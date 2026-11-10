# SPDX-License-Identifier: Apache-2.0
"""HeartCodec transformer."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.rope_cache = {}

    def get_sin_cos(
        self, sequence_length: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (sequence_length, device, dtype)
        cached_angles = self.rope_cache.get(cache_key, None)
        if cached_angles is not None and cached_angles[0].device == device:
            return cached_angles
        else:
            pass
        inverse_frequencies = 1.0 / self.base ** (
            torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim
        )
        sequence_length = torch.arange(sequence_length, device=device, dtype=dtype)
        frequencies = torch.einsum("i,j->ij", sequence_length, inverse_frequencies)
        sin = frequencies.sin()
        cos = frequencies.cos()
        self.rope_cache[cache_key] = (sin, cos)
        return (sin, cos)


class LlamaAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
        rope_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.q_proj = nn.Linear(dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=bias)
        self.o_proj = nn.Linear(self.inner_dim, dim, bias=bias)
        self.dropout = dropout
        self.rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope = RotaryEmbedding(self.rope_dim)

    def reshape_heads(
        self, x: torch.Tensor, batch_size: int, sequence_length: int
    ) -> torch.Tensor:
        return x.view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        q = self.reshape_heads(self.q_proj(x), batch_size, sequence_length)
        k = self.reshape_heads(self.k_proj(x), batch_size, sequence_length)
        v = self.reshape_heads(self.v_proj(x), batch_size, sequence_length)
        rotary_width = min(self.rope_dim, self.head_dim)
        sine, cosine = self.rope.get_sin_cos(
            sequence_length, device=x.device, dtype=x.dtype
        )
        sine = sine.view(1, 1, sequence_length, rotary_width // 2, 1)
        cosine = cosine.view(1, 1, sequence_length, rotary_width // 2, 1)

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            head, tail = (tensor[..., :rotary_width], tensor[..., rotary_width:])
            head = head.view(
                batch_size, self.n_heads, sequence_length, rotary_width // 2, 2
            )
            x1, x2 = (head[..., 0:1], head[..., 1:2])
            rotated = torch.cat(
                [x1 * cosine - x2 * sine, x1 * sine + x2 * cosine], dim=-1
            )
            return torch.cat(
                [
                    rotated.view(
                        batch_size, self.n_heads, sequence_length, rotary_width
                    ),
                    tail,
                ],
                dim=-1,
            )

        attended = F.scaled_dot_product_attention(
            rotate(q),
            rotate(k),
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.o_proj(
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.inner_dim)
        )


class LlamaMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.up(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.down(x)


class LlamaTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        mlp_multiple_of: int = 256,
        dropout: float = 0.0,
        attention_bias: bool = False,
        use_ada_layer_norm_single: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim, 1e-06)
        self.attn = LlamaAttention(
            dim,
            n_heads,
            head_dim,
            bias=attention_bias,
            dropout=dropout,
            rope_dim=head_dim,
        )
        self.mlp_norm = RMSNorm(dim, 1e-06)
        self.mlp = LlamaMLP(dim, multiple_of=mlp_multiple_of, dropout=dropout)
        self.use_ada_layer_norm_single = use_ada_layer_norm_single
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        else:
            pass

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_ada_layer_norm_single:
            batch_size = x.shape[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.attn_norm(x)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            attention_output = self.attn(norm_hidden_states)
            attention_output = gate_msa * attention_output
            x = x + attention_output
            norm_hidden_states = self.mlp_norm(x)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            attention_output = self.mlp(norm_hidden_states)
            attention_output = gate_mlp * attention_output
            x = x + attention_output
            return x
        else:
            attention_output = self.attn(self.attn_norm(x))
            x = x + attention_output
            attention_output = self.mlp(self.mlp_norm(x))
            x = x + attention_output
            return x


class ProjectLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        filter_size: int,
        kernel_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.ffn_1 = nn.Conv1d(
            hidden_size, filter_size, kernel_size, padding=kernel_size // 2
        )
        self.ffn_2 = nn.Linear(filter_size, filter_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn_1(x.transpose(1, 2)).transpose(1, 2)
        x = x * self.kernel_size ** (-0.5)
        x = self.ffn_2(x)
        return x


class LlamaTransformer(nn.Module):

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        out_channels: int,
        num_layers: int = 12,
        num_layers_2: int = 2,
        dropout: float = 0.0,
        norm_type: str = "layer_norm",
    ) -> None:
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        inner_dim_2 = inner_dim * 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = inner_dim
        self.inner_dim_2 = inner_dim_2
        self.dropout = dropout
        self.proj_in = ProjectLayer(in_channels, inner_dim, kernel_size=3)
        use_ada_single = norm_type == "ada_norm_single"
        self.transformer_blocks = nn.ModuleList(
            [
                LlamaTransformerBlock(
                    dim=inner_dim,
                    n_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    dropout=dropout,
                    attention_bias=False,
                    use_ada_layer_norm_single=use_ada_single,
                )
                for _ in range(num_layers)
            ]
        )
        self.transformer_blocks_2 = nn.ModuleList(
            [
                LlamaTransformerBlock(
                    dim=inner_dim_2,
                    n_heads=num_attention_heads,
                    head_dim=attention_head_dim * 2,
                    dropout=dropout,
                    attention_bias=False,
                    use_ada_layer_norm_single=use_ada_single,
                )
                for _ in range(num_layers_2)
            ]
        )
        self.connection_proj = ProjectLayer(
            in_channels + inner_dim, inner_dim_2, kernel_size=3
        )
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-06)
        self.norm_out_2 = nn.LayerNorm(inner_dim_2, elementwise_affine=False, eps=1e-06)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )
        self.scale_shift_table_2 = nn.Parameter(
            torch.randn(2, inner_dim_2) / inner_dim_2**0.5
        )
        self.proj_out = ProjectLayer(inner_dim_2, out_channels, kernel_size=3)
        self.adaln_single = AdaLayerNormSingleFlow(inner_dim)
        self.adaln_single_2 = AdaLayerNormSingleFlow(inner_dim_2)

    def forward(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        projected_states = self.proj_in(hidden_states)
        timestep_mod, embedded_timestep = self.adaln_single(
            timestep, hidden_dtype=projected_states.dtype
        )
        for block in self.transformer_blocks:
            projected_states = block(projected_states, timestep=timestep_mod)
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None]
        ).chunk(2, dim=1)
        projected_states = self.norm_out(projected_states)
        projected_states = projected_states * (1 + scale) + shift
        x = torch.cat([hidden_states, projected_states], dim=-1)
        x = self.connection_proj(x)
        timestep_mod_2, embedded_timestep_2 = self.adaln_single_2(
            timestep, hidden_dtype=x.dtype
        )
        for block in self.transformer_blocks_2:
            x = block(x, timestep=timestep_mod_2)
        shift_2, scale_2 = (
            self.scale_shift_table_2[None] + embedded_timestep_2[:, None]
        ).chunk(2, dim=1)
        x = self.norm_out_2(x)
        x = x * (1 + scale_2) + shift_2
        out = self.proj_out(x)
        return out


class PixArtAlphaCombinedFlowEmbeddings(nn.Module):

    def __init__(self, embedding_dim: int, size_emb_dim: int) -> None:
        super().__init__()
        self.flow_t_size = 512
        self.outdim = size_emb_dim
        self.timestep_embedder = TimestepEmbedding(
            in_channels=self.flow_t_size, time_embed_dim=embedding_dim
        )

    def timestep_embedding(
        self, timesteps: torch.Tensor, max_period: int = 10000, scale: int = 1000
    ) -> torch.Tensor:
        half = self.flow_t_size // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, device=timesteps.device)
            / half
        ).type(timesteps.type())
        angles = timesteps[:, None] * frequencies[None] * scale
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.flow_t_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        else:
            pass
        return embedding

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype
    ) -> torch.Tensor:
        timesteps_proj = self.timestep_embedding(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        conditioning = timesteps_emb
        return conditioning


class AdaLayerNormSingleFlow(nn.Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.emb = PixArtAlphaCombinedFlowEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return (self.linear(self.silu(embedded_timestep)), embedded_timestep)


class TimestepEmbedding(nn.Module):

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x
