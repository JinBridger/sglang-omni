# SPDX-License-Identifier: Apache-2.0
"""Batched HeartMoE depth decoder with the checkpoint's native tensor layout."""

import math
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

LocalCache = list[tuple[torch.Tensor, torch.Tensor]]


class AudioConfig(TypedDict):
    codebook_size: int
    hidden_size: int
    special_token_ids: list[int]
    local_decoder: dict[str, int | float | bool]


def rms_norm(
    values: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(-1, keepdim=True) + epsilon
    )
    return (normalized * (weight.float() + 1)).to(values.dtype)


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return rms_norm(values, self.weight, self.epsilon)


class NormLinear(nn.Linear):
    def __init__(
        self, input_features: int, output_features: int, epsilon: float
    ) -> None:
        super().__init__(input_features, output_features, bias=False)
        self.layer_norm_weight = nn.Parameter(torch.zeros(input_features))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(
            rms_norm(values, self.layer_norm_weight, self.epsilon), self.weight
        )


class LocalAttention(nn.Module):
    def __init__(self, config: dict[str, int | float | bool]) -> None:
        super().__init__()
        self.heads = config["num_attention_heads"]
        self.groups = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.rotary_dim = int(self.head_dim * config["partial_rotary_factor"])
        self.linear_qkv = NormLinear(
            config["hidden_size"],
            (2 * self.heads + 2 * self.groups) * self.head_dim,
            config["rms_norm_eps"],
        )
        self.linear_proj = nn.Linear(
            self.heads * self.head_dim, config["hidden_size"], bias=False
        )
        frequencies = 1 / (
            config["rope_theta"]
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        angles = torch.outer(
            torch.arange(8, dtype=torch.float32), frequencies
        ).repeat_interleave(2, -1)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def rotary(self, values: torch.Tensor, offset: int) -> torch.Tensor:
        cosine = self.rope_cos[offset : offset + values.shape[1]][None, :, None].to(
            values.dtype
        )
        sine = self.rope_sin[offset : offset + values.shape[1]][None, :, None].to(
            values.dtype
        )
        rotating, remainder = (
            values[..., : self.rotary_dim],
            values[..., self.rotary_dim :],
        )
        pairs = rotating.reshape(*rotating.shape[:-1], -1, 2)
        perpendicular = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return torch.cat((rotating * cosine + perpendicular * sine, remainder), dim=-1)

    def forward(
        self, values: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, length, _ = values.shape
        head_width = self.heads // self.groups * self.head_dim
        packed = self.linear_qkv(values).view(batch_size, length, self.groups, -1)
        q, gate, k, v = packed.split(
            [head_width, head_width, self.head_dim, self.head_dim], -1
        )
        offset = 0 if cache is None else cache[0].shape[2]
        q = self.rotary(
            q.reshape(batch_size, length, self.heads, self.head_dim), offset
        ).transpose(1, 2)
        k = self.rotary(k, offset).transpose(1, 2)
        v = v.transpose(1, 2)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        else:
            pass
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=offset == 0, dropout_p=0.0, enable_gqa=True
        )
        attended = attended.transpose(1, 2).reshape(batch_size, length, -1)
        attended = (
            attended.float() * gate.reshape(batch_size, length, -1).float().sigmoid()
        ).to(values.dtype)
        return self.linear_proj(attended), (k, v)


class LocalMLP(nn.Module):
    def __init__(self, config: dict[str, int | float | bool]) -> None:
        super().__init__()
        self.linear_fc1 = NormLinear(
            config["hidden_size"],
            2 * config["intermediate_size"],
            config["rms_norm_eps"],
        )
        self.linear_fc2 = nn.Linear(
            config["intermediate_size"], config["hidden_size"], bias=False
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate, values = self.linear_fc1(values).chunk(2, dim=-1)
        return self.linear_fc2((F.silu(gate.float()) * values.float()).to(values.dtype))


class LocalLayer(nn.Module):
    def __init__(self, config: dict[str, int | float | bool]) -> None:
        super().__init__()
        self.self_attention = LocalAttention(config)
        self.mlp = LocalMLP(config)

    def forward(
        self, values: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attention, cache = self.self_attention(values, cache)
        values = values + attention
        return values + self.mlp(values), cache


class LocalLayerAdapter(nn.Module):
    def __init__(self, config: dict[str, int | float | bool]) -> None:
        super().__init__()
        self.layer = LocalLayer(config)


class LocalDecoder(nn.Module):
    def __init__(
        self, config: dict[str, int | float | bool], codebook_size: int
    ) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.layers = nn.ModuleList(
            [LocalLayerAdapter(config) for _ in range(config["num_hidden_layers"])]
        )
        self.norm = ZeroCenteredRMSNorm(hidden_size, config["rms_norm_eps"])
        self.input_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(codebook_size, hidden_size) for _ in range(7)]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, codebook_size, bias=False) for _ in range(7)]
        )

    def run_layers(
        self, values: torch.Tensor, cache: LocalCache | None
    ) -> tuple[torch.Tensor, LocalCache]:
        next_cache = []
        for i, block in enumerate(self.layers):
            values, layer_cache = block.layer(
                values, None if cache is None else cache[i]
            )
            next_cache.append(layer_cache)
        return self.norm(values)[:, -1], next_cache


class AudioModules(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        size, hidden_size = config["codebook_size"], config["hidden_size"]
        self.mtp = LocalDecoder(config["local_decoder"], size)
        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(size, hidden_size) for _ in range(8)]
        )
        self.audio_head = nn.Linear(hidden_size, size, bias=False)
        self.audio_end_head = nn.Linear(hidden_size, 1, bias=False)
        self.special_token_embeddings = nn.Module()
        self.special_token_embeddings.embedding = nn.Embedding(
            len(config["special_token_ids"]), hidden_size
        )

    def audio_input(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(
            table(codes[:, i]) for i, table in enumerate(self.audio_embeddings[:3])
        ) / math.sqrt(3)

    def audio_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.audio_head(hidden), self.audio_end_head(hidden)), dim=-1)


def sample_codes(
    logits: torch.Tensor, uniforms: torch.Tensor, sampling: torch.Tensor
) -> torch.Tensor:
    temperature, top_k, top_p = sampling.unbind(-1)
    values, indices = (logits.float() / temperature.clamp_min(1e-5)[:, None]).sort(
        descending=True
    )
    ranks = torch.arange(values.shape[-1], device=values.device)
    values = values.masked_fill(
        (top_k[:, None] > 0) & (ranks >= top_k[:, None]), -torch.inf
    )
    probabilities = values.softmax(-1)
    values = values.masked_fill(
        (probabilities.cumsum(-1) - probabilities) >= top_p[:, None], -torch.inf
    )
    cumulative = values.softmax(-1).cumsum(-1)
    selected = (cumulative < uniforms[:, None]).sum(-1).clamp_max(values.shape[-1] - 1)
    selected = indices.gather(1, selected[:, None]).squeeze(1)
    return torch.where(temperature == 0, logits.argmax(-1), selected)


class FrameDecoder:
    def __init__(
        self, audio: AudioModules, batch_sizes: list[int], use_graph: bool
    ) -> None:
        self.audio = audio
        parameter = audio.audio_head.weight
        self.hidden = torch.zeros(
            max(batch_sizes),
            parameter.shape[1],
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.uniforms = torch.full((max(batch_sizes), 8), 0.5, device=parameter.device)
        self.sampling = torch.tensor([0.8, 50, 0.95], device=parameter.device).repeat(
            max(batch_sizes), 1
        )
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if use_graph:
            stream = torch.cuda.Stream(device=parameter.device)
            stream.wait_stream(torch.cuda.current_stream(parameter.device))
            with torch.cuda.stream(stream), torch.inference_mode():
                for batch_size in batch_sizes:
                    for _ in range(3):
                        self.run(
                            self.hidden[:batch_size],
                            self.uniforms[:batch_size],
                            self.sampling[:batch_size],
                        )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        self.outputs[batch_size] = self.run(
                            self.hidden[:batch_size],
                            self.uniforms[:batch_size],
                            self.sampling[:batch_size],
                        )
                    self.graphs[batch_size] = graph
            torch.cuda.current_stream(parameter.device).wait_stream(stream)
        else:
            pass

    def run(
        self, hidden: torch.Tensor, uniforms: torch.Tensor, sampling: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_code = sample_codes(
            self.audio.audio_logits(hidden), uniforms[:, 0], sampling
        )
        codes = [first_code]
        decoder = self.audio.mtp
        values = torch.stack(
            (
                decoder.input_projection(hidden),
                decoder.embeddings[0](first_code.clamp_max(8191)),
            ),
            dim=1,
        )
        cache = None
        for i, head in enumerate(decoder.heads):
            if i:
                values = decoder.embeddings[i](codes[-1]).unsqueeze(1)
            else:
                pass
            last, cache = decoder.run_layers(values, cache)
            codes.append(sample_codes(head(last), uniforms[:, i + 1], sampling))
        codes = torch.stack(codes, dim=1)
        return codes, self.audio.audio_input(codes.clamp_max(8191))

    @torch.inference_mode()
    def __call__(
        self, hidden: torch.Tensor, uniforms: torch.Tensor, sampling: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.graphs:
            return self.run(hidden, uniforms, sampling)
        else:
            batch_size = hidden.shape[0]
            bucket = next(size for size in self.graphs if size >= batch_size)
            self.hidden[:batch_size].copy_(hidden)
            self.uniforms[:batch_size].copy_(uniforms)
            self.sampling[:batch_size].copy_(sampling)
            self.graphs[bucket].replay()
            codes, feedback = self.outputs[bucket]
            return codes[:batch_size], feedback[:batch_size]
