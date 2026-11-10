# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The ACESTEO Team. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
"""Inference attention, normalization, and conditioning layers."""

import math
from dataclasses import dataclass

import torch
from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_varlen_func
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.mula_edit.model_config import MulaModelConfig


@torch.library.custom_op("sglang_omni::mula_edit_local_attention", mutates_args=())
def local_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    window_frames: int,
) -> torch.Tensor:
    attended = flash_attn_varlen_func(
        q=queries.transpose(1, 2),
        k=keys.transpose(1, 2),
        v=values.transpose(1, 2),
        causal=False,
        window_size=(window_frames, window_frames),
    )
    return attended.transpose(1, 2).contiguous()


@local_attention.register_fake
def local_attention_shape(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    window_frames: int,
) -> torch.Tensor:
    """Only queries determine the output shape of the registered attention operator."""
    return torch.empty(queries.shape, device=queries.device, dtype=queries.dtype)


@dataclass(kw_only=True)
class AttentionKeysValues:
    keys: torch.Tensor
    values: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.epsilon
        )
        return self.weight * normalized.to(hidden_states.dtype)


class FeedForward(nn.Module):
    def __init__(self, config: MulaModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def rotary_embeddings(
    hidden_states: torch.Tensor, config: MulaModelConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(0, config.head_dim, 2, device=hidden_states.device).float()
    # note (Codex): Checkpoint inference rounds the inverse frequencies when moving the model to bfloat16.
    inverse_frequencies = (
        (1.0 / config.rope_theta ** (indices / config.head_dim))
        .to(hidden_states.dtype)
        .float()
    )
    positions = torch.arange(
        hidden_states.shape[1], device=hidden_states.device
    ).float()
    frequencies = inverse_frequencies[None, :, None] @ positions[None, None, :]
    angles = torch.cat(
        (frequencies.transpose(1, 2), frequencies.transpose(1, 2)), dim=-1
    )
    return angles.cos().to(hidden_states.dtype), angles.sin().to(hidden_states.dtype)


def rotate_hidden(
    hidden_states: torch.Tensor, embeddings: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cosine, sine = embeddings
    first, second = hidden_states.chunk(2, dim=-1)
    return (
        hidden_states * cosine[:, None]
        + torch.cat((-second, first), dim=-1) * sine[:, None]
    )


def attention_mask(
    valid_frames: torch.Tensor, dtype: torch.dtype, window_frames: int | None
) -> torch.Tensor:
    frames = valid_frames.shape[1]
    valid = valid_frames[:, None, None, :].expand(-1, 1, frames, -1)
    if window_frames is not None:
        positions = torch.arange(frames, device=valid_frames.device)
        valid = (
            valid
            & ((positions[:, None] - positions[None, :]).abs() <= window_frames)[
                None, None
            ]
        )
    else:
        pass
    return torch.zeros(valid.shape, device=valid.device, dtype=dtype).masked_fill(
        ~valid, torch.finfo(dtype).min
    )


class Attention(nn.Module):
    def __init__(self, config: MulaModelConfig) -> None:
        super().__init__()
        self.head_dimension = config.head_dim
        self.key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.use_native_attention: bool = False
        self.use_flash_attention: bool = False
        self.window_frames: int | None = None
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def project_keys_values(self, hidden_states: torch.Tensor) -> AttentionKeysValues:
        shape = (*hidden_states.shape[:-1], -1, self.head_dimension)
        return AttentionKeysValues(
            keys=self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2),
            values=self.v_proj(hidden_states).view(shape).transpose(1, 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None,
        embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cached: AttentionKeysValues | None = None,
    ) -> torch.Tensor:
        shape = (*hidden_states.shape[:-1], -1, self.head_dimension)
        queries = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        keys_values = (
            self.project_keys_values(hidden_states) if cached is None else cached
        )
        keys, values = keys_values.keys, keys_values.values
        if embeddings is not None:
            queries = rotate_hidden(queries, embeddings)
            keys = rotate_hidden(keys, embeddings)
        else:
            pass
        if self.use_flash_attention:
            assert self.window_frames is not None
            attended = local_attention(queries, keys, values, self.window_frames)
        elif self.use_native_attention and mask is None:
            attended = F.scaled_dot_product_attention(
                queries, keys, values, dropout_p=0.0, is_causal=False, enable_gqa=True
            )
        else:
            keys = keys.repeat_interleave(self.key_value_groups, dim=1)
            values = values.repeat_interleave(self.key_value_groups, dim=1)
            attended = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        return self.o_proj(attended.transpose(1, 2).contiguous().flatten(2))


class EncoderLayer(nn.Module):
    def __init__(self, config: MulaModelConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), mask, embeddings
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class DiTLayer(nn.Module):
    def __init__(self, config: MulaModelConfig) -> None:
        super().__init__()
        self.self_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.cross_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.cross_attn = Attention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = FeedForward(config)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None,
        embeddings: tuple[torch.Tensor, torch.Tensor],
        timestep_projection: torch.Tensor,
        cross_attention: AttentionKeysValues,
        cross_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = (
            self.scale_shift_table + timestep_projection
        ).chunk(6, dim=1)
        normalized = (self.self_attn_norm(hidden_states) * (1 + scale) + shift).to(
            hidden_states.dtype
        )
        hidden_states = (
            hidden_states + self.self_attn(normalized, mask, embeddings) * gate
        ).to(hidden_states.dtype)
        hidden_states = hidden_states + self.cross_attn(
            self.cross_attn_norm(hidden_states),
            cross_attention_mask,
            cached=cross_attention,
        )
        normalized = (self.mlp_norm(hidden_states) * (1 + mlp_scale) + mlp_shift).to(
            hidden_states.dtype
        )
        return (hidden_states + self.mlp(normalized) * mlp_gate).to(hidden_states.dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden_dimension)
        self.linear_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.time_proj = nn.Linear(hidden_dimension, hidden_dimension * 6)

    def forward(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(128, dtype=torch.float32, device=timestep.device)
            / 128
        )
        angles = (timestep * 1000)[:, None].float() * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1).to(timestep.dtype)
        embedding = self.linear_2(F.silu(self.linear_1(embedding)))
        return embedding, self.time_proj(F.silu(embedding)).unflatten(1, (6, -1))
