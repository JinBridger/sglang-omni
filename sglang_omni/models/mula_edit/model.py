# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The ACESTEO Team. All rights reserved.
"""Checkpoint-compatible condition encoders and diffusion transformer."""

from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.mula_edit.layers import (
    AttentionKeysValues,
    DiTLayer,
    EncoderLayer,
    RMSNorm,
    TimestepEmbedding,
    attention_mask,
    rotary_embeddings,
)
from sglang_omni.models.mula_edit.model_config import MulaModelConfig
from sglang_omni.models.mula_edit.symbols import SymbolicSlotEncoder


def pack_sequences(
    first: torch.Tensor,
    second: torch.Tensor,
    first_mask: torch.Tensor,
    second_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.cat((first, second), dim=1)
    mask = torch.cat((first_mask, second_mask), dim=1)
    indices = mask.argsort(dim=1, descending=True, stable=True)
    hidden_states = hidden_states.gather(
        1, indices.unsqueeze(-1).expand_as(hidden_states)
    )
    valid = torch.arange(mask.shape[1], device=mask.device)[None] < mask.sum(
        dim=1, keepdim=True
    )
    return hidden_states, valid


class ConditionSequenceEncoder(nn.Module):
    def __init__(self, config: MulaModelConfig, *, timbre: bool) -> None:
        super().__init__()
        self.config = config
        self.is_timbre = timbre
        input_dimension = config.timbre_hidden_dim if timbre else config.text_hidden_dim
        layer_count = (
            config.num_timbre_encoder_hidden_layers
            if timbre
            else config.num_lyric_encoder_hidden_layers
        )
        self.embed_tokens = nn.Linear(input_dimension, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(layer_count)])
        if timbre:
            self.special_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        else:
            pass

    def forward(
        self, embeddings: torch.Tensor, valid_tokens: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(embeddings)
        positions = rotary_embeddings(hidden_states, self.config)
        full_mask = attention_mask(valid_tokens.bool(), hidden_states.dtype, None)
        sliding_mask = (
            attention_mask(
                valid_tokens.bool(), hidden_states.dtype, self.config.sliding_window
            )
            if self.config.use_sliding_window
            else full_mask
        )
        for index, layer in enumerate(self.layers):
            mask = (
                sliding_mask
                if self.config.layer_types[index] == "sliding_attention"
                else full_mask
            )
            hidden_states = layer(hidden_states, mask, positions)
        hidden_states = self.norm(hidden_states)
        return hidden_states[:, :1] if self.is_timbre else hidden_states


class ConditionEncoder(nn.Module):
    def __init__(self, config: MulaModelConfig) -> None:
        super().__init__()
        self.text_projector = nn.Linear(
            config.text_hidden_dim, config.hidden_size, bias=False
        )
        self.lyric_encoder = ConditionSequenceEncoder(config, timbre=False)
        self.timbre_encoder = ConditionSequenceEncoder(config, timbre=True)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        lyric_embeddings: torch.Tensor,
        lyric_mask: torch.Tensor,
    ) -> torch.Tensor:
        lyrics = self.lyric_encoder(lyric_embeddings, lyric_mask)
        timbre_mask = torch.ones(
            lyrics.shape[0], 1, device=lyrics.device, dtype=torch.long
        )
        timbre = self.timbre_encoder(
            torch.zeros(
                lyrics.shape[0], 1, 64, device=lyrics.device, dtype=lyrics.dtype
            ),
            timbre_mask,
        )
        hidden_states, mask = pack_sequences(lyrics, timbre, lyric_mask, timbre_mask)
        hidden_states, _ = pack_sequences(
            hidden_states, self.text_projector(text_embeddings), mask, text_mask
        )
        return hidden_states


class InputProjection(nn.Module):
    def __init__(self, config: MulaModelConfig) -> None:
        super().__init__()
        self.base_proj = nn.Conv1d(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.sym_proj = nn.Conv1d(
            3 * (config.symbolic_slot_dim + 1),
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.joint_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, base: torch.Tensor, symbols: torch.Tensor) -> torch.Tensor:
        hidden_states = self.base_proj(base.transpose(1, 2)).transpose(1, 2)
        symbolic_states = self.sym_proj(symbols.transpose(1, 2)).transpose(1, 2)
        return hidden_states + self.joint_proj(
            torch.cat((hidden_states, symbolic_states), dim=-1)
        )


class MulaDiT(nn.Module):
    def __init__(self, config: MulaModelConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [DiTLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.proj_in = InputProjection(config)
        self.time_embed = TimestepEmbedding(config.hidden_size)
        self.time_embed_r = TimestepEmbedding(config.hidden_size)
        self.condition_embedder = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm_out = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.proj_out = nn.Sequential(
            nn.Identity(),
            nn.ConvTranspose1d(
                config.hidden_size, 64, config.patch_size, stride=config.patch_size
            ),
        )
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, config.hidden_size))
        self.symbolic_encoder = SymbolicSlotEncoder(
            config.symbolic_slot_dim, config.rms_norm_eps
        )
        self.context_null = nn.Embedding(1, 64)

    def prepare_cross_attention(
        self, conditions: torch.Tensor
    ) -> list[AttentionKeysValues]:
        hidden_states = self.condition_embedder(conditions)
        return [
            layer.cross_attn.project_keys_values(hidden_states) for layer in self.layers
        ]

    def forward(
        self,
        noisy_latents: torch.Tensor,
        context_latents: torch.Tensor,
        symbols: torch.Tensor,
        timestep: torch.Tensor,
        frame_lengths: torch.Tensor,
        cross_attention: list[AttentionKeysValues],
    ) -> torch.Tensor:
        frames = noisy_latents.shape[1]
        base = torch.cat((context_latents, noisy_latents), dim=-1)
        padding = frames % self.config.patch_size
        if padding:
            base = F.pad(base, (0, 0, 0, padding))
            symbols = F.pad(symbols, (0, 0, 0, padding))
        else:
            pass
        # note (Codex): Contiguous token rows let linear projections use one matrix multiplication across requests.
        hidden_states = self.proj_in(base, symbols).contiguous()
        embedding, projection = self.time_embed(timestep)
        relative_embedding, relative_projection = self.time_embed_r(
            torch.zeros_like(timestep)
        )
        embedding = embedding + relative_embedding
        projection = projection + relative_projection
        positions = rotary_embeddings(hidden_states, self.config)
        patch_lengths = (
            frame_lengths + self.config.patch_size - 1
        ) // self.config.patch_size
        valid = (
            torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
            < patch_lengths[:, None]
        )
        full_mask = attention_mask(valid, hidden_states.dtype, None)
        # note (Codex): An explicit additive mask preserves the trained decoder's SDPA path, including padded condition tokens.
        cross_attention_mask = hidden_states.new_zeros(
            1, 1, hidden_states.shape[1], cross_attention[0].keys.shape[2]
        )
        sliding_mask = (
            attention_mask(valid, hidden_states.dtype, self.config.sliding_window)
            if self.config.use_sliding_window
            else full_mask
        )
        for index, layer in enumerate(self.layers):
            mask = (
                sliding_mask
                if self.config.layer_types[index] == "sliding_attention"
                else full_mask
            )
            hidden_states = layer(
                hidden_states,
                mask,
                positions,
                projection,
                cross_attention[index],
                cross_attention_mask,
            )
        shift, scale = (self.scale_shift_table + embedding.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).to(
            hidden_states.dtype
        )
        return self.proj_out(hidden_states.transpose(1, 2)).transpose(1, 2)[:, :frames]


class MulaModel(nn.Module):
    def __init__(self, config: MulaModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConditionEncoder(config)
        self.decoder = MulaDiT(config)
        self.null_condition_emb = nn.Parameter(torch.empty(1, 1, config.hidden_size))

    @classmethod
    def load(
        cls, model_path: str, device: torch.device, dtype: torch.dtype
    ) -> "MulaModel":
        directory = Path(model_path)
        config = MulaModelConfig.model_validate_json(
            (directory / "config.json").read_text()
        )
        with torch.device("meta"):
            model = cls(config)
        weights: dict[str, torch.Tensor] = {}
        with safe_open(
            directory / "model.safetensors", framework="pt", device="cpu"
        ) as checkpoint:
            for name in checkpoint.keys():
                if name.startswith(("tokenizer.", "detokenizer.")):
                    continue
                else:
                    tensor = checkpoint.get_tensor(name)
                if name == "decoder.context_null.weight":
                    tensor = tensor.unsqueeze(0)
                else:
                    pass
                weights[name] = tensor
        model.load_state_dict(weights, strict=True, assign=True)
        return model.to(device=device, dtype=dtype).eval().requires_grad_(False)
