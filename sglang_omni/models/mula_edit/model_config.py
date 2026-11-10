# SPDX-License-Identifier: Apache-2.0
"""Checkpoint architecture fields used by Mula inference."""

from typing import Literal, Self

from pydantic import BaseModel, model_validator


class MulaModelConfig(BaseModel):
    model_type: Literal["acestep_v4"] = "acestep_v4"
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000
    use_sliding_window: bool = True
    sliding_window: int = 128
    layer_types: list[Literal["sliding_attention", "full_attention"]]
    attention_bias: bool = False
    num_lyric_encoder_hidden_layers: int = 8
    num_timbre_encoder_hidden_layers: int = 4
    text_hidden_dim: int = 1024
    timbre_hidden_dim: int = 64
    audio_acoustic_hidden_dim: Literal[64] = 64
    in_channels: Literal[192] = 192
    num_symbolic_slots: Literal[3] = 3
    symbolic_slot_dim: int = 32
    patch_size: Literal[2] = 2
    use_source_latents: Literal[True] = True
    hidden_act: Literal["silu"] = "silu"

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must describe every decoder layer")
        elif self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Attention heads must be divisible by key/value heads")
        elif self.symbolic_slot_dim <= 0 or self.symbolic_slot_dim % 2:
            raise ValueError("symbolic_slot_dim must be a positive even integer")
        else:
            pass
        return self
