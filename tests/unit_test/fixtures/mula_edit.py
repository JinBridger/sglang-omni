# SPDX-License-Identifier: Apache-2.0
"""Small music decoder and seeded sampling inputs for inference regressions."""

import torch

from sglang_omni.models.mula_edit.model import MulaDiT
from sglang_omni.models.mula_edit.model_config import MulaModelConfig
from sglang_omni.models.mula_edit.sampling import SamplingState


def small_decoder(
    *, head_dimension: int = 8, uses_sliding_window: bool = True
) -> MulaDiT:
    torch.manual_seed(21)
    config = MulaModelConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dimension,
        use_sliding_window=uses_sliding_window,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=2,
        text_hidden_dim=16,
        num_lyric_encoder_hidden_layers=1,
        num_timbre_encoder_hidden_layers=1,
    )
    decoder = MulaDiT(config).eval()
    with torch.no_grad():
        for name, parameter in decoder.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.1)
    return decoder


def sampling_state(
    decoder: MulaDiT, frames: int, steps: int, guidance: float
) -> SamplingState:
    device = decoder.scale_shift_table.device
    dtype = decoder.scale_shift_table.dtype
    clean = torch.randn(1, frames, 64, device=device, dtype=dtype)
    noise = torch.randn_like(clean)
    mask = torch.zeros(1, frames, dtype=torch.bool, device=device)
    mask[:, 1:-1] = True
    conditions = torch.randn(
        2 if guidance > 1 else 1,
        9,
        decoder.config.hidden_size,
        device=device,
        dtype=dtype,
    )
    return SamplingState(
        clean_latents=clean,
        noise=noise,
        latents=noise,
        context_latents=torch.cat(
            (
                clean.masked_fill(mask.unsqueeze(-1), 0),
                mask.unsqueeze(-1).expand_as(clean),
            ),
            dim=-1,
        ),
        symbols=torch.randn(1, frames, 99, device=device, dtype=dtype),
        mask=mask,
        timesteps=torch.linspace(1, 0, steps + 1, device=device, dtype=dtype),
        cross_attention=decoder.prepare_cross_attention(conditions),
        guidance_scale=guidance,
        crossfade_frames=0,
    )
