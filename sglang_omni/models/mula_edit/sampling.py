# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ACEStep
"""Euler/APG sampling with independently advancing requests in one DiT batch."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sglang_omni.models.mula_edit.layers import AttentionKeysValues
from sglang_omni.models.mula_edit.model import MulaDiT


@dataclass(kw_only=True)
class SamplingState:
    clean_latents: torch.Tensor
    noise: torch.Tensor
    latents: torch.Tensor
    context_latents: torch.Tensor
    symbols: torch.Tensor
    mask: torch.Tensor
    timesteps: torch.Tensor
    cross_attention: list[AttentionKeysValues]
    guidance_scale: float
    crossfade_frames: int
    step_index: int = 0
    momentum: torch.Tensor | None = None

    @property
    def done(self) -> bool:
        return self.step_index == len(self.timesteps) - 1

    @property
    def branches(self) -> int:
        return 2 if self.guidance_scale > 1 else 1

    def finish(self) -> torch.Tensor:
        """Restore source latents with the checkpoint's boundary blend."""
        soft = self.mask.float().clone()
        positions = torch.nonzero(self.mask[0], as_tuple=False).flatten()
        first, last = int(positions[0]), int(positions[-1]) + 1
        before = min(first, self.crossfade_frames)
        after = min(soft.shape[1] - last, self.crossfade_frames)
        if before:
            soft[:, first - before : first] = torch.linspace(
                0, 1, before + 2, device=soft.device
            )[1:-1]
        else:
            pass
        if after:
            soft[:, last : last + after] = torch.linspace(
                1, 0, after + 2, device=soft.device
            )[1:-1]
        else:
            pass
        return (
            soft.unsqueeze(-1) * self.latents
            + (1 - soft.unsqueeze(-1)) * self.clean_latents
        )


@torch.inference_mode()
def advance_batch(decoder: MulaDiT, states: list[SamplingState]) -> None:
    """Advance each request exactly once, masking padding from other recordings."""
    longest = max(state.latents.shape[1] for state in states)
    noisy_rows: list[torch.Tensor] = []
    context_rows: list[torch.Tensor] = []
    symbolic_rows: list[torch.Tensor] = []
    timestep_rows: list[torch.Tensor] = []
    frame_lengths: list[int] = []
    for state in states:
        frames = state.latents.shape[1]
        padding = (0, 0, 0, longest - frames)
        noisy_rows.append(F.pad(state.latents, padding).expand(state.branches, -1, -1))
        context_rows.append(
            F.pad(state.context_latents, padding).expand(state.branches, -1, -1)
        )
        symbolic_rows.append(
            F.pad(state.symbols, padding).expand(state.branches, -1, -1)
        )
        timestep_rows.append(state.timesteps[state.step_index].expand(state.branches))
        frame_lengths.extend([frames] * state.branches)
    cross_attention = [
        AttentionKeysValues(
            keys=torch.cat([state.cross_attention[index].keys for state in states]),
            values=torch.cat([state.cross_attention[index].values for state in states]),
        )
        for index in range(len(decoder.layers))
    ]
    velocities = decoder(
        torch.cat(noisy_rows),
        torch.cat(context_rows),
        torch.cat(symbolic_rows),
        torch.cat(timestep_rows),
        torch.tensor(frame_lengths, device=states[0].latents.device),
        cross_attention,
    )
    offset = 0
    for state in states:
        frames = state.latents.shape[1]
        velocity = velocities[offset : offset + 1, :frames]
        if state.branches == 2:
            difference = velocity - velocities[offset + 1 : offset + 2, :frames]
            state.momentum = (
                difference
                if state.momentum is None
                else difference + (-0.75 * state.momentum)
            )
            update = state.momentum * torch.minimum(
                torch.ones_like(state.momentum),
                2.5 / state.momentum.norm(p=2, dim=1, keepdim=True),
            )
            direction = F.normalize(velocity.double(), dim=1)
            parallel = (update.double() * direction).sum(
                dim=1, keepdim=True
            ) * direction
            orthogonal = (update.double() - parallel).to(velocity.dtype)
            velocity = velocity + (state.guidance_scale - 1) * orthogonal
        else:
            pass
        next_timestep = state.timesteps[state.step_index + 1]
        delta = state.timesteps[state.step_index] - next_timestep
        generated = state.latents - velocity * delta
        source = next_timestep * state.noise + (1 - next_timestep) * state.clean_latents
        state.latents = torch.where(state.mask.unsqueeze(-1), generated, source)
        state.step_index += 1
        offset += state.branches
