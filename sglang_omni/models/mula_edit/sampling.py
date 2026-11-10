# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ACEStep
"""Serial Euler/APG sampling with a compiled DiT and reusable CUDA Graph."""

import logging
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sglang_omni.models.mula_edit.layers import AttentionKeysValues
from sglang_omni.models.mula_edit.model import MulaDiT

logger = logging.getLogger(__name__)
GRAPH_WARMUP_STEPS = 3


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


@dataclass(kw_only=True)
class DiTInputs:
    noisy_latents: torch.Tensor
    context_latents: torch.Tensor
    symbols: torch.Tensor
    timestep: torch.Tensor
    cross_attention: list[AttentionKeysValues]

    def predict(self, decoder: MulaDiT) -> torch.Tensor:
        return decoder(
            self.noisy_latents,
            self.context_latents,
            self.symbols,
            self.timestep,
            self.cross_attention,
        )


class MulaSampler:
    """Reuse one fixed-shape decoder graph across steps and consecutive matching edits."""

    def __init__(
        self,
        decoder: MulaDiT,
        *,
        enable_torch_compile: bool,
        enable_cuda_graph: bool,
    ) -> None:
        self.decoder: MulaDiT = decoder
        self.enable_cuda_graph: bool = enable_cuda_graph
        self.inputs: DiTInputs | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.velocities: torch.Tensor | None = None
        if enable_torch_compile:
            # note (Codex): Keep BF16 rounding and leave graph ownership to the sampler.
            self.decoder.compile(
                fullgraph=True,
                dynamic=True,
                options={"triton.cudagraphs": False, "emulate_precision_casts": True},
            )
        else:
            pass

    @torch.inference_mode()
    def prepare(self, state: SamplingState) -> None:
        frames = state.latents.shape[1]
        noisy_latents = state.latents.expand(state.branches, -1, -1)
        if self.enable_cuda_graph and noisy_latents.device.type != "cuda":
            raise ValueError("Mula Edit CUDA Graph requires a CUDA decoder")
        else:
            pass
        if (
            self.inputs is None
            or self.inputs.noisy_latents.shape != noisy_latents.shape
            or self.inputs.cross_attention[0].keys.shape
            != state.cross_attention[0].keys.shape
        ):
            self.graph = None
            self.velocities = None
            self.inputs = DiTInputs(
                noisy_latents=noisy_latents.clone(),
                context_latents=state.context_latents.expand(
                    state.branches, -1, -1
                ).clone(),
                symbols=state.symbols.expand(state.branches, -1, -1).clone(),
                timestep=state.timesteps[:1].expand(state.branches).clone(),
                cross_attention=[
                    AttentionKeysValues(
                        keys=layer.keys.clone(), values=layer.values.clone()
                    )
                    for layer in state.cross_attention
                ],
            )
        else:
            self.inputs.noisy_latents.copy_(noisy_latents)
            self.inputs.context_latents.copy_(state.context_latents)
            self.inputs.symbols.copy_(state.symbols)
            self.inputs.timestep.copy_(state.timesteps[:1])
            for cached, layer in zip(
                self.inputs.cross_attention, state.cross_attention
            ):
                cached.keys.copy_(layer.keys)
                cached.values.copy_(layer.values)
        if self.enable_cuda_graph and self.graph is None:
            logger.info(
                f"Capturing music edit DiT: frames={frames}, branches={state.branches}"
            )
            capture_stream = torch.cuda.Stream(device=noisy_latents.device)
            current_stream = torch.cuda.current_stream(noisy_latents.device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                for _ in range(GRAPH_WARMUP_STEPS):
                    self.inputs.predict(self.decoder)
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(noisy_latents.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                self.velocities = self.inputs.predict(self.decoder)
            current_stream.wait_stream(capture_stream)
            self.graph = graph
        else:
            pass

    @torch.inference_mode()
    def advance(self, state: SamplingState) -> None:
        assert self.inputs is not None
        self.inputs.noisy_latents.copy_(state.latents)
        self.inputs.timestep.copy_(state.timesteps[state.step_index])
        if self.graph is None:
            velocities = self.inputs.predict(self.decoder)
        else:
            self.graph.replay()
            assert self.velocities is not None
            velocities = self.velocities
        velocity = velocities[:1]
        if state.branches == 2:
            difference = velocity - velocities[1:2]
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
