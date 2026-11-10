# SPDX-License-Identifier: Apache-2.0
"""HeartCodec flow matching."""

from threading import Event

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    DiffusionBreakableCudaGraphRunner,
)

from sglang_omni.models.heartmoe.codec_transformer import LlamaTransformer


class ResidualVQ(nn.Module):
    """Decode residual codebooks while retaining the checkpoint's tensor names."""

    def __init__(
        self, dim: int, codebook_size: int, codebook_dim: int, num_quantizers: int
    ) -> None:
        super().__init__()
        self.project_in = nn.Linear(dim, codebook_dim)
        self.project_out = nn.Linear(codebook_dim, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_quantizers):
            codebook = nn.Module()
            codebook.register_buffer(
                "embed", torch.zeros(1, codebook_size, codebook_dim)
            )
            codebook.register_buffer(
                "embed_avg", torch.zeros(1, codebook_size, codebook_dim)
            )
            codebook.register_buffer("cluster_size", torch.zeros(1, codebook_size))
            codebook.register_buffer("initted", torch.zeros(1))
            layer = nn.Module()
            layer.add_module("_codebook", codebook)
            self.layers.append(layer)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codes = torch.stack(
            [
                F.embedding(indices[..., i], layer.get_submodule("_codebook").embed[0])
                for i, layer in enumerate(self.layers)
            ]
        )
        return self.project_out(codes.sum(dim=0))


class FlowMatching(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 8192,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.vq_embed = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            num_quantizers=num_quantizers,
        )
        self.cond_feature_emb = nn.Linear(dim, dim)
        self.zero_cond_embedding1 = nn.Parameter(torch.randn(dim))
        self.estimator = LlamaTransformer(
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            norm_type=norm_type,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            num_layers_2=num_layers_2,
            out_channels=out_channels,
        )
        self.latent_dim = out_channels
        self.estimator_runner: DiffusionBreakableCudaGraphRunner | None = None

    @torch.inference_mode()
    def capture_estimator(
        self, latent_frames: int, guidance_scale: float, max_batch_size: int
    ) -> None:
        device = next(self.parameters()).device
        runner = DiffusionBreakableCudaGraphRunner(self.estimator, device)
        for batch_size in range(1, max_batch_size + 1):
            guided_batch_size = batch_size * (2 if guidance_scale > 1 else 1)
            captured = runner.capture(
                hidden_states=torch.zeros(
                    guided_batch_size,
                    latent_frames,
                    self.estimator.in_channels,
                    device=device,
                ),
                timestep=torch.zeros(guided_batch_size, device=device),
            )
            if not captured:
                raise RuntimeError(
                    "HeartCodec could not capture its estimator CUDA graph"
                )
            else:
                pass
        self.estimator_runner = runner

    @torch.inference_mode()
    def inference_codes(
        self,
        codes: list[torch.Tensor],
        true_latents: torch.Tensor,
        latent_length: int,
        incontext_length: int,
        guidance_scale: float = 2.0,
        num_steps: int = 20,
        disable_progress: bool = True,
        scenario: str = "start_seg",
        abort_event: Event | None = None,
        context_lengths: list[int] | None = None,
        initial_noise: torch.Tensor | None = None,
        abort_events: tuple[Event, ...] = (),
    ) -> torch.Tensor:
        device = true_latents.device
        dtype = true_latents.dtype
        codes_bestrq_emb = codes[0]
        batch_size = codes_bestrq_emb.shape[0]
        self.vq_embed.eval()
        quantized_feature_emb = self.vq_embed.get_output_from_indices(
            codes_bestrq_emb.transpose(1, 2)
        )
        quantized_feature_emb = self.cond_feature_emb(quantized_feature_emb)
        quantized_feature_emb = F.interpolate(
            quantized_feature_emb.permute(0, 2, 1), scale_factor=2, mode="nearest"
        ).permute(0, 2, 1)
        num_frames = quantized_feature_emb.shape[1]
        if initial_noise is None:
            latents = torch.randn(
                (batch_size, num_frames, self.latent_dim), device=device, dtype=dtype
            )
        else:
            latents = initial_noise
        latent_masks = torch.zeros(
            latents.shape[0], latents.shape[1], dtype=torch.int64, device=latents.device
        )
        latent_masks[:, 0:latent_length] = 2
        if scenario == "other_seg":
            if context_lengths is None:
                latent_masks[:, 0:incontext_length] = 1
            else:
                context_mask = (
                    torch.arange(num_frames, device=device)[None]
                    < torch.tensor(context_lengths, device=device)[:, None]
                )
                latent_masks = torch.where(context_mask, 1, latent_masks)
        else:
            pass
        quantized_feature_emb = (latent_masks > 0.5).unsqueeze(
            -1
        ) * quantized_feature_emb + (latent_masks < 0.5).unsqueeze(
            -1
        ) * self.zero_cond_embedding1.unsqueeze(
            0
        )
        context_mask = ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1)
        incontext_latents = true_latents * context_mask.float()
        additional_model_input = torch.cat([quantized_feature_emb], 1)
        temperature = 1.0
        t_span = torch.linspace(
            0, 1, num_steps + 1, device=quantized_feature_emb.device
        )
        latents = self.solve_euler(
            latents * temperature,
            incontext_latents.to(dtype),
            context_mask,
            t_span,
            additional_model_input,
            guidance_scale,
            abort_event,
            abort_events,
        )
        return torch.where(context_mask, incontext_latents, latents)

    def solve_euler(
        self,
        x: torch.Tensor,
        incontext_x: torch.Tensor,
        context_mask: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        guidance_scale: float,
        abort_event: Event | None,
        abort_events: tuple[Event, ...],
    ) -> torch.Tensor:
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
        """
        t, _, dt = (t_span[0], t_span[-1], t_span[1] - t_span[0])
        noise = x.clone()
        sol = []
        estimator = self.estimator_runner or self.estimator
        for step in range(1, len(t_span)):
            if (abort_event is not None and abort_event.is_set()) or (
                abort_events and all(event.is_set() for event in abort_events)
            ):
                raise InterruptedError("HeartCodec generation aborted")
            else:
                pass
            # note (Codex): Tensor masks support mixed stream positions without host synchronization.
            x = torch.where(
                context_mask,
                (1 - (1 - 1e-06) * t) * noise + t * incontext_x,
                x,
            )
            if guidance_scale > 1.0:
                dphi_dt = estimator(
                    hidden_states=torch.cat(
                        [
                            torch.cat([x, x], 0),
                            torch.cat([incontext_x, incontext_x], 0),
                            torch.cat([torch.zeros_like(mu), mu], 0),
                        ],
                        2,
                    ),
                    timestep=t.unsqueeze(-1).repeat(2 * x.shape[0]),
                )
                dphi_dt_uncond, dhpi_dt_cond = dphi_dt.chunk(2, 0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (
                    dhpi_dt_cond - dphi_dt_uncond
                )
            else:
                dphi_dt = estimator(
                    hidden_states=torch.cat([x, incontext_x, mu], 2),
                    timestep=t.unsqueeze(-1).repeat(x.shape[0]),
                )
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            else:
                pass
        result = sol[-1]
        return result
