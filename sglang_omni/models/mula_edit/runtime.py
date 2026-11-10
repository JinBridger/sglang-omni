# SPDX-License-Identifier: Apache-2.0
"""Resident music model, VAE, and text encoder for serial edits."""

from pathlib import Path
from typing import Literal

import torch
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from sglang_omni.models.mula_edit.audio import PreparedEdit, decode_audio, encode_audio
from sglang_omni.models.mula_edit.model import MulaModel
from sglang_omni.models.mula_edit.protocol import MulaEditRequest
from sglang_omni.models.mula_edit.sampling import MulaSampler, SamplingState
from sglang_omni.models.mula_edit.symbols import FRAMES_PER_SECOND


class MulaEditRuntime:
    def __init__(
        self,
        model_path: str,
        checkpoint_dir: str,
        device: torch.device,
        dtype: torch.dtype,
        *,
        attention_backend: Literal["auto", "sdpa"],
        enable_torch_compile: bool,
        enable_cuda_graph: bool,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model = MulaModel.load(model_path, device, dtype)
        self.model.decoder.configure_attention(attention_backend)
        self.sampler: MulaSampler = MulaSampler(
            self.model.decoder,
            enable_torch_compile=enable_torch_compile,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.silence = (
            torch.load(
                Path(model_path) / "silence_latent.pt",
                map_location="cpu",
                weights_only=True,
            )
            .transpose(1, 2)
            .to(device=device, dtype=dtype)
        )
        directory = Path(checkpoint_dir)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            directory / "Qwen3-Embedding-0.6B"
        )
        self.text_encoder = (
            Qwen3Model.from_pretrained(
                directory / "Qwen3-Embedding-0.6B",
                dtype=dtype,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.vae = (
            AutoencoderOobleck.from_pretrained(directory / "vae", torch_dtype=dtype)
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    @torch.inference_mode()
    def encode_conditions(self, request: MulaEditRequest, frames: int) -> torch.Tensor:
        prompt = (
            "# Instruction\nRepaint the mask area based on the given conditions and the symbolic lead sheet and multi-track conditions:\n\n"
            f"# Caption\n{request.caption}\n\n# Metas\n"
            f"- bpm: {request.bpm or 'N/A'}\n"
            f"- timesignature: {request.timesignature or 'N/A'}\n"
            f"- keyscale: {request.keyscale or 'N/A'}\n"
            f"- duration: {frames // FRAMES_PER_SECOND} seconds\n<|endoftext|>\n"
        )
        text_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        lyric_tokens = self.tokenizer(
            request.lyrics,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        # note (Codex): Training encoded padded prompt tokens without an attention mask.
        text_embeddings = self.text_encoder(text_tokens.input_ids).last_hidden_state.to(
            self.dtype
        )
        lyric_embeddings = self.text_encoder.embed_tokens(lyric_tokens.input_ids).to(
            self.dtype
        )
        return self.model.encoder(
            text_embeddings,
            text_tokens.attention_mask.to(self.dtype),
            lyric_embeddings,
            lyric_tokens.attention_mask.to(self.dtype),
        )

    @torch.inference_mode()
    def begin(self, prepared: PreparedEdit) -> SamplingState:
        request = prepared.request
        clean = encode_audio(self.vae, prepared.waveform, request.seed).to(self.dtype)
        frames = clean.shape[1]
        if frames > self.silence.shape[1]:
            raise ValueError("Recording exceeds the checkpoint's silence latent length")
        else:
            pass
        first = int(request.start_seconds * FRAMES_PER_SECOND)
        last = min(frames, max(first + 1, int(request.end_seconds * FRAMES_PER_SECOND)))
        mask = (torch.arange(frames, device=self.device) >= first) & (
            torch.arange(frames, device=self.device) < last
        )
        mask = mask.unsqueeze(0)
        source = torch.where(mask.unsqueeze(-1), self.silence[:, :frames], clean)
        context = torch.cat(
            (source, mask.unsqueeze(-1).expand_as(clean).to(self.dtype)), dim=-1
        )
        symbols = self.model.decoder.symbolic_encoder(
            prepared.events.to(self.device), prepared.provided.to(self.device)
        )
        conditions = self.encode_conditions(request, frames)
        if request.guidance_scale > 1:
            conditions = torch.cat(
                (conditions, self.model.null_condition_emb.expand_as(conditions))
            )
        else:
            pass
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        noise = torch.randn(
            clean.shape, device=self.device, dtype=self.dtype, generator=generator
        )
        timesteps = torch.linspace(
            1, 0, request.steps + 1, device=self.device, dtype=self.dtype
        )
        timesteps = request.shift * timesteps / (1 + (request.shift - 1) * timesteps)
        state = SamplingState(
            clean_latents=clean,
            noise=noise,
            latents=noise,
            context_latents=context,
            symbols=symbols,
            mask=mask,
            timesteps=timesteps,
            cross_attention=self.model.decoder.prepare_cross_attention(conditions),
            guidance_scale=request.guidance_scale,
            crossfade_frames=int(request.crossfade_seconds * FRAMES_PER_SECOND),
        )
        self.sampler.prepare(state)
        return state

    def advance(self, state: SamplingState) -> None:
        self.sampler.advance(state)

    @torch.inference_mode()
    def finish(self, state: SamplingState, samples: int) -> torch.Tensor:
        return decode_audio(self.vae, state.finish(), samples)
