# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ACEStep
"""Bounded audio decoding and overlapping VAE encode/decode windows."""

import base64
import binascii
import io
import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
from scipy.signal import resample_poly
from torch.nn import functional as F

from sglang_omni.models.mula_edit.protocol import MAX_AUDIO_BYTES, MulaEditRequest
from sglang_omni.models.mula_edit.symbols import score_events

SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = 1920
ENCODE_SAMPLES = 15 * SAMPLE_RATE
ENCODE_OVERLAP_SAMPLES = 2 * SAMPLE_RATE
DECODE_FRAMES = 512
DECODE_OVERLAP_FRAMES = 64


@dataclass(kw_only=True)
class PreparedEdit:
    request: MulaEditRequest
    waveform: torch.Tensor
    samples: int
    events: torch.Tensor
    provided: torch.Tensor


def prepare_audio(request: MulaEditRequest, max_audio_seconds: float) -> PreparedEdit:
    try:
        encoded = base64.b64decode(request.audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(
            "Mula edit request: audio_base64 must contain a valid base64 audio file"
        ) from error
    if len(encoded) > MAX_AUDIO_BYTES:
        raise ValueError("Mula edit request: Encoded audio exceeds the 128 MiB limit")
    else:
        pass
    try:
        with sf.SoundFile(io.BytesIO(encoded)) as audio_file:
            sample_rate = audio_file.samplerate
            if audio_file.frames / sample_rate > max_audio_seconds:
                raise ValueError(
                    f"Mula edit request: Audio exceeds max_audio_seconds={max_audio_seconds}"
                )
            else:
                waveform = audio_file.read(dtype="float32", always_2d=True)
    except sf.LibsndfileError as error:
        raise ValueError(
            "Mula edit request: Could not decode audio_base64 as an audio file"
        ) from error
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(
            "Mula edit request: Audio must contain finite, nonempty samples"
        )
    elif request.end_seconds > len(waveform) / sample_rate:
        raise ValueError(
            "Mula edit request: The edit interval extends beyond the recording"
        )
    else:
        pass
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        waveform = resample_poly(
            waveform, SAMPLE_RATE // divisor, sample_rate // divisor, axis=0
        )
    else:
        pass
    if waveform.shape[1] == 1:
        waveform = np.repeat(waveform, 2, axis=1)
    else:
        waveform = waveform[:, :2]
    samples = len(waveform)
    frames = math.ceil(samples / SAMPLES_PER_FRAME)
    waveform_tensor = F.pad(
        torch.from_numpy(waveform.T.copy()), (0, frames * SAMPLES_PER_FRAME - samples)
    )
    events, provided = score_events(request, frames)
    return PreparedEdit(
        request=request,
        waveform=waveform_tensor,
        samples=samples,
        events=events,
        provided=provided,
    )


@torch.inference_mode()
def encode_audio(
    vae: AutoencoderOobleck, waveform: torch.Tensor, seed: int
) -> torch.Tensor:
    samples = waveform.shape[-1]
    generator = torch.Generator(device=vae.device).manual_seed(seed)
    if samples <= ENCODE_SAMPLES:
        return (
            vae.encode(waveform[None].to(vae.device, vae.dtype))
            .latent_dist.sample(generator=generator)
            .transpose(1, 2)
        )
    else:
        pass
    stride = ENCODE_SAMPLES - 2 * ENCODE_OVERLAP_SAMPLES
    pieces: list[torch.Tensor] = []
    for first in range(0, samples, stride):
        last = min(first + stride, samples)
        window_first = max(0, first - ENCODE_OVERLAP_SAMPLES)
        window_last = min(samples, last + ENCODE_OVERLAP_SAMPLES)
        audio_window = waveform[None, :, window_first:window_last].to(
            vae.device, vae.dtype
        )
        latents = vae.encode(audio_window).latent_dist.sample(generator=generator)
        trim_first = (first - window_first) // SAMPLES_PER_FRAME
        trim_last = (last - window_first) // SAMPLES_PER_FRAME
        pieces.append(latents[:, :, trim_first:trim_last])
    return torch.cat(pieces, dim=-1).transpose(1, 2)


@torch.inference_mode()
def decode_audio(
    vae: AutoencoderOobleck, latents: torch.Tensor, samples: int
) -> torch.Tensor:
    latents = latents.transpose(1, 2).contiguous().to(vae.device, vae.dtype)
    frames = latents.shape[-1]
    if frames <= DECODE_FRAMES:
        return vae.decode(latents).sample[0, :, :samples].float().cpu()
    else:
        pass
    stride = DECODE_FRAMES - 2 * DECODE_OVERLAP_FRAMES
    pieces: list[torch.Tensor] = []
    for first in range(0, frames, stride):
        last = min(first + stride, frames)
        window_first = max(0, first - DECODE_OVERLAP_FRAMES)
        window_last = min(frames, last + DECODE_OVERLAP_FRAMES)
        waveform = vae.decode(latents[:, :, window_first:window_last]).sample
        trim_first = (first - window_first) * SAMPLES_PER_FRAME
        trim_last = (last - window_first) * SAMPLES_PER_FRAME
        pieces.append(waveform[0, :, trim_first:trim_last].float().cpu())
    return torch.cat(pieces, dim=-1)[:, :samples]
