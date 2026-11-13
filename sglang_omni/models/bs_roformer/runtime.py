# SPDX-License-Identifier: Apache-2.0
"""Local model loading, overlapping audio windows, and stem export."""

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import soundfile as sf
import torch
import yaml
from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import (
    Attention,
    BSRoformer,
)
from pydantic import BaseModel, Field
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F

from sglang_omni.proto.request import StagePayload


class AudioConfig(BaseModel):
    sample_rate: int = Field(gt=0)
    chunk_size: int = Field(gt=0)


class InferenceConfig(BaseModel):
    num_overlap: int = Field(ge=1)


class TrainingConfig(BaseModel):
    target_instrument: Literal["Vocals"]


class CheckpointConfig(BaseModel):
    audio: AudioConfig
    model: dict[str, int | float | bool | tuple[int, ...]]
    inference: InferenceConfig
    training: TrainingConfig


class SDPAttention(nn.Module):
    """Let PyTorch select flash attention on current CUDA architectures."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


@dataclass(kw_only=True)
class Separation:
    payload: StagePayload
    waveform: torch.Tensor
    vocals: torch.Tensor
    weights: torch.Tensor
    window_starts: list[int]

    @classmethod
    def prepare(
        cls,
        payload: StagePayload,
        sample_rate: int,
        chunk_samples: int,
        stride_samples: int,
        max_audio_seconds: float,
    ) -> "Separation":
        inputs = payload.request.inputs
        if payload.request.metadata.get("task") != "music_separate":
            raise ValueError("Music separation request: use /v1/music/separate")
        elif not isinstance(inputs, dict) or not isinstance(
            audio_bytes := inputs.get("audio_bytes"), bytes
        ):
            raise ValueError("Music separation request: expected uploaded audio bytes")
        else:
            pass
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as audio_file:
                source_sample_rate = audio_file.samplerate
                waveform = audio_file.read(
                    frames=int(max_audio_seconds * source_sample_rate) + 1,
                    dtype="float32",
                    always_2d=True,
                )
        except sf.LibsndfileError as error:
            raise ValueError(
                "Music separation request: could not decode the uploaded audio"
            ) from error
        if len(waveform) > max_audio_seconds * source_sample_rate:
            raise ValueError(
                f"Music separation request: audio exceeds {max_audio_seconds} seconds"
            )
        elif waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(
                "Music separation request: audio must be finite and nonempty"
            )
        elif waveform.shape[1] not in (1, 2):
            raise ValueError("Music separation request: expected mono or stereo audio")
        else:
            pass
        if source_sample_rate != sample_rate:
            divisor = math.gcd(source_sample_rate, sample_rate)
            waveform = resample_poly(
                waveform, sample_rate // divisor, source_sample_rate // divisor, axis=0
            )
        else:
            pass
        if waveform.shape[1] == 1:
            waveform = np.repeat(waveform, 2, axis=1)
        else:
            pass
        waveform_tensor = torch.from_numpy(waveform.T.copy())
        last_start = max(waveform_tensor.shape[-1] - chunk_samples, 0)
        window_starts = [*range(0, last_start, stride_samples), last_start]
        return cls(
            payload=payload,
            waveform=waveform_tensor,
            vocals=torch.zeros_like(waveform_tensor),
            weights=torch.zeros(waveform_tensor.shape[-1]),
            window_starts=window_starts,
        )

    def export(self, sample_rate: int) -> StagePayload:
        self.vocals.div_(self.weights)
        audio_format: Literal["FLAC", "WAV"]
        audio_subtype: Literal["PCM_16", "FLOAT"]
        if self.payload.request.metadata.get("format") == "flac":
            audio_format, audio_subtype = "FLAC", "PCM_16"
        else:
            audio_format, audio_subtype = "WAV", "FLOAT"
        archive_bytes = io.BytesIO()
        with ZipFile(archive_bytes, "w", compression=ZIP_STORED) as archive:
            for stem_name, waveform in (
                ("vocals", self.vocals),
                ("instrumental", self.waveform - self.vocals),
            ):
                audio_bytes = io.BytesIO()
                sf.write(
                    audio_bytes,
                    waveform.numpy().T,
                    sample_rate,
                    format=audio_format,
                    subtype=audio_subtype,
                )
                archive.writestr(
                    f"{stem_name}.{audio_format.lower()}", audio_bytes.getvalue()
                )
        return StagePayload(
            request_id=self.payload.request_id,
            request=self.payload.request,
            data={
                "media": [
                    {
                        "type": "file",
                        "mime_type": "application/zip",
                        "data": archive_bytes.getvalue(),
                    }
                ]
            },
        )


class BSRoFormerRuntime:
    def __init__(
        self,
        model_path: str,
        checkpoint_filename: str,
        device: torch.device,
        dtype: torch.dtype,
        enable_torch_compile: bool,
    ) -> None:
        checkpoint_path = Path(model_path) / checkpoint_filename
        with checkpoint_path.with_suffix(".yaml").open() as configuration_file:
            configuration = CheckpointConfig.model_validate(
                yaml.load(configuration_file, Loader=yaml.FullLoader)
            )
        if (
            configuration.model.get("stereo") is not True
            or configuration.model.get("num_stems") != 1
        ):
            raise ValueError(
                "Expected a stereo BS-RoFormer vocal separation checkpoint"
            )
        else:
            pass
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.sample_rate: int = configuration.audio.sample_rate
        self.chunk_samples: int = configuration.audio.chunk_size
        hop_samples = configuration.model["stft_hop_length"]
        if self.chunk_samples % hop_samples:
            raise ValueError(
                "audio.chunk_size must be divisible by model.stft_hop_length"
            )
        else:
            pass
        self.stride_samples: int = (
            self.chunk_samples // configuration.inference.num_overlap
        )
        if self.stride_samples < 1:
            raise ValueError("num_overlap must not exceed the audio chunk size")
        else:
            pass
        self.model: BSRoformer = BSRoformer(**configuration.model)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.attend = SDPAttention()
            elif isinstance(module, nn.Linear):
                module.to(dtype=dtype)
            else:
                pass
        self.model.to(device).eval().requires_grad_(False)
        if enable_torch_compile:
            self.model.band_split.compile(fullgraph=True, dynamic=False)
            for layer in self.model.layers:
                for transformer in layer:
                    transformer.compile(fullgraph=True, dynamic=False)
            for estimator in self.model.mask_estimators:
                estimator.compile(fullgraph=True, dynamic=False)
        else:
            pass
        self.window: torch.Tensor = torch.hamming_window(
            self.chunk_samples, periodic=False
        )

    @torch.inference_mode()
    def infer(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            return self.model(waveform.to(self.device)).float()

    def warm_up(self) -> None:
        waveform = torch.zeros(1, 2, self.chunk_samples, device=self.device)
        self.infer(waveform)
