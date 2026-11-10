# SPDX-License-Identifier: Apache-2.0
"""Serial request execution through the shared scheduler."""

from typing import Literal

import torch
from torch.nn import functional as F

from sglang_omni.models.bs_roformer.runtime import BSRoFormerRuntime, Separation
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device


class BSRoFormerScheduler(SimpleScheduler[StagePayload, StagePayload]):
    def __init__(self, runtime: BSRoFormerRuntime, max_audio_seconds: float) -> None:
        self.runtime: BSRoFormerRuntime = runtime
        self.max_audio_seconds: float = max_audio_seconds
        super().__init__(self.separate)

    def warm_up_serving_thread(self) -> None:
        if self.runtime.device.type == "cuda":
            torch.cuda.set_device(self.runtime.device)
        else:
            pass
        self.runtime.warm_up()

    def separate(self, payload: StagePayload) -> StagePayload:
        separation = Separation.prepare(
            payload,
            self.runtime.sample_rate,
            self.runtime.chunk_samples,
            self.runtime.stride_samples,
            self.max_audio_seconds,
        )
        for start_sample in separation.window_starts:
            if self.is_aborted(payload.request_id):
                raise InterruptedError(
                    f"Music separation {payload.request_id} was cancelled"
                )
            else:
                pass
            waveform = separation.waveform[
                :, start_sample : start_sample + self.runtime.chunk_samples
            ]
            valid_samples = waveform.shape[-1]
            waveform = F.pad(waveform, (0, self.runtime.chunk_samples - valid_samples))
            prediction = self.runtime.infer(waveform[None]).cpu()[0, :, :valid_samples]
            end_sample = start_sample + valid_samples
            separation.vocals[:, start_sample:end_sample].add_(
                prediction * self.runtime.window[:valid_samples]
            )
            separation.weights[start_sample:end_sample].add_(
                self.runtime.window[:valid_samples]
            )
        return separation.export(self.runtime.sample_rate)


def create_executor(
    model_path: str,
    *,
    checkpoint_filename: str,
    dtype: Literal["bfloat16", "float16", "float32"],
    max_audio_seconds: float,
    enable_torch_compile: bool,
    device: str | None = None,
    gpu_id: int | None = None,
) -> BSRoFormerScheduler:
    resolved_device = resolve_concrete_device(device, gpu_id)
    if resolved_device.type != "cuda":
        raise ValueError("BS-RoFormer serving currently requires CUDA")
    else:
        torch.cuda.set_device(resolved_device)
    runtime = BSRoFormerRuntime(
        model_path,
        checkpoint_filename,
        resolved_device,
        getattr(torch, dtype),
        enable_torch_compile,
    )
    return BSRoFormerScheduler(runtime, max_audio_seconds)
