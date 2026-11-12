# SPDX-License-Identifier: Apache-2.0
"""Bounded FIFO admission and serial diffusion edits."""

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch

from sglang_omni.admission import QueueFullError
from sglang_omni.models.mula_edit.audio import SAMPLE_RATE, prepare_audio
from sglang_omni.models.mula_edit.protocol import MulaEditRequest
from sglang_omni.models.mula_edit.runtime import MulaEditRuntime
from sglang_omni.models.mula_edit.sampling import SamplingState
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)
ABORTED_REQUEST_ID_LIMIT = 10000


@dataclass(kw_only=True)
class ActiveEdit:
    payload: StagePayload
    state: SamplingState
    samples: int


class MulaEditScheduler:
    def __init__(
        self,
        runtime: MulaEditRuntime,
        max_queued_requests: int,
        max_audio_seconds: float,
    ) -> None:
        self.runtime: MulaEditRuntime = runtime
        self.max_queued_requests: int = max_queued_requests
        self.max_audio_seconds: float = max_audio_seconds
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.pending: deque[StagePayload] = deque()
        self.active: ActiveEdit | None = None
        self.aborted: dict[str, None] = {}
        self.abort_lock: threading.Lock = threading.Lock()
        self.running: bool = True

    def warm_up_serving_thread(self) -> None:
        torch.cuda.set_device(self.runtime.device)

    def abort(self, request_id: str) -> None:
        with self.abort_lock:
            self.aborted[request_id] = None
            if len(self.aborted) > ABORTED_REQUEST_ID_LIMIT:
                self.aborted.pop(next(iter(self.aborted)))
            else:
                pass
        self.inbox.put(IncomingMessage(request_id, "wakeup"))

    def stop(self) -> None:
        self.running = False
        self.inbox.put(IncomingMessage("", "wakeup"))

    def emit_error(self, request_id: str, error: Exception) -> None:
        logger.exception(f"Music edit {request_id} failed")
        self.outbox.put(OutgoingMessage(request_id, "error", data=error))

    def receive(self, message: IncomingMessage) -> None:
        with self.abort_lock:
            aborted = message.request_id in self.aborted
        if message.type != "new_request" or aborted:
            return
        elif not isinstance(message.data, StagePayload):
            self.outbox.put(
                OutgoingMessage(
                    message.request_id,
                    "error",
                    data=ValueError("Expected StagePayload"),
                )
            )
        elif (
            len(self.pending) + int(self.active is not None)
            >= 1 + self.max_queued_requests
        ):
            self.outbox.put(
                OutgoingMessage(message.request_id, "error", data=QueueFullError())
            )
        else:
            self.pending.append(message.data)

    def is_cancelled(self, request_id: str) -> bool:
        with self.abort_lock:
            return not self.running or request_id in self.aborted

    def settle(self) -> None:
        with self.abort_lock:
            aborted = set(self.aborted)
        self.pending = deque(
            payload for payload in self.pending if payload.request_id not in aborted
        )
        if self.active is not None and self.active.payload.request_id in aborted:
            self.active = None
        else:
            pass
        if self.active is None and self.pending and self.running:
            payload = self.pending.popleft()
            try:
                if payload.request.metadata.get("task") != "mula_edit":
                    raise ValueError(
                        "This model accepts requests through /v1/music/edit"
                    )
                else:
                    request = MulaEditRequest.model_validate(payload.request.inputs)
                prepared = prepare_audio(request, self.max_audio_seconds)
                if self.is_cancelled(payload.request_id):
                    return
                else:
                    self.active = ActiveEdit(
                        payload=payload,
                        state=self.runtime.begin(prepared),
                        samples=prepared.samples,
                    )
            except Exception as error:
                self.emit_error(payload.request_id, error)
        else:
            pass

    def advance(self) -> None:
        edit = self.active
        assert edit is not None
        if self.is_cancelled(edit.payload.request_id):
            self.active = None
            return
        else:
            pass
        try:
            self.runtime.advance(edit.state)
            if self.is_cancelled(edit.payload.request_id):
                self.active = None
                return
            elif not edit.state.done:
                return
            else:
                self.active = None
            waveform = self.runtime.finish(edit.state, edit.samples).numpy()
            if self.is_cancelled(edit.payload.request_id):
                return
            else:
                pass
            result = StagePayload(
                request_id=edit.payload.request_id,
                request=edit.payload.request,
                data={
                    "audio_waveform": waveform.tobytes(),
                    "audio_waveform_dtype": "float32",
                    "audio_waveform_shape": list(waveform.shape),
                    "sample_rate": SAMPLE_RATE,
                },
            )
            self.outbox.put(
                OutgoingMessage(edit.payload.request_id, "result", data=result)
            )
        except Exception as error:
            self.active = None
            self.emit_error(edit.payload.request_id, error)

    def start(self) -> None:
        self.warm_up_serving_thread()
        try:
            while self.running:
                while True:
                    try:
                        self.receive(self.inbox.get_nowait())
                    except queue.Empty:
                        break
                self.settle()
                if self.active is not None:
                    self.advance()
                else:
                    try:
                        self.receive(self.inbox.get(timeout=0.01))
                    except queue.Empty:
                        pass
        finally:
            self.active = None
            self.pending.clear()


def create_executor(
    model_path: str,
    *,
    checkpoint_dir: str | None,
    dtype: Literal["bfloat16", "float32"],
    attention_backend: Literal["auto", "sdpa"],
    enable_torch_compile: bool,
    enable_cuda_graph: bool,
    max_queued_requests: int,
    max_audio_seconds: float,
    device: str | None = None,
    gpu_id: int | None = None,
) -> MulaEditScheduler:
    if checkpoint_dir is None:
        raise ValueError(
            "Set edit.factory.checkpoint_dir to the directory containing vae and Qwen3-Embedding-0.6B"
        )
    else:
        resolved_device = torch.device(resolve_concrete_device(device, gpu_id))
    if resolved_device.type != "cuda":
        raise ValueError("Mula Edit currently requires a CUDA device")
    else:
        torch.cuda.set_device(resolved_device)
    runtime = MulaEditRuntime(
        model_path,
        checkpoint_dir,
        resolved_device,
        getattr(torch, dtype),
        attention_backend=attention_backend,
        enable_torch_compile=enable_torch_compile,
        enable_cuda_graph=enable_cuda_graph,
    )
    return MulaEditScheduler(runtime, max_queued_requests, max_audio_seconds)
