# SPDX-License-Identifier: Apache-2.0
"""Continuous decoding with bounded request slots and CPU audio preparation."""

import logging
import queue
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from sglang_omni.admission import QueueFullError
from sglang_omni.models.sheetsage2.model import BatchedDecoder, SheetSage2Model
from sglang_omni.models.sheetsage2.transcription import Transcription
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)
ABORTED_REQUEST_ID_LIMIT = 10000


class SheetSage2Scheduler:
    def __init__(
        self,
        model: SheetSage2Model,
        decoder: BatchedDecoder,
        max_batch_size: int,
        max_queued_requests: int,
        max_audio_seconds: float,
    ) -> None:
        self.model = model
        self.decoder = decoder
        self.max_batch_size = max_batch_size
        self.max_queued_requests = max_queued_requests
        self.max_audio_seconds = max_audio_seconds
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.pending: deque[StagePayload] = deque()
        self.preparing: dict[int, tuple[str, Future[Transcription]]] = {}
        self.exporting: dict[str, Future[StagePayload]] = {}
        self.active: dict[int, Transcription] = {}
        self.aborted: dict[str, None] = {}
        self.abort_lock = threading.Lock()
        self.running: bool = True
        self.grammar_masks: dict[
            tuple[bool, int, bool, int, str | None], torch.Tensor
        ] = {}

    def warm_up_serving_thread(self) -> None:
        torch.cuda.set_device(self.decoder.device)

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

    def receive(self, message: IncomingMessage) -> None:
        with self.abort_lock:
            is_aborted = message.request_id in self.aborted
        if message.type != "new_request" or is_aborted:
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
            len(self.pending)
            + len(self.preparing)
            + len(self.active)
            + len(self.exporting)
            >= self.max_batch_size + self.max_queued_requests
        ):
            self.outbox.put(
                OutgoingMessage(message.request_id, "error", data=QueueFullError())
            )
        else:
            self.pending.append(message.data)

    def emit_error(self, request_id: str, error: Exception) -> None:
        logger.exception(f"SheetSage2 request {request_id} failed")
        self.outbox.put(OutgoingMessage(request_id, "error", data=error))

    def settle(self, executor: ThreadPoolExecutor) -> None:
        with self.abort_lock:
            aborted = set(self.aborted)
        self.pending = deque(
            payload for payload in self.pending if payload.request_id not in aborted
        )
        for slot, request in list(self.active.items()):
            if request.payload.request_id in aborted:
                self.active.pop(slot)
            else:
                pass
        for slot, (request_id, future) in list(self.preparing.items()):
            if request_id in aborted:
                future.cancel()
                # note (Codex): A running CPU decoder retains its slot until it settles.
                if not future.done():
                    continue
                else:
                    self.preparing.pop(slot)
            elif future.done():
                self.preparing.pop(slot)
                try:
                    self.active[slot] = future.result()
                except Exception as error:
                    self.emit_error(request_id, error)
            else:
                pass
        for request_id, future in list(self.exporting.items()):
            if request_id in aborted:
                future.cancel()
                if not future.done():
                    continue
                else:
                    self.exporting.pop(request_id)
            elif future.done():
                self.exporting.pop(request_id)
                try:
                    self.outbox.put(
                        OutgoingMessage(request_id, "result", data=future.result())
                    )
                except Exception as error:
                    self.emit_error(request_id, error)
            else:
                pass
        for slot in range(self.max_batch_size):
            if not self.pending:
                break
            elif slot in self.active or slot in self.preparing:
                continue
            else:
                payload = self.pending.popleft()
                self.preparing[slot] = (
                    payload.request_id,
                    executor.submit(
                        Transcription.prepare,
                        payload,
                        self.model,
                        self.max_audio_seconds,
                    ),
                )

    def finish_steps(self, executor: ThreadPoolExecutor) -> None:
        for slot, request in list(self.active.items()):
            if request.done:
                self.active.pop(slot)
                self.exporting[request.payload.request_id] = executor.submit(
                    request.export, self.model
                )
            else:
                pass

    @torch.inference_mode()
    def advance(self, executor: ThreadPoolExecutor) -> None:
        for slot, request in list(self.active.items()):
            if request.needs_prefill:
                try:
                    token = request.prefill(self.model, self.decoder, slot)
                    request.accept(token, self.model)
                except Exception as error:
                    self.active.pop(slot)
                    self.emit_error(request.payload.request_id, error)
            else:
                pass
        self.finish_steps(executor)
        decoding = {
            slot: request
            for slot, request in self.active.items()
            if not request.needs_prefill
        }
        if not decoding:
            return
        else:
            pass
        batch_size = max(decoding) + 1
        token_ids = [self.model.tokenizer.sos_token] * batch_size
        positions = [0] * batch_size
        masks = [
            torch.ones(self.model.config.vocab_size, dtype=torch.bool)
        ] * batch_size
        for slot, request in decoding.items():
            token_ids[slot] = request.token_ids[-1]
            positions[slot] = len(request.token_ids) - 1
            grammar = request.grammar
            assert grammar is not None
            mask_key = (
                grammar.in_shift,
                grammar.shift_run,
                grammar.payload_count > 0,
                grammar.last_field_index,
                grammar.incomplete,
            )
            if mask_key not in self.grammar_masks:
                self.grammar_masks[mask_key] = grammar.allowed("cpu")
            else:
                pass
            masks[slot] = self.grammar_masks[mask_key]
        try:
            next_tokens = self.decoder.step(token_ids, positions, torch.stack(masks))
        except Exception as error:
            for slot, request in decoding.items():
                self.active.pop(slot)
                self.emit_error(request.payload.request_id, error)
            return
        for slot, request in decoding.items():
            try:
                request.accept(next_tokens[slot], self.model)
            except Exception as error:
                self.active.pop(slot)
                self.emit_error(request.payload.request_id, error)
        self.finish_steps(executor)

    def start(self) -> None:
        self.warm_up_serving_thread()
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="sheetsage2-cpu"
        ) as executor:
            try:
                while self.running:
                    while True:
                        try:
                            self.receive(self.inbox.get_nowait())
                        except queue.Empty:
                            break
                    self.settle(executor)
                    if self.active:
                        self.advance(executor)
                    else:
                        try:
                            self.receive(self.inbox.get(timeout=0.01))
                        except queue.Empty:
                            pass
            finally:
                for _, future in self.preparing.values():
                    future.cancel()
                for future in self.exporting.values():
                    future.cancel()
                self.active.clear()
                self.pending.clear()


def create_executor(
    model_path: str,
    *,
    base_model_path: str | None = None,
    dtype: str,
    max_batch_size: int,
    max_queued_requests: int,
    max_audio_seconds: float,
    enable_cuda_graph: bool,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SheetSage2Scheduler:
    resolved_device = torch.device(resolve_concrete_device(device, gpu_id))
    if resolved_device.type != "cuda":
        raise ValueError("SheetSage2 currently requires a CUDA device")
    else:
        torch.cuda.set_device(resolved_device)
    model = SheetSage2Model(model_path, base_model_path).to(resolved_device)
    decoder = BatchedDecoder(
        model, max_batch_size, getattr(torch, dtype), enable_cuda_graph
    )
    return SheetSage2Scheduler(
        model, decoder, max_batch_size, max_queued_requests, max_audio_seconds
    )
