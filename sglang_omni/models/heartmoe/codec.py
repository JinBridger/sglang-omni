# SPDX-License-Identifier: Apache-2.0
"""Incremental HeartCodec flow matching with latent context and overlap-add."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from types import MethodType

import torch
from safetensors.torch import load
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from sglang_omni.models.heartmoe.codec_model import HeartCodec, HeartCodecConfig
from sglang_omni.models.heartmoe.codec_transformer import LlamaTransformerBlock
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = 3840


@dataclass(kw_only=True)
class CodecState:
    codes: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 8), dtype=torch.long)
    )
    latent: torch.Tensor | None = None
    overlap: torch.Tensor | None = None
    waveform: list[torch.Tensor] = field(default_factory=list)
    chunks: int = 0
    emitted_samples: int = 0
    abort_event: Event = field(default_factory=Event)


class HeartCodecScheduler(StreamingSimpleScheduler):
    def __init__(
        self,
        model: HeartCodec,
        chunk_frames: int,
        overlap_frames: int,
        num_steps: int,
        guidance_scale: float,
        max_batch_size: int,
    ) -> None:
        super().__init__(compute_fn=None, max_batch_size=max_batch_size)
        self.can_batch_stream_chunks = max_batch_size > 1
        self.stream_chunk_batch_distinct_requests = True
        self.model = model
        self.device = next(model.parameters()).device
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.hop_frames = chunk_frames - overlap_frames
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.states: dict[str, CodecState] = {}

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return True

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if not isinstance(item.data, torch.Tensor):
            raise TypeError("HeartCodec expects a tensor of audio frames")
        else:
            pass
        state = self.states.setdefault(request_id, CodecState())
        state.codes = torch.cat((state.codes, item.data.cpu()), dim=0)
        messages = []
        try:
            while state.codes.shape[0] >= self.chunk_frames:
                waveform = self.decode_window(state, final=False)
                messages.append(self.audio_message(request_id, waveform))
        except InterruptedError:
            if state.abort_event.is_set():
                return []
            else:
                raise
        return messages

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        with self.state_lock:
            requests: dict[str, CodecState] = {}
            for request_id, item in items:
                if self.is_aborted(request_id):
                    continue
                elif not isinstance(item.data, torch.Tensor):
                    self.emit_error(
                        request_id,
                        TypeError("HeartCodec expects a tensor of audio frames"),
                    )
                    self.abort(request_id)
                else:
                    state = self.states.setdefault(request_id, CodecState())
                    state.codes = torch.cat((state.codes, item.data.cpu()), dim=0)
                    requests[request_id] = state
        while ready := [
            (request_id, state)
            for request_id, state in requests.items()
            if not state.abort_event.is_set()
            and state.codes.shape[0] >= self.chunk_frames
        ][: self.max_batch_size]:
            try:
                waveforms = self.decode_windows(
                    [state for _, state in ready], final=False
                )
            except InterruptedError:
                if all(state.abort_event.is_set() for _, state in ready):
                    return
                else:
                    raise
            except Exception as error:
                for request_id, state in ready:
                    if not state.abort_event.is_set():
                        self.emit_error(request_id, error)
                        self.abort(request_id)
                    else:
                        pass
                return
            else:
                for (request_id, state), waveform in zip(ready, waveforms):
                    if not state.abort_event.is_set():
                        self.outbox.put(self.audio_message(request_id, waveform))
                    else:
                        pass

    def decode_window(self, state: CodecState, *, final: bool) -> torch.Tensor:
        return self.decode_windows([state], final=final)[0]

    @torch.inference_mode()
    def decode_windows(
        self, states: list[CodecState], *, final: bool
    ) -> list[torch.Tensor]:
        torch.cuda.set_device(self.device)
        frame_counts = [
            min(state.codes.shape[0], self.chunk_frames) for state in states
        ]
        codes = torch.stack(
            [
                state.codes[:frame_count]
                .repeat((self.chunk_frames + frame_count - 1) // frame_count, 1)[
                    : self.chunk_frames
                ]
                .T
                for state, frame_count in zip(states, frame_counts)
            ]
        ).to(self.device)
        latent = torch.zeros(
            (len(states), self.chunk_frames * 2, 256), device=self.device
        )
        context_lengths = []
        noises = []
        for i, state in enumerate(states):
            context_frames = 0 if state.latent is None else self.overlap_frames * 2
            context_lengths.append(context_frames)
            if state.latent is not None:
                latent[i : i + 1, :context_frames] = state.latent
            else:
                pass
            # note (Codex): A request's noise must not depend on batch membership or row order.
            generator = torch.Generator(device=self.device).manual_seed(
                999 + state.chunks
            )
            noises.append(
                torch.randn(
                    (1, self.chunk_frames * 2, 256),
                    device=self.device,
                    generator=generator,
                )
            )
        latent = self.model.flow_matching.inference_codes(
            [codes],
            latent,
            self.chunk_frames * 2,
            0,
            guidance_scale=self.guidance_scale,
            num_steps=self.num_steps,
            disable_progress=True,
            scenario="other_seg",
            context_lengths=context_lengths,
            initial_noise=torch.cat(noises),
            abort_events=tuple(state.abort_event for state in states),
        )
        if all(state.abort_event.is_set() for state in states):
            raise InterruptedError("HeartCodec generation aborted")
        else:
            pass
        stereo_latent = (
            latent.reshape(len(states), self.chunk_frames * 2, 2, 128)
            .permute(0, 2, 3, 1)
            .reshape(len(states) * 2, 128, self.chunk_frames * 2)
        )
        waveforms = (
            self.model.scalar_model.decode(stereo_latent)
            .reshape(len(states), 2, -1)
            .float()
            .cpu()
        )
        overlap_samples = self.overlap_frames * SAMPLES_PER_FRAME
        outputs = []
        for i, (state, frame_count) in enumerate(zip(states, frame_counts)):
            waveform = waveforms[i]
            state.latent = latent[i : i + 1, -self.overlap_frames * 2 :].clone()
            if state.overlap is not None:
                ramp = torch.linspace(0, 1, overlap_samples)
                waveform[:, :overlap_samples] = (
                    state.overlap * (1 - ramp) + waveform[:, :overlap_samples] * ramp
                )
            else:
                pass
            if final:
                waveform = waveform[:, : frame_count * SAMPLES_PER_FRAME]
                state.codes = state.codes[:0]
            else:
                state.overlap = waveform[
                    :, self.hop_frames * SAMPLES_PER_FRAME :
                ].clone()
                waveform = waveform[:, : self.hop_frames * SAMPLES_PER_FRAME]
                state.codes = state.codes[self.hop_frames :]
            state.chunks += 1
            state.emitted_samples += waveform.shape[-1]
            state.waveform.append(waveform)
            outputs.append(waveform)
        if len(states) == 1 and states[0].abort_event.is_set():
            raise InterruptedError("HeartCodec generation aborted")
        else:
            return outputs

    def audio_message(self, request_id: str, waveform: torch.Tensor) -> OutgoingMessage:
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=audio_waveform_payload(
                waveform, sample_rate=SAMPLE_RATE, keep_channels=True
            ),
            metadata={"modality": "audio"},
        )

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.states[request_id]
        payload = self.stream_payloads[request_id]
        target_samples = payload.data["frames"] * SAMPLES_PER_FRAME
        remaining_samples = target_samples - state.emitted_samples
        messages = []
        if remaining_samples > 0:
            if (
                state.overlap is not None
                and remaining_samples <= state.overlap.shape[-1]
            ):
                waveform = state.overlap[:, :remaining_samples]
                state.waveform.append(waveform)
            else:
                try:
                    waveform = self.decode_window(state, final=True)
                except InterruptedError:
                    if state.abort_event.is_set():
                        return []
                    else:
                        raise
            messages.append(self.audio_message(request_id, waveform))
        else:
            pass
        waveform = torch.cat(state.waveform, dim=-1)
        payload.data.update(
            audio_waveform_payload(
                waveform, sample_rate=SAMPLE_RATE, modality="audio", keep_channels=True
            )
        )
        messages.append(
            OutgoingMessage(request_id=request_id, type="result", data=payload)
        )
        return messages

    def abort(self, request_id: str) -> None:
        state = self.states.get(request_id)
        if state is not None:
            state.abort_event.set()
        else:
            pass
        super().abort(request_id)

    def clear_stream_state(self, request_id: str) -> None:
        state = self.states.pop(request_id, None)
        if state is not None:
            state.abort_event.set()
        else:
            pass


def create_codec_executor(
    model_path: str,
    *,
    chunk_frames: int,
    overlap_frames: int,
    num_steps: int,
    guidance_scale: float,
    max_batch_size: int,
    compile_codec: bool,
    use_cuda_graph: bool,
    gpu_id: int | None = None,
    device: str | None = None,
) -> HeartCodecScheduler:
    device = resolve_concrete_device(device, gpu_id)
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = Path(resolve_checkpoint(model_path))
    config = json.loads((checkpoint / "config.json").read_text())
    if "heartcodec_config" not in config:
        raise ValueError("Bundle HeartCodec with scripts/pack_heartmoe_codec.py first")
    else:
        pass
    config = HeartCodecConfig(**config["heartcodec_config"])
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = {
        filename
        for name, filename in index["weight_map"].items()
        if name.startswith("heartcodec.")
    }
    model = HeartCodec(config)
    weights = {}
    for shard in sorted(shards):
        weights.update(
            {
                name.removeprefix("heartcodec."): tensor
                for name, tensor in load((checkpoint / shard).read_bytes()).items()
                if name.startswith("heartcodec.")
            }
        )
    model.load_state_dict(weights, strict=True)
    model = model.float().to(device).eval().requires_grad_(False)
    if compile_codec:
        for module in model.scalar_model.modules():
            if is_parametrized(module, "weight"):
                remove_parametrizations(module, "weight", leave_parametrized=True)
            else:
                pass
        compiled = torch.compile(LlamaTransformerBlock.forward, dynamic=True)
        for block in (
            *model.flow_matching.estimator.transformer_blocks,
            *model.flow_matching.estimator.transformer_blocks_2,
        ):
            block.forward = MethodType(compiled, block)
        model.scalar_model.decode = torch.compile(
            model.scalar_model.decode, dynamic=True
        )
    else:
        pass
    if use_cuda_graph:
        model.flow_matching.capture_estimator(
            chunk_frames * 2, guidance_scale, max_batch_size
        )
    else:
        pass
    scheduler = HeartCodecScheduler(
        model, chunk_frames, overlap_frames, num_steps, guidance_scale, max_batch_size
    )
    if compile_codec:
        with torch.inference_mode():
            for batch_size in range(1, max_batch_size + 1):
                latent = (
                    torch.zeros(batch_size, chunk_frames * 2, 2, 128, device=device)
                    .permute(0, 2, 3, 1)
                    .reshape(batch_size * 2, 128, chunk_frames * 2)
                )
                model.scalar_model.decode(latent)
    else:
        pass
    logger.info(
        f"HeartCodec ready max_batch_size={max_batch_size} compile_codec={compile_codec} use_cuda_graph={use_cuda_graph}"
    )
    return scheduler
