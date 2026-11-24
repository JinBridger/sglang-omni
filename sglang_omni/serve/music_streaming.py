# SPDX-License-Identifier: Apache-2.0
"""Music audio streaming with stereo PCM, WAV, and SSE output."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Literal

import anyio
import numpy as np
from fastapi import Request

from sglang_omni.client.audio import (
    DEFAULT_SAMPLE_RATE,
    apply_speed,
    encode_pcm,
    to_numpy,
)
from sglang_omni.client.client import Client
from sglang_omni.client.types import (
    UNKNOWN_FINISH_REASON,
    GenerateChunk,
    GenerateRequest,
)
from sglang_omni.serve.music_audio import music_generation_error
from sglang_omni.serve.protocol import SpeechStreamOutcome
from sglang_omni.serve.speech_stream_outcomes import SpeechStreamOutcomes
from sglang_omni.serve.speech_to_text import (
    abort_and_close_speech_to_text_stream,
    cancel_task_bounded,
    wait_for_request_disconnect,
)
from sglang_omni.serve.streaming import (
    ClosableStreamingResponse,
    close_async_iterator_if_supported,
)

logger = logging.getLogger(__name__)
PCM_SAMPLE_WIDTH_BYTES = 2
STREAMING_WAV_SIZE = 0xFFFFFFFF


def create_streaming_wav_header(sample_rate: int, channels: int) -> bytes:
    """Describe a PCM stream whose final size is not known yet."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        STREAMING_WAV_SIZE,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * PCM_SAMPLE_WIDTH_BYTES,
        channels * PCM_SAMPLE_WIDTH_BYTES,
        PCM_SAMPLE_WIDTH_BYTES * 8,
        b"data",
        STREAMING_WAV_SIZE,
    )


def select_music_audio_delta(
    audio_data: object,
    *,
    emitted_samples: int,
    is_terminal: bool,
) -> tuple[np.ndarray | None, int]:
    """Keep stereo channels and remove audio already sent before a terminal chunk."""
    audio = to_numpy(audio_data)
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 2 and audio.shape[0] in (1, 2):
        pass
    elif audio.ndim == 2 and audio.shape[-1] in (1, 2):
        audio = audio.T
    elif audio.ndim == 0:
        audio = audio.reshape(1)
    elif audio.ndim == 1:
        pass
    else:
        raise ValueError(
            f"Music streaming requires mono or stereo audio, got shape {audio.shape}"
        )
    total_samples = int(audio.shape[-1])
    if not is_terminal:
        return audio, emitted_samples + total_samples
    elif total_samples <= emitted_samples:
        return None, emitted_samples
    else:
        return audio[..., emitted_samples:], total_samples


def music_pcm_chunk_bytes(
    chunk: GenerateChunk,
    *,
    emitted_samples: int,
    speed: float,
) -> tuple[bytes | None, int, int, int]:
    sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
    audio_data, emitted_samples = select_music_audio_delta(
        chunk.audio_data,
        emitted_samples=emitted_samples,
        is_terminal=chunk.finish_reason is not None,
    )
    if audio_data is None or audio_data.size == 0:
        return None, emitted_samples, sample_rate, 0
    else:
        pass

    channels = int(audio_data.shape[0]) if audio_data.ndim == 2 else 1
    if speed != 1.0:
        audio_data, sample_rate = apply_speed(audio_data, speed, sample_rate)
    else:
        pass
    audio_bytes = encode_pcm(audio_data, sample_rate)
    if not audio_bytes:
        return None, emitted_samples, sample_rate, channels
    else:
        pass
    return audio_bytes, emitted_samples, sample_rate, channels


def create_music_sse_response(
    pcm_chunks: AsyncGenerator[bytes, None],
    *,
    stream_outcome: SpeechStreamOutcome,
    sample_rate: int,
    channels: int,
    response_format: Literal["pcm", "wav"],
    enable_speech_sse_named_events: bool,
    headers: dict[str, str],
) -> ClosableStreamingResponse:
    """Wrap PCM chunks in SSE events with terminal metadata."""

    def sse_frame(event_type: str, event_json: str) -> str:
        # note (Xin Gao): existing clients listen for unnamed "message" events.
        if enable_speech_sse_named_events:
            return f"event: {event_type}\ndata: {event_json}\n\n"
        else:
            return f"data: {event_json}\n\n"

    async def stream_events() -> AsyncIterator[str]:
        emitted_audio = False
        duration_seconds = 0.0
        try:
            async for pcm_bytes in pcm_chunks:
                audio_packets: tuple[bytes, ...]
                if not emitted_audio and response_format == "wav":
                    audio_packets = (
                        create_streaming_wav_header(sample_rate, channels),
                        pcm_bytes,
                    )
                else:
                    audio_packets = (pcm_bytes,)

                duration_seconds += len(pcm_bytes) / (
                    PCM_SAMPLE_WIDTH_BYTES * channels * sample_rate
                )
                for audio_packet in audio_packets:
                    delta = {
                        "type": "speech.audio.delta",
                        "audio": base64.b64encode(audio_packet).decode("ascii"),
                        "response_format": response_format,
                    }
                    emitted_audio = True
                    yield sse_frame(delta["type"], json.dumps(delta))

            final_usage = stream_outcome.usage
            usage: dict[str, int | None] | None
            if final_usage is None:
                usage = None
            else:
                input_tokens = (
                    final_usage.speech_input_tokens
                    if final_usage.speech_input_tokens is not None
                    else final_usage.prompt_tokens
                )
                output_tokens = (
                    final_usage.speech_output_tokens
                    if final_usage.speech_output_tokens is not None
                    else final_usage.completion_tokens
                )
                total_tokens = (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else final_usage.total_tokens
                )
                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            done = {
                "type": "speech.audio.done",
                "duration": round(duration_seconds, 3),
                "usage": usage,
                "finish_reason": stream_outcome.finish_reason,
            }
            if stream_outcome.sglext is not None:
                done["sglext"] = stream_outcome.sglext.model_dump()
            else:
                pass
            yield sse_frame(done["type"], json.dumps(done))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                f"Speech SSE stream failed for request {stream_outcome.request_id}"
            )
            failure = music_generation_error(exc)
            error: dict[str, str | int | bool | None] = {
                "message": failure.message,
                "type": failure.error_type,
                "param": failure.param,
                "code": (
                    failure.code if failure.code is not None else failure.status_code
                ),
            }
            if emitted_audio:
                error.update(partial_audio=True, action="discard")
            else:
                pass
            error_type = (
                "speech.audio.error" if enable_speech_sse_named_events else "error"
            )
            payload = {"type": error_type, "error": error}
            yield sse_frame(error_type, json.dumps(payload))
        finally:
            with anyio.CancelScope(shield=True):
                await pcm_chunks.aclose()

    return ClosableStreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers=headers,
    )


async def music_audio_response(
    request: Request,
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    speed: float,
    stream_format: Literal["audio", "sse"] = "audio",
    response_format: Literal["pcm", "wav"] = "pcm",
    *,
    speech_stream_outcomes: SpeechStreamOutcomes,
    enable_speech_sse_named_events: bool,
) -> ClosableStreamingResponse:
    """Stream PCM/WAV bytes or SSE after the first audio sets the headers."""
    emitted_samples = 0
    stream_outcome = SpeechStreamOutcome(
        request_id=request_id, finish_reason=UNKNOWN_FINISH_REASON
    )

    def record_terminal_chunk_state(chunk: GenerateChunk) -> None:
        # note (Yucheng Hu): the terminal chunk carries these; the last reported
        # value wins so the outcome can be served once the stream has ended.
        if chunk.reported_finish_reason:
            stream_outcome.finish_reason = chunk.reported_finish_reason
        else:
            pass
        if chunk.usage is not None:
            stream_outcome.usage = chunk.usage
        else:
            pass

    chunk_stream = client.generate(gen_req, request_id=request_id)
    first_audio_bytes: bytes | None = None
    stream_sample_rate: int | None = None
    stream_channels: int | None = None
    stream_completed = False
    stream_closed = False
    disconnect_task = asyncio.create_task(wait_for_request_disconnect(request))
    next_chunk_task: asyncio.Task[GenerateChunk] | None = None

    try:
        while True:
            next_chunk_task = asyncio.create_task(anext(chunk_stream))
            done, _ = await asyncio.wait(
                {next_chunk_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                if not next_chunk_task.done():
                    await cancel_task_bounded(next_chunk_task)
                else:
                    pass
                await abort_and_close_speech_to_text_stream(
                    client, request_id, chunk_stream
                )
                stream_closed = True
                raise asyncio.CancelledError
            else:
                pass

            try:
                chunk = next_chunk_task.result()
            except StopAsyncIteration:
                stream_completed = True
                break
            record_terminal_chunk_state(chunk)
            if chunk.audio_data is None:
                continue
            else:
                pass

            first_audio_bytes, emitted_samples, stream_sample_rate, stream_channels = (
                music_pcm_chunk_bytes(
                    chunk,
                    emitted_samples=emitted_samples,
                    speed=speed,
                )
            )
            if first_audio_bytes is not None:
                break
            else:
                pass

        if (
            first_audio_bytes is None
            or stream_sample_rate is None
            or stream_channels is None
        ):
            raise RuntimeError("No audio output generated from the pipeline.")
        elif stream_sample_rate <= 0:
            raise ValueError("Music stream sample rate must be positive")
        else:
            pass
    except asyncio.CancelledError:
        if not stream_closed:
            await abort_and_close_speech_to_text_stream(
                client, request_id, chunk_stream
            )
        else:
            pass
        raise
    except Exception:
        if not stream_completed:
            await abort_and_close_speech_to_text_stream(
                client, request_id, chunk_stream
            )
        else:
            await close_async_iterator_if_supported(chunk_stream)
        raise
    finally:
        if next_chunk_task is not None and not next_chunk_task.done():
            await cancel_task_bounded(next_chunk_task)
        else:
            pass
        if not disconnect_task.done():
            await cancel_task_bounded(disconnect_task)
        else:
            pass

    async def stream_audio() -> AsyncGenerator[bytes, None]:
        nonlocal emitted_samples
        active_request = True
        try:
            if stream_format == "audio" and response_format == "wav":
                yield create_streaming_wav_header(stream_sample_rate, stream_channels)
            else:
                pass
            yield first_audio_bytes

            async for chunk in chunk_stream:
                record_terminal_chunk_state(chunk)
                if chunk.audio_data is None:
                    continue
                else:
                    pass

                audio_bytes, emitted_samples, sample_rate, channels = (
                    music_pcm_chunk_bytes(
                        chunk,
                        emitted_samples=emitted_samples,
                        speed=speed,
                    )
                )
                if audio_bytes is None:
                    continue
                else:
                    pass
                if sample_rate != stream_sample_rate:
                    raise RuntimeError(
                        "Music stream sample rate changed from "
                        f"{stream_sample_rate} to {sample_rate}"
                    )
                elif channels != stream_channels:
                    raise RuntimeError(
                        "Music stream channel count changed from "
                        f"{stream_channels} to {channels}"
                    )
                else:
                    pass
                yield audio_bytes
            active_request = False
            if stream_format == "audio":
                speech_stream_outcomes.record(
                    request_id,
                    stream_outcome.finish_reason,
                    stream_outcome.usage,
                    stream_outcome.sglext,
                )
            else:
                pass
        finally:
            with anyio.CancelScope(shield=True):
                if active_request:
                    await abort_and_close_speech_to_text_stream(
                        client, request_id, chunk_stream
                    )
                else:
                    await close_async_iterator_if_supported(chunk_stream)

    headers = {
        "X-Request-Id": request_id,
        "X-Sample-Rate": str(stream_sample_rate),
        "X-Channels": str(stream_channels),
        "X-Bit-Depth": "16",
    }
    if stream_format == "sse":
        return create_music_sse_response(
            stream_audio(),
            stream_outcome=stream_outcome,
            sample_rate=stream_sample_rate,
            channels=stream_channels,
            response_format=response_format,
            enable_speech_sse_named_events=enable_speech_sse_named_events,
            headers=headers,
        )
    else:
        headers["X-SGLang-Omni-Speech-Id"] = request_id
        return ClosableStreamingResponse(
            stream_audio(), media_type=f"audio/{response_format}", headers=headers
        )
