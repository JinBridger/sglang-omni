# SPDX-License-Identifier: Apache-2.0
"""JSON validation, audio downloads, and failure handling for music endpoints."""

import logging

from fastapi import HTTPException, Request

from sglang_omni.client.client import Client
from sglang_omni.client.types import CompletionResult, GenerateRequest
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector
from sglang_omni.serve.openai_errors import generation_error_status_code
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    speech_generation_error,
)
from sglang_omni.serve.speech_to_text import complete_unless_disconnected
from sglang_omni.utils.audio import audio_request_timeout

logger = logging.getLogger(__name__)
MUSIC_BAD_REQUEST_MARKERS = (
    "Music separation request:",
    "Mula edit request:",
    "HeartMoE Cover requires",
    "HeartMoE Cover score contains",
    "HeartMoE score is only supported",
)


def music_generation_error(error: Exception) -> SpeechAPIError:
    if generation_error_status_code(error) == 500 and any(
        marker in str(error) for marker in MUSIC_BAD_REQUEST_MARKERS
    ):
        return bad_request(str(error))
    else:
        return speech_generation_error(error)


async def complete_music_request(
    request: Request,
    client: Client,
    generation: GenerateRequest,
    *,
    request_id: str,
) -> CompletionResult:
    try:
        return await complete_unless_disconnected(
            request,
            client,
            client.completion(generation, request_id=request_id),
            request_id=request_id,
        )
    except Exception as error:
        failure = music_generation_error(error)
        if failure.status_code == 500:
            logger.exception(f"Music request {request_id} failed")
        else:
            pass
        raise HTTPException(
            status_code=failure.status_code, detail=failure.message
        ) from error


async def require_music_json(request: Request) -> None:
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type == "application/json" or (
        content_type.startswith("application/") and content_type.endswith("+json")
    ):
        return
    else:
        raise HTTPException(
            status_code=415, detail="Content-Type must be application/json"
        )


async def download_music_audio(music_url: str, *, max_bytes: int | None) -> bytes:
    connector = get_global_resource_connector()
    try:
        audio_bytes, _ = await connector.load_http_bytes_async(
            music_url, timeout=audio_request_timeout(), max_bytes=max_bytes
        )
    except ValueError as error:
        # note (Codex): The connector reports its byte limit through ValueError.
        status_code = (
            413
            if max_bytes is not None
            and str(error) == f"Media URL response exceeds {max_bytes} bytes."
            else 400
        )
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    else:
        pass
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio URL returned an empty file")
    else:
        return audio_bytes
