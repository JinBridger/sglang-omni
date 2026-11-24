# SPDX-License-Identifier: Apache-2.0
"""JSON request validation and audio downloads for music endpoints."""

from fastapi import HTTPException, Request

from sglang_omni.preprocessing.resource_connector import (
    MediaSizeLimitError,
    get_global_resource_connector,
)
from sglang_omni.utils.audio import audio_request_timeout


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
    except MediaSizeLimitError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    else:
        pass
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio URL returned an empty file")
    else:
        return audio_bytes
