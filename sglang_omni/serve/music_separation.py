# SPDX-License-Identifier: Apache-2.0
"""Multipart music uploads with a ZIP response containing two stems."""

import uuid
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from sglang_omni.client.types import GenerateRequest
from sglang_omni.serve.speech_to_text import complete_speech_to_text_request

MAX_AUDIO_BYTES = 128 * 1024 * 1024


def register_music_separation(app: FastAPI) -> None:
    @app.post(
        "/v1/music/separate",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                }
            }
        },
    )
    async def separate_music(
        request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
    ) -> Response:
        if "BSRoFormer" not in app.state.architectures:
            raise HTTPException(
                status_code=400, detail="This model does not support music separation"
            )
        elif model is not None and model != app.state.model_name:
            raise HTTPException(
                status_code=404, detail=f"Model {model!r} is not served"
            )
        else:
            pass
        audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        elif len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413, detail="Uploaded audio exceeds 128 MiB"
            )
        else:
            pass
        request_id = f"music-separate-{uuid.uuid4()}"
        generation = GenerateRequest(
            model=model or app.state.model_name,
            prompt={"audio_bytes": audio_bytes},
            stream=False,
            metadata={"task": "music_separate"},
        )
        result = await complete_speech_to_text_request(
            request,
            app.state.client,
            generation,
            request_id=request_id,
            error_log_message="Music separation failed for %s",
        )
        if not result.media or not isinstance(
            archive_bytes := result.media[0].get("data"), bytes
        ):
            raise HTTPException(
                status_code=500, detail="Model returned no separated audio"
            )
        else:
            return Response(
                content=archive_bytes,
                media_type="application/zip",
                headers={
                    "Content-Disposition": 'attachment; filename="separated.zip"',
                    "X-Request-Id": request_id,
                },
            )
