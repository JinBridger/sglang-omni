# SPDX-License-Identifier: Apache-2.0
"""Music separation from audio URLs with a ZIP response containing two stems."""

import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from sglang_omni.client.types import GenerateRequest
from sglang_omni.serve.music_audio import download_music_audio, require_music_json
from sglang_omni.serve.speech_to_text import complete_speech_to_text_request

MAX_AUDIO_BYTES = 128 * 1024 * 1024


class MusicSeparationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    music_url: AnyHttpUrl
    model: str | None = None
    output_format: str = Field(default="wav", alias="format")


def register_music_separation(app: FastAPI) -> None:
    @app.post(
        "/v1/music/separate",
        dependencies=[Depends(require_music_json)],
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
        body: MusicSeparationRequest,
        request: Request,
    ) -> Response:
        if "BSRoFormer" not in app.state.architectures:
            raise HTTPException(
                status_code=400, detail="This model does not support music separation"
            )
        elif body.model is not None and body.model != app.state.model_name:
            raise HTTPException(
                status_code=404, detail=f"Model {body.model!r} is not served"
            )
        else:
            pass
        audio_bytes = await download_music_audio(
            str(body.music_url), max_bytes=MAX_AUDIO_BYTES
        )
        request_id = f"music-separate-{uuid.uuid4()}"
        generation = GenerateRequest(
            model=body.model or app.state.model_name,
            prompt={"audio_bytes": audio_bytes},
            stream=False,
            metadata={"task": "music_separate", "format": body.output_format},
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
