# SPDX-License-Identifier: Apache-2.0
"""JSON music editing requests with a binary WAV response."""

import base64
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.mula_edit.protocol import MulaEditRequest
from sglang_omni.serve.speech_to_text import complete_speech_to_text_request


def register_music_edit(app: FastAPI) -> None:
    @app.post(
        "/v1/music/edit",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                }
            }
        },
    )
    async def edit_music(body: MulaEditRequest, request: Request) -> Response:
        if "AceStepV4ConditionGenerationModel" not in app.state.architectures:
            raise HTTPException(
                status_code=400, detail="This model does not support music editing"
            )
        elif body.model is not None and body.model != app.state.model_name:
            raise HTTPException(
                status_code=404, detail=f"Model {body.model!r} is not served"
            )
        else:
            pass
        request_id = f"mula-edit-{uuid.uuid4()}"
        generation = GenerateRequest(
            model=body.model or app.state.model_name,
            prompt=body.model_dump(mode="json"),
            stream=False,
            metadata={"task": "mula_edit"},
        )
        result = await complete_speech_to_text_request(
            request,
            app.state.client,
            generation,
            request_id=request_id,
            error_log_message="Music edit failed for %s",
        )
        if result.audio is None:
            raise HTTPException(
                status_code=500, detail="Model returned no edited audio"
            )
        else:
            return Response(
                content=base64.b64decode(result.audio.data),
                media_type="audio/wav",
                headers={"X-Request-Id": request_id},
            )
