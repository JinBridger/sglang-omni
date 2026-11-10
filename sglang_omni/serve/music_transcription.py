# SPDX-License-Identifier: Apache-2.0
"""Music transcription from audio URLs with structured ABC and LAB output."""

import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError

from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.serve.music_audio import (
    complete_music_request,
    download_music_audio,
    require_music_json,
)


class MusicTranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    music_url: AnyHttpUrl
    model: str | None = None
    melody_only: bool = False
    max_new_tokens: int | None = Field(default=None, ge=1)


class MusicTranscriptionResponse(BaseModel):
    melody_vocal_lab: str = Field(alias="melody_vocal.lab")
    melody_instrumental_lab: str = Field(alias="melody_instrumental.lab")
    beat_lab: str = Field(alias="beat.lab")
    chord_lab: str = Field(alias="chord.lab")
    key_lab: str = Field(alias="key.lab")
    score_abc: str | None = Field(default=None, alias="score.abc")


def register_music_transcription(app: FastAPI) -> None:
    @app.post("/v1/music/transcribe", dependencies=[Depends(require_music_json)])
    async def transcribe_music(
        body: MusicTranscriptionRequest,
        request: Request,
        response: Response,
    ) -> MusicTranscriptionResponse:
        if body.model is not None and body.model != app.state.model_name:
            raise HTTPException(
                status_code=404, detail=f"Model {body.model!r} is not served"
            )
        else:
            pass
        audio_bytes = await download_music_audio(str(body.music_url), max_bytes=None)
        request_id = f"music-transcribe-{uuid.uuid4()}"
        generation = GenerateRequest(
            model=body.model or app.state.model_name,
            prompt={"audio_bytes": audio_bytes},
            sampling=SamplingParams(max_new_tokens=body.max_new_tokens),
            extra_params={"task": "transcribe", "melody_only": body.melody_only},
            stream=False,
            metadata={"task": "music_transcribe"},
        )
        result = await complete_music_request(
            request,
            app.state.client,
            generation,
            request_id=request_id,
        )
        try:
            transcription = MusicTranscriptionResponse.model_validate_json(result.text)
        except ValidationError as error:
            raise HTTPException(
                status_code=500, detail="Model returned invalid music transcription"
            ) from error
        else:
            response.headers["X-Request-Id"] = request_id
            if result.finish_reason:
                response.headers["X-Finish-Reason"] = result.finish_reason
            else:
                pass
            return transcription
