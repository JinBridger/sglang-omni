# SPDX-License-Identifier: Apache-2.0
"""HeartMoE music request validation and audio responses."""

import asyncio
import logging
import uuid
from dataclasses import replace
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SpeechResult
from sglang_omni.serve.music_audio import music_generation_error, require_music_json
from sglang_omni.serve.music_streaming import music_audio_response
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    speech_error_response,
)
from sglang_omni.serve.speech_service import (
    PreparedSpeechRequest,
    SpeechRequestValidator,
)
from sglang_omni.serve.speech_to_text import (
    cancel_task_bounded,
    wait_for_request_disconnect,
)

logger = logging.getLogger(__name__)
MUSIC_TASK_TYPES = {
    task.replace("_", "").replace("-", "").lower(): task
    for task in ("Direct", "PlanFull", "PlanMelody", "Cover", "Instrumental")
}

# note (Codex): HeartCodec decodes each frame to 3,840 samples at 48 kHz.
HEARTMOE_FRAME_DURATION_SECONDS = Decimal("0.08")
MusicDurationSeconds = Annotated[
    float,
    Field(strict=True, ge=0.08, le=600, allow_inf_nan=False),
]


class MusicGenerationRequest(BaseModel):
    """Generate music from lyrics, style instructions, and an optional ABC score."""

    model_config = ConfigDict(extra="forbid", strict=True)

    lyrics: str
    score: str | None = None
    model: str | None = None
    instructions: str | None = None
    task_type: str | None = None
    voice: str = Field(
        default="default", validation_alias=AliasChoices("voice", "speaker")
    )
    speed: float = 1.0
    response_format: str = "wav"
    stream: bool = False
    stream_format: Literal["audio", "sse"] = "audio"
    max_duration: MusicDurationSeconds | None = Field(
        default=None,
        description="Maximum audio duration in seconds; defaults to 30 when omitted or null.",
    )
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stage_params: dict[str, dict[str, JsonValue]] | None = None

    def prepare_generation(
        self, speech_service: SpeechRequestValidator
    ) -> PreparedSpeechRequest:
        task_type = (
            "PlanFull"
            if self.task_type is None
            else MUSIC_TASK_TYPES.get(
                self.task_type.strip().replace("_", "").replace("-", "").lower()
            )
        )
        if task_type not in {
            "Direct",
            "PlanFull",
            "PlanMelody",
            "Cover",
            "Instrumental",
        }:
            raise bad_request(
                "task_type must be Direct, PlanFull, PlanMelody, Cover, or Instrumental",
                param="task_type",
            )
        elif self.score is not None and task_type != "Cover":
            raise bad_request(
                "score is only supported for task_type='Cover'", param="score"
            )
        else:
            pass
        payload = self.model_dump()
        payload["input"] = payload.pop("lyrics")
        payload.pop("score")
        payload.pop("task_type")
        if task_type == "Cover" and (self.score is None or not self.score.strip()):
            raise bad_request(
                "Cover requires non-empty ABC text in score", param="score"
            )
        else:
            pass
        parameter_groups = [(payload, "")]
        if self.stage_params is not None and "heartmoe_ar" in self.stage_params:
            parameter_groups.append(
                (payload["stage_params"]["heartmoe_ar"], "stage_params.heartmoe_ar.")
            )
        else:
            pass
        for parameters, prefix in parameter_groups:
            if "max_new_tokens" in parameters:
                raise bad_request(
                    f"{prefix}max_new_tokens has been replaced by {prefix}max_duration in seconds",
                    param=f"{prefix}max_new_tokens",
                )
            else:
                pass
            duration_seconds = parameters.pop("max_duration", None)
            if duration_seconds is None:
                continue
            else:
                pass
            try:
                duration_seconds = TypeAdapter(MusicDurationSeconds).validate_python(
                    duration_seconds
                )
            except ValidationError as error:
                raise bad_request(
                    f"{prefix}max_duration must be a finite number between 0.08 and 600 seconds",
                    param=f"{prefix}max_duration",
                ) from error
            parameters["max_new_tokens"] = int(
                Decimal(str(duration_seconds)) / HEARTMOE_FRAME_DURATION_SECONDS
            )
        try:
            prepared = speech_service.parse_generation_request(payload)
        except SpeechAPIError as error:
            field_names = {"input": "lyrics", "ref_audio": "score"}
            if error.param in field_names:
                field_name = field_names[error.param]
                raise replace(
                    error,
                    message=error.message.replace(error.param, field_name),
                    param=field_name,
                ) from error
            else:
                raise

        return replace(
            prepared,
            request=prepared.request.model_copy(
                update={
                    "task_type": task_type if self.task_type is not None else None,
                    "ref_audio": self.score,
                }
            ),
        )


def register_music_generation(app: FastAPI) -> None:
    @app.post("/v1/music/generate", dependencies=[Depends(require_music_json)])
    async def create_music(request: Request, body: MusicGenerationRequest) -> Response:
        try:
            prepared = await asyncio.to_thread(
                body.prepare_generation, app.state.speech_service
            )
        except SpeechAPIError as error:
            return speech_error_response(error)
        return await create_music_response(request, prepared)

    @app.get("/v1/music/generate/{request_id}")
    async def get_music_stream_outcome(request_id: str) -> JSONResponse:
        outcome = app.state.speech_stream_outcomes.get(request_id)
        if outcome is None:
            raise HTTPException(
                status_code=404,
                detail=f"No finished audio stream with request id {request_id}",
            )
        else:
            return JSONResponse(outcome.model_dump())


def music_generation_failure_response(
    request_id: str,
    error: Exception,
) -> JSONResponse:
    failure = music_generation_error(error)
    if failure.status_code == 500:
        logger.exception(f"Music generation failed for request {request_id}")
    else:
        pass
    return speech_error_response(failure)


async def create_music_response(
    request: Request, prepared: PreparedSpeechRequest
) -> Response:
    app = request.app
    client: Client = app.state.client
    speech_service: SpeechRequestValidator = app.state.speech_service
    request_id = f"speech-{uuid.uuid4()}"
    req = prepared.request
    gen_req = speech_service.build_generate_request(
        req,
        validate=False,
        reference_descriptors=prepared.reference_descriptors,
        uploaded_voice=prepared.uploaded_voice,
    )
    if req.stream:
        try:
            return await music_audio_response(
                request=request,
                client=client,
                gen_req=gen_req,
                request_id=request_id,
                speed=req.speed,
                speech_stream_outcomes=app.state.speech_stream_outcomes,
                stream_format=req.stream_format,
                response_format="wav" if req.response_format == "wav" else "pcm",
                enable_speech_sse_named_events=app.state.enable_speech_sse_named_events,
            )
        except Exception as exc:
            return music_generation_failure_response(request_id, exc)
    else:
        pass

    try:
        result = await await_music_response(
            request=request,
            client=client,
            gen_req=gen_req,
            request_id=request_id,
            response_format=req.response_format,
            speed=req.speed,
        )
    except Exception as exc:
        return music_generation_failure_response(request_id, exc)

    headers = {
        "Content-Disposition": f'attachment; filename="speech.{result.format}"',
        "X-Request-Id": request_id,
        # note (Junnan Li): the body is binary audio, so the terminal state
        # travels in the same X- header channel as usage.
        "X-Finish-Reason": result.finish_reason,
    }
    if result.usage is not None:
        headers.update(
            {
                name: str(tokens)
                for name, tokens in (
                    ("X-Prompt-Tokens", result.usage.prompt_tokens),
                    ("X-Completion-Tokens", result.usage.completion_tokens),
                    (
                        "X-SGLang-Omni-Input-Tokens",
                        result.usage.prompt_tokens,
                    ),
                    (
                        "X-SGLang-Omni-Output-Tokens",
                        result.usage.completion_tokens,
                    ),
                    ("X-SGLang-Omni-Total-Tokens", result.usage.total_tokens),
                )
                if tokens is not None
            }
        )
        if result.usage.engine_time_s is not None:
            headers["X-Engine-Time"] = str(result.usage.engine_time_s)
        else:
            pass
    else:
        pass
    return Response(
        content=result.audio_bytes,
        media_type=result.mime_type,
        headers=headers,
    )


async def await_music_response(
    request: Request,
    client: Client,
    gen_req: GenerateRequest,
    *,
    request_id: str,
    response_format: str,
    speed: float,
) -> SpeechResult:
    speech_task = asyncio.create_task(
        client.speech(
            gen_req,
            request_id=request_id,
            response_format=response_format,
            speed=speed,
            allow_format_fallback=False,
        )
    )
    disconnect_task = asyncio.create_task(wait_for_request_disconnect(request))
    aborted = False
    try:
        done, _ = await asyncio.wait(
            {speech_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if speech_task in done:
            return speech_task.result()
        else:
            pass

        await client.abort(request_id)
        aborted = True
        speech_task.cancel()
        raise asyncio.CancelledError
    except asyncio.CancelledError:
        if not aborted:
            await client.abort(request_id)
        else:
            pass
        raise
    finally:
        if not speech_task.done():
            await cancel_task_bounded(speech_task)
        else:
            pass
        if not disconnect_task.done():
            await cancel_task_bounded(disconnect_task)
        else:
            pass
