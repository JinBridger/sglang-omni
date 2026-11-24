# SPDX-License-Identifier: Apache-2.0
"""HeartMoE music requests adapted to the shared audio generation service."""

from dataclasses import replace
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from sglang_omni.serve.speech_errors import SpeechAPIError, bad_request
from sglang_omni.serve.speech_service import (
    PreparedSpeechRequest,
    SpeechRequestValidator,
    normalize_task_type,
)

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
            else normalize_task_type(self.task_type)
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
        payload["ref_audio"] = payload.pop("score")
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
            return speech_service.parse_generation_request(payload)
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
