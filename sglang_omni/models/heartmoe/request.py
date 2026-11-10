# SPDX-License-Identifier: Apache-2.0
"""Speech prompts, per-request sampling state, and codec frame transport."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput

ABC_START = 248070
ABC_END = 248071
AUDIO_START = 248072
AUDIO_END = 248073
AUDIO_PLACEHOLDER = 248074
CODEBOOK_SIZE = 8192
FRAMES_PER_SECOND = 12.5
HeartMoETask = Literal["Direct", "PlanFull", "PlanMelody", "Cover"]
TASK_INSTRUCTIONS: dict[HeartMoETask, str] = {
    "Direct": "Generate the complete song audio directly, without an ABC score.",
    "PlanFull": "Write a full ABC score, then generate the complete corresponding song audio.",
    "PlanMelody": "Write melody-only ABC notation, then generate the complete corresponding song audio.",
}
RESERVED_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|music_abc_start|>",
    "<|music_abc_end|>",
    "<|music_audio_start|>",
    "<|music_audio_end|>",
    "<|music_frame|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|style_start|>",
    "<|style_end|>",
)


class SamplingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    max_new_tokens: int = Field(default=375, ge=1, le=7500)
    seed: int = Field(default=20261105, ge=0, lt=2**64)
    temperature: float = Field(default=0.8, ge=0)
    top_k: int = Field(default=50, ge=0, le=CODEBOOK_SIZE + 1)
    top_p: float = Field(default=0.95, gt=0, le=1)


@dataclass
class HeartMoERequest(SGLangARRequestData):
    enforce_request_limits: bool = True
    options: SamplingOptions = field(default_factory=SamplingOptions)
    uniforms: torch.Tensor | None = None
    feedback: torch.Tensor | None = None
    frames: list[torch.Tensor] = field(default_factory=list)
    generated_frames: int = 0
    phase: Literal["abc", "audio_start", "audio"] = "audio"
    abc_token_ids: list[int] = field(default_factory=list)
    max_abc_tokens: int = 0
    abc_uniforms: torch.Tensor | None = None


def render_prompt(lyrics: str, style: str, task_type: HeartMoETask = "PlanFull") -> str:
    if not lyrics.strip():
        raise ValueError("HeartMoE input must contain lyrics or '[Instrumental]'")
    elif any(marker in text for text in (lyrics, style) for marker in RESERVED_MARKERS):
        raise ValueError("HeartMoE lyrics and instructions contain a reserved marker")
    else:
        pass
    task_type = "PlanFull" if task_type == "Cover" else task_type
    style_prompt = f"<|style_start|>{style}<|style_end|>\n" if style.strip() else ""
    return (
        "<|im_start|>system\nGenerate complete songs from the supplied conditions. "
        "Follow the requested output order, ABC type and modality boundaries.<|im_end|>\n"
        f"<|im_start|>user\n{TASK_INSTRUCTIONS[task_type]}\n"
        f"{style_prompt}<|lyrics_start|>{lyrics}<|lyrics_end|>\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def build_request(
    payload: StagePayload,
    tokenizer: PreTrainedTokenizerBase,
    context_length: int,
    max_abc_tokens: int,
) -> HeartMoERequest:
    speech = payload.request.metadata.get("tts_params")
    if not isinstance(speech, dict) or not isinstance(payload.request.inputs, str):
        raise ValueError("HeartMoE requires a /v1/audio/speech request")
    elif speech.get("voice") not in (None, "", "default"):
        raise ValueError("HeartMoE supports voice='default' only")
    elif speech.get("speed", 1.0) != 1.0:
        raise ValueError("HeartMoE supports speed=1.0 only")
    else:
        pass
    sampling = {
        name: value
        for name, value in payload.request.params.items()
        if name in SamplingOptions.model_fields and value is not None
    }
    if speech.get("seed") is not None:
        sampling["seed"] = speech["seed"]
    else:
        pass
    options = SamplingOptions.model_validate(sampling)
    task_type = TypeAdapter(HeartMoETask).validate_python(
        "PlanFull" if speech.get("task_type") is None else speech["task_type"]
    )
    prompt = render_prompt(
        payload.request.inputs, speech.get("instructions") or "", task_type
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ref_audio = speech.get("ref_audio")
    if task_type == "Cover":
        if not isinstance(ref_audio, str) or not ref_audio.strip():
            raise ValueError("HeartMoE Cover requires non-empty ABC text in ref_audio")
        elif any(marker in ref_audio for marker in RESERVED_MARKERS):
            raise ValueError("HeartMoE Cover ref_audio contains a reserved marker")
        else:
            prompt_ids += (
                [ABC_START]
                + tokenizer.encode(ref_audio, add_special_tokens=False)
                + [ABC_END, AUDIO_START]
            )
            prompt += f"<|music_abc_start|>{ref_audio}<|music_abc_end|>"
    elif ref_audio is not None:
        raise ValueError("HeartMoE ref_audio is only supported for task_type='Cover'")
    else:
        prompt_ids.append(AUDIO_START if task_type == "Direct" else ABC_START)
    starts_with_audio = task_type in ("Direct", "Cover")
    generation_tokens = options.max_new_tokens
    if starts_with_audio:
        max_abc_tokens = 0
    else:
        max_abc_tokens = min(
            # note (Codex): SGLang reserves one context position beyond the generation budget.
            max_abc_tokens,
            context_length - len(prompt_ids) - generation_tokens - 3,
        )
        if max_abc_tokens < 1:
            raise ValueError("HeartMoE prompt and audio frames leave no room for ABC")
        else:
            generation_tokens += max_abc_tokens + 2
    if len(prompt_ids) + generation_tokens >= context_length:
        raise ValueError(
            f"HeartMoE prompt and audio frames exceed {context_length} positions"
        )
    else:
        pass
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=generation_tokens,
        stop_token_ids=[AUDIO_END],
    )
    sampling_params.normalize(None)
    request = Req(
        rid=payload.request_id,
        origin_input_text=prompt,
        origin_input_ids=prompt_ids,
        sampling_params=sampling_params,
        eos_token_ids={AUDIO_END},
        vocab_size=len(tokenizer),
    )
    request.tokenizer = None
    return HeartMoERequest(
        req=request,
        stage_payload=payload,
        options=options,
        input_ids=torch.tensor([prompt_ids]),
        output_ids=request.output_ids,
        max_new_tokens=generation_tokens,
        phase="audio" if starts_with_audio else "abc",
        max_abc_tokens=max_abc_tokens,
    )


class FrameStream:
    """Flush small CPU frame blocks without retaining CUDA graph output storage."""

    def __call__(
        self, request_id: str, request: HeartMoERequest, output: RequestOutput
    ) -> Iterator[OutgoingMessage]:
        if len(request.frames) >= 8:
            yield from self.flush(request_id, request)
        else:
            pass

    def flush(
        self, request_id: str, request: HeartMoERequest
    ) -> Iterator[OutgoingMessage]:
        if request.frames:
            frames = torch.stack(request.frames).cpu()
            request.frames.clear()
            yield OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=frames,
                metadata={"modality": "audio_codes", "stream": True},
            )
        else:
            pass


def build_result(request: HeartMoERequest) -> StagePayload:
    if request.phase != "audio":
        raise ValueError("HeartMoE generation stopped before reaching audio")
    elif request.generated_frames == 0:
        raise ValueError("HeartMoE generated no audio frames")
    else:
        pass
    payload = request.stage_payload
    prompt_tokens = len(request.req.origin_input_ids)
    completion_tokens = len(request.abc_token_ids) + request.generated_frames
    payload.data = {
        "frames": request.generated_frames,
        "abc_tokens": len(request.abc_token_ids),
        "finish_reason": "length" if request.finish_reason == "length" else "stop",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return payload
