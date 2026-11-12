# SPDX-License-Identifier: Apache-2.0
"""JSON contract for music edits on the original recording's timeline."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CHORD_QUALITIES = (
    "maj",
    "min",
    "dim",
    "aug",
    "sus2",
    "sus4",
    "7",
    "maj7",
    "min7",
    "hdim7",
    "dim7",
    "sus4(b7)",
    "maj6",
    "min6",
    "minmaj7",
    "other",
)
MAX_AUDIO_BYTES = 128 * 1024 * 1024
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NoteEvent = tuple[Seconds, Seconds, Annotated[int, Field(strict=True, ge=0, le=127)]]
ChordEvent = tuple[Seconds, Seconds, Annotated[str, Field(min_length=1, max_length=64)]]


class SymbolicScore(BaseModel):
    """Each slot is a complete event list, or null when its condition is unavailable."""

    model_config = ConfigDict(extra="forbid")
    vocal_melody: list[NoteEvent] | None
    instrumental_melody: list[NoteEvent] | None
    chord_progression: list[ChordEvent] | None

    @model_validator(mode="after")
    def validate_intervals(self) -> Self:
        for events in (
            self.vocal_melody,
            self.instrumental_melody,
            self.chord_progression,
        ):
            for start_seconds, end_seconds, _ in events or []:
                if end_seconds <= start_seconds:
                    raise ValueError("Every score event must end after it starts")
                else:
                    pass
        return self


class MulaEditRequest(BaseModel):
    """One edit; audio_base64 contains an encoded audio file, without a data URL prefix."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str | None = None
    audio_base64: str = Field(min_length=1, max_length=4 * ((MAX_AUDIO_BYTES + 2) // 3))
    symbols: SymbolicScore
    lyrics: str = Field(max_length=32768)
    start_seconds: Seconds
    end_seconds: Seconds
    symbolic_scope: Literal["whole", "span"] = "whole"
    caption: str = Field(default="", max_length=8192)
    bpm: int | None = Field(default=None, gt=0, le=1000)
    keyscale: str | None = Field(default=None, max_length=64)
    timesignature: str | None = Field(default=None, max_length=16)
    steps: int = Field(default=32, ge=1, le=200)
    guidance_scale: float = Field(default=7.0, ge=0, le=30)
    shift: float = Field(default=3.0, gt=0, le=20)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)
    crossfade_seconds: Seconds = 1.0

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        else:
            pass
        return self
