# SPDX-License-Identifier: Apache-2.0
"""Request-local window stitching, grammar state, and JSON music export."""

import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict

import numpy as np
import pretty_midi
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sglang_omni.models.sheetsage2.model import BatchedDecoder, SheetSage2Model
from sglang_omni.preprocessing.transcription import (
    police_request_audio,
    resolve_audio_source,
)
from sglang_omni.proto.request import StagePayload

logger = logging.getLogger(__name__)
PREFIX_RESERVE_TOKENS = 128


class TranscriptionParameters(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task: Literal["transcribe"] = "transcribe"
    temperature: Literal[0, 0.0] = 0
    repetition_penalty: Literal[1, 1.0] | None = None
    prompt: Literal[""] | None = None
    language: Literal[""] | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)
    melody_only: bool = False


class Window(TypedDict):
    start: float
    end: float
    accept_start: float
    accept_end: float
    prefix_end: float
    generation_stop: float | None


class MelodyNote(TypedDict):
    pitch: int
    track: int
    duration_bin: int
    duration_steps: int
    end_time: float


class Rhythm(TypedDict, total=False):
    meter: tuple[int, int]
    eighth_position: int


class DecodedEvent(TypedDict):
    subbeat: int
    time: float
    global_subbeat: int
    tokens_by_field: dict[str, list[int]]
    values: dict[str, str | float | Rhythm | list[MelodyNote]]


class GrammarState(Protocol):
    in_shift: bool
    shift_run: int
    payload_count: int
    last_field_index: int
    incomplete: str | None

    def allowed(self, device: str) -> torch.Tensor: ...

    def update(self, token: int) -> bool: ...


@dataclass(kw_only=True)
class Transcription:
    payload: StagePayload
    audio: torch.Tensor
    duration_seconds: float
    windows: list[Window]
    parameters: TranscriptionParameters
    events: list[DecodedEvent] = field(default_factory=list)
    window_index: int = 0
    token_ids: list[int] = field(default_factory=list)
    generated_tokens: int = 0
    global_subbeat_base: int = 0
    grammar: GrammarState | None = None
    finish_reason: Literal["stop", "length"] = "stop"
    needs_prefill: bool = True
    done: bool = False

    @classmethod
    def prepare(
        cls, payload: StagePayload, model: SheetSage2Model, max_audio_seconds: float
    ) -> "Transcription":
        try:
            parameters = TranscriptionParameters.model_validate(payload.request.params)
        except ValidationError as error:
            raise ValueError(
                f"Native generation request has invalid SheetSage2 parameters: {error}"
            ) from error
        source = police_request_audio(resolve_audio_source(payload))
        if not isinstance(source, (bytes, str)):
            raise ValueError("SheetSage2 expects encoded audio bytes or an audio path")
        else:
            try:
                audio = model.pipeline.load_audio(source)
            except ValueError as error:
                raise ValueError(
                    f"SheetSage2 could not decode the uploaded audio: {error}"
                ) from error
        duration_seconds = len(audio) / model.config.sampling_rate
        if duration_seconds > max_audio_seconds:
            raise ValueError(
                f"SheetSage2 accepts audio up to {max_audio_seconds:g} seconds"
            )
        else:
            pass
        windows = model.pipeline.sliding_window_plan(
            duration_seconds, window_seconds=model.config.input_audio_length
        )
        return cls(
            payload=payload,
            audio=audio,
            duration_seconds=duration_seconds,
            windows=windows,
            parameters=parameters,
        )

    def prefill(
        self, model: SheetSage2Model, decoder: BatchedDecoder, slot: int
    ) -> int:
        window = self.windows[self.window_index]
        tokenizer = model.tokenizer
        prompts = model.generation.FULL_TASK_PROMPTS
        if self.window_index:
            _, prefix, base = model.generation.build_overlap_prefix_tokens(
                self.events, tokenizer, prompts, window["start"], window["prefix_end"]
            )
            self.global_subbeat_base = base or 0
        else:
            prefix = None
        self.token_ids = (
            list(prefix) if prefix is not None else tokenizer.prompt_prefix(prompts)
        )
        if self.token_ids[-1] == tokenizer.eos_token:
            self.token_ids.pop()
        else:
            pass
        if len(self.token_ids) >= model.max_output_seq_len - PREFIX_RESERVE_TOKENS:
            raise ValueError("SheetSage2 overlap prefix fills the decoder context")
        else:
            pass
        self.grammar = model.generation.PromptGrammarState(tokenizer)
        for token in self.token_ids[self.token_ids.index(tokenizer.out_token) + 1 :]:
            self.grammar.update(token)
        segment = model.pipeline.slice_audio(
            self.audio, window["start"], model.config.input_audio_length
        )[None].to(decoder.device)
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", dtype=decoder.dtype, enabled=decoder.dtype != torch.float32
            ),
        ):
            memory = model.encode(segment)
            logits = decoder.prefill(slot, memory, self.token_ids)
        self.needs_prefill = False
        allowed = self.grammar.allowed("cpu").to(decoder.device)
        return logits.masked_fill(~allowed, float("-inf")).argmax().item()

    def accept(self, token: int, model: SheetSage2Model) -> None:
        assert self.grammar is not None
        self.token_ids.append(token)
        self.generated_tokens += 1
        tokenizer = model.tokenizer
        window = self.windows[self.window_index]
        finished = self.grammar.update(token)
        stop_seconds = window["generation_stop"]
        if stop_seconds is None:
            stop_seconds = min(
                self.duration_seconds - window["start"], model.config.input_audio_length
            )
        else:
            pass
        if tokenizer.time_token_start <= token < tokenizer.time_token_end:
            finished |= (
                tokenizer.token_to_time_id(token) / tokenizer.time_hz >= stop_seconds
            )
        else:
            pass
        budget = self.parameters.max_new_tokens
        reached_limit = len(self.token_ids) >= model.max_output_seq_len or (
            budget is not None and self.generated_tokens >= budget
        )
        if not finished and not reached_limit:
            return
        else:
            pass
        if self.token_ids[-1] != tokenizer.eos_token:
            self.token_ids.append(tokenizer.eos_token)
        else:
            pass
        decoded, warning = model.generation.decode_generated_tokens(
            tokenizer,
            torch.tensor(self.token_ids),
            self.payload.request_id,
            self.window_index,
        )
        if warning is not None:
            logger.warning(f"SheetSage2 {self.payload.request_id}: {warning['error']}")
        else:
            pass
        lookup = model.generation.event_time_map(
            decoded, model.config.input_audio_length
        )
        self.events.extend(
            model.generation.stitched_window_events(
                decoded,
                lookup,
                window["start"],
                window["accept_start"],
                window["accept_end"],
                self.duration_seconds,
                self.window_index,
                global_subbeat_base=self.global_subbeat_base,
            )
        )
        self.window_index += 1
        self.needs_prefill = True
        if reached_limit and not finished:
            self.finish_reason = "length"
            self.done = True
        else:
            self.done = self.window_index == len(self.windows)
        if budget is not None and self.generated_tokens >= budget and not self.done:
            self.finish_reason = "length"
            self.done = True
        else:
            pass

    def export(self, model: SheetSage2Model) -> StagePayload:
        self.events.sort(key=lambda event: (event["time"], event["global_subbeat"]))
        exports = model.exports
        notes = sorted(
            (
                float(event["time"]),
                min(self.duration_seconds, float(note["end_time"])),
                int(note["pitch"]),
                int(note["track"]),
            )
            for event in self.events
            for note in event["values"].get("melody", ())
            if min(self.duration_seconds, float(note["end_time"])) > event["time"]
        )
        # note (Codex): LAB outputs preserve decoded timing before notation-only adjustments.
        file_contents: dict[str, str] = {
            f"melody_{track_name}.lab": exports.rows_text(
                [
                    (start_seconds, end_seconds, pitch)
                    for start_seconds, end_seconds, pitch, source_track in notes
                    if source_track == track
                ]
            )
            for track, track_name in ((0, "vocal"), (1, "instrumental"))
        }
        try:
            beats = exports.rhythm_rows(self.events)
            file_contents["beat.lab"] = exports.rows_text(beats)
            if len(beats) < 2:
                raise ValueError("At least two decoded beats are required for ABC")
            else:
                period_seconds = float(np.median(np.diff([b[0] for b in beats[-9:]])))
            if period_seconds <= 0:
                raise ValueError("Decoded beats must increase in time")
            else:
                pass
            while beats[-1][0] < self.duration_seconds - 1e-6:
                previous = beats[-1]
                beats.append(
                    [
                        previous[0] + period_seconds,
                        previous[1] % previous[2] + 1,
                        previous[2],
                        previous[3],
                    ]
                )
            intervals = {
                name: exports.interval_rows(self.events, name, self.duration_seconds)
                for name in ("chord", "key", "structure")
            }
            if not intervals["key"]:
                raise ValueError(
                    "No key was decoded; cannot construct a keyed ABC score"
                )
            else:
                pass
            intervals["chord"] = exports.correct_chord_rows(
                intervals["chord"], intervals["key"]
            )
            intervals["key"] = [
                [start, end, exports.normalize_key_name(key)]
                for start, end, key in intervals["key"]
            ]
            file_contents.update(
                {
                    f"{field_name}.lab": exports.rows_text(intervals[field_name])
                    for field_name in ("chord", "key")
                }
            )
            intervals = {
                name: [
                    [max(beats[0][0], start), min(beats[-1][0], end), value]
                    for start, end, value in rows
                    if end > beats[0][0] and start < beats[-1][0]
                ]
                for name, rows in intervals.items()
            }
            clean_notes, _ = exports.notation_notes(notes)
            melody = pretty_midi.PrettyMIDI(resolution=960)
            for track, name in ((0, "Vocal"), (1, "Ins")):
                instrument = pretty_midi.Instrument(program=0, name=name)
                instrument.notes = [
                    pretty_midi.Note(100, pitch, start, end)
                    for start, end, pitch, source in clean_notes
                    if source == track and end > start
                ]
                melody.instruments.append(instrument)
            # note (Codex): The MIDI roundtrip preserves the checkpoint's timing quantization.
            abc, _ = exports.generate_abc_from_data(
                exports.midi_bytes(melody),
                beats,
                intervals["chord"],
                intervals["key"],
                intervals["structure"],
                melody_only=self.parameters.melody_only,
            )
        except (ValueError, FileNotFoundError) as error:
            raise ValueError(f"SheetSage2 could not construct ABC: {error}") from error
        file_contents["score.abc"] = abc
        self.payload.data = {
            "text": json.dumps(file_contents, ensure_ascii=False),
            "finish_reason": self.finish_reason,
            "usage": {"completion_tokens": self.generated_tokens},
        }
        return self.payload
