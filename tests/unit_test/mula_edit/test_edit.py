# SPDX-License-Identifier: Apache-2.0
"""Music edit contracts, source preservation, and mixed-length batching."""

import base64
import copy
import io
import threading
from typing import TypedDict

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.client.client import Client
from sglang_omni.models.mula_edit.audio import PreparedEdit, prepare_audio
from sglang_omni.models.mula_edit.model import MulaDiT
from sglang_omni.models.mula_edit.model_config import MulaModelConfig
from sglang_omni.models.mula_edit.protocol import MulaEditRequest
from sglang_omni.models.mula_edit.sampling import SamplingState, advance_batch
from sglang_omni.models.mula_edit.stages import MulaEditScheduler
from sglang_omni.models.mula_edit.symbols import parse_chord, score_events
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.serve.openai_api import create_app


def edit_request(**changes: object) -> MulaEditRequest:
    stream = io.BytesIO()
    sf.write(stream, np.zeros((48001, 2), dtype=np.float32), 48000, format="WAV")
    return MulaEditRequest.model_validate(
        {
            "audio_base64": base64.b64encode(stream.getvalue()).decode(),
            "symbols": {
                "vocal_melody": [[0.2, 0.4, 60]],
                "instrumental_melody": [],
                "chord_progression": [[0.0, 1.0, "C:maj/3"]],
            },
            "lyrics": "[Instrumental]",
            "start_seconds": 0.2,
            "end_seconds": 0.8,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"end_seconds": 0.1},
        {"start_seconds": float("nan")},
        {"steps": 0},
        {"stream": True},
        {
            "symbols": {
                "vocal_melody": [[0.5, 0.1, 60]],
                "instrumental_melody": [],
                "chord_progression": [],
            }
        },
    ],
)
def test_invalid_edit_contract_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        edit_request(**changes)


def test_score_preserves_onsets_chord_bass_and_unknown_slots() -> None:
    request = edit_request(
        symbols={
            "vocal_melody": [[0.2, 0.4, 60]],
            "instrumental_melody": None,
            "chord_progression": [[0.0, 1.0, "C:maj/3"]],
        }
    )
    events, provided = score_events(request, 25)
    assert events[(events[:, 0] == 5) & (events[:, 1] == 0)].tolist() == [
        [5, 0, 60, 1, 0]
    ]
    assert events[(events[:, 0] == 6) & (events[:, 1] == 0)].tolist() == [
        [6, 0, 60, 0, 0]
    ]
    assert events[(events[:, 0] == 0) & (events[:, 1] == 2)].tolist() == [
        [0, 2, 0, 1, 1],
        [0, 2, 4, 1, 17],
    ]
    assert provided[0, :, 0].all()
    assert not provided[0, :, 1].any()
    assert parse_chord("N") is None
    with pytest.raises(ValueError, match="Harte"):
        parse_chord("C/E")


def test_audio_keeps_partial_last_frame_and_rejects_truncation() -> None:
    prepared = prepare_audio(edit_request(), 2)
    assert prepared.samples == 48001
    assert prepared.waveform.shape == (2, 26 * 1920)
    assert prepared.provided.shape == (1, 26, 3)
    with pytest.raises(ValueError, match="exceeds"):
        prepare_audio(edit_request(), 1)
    with pytest.raises(ValueError, match="beyond"):
        prepare_audio(edit_request(end_seconds=2), 3)
    with pytest.raises(ValueError, match="base64"):
        prepare_audio(edit_request(audio_base64="not an audio file"), 2)


def small_decoder() -> MulaDiT:
    torch.manual_seed(21)
    config = MulaModelConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=2,
        text_hidden_dim=16,
        num_lyric_encoder_hidden_layers=1,
        num_timbre_encoder_hidden_layers=1,
    )
    decoder = MulaDiT(config).eval()
    with torch.no_grad():
        for name, parameter in decoder.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.1)
    return decoder


def sampling_state(
    decoder: MulaDiT, frames: int, steps: int, guidance: float
) -> SamplingState:
    clean = torch.randn(1, frames, 64)
    noise = torch.randn_like(clean)
    mask = torch.zeros(1, frames, dtype=torch.bool)
    mask[:, 1:-1] = True
    conditions = torch.randn(2 if guidance > 1 else 1, 9, decoder.config.hidden_size)
    return SamplingState(
        clean_latents=clean,
        noise=noise,
        latents=noise,
        context_latents=torch.cat(
            (
                clean.masked_fill(mask.unsqueeze(-1), 0),
                mask.unsqueeze(-1).expand_as(clean),
            ),
            dim=-1,
        ),
        symbols=torch.randn(1, frames, 99),
        mask=mask,
        timesteps=torch.linspace(1, 0, steps + 1),
        cross_attention=decoder.prepare_cross_attention(conditions),
        guidance_scale=guidance,
        crossfade_frames=0,
    )


@torch.inference_mode()
def test_late_arrival_mixed_lengths_and_guidance_match_serial_sampling() -> None:
    decoder = small_decoder()
    first = sampling_state(decoder, frames=9, steps=4, guidance=7)
    second = sampling_state(decoder, frames=16, steps=2, guidance=1)
    serial = copy.deepcopy([first, second])
    for state in serial:
        while not state.done:
            advance_batch(decoder, [state])
    advance_batch(decoder, [first])
    active = [first, second]
    while active:
        advance_batch(decoder, active)
        active = [state for state in active if not state.done]
    for batched, independent in zip((first, second), serial):
        torch.testing.assert_close(
            batched.latents, independent.latents, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            batched.latents[~batched.mask],
            batched.clean_latents[~batched.mask],
            rtol=0,
            atol=0,
        )


class AudioResult(TypedDict):
    audio_waveform: bytes
    audio_waveform_shape: list[int]
    sample_rate: int


class Coordinator:
    def __init__(self) -> None:
        self.requests: list[tuple[str, OmniRequest]] = []

    async def submit(self, request_id: str, request: OmniRequest) -> AudioResult:
        self.requests.append((request_id, request))
        return {
            "audio_waveform": np.zeros((2, 4800), dtype=np.float32).tobytes(),
            "audio_waveform_shape": [2, 4800],
            "sample_rate": 48000,
        }

    async def abort(self, request_id: str) -> None:
        pass


def test_endpoint_delivers_stereo_wav_through_real_client() -> None:
    coordinator = Coordinator()
    application = create_app(
        Client(coordinator),
        model_name="mula-edit",
        architectures=["AceStepV4ConditionGenerationModel"],
    )
    with TestClient(application) as client:
        response = client.post(
            "/v1/music/edit", json=edit_request().model_dump(mode="json")
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        audio, sample_rate = sf.read(io.BytesIO(response.content))
        assert audio.shape == (4800, 2)
        assert sample_rate == 48000
        [(request_id, request)] = coordinator.requests
        assert response.headers["x-request-id"] == request_id
        assert request.metadata["task"] == "mula_edit"
        assert request.inputs["symbols"]["vocal_melody"] == [[0.2, 0.4, 60]]
        response = client.post(
            "/v1/music/edit", json=edit_request(model="other").model_dump(mode="json")
        )
        assert response.status_code == 404
        assert len(coordinator.requests) == 1


def test_scheduler_admits_during_sampling_and_cancels_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_prepared = threading.Event()
    batches: list[int] = []

    class Runtime:
        def begin(self, prepared: PreparedEdit) -> SamplingState:
            state = sampling_state(small_decoder(), 4, prepared.request.steps, 1)
            return state

        def advance(self, states: list[SamplingState]) -> None:
            if not batches:
                first_started.set()
                assert release_first.wait(10)
            elif len(batches) == 1:
                assert second_prepared.wait(10)
            batches.append(len(states))
            for state in states:
                state.step_index += 1

        def finish(self, state: SamplingState, samples: int) -> torch.Tensor:
            return torch.zeros(2, samples)

    def prepare(request: MulaEditRequest, max_audio_seconds: float) -> PreparedEdit:
        prepared = prepare_audio(request, max_audio_seconds)
        if request.seed == 2:
            second_prepared.set()
        return prepared

    def submit(
        scheduler: MulaEditScheduler, request_id: str, steps: int, seed: int
    ) -> None:
        request = OmniRequest(
            inputs=edit_request(steps=steps, seed=seed).model_dump(mode="json"),
            metadata={"task": "mula_edit"},
        )
        payload = StagePayload(request_id=request_id, request=request, data=None)
        scheduler.inbox.put(IncomingMessage(request_id, "new_request", data=payload))

    monkeypatch.setattr("sglang_omni.models.mula_edit.stages.prepare_audio", prepare)
    scheduler = MulaEditScheduler(Runtime(), 2, 2, 2)
    monkeypatch.setattr(scheduler, "warm_up_serving_thread", lambda: None)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        submit(scheduler, "first", 4, 1)
        assert first_started.wait(10)
        submit(scheduler, "second", 2, 2)
        scheduler.abort("cancelled")
        submit(scheduler, "cancelled", 2, 3)
        release_first.set()
        results = [scheduler.outbox.get(timeout=10), scheduler.outbox.get(timeout=10)]
        assert {result.request_id for result in results} == {"first", "second"}
        assert all(result.type == "result" for result in results)
        assert max(batches) == 2
        assert scheduler.outbox.empty()
    finally:
        release_first.set()
        second_prepared.set()
        scheduler.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()
