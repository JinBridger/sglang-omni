# SPDX-License-Identifier: Apache-2.0
"""Music edit contracts, source preservation, and serial execution."""

import copy
import io
import threading
from typing import TypedDict

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.models.mula_edit.audio import PreparedEdit, prepare_audio
from sglang_omni.models.mula_edit.protocol import MulaEditInput, MulaEditRequest
from sglang_omni.models.mula_edit.sampling import MulaSampler, SamplingState
from sglang_omni.models.mula_edit.stages import MulaEditScheduler
from sglang_omni.models.mula_edit.symbols import parse_chord, score_events
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.serve import music_edit
from sglang_omni.serve.openai_api import create_app
from tests.unit_test.fixtures.mula_edit import sampling_state, small_decoder

pytest_plugins = ["tests.unit_test.fixtures.media_download"]


def audio_recording() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.zeros((48001, 2), dtype=np.float32), 48000, format="WAV")
    return stream.getvalue()


def edit_request(**changes: object) -> MulaEditRequest:
    return MulaEditRequest.model_validate(
        {
            "music_url": "https://media.example/music.wav?signature=a%2Fb%2Bc",
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


@pytest.mark.parametrize(
    ("label", "expected_chord"),
    [
        ("C:maj9", (0, 16, 0)),
        ("E:9", (4, 16, 4)),
        ("F#:maj9/b3", (6, 16, 9)),
        ("C:maj/8", None),
        ("C:maj9/8", None),
        ("H:maj", None),
        ("C/E", None),
        (" N ", None),
        ("X", None),
        (" C# ", (1, 1, 1)),
        ("Bb:min/b3", (10, 2, 1)),
        ("C:maj/13", (0, 1, 9)),
    ],
)
def test_score_accepts_unknown_chord_qualities_and_skips_unparseable_labels(
    label: str, expected_chord: tuple[int, int, int] | None
) -> None:
    """Unknown qualities and unparseable chord labels must not reject an edit."""
    assert parse_chord(label) == expected_chord
    request = edit_request(
        symbols={
            "vocal_melody": None,
            "instrumental_melody": None,
            "chord_progression": [[0.0, 0.04, label]],
        }
    )
    events, provided = score_events(request, 1)
    if expected_chord is None:
        assert events.shape == (0, 5)
    else:
        root, quality, bass = expected_chord
        assert events.tolist() == [[0, 2, root, 1, quality], [0, 2, bass, 1, 17]]
    assert provided.tolist() == [[[False, False, True]]]


def test_audio_keeps_partial_last_frame_and_rejects_truncation() -> None:
    audio_bytes = audio_recording()
    prepared = prepare_audio(edit_request(), audio_bytes, 2)
    assert prepared.samples == 48001
    assert prepared.waveform.shape == (2, 26 * 1920)
    assert prepared.provided.shape == (1, 26, 3)
    with pytest.raises(ValueError, match="exceeds"):
        prepare_audio(edit_request(), audio_bytes, 1)
    with pytest.raises(ValueError, match="beyond"):
        prepare_audio(edit_request(end_seconds=2), audio_bytes, 3)
    with pytest.raises(ValueError, match="Could not decode"):
        prepare_audio(edit_request(), b"not an audio file", 2)


@pytest.mark.parametrize("guidance", [1, 7])
@torch.inference_mode()
def test_serial_sampling_restores_source_outside_edit(guidance: float) -> None:
    decoder = small_decoder()
    sampler = MulaSampler(decoder, enable_torch_compile=False, enable_cuda_graph=False)
    state = sampling_state(decoder, frames=9, steps=4, guidance=guidance)
    sampler.prepare(state)
    while not state.done:
        sampler.advance(state)
    torch.testing.assert_close(
        state.latents[~state.mask], state.clean_latents[~state.mask], rtol=0, atol=0
    )
    assert torch.isfinite(state.latents).all()
    assert not torch.equal(state.latents[state.mask], state.clean_latents[state.mask])


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("enable_torch_compile", "enable_cuda_graph"),
    [(True, False), (False, True), (True, True)],
)
@torch.inference_mode()
def test_accelerated_sampling_matches_eager_across_requests(
    enable_torch_compile: bool, enable_cuda_graph: bool, dtype: torch.dtype
) -> None:
    torch.compiler.reset()
    decoder = small_decoder().to(device="cuda", dtype=dtype)
    eager = MulaSampler(decoder, enable_torch_compile=False, enable_cuda_graph=False)
    accelerated = MulaSampler(
        copy.deepcopy(decoder),
        enable_torch_compile=enable_torch_compile,
        enable_cuda_graph=enable_cuda_graph,
    )
    for frames, guidance in ((9, 7), (9, 7), (16, 1), (16, 7), (9, 7)):
        state = sampling_state(decoder, frames, steps=3, guidance=guidance)
        expected = copy.deepcopy(state)
        eager.prepare(expected)
        accelerated.prepare(state)
        while not state.done:
            eager.advance(expected)
            accelerated.advance(state)
            tolerance = 0 if dtype == torch.bfloat16 else 1e-4
            torch.testing.assert_close(
                state.latents, expected.latents, rtol=tolerance, atol=tolerance
            )
        torch.testing.assert_close(
            state.latents[~state.mask], state.clean_latents[~state.mask], rtol=0, atol=0
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
        edit_input = MulaEditInput.model_validate(request.inputs)
        prepare_audio(edit_input.request, edit_input.audio_bytes, 2)
        return {
            "audio_waveform": np.zeros((2, 4800), dtype=np.float32).tobytes(),
            "audio_waveform_shape": [2, 4800],
            "sample_rate": 48000,
        }

    async def abort(self, request_id: str) -> None:
        pass


def test_endpoint_delivers_stereo_wav_through_real_client(
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
) -> None:
    audio_bytes = audio_recording()
    audio_download[1][:] = [httpx.Response(200, content=audio_bytes)]
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
        assert request.inputs["request"]["symbols"]["vocal_melody"] == [[0.2, 0.4, 60]]
        assert request.inputs["request"]["start_seconds"] == 0.2
        assert request.inputs["request"]["end_seconds"] == 0.8
        assert request.inputs["audio_bytes"] == audio_bytes
        assert [str(request.url) for request in audio_download[0]] == [
            str(edit_request().music_url)
        ]
        request_schema = client.get("/openapi.json").json()["components"]["schemas"][
            "MulaEditRequest"
        ]
        assert "music_url" in request_schema["required"]
        assert "audio_base64" not in request_schema["properties"]
        response = client.post(
            "/v1/music/edit", json=edit_request(model="other").model_dump(mode="json")
        )
        assert response.status_code == 404
        assert len(coordinator.requests) == 1
        assert len(audio_download[0]) == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"music_url": None},
        {"music_url": ""},
        {"music_url": "file:///tmp/music.wav"},
        {"audio_base64": "YXVkaW8="},
    ],
)
def test_endpoint_rejects_invalid_url_or_legacy_audio_before_download(
    fields: dict[str, str | None],
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
) -> None:
    coordinator = Coordinator()
    application = create_app(
        Client(coordinator), architectures=["AceStepV4ConditionGenerationModel"]
    )
    body = {**edit_request().model_dump(mode="json"), **fields}
    with TestClient(application) as client:
        response = client.post("/v1/music/edit", json=body)
    assert response.status_code == 422
    assert audio_download[0] == []
    assert coordinator.requests == []


@pytest.mark.parametrize(
    ("download_result", "status_code"),
    [
        (httpx.Response(200, content=b""), 400),
        (httpx.Response(404), 400),
        (httpx.ReadTimeout("Timed out"), 400),
        (httpx.Response(200, headers={"content-length": "33"}), 413),
        (httpx.Response(200, stream=httpx.ByteStream(b"x" * 33)), 413),
    ],
)
def test_endpoint_rejects_failed_or_oversized_downloads(
    download_result: httpx.Response | httpx.HTTPError,
    status_code: int,
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(music_edit, "MAX_AUDIO_BYTES", 32)
    audio_download[1][:] = [download_result]
    coordinator = Coordinator()
    application = create_app(
        Client(coordinator), architectures=["AceStepV4ConditionGenerationModel"]
    )
    with TestClient(application) as client:
        response = client.post(
            "/v1/music/edit", json=edit_request().model_dump(mode="json")
        )
    assert response.status_code == status_code, response.text
    assert coordinator.requests == []


@pytest.mark.parametrize("audio_bytes", [b"broken audio", None])
def test_endpoint_reports_invalid_audio_or_non_json_requests(
    audio_bytes: bytes | None,
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
) -> None:
    coordinator = Coordinator()
    application = create_app(
        Client(coordinator), architectures=["AceStepV4ConditionGenerationModel"]
    )
    with TestClient(application) as client:
        if audio_bytes is None:
            response = client.post(
                "/v1/music/edit", files={"file": ("music.wav", b"\xffaudio")}
            )
            assert response.status_code == 415
            assert audio_download[0] == []
            assert coordinator.requests == []
        else:
            audio_download[1][:] = [httpx.Response(200, content=audio_bytes)]
            response = client.post(
                "/v1/music/edit", json=edit_request().model_dump(mode="json")
            )
            assert response.status_code == 400


@pytest.mark.parametrize("cancel_active", [False, True])
def test_scheduler_serializes_requests_and_releases_cancelled_work(
    monkeypatch: pytest.MonkeyPatch, cancel_active: bool
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    operations: list[tuple[str, int]] = []
    states: dict[int, int] = {}

    class Runtime:
        def begin(self, prepared: PreparedEdit) -> SamplingState:
            operations.append(("begin", prepared.request.seed))
            state = sampling_state(small_decoder(), 4, prepared.request.steps, 1)
            states[id(state)] = prepared.request.seed
            return state

        def advance(self, state: SamplingState) -> None:
            seed = states[id(state)]
            operations.append(("advance", seed))
            if seed == 1 and state.step_index == 0:
                first_started.set()
                assert release_first.wait(10)
            state.step_index += 1

        def finish(self, state: SamplingState, samples: int) -> torch.Tensor:
            operations.append(("finish", states[id(state)]))
            return torch.zeros(2, samples)

    def submit(scheduler: MulaEditScheduler, request_id: str, seed: int) -> None:
        request = OmniRequest(
            inputs={
                "request": edit_request(steps=3, seed=seed).model_dump(mode="json"),
                "audio_bytes": audio_recording(),
            },
            metadata={"task": "mula_edit"},
        )
        payload = StagePayload(request_id=request_id, request=request, data=None)
        scheduler.inbox.put(IncomingMessage(request_id, "new_request", data=payload))

    scheduler = MulaEditScheduler(Runtime(), 1, 2)
    monkeypatch.setattr(scheduler, "warm_up_serving_thread", lambda: None)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        submit(scheduler, "first", 1)
        assert first_started.wait(10)
        scheduler.abort("cancelled")
        submit(scheduler, "cancelled", 3)
        submit(scheduler, "second", 2)
        submit(scheduler, "overflow", 4)
        if cancel_active:
            scheduler.abort("first")
        release_first.set()
        results = [
            scheduler.outbox.get(timeout=10) for _ in range(2 if cancel_active else 3)
        ]
        successful = [
            result.request_id for result in results if result.type == "result"
        ]
        assert successful == (
            ["second", "overflow"] if cancel_active else ["first", "second"]
        )
        if cancel_active:
            assert all(result.type == "result" for result in results)
        else:
            [rejected] = [result for result in results if result.type == "error"]
            assert rejected.request_id == "overflow"
            assert isinstance(rejected.data, QueueFullError)
        first_operations = (
            [("begin", 1), ("advance", 1)]
            if cancel_active
            else [("begin", 1), *[("advance", 1)] * 3, ("finish", 1)]
        )
        expected_operations = first_operations + [
            ("begin", 2),
            *[("advance", 2)] * 3,
            ("finish", 2),
        ]
        if cancel_active:
            expected_operations += [("begin", 4), *[("advance", 4)] * 3, ("finish", 4)]
        assert operations == expected_operations
        assert scheduler.outbox.empty()
    finally:
        release_first.set()
        scheduler.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()
