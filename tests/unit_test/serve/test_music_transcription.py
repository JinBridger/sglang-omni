# SPDX-License-Identifier: Apache-2.0
"""Music transcription HTTP contracts through the pipeline client."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_api import create_app

pytest_plugins = ["tests.unit_test.fixtures.media_download"]


class MusicCoordinator:
    def __init__(self, result_text: str | Exception) -> None:
        self.result_text = result_text
        self.requests: list[tuple[str, OmniRequest]] = []

    async def submit(self, request_id: str, request: OmniRequest) -> dict[str, str]:
        self.requests.append((request_id, request))
        if isinstance(self.result_text, Exception):
            raise self.result_text
        else:
            return {"text": self.result_text, "finish_reason": "length"}

    async def abort(self, request_id: str) -> None:
        pass


def music_client(
    coordinator: MusicCoordinator, architecture: str = "SheetSage2Model"
) -> TestClient:
    return TestClient(
        create_app(
            Client(coordinator), model_name="sheetsage2", architectures=[architecture]
        )
    )


@pytest.mark.parametrize(
    "architecture", ["SheetSage2Model", "SheetSage2ForConditionalGeneration"]
)
@pytest.mark.parametrize("include_score", [True, False])
def test_music_transcription_returns_files_as_json_fields(
    architecture: str,
    include_score: bool,
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
) -> None:
    """Music outputs must not require a second JSON decode of a text field."""
    file_contents = {
        "melody_vocal.lab": "0.0\t0.5\t60\n",
        "melody_instrumental.lab": "",
        "beat.lab": "0.0\t1\t4\t0\n",
        "chord.lab": "0.0\t1.0\tC\n",
        "key.lab": "0.0\t1.0\tC:maj\n",
    }
    if include_score:
        file_contents["score.abc"] = "X:1\nT:音乐\nK:C\nC D E F|\n"
    else:
        pass
    coordinator = MusicCoordinator(json.dumps(file_contents, ensure_ascii=False))
    audio_url = "https://media.example/song.wav?signature=a%2Fb%2Bc&expires=123"
    with music_client(coordinator, architecture) as client:
        response = client.post(
            "/v1/music/transcribe",
            json={
                "music_url": audio_url,
                "model": "sheetsage2",
                "melody_only": True,
                "max_new_tokens": 32,
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {
            **file_contents,
            "score.abc": file_contents.get("score.abc"),
        }
        assert response.headers["x-finish-reason"] == "length"
        [(request_id, request)] = coordinator.requests
        assert response.headers["x-request-id"] == request_id
        assert request.inputs == {"audio_bytes": b"encoded audio"}
        assert [str(request.url) for request in audio_download[0]] == [audio_url]
        assert request.params["task"] == "transcribe"
        assert request.params["melody_only"] is True
        assert request.params["max_new_tokens"] == 32
        assert request.params["stream"] is False
        assert request.metadata["model"] == "sheetsage2"
        assert request.metadata["task"] == "music_transcribe"
        assert client.post("/v1/audio/transcriptions").status_code == 404
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/music/transcribe" in paths
        assert "/v1/audio/transcriptions" not in paths
        assert set(paths["/v1/music/transcribe"]["post"]["requestBody"]["content"]) == {
            "application/json"
        }


@pytest.mark.parametrize(
    ("body", "status_code"),
    [
        ({}, 422),
        ({"music_url": ""}, 422),
        ({"music_url": "not-a-url"}, 422),
        ({"music_url": "file:///tmp/song.wav"}, 422),
        ({"music_url": "https://media.example/song.wav", "model": "unknown"}, 404),
        ({"music_url": "https://media.example/song.wav", "max_new_tokens": 0}, 422),
    ],
)
def test_invalid_requests_are_rejected_before_download(
    body: dict[str, str | int],
    status_code: int,
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
) -> None:
    coordinator = MusicCoordinator("{}")
    with music_client(coordinator) as client:
        response = client.post("/v1/music/transcribe", json=body)
    assert response.status_code == status_code
    assert coordinator.requests == []
    assert audio_download[0] == []


def test_multipart_uploads_are_rejected() -> None:
    """Binary uploads must not crash JSON encoding of body validation errors."""
    coordinator = MusicCoordinator("{}")
    with music_client(coordinator) as client:
        response = client.post(
            "/v1/music/transcribe", files={"file": ("music.wav", b"\xffaudio")}
        )
    assert response.status_code == 415
    assert coordinator.requests == []


@pytest.mark.parametrize("result_text", ["not json", "[]", '{"beat.lab": 123}'])
def test_invalid_backend_output_returns_server_error(
    result_text: str,
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
) -> None:
    with music_client(MusicCoordinator(result_text)) as client:
        response = client.post(
            "/v1/music/transcribe", json={"music_url": "https://media.example/song.wav"}
        )
    assert response.status_code == 500
    assert response.json()["detail"] == "Model returned invalid music transcription"


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (
            RuntimeError(
                "SheetSage2 could not decode the uploaded audio: invalid header"
            ),
            400,
        ),
        (RuntimeError(QueueFullError.MESSAGE), 503),
    ],
)
def test_backend_errors_preserve_http_status(
    error: Exception,
    status_code: int,
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
) -> None:
    with music_client(MusicCoordinator(error)) as client:
        response = client.post(
            "/v1/music/transcribe", json={"music_url": "https://media.example/song.wav"}
        )
    assert response.status_code == status_code


@pytest.mark.parametrize(
    "download_result",
    [
        httpx.Response(200, content=b""),
        httpx.Response(404),
        httpx.ReadTimeout("Timed out"),
    ],
)
def test_download_failures_are_rejected_before_dispatch(
    download_result: httpx.Response | httpx.HTTPError,
    audio_download: tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]],
) -> None:
    audio_download[1][:] = [download_result]
    coordinator = MusicCoordinator("{}")
    with music_client(coordinator) as client:
        response = client.post(
            "/v1/music/transcribe", json={"music_url": "https://media.example/song.wav"}
        )
    assert response.status_code == 400
    assert coordinator.requests == []


def test_speech_models_keep_their_transcription_route() -> None:
    with music_client(
        MusicCoordinator("hello"), "WhisperForConditionalGeneration"
    ) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/audio/transcriptions" in paths
        assert "/v1/music/transcribe" not in paths
        assert client.post("/v1/music/transcribe").status_code == 404
