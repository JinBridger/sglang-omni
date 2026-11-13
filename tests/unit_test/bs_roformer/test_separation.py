# SPDX-License-Identifier: Apache-2.0
"""Source reconstruction, serial scheduling, and compiled inference."""

import copy
import io
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from fastapi.testclient import TestClient

pytest.importorskip("audio_separator")

from audio_separator.separator.uvr_lib_v5.roformer.attend import Attend
from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import (
    Attention,
    BSRoformer,
)

from sglang_omni.client.client import Client
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.bs_roformer.config import CHECKPOINT_FILENAME
from sglang_omni.models.bs_roformer.runtime import BSRoFormerRuntime, Separation
from sglang_omni.models.bs_roformer.stages import BSRoFormerScheduler
from sglang_omni.pipeline.mp_runner import resolve_coordinator_max_in_flight
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.serve.openai_api import create_app


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[BSRoFormerRuntime]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    configuration = {
        "audio": {"sample_rate": 8000, "chunk_size": 256},
        "model": {
            "dim": 8,
            "depth": 1,
            "stereo": True,
            "num_stems": 1,
            "time_transformer_depth": 1,
            "freq_transformer_depth": 1,
            "freqs_per_bands": (4, 13),
            "dim_head": 4,
            "heads": 2,
            "stft_n_fft": 32,
            "stft_hop_length": 8,
            "stft_win_length": 32,
            "mask_estimator_depth": 1,
        },
        "training": {"target_instrument": "Vocals"},
        "inference": {"num_overlap": 2},
    }
    checkpoint_path = tmp_path / CHECKPOINT_FILENAME
    checkpoint_path.with_suffix(".yaml").write_text(yaml.dump(configuration))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        torch.save(BSRoformer(**configuration["model"]).state_dict(), checkpoint_path)
    configuration_manager = ConfigManager.from_model_path(str(tmp_path))
    assert configuration_manager.config.architecture == "BSRoFormer"
    try:
        yield BSRoFormerRuntime(
            str(tmp_path),
            CHECKPOINT_FILENAME,
            torch.device("cpu"),
            torch.float32,
            False,
        )
    finally:
        torch.set_num_threads(previous_threads)


def test_sdpa_preserves_reference_model_output(runtime: BSRoFormerRuntime) -> None:
    reference = copy.deepcopy(runtime.model)
    for module in reference.modules():
        if isinstance(module, Attention):
            module.attend = Attend(flash=False)
        else:
            pass
    waveform = torch.from_numpy(
        np.random.default_rng(7).normal(size=(2, 2, 256)).astype(np.float32)
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            runtime.model(waveform), reference(waveform), atol=1e-6, rtol=1e-4
        )


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_cuda_inference_matches_eager_across_inputs(
    runtime: BSRoFormerRuntime, tmp_path: Path, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    eager = BSRoFormerRuntime(
        str(tmp_path), CHECKPOINT_FILENAME, torch.device("cuda:0"), dtype, False
    )
    compiled = BSRoFormerRuntime(
        str(tmp_path), CHECKPOINT_FILENAME, torch.device("cuda:0"), dtype, True
    )
    compiled.warm_up()
    generator = np.random.default_rng(9)
    for _ in range(3):
        waveform = torch.from_numpy(
            generator.normal(0, 0.1, (1, 2, 256)).astype(np.float32)
        )
        expected = eager.infer(waveform)
        actual = compiled.infer(waveform)
        relative_error = (actual - expected).norm() / expected.norm()
        assert torch.isfinite(actual).all()
        assert relative_error < (0.02 if dtype == torch.bfloat16 else 2e-5)


def test_queue_setting_reaches_coordinator_admission(
    runtime: BSRoFormerRuntime, tmp_path: Path
) -> None:
    manager = ConfigManager.from_model_path(str(tmp_path))
    configuration = manager.merge_config({"separate.max_queued_requests": 2})
    plan, _ = compile_logical_processes(configuration)
    assert (
        resolve_coordinator_max_in_flight(configuration, logical_process_plan=plan) == 3
    )


def audio_upload(samples: int, channels: int = 2, sample_rate: int = 8000) -> bytes:
    waveform = (
        np.random.default_rng(4).normal(0, 0.1, (samples, channels)).astype(np.float32)
    )
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, waveform, sample_rate, format="WAV", subtype="FLOAT")
    return audio_bytes.getvalue()


def payload(request_id: str, audio_bytes: bytes) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={"audio_bytes": audio_bytes}, metadata={"task": "music_separate"}
        ),
        data=None,
    )


def prepare(runtime: BSRoFormerRuntime, request: StagePayload) -> Separation:
    return Separation.prepare(
        request, runtime.sample_rate, runtime.chunk_samples, runtime.stride_samples, 1
    )


@pytest.mark.parametrize(
    ("samples", "channels", "sample_rate"),
    [(1, 1, 8000), (257, 2, 8000), (667, 1, 16000)],
)
def test_separation_preserves_length_and_reconstructs_source(
    runtime: BSRoFormerRuntime, samples: int, channels: int, sample_rate: int
) -> None:
    request = payload("test", audio_upload(samples, channels, sample_rate))
    source = prepare(runtime, request)
    result = BSRoFormerScheduler(runtime, 1).separate(request)
    with ZipFile(io.BytesIO(result.data["media"][0]["data"])) as archive:
        assert set(archive.namelist()) == {"vocals.wav", "instrumental.wav"}
        vocals, output_rate = sf.read(io.BytesIO(archive.read("vocals.wav")))
        instrumental, _ = sf.read(io.BytesIO(archive.read("instrumental.wav")))
    assert output_rate == runtime.sample_rate
    assert vocals.shape == (int(np.ceil(samples * output_rate / sample_rate)), 2)
    assert np.isfinite(vocals).all()
    np.testing.assert_allclose(
        vocals + instrumental, source.waveform.numpy().T, atol=3e-8
    )


@pytest.mark.parametrize(
    "audio_bytes", [b"not audio", audio_upload(8001), audio_upload(10, 3)]
)
def test_bad_audio_is_rejected(runtime: BSRoFormerRuntime, audio_bytes: bytes) -> None:
    with pytest.raises(ValueError, match="Music separation request:"):
        prepare(runtime, payload("bad", audio_bytes))


@pytest.mark.parametrize("cancel_active", [False, True])
def test_scheduler_finishes_requests_in_order_and_skips_cancelled_work(
    runtime: BSRoFormerRuntime, monkeypatch: pytest.MonkeyPatch, cancel_active: bool
) -> None:
    started = threading.Event()
    release = threading.Event()
    window_calls: list[int] = []
    original_infer = runtime.infer

    def infer(waveform: torch.Tensor) -> torch.Tensor:
        window_calls.append(waveform.shape[-1])
        if len(window_calls) == 1:
            started.set()
            assert release.wait(timeout=10)
        else:
            pass
        return original_infer(waveform)

    monkeypatch.setattr(runtime, "infer", infer)
    scheduler = BSRoFormerScheduler(runtime, 1)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        scheduler.inbox.put(
            IncomingMessage(
                "first", "new_request", payload("first", audio_upload(8000))
            )
        )
        assert started.wait(timeout=10)
        scheduler.inbox.put(
            IncomingMessage(
                "second", "new_request", payload("second", audio_upload(301))
            )
        )
        scheduler.inbox.put(
            IncomingMessage(
                "cancelled", "new_request", payload("cancelled", audio_upload(301))
            )
        )
        scheduler.abort("cancelled")
        if cancel_active:
            scheduler.abort("first")
        else:
            pass
        release.set()
        expected_ids = ["second"] if cancel_active else ["first", "second"]
        results = [scheduler.outbox.get(timeout=20) for _ in expected_ids]
        assert [(result.request_id, result.type) for result in results] == [
            (request_id, "result") for request_id in expected_ids
        ]
        if cancel_active:
            assert len(window_calls) == 3
        else:
            pass
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()


@pytest.mark.parametrize(
    ("output_format", "audio_format", "audio_subtype"),
    [(None, "WAV", "FLOAT"), ("flac", "FLAC", "PCM_16"), ("mp3", "WAV", "FLOAT")],
)
def test_endpoint_returns_zip_through_pipeline_client(
    runtime: BSRoFormerRuntime,
    output_format: str | None,
    audio_format: Literal["WAV", "FLAC"],
    audio_subtype: Literal["FLOAT", "PCM_16"],
) -> None:
    requests: list[OmniRequest] = []

    class Coordinator:
        async def submit(
            self, request_id: str, request: OmniRequest
        ) -> dict[str, list[dict[str, str | bytes]]]:
            requests.append(request)
            return (
                BSRoFormerScheduler(runtime, 1)
                .separate(StagePayload(request_id, request, None))
                .data
            )

        async def abort(self, request_id: str) -> None:
            pass

    application = create_app(
        Client(Coordinator()), model_name="bs-roformer", architectures=["BSRoFormer"]
    )
    audio_bytes = audio_upload(313)
    reference = BSRoFormerScheduler(runtime, 1).separate(
        payload("reference", audio_bytes)
    )
    with TestClient(application) as client:
        response = client.post(
            "/v1/music/separate",
            files={"file": ("music.wav", audio_bytes)},
            data={} if output_format is None else {"format": output_format},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/zip"
        assert 'filename="separated.zip"' in response.headers["content-disposition"]
        with (
            ZipFile(io.BytesIO(response.content)) as archive,
            ZipFile(
                io.BytesIO(reference.data["media"][0]["data"])
            ) as reference_archive,
        ):
            assert set(archive.namelist()) == {
                f"vocals.{audio_format.lower()}",
                f"instrumental.{audio_format.lower()}",
            }
            for stem_name in ("vocals", "instrumental"):
                with sf.SoundFile(
                    io.BytesIO(archive.read(f"{stem_name}.{audio_format.lower()}"))
                ) as audio_file:
                    assert audio_file.format == audio_format
                    assert audio_file.subtype == audio_subtype
                    assert audio_file.samplerate == runtime.sample_rate
                    assert audio_file.channels == 2
                    assert audio_file.frames == 313
                    waveform = audio_file.read(dtype="float32")
                reference_waveform, _ = sf.read(
                    io.BytesIO(reference_archive.read(f"{stem_name}.wav")),
                    dtype="float32",
                )
                np.testing.assert_allclose(
                    waveform,
                    reference_waveform,
                    rtol=0,
                    atol=1 / 32768 if output_format == "flac" else 0,
                )
        for audio_bytes, form, status_code in [
            (b"", {}, 400),
            (b"broken", {}, 400),
            (audio_upload(10), {"model": "other"}, 404),
        ]:
            response = client.post(
                "/v1/music/separate",
                files={"file": ("music.wav", audio_bytes)},
                data=form,
            )
            assert response.status_code == status_code, response.text
        assert len(requests) == 2


def test_endpoint_rejects_other_model_architectures() -> None:
    application = create_app(None, model_name="other", architectures=["OtherModel"])
    with TestClient(application) as client:
        response = client.post(
            "/v1/music/separate", files={"file": ("music.wav", audio_upload(10))}
        )
        assert response.status_code == 400
