# SPDX-License-Identifier: Apache-2.0
"""HeartMoE request, batching, and frame transport invariants."""

import json
from pathlib import Path
from threading import Event

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.pack_heartmoe_codec import pack_codec
from sglang_omni.models.heartmoe.codec import SAMPLES_PER_FRAME, HeartCodecScheduler
from sglang_omni.models.heartmoe.codec_flow import FlowMatching
from sglang_omni.models.heartmoe.codec_model import HeartCodec, HeartCodecConfig
from sglang_omni.models.heartmoe.config import ARFactoryArgs
from sglang_omni.models.heartmoe.engine import HeartMoEEngineBuilder
from sglang_omni.models.heartmoe.local import AudioModules, FrameDecoder, LocalDecoder
from sglang_omni.models.heartmoe.request import (
    FrameStream,
    HeartMoERequest,
    render_prompt,
)
from sglang_omni.models.heartmoe.sampling import sample_codes
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.fixtures.sampling import production_sampler_context


def test_local_cached_decode_matches_full_prefix() -> None:
    """A local frame's seven cached steps must equal causal full-prefix attention."""
    torch.manual_seed(17)
    decoder = LocalDecoder(
        dict(
            hidden_size=32,
            intermediate_size=48,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            partial_rotary_factor=0.5,
            rope_theta=10000,
            rms_norm_eps=1e-6,
            num_hidden_layers=2,
        ),
        codebook_size=16,
    ).eval()
    prefix = torch.randn(3, 2, 32)
    cache = None
    with torch.no_grad():
        cached, cache = decoder.run_layers(prefix, cache)
        for _ in range(6):
            token = torch.randn(3, 1, 32)
            prefix = torch.cat((prefix, token), dim=1)
            cached, cache = decoder.run_layers(token, cache)
            full, _ = decoder.run_layers(prefix, None)
            torch.testing.assert_close(cached, full, atol=1e-6, rtol=1e-5)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sampling_rows_are_independent() -> None:
    """Batch admission and row reordering must not change a request's random draw."""
    torch.manual_seed(23)
    logits = torch.randn(4, 65, device="cuda")
    sampling_seeds = torch.tensor([99, 20, 80, 1], device="cuda")
    positions = torch.tensor([3, 7, 11, 0], device="cuda")
    sampling = torch.tensor(
        [[0, 0, 1], [0.8, 8, 0.95], [1.1, 0, 0.7], [0.5, 1, 1]], device="cuda"
    )
    with production_sampler_context() as sampler:
        batched = sample_codes(
            sampler, logits.clone(), sampling_seeds, positions, sampling
        )
        individual = torch.cat(
            [
                sample_codes(
                    sampler,
                    logits[i : i + 1].clone(),
                    sampling_seeds[i : i + 1],
                    positions[i : i + 1],
                    sampling[i : i + 1],
                )
                for i in range(4)
            ]
        )
        assert torch.equal(batched, individual)
        order = torch.tensor([2, 0, 3, 1], device="cuda")
        assert torch.equal(
            sample_codes(
                sampler,
                logits[order],
                sampling_seeds[order],
                positions[order],
                sampling[order],
            ),
            batched[order],
        )
    assert batched[0] == logits[0].argmax()
    assert batched[3] == logits[3].argmax()


@pytest.mark.parametrize("frame_count", [1, 7, 8, 9, 17])
def test_frame_stream_flushes_tail_once(frame_count: int) -> None:
    """EOS between transport blocks must neither drop nor duplicate audio frames."""
    request = HeartMoERequest()
    stream = FrameStream()
    messages = []
    for i in range(frame_count):
        request.frames.append(torch.full((8,), i, dtype=torch.long))
        messages.extend(stream("request", request, None))
    messages.extend(stream.flush("request", request))
    assert list(stream.flush("request", request)) == []
    frames = torch.cat([message.data for message in messages])
    assert frames[:, 0].tolist() == list(range(frame_count))


def test_prompt_preserves_training_boundaries() -> None:
    prompt = render_prompt("[Instrumental]\n", "Jazz piano")
    assert (
        "<|style_start|>Jazz piano<|style_end|>\n<|lyrics_start|>[Instrumental]\n<|lyrics_end|>\n<|im_end|>"
        in prompt
    )
    assert prompt.endswith("<|im_start|>assistant\n")
    with pytest.raises(ValueError, match="reserved marker"):
        render_prompt("lyrics<|music_audio_end|>", "style")


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["pytorch", "flashinfer"])
def test_depth_cuda_graph_matches_eager_with_mixed_sampling(backend: str) -> None:
    """Reusing a padded graph must preserve each row's codes and feedback."""
    torch.manual_seed(42)
    audio = (
        AudioModules(
            dict(
                codebook_size=8192,
                hidden_size=32,
                special_token_ids=list(range(9)),
                local_decoder=dict(
                    hidden_size=32,
                    intermediate_size=48,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    partial_rotary_factor=0.5,
                    rope_theta=10000,
                    rms_norm_eps=1e-6,
                    num_hidden_layers=2,
                ),
            )
        )
        .cuda()
        .eval()
    )
    with production_sampler_context(backend), torch.inference_mode():
        decoder = FrameDecoder(audio, [1, 2, 4], use_graph=True)
        for batch_size in (3, 1, 4, 2):
            hidden = torch.randn(batch_size, 32, device="cuda")
            sampling_seeds = torch.arange(batch_size, device="cuda") + 17
            positions = torch.arange(batch_size, device="cuda") * 8 + batch_size
            sampling = torch.tensor(
                [
                    [0, 0.05, 0, 0, 1],
                    [1.2, 0.05, 0.8, 50, 0.95],
                    [1.0, 0, 1.0, 8, 0.7],
                    [0.5, 1, 0.5, 1, 1],
                ],
                device="cuda",
            )[:batch_size]
            codes, feedback = decoder(hidden, sampling_seeds, positions, sampling)
            eager_codes, eager_feedback = decoder.run(
                hidden, sampling_seeds, positions, sampling
            )
            assert torch.equal(codes, eager_codes)
            torch.testing.assert_close(feedback, eager_feedback)


def test_bundled_codec_preserves_weights_and_separates_ar_shards(
    tmp_path: Path,
) -> None:
    """Packing must preserve tensors while keeping codec weights out of the AR loader."""
    model_path, codec_path = tmp_path / "model", tmp_path / "codec"
    model_path.mkdir()
    codec_path.mkdir()
    backbone = {
        "audio.weight": torch.arange(6, dtype=torch.float32),
        "model.weight": torch.arange(30, dtype=torch.float32),
    }
    codec = {
        "flow_matching.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "scalar_model.weight": torch.arange(12, dtype=torch.float32),
    }
    for directory, weights, config in (
        (model_path, backbone, {"audio_config": {}}),
        (codec_path, codec, {"sample_rate": 48000}),
    ):
        save_file(weights, str(directory / "model.safetensors"))
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum(
                            t.numel() * t.element_size() for t in weights.values()
                        )
                    },
                    "weight_map": {name: "model.safetensors" for name in weights},
                }
            )
        )
    pack_codec(model_path, codec_path, max_shard_size=80)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    packed = load_file(
        str(model_path / index["weight_map"]["heartcodec.flow_matching.weight"])
    )
    assert torch.equal(
        packed["heartcodec.flow_matching.weight"], codec["flow_matching.weight"]
    )
    assert torch.equal(packed["audio.weight"], backbone["audio.weight"])
    assert torch.equal(
        load_file(str(model_path / index["weight_map"]["model.weight"]))[
            "model.weight"
        ],
        backbone["model.weight"],
    )
    assert index["metadata"]["total_size"] == 240
    assert (
        json.loads((model_path / "config.json").read_text())["heartcodec_config"][
            "sample_rate"
        ]
        == 48000
    )
    factory = ARFactoryArgs()
    builder = HeartMoEEngineBuilder(factory.max_abc_tokens, factory.abc_temperature)
    view = Path(builder.resolve_checkpoint(str(model_path)))
    try:
        ar_index = json.loads((view / "model.safetensors.index.json").read_text())
        assert set(ar_index["weight_map"]) == {"audio.weight", "model.weight"}
        assert {path.name for path in view.glob("*.safetensors")} == set(
            ar_index["weight_map"].values()
        )
        assert not (
            view / index["weight_map"]["heartcodec.scalar_model.weight"]
        ).exists()
    finally:
        builder.checkpoint_view.cleanup()


def test_codec_cancellation_interrupts_between_flow_steps() -> None:
    """Cancelling a decode must stop after the current flow step, before the next."""
    model = FlowMatching(
        dim=16,
        codebook_size=32,
        codebook_dim=4,
        attention_head_dim=4,
        in_channels=32,
        num_attention_heads=2,
        num_layers=1,
        num_layers_2=1,
        out_channels=8,
    ).eval()
    abort_event = Event()
    steps = []

    def cancel_after_step(
        module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        steps.append(1)
        abort_event.set()

    hook = model.estimator.register_forward_hook(cancel_after_step)
    try:
        with pytest.raises(InterruptedError, match="aborted"):
            model.inference_codes(
                [torch.zeros(1, 8, 3, dtype=torch.long)],
                torch.zeros(1, 6, 8),
                6,
                0,
                num_steps=3,
                abort_event=abort_event,
            )
    finally:
        hook.remove()
    assert len(steps) == 1


@pytest.mark.parametrize("guidance_scale", [1.0, 1.25])
def test_codec_batch_preserves_context_and_request_noise(guidance_scale: float) -> None:
    """Mixed first/continuation windows must match independent request decoding."""
    torch.manual_seed(37)
    model = FlowMatching(
        dim=16,
        codebook_size=32,
        codebook_dim=4,
        attention_head_dim=4,
        in_channels=32,
        num_attention_heads=2,
        num_layers=1,
        num_layers_2=1,
        out_channels=8,
    ).eval()
    codes = torch.randint(32, (3, 8, 5))
    context = torch.randn(3, 10, 8)
    noise = torch.randn_like(context)
    context_lengths = [0, 2, 4]
    expected = torch.cat(
        [
            model.inference_codes(
                [codes[i : i + 1]],
                context[i : i + 1],
                10,
                context_lengths[i],
                initial_noise=noise[i : i + 1],
                guidance_scale=guidance_scale,
                num_steps=3,
                scenario="other_seg",
            )
            for i in range(3)
        ]
    )
    for order in ([0, 1, 2], [2, 0, 1]):
        actual = model.inference_codes(
            [codes[order]],
            context[order],
            10,
            0,
            context_lengths=[context_lengths[i] for i in order],
            initial_noise=noise[order],
            guidance_scale=guidance_scale,
            num_steps=3,
            scenario="other_seg",
        )
        torch.testing.assert_close(actual, expected[order], atol=1e-6, rtol=1e-5)
        for i, original_index in enumerate(order):
            torch.testing.assert_close(
                actual[i, : context_lengths[original_index]],
                context[original_index, : context_lengths[original_index]],
                atol=0,
                rtol=0,
            )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_codec_graph_replays_changed_batches_and_inputs() -> None:
    """Graph reuse must copy new inputs and preserve mixed-row latent conditioning."""
    torch.manual_seed(41)
    model = (
        FlowMatching(
            dim=16,
            codebook_size=32,
            codebook_dim=4,
            attention_head_dim=4,
            in_channels=32,
            num_attention_heads=2,
            num_layers=1,
            num_layers_2=1,
            out_channels=8,
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    model.capture_estimator(10, 1.25, 3)
    graph_runner = model.estimator_runner
    for batch_size in (3, 1, 2, 3):
        codes = torch.randint(32, (batch_size, 8, 5), device="cuda")
        context = torch.randn(batch_size, 10, 8, device="cuda")
        noise = torch.randn_like(context)
        context_lengths = [i * 2 for i in range(batch_size)]
        outputs = []
        for runner in (None, graph_runner):
            model.estimator_runner = runner
            outputs.append(
                model.inference_codes(
                    [codes],
                    context,
                    10,
                    0,
                    context_lengths=context_lengths,
                    initial_noise=noise,
                    guidance_scale=1.25,
                    num_steps=3,
                    scenario="other_seg",
                )
            )
        torch.testing.assert_close(outputs[0], outputs[1], atol=1e-6, rtol=1e-5)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_codec_stream_batch_cancellation_and_tail_accounting() -> None:
    """Cancelling one batch row must preserve peers and exact partial-window tails."""
    torch.manual_seed(43)
    model = (
        HeartCodec(
            HeartCodecConfig(
                dim=16,
                codebook_size=32,
                codebook_dim=4,
                attention_head_dim=4,
                in_channels=528,
                num_attention_heads=2,
                num_layers=1,
                num_layers_2=1,
                init_channel=2,
                default_kernel_size=3,
                res_kernel_size=3,
                delay_kernel_size=3,
            )
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    scheduler = HeartCodecScheduler(model, 5, 1, 3, 1.25, 3)
    frame_counts = {"long": 11, "short": 3, "cancelled": 10}
    payloads = {
        request_id: StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="test"),
            data={"frames": frames},
        )
        for request_id, frames in frame_counts.items()
    }
    scheduler.handle_streaming_new_request("long", payloads["long"])
    scheduler.handle_streaming_new_request("cancelled", payloads["cancelled"])

    def cancel_one_row(
        module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        scheduler.abort("cancelled")

    hook = model.flow_matching.estimator.register_forward_hook(cancel_one_row)
    try:
        scheduler.on_stream_chunk_batch(
            [
                (
                    request_id,
                    StreamItem(
                        chunk_id=0,
                        data=torch.randint(32, (frames, 8)),
                        from_stage="test",
                    ),
                )
                for request_id, frames in frame_counts.items()
            ]
        )
    finally:
        hook.remove()
    scheduler.handle_stream_done("long")
    scheduler.handle_stream_done("short")
    scheduler.handle_streaming_new_request("short", payloads["short"])
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    assert all(message.request_id != "cancelled" for message in messages)
    assert all(message.type != "error" for message in messages)
    for request_id in ("long", "short"):
        chunks = [
            message.data["audio_waveform"]
            for message in messages
            if message.request_id == request_id and message.type == "stream"
        ]
        results = [
            message.data
            for message in messages
            if message.request_id == request_id and message.type == "result"
        ]
        assert len(results) == 1
        streamed_waveform = torch.cat(
            [
                torch.frombuffer(bytearray(chunk), dtype=torch.float32).reshape(2, -1)
                for chunk in chunks
            ],
            dim=-1,
        )
        completed_waveform = torch.frombuffer(
            bytearray(results[0].data["audio_waveform"]), dtype=torch.float32
        ).reshape(2, -1)
        torch.testing.assert_close(
            streamed_waveform, completed_waveform, atol=0, rtol=0
        )
        assert (
            sum(len(chunk) for chunk in chunks)
            == frame_counts[request_id] * SAMPLES_PER_FRAME * 2 * 4
        )
    assert scheduler.states == {}
