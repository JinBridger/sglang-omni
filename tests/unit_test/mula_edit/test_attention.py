# SPDX-License-Identifier: Apache-2.0
"""Native GQA, local window boundaries, and compiled music attention."""

import copy

import pytest
import torch
from sglang.kernels.ops.attention.flash_attention_v4 import (
    is_flash_attention_v4_available,
)
from torch.nn import functional as F

from sglang_omni.models.mula_edit.layers import local_attention, rotary_embeddings
from sglang_omni.models.mula_edit.sampling import MulaSampler
from tests.unit_test.fixtures.mula_edit import sampling_state, small_decoder


@pytest.mark.parametrize("uses_cross_attention", [False, True])
@torch.inference_mode()
def test_native_gqa_keeps_all_condition_tokens(uses_cross_attention: bool) -> None:
    decoder = small_decoder()
    attention = decoder.layers[0].self_attn
    hidden_states = torch.randn(2, 7, decoder.config.hidden_size)
    conditions = torch.randn(2, 11, decoder.config.hidden_size)
    if uses_cross_attention:
        cached = attention.project_keys_values(conditions)
        positions = None
        mask = torch.zeros(1, 1, 7, 11)
    else:
        cached = None
        positions = rotary_embeddings(hidden_states, decoder.config)
        mask = torch.zeros(1, 1, 7, 7)
    expected = attention(hidden_states, mask, positions, cached)
    attention.use_native_attention = True
    actual = attention(hidden_states, None, positions, cached)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("frames", "window_frames"), [(1, 0), (17, 3), (257, 128)])
@torch.inference_mode()
def test_flash_attention_preserves_inclusive_window(
    frames: int, window_frames: int
) -> None:
    if (
        torch.cuda.get_device_capability()[0] != 12
        or not is_flash_attention_v4_available()
    ):
        pytest.skip("requires FlashAttention-4 on SM12x")
    else:
        pass
    torch.manual_seed(42)
    queries = torch.randn(2, 4, frames, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(2, 2, frames, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    positions = torch.arange(frames, device="cuda")
    mask = torch.zeros(frames, frames, device="cuda", dtype=queries.dtype).masked_fill(
        (positions[:, None] - positions[None, :]).abs() > window_frames,
        torch.finfo(queries.dtype).min,
    )
    expected = F.scaled_dot_product_attention(
        queries,
        keys.repeat_interleave(2, 1),
        values.repeat_interleave(2, 1),
        attn_mask=mask,
    )
    actual = local_attention(queries, keys, values, window_frames)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=4e-3)
    if frames > window_frames + 1:
        values[:, :, window_frames + 1 :] += 10
        outside_changed = local_attention(queries, keys, values, window_frames)
        torch.testing.assert_close(
            outside_changed[:, :, 0], actual[:, :, 0], rtol=0, atol=0
        )
    else:
        pass
    values[:, :, window_frames] += 10
    boundary_changed = local_attention(queries, keys, values, window_frames)
    assert not torch.equal(boundary_changed[:, :, 0], actual[:, :, 0])


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("uses_sliding_window", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_auto_attention_without_flash_keeps_decoder_output(
    monkeypatch: pytest.MonkeyPatch, uses_sliding_window: bool, dtype: torch.dtype
) -> None:
    decoder = small_decoder(
        head_dimension=128, uses_sliding_window=uses_sliding_window
    ).to(device="cuda", dtype=dtype)
    state = sampling_state(decoder, 9, 3, 7)
    sampler = MulaSampler(decoder, enable_torch_compile=False, enable_cuda_graph=False)
    sampler.prepare(state)
    assert sampler.inputs is not None
    expected = sampler.inputs.predict(decoder)
    monkeypatch.setattr(
        "sglang_omni.models.mula_edit.model.is_flash_attention_v4_available",
        lambda: False,
    )
    decoder.configure_attention("auto")
    assert not decoder.use_flash_attention
    actual = sampler.inputs.predict(decoder)
    tolerance = 0 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    decoder.configure_attention("sdpa")
    torch.testing.assert_close(
        sampler.inputs.predict(decoder), expected, rtol=0, atol=0
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("enable_torch_compile", "enable_cuda_graph"),
    [(True, False), (False, True), (True, True)],
)
@torch.inference_mode()
def test_native_attention_sampling_compiles_and_replays_across_edits(
    enable_torch_compile: bool, enable_cuda_graph: bool
) -> None:
    if (
        torch.cuda.get_device_capability()[0] != 12
        or not is_flash_attention_v4_available()
    ):
        pytest.skip("requires FlashAttention-4 on SM12x")
    else:
        pass
    torch.compiler.reset()
    decoder = small_decoder(head_dimension=128).to(device="cuda", dtype=torch.bfloat16)
    decoder.configure_attention("auto")
    eager = MulaSampler(decoder, enable_torch_compile=False, enable_cuda_graph=False)
    accelerated = MulaSampler(
        copy.deepcopy(decoder),
        enable_torch_compile=enable_torch_compile,
        enable_cuda_graph=enable_cuda_graph,
    )
    for frames, guidance in ((9, 7), (9, 7), (16, 1), (16, 7)):
        state = sampling_state(decoder, frames, 3, guidance)
        expected = copy.deepcopy(state)
        eager.prepare(expected)
        accelerated.prepare(state)
        while not state.done:
            eager.advance(expected)
            accelerated.advance(state)
            torch.testing.assert_close(state.latents, expected.latents, rtol=0, atol=0)
        torch.testing.assert_close(
            state.latents[~state.mask], state.clean_latents[~state.mask], rtol=0, atol=0
        )
