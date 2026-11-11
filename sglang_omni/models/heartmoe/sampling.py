# SPDX-License-Identifier: Apache-2.0
"""Seeded HeartMoE draws through the production SGLang sampler."""

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

MASK_LOGIT_OFFSET = 1e4


def sample_codes(
    sampler: Sampler,
    logits: torch.Tensor,
    sampling_seeds: torch.Tensor,
    positions: torch.Tensor,
    sampling: torch.Tensor,
    *,
    apply_top_k_top_p: bool = True,
) -> torch.Tensor:
    temperature, top_k, top_p = sampling.unbind(-1)
    greedy_codes = logits.argmax(-1)
    sampling_info = SamplingBatchInfo(
        temperatures=torch.where(temperature == 0, 1, temperature).unsqueeze(-1),
        top_ks=torch.where(top_k > 0, top_k, logits.shape[-1]).to(torch.int32),
        top_ps=top_p,
        min_ps=torch.zeros_like(temperature),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_k_sampling=apply_top_k_top_p,
        need_top_p_sampling=apply_top_k_top_p,
        need_min_p_sampling=False,
        vocab_size=logits.shape[-1],
        sampling_seed=sampling_seeds,
        device=logits.device.type,
    )
    selected = sampler(
        LogitsProcessorOutput(next_token_logits=logits.float()),
        sampling_info,
        False,
        [0] * logits.shape[0],
        [[] for _ in range(logits.shape[0])],
        positions,
    )
    return torch.where(temperature == 0, greedy_codes, selected)
