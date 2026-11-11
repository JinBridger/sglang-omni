# SPDX-License-Identifier: Apache-2.0
"""Run the production sampler without a distributed model worker."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from sglang.srt.layers.sampler import Sampler, create_sampler
from sglang.srt.runtime_context import get_context, get_parallel

from sglang_omni.model_runner.sglang_model_runner import (
    register_flashinfer_seeded_sampler,
)


@dataclass
class SingleRankSamplingGroup:
    device_group: None = None


@contextmanager
def production_sampler_context(
    backend: Literal["pytorch", "flashinfer"] = "pytorch",
) -> Iterator[Sampler]:
    register_flashinfer_seeded_sampler()
    with (
        get_context().override_server_args(
            sampling_backend=backend,
            enable_deterministic_inference=False,
        ),
        get_parallel().override(tp_group=SingleRankSamplingGroup()),
    ):
        yield create_sampler()
