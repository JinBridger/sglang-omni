# SPDX-License-Identifier: Apache-2.0
"""HeartMoE hooks for the shared continuous batching scheduler."""

import json
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import torch
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from transformers import AutoTokenizer

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.heartmoe.local import FrameDecoder
from sglang_omni.models.heartmoe.model import HeartMoEModel
from sglang_omni.models.heartmoe.request import (
    FrameStream,
    HeartMoERequest,
    build_request,
    build_result,
)
from sglang_omni.models.heartmoe.runner import HeartMoERunner
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.generation_batch_policy import build_default_cuda_graph_bs
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class ResultAdapter(Protocol):
    def __call__(self, request: HeartMoERequest) -> StagePayload: ...


class HeartMoEEngineBuilder(TtsEngineBuilder[HeartMoERequest]):
    model_name = "heartmoe"
    context_length = 16384

    def __init__(self, max_abc_tokens: int, abc_temperature: float) -> None:
        self.max_abc_tokens = max_abc_tokens
        self.abc_temperature = abc_temperature

    def resolve_checkpoint(self, model_path: str) -> str:
        source = Path(model_path).resolve()
        original = json.loads((source / "config.json").read_text())
        original.pop("heartcodec_config", None)
        index = json.loads((source / "model.safetensors.index.json").read_text())
        index["weight_map"] = {
            name: filename
            for name, filename in index["weight_map"].items()
            if not name.startswith("heartcodec.")
        }
        index.pop("metadata", None)
        shards = set(index["weight_map"].values())
        self.checkpoint_view = TemporaryDirectory(prefix="heartmoe-")
        destination = Path(self.checkpoint_view.name)
        for path in source.iterdir():
            if (
                path.is_file()
                and path.name not in {"config.json", "model.safetensors.index.json"}
                and (path.suffix != ".safetensors" or path.name in shards)
            ):
                (destination / path.name).symlink_to(path)
            else:
                pass
        (destination / "model.safetensors.index.json").write_text(json.dumps(index))
        original.update(
            model_type="qwen3_5_moe_text", architectures=["Qwen3_5MoeForCausalLM"]
        )
        config = dict(
            model_type="qwen3_5_moe",
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            language_model_only=True,
            tie_word_embeddings=False,
            dtype="bfloat16",
            text_config=original,
            audio_config=original["audio_config"],
        )
        (destination / "config.json").write_text(json.dumps(config))
        return str(destination)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        ModelRegistry.models["Qwen3_5MoeForConditionalGeneration"] = HeartMoEModel
        torch.set_num_threads(8)
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_cudnn_sdp(False)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": 16,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.5,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("HeartMoE currently supports tp_size=1")
        else:
            pass
        self.max_running_requests = int(overrides["max_running_requests"])
        self.batch_sizes = build_default_cuda_graph_bs(self.max_running_requests)
        overrides.update(
            cuda_graph_max_bs=self.max_running_requests,
            cuda_graph_bs=self.batch_sizes,
            enable_return_hidden_states=True,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            chunked_prefill_size=0,
            mamba_ssm_dtype="float32",
        )
        overrides.setdefault("attention_backend", "triton")
        overrides.setdefault("weight_loader_prefetch_checkpoints", True)
        overrides.setdefault(
            "max_total_tokens", self.context_length * self.max_running_requests
        )
        overrides.setdefault("max_mamba_cache_size", self.max_running_requests + 4)

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        model = model_worker.model_runner.model
        model.feedback = torch.zeros(
            max(self.batch_sizes),
            model.config.hidden_size,
            device=device,
            dtype=model.audio.audio_head.weight.dtype,
        )
        model.eval().requires_grad_(False)

    def setup_model_resources(
        self,
        model: HeartMoEModel,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        self.decoder = FrameDecoder(
            model.audio, self.batch_sizes, generation_cuda_graph_enabled
        )

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> HeartMoERunner:
        return HeartMoERunner(
            model_worker, output_proc, self.decoder, self.abc_temperature
        )

    def make_adapters(
        self, model: HeartMoEModel
    ) -> tuple[partial[HeartMoERequest], ResultAdapter]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_dir, trust_remote_code=False
        )
        return (
            partial(
                build_request,
                tokenizer=tokenizer,
                context_length=self.context_length,
                max_abc_tokens=self.max_abc_tokens,
            ),
            build_result,
        )

    def extra_scheduler_kwargs(self) -> SchedulerExtras[HeartMoERequest]:
        return {"stream_output_builder": FrameStream(), "enable_async_decode": False}

    def post_scheduler_setup(
        self, scheduler: OmniScheduler[HeartMoERequest], model_runner: HeartMoERunner
    ) -> None:
        # note (Codex): Audio feedback cannot be reconstructed from placeholder token IDs after retraction.
        scheduler.new_token_ratio_tracker = NewTokenRatioTracker(
            init=1.0, min=1.0, decay=0.0, current=1.0
        )


def create_ar_executor(
    model_path: str,
    *,
    max_abc_tokens: int,
    abc_temperature: float,
    gpu_id: int | None = None,
    device: str | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[HeartMoERequest]:
    builder = HeartMoEEngineBuilder(max_abc_tokens, abc_temperature)
    scheduler = builder.build(
        model_path,
        gpu_id=gpu_id,
        device=device,
        server_args_overrides=server_args_overrides,
    )
    # note (Codex): The scheduler keeps the temporary config view alive until shutdown.
    scheduler.heartmoe_builder = builder
    return scheduler
