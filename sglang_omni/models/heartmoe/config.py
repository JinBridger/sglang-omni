# SPDX-License-Identifier: Apache-2.0
"""HeartMoE autoregressive generation followed by streaming HeartCodec."""

import sysconfig
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.heartmoe"


class ARFactoryArgs(FactoryArgs):
    max_abc_tokens: int = Field(default=4096, ge=1, le=4096)
    abc_temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)


class ARStageConfig(EngineStageConfig):
    factory: ARFactoryArgs = Field(default_factory=ARFactoryArgs)


class CodecFactoryArgs(FactoryArgs):
    chunk_frames: int = Field(default=93, ge=2)
    overlap_frames: int = Field(default=13, ge=1)
    num_steps: int = Field(default=10, ge=1)
    guidance_scale: float = Field(default=1.25, ge=0)
    max_batch_size: int = Field(default=1, ge=1, le=8)
    compile_codec: bool = False
    use_cuda_graph: bool = False


class CodecStageConfig(StageConfig):
    factory: CodecFactoryArgs = Field(default_factory=CodecFactoryArgs)


def stages() -> list[StageConfig]:
    return [
        ARStageConfig(
            name="heartmoe_ar",
            process="heartmoe_ar",
            factory_path=f"{PACKAGE}.engine.create_ar_executor",
            gpu=0,
            next="heartcodec",
            stream_to=["heartcodec"],
        ),
        CodecStageConfig(
            name="heartcodec",
            process="heartcodec",
            factory_path=f"{PACKAGE}.codec.create_codec_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class HeartMoEPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "HeartMoEForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    supported_speech_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "input",
            "instructions",
            "task_type",
            "ref_audio",
            "voice",
            "speed",
            "response_format",
            "stream",
            "stream_format",
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "seed",
        }
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "heartmoe_ar": ARStageConfig,
        "heartcodec": CodecStageConfig,
    }
    stages: list[StageConfig] = Field(default_factory=stages)
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )

    def resolved_stage_env_defaults(self, stage_name: str) -> dict[str, str]:
        environment = super().resolved_stage_env_defaults(stage_name)
        if stage_name == "heartcodec":
            directory = Path(sysconfig.get_path("purelib")) / "nvidia" / "cudnn" / "lib"
            libraries = sorted(directory.glob("libcudnn*.so.9"))
            if libraries:
                # note (Codex): Preload matching sublibraries even when the host exports a system CUDA library path.
                environment.setdefault("LD_PRELOAD", ":".join(map(str, libraries)))
            else:
                pass
        else:
            pass
        return environment

    def validate_processes(self) -> None:
        super().validate_processes()
        autoregressive = self.stage_named("heartmoe_ar")
        codec = self.stage_named("heartcodec")
        if (autoregressive.process or autoregressive.name) == (
            codec.process or codec.name
        ):
            raise ValueError("HeartMoE and HeartCodec require separate processes")
        elif any(stage.tp_size != 1 for stage in self.stages):
            raise ValueError("HeartMoE currently supports tp_size=1")
        elif codec.factory.overlap_frames >= codec.factory.chunk_frames:
            raise ValueError("HeartCodec overlap_frames must be less than chunk_frames")
        else:
            pass


EntryClass = HeartMoEPipelineConfig
