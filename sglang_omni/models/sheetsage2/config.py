# SPDX-License-Identifier: Apache-2.0
"""SheetSage2 pipeline configuration."""

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config import (
    AudioChunkingConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)


class SheetSage2FactoryArgs(FactoryArgs):
    base_model_path: str | None = None
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    max_batch_size: int = Field(default=8, ge=1)
    max_queued_requests: int = Field(default=64, ge=0)
    max_audio_seconds: float = Field(default=3600, gt=0)
    enable_cuda_graph: bool = True


class SheetSage2StageConfig(StageConfig):
    factory: SheetSage2FactoryArgs = Field(default_factory=SheetSage2FactoryArgs)


class SheetSage2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "SheetSage2Model"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "SheetSage2ForConditionalGeneration",
    )
    allow_audio_chunking: ClassVar[bool] = False
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "transcribe": SheetSage2StageConfig,
    }
    audio_chunking: AudioChunkingConfig = AudioChunkingConfig(
        max_audio_clip_s=3600, max_total_audio_s=3600
    )
    stages: list[StageConfig] = [
        SheetSage2StageConfig(
            name="transcribe",
            process="transcribe",
            factory_path="sglang_omni.models.sheetsage2.stages.create_executor",
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = SheetSage2PipelineConfig
