# SPDX-License-Identifier: Apache-2.0
"""Deployment settings for continuously batched music edits."""

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig


class MulaEditFactoryArgs(FactoryArgs):
    checkpoint_dir: str | None = None
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    max_batch_size: int = Field(default=4, ge=1)
    max_queued_requests: int = Field(default=32, ge=0)
    max_audio_seconds: float = Field(default=240, gt=0, le=600)


class MulaEditStageConfig(StageConfig):
    factory: MulaEditFactoryArgs = Field(default_factory=MulaEditFactoryArgs)


class MulaEditPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "AceStepV4ConditionGenerationModel"
    allow_audio_chunking: ClassVar[bool] = False
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "edit": MulaEditStageConfig,
    }
    stages: list[StageConfig] = [
        MulaEditStageConfig(
            name="edit",
            process="edit",
            factory_path="sglang_omni.models.mula_edit.stages.create_executor",
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = MulaEditPipelineConfig
