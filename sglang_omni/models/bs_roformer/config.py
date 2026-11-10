# SPDX-License-Identifier: Apache-2.0
"""Serving settings for BS-RoFormer source separation."""

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig

CHECKPOINT_FILENAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


class BSRoFormerFactoryArgs(FactoryArgs):
    checkpoint_filename: str = CHECKPOINT_FILENAME
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    max_audio_seconds: float = Field(default=600, gt=0)
    enable_torch_compile: bool = True


class BSRoFormerStageConfig(StageConfig):
    factory: BSRoFormerFactoryArgs = Field(default_factory=BSRoFormerFactoryArgs)
    max_queued_requests: int = Field(default=32, ge=0)


class BSRoFormerPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "BSRoFormer"
    allow_audio_chunking: ClassVar[bool] = False
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "separate": BSRoFormerStageConfig,
    }
    stages: list[StageConfig] = [
        BSRoFormerStageConfig(
            name="separate",
            process="separate",
            factory_path="sglang_omni.models.bs_roformer.stages.create_executor",
            gpu=0,
            terminal=True,
        )
    ]

    def entry_stage_admission_limit(self) -> int:
        stage = self.stages[0]
        assert isinstance(stage, BSRoFormerStageConfig)
        return 1 + stage.max_queued_requests


EntryClass = BSRoFormerPipelineConfig
