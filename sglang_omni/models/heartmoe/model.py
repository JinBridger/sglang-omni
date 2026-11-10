# SPDX-License-Identifier: Apache-2.0
"""SGLang Qwen3.5-MoE backbone with HeartMoE audio weights and feedback."""

from collections.abc import Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from sglang_omni.models.heartmoe.local import AudioModules
from sglang_omni.models.heartmoe.request import AUDIO_PLACEHOLDER


class HeartMoEModel(Qwen3_5MoeForConditionalGeneration):
    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.audio = AudioModules(config.audio_config)
        self.audio_config = config.audio_config
        self.register_buffer(
            "feedback", torch.zeros(1, self.config.hidden_size), persistent=False
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        audio_state: dict[str, torch.Tensor] = {}

        def backbone_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if name.startswith("heartcodec."):
                    continue
                elif name.endswith("._extra_state"):
                    if tensor.numel():
                        raise ValueError(f"Unsupported nonempty TE state: {name}")
                    else:
                        pass
                elif name.startswith(("model.", "lm_head.")):
                    yield name, tensor
                else:
                    audio_state[name] = tensor

        loaded = super().load_weights(backbone_weights())
        self.audio.load_state_dict(audio_state, strict=True)
        loaded.update(f"audio.{name}" for name in audio_state)
        with torch.no_grad():
            for i, token_id in enumerate(self.audio_config["special_token_ids"]):
                self.model.embed_tokens.weight[token_id].copy_(
                    self.audio.special_token_embeddings.embedding.weight[i]
                )
        return loaded

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        if forward_batch.forward_mode.is_decode():
            input_embeds = torch.where(
                (input_ids == AUDIO_PLACEHOLDER).unsqueeze(-1),
                self.feedback[: input_ids.shape[0]],
                self.model.embed_tokens(input_ids),
            )
        else:
            pass
        hidden = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if forward_batch.forward_mode.is_extend():
            hidden = hidden[torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1]
        else:
            pass
        return LogitsProcessorOutput(
            next_token_logits=self.audio.audio_logits(hidden).float(),
            hidden_states=hidden,
        )
