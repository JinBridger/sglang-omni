# SPDX-License-Identifier: Apache-2.0
"""HeartMoE ABC and audio generation in continuous batches."""

import logging

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.heartmoe.local import FrameDecoder
from sglang_omni.models.heartmoe.model import HeartMoEModel
from sglang_omni.models.heartmoe.request import (
    ABC_END,
    AUDIO_END,
    AUDIO_PLACEHOLDER,
    AUDIO_START,
    CODEBOOK_SIZE,
    HeartMoERequest,
)
from sglang_omni.models.heartmoe.sampling import MASK_LOGIT_OFFSET, sample_codes
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest

logger = logging.getLogger(__name__)


class HeartMoERunner(ModelRunner[HeartMoERequest]):
    model: HeartMoEModel

    def __init__(
        self,
        worker: ModelWorker,
        output: SGLangOutputProcessor,
        decoder: FrameDecoder,
        abc_temperature: float,
    ) -> None:
        super().__init__(worker, output)
        self.decoder = decoder
        self.abc_temperature = abc_temperature
        self.previous_batch_size = 0

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.LAST

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.FULL

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        for request in requests:
            state: HeartMoERequest = request.data
            state.feedback = torch.zeros_like(self.model.feedback[0])

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        self.model.feedback[: len(requests)].copy_(
            torch.stack([request.data.feedback for request in requests])
        )

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.advance(result, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.advance(result, requests)

    @torch.inference_mode()
    def advance(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> None:
        states: list[HeartMoERequest] = [request.data for request in requests]
        hidden_states = result.logits_output.hidden_states
        result.next_token_ids = torch.full(
            (len(states),), AUDIO_START, dtype=torch.long, device=hidden_states.device
        )
        abc_indices: list[int] = []
        abc_end_indices: list[int] = []
        audio_indices: list[int] = []
        for index, state in enumerate(states):
            if (
                state.phase == "abc"
                and len(state.abc_token_ids) >= state.max_abc_tokens
            ):
                abc_end_indices.append(index)
                logger.info(
                    f"HeartMoE ending ABC at token limit request_id={state.req.rid} "
                    f"abc_tokens={len(state.abc_token_ids)}"
                )
            elif state.phase == "abc":
                abc_indices.append(index)
            elif state.phase == "audio_start":
                state.phase = "audio"
            else:
                audio_indices.append(index)

        if abc_indices:
            logits = self.model.lm_head.quant_method.apply(
                self.model.lm_head, hidden_states[abc_indices], self.model.lm_head.bias
            )[:, : self.model.audio_config["layout"]["text_vocab_size"]].float()
            forbidden_tokens = [
                token_id
                for token_id in self.model.audio_config["special_token_ids"]
                if token_id != ABC_END
            ]
            logits[:, forbidden_tokens] = (
                logits.amax(dim=-1, keepdim=True) - MASK_LOGIT_OFFSET
            )
            sampling_seeds = torch.tensor(
                [states[index].sampling_seed for index in abc_indices],
                device=hidden_states.device,
            )
            positions = torch.tensor(
                [len(states[index].abc_token_ids) for index in abc_indices],
                device=hidden_states.device,
            )
            sampling = torch.tensor(
                [
                    [
                        (
                            self.abc_temperature
                            if states[index].options.abc_temperature is None
                            else states[index].options.abc_temperature
                        ),
                        states[index].options.top_k,
                        states[index].options.top_p,
                    ]
                    for index in abc_indices
                ],
                device=hidden_states.device,
            )
            token_ids = sample_codes(
                self.decoder.sampler, logits, sampling_seeds, positions, sampling
            )
            result.next_token_ids[abc_indices] = token_ids
            for index, token_id in zip(abc_indices, token_ids.tolist(), strict=True):
                state = states[index]
                if token_id == ABC_END:
                    abc_end_indices.append(index)
                else:
                    state.abc_token_ids.append(token_id)
        else:
            pass

        for index in abc_end_indices:
            state = states[index]
            result.next_token_ids[index] = ABC_END
            state.phase = "audio_start"
            # note (Codex): Reserve both modality markers without spending the audio frame budget on ABC.
            state.max_new_tokens = min(
                state.req.sampling_params.max_new_tokens,
                len(state.req.output_ids) + 2 + state.options.max_new_tokens,
            )
            state.req.sampling_params.max_new_tokens = state.max_new_tokens

        if audio_indices:
            sampling_seeds = torch.tensor(
                [states[index].sampling_seed for index in audio_indices],
                device=hidden_states.device,
            )
            positions = torch.tensor(
                [
                    states[index].max_abc_tokens + states[index].generated_frames * 8
                    for index in audio_indices
                ],
                device=hidden_states.device,
            )
            sampling = torch.tensor(
                [
                    [
                        states[index].options.first_codebook_temperature,
                        states[index].options.first_codebook_min_p,
                        states[index].options.temperature,
                        states[index].options.top_k,
                        states[index].options.top_p,
                    ]
                    for index in audio_indices
                ],
                device=hidden_states.device,
            )
            codes, feedback = self.decoder(
                hidden_states[audio_indices], sampling_seeds, positions, sampling
            )
            stopped = codes[:, 0] == CODEBOOK_SIZE
            result.next_token_ids[audio_indices] = torch.where(
                stopped, AUDIO_END, AUDIO_PLACEHOLDER
            )
            for row, (index, stop) in enumerate(
                zip(audio_indices, stopped.tolist(), strict=True)
            ):
                if not stop:
                    states[index].frames.append(codes[row].clone())
                    states[index].feedback = feedback[row].clone()
                    states[index].generated_frames += 1
                else:
                    pass
        else:
            pass
        if len(requests) != self.previous_batch_size:
            logger.info(f"HeartMoE active batch size={len(requests)}")
            self.previous_batch_size = len(requests)
        else:
            pass
