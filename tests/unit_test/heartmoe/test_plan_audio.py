# SPDX-License-Identifier: Apache-2.0
"""HeartMoE task selection, mixed generation phases, and decode embeddings."""

from collections.abc import Iterator
from typing import Literal

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from torch import nn
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.heartmoe.config import ARFactoryArgs, HeartMoEPipelineConfig
from sglang_omni.models.heartmoe.local import AudioModules, FrameDecoder
from sglang_omni.models.heartmoe.model import HeartMoEModel
from sglang_omni.models.heartmoe.request import (
    ABC_END,
    ABC_START,
    AUDIO_PLACEHOLDER,
    AUDIO_START,
    HeartMoERequest,
    build_request,
    build_result,
)
from sglang_omni.models.heartmoe.runner import HeartMoERunner
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.serve.openai_errors import generation_error_status_code
from sglang_omni.serve.speech_service import SpeechRequestValidator

ABC_SCORE = "X:1\nT:Cover test\nM:4/4\nL:1/4\nQ:1/4=100\nK:C\nC D E G | G E D C |]\n"


class EmbeddingBackbone(nn.Module):
    """Isolate modality embedding selection from the transformer layers."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(248080, 8)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids) if input_embeds is None else input_embeds


def text_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(
        BPE(
            {
                character: index
                for index, character in enumerate(sorted(ByteLevel.alphabet()))
            },
            merges=[],
        )
    )
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer)


def speech_request(
    task_type: str | None,
    context_length: int = 4096,
    *,
    max_abc_tokens: int = 4,
    ref_audio: str | None = None,
) -> HeartMoERequest:
    validator = SpeechRequestValidator(
        default_model="heartmoe",
        supported_speech_fields=HeartMoEPipelineConfig.supported_speech_fields,
    )
    request = validator.parse_request(
        {
            "input": "[Instrumental]\n",
            "instructions": "Jazz piano",
            "task_type": task_type,
            "ref_audio": ref_audio,
            "max_new_tokens": 3,
            "temperature": 0,
            "response_format": "pcm",
        }
    )
    generated = validator.build_generate_request(request)
    state = build_request(
        StagePayload(
            request_id=task_type or "default",
            request=OmniRequest(
                inputs=generated.prompt,
                params=generated.sampling.to_dict(),
                metadata=generated.metadata,
            ),
            data={},
        ),
        text_tokenizer(),
        context_length=context_length,
        max_abc_tokens=max_abc_tokens,
    )
    # note (Codex): The byte tokenizer isolates prompts; generation retains the checkpoint vocabulary.
    state.req.vocab_size = 248080
    return state


@pytest.fixture
def runner() -> Iterator[HeartMoERunner]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(51)
    model = HeartMoEModel.__new__(HeartMoEModel)
    nn.Module.__init__(model)
    model.model = EmbeddingBackbone()
    model.lm_head = nn.Linear(8, 248080, bias=False)
    model.lm_head.quant_method = UnquantizedEmbeddingMethod()
    model.audio_config = {
        "layout": {"text_vocab_size": 248079},
        "special_token_ids": list(range(ABC_START, 248079)),
    }
    model.audio = AudioModules(
        dict(
            codebook_size=8192,
            hidden_size=8,
            special_token_ids=list(range(ABC_START, 248079)),
            local_decoder=dict(
                hidden_size=8,
                intermediate_size=16,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                partial_rotary_factor=0.5,
                rope_theta=10000,
                rms_norm_eps=1e-6,
                num_hidden_layers=1,
            ),
        )
    )
    model.register_buffer("feedback", torch.zeros(4, 8))
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[7, 0] = 10
        model.lm_head.weight[ABC_END, 1] = 10
        model.lm_head.weight[AUDIO_START, 0] = 20
        model.lm_head.weight[248079, 0] = 30
        model.audio.audio_end_head.weight.fill_(-20)
    model.eval().requires_grad_(False)
    instance = HeartMoERunner.__new__(HeartMoERunner)
    instance.model = model
    instance.device = torch.device("cpu")
    instance.decoder = FrameDecoder(model.audio, [1, 2, 4], use_graph=False)
    instance.abc_temperature = 0
    instance.previous_batch_size = 0
    try:
        yield instance
    finally:
        torch.set_num_threads(previous_threads)


def advance(
    runner: HeartMoERunner,
    states: list[HeartMoERequest],
    hidden_states: torch.Tensor,
) -> list[int]:
    requests = [
        SchedulerRequest(request_id=state.req.rid, data=state) for state in states
    ]
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )
    )
    runner.advance(result, requests)
    token_ids = result.next_token_ids.tolist()
    for state, token_id in zip(states, token_ids, strict=True):
        state.req.output_ids.append(token_id)
        state.req.update_finish_state()
    return token_ids


@pytest.mark.parametrize(
    ("task_type", "instruction", "first_token"),
    [
        (
            None,
            "Write a full ABC score, then generate the complete corresponding song audio.",
            ABC_START,
        ),
        (
            "Direct",
            "Generate the complete song audio directly, without an ABC score.",
            AUDIO_START,
        ),
        (
            "PlanFull",
            "Write a full ABC score, then generate the complete corresponding song audio.",
            ABC_START,
        ),
        (
            "PlanMelody",
            "Write melody-only ABC notation, then generate the complete corresponding song audio.",
            ABC_START,
        ),
        (
            "plan_melody",
            "Write melody-only ABC notation, then generate the complete corresponding song audio.",
            ABC_START,
        ),
    ],
)
def test_speech_tasks_select_prompt_and_first_modality(
    task_type: str | None, instruction: str, first_token: int
) -> None:
    """The HTTP speech path must preserve the task through model request building."""
    state = speech_request(task_type)
    prompt = text_tokenizer().decode(state.req.origin_input_ids[:-1])
    assert f"<|im_start|>user\n{instruction}\n" in prompt
    assert state.req.origin_input_ids[-1] == first_token
    assert state.options.max_new_tokens == 3


def test_cover_prefills_planfull_score_and_generates_audio_immediately(
    runner: HeartMoERunner,
) -> None:
    """A caller's ABC must condition audio as a completed PlanFull score without being generated."""
    cover = speech_request("cover", ref_audio=ABC_SCORE)
    plan = speech_request("PlanFull")
    score_tokens = text_tokenizer().encode(ABC_SCORE, add_special_tokens=False)
    assert cover.req.origin_input_ids == (
        plan.req.origin_input_ids + score_tokens + [ABC_END, AUDIO_START]
    )
    assert cover.phase == "audio"
    assert cover.max_abc_tokens == 0
    assert cover.max_new_tokens == cover.options.max_new_tokens == 3
    states = [cover, plan]
    runner.before_prefill(
        None,
        None,
        [SchedulerRequest(request_id=state.req.rid, data=state) for state in states],
    )
    assert advance(runner, states, torch.eye(8)[[0, 0]]) == [AUDIO_PLACEHOLDER, 7]
    assert cover.generated_frames == 1
    assert plan.generated_frames == 0
    for _ in range(2):
        advance(runner, [cover], torch.eye(8)[[0]])
    assert cover.req.finished()
    result = build_result(cover).data
    assert result["abc_tokens"] == 0
    assert result["frames"] == 3
    assert result["usage"]["completion_tokens"] == 3
    assert result["usage"]["prompt_tokens"] == len(cover.req.origin_input_ids)


def test_cover_score_counts_against_context_without_reducing_audio_budget() -> None:
    """User-supplied ABC consumes context even though it consumes no generation budget."""
    prompt_length = len(
        speech_request("Cover", ref_audio=ABC_SCORE).req.origin_input_ids
    )
    cover = speech_request(
        "Cover", context_length=prompt_length + 4, ref_audio=ABC_SCORE
    )
    assert cover.max_new_tokens == 3
    with pytest.raises(ValueError, match="prompt and audio frames exceed"):
        speech_request("Cover", context_length=prompt_length + 3, ref_audio=ABC_SCORE)


def test_cover_score_cannot_insert_modality_markers() -> None:
    """Score text cannot close the ABC block or open an extra audio block."""
    with pytest.raises(ValueError, match="reserved marker") as raised:
        speech_request("Cover", ref_audio=ABC_SCORE + "<|music_audio_start|>")
    assert generation_error_status_code(raised.value) == 400


def test_plan_context_reserves_audio_and_rejects_unsupported_task() -> None:
    """Planning cannot exhaust the positions reserved for the requested audio."""
    prompt_length = len(speech_request("PlanFull").req.origin_input_ids)
    context_length = prompt_length + 8
    state = speech_request("PlanFull", context_length=context_length)
    assert state.max_abc_tokens == 2
    assert len(state.req.origin_input_ids) + state.max_new_tokens == context_length - 1
    with pytest.raises(ValueError, match="no room for ABC"):
        speech_request("PlanFull", context_length=prompt_length + 5)
    with pytest.raises(ValueError, match="Direct.*PlanFull.*PlanMelody"):
        speech_request("Base")


def test_mixed_plan_audio_transitions_preserve_frames_and_budgets(
    runner: HeartMoERunner,
) -> None:
    """ABC markers must use text embeddings and leave every request's audio budget intact."""
    direct, full, melody = [
        speech_request(task) for task in ("Direct", "PlanFull", "PlanMelody")
    ]
    states = [direct, full, melody]
    runner.before_prefill(
        None,
        None,
        [SchedulerRequest(request_id=state.req.rid, data=state) for state in states],
    )
    text_hidden = torch.eye(8)[0]
    end_hidden = torch.eye(8)[1]
    assert advance(
        runner, states, torch.stack([text_hidden, text_hidden, end_hidden])
    ) == [AUDIO_PLACEHOLDER, 7, ABC_END]
    assert [state.generated_frames for state in states] == [1, 0, 0]
    assert full.abc_token_ids == [7]
    assert advance(
        runner,
        [melody, direct, full],
        torch.stack([end_hidden, text_hidden, end_hidden]),
    ) == [AUDIO_START, AUDIO_PLACEHOLDER, ABC_END]
    assert advance(
        runner,
        [full, melody, direct],
        torch.stack([end_hidden, text_hidden, text_hidden]),
    ) == [AUDIO_START, AUDIO_PLACEHOLDER, AUDIO_PLACEHOLDER]
    assert direct.req.finished()
    assert direct.generated_frames == 3
    advance(runner, [full, melody], torch.stack([text_hidden, text_hidden]))
    advance(runner, [melody, full], torch.stack([text_hidden, text_hidden]))
    assert melody.req.finished()
    advance(runner, [full], text_hidden[None])
    assert full.req.finished()
    for state in states:
        assert state.generated_frames == 3
        state.finish_reason = "length"
        result = build_result(state).data
        assert result["frames"] == 3
        assert result["finish_reason"] == "length"
        assert result["usage"]["completion_tokens"] == 3 + len(state.abc_token_ids)
    assert (
        list(full.req.output_ids) == [7, ABC_END, AUDIO_START] + [AUDIO_PLACEHOLDER] * 3
    )
    assert (
        list(melody.req.output_ids) == [ABC_END, AUDIO_START] + [AUDIO_PLACEHOLDER] * 3
    )


@pytest.mark.parametrize("ends_score", [False, True])
def test_plan_transition_preserves_scheduler_limit(
    runner: HeartMoERunner, ends_score: bool
) -> None:
    """Switching to audio must retain a stricter token cap applied at admission."""
    state = speech_request("PlanFull", max_abc_tokens=1)
    state.req.sampling_params.max_new_tokens = 4
    requests = [SchedulerRequest(request_id=state.req.rid, data=state)]
    runner.before_prefill(None, None, requests)
    advance(runner, [state], torch.eye(8)[[int(ends_score)]])
    for _ in range(3):
        advance(runner, [state], torch.eye(8)[[0]])
    assert state.req.finished()
    assert state.req.sampling_params.max_new_tokens == 4
    assert state.generated_frames == (2 if ends_score else 1)


@pytest.mark.parametrize("task_type", ["PlanFull", "PlanMelody"])
@pytest.mark.parametrize(
    ("abc_tokens", "ends_score"), [(4096, False), (4096, True), (2, False)]
)
def test_abc_limit_transitions_to_audio_without_losing_frames(
    runner: HeartMoERunner, task_type: str, abc_tokens: int, ends_score: bool
) -> None:
    """A capped score must retain its last token and emit both boundaries before the full audio budget."""
    prompt_length = len(speech_request(task_type).req.origin_input_ids)
    context_length = 16384 if abc_tokens == 4096 else prompt_length + 8
    plan = speech_request(
        task_type,
        context_length=context_length,
        max_abc_tokens=ARFactoryArgs().max_abc_tokens,
    )
    assert plan.max_abc_tokens == abc_tokens
    direct = speech_request("Direct")
    states = [plan, direct]
    runner.before_prefill(
        None,
        None,
        [SchedulerRequest(request_id=state.req.rid, data=state) for state in states],
    )
    plan.abc_token_ids.extend([7] * (abc_tokens - 1))
    plan.req.output_ids.extend(plan.abc_token_ids)
    assert advance(runner, states, torch.eye(8)[[int(ends_score), 0]]) == [
        ABC_END if ends_score else 7,
        AUDIO_PLACEHOLDER,
    ]
    assert plan.generated_frames == 0
    assert not plan.req.finished()
    if ends_score:
        expected_abc_tokens = abc_tokens - 1
    else:
        expected_abc_tokens = abc_tokens
        assert advance(runner, [plan], torch.eye(8)[[0]]) == [ABC_END]
        assert not plan.req.finished()
    assert plan.abc_token_ids == [7] * expected_abc_tokens
    assert advance(runner, states, torch.eye(8)[[0, 0]]) == [
        AUDIO_START,
        AUDIO_PLACEHOLDER,
    ]
    assert advance(runner, states, torch.eye(8)[[0, 0]]) == [
        AUDIO_PLACEHOLDER,
        AUDIO_PLACEHOLDER,
    ]
    assert direct.req.finished()
    assert build_result(direct).data["frames"] == 3
    for _ in range(2):
        assert advance(runner, [plan], torch.eye(8)[[0]]) == [AUDIO_PLACEHOLDER]
    assert plan.req.finished()
    assert list(plan.req.output_ids) == (
        [7] * expected_abc_tokens + [ABC_END, AUDIO_START] + [AUDIO_PLACEHOLDER] * 3
    )
    result = build_result(plan).data
    assert result["frames"] == 3
    assert result["abc_tokens"] == expected_abc_tokens
    assert result["usage"]["completion_tokens"] == expected_abc_tokens + 3


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is required"
                ),
            ],
        ),
    ],
)
def test_decode_embeddings_follow_modality_with_graph_replay(
    runner: HeartMoERunner, device: Literal["cpu", "cuda"]
) -> None:
    """A reused decode graph must follow changed tokens and feedback after row reordering."""
    model = runner.model.to(device)
    input_ids = torch.tensor(
        [7, ABC_END, AUDIO_START, AUDIO_PLACEHOLDER], device=device
    )
    positions = torch.arange(4, device=device)
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=4,
        input_ids=input_ids,
        req_pool_indices=positions,
        seq_lens=torch.ones(4, dtype=torch.long, device=device),
        out_cache_loc=positions,
        seq_lens_sum=4,
    )
    with torch.inference_mode():
        model.feedback.copy_(torch.randn_like(model.feedback))
        if device == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                model(input_ids, positions, batch)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = model(input_ids, positions, batch)
        else:
            graph = None
        for token_ids in (
            [7, ABC_END, AUDIO_START, AUDIO_PLACEHOLDER],
            [AUDIO_PLACEHOLDER, 7, ABC_END, AUDIO_START],
        ):
            input_ids.copy_(torch.tensor(token_ids, device=device))
            model.feedback.copy_(torch.randn_like(model.feedback))
            expected = torch.stack(
                [
                    (
                        model.feedback[index]
                        if token_id == AUDIO_PLACEHOLDER
                        else model.model.embed_tokens.weight[token_id]
                    )
                    for index, token_id in enumerate(token_ids)
                ]
            )
            eager = model(input_ids, positions, batch)
            torch.testing.assert_close(eager.hidden_states, expected)
            if graph is not None:
                graph.replay()
                torch.testing.assert_close(captured.hidden_states, expected)
                torch.testing.assert_close(
                    captured.next_token_logits, eager.next_token_logits
                )
