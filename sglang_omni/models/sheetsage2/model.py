# SPDX-License-Identifier: Apache-2.0
"""SheetSage2 weights and a BART decoder with independent request positions."""

import importlib
from pathlib import Path
from types import ModuleType

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoModel, BartConfig, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.bart.modeling_bart import BartDecoder

PROJECTIONS = ("query_proj", "key_proj", "value_proj", "out_proj")
ATTENTION_WORKSPACE_BYTES = 128 * 1024**2


class SheetSage2Model(nn.Module):
    def __init__(self, model_path: str, base_model_path: str | None) -> None:
        super().__init__()
        self.config: PretrainedConfig = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer_class = get_class_from_dynamic_module(
            "tokenization_sheetsage2.SheetSage2Tokenizer", model_path
        )
        self.tokenizer = tokenizer_class(
            self.config.input_audio_length,
            self.config.time_hz,
            self.config.tokenizer_schema_version,
            expected_fingerprint=self.config.tokenizer_fingerprint,
        )
        transcriber_class = get_class_from_dynamic_module(
            "pipeline_sheetsage2.Transcriber", model_path
        )
        self.pipeline: ModuleType = importlib.import_module(
            transcriber_class.__module__
        )
        self.generation: ModuleType = importlib.import_module(
            f"{transcriber_class.__module__.rsplit('.', 1)[0]}.generation_sheetsage2"
        )
        self.exports: ModuleType = importlib.import_module(
            f"{transcriber_class.__module__.rsplit('.', 1)[0]}.exports_sheetsage2"
        )
        self.max_output_seq_len: int = self.config.max_output_seq_len
        self.layer_weight: nn.Parameter = nn.Parameter(
            torch.zeros(self.config.backbone_config["num_hidden_layers"] + 1)
        )
        self.encoder_projection: nn.Linear = nn.Linear(
            self.config.backbone_config["hidden_size"], self.config.hidden_size
        )
        decoder_config = BartConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.hidden_size,
            decoder_layers=self.config.decoder_layers,
            decoder_attention_heads=self.config.num_attention_heads,
            decoder_ffn_dim=self.config.intermediate_size,
            max_position_embeddings=self.max_output_seq_len,
            activation_function="gelu",
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            attn_implementation="sdpa",
        )
        self.decoder: BartDecoder = BartDecoder(decoder_config)
        self.token_embedding: nn.Embedding = self.decoder.embed_tokens
        self.output_projection: nn.Linear = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.output_projection.weight = self.token_embedding.weight
        weights = load_file(str(Path(model_path) / "model.safetensors"))
        if self.config.weights_format != "adapter":
            raise ValueError(
                "SheetSage2 requires an adapter checkpoint and its MERT base"
            )
        else:
            parent = base_model_path or self.config.base_model_name_or_path
        self.encoder = AutoModel.from_pretrained(
            parent,
            revision=self.config.base_model_revision,
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        # note (Codex): Transformers 5's meta loader does not restore MERT's nonpersistent rotary buffer.
        self.encoder.embed_positions = type(self.encoder.embed_positions)(
            self.encoder.config
        )
        scale = self.config.lora_alpha / self.config.lora_rank
        with torch.no_grad():
            for i, layer in enumerate(self.encoder.layers):
                for name in PROJECTIONS:
                    prefix = f"adapter.layers.{i}.attn.{name}"
                    update = weights.pop(f"{prefix}.lora_B.weight") @ weights.pop(
                        f"{prefix}.lora_A.weight"
                    )
                    getattr(layer.attn, name).weight.add_(update, alpha=scale)
        weights["decoder.embed_tokens.weight"] = weights["token_embedding.weight"]
        weights["output_projection.weight"] = weights["token_embedding.weight"]
        # note (Codex): The encoder checkpoint was loaded separately before merging its adapters.
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if unexpected or any(not name.startswith("encoder.") for name in missing):
            raise ValueError(
                f"Incompatible SheetSage2 weights: {missing=}, {unexpected=}"
            )
        else:
            pass
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        window_samples = round(
            self.config.input_audio_length * self.config.sampling_rate
        )
        audio = F.pad(audio.float(), (0, window_samples - audio.shape[-1]))
        mel = self.encoder.feature_extractor(audio)
        hidden_states = self.encoder.subsampling_module(mel)
        weights = self.layer_weight.softmax(dim=0)
        mixed_hidden_states = hidden_states * weights[0]
        positions = self.encoder.embed_positions(hidden_states)
        for weight, layer in zip(weights[1:], self.encoder.layers):
            hidden_states = layer(hidden_states, positions)
            mixed_hidden_states = mixed_hidden_states + hidden_states * weight
        return self.encoder_projection(mixed_hidden_states)


class BatchedDecoder:
    def __init__(
        self,
        model: SheetSage2Model,
        max_batch_size: int,
        dtype: torch.dtype,
        enable_cuda_graph: bool,
    ) -> None:
        self.model = model
        self.device: torch.device = next(model.parameters()).device
        self.dtype = dtype
        self.enable_cuda_graph = enable_cuda_graph
        decoder = model.decoder
        self.head_count: int = decoder.config.decoder_attention_heads
        self.head_width: int = decoder.config.d_model // self.head_count
        self.max_tokens: int = model.max_output_seq_len
        frame_count = round(
            model.config.input_audio_length * model.config.backbone_config["frame_rate"]
        )
        self.self_keys: list[torch.Tensor] = []
        self.self_values: list[torch.Tensor] = []
        self.cross_keys: list[torch.Tensor] = []
        self.cross_values: list[torch.Tensor] = []
        for _ in decoder.layers:
            for buffers, sequence_length in (
                (self.self_keys, self.max_tokens),
                (self.self_values, self.max_tokens),
                (self.cross_keys, frame_count),
                (self.cross_values, frame_count),
            ):
                buffers.append(
                    torch.zeros(
                        max_batch_size,
                        self.head_count,
                        sequence_length,
                        self.head_width,
                        device=self.device,
                        dtype=dtype,
                    )
                )
        self.token_ids = torch.zeros(
            max_batch_size, 1, device=self.device, dtype=torch.long
        )
        self.positions = torch.zeros(
            max_batch_size, device=self.device, dtype=torch.long
        )
        self.allowed = torch.ones(
            max_batch_size,
            model.config.vocab_size,
            device=self.device,
            dtype=torch.bool,
        )
        self.graphs: dict[
            tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]
        ] = {}
        self.graph_pool = torch.cuda.graph_pool_handle() if enable_cuda_graph else None
        self.sequence_lengths = torch.full(
            (max_batch_size,), self.max_tokens, device=self.device, dtype=torch.int32
        )
        self.workspace = torch.empty(
            ATTENTION_WORKSPACE_BYTES, device=self.device, dtype=torch.uint8
        )
        self.attention_wrappers: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}

    @torch.inference_mode()
    def prefill(
        self, slot: int, memory: torch.Tensor, token_ids: list[int]
    ) -> torch.Tensor:
        with torch.autocast(
            "cuda", dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            decoded = self.model.decoder(
                input_ids=torch.tensor([token_ids], device=self.device),
                encoder_hidden_states=memory,
                use_cache=True,
                return_dict=True,
            )
            logits = self.model.output_projection(decoded.last_hidden_state[:, -1])
        cache = decoded.past_key_values
        for i in range(len(self.self_keys)):
            self.self_keys[i][slot, :, : len(token_ids)].copy_(
                cache.self_attention_cache.layers[i].keys[0]
            )
            self.self_values[i][slot, :, : len(token_ids)].copy_(
                cache.self_attention_cache.layers[i].values[0]
            )
            self.cross_keys[i][slot].copy_(
                cache.cross_attention_cache.layers[i].keys[0]
            )
            self.cross_values[i][slot].copy_(
                cache.cross_attention_cache.layers[i].values[0]
            )
        return logits[0]

    def forward(self, batch_size: int, token_capacity: int) -> torch.Tensor:
        decoder = self.model.decoder
        positions = self.positions[:batch_size]
        batch_indices = torch.arange(batch_size, device=self.device)
        token_ids = self.token_ids[:batch_size]
        hidden_states = decoder.embed_tokens(token_ids) + F.embedding(
            positions[:, None] + decoder.embed_positions.offset,
            decoder.embed_positions.weight,
        )
        hidden_states = decoder.layernorm_embedding(hidden_states)
        attention_mask = (
            torch.arange(token_capacity, device=self.device)[None, :]
            <= positions[:, None]
        )[:, None, None, :]
        self.sequence_lengths[:batch_size].copy_((positions + 1).int())
        for i, layer in enumerate(decoder.layers):
            residual = hidden_states
            attention = layer.self_attn
            q = (
                attention.q_proj(hidden_states)
                .view(batch_size, 1, self.head_count, self.head_width)
                .transpose(1, 2)
            )
            k = attention.k_proj(hidden_states).view(
                batch_size, self.head_count, self.head_width
            )
            v = attention.v_proj(hidden_states).view(
                batch_size, self.head_count, self.head_width
            )
            self.self_keys[i][batch_indices, :, positions] = k
            self.self_values[i][batch_indices, :, positions] = v
            if self.dtype == torch.float32:
                hidden_states = F.scaled_dot_product_attention(
                    q,
                    self.self_keys[i][:batch_size, :, :token_capacity],
                    self.self_values[i][:batch_size, :, :token_capacity],
                    attn_mask=attention_mask,
                ).reshape(batch_size, 1, -1)
            else:
                hidden_states = (
                    self.attention_wrappers[batch_size]
                    .run(
                        q[:, :, 0],
                        (
                            self.self_keys[i][:batch_size],
                            self.self_values[i][:batch_size],
                        ),
                    )
                    .reshape(batch_size, 1, -1)
                )
            hidden_states = layer.self_attn_layer_norm(
                residual + attention.out_proj(hidden_states)
            )
            residual = hidden_states
            attention = layer.encoder_attn
            q = (
                attention.q_proj(hidden_states)
                .view(batch_size, 1, self.head_count, self.head_width)
                .transpose(1, 2)
            )
            hidden_states = (
                F.scaled_dot_product_attention(
                    q,
                    self.cross_keys[i][:batch_size],
                    self.cross_values[i][:batch_size],
                )
                .transpose(1, 2)
                .reshape(batch_size, 1, -1)
            )
            hidden_states = layer.encoder_attn_layer_norm(
                residual + attention.out_proj(hidden_states)
            )
            residual = hidden_states
            hidden_states = layer.fc2(layer.activation_fn(layer.fc1(hidden_states)))
            hidden_states = layer.final_layer_norm(residual + hidden_states)
        logits = self.model.output_projection(hidden_states[:, 0])
        return logits.masked_fill(~self.allowed[:batch_size], float("-inf")).argmax(
            dim=-1
        )

    @torch.inference_mode()
    def step(
        self, token_ids: list[int], positions: list[int], allowed: torch.Tensor
    ) -> list[int]:
        batch_size = len(token_ids)
        token_capacity = self.max_tokens
        if self.dtype == torch.float32:
            token_capacity = min(
                self.max_tokens, 1 << max(5, max(positions).bit_length())
            )
        elif batch_size not in self.attention_wrappers:
            page_offsets = torch.arange(batch_size + 1, dtype=torch.int32)
            page_indices = torch.arange(batch_size, dtype=torch.int32)
            page_lengths = torch.full((batch_size,), self.max_tokens, dtype=torch.int32)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace,
                kv_layout="HND",
                use_cuda_graph=True,
                paged_kv_indptr_buffer=page_offsets.to(self.device),
                paged_kv_indices_buffer=page_indices.to(self.device),
                paged_kv_last_page_len_buffer=self.sequence_lengths[:batch_size],
            )
            wrapper.plan(
                page_offsets,
                page_indices,
                page_lengths,
                self.head_count,
                self.head_count,
                self.head_width,
                self.max_tokens,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            self.attention_wrappers[batch_size] = wrapper
        else:
            pass
        self.token_ids[:batch_size, 0].copy_(torch.tensor(token_ids))
        self.positions[:batch_size].copy_(torch.tensor(positions))
        self.allowed[:batch_size].copy_(allowed)
        with (
            torch.autocast(
                "cuda", dtype=self.dtype, enabled=self.dtype != torch.float32
            ),
            sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.MATH,
                ],
                set_priority=True,
            ),
        ):
            if not self.enable_cuda_graph:
                next_tokens = self.forward(batch_size, token_capacity)
            else:
                graph_key = (batch_size, token_capacity)
                if graph_key not in self.graphs:
                    stream = torch.cuda.Stream(device=self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(stream):
                        self.forward(batch_size, token_capacity)
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.graph_pool):
                        next_tokens = self.forward(batch_size, token_capacity)
                    self.graphs[graph_key] = (graph, next_tokens)
                else:
                    graph, next_tokens = self.graphs[graph_key]
                graph.replay()
        return next_tokens.tolist()
