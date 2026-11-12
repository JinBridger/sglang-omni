# HeartMoE

HeartMoE uses SGLang's Qwen3.5-MoE backbone, a batched eight-codebook local decoder, and a separate HeartCodec process. The AR stage uses OmniScheduler continuous batching: requests can enter and leave between decode steps, including batches that mix ABC planning and audio generation. Both backbone and local decoder support CUDA graphs. Direct and planning tasks use Chinese system and task prompts; Cover uses a dedicated Chinese system prompt with an English lead-sheet task instruction. Tensor parallelism is not implemented.

## Setup and serving

Copy the complete HeartMoE checkpoint directory and this SGLang-Omni repository to the serving machine, then install the repository using the [installation guide](../get_started/installation.md). HeartCodec's model, residual codebook lookup, flow matching, and waveform decoder live in `sglang_omni/models/heartmoe/` and use the repository's existing dependencies. No separate HeartCodec Python checkout, `vector_quantize_pytorch` package, or Python path setup is needed.

The combined checkpoint uses one standard shard sequence, `model-00001-of-00017.safetensors` through `model-00017-of-00017.safetensors`. Codec tensors use the `heartcodec.` prefix in the shared `model.safetensors.index.json`; the main `config.json` contains `heartcodec_config`. Both stages resolve weights from `--model-path` and load only their own tensors. Copy all files in that directory, including the updated configuration and index.

For an older checkpoint that has not yet been bundled, run this once before copying it:

```bash
python scripts/pack_heartmoe_codec.py \
  --model-path /path/to/heartmoe_hf \
  --codec-path /path/to/HeartCodec
```

The packer combines AR and codec tensors into one state dictionary and writes new safetensors shards with a shared 5 GB size limit. A shard can contain both AR and codec tensors; each stage selects its own weights using their names. Tensor values and dtypes are preserved. A checkpoint that already contains codec weights does not need this step.

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path /path/to/heartmoe_hf --port 8000
```

For the first launch on a new environment, set `SGLANG_OMNI_STARTUP_TIMEOUT=1800` before the command to allow time for FlashInfer kernel compilation and checkpoint loading from shared storage. Subsequent launches reuse the compiled kernels. The AR loader enables SGLang's checkpoint prefetching.

The default places both stages on one GPU and admits up to 16 AR requests. The BF16 model weights occupy about 66 GiB; HeartCodec weights, caches, and temporary buffers need additional memory. This configuration is intended for GPUs with more than 80 GiB. To change admission or move the codec to another visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path /path/to/heartmoe_hf --port 8000 \
  --heartmoe_ar.engine.max_running_requests 8 \
  --heartcodec.gpu 1
```

## Speech requests

`input` contains lyrics, and `instructions` describes the style. For instrumental music, set `task_type` to `Instrumental` and describe the music in `instructions`. The speech API still requires non-empty `input`; use `[Instrumental]` as a placeholder. This task ignores the input lyrics and always sends an empty lyrics block. One generated frame represents 0.08 seconds; `max_new_tokens=375` caps audio at 30 seconds. The model can finish earlier on EOS.

Select the generation task with `task_type`:

| `task_type` | Task | Mode | Generation order |
| --- | --- | --- | --- |
| `Direct` | `direct_audio` | `direct` | Audio only |
| Omitted, `null`, or `PlanFull` | `plan_audio` | `full` | Full ABC score, then audio |
| `PlanMelody` | `plan_audio` | `melody` | Melody-only ABC score, then audio |
| `Cover` | `plan_audio` | `full` | User-supplied full ABC score, then audio |
| `Instrumental` | `plan_audio` | `full` | Full ABC score, then instrumental audio without vocals or lyrics |

The requests below use `Instrumental`. Omit `task_type` to select `PlanFull`, or set it to `Direct` for direct audio generation. The speech endpoint returns audio; the ABC score conditions generation internally. Streaming audio begins after planning completes. ABC tokens do not consume the `max_new_tokens` audio frame budget. For `PlanFull`, `PlanMelody`, and `Instrumental`, the server allows at most 4,096 ABC content tokens, reduced when needed to reserve context for the requested audio. If the score reaches that limit without its end marker, the server preserves those tokens, inserts the ABC end marker followed by the audio start marker, and continues generating audio. A model-generated ABC end marker can start audio earlier. Configure a lower ABC limit and the sampling temperature with `--heartmoe_ar.factory.max_abc_tokens` and `--heartmoe_ar.factory.abc_temperature` (defaults: 4096 and 0.7); ABC sampling uses the request's `top_k`, `top_p`, and `seed`.

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "Instrumental",
    "input": "[Instrumental]\n",
    "instructions": "Jazz piano, upright bass, brushed drums",
    "max_new_tokens": 375,
    "seed": 20261105,
    "response_format": "wav"
  }' --output music.wav
```

For actual incremental delivery, set `stream=true`; merely streaming the HTTP download without this field does not enable incremental generation.

```bash
curl --no-buffer --fail-with-body http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "Instrumental",
    "input": "[Instrumental]\n",
    "instructions": "Jazz piano, upright bass, brushed drums",
    "max_new_tokens": 375,
    "seed": 20261105,
    "stream": true,
    "response_format": "pcm"
  }' --output music.pcm
```

The sample rate is 48 kHz. Non-streaming WAV/FLAC/MP3 retain stereo. HTTP streaming PCM/WAV and SSE preserve HeartCodec's two channels. Streamed PCM is signed 16-bit little-endian, interleaved left/right; read `X-Channels`, `X-Sample-Rate`, and `X-Bit-Depth` from the response instead of assuming mono. WAV headers describe both channels, and SSE duration accounts for both channels. At 48 kHz, stereo PCM uses 192,000 bytes per second. Non-streaming raw PCM retains the existing mono encoding. The SSE stream includes `speech.audio.delta` events and a terminal `speech.audio.done` event.

For `Cover`, put the reference lead-sheet ABC score as raw text in `ref_audio`. Its dedicated prompt asks the model to preserve the melody, section structure, and any chord progression while following `instructions` for genre, mood, instruments, vocal gender and timbre, tempo, and any specified key. Otherwise, it retains the score's key. The supplied score remains between the assistant's ABC markers, and generation starts at the audio boundary without generating or rewriting ABC. Do not supply an audio file, URL, base64 audio, or modality markers in this field. Missing or blank scores are rejected. The score counts as input context, not generated ABC, and is not subject to the ABC generation limit. The score and requested audio must still fit the model's context. All other request requirements and audio sampling settings are the same as `Direct`; the default remains `PlanFull`.

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "Cover",
    "input": "[Instrumental]",
    "instructions": "Solo piano, gentle and clear",
    "ref_audio": "X:1\nT:Example\nM:4/4\nL:1/4\nQ:1/4=100\nK:C\nC D E G | G E D C |]",
    "max_new_tokens": 375,
    "response_format": "wav"
  }' --output cover.wav
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `task_type` | `null` | `Direct`, `PlanFull`, `PlanMelody`, `Cover`, or `Instrumental`; `null` selects `PlanFull` |
| `ref_audio` | `null` | Required raw ABC score for `Cover`; unsupported for other HeartMoE tasks |
| `max_new_tokens` | 375 | Audio frame cap, from 1 to 7,500 |
| `temperature` | CB0: 1.2; CB1–7: 0.8 | An explicit value overrides both audio temperatures; 0 selects greedy decoding |
| `top_k` | 50 | Candidate count for CB1–7 and ABC; 0 disables top-k |
| `top_p` | 0.95 | Nucleus probability for CB1–7 and ABC |
| `seed` | 20261105 | Per-request sampling seed |
| `voice` | `default` | Only the default voice is supported |
| `speed` | 1.0 | Describe tempo in `instructions` |

All three generation paths use the production SGLang sampler. CB0 first keeps raw logits at least `max(logits) + ln(0.05)`, before applying temperature. Rejected logits become `max(logits) - 10000`, keeping sampler inputs finite. CB0 then samples at temperature 1.2 without top-k or top-p; EOS participates in the same filter as ordinary codes. CB1–7 sample at temperature 0.8, top-k 50, and top-p 0.95. ABC uses temperature 0.7, top-k 50, and top-p 0.95 by default.

For independent per-request settings, use `stage_params.heartmoe_ar`. It accepts `first_codebook_temperature`, `first_codebook_min_p`, `abc_temperature`, and the sampling fields in the table. Stage values override top-level values. An explicit `temperature` sets both audio temperatures unless `first_codebook_temperature` is also supplied; `top_k` and `top_p` never filter CB0. `first_codebook_min_p=0` disables the CB0 gate, and `1` retains only maximum-logit codes. For example:

```json
{
  "input": "[Instrumental]",
  "stage_params": {
    "heartmoe_ar": {
      "first_codebook_temperature": 1.2,
      "first_codebook_min_p": 0.05,
      "temperature": 0.8,
      "abc_temperature": 0.7,
      "top_k": 50,
      "top_p": 0.95
    }
  }
}
```

Sampling seeds and positions remain separate for each request and codebook step. Fused backbone numerics can still vary with batch size; an identical seed is not a promise of identical samples across different concurrency levels or the previous sampler.

## Streaming windows and measurement

HeartCodec runs as an independent `StreamingSimpleScheduler` stage. The AR scheduler sends frame blocks through `stream_to`; the codec scheduler decodes ready windows, maintains each request's latent and waveform overlap, and emits audio immediately. It handles stream completion and cancellation through the pipeline lifecycle, including cancellation checks between flow steps. AR requests use continuous batching. Codec windows run serially by default; an optional batch combines ready windows from different requests without waiting for an incomplete batch.

HeartCodec defaults to 93 frames per window with 13 overlap frames. It conditions each window on the preceding latent tail and crossfades the overlapping waveform. The first window provides 6.4 seconds of playable audio. Frames are transported in blocks of eight, so a long request starts its first decode after 96 frames (7.68 seconds of audio content) have been generated. The final partial window is padded for decoding, then trimmed to the exact generated duration. Both streaming and non-streaming responses use the same decoded windows.

Tune the window with `--heartcodec.factory.chunk_frames` and `--heartcodec.factory.overlap_frames`. Shorter windows reduce first-audio latency but change the codec's conditioning context and may affect sound quality. The default window is shorter than the old offline decoder's 372-frame window, so its waveform is not expected to match that decoder sample for sample. Flow matching retains 10 steps, guidance 1.25, and FP32 weights with TF32 enabled.

For higher decoder throughput on a separate GPU, enable window batches, compilation, and SGLang diffusion CUDA graphs:

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path /path/to/heartmoe_hf --port 8000 \
  --heartcodec.gpu 1 \
  --heartcodec.factory.max_batch_size 8 \
  --heartcodec.factory.compile_codec true \
  --heartcodec.factory.use_cuda_graph true
```

`max_batch_size` accepts 1 through 8 and defaults to 1. Each request keeps its own random noise, latent context, overlap, and audio length; cancelling one batch row does not cancel its peers. Partial final windows still flush individually. `compile_codec` and `use_cuda_graph` default to false. Compilation folds inference weight normalization and compiles the transformer blocks and waveform decoder. CUDA graphs replay the transformer inside each flow step, so cancellation can still interrupt between steps. Startup warms the supported batch sizes; cold compilation increases startup time. Larger batches and graph caches need additional GPU memory, so measure memory before colocating the optimized codec with the AR model.

These options preserve FP32 weights, TF32, the window geometry, and the flow step count. Batch-dependent matrix operations and compiler fusion can change floating-point rounding, including the waveform decoder's quantization boundaries; optimized audio is not guaranteed to match eager output sample for sample. Benchmark speed and evaluate audio quality on representative songs before selecting deployment settings.

```bash
python benchmarks/bench_heartmoe.py \
  --url http://localhost:8000 --frames 375 \
  --concurrency 1 2 4 8 16 --repeats 3 \
  --output heartmoe-benchmark.json
```

The benchmark warms each concurrency level, then measures first PCM audio latency, end-to-end latency, and aggregate generated audio seconds per wall-clock second. First-audio latency includes AR generation, HeartCodec, queueing, and HTTP transport; it excludes loading and warmup. The first byte measured is audio, not a WAV header or an SSE control event.

It also reports each request's real-time factor (RTF = elapsed seconds / generated audio seconds), the number exceeding RTF 1, and the additional startup buffer needed for uninterrupted playback after the first audio arrives. An RTF below 1 alone does not guarantee that intermediate audio chunks arrive before playback exhausts its buffer. Raise `--heartmoe_ar.engine.max_running_requests` when measuring above the default admission limit of 16.
