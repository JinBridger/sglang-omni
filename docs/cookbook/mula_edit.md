# Mula Edit

Mula Edit regenerates a selected interval of an existing recording using vocal melody, instrumental melody, chords, and lyrics. `POST /v1/music/edit` accepts JSON and returns a 48 kHz stereo WAV file. The implementation loads the v5 checkpoint with the `AceStepV4ConditionGenerationModel` architecture.

## Launch

The model directory must contain `config.json`, `model.safetensors`, and `silence_latent.pt`. The separate checkpoint directory must contain `vae/` and `Qwen3-Embedding-0.6B/`.

```bash
sgl-omni serve /path/to/v5/checkpoint \
  --model-name mula-edit \
  --edit.factory.checkpoint_dir /path/to/checkpoints \
  --edit.factory.max_queued_requests 32 \
  --edit.gpu 5 \
  --port 8000
```

Use the same launch environment as other Omni CUDA models. If a system CUDA installation overrides the virtual environment's cuDNN libraries, launch with `env -u LD_LIBRARY_PATH sgl-omni serve ...`.

## Request

All timestamps are seconds from the beginning of the original recording. Supply the desired score directly; this endpoint does not transcribe the source audio or parse ABC notation.

```python
import base64
from pathlib import Path

import httpx

request = {
    "model": "mula-edit",
    "audio_base64": base64.b64encode(Path("input.wav").read_bytes()).decode(),
    "start_seconds": 10.0,
    "end_seconds": 12.0,
    "symbols": {
        "vocal_melody": [[10.0, 10.4, 60], [10.4, 10.8, 62]],
        "instrumental_melody": [[10.0, 11.2, 67]],
        "chord_progression": [[10.0, 11.2, "C:maj"], [11.2, 12.0, "G:7"]],
    },
    "symbolic_scope": "span",
    "lyrics": "[Verse]\nThe replacement lyrics",
    "caption": "Piano and vocals",
    "bpm": 120,
    "keyscale": "C major",
    "timesignature": "4",
    "steps": 32,
    "guidance_scale": 7.0,
    "seed": 42,
}
response = httpx.post("http://localhost:8000/v1/music/edit", json=request, timeout=300)
response.raise_for_status()
Path("edited.wav").write_bytes(response.content)
```

The three `symbols` fields are required. Melody events are `[start_seconds, end_seconds, midi_pitch]`, with integer MIDI pitches from 0 through 127. Chord events are `[start_seconds, end_seconds, harte_label]`; examples include `C:maj`, `A:min`, `G:7`, `C:maj/3`, and `N` for no chord. Supported qualities are `maj`, `min`, `dim`, `aug`, `sus2`, `sus4`, `7`, `maj7`, `min7`, `hdim7`, `dim7`, `sus4(b7)`, `maj6`, `min6`, `minmaj7`, and `other`.

Chord parsing follows the training behavior. Unknown qualities, such as `C:maj9` and `E:9`, map to `other` while preserving the root and bass. Labels with unsupported bass degrees, such as `C:maj/8`, or unparseable syntax, such as `H:maj` and `C/E`, are treated as no chord and skipped, as are `N` and `X`.

An empty slot `[]` explicitly supplies no notes. A null slot leaves that condition unknown. With `symbolic_scope="whole"` (the default), the supplied score describes the entire recording. Use `"span"` when only the selected interval's score is available; symbols outside it are hidden. Lyrics are supplied for the whole recording and use the model's 512-token encoding limit; captions and musical metadata share its 256-token prompt limit.

`audio_base64` contains a complete encoded audio file without a data URL prefix. WAV and FLAC are supported by the audio decoder. Mono input is duplicated to stereo, additional channels are discarded, and audio is resampled to 48 kHz. The encoded file limit is 128 MiB. The duration limit defaults to 360 seconds (inclusive) and can be configured with `--edit.factory.max_audio_seconds`, up to 600 seconds. Oversized recordings and edit intervals extending beyond the recording are rejected instead of truncated.

Sampling controls are `steps` (default 32), `guidance_scale` (7), `shift` (3), `seed` (42), and `crossfade_seconds` (1). The seed controls both VAE posterior sampling and diffusion noise. Successful responses have `Content-Type: audio/wav` and an `X-Request-Id` header. Invalid JSON fields return 422, invalid audio or edit intervals extending beyond the recording return 400, and a full queue returns 503. Streaming is not supported by this endpoint.

## Serial execution and acceleration

The scheduler runs one edit at a time in arrival order. Audio preparation, VAE encoding, text conditioning, all diffusion steps, and VAE decoding finish before the next request starts. Additional requests wait in a bounded queue, which defaults to 32 requests; a full queue returns 503. Set `--edit.factory.max_queued_requests 0` to accept only the active request. The former `max_batch_size` option has been removed. Cancellation is checked between diffusion steps and around audio preparation and decoding.

DiT inference enables `torch.compile` and CUDA Graph by default. Compilation uses dynamic shapes and disables Inductor's automatic CUDA Graph capture so the sampler owns graph capture and replay. Each request prepares its symbolic conditions and cross-attention keys and values once. Each diffusion step updates fixed noise and timestep buffers and replays the decoder graph; Euler/APG updates run outside the graph.

The sampler retains one graph for the current audio frame count, guidance branch count, and condition shape. Consecutive requests with the same shape reuse it after replacing all request conditions. A different shape replaces the previous graph. Compilation and capture happen lazily, so the first request and a request requiring a new graph have additional preparation latency. VAE and text encoding remain eager; VAE windows bound their memory use.

With the default `--edit.factory.attention_backend auto`, the BF16 DiT uses mask-free SDPA with native grouped-query attention for full self-attention and cross-attention. On SM12x CUDA devices with FlashAttention-4 available and 128-dimensional heads, local self-attention uses a native symmetric window. Other devices, missing FlashAttention-4, and other head dimensions use the dense SDPA window; float32 keeps the original attention path. Condition encoders retain their padding masks, and the DiT continues to attend to all condition tokens, including padding, as required by the checkpoint. No quantization is applied.

Set `--edit.factory.attention_backend sdpa` to use the original dense-mask attention for comparison or rollback. Attention backends can differ in floating-point rounding; compare musical quality inside the edit interval and around its boundaries when changing backends.

The two optimizations can be disabled independently. To run eager inference, add:

```bash
  --edit.factory.enable_torch_compile false \
  --edit.factory.enable_cuda_graph false
```

The DiT processes the entire recording on every diffusion step, even when the edited interval is short. A 120-second recording contains 3,000 latent frames, reduced to 1,500 transformer tokens. Guidance above 1 evaluates two decoder rows for the single request. Longer recordings and more diffusion steps increase latency; queue capacity only controls how many requests can wait.

The output keeps the source duration after resampling. Outside the selected interval, source latents are restored before decoding. Those samples still pass through the VAE and are not guaranteed to match the original waveform exactly. The result is a single mixed recording, not separate stems.
