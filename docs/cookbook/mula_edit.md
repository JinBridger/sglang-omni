# Mula Edit

Mula Edit regenerates a selected interval of an existing recording using vocal melody, instrumental melody, chords, and lyrics. `POST /v1/music/edit` accepts JSON and returns a 48 kHz stereo WAV file. The implementation loads the v5 checkpoint with the `AceStepV4ConditionGenerationModel` architecture.

## Launch

The model directory must contain `config.json`, `model.safetensors`, and `silence_latent.pt`. The separate checkpoint directory must contain `vae/` and `Qwen3-Embedding-0.6B/`.

```bash
sgl-omni serve /path/to/v5/checkpoint \
  --model-name mula-edit \
  --edit.factory.checkpoint_dir /path/to/checkpoints \
  --edit.factory.max_batch_size 4 \
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

`audio_base64` contains a complete encoded audio file without a data URL prefix. WAV and FLAC are supported by the audio decoder. Mono input is duplicated to stereo, additional channels are discarded, and audio is resampled to 48 kHz. The encoded file limit is 128 MiB. The duration limit defaults to 240 seconds and can be configured with `--edit.factory.max_audio_seconds`, up to 600 seconds. Oversized recordings and edit intervals extending beyond the recording are rejected instead of truncated.

Sampling controls are `steps` (default 32), `guidance_scale` (7), `shift` (3), `seed` (42), and `crossfade_seconds` (1). The seed controls both VAE posterior sampling and diffusion noise. Batch composition can still cause floating-point differences. Successful responses have `Content-Type: audio/wav` and an `X-Request-Id` header. Invalid JSON fields return 422, invalid audio or edit intervals extending beyond the recording return 400, and a full queue returns 503. Streaming is not supported by this endpoint.

## Concurrent requests

The scheduler admits requests between diffusion steps and combines active requests into one DiT forward pass. Each request retains its own diffusion timestep, source latents, symbolic conditions, guidance, and cross-attention cache. Different recording lengths are padded with attention masks so padding cannot affect another recording. Requests leave the batch independently when they finish or are cancelled.

`max_batch_size` counts requests; guidance can require two decoder rows per request. `max_queued_requests` bounds the waiting queue. CPU audio preparation overlaps GPU work. VAE encoding, text conditioning, and VAE decoding run serially in bounded chunks on the same GPU, so these phases can pause diffusion steps for other requests. This initial implementation batches DiT steps and does not batch VAE calls.

The defaults are 4 active requests and 32 queued requests. Four is not a fixed concurrency limit. To allow 16 active requests and another 64 waiting requests, add these launch options:

```bash
  --edit.factory.max_batch_size 16 \
  --edit.factory.max_queued_requests 64
```

Increasing the active batch size shares GPU time among more requests; it does not guarantee higher throughput. For whole recordings lasting 1–3 minutes, start with 4 active requests when throughput and latency matter, and use 16 when more requests should advance together. Queue capacity absorbs bursts without increasing the DiT batch size. Longer queues require more host memory for the encoded audio and longer client timeouts.

The DiT processes the entire recording on every diffusion step, even when the edited interval is short. A 120-second recording contains 3,000 latent frames, reduced to 1,500 transformer tokens; classifier-free guidance doubles the decoder rows. Matrix operations, attention, and intermediate tensor reads and writes dominate GPU execution. VAE encoding and decoding add serial work, and mixing recording lengths adds padding up to the longest recording in the batch. Tune concurrency for the longest expected recordings and the chosen guidance and step count.

On an RTX PRO 6000 Blackwell with 96 GB of memory, a warm resident runtime measured the following for synthetic 120-second stereo recordings, 32 diffusion steps, guidance 7, bfloat16, and PyTorch 2.14.1. Each batch was prepared before sampling, and GPU work was synchronized for timing. These are complete batch runtimes including serial VAE encoding, conditioning, and decoding; they exclude CPU input preparation, HTTP transfer, queueing, and WAV serialization. They are not HTTP latency measurements.

| Active requests | Complete batch | DiT time | Requests/minute | Peak allocated GPU memory |
| --- | --- | --- | --- | --- |
| 4 | 10.9 s | 7.6 s | 22.0 | 8.4 GiB |
| 16 | 48.1 s | 34.9 s | 20.0 | 13.0 GiB |

PyTorch reserved 10.5 GiB and 16.4 GiB respectively; CUDA context and library allocations require additional memory. The measurements show spare memory capacity at 16 requests, while DiT execution accounts for roughly 70–73% of runtime. They do not establish the maximum safe concurrency for other recording lengths, dtypes, or GPUs.

A separate local HTTP run submitted 16 concurrent 180-second synthetic recordings with the same sampling settings and an active request limit of 16. All returned valid 48 kHz stereo WAV files with the expected duration. Response times ranged from 54.0 to 76.2 seconds, with a median of 68.2 seconds. This burst test includes request admission, audio processing, and local HTTP transfer; production latency also depends on arrival patterns and audio file sizes.

The output keeps the source duration after resampling. Outside the selected interval, source latents are restored before decoding. Those samples still pass through the VAE and are not guaranteed to match the original waveform exactly. The result is a single mixed recording, not separate stems.
