# BS-RoFormer

BS-RoFormer separates a recording into vocals and instrumental accompaniment. `POST /v1/music/separate` accepts an uploaded audio file and returns a ZIP containing `vocals.wav` and `instrumental.wav` by default, or `vocals.flac` and `instrumental.flac` when the optional `format` form field is `flac`.

## Launch

Install the model dependency in your Omni environment:

```bash
pip install -e '.[bs-roformer]'
```

The local model directory must contain `model_bs_roformer_ep_317_sdr_12.9755.ckpt` and its matching `.yaml` file. Omni recognizes this layout directly and loads both files locally.

```bash
OMP_NUM_THREADS=4 sgl-omni serve /path/to/BS-RoFormer \
  --model-name bs-roformer \
  --separate.gpu 6 \
  --separate.max_queued_requests 32 \
  --port 8000
```

The stage requires CUDA. The default precision is bfloat16; `--separate.factory.dtype float32` disables mixed precision. Linear weights remain in the selected precision, and attention uses PyTorch SDPA. The STFT, inverse STFT, normalization weights, rotary frequencies, and overlap accumulation retain float32 precision. Output files use float32 WAV by default or 16-bit FLAC when requested.

Compilation is enabled by default for the frequency-band encoder, Transformer blocks, and mask prediction heads. These regions use fixed window shapes so PyTorch can fuse normalization, rotary embeddings, activations, and other tensor operations. Compilation and a warmup forward finish before the worker reports ready. Use `--separate.factory.enable_torch_compile false` to run eagerly.

## Request

```bash
curl --fail-with-body http://localhost:8000/v1/music/separate \
  -F 'file=@song.mp3' \
  -F 'model=bs-roformer' \
  --output separated.zip
unzip separated.zip
```

The `model` form field is optional. WAV, FLAC, MP3, and OGG inputs supported by libsndfile are accepted. Mono input is duplicated to stereo, and input is resampled to the checkpoint's sample rate, 44.1 kHz for this model. The two WAV files use 32-bit floating-point samples, retain the resampled recording length, and sum back to the input without independent normalization or clipping.

Add `-F 'format=flac'` to return `vocals.flac` and `instrumental.flac` encoded as 16-bit PCM FLAC for smaller downloads. Omitting `format` or supplying any other value preserves the default 32-bit floating-point WAV output. Instrumental audio is still computed as input minus vocals before encoding; FLAC export quantizes each stem to 16-bit PCM.

Successful responses use `Content-Type: application/zip`, `Content-Disposition: attachment; filename="separated.zip"`, and an `X-Request-Id` header. Missing uploads return 422; empty or invalid audio returns 400; unknown model names return 404. Uploads over 128 MiB return 413. The duration limit defaults to 600 seconds and is configurable with `--separate.factory.max_audio_seconds`.

## Request execution

Each GPU worker completes one request before starting the next, using Omni's shared serial scheduler. It decodes the recording, processes its overlapping eight-second windows in order, combines the predictions, and exports both stems. Window size and overlap come from the checkpoint configuration; the supplied model uses a two-second stride. Requests are not interleaved, and windows from different requests are not batched together.

`--separate.max_queued_requests` defaults to 32 and bounds waiting requests in addition to the one running request; excess requests receive 503. Disconnected requests are cancelled between windows, and cancelled queued requests are skipped. Complete recordings and overlap buffers remain in host memory; only the current window is placed on the GPU. A steady queue can keep the worker busy, while increasing queue depth mainly increases waiting time.

## Throughput

Measured on one NVIDIA RTX PRO 6000 Blackwell Server Edition with bfloat16, PyTorch 2.14.1, and `OMP_NUM_THREADS=4`. Each case sends 32 requests containing the same 30-second WAV through local HTTP after two warmup requests. Throughput includes upload, decoding, inference, overlap accumulation, WAV export, and response transfer; model loading and compilation are excluded.

| Execution | Client concurrency | Audio seconds per wall second | GPU memory peak (GiB) |
| --- | --- | --- | --- |
| Eager | 4 | 19.90 | 2.72 |
| Compiled | 1 | 44.69 | 2.40 |
| Compiled | 4 | 46.00 | 2.40 |
| Compiled | 16 | 45.90 | 2.40 |

Compilation improves sustained throughput by 2.31 times at concurrency 4. Increasing concurrency from 1 to 4 keeps the worker supplied with requests and adds about 3%; increasing it to 16 gives no further gain. These results apply to this checkpoint, hardware, and workload. Startup took 12 seconds eagerly and 31 seconds with an existing compilation cache; a first compilation can take longer.
