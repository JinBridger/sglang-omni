# SheetSage2

SheetSage2 transcribes music into ABC notation and LAB annotations. `POST /v1/music/transcribe` accepts a JSON body containing an audio URL and returns a JSON object keyed by output file name.

## Request

```bash
curl --fail-with-body "${BASE_URL}/v1/music/transcribe" \
  -H 'Content-Type: application/json' \
  -d '{"music_url":"https://example.com/song.wav"}' \
  --output transcription.json
```

Set `BASE_URL` to the running SheetSage2 server and replace the example URL with an accessible audio link. The required `music_url` field accepts HTTP and HTTPS links, including signed URLs; the server downloads the audio into memory before inference. The downloaded bytes are passed to the model and decoded through FFmpeg's standard input, without creating an audio file on disk. Multipart uploads are not accepted. The optional JSON fields are `model`, which must match the served model name, `melody_only` (boolean, default `false`), and `max_new_tokens` (a positive integer). Requests complete without streaming. The model's audio duration limit defaults to 3600 seconds and is configurable with `--transcribe.factory.max_audio_seconds`.

## Response

The response has `Content-Type: application/json`. Parse it once to access each output:

```json
{
  "melody_vocal.lab": "0.0\t0.5\t60\n",
  "melody_instrumental.lab": "",
  "beat.lab": "0.0\t1\t4\t0\n",
  "chord.lab": "0.0\t1.0\tC\n",
  "key.lab": "0.0\t1.0\tC:maj\n",
  "score.abc": "X:1\nK:C\nC D E F|\n"
}
```

`score.abc` is always present and is `null` when ABC construction fails; the LAB outputs are still returned. `X-Request-Id` identifies the request, and `X-Finish-Reason: length` indicates that generation reached the token limit. Missing or invalid URLs return 422. Failed downloads, empty files, and invalid audio return 400; unknown model names return 404, and a full model queue returns 503. Non-JSON requests, including multipart uploads, return 415.

SheetSage2 servers expose this endpoint in place of `/v1/audio/transcriptions`. Clients migrating from that endpoint should send a JSON body with `music_url` instead of uploading a file, and read the returned JSON object directly instead of parsing a JSON string in `text`. Speech recognition models continue to use `/v1/audio/transcriptions`.
