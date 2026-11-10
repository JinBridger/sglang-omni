# SPDX-License-Identifier: Apache-2.0
"""Measure speech streaming latency, throughput, and playback buffering."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np


async def generate(
    client: httpx.AsyncClient, endpoint: str, frames: int, seed: int
) -> dict[str, float]:
    started = time.perf_counter()
    first_audio_seconds = None
    audio_bytes = 0
    startup_buffer_seconds = 0.0
    async with client.stream(
        "POST",
        endpoint,
        json={
            "input": "[Instrumental]\n",
            "instructions": "Jazz piano, upright bass, brushed drums",
            "max_new_tokens": frames,
            "seed": seed,
            "stream": True,
            "response_format": "pcm",
        },
    ) as response:
        response.raise_for_status()
        bytes_per_second = (
            int(response.headers["x-sample-rate"])
            * int(response.headers["x-channels"])
            * int(response.headers["x-bit-depth"])
            // 8
        )
        async for chunk in response.aiter_bytes():
            if chunk:
                elapsed_seconds = time.perf_counter() - started
                if first_audio_seconds is None:
                    first_audio_seconds = elapsed_seconds
                else:
                    startup_buffer_seconds = max(
                        startup_buffer_seconds,
                        elapsed_seconds
                        - first_audio_seconds
                        - audio_bytes / bytes_per_second,
                    )
                audio_bytes += len(chunk)
            else:
                pass
    elapsed_seconds = time.perf_counter() - started
    if first_audio_seconds is None or audio_bytes == 0:
        raise RuntimeError("Server returned no PCM audio")
    else:
        pass
    audio_seconds = audio_bytes / bytes_per_second
    return {
        "first_audio_seconds": first_audio_seconds,
        "elapsed_seconds": elapsed_seconds,
        "audio_seconds": audio_seconds,
        "rtf": elapsed_seconds / audio_seconds,
        "startup_buffer_seconds": startup_buffer_seconds,
    }


async def benchmark(arguments: argparse.Namespace) -> None:
    endpoint = arguments.url.rstrip("/") + "/v1/audio/speech"
    results = []
    async with httpx.AsyncClient(
        timeout=600, limits=httpx.Limits(max_connections=64)
    ) as client:
        for concurrency in arguments.concurrency:
            await asyncio.gather(
                *(
                    generate(client, endpoint, arguments.frames, i)
                    for i in range(concurrency)
                )
            )
            started = time.perf_counter()
            measurements = []
            for repeat in range(arguments.repeats):
                measurements.extend(
                    await asyncio.gather(
                        *(
                            generate(
                                client,
                                endpoint,
                                arguments.frames,
                                20261105 + i + repeat * concurrency,
                            )
                            for i in range(concurrency)
                        )
                    )
                )
            wall_seconds = time.perf_counter() - started
            row = {
                "concurrency": concurrency,
                "requests": len(measurements),
                "ttfa_mean_seconds": float(
                    np.mean([result["first_audio_seconds"] for result in measurements])
                ),
                "ttfa_p95_seconds": float(
                    np.percentile(
                        [result["first_audio_seconds"] for result in measurements], 95
                    )
                ),
                "latency_mean_seconds": float(
                    np.mean([result["elapsed_seconds"] for result in measurements])
                ),
                "audio_seconds_per_second": sum(
                    result["audio_seconds"] for result in measurements
                )
                / wall_seconds,
                "rtf_mean": float(np.mean([result["rtf"] for result in measurements])),
                "requests_above_rtf_one": sum(
                    result["rtf"] > 1 for result in measurements
                ),
                "startup_buffer_max_seconds": max(
                    result["startup_buffer_seconds"] for result in measurements
                ),
            }
            print(json.dumps(row), flush=True)
            results.append(dict(**row, measurements=measurements))
            arguments.output.write_text(
                json.dumps({"frames": arguments.frames, "results": results}, indent=2)
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--frames", type=int, default=375)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(benchmark(parser.parse_args()))


if __name__ == "__main__":
    main()
