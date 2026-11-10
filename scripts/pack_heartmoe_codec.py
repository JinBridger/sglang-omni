# SPDX-License-Identifier: Apache-2.0
"""Merge HeartMoE and HeartCodec tensors, then shard the combined checkpoint."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load, save_file


def pack_codec(
    model_path: Path, codec_path: Path, *, max_shard_size: int | str
) -> None:
    config = json.loads((model_path / "config.json").read_text())
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    codec_config = json.loads((codec_path / "config.json").read_text())
    codec_index = json.loads((codec_path / "model.safetensors.index.json").read_text())
    original_shards = set(index["weight_map"].values())
    weights = {}
    for filename in sorted(original_shards):
        weights.update(
            {
                name: tensor
                for name, tensor in load((model_path / filename).read_bytes()).items()
                if not name.startswith("heartcodec.")
            }
        )
        print(f"Loaded {filename}", flush=True)
    for filename in sorted(set(codec_index["weight_map"].values())):
        weights.update(
            {
                f"heartcodec.{name}": tensor
                for name, tensor in load((codec_path / filename).read_bytes()).items()
            }
        )
    weights = dict(sorted(weights.items()))
    plan = split_torch_state_dict_into_shards(weights, max_shard_size=max_shard_size)
    print(
        f"Merged {len(weights)} tensors into {len(plan.filename_to_tensors)} shards",
        flush=True,
    )
    config["heartcodec_config"] = codec_config
    with TemporaryDirectory(
        prefix=".heartmoe-pack-", dir=model_path.parent
    ) as directory:
        staging = Path(directory)
        for filename, names in plan.filename_to_tensors.items():
            save_file(
                {name: weights.pop(name) for name in names},
                str(staging / filename),
                metadata={"format": "pt"},
            )
            print(f"Wrote {filename}: {len(names)} tensors", flush=True)
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename},
                indent=2,
            )
            + "\n"
        )
        for filename in (
            *plan.filename_to_tensors,
            "config.json",
            "model.safetensors.index.json",
        ):
            (staging / filename).replace(model_path / filename)
        for filename in original_shards - plan.filename_to_tensors.keys():
            (model_path / filename).unlink()
    print(f"Combined checkpoint saved to {model_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    arguments = parser.parse_args()
    pack_codec(
        arguments.model_path,
        arguments.codec_path,
        max_shard_size=arguments.max_shard_size,
    )


if __name__ == "__main__":
    main()
