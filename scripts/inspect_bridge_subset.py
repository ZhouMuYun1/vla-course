"""Create an instruction inventory and report sample from a Bridge RLDS subset."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from scripts.bridge_filters import classify_instruction
except ModuleNotFoundError:
    from bridge_filters import classify_instruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_report"))
    return parser.parse_args()


def to_text(value) -> str:
    item = np.asarray(value).reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = tfds.builder("bridge_dataset", data_dir=str(args.data_dir))
    rows = []
    counts = Counter()
    sample_written = False

    for split in ("train", "val"):
        dataset = builder.as_dataset(split=split, shuffle_files=False)
        for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
            steps = list(episode["steps"])
            if not steps:
                continue
            instruction = to_text(steps[0]["language_instruction"])
            category = classify_instruction(instruction)
            counts[(split, category)] += 1
            rows.append({
                "split": split,
                "episode_index": episode_index,
                "category": category,
                "instruction": instruction,
                "steps": len(steps),
            })
            if not sample_written and category != "other":
                first = steps[0]
                image = np.asarray(first["observation"]["image_0"])
                tf.io.write_file(
                    str(args.output_dir / "sample_image.png"), tf.io.encode_png(image)
                )
                (args.output_dir / "sample.json").write_text(
                    json.dumps({
                        "split": split,
                        "episode_index": episode_index,
                        "instruction": instruction,
                        "raw_action": np.asarray(first["action"]).tolist(),
                        "trajectory_steps": len(steps),
                        "image_shape": list(image.shape),
                    }, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                sample_written = True

    with (args.output_dir / "data_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "total_trajectories": len(rows),
        "total_steps": sum(row["steps"] for row in rows),
        "counts": {
            f"{split}/{category}": value
            for (split, category), value in sorted(counts.items())
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
