"""Evaluate OpenVLA-7B on the BridgeData V2 subset used by this project.

The test split is the official RLDS validation trajectories listed in
artifacts/openvla_bridge_split/test.csv. Metrics compare the 7-D action
predicted from image_0 and the language instruction with the logged Bridge
action. A few episodes are also saved as MP4 so the prediction can be watched
on the original camera frames.

Run this inside the openvla conda environment. TensorFlow is kept on CPU so
the GPU stays available for the policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

import numpy as np
import tensorflow_datasets as tfds
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForVision2Seq, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
UNNORM_KEY = "bridge_orig"
ACTION_LABELS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "bridge_course_subset",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "openvla-7b",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=REPO_ROOT / "artifacts" / "openvla_bridge_split" / "test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "openvla_bridge_test",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Evaluate only the first N selected episodes. 0 means the full split.",
    )
    parser.add_argument("--videos-per-category", type=int, default=3)
    parser.add_argument("--fps", type=int, default=5)
    return parser.parse_args()


def load_selected(path: Path, limit: int) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["episode_index"]))
    if limit > 0:
        rows = rows[:limit]
    return {int(row["episode_index"]): row for row in rows}


def to_text(value) -> str:
    item = np.asarray(value).reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def load_policy(model_dir: Path, device: str):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return processor, model


def predict_action(processor, model, image: np.ndarray, instruction: str, device: str) -> np.ndarray:
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(device, dtype=torch.bfloat16)
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    return np.asarray(action, dtype=np.float64)


def step_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - ground_truth)
    return {
        "mae": float(error.mean()),
        "mae_xyz": float(error[:3].mean()),
        "mae_rotation": float(error[3:6].mean()),
        "mae_gripper": float(error[6]),
        "gripper_correct": float((prediction[6] >= 0.5) == (ground_truth[6] >= 0.5)),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("mae", "mae_xyz", "mae_rotation", "mae_gripper", "gripper_correct")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    text_font = font(13)
    line_height = 16
    bar_height = 8 + line_height * len(lines)
    height = image.height + bar_height
    height += (16 - height % 16) % 16
    canvas = Image.new("RGB", (image.width, height), (16, 16, 16))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((6, image.height + 4 + index * line_height), line, fill=(235, 235, 235), font=text_font)
    return np.asarray(canvas)


def format_action(action: np.ndarray) -> str:
    return " ".join(f"{label}={value:+.3f}" for label, value in zip(ACTION_LABELS, action))


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


def evaluate(args: argparse.Namespace) -> None:
    selected = load_selected(args.split_csv, args.max_episodes)
    if not selected:
        raise SystemExit(f"no episodes found in {args.split_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    video_counts: dict[str, int] = {}

    print(f"loading {args.model}", flush=True)
    processor, model = load_policy(args.model, args.device)
    builder = tfds.builder("bridge_dataset", data_dir=str(args.data_dir))
    dataset = builder.as_dataset(split="val", shuffle_files=False)

    episode_rows = []
    all_steps: list[dict[str, float]] = []
    by_category: dict[str, list[dict[str, float]]] = {}
    matched = 0

    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        record = selected.get(episode_index)
        if record is None:
            continue
        steps = list(episode["steps"])
        if not steps:
            continue
        matched += 1
        instruction = to_text(steps[0]["language_instruction"])
        category = record["category"]
        save_video = video_counts.get(category, 0) < args.videos_per_category
        frames: list[np.ndarray] = []
        episode_steps: list[dict[str, float]] = []

        for step_index, step in enumerate(steps):
            image = np.asarray(step["observation"]["image_0"])
            ground_truth = np.asarray(step["action"], dtype=np.float64)
            prediction = predict_action(processor, model, image, instruction, args.device)
            metrics = step_metrics(ground_truth, prediction)
            episode_steps.append(metrics)
            if save_video:
                frames.append(
                    annotate(
                        image,
                        [
                            instruction[:110],
                            f"{category}  step {step_index + 1}/{len(steps)}  L1={metrics['mae']:.4f}",
                            "gt   " + format_action(ground_truth),
                            "pred " + format_action(prediction),
                        ],
                    )
                )

        summary = mean_metrics(episode_steps)
        video_name = ""
        if save_video:
            video_name = f"val_{episode_index:04d}_{category}.mp4"
            write_video(video_dir / video_name, frames, args.fps)
            video_counts[category] = video_counts.get(category, 0) + 1
        episode_rows.append({
            "episode_index": episode_index,
            "category": category,
            "instruction": instruction,
            "steps": len(steps),
            "video": video_name,
            **summary,
        })
        all_steps.extend(episode_steps)
        by_category.setdefault(category, []).extend(episode_steps)
        print(
            f"[{matched}/{len(selected)}] val/{episode_index} {category} "
            f"steps={len(steps)} mae={summary['mae']:.4f} "
            f"xyz={summary['mae_xyz']:.4f} grip_acc={summary['gripper_correct']:.2f}",
            flush=True,
        )
        if matched == len(selected):
            break

    if matched != len(selected):
        raise SystemExit(f"matched {matched} episodes, expected {len(selected)}")

    result = {
        "model": str(args.model),
        "data_dir": str(args.data_dir),
        "split_csv": str(args.split_csv),
        "unnorm_key": UNNORM_KEY,
        "image_key": "observation/image_0",
        "episodes": matched,
        "steps": len(all_steps),
        "overall": mean_metrics(all_steps),
        "by_category": {name: mean_metrics(rows) for name, rows in sorted(by_category.items())},
        "notes": [
            "mae_xyz and mae_rotation compare OpenVLA's unnormalized bridge_orig action with the raw Bridge action.",
            "The gripper dimension is not rescaled by bridge_orig q01/q99. gripper_correct treats both sides as open when the value is at least 0.5.",
            "Videos replay the dataset's real camera frames. They show the logged action beside the model's prediction; they are not closed-loop rollouts.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)
    print(json.dumps(result["overall"], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
