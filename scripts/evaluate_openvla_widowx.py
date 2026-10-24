"""Roll out a remote OpenVLA server in SimplerEnv WidowX and save videos.

Run this in the simpler_env environment while scripts/serve_openvla.py is
already serving actions. The camera frame is resized to 256x256 with the same
Lanczos filter SimplerEnv uses for Octo, because Bridge training images are
256x256 and a raw 640x480 frame would be stretched. Rotation is converted from
Bridge Euler deltas to the axis-angle command used by the WidowX controller,
and the gripper is binarized to -1/1.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("VK_ICD_FILENAMES", str(Path(__file__).resolve().parents[1] / "config" / "nvidia_icd.json"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mediapy as media
import numpy as np
import requests
import tensorflow as tf
from transforms3d.euler import euler2axangle

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("widowx_spoon_on_towel", "widowx_put_eggplant_in_basket")
MODEL_SEED = "greedy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "openvla_widowx")
    parser.add_argument("--model-label", default="unknown")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=60)
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=REPO_ROOT / "config" / "eval_states_b_formal.json",
    )
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def load_states(path: Path, tasks: list[str], episodes: int) -> dict[str, list[int]]:
    payload = json.loads(path.read_text())
    selected = {}
    for task in tasks:
        rows = payload["tasks"][task][:episodes]
        if len(rows) < episodes:
            raise ValueError(f"{path} has {len(rows)} states for {task}, need {episodes}")
        selected[task] = [int(row["env_seed"]) for row in rows]
    return selected


def resize_for_openvla(image: np.ndarray, size: int) -> np.ndarray:
    resized = tf.image.resize(image, size=(size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8).numpy()


def bridge_action_to_widowx(action: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(action[3:6], dtype=np.float64)
    axis, angle = euler2axangle(float(roll), float(pitch), float(yaw))
    rotation = np.asarray(axis, dtype=np.float64) * float(angle)
    gripper = np.array([1.0 if action[6] >= 0.5 else -1.0], dtype=np.float64)
    return np.concatenate([np.asarray(action[:3], dtype=np.float64), rotation, gripper])


def query_action(server: str, image: np.ndarray, instruction: str) -> np.ndarray:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    response = requests.post(
        server.rstrip("/") + "/",
        json={
            "image": base64.b64encode(image.tobytes()).decode("ascii"),
            "shape": list(image.shape),
            "instruction": instruction,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return np.asarray(payload["action"], dtype=np.float64)


def run_episode(
    env,
    server: str,
    task: str,
    episode_id: int,
    env_seed: int,
    output_dir: Path,
    image_size: int,
    model_label: str,
) -> dict:
    started = time.time()
    obs, reset_info = env.reset(seed=env_seed)
    instruction = env.get_language_instruction()
    image = get_image_from_maniskill2_obs_dict(env, obs)
    frames = [image]
    raw_actions = []
    env_actions = []
    success = False
    truncated = False
    info = {}
    error = ""
    try:
        for _step in range(env.spec.max_episode_steps if hasattr(env, "spec") else 10**9):
            model_image = resize_for_openvla(image, image_size)
            raw_action = query_action(server, model_image, instruction)
            env_action = bridge_action_to_widowx(raw_action)
            raw_actions.append(raw_action)
            env_actions.append(env_action)
            obs, _reward, success, truncated, info = env.step(env_action)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            frames.append(image)
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction
            if success or truncated:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        success = False
    success = bool(success)
    video_name = f"{task}_seed_{env_seed}_success_{success}.mp4"
    video_path = output_dir / video_name
    if frames:
        media.write_video(str(video_path), frames, fps=5)
    raw = np.asarray(raw_actions, dtype=np.float64) if raw_actions else np.zeros((0, 7))
    executed = np.asarray(env_actions, dtype=np.float64) if env_actions else np.zeros((0, 7))
    return {
        "model": model_label,
        "task": task,
        "episode": episode_id,
        "env_seed": env_seed,
        "model_seed": MODEL_SEED,
        "instruction": instruction,
        "success": success,
        "truncated": bool(truncated),
        "control_steps": int(len(executed)),
        "elapsed_seconds": round(time.time() - started, 3),
        "exception": error,
        "mean_abs_xyz": float(np.mean(np.abs(raw[:, :3]))) if len(raw) else None,
        "gripper_close_fraction": float(np.mean(executed[:, 6] < 0)) if len(executed) else None,
        "reset_info": {key: str(value) for key, value in dict(reset_info).items()},
        "episode_stats": info.get("episode_stats", {}),
        "video": str(video_path),
    }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_episodes_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "model",
        "task",
        "episode",
        "env_seed",
        "model_seed",
        "success",
        "control_steps",
        "truncated",
        "exception",
        "is_src_obj_grasped",
        "moved_correct_obj",
        "moved_wrong_obj",
        "src_on_target",
        "consecutive_grasp",
        "mean_abs_xyz",
        "gripper_close_fraction",
        "video",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            stats = record.get("episode_stats") or {}
            writer.writerow(
                {
                    "model": record["model"],
                    "task": record["task"],
                    "episode": record["episode"],
                    "env_seed": record["env_seed"],
                    "model_seed": record["model_seed"],
                    "success": record["success"],
                    "control_steps": record["control_steps"],
                    "truncated": record["truncated"],
                    "exception": record["exception"],
                    "is_src_obj_grasped": bool(stats.get("is_src_obj_grasped")),
                    "moved_correct_obj": bool(stats.get("moved_correct_obj")),
                    "moved_wrong_obj": bool(stats.get("moved_wrong_obj")),
                    "src_on_target": bool(stats.get("src_on_target")),
                    "consecutive_grasp": bool(stats.get("consecutive_grasp")),
                    "mean_abs_xyz": record["mean_abs_xyz"],
                    "gripper_close_fraction": record["gripper_close_fraction"],
                    "video": record["video"],
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ready = requests.get(args.server, timeout=10)
    ready.raise_for_status()
    states = load_states(args.eval_config, args.tasks, args.episodes)
    records = []
    for task in args.tasks:
        env = simpler_env.make(
            task,
            max_episode_steps=args.max_episode_steps,
            control_freq=args.control_freq,
        )
        for episode_id, env_seed in enumerate(states[task]):
            record = run_episode(
                env,
                args.server,
                task,
                episode_id,
                env_seed,
                args.output_dir,
                args.image_size,
                args.model_label,
            )
            records.append(record)
            print(
                json.dumps(
                    {
                        key: record[key]
                        for key in ("model", "task", "env_seed", "success", "control_steps", "exception", "video")
                    }
                ),
                flush=True,
            )
        env.close()
    failures = [record for record in records if not record["success"]]
    summary = {
        "model": args.model_label,
        "eval_config": str(args.eval_config),
        "image_size": args.image_size,
        "control_freq": args.control_freq,
        "max_episode_steps": args.max_episode_steps,
        "model_seed": MODEL_SEED,
        "episodes": len(records),
        "successes": sum(record["success"] for record in records),
        "failures": len(failures),
        "failure_videos": [record["video"] for record in failures],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "episodes": records}, indent=2, default=json_default) + "\n"
    )
    write_episodes_csv(args.output_dir / "episodes.csv", records)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
