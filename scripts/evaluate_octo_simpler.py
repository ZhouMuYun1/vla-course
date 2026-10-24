"""Evaluate a local Octo checkpoint on a SimplerEnv WidowX task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import mediapy as media
import jax
import numpy as np
import tensorflow as tf

import simpler_env
from octo.model.octo_model import OctoModel
from simpler_env.policies.octo.octo_model import OctoInference
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


WIDOWX_TASKS = (
    "widowx_spoon_on_towel",
    "widowx_put_eggplant_in_basket",
)


class SimplerOctoCompatibility:
    """Translate SimplerEnv's legacy pad-mask key to Octo's current API."""

    def __init__(self, model: OctoModel) -> None:
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def sample_actions(self, observations, tasks, **kwargs):
        observations = dict(observations)
        timestep_pad_mask = observations.pop("pad_mask")
        return self.model.sample_actions(
            observations,
            tasks,
            timestep_pad_mask=timestep_pad_mask,
            **kwargs,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=300000)
    parser.add_argument("--task", choices=WIDOWX_TASKS, default=WIDOWX_TASKS[0])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model-name", default="octo")
    parser.add_argument("--model-seed", "--seed", dest="model_seed", type=int, default=0)
    parser.add_argument("--env-seed-start", type=int, default=0)
    parser.add_argument("--eval-states", type=Path)
    parser.add_argument("--max-episode-steps", type=int, default=60)
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--tf-memory-limit", type=int, default=3072)
    parser.add_argument("--output-dir", type=Path, default=Path("results/smoke"))
    return parser.parse_args()


def configure_tensorflow(memory_limit: int) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)],
        )


def load_policy(checkpoint: Path, step: int, seed: int) -> OctoInference:
    model = OctoModel.load_pretrained(str(checkpoint), step=step)
    return OctoInference(
        model=SimplerOctoCompatibility(model),
        dataset_id="bridge_dataset",
        policy_setup="widowx_bridge",
        init_rng=seed,
    )


def reset_policy_rng(policy: OctoInference, seed: int) -> None:
    policy.rng = jax.random.PRNGKey(seed)
    for _ in range(5):
        policy.rng, _ = jax.random.split(policy.rng)


def load_eval_states(args: argparse.Namespace) -> list[dict]:
    if args.eval_states is None:
        return [
            {"episode": index, "env_seed": args.env_seed_start + index, "options": {}}
            for index in range(args.episodes)
        ]
    payload = json.loads(args.eval_states.read_text(encoding="utf-8"))
    states = payload["tasks"][args.task]
    if args.episodes > len(states):
        raise ValueError(
            f"{args.eval_states} has {len(states)} states for {args.task}, "
            f"but {args.episodes} were requested"
        )
    return states[: args.episodes]


def run_episode(
    env,
    policy: OctoInference,
    episode_id: int,
    env_seed: int,
    model_seed: int,
    reset_options: dict,
    output_dir: Path,
) -> dict:
    started_at = time.time()
    reset_policy_rng(policy, model_seed)
    obs, reset_info = env.reset(seed=env_seed, options=reset_options or None)
    instruction = env.get_language_instruction()
    policy.reset(instruction)

    image = get_image_from_maniskill2_obs_dict(env, obs)
    frames = [image]
    predicted_terminated = False
    truncated = False
    success = False
    timestep = 0
    last_info = {}

    while not (predicted_terminated or truncated):
        _, action = policy.step(image, instruction)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated and not env.is_final_subtask():
            predicted_terminated = False
            env.advance_to_next_subtask()

        env_action = np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]]
        )
        obs, _, success, truncated, last_info = env.step(env_action)

        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

        image = get_image_from_maniskill2_obs_dict(env, obs)
        frames.append(image)
        timestep += 1

    success = bool(success)
    video_path = output_dir / f"episode_{episode_id:03d}_success_{success}.mp4"
    media.write_video(video_path, frames, fps=5)
    return {
        "episode": episode_id,
        "env_seed": env_seed,
        "model_seed": model_seed,
        "instruction": instruction,
        "success": success,
        "control_steps": timestep,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "reset_info": reset_info,
        "episode_stats": last_info.get("episode_stats", {}),
        "video": str(video_path),
    }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_tensorflow(args.tf_memory_limit)

    policy = load_policy(args.checkpoint, args.checkpoint_step, args.model_seed)
    env = simpler_env.make(
        args.task,
        max_episode_steps=args.max_episode_steps,
        control_freq=args.control_freq,
    )
    states = load_eval_states(args)
    records = []
    episode_log = args.output_dir / "episodes.jsonl"

    with episode_log.open("w", encoding="utf-8") as log_file:
        for run_index, state in enumerate(states):
            episode_id = int(state.get("episode", run_index))
            episode_model_seed = args.model_seed + episode_id
            record = run_episode(
                env,
                policy,
                episode_id,
                int(state["env_seed"]),
                episode_model_seed,
                state.get("options", {}),
                args.output_dir,
            )
            records.append(record)
            log_file.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
            log_file.flush()
            print(json.dumps(record, ensure_ascii=False, default=json_default))

    success_count = sum(record["success"] for record in records)
    summary = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": args.checkpoint_step,
        "task": args.task,
        "episodes": args.episodes,
        "successes": success_count,
        "success_rate": success_count / args.episodes,
        "model_seed_base": args.model_seed,
        "eval_states": str(args.eval_states) if args.eval_states else None,
        "max_episode_steps": args.max_episode_steps,
        "control_freq": args.control_freq,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
