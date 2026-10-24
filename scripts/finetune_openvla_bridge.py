"""LoRA-finetune OpenVLA-7B on the Bridge subset used by this project.

The training trajectories are the official RLDS train split listed in
artifacts/openvla_bridge_split/train.csv. The official val split is used only
to pick the Final adapter by action-token loss. SimplerEnv success is not
used for checkpoint selection.

Actions are normalized with the base checkpoint's bridge_orig q01/q99
statistics, so inference keeps unnorm_key=bridge_orig. The gripper dimension
is left in its recorded 0/1 range, matching the base model's mask.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import random
import time

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
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.optim import AdamW
from transformers import AutoModelForVision2Seq, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100
LORA_TARGETS = ["q_proj", "v_proj"]
TASK_KEYWORDS = ("spoon", "towel", "eggplant", "basket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "openvla-7b")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "bridge_course_subset")
    parser.add_argument("--train-csv", type=Path, default=REPO_ROOT / "artifacts" / "openvla_bridge_split" / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=REPO_ROOT / "artifacts" / "openvla_bridge_split" / "test.csv")
    parser.add_argument("--extra-data-dir", type=Path, default=REPO_ROOT / "data" / "bridge_keyword_extra")
    parser.add_argument(
        "--extra-train-csv",
        type=Path,
        default=REPO_ROOT / "artifacts" / "openvla_bridge_split" / "train_extra_keywords.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoints" / "openvla_bridge_lora_v2")
    parser.add_argument("--merged-dir", type=Path, default=REPO_ROOT / "models" / "openvla-7b-bridge-lora-v2")
    parser.add_argument("--log-path", type=Path, default=REPO_ROOT / "logs" / "openvla_bridge_lora_v2.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--val-interval", type=int, default=25)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-delta", type=float, default=0.02)
    parser.add_argument("--min-xyz", type=float, default=0.005)
    parser.add_argument("--min-rotation", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_episode_ids(path: Path) -> set[int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {int(row["episode_index"]) for row in csv.DictReader(handle)}


def to_text(value) -> str:
    item = np.asarray(value).reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def keeps_motion(action: np.ndarray, previous: np.ndarray | None, min_xyz: float, min_rotation: float) -> bool:
    moving = np.linalg.norm(action[:3]) >= min_xyz or np.linalg.norm(action[3:6]) >= min_rotation
    gripper_change = previous is not None and abs(float(action[6]) - float(previous[6])) >= 0.5
    return bool(moving or gripper_change)


def load_split(
    data_dir: Path,
    split: str,
    episode_ids: set[int],
    min_xyz: float,
    min_rotation: float,
) -> tuple[list[dict], dict[str, int]]:
    builder = tfds.builder("bridge_dataset", data_dir=str(data_dir))
    dataset = builder.as_dataset(split=split, shuffle_files=False)
    samples = []
    seen = 0
    kept_episodes = 0
    skipped_keyword = 0
    skipped_still = 0
    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        if episode_index not in episode_ids:
            continue
        steps = list(episode["steps"])
        if not steps:
            continue
        seen += 1
        instruction = to_text(steps[0]["language_instruction"]).lower()
        if not any(keyword in instruction for keyword in TASK_KEYWORDS):
            skipped_keyword += 1
            if seen == len(episode_ids):
                break
            continue
        kept_episodes += 1
        previous = None
        for step in steps:
            action = np.asarray(step["action"], dtype=np.float32)
            if not keeps_motion(action, previous, min_xyz, min_rotation):
                skipped_still += 1
                previous = action
                continue
            image = Image.fromarray(np.asarray(step["observation"]["image_0"])).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            samples.append({
                "jpeg": buffer.getvalue(),
                "instruction": instruction,
                "action": action,
            })
            previous = action
        if seen == len(episode_ids):
            break
        if kept_episodes % 20 == 0:
            print(f"cached {kept_episodes} keyword {split} trajectories", flush=True)
    if not samples:
        raise RuntimeError(f"{split}: no training samples left after keyword and motion filters")
    return samples, {
        "episodes_in_split_list": seen,
        "keyword_episodes": kept_episodes,
        "skipped_other_instructions": skipped_keyword,
        "skipped_still_steps": skipped_still,
        "kept_steps": len(samples),
    }


def bridge_stats(model_dir: Path) -> dict:
    config = json.loads((model_dir / "config.json").read_text())
    return config["norm_stats"]["bridge_orig"]["action"]


def normalize_action(action: np.ndarray, stats: dict) -> np.ndarray:
    mask = np.asarray(stats["mask"], dtype=bool)
    low = np.asarray(stats["q01"], dtype=np.float64)
    high = np.asarray(stats["q99"], dtype=np.float64)
    scale = np.maximum(high - low, 1e-8)
    normalized = np.clip(2.0 * (action - low) / scale - 1.0, -1.0, 1.0)
    raw = np.clip(action, -1.0, 1.0)
    return np.where(mask, normalized, raw).astype(np.float64)


class ActionTokenizer:
    def __init__(self, tokenizer, bins: int = 256):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.bins = np.linspace(-1, 1, bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = int(tokenizer.vocab_size - (bins + 1))

    def encode_action(self, action: np.ndarray) -> str:
        discretized = np.digitize(np.clip(action, -1, 1), self.bins)
        token_ids = list(self.tokenizer.vocab_size - discretized)
        return self.tokenizer.decode(token_ids)


def build_example(sample: dict, processor, action_tokenizer: ActionTokenizer, stats: dict, device: str):
    image = Image.open(io.BytesIO(sample["jpeg"])).convert("RGB")
    normalized = normalize_action(sample["action"], stats)
    action_text = action_tokenizer.encode_action(normalized)
    prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut: {action_text}</s>"
    encoded = processor.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    image_inputs = processor.image_processor(image, return_tensors="pt")
    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    labels[:, : -(normalized.shape[0] + 1)] = IGNORE_INDEX
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "pixel_values": image_inputs["pixel_values"].to(device=device, dtype=torch.bfloat16),
        "labels": labels.to(device),
    }


def evaluate_loss(model, samples: list[dict], processor, action_tokenizer, stats: dict, device: str) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for sample in samples:
            batch = build_example(sample, processor, action_tokenizer, stats, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
            losses.append(float(output.loss.item()))
    model.train()
    return float(np.mean(losses))


def count_parameters(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_ratio": trainable / total}


def merge_adapter(base_dir: Path, adapter_dir: Path, output_dir: Path, device: str) -> None:
    processor = AutoProcessor.from_pretrained(base_dir, trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(
        base_dir,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    for name in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
        source = base_dir / name
        target = output_dir / name
        if source.exists() and not target.exists():
            target.write_bytes(source.read_bytes())
    del merged, base
    torch.cuda.empty_cache()
    print(f"merged model saved to {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("loading Bridge trajectories", flush=True)
    train_samples, train_filter = load_split(
        args.data_dir, "train", read_episode_ids(args.train_csv), args.min_xyz, args.min_rotation
    )
    if args.extra_train_csv.exists() and args.extra_data_dir.exists():
        extra_samples, extra_filter = load_split(
            args.extra_data_dir, "train", read_episode_ids(args.extra_train_csv), args.min_xyz, args.min_rotation
        )
        train_samples.extend(extra_samples)
        train_filter["extra"] = extra_filter
        print(f"added extra keyword samples {len(extra_samples)}", flush=True)
    val_samples, val_filter = load_split(
        args.data_dir, "val", read_episode_ids(args.val_csv), args.min_xyz, args.min_rotation
    )
    print(f"train samples {len(train_samples)} val samples {len(val_samples)}", flush=True)
    print(f"train filter {train_filter}", flush=True)
    print(f"val filter {val_filter}", flush=True)
    rng = random.Random(args.seed)
    val_probe = rng.sample(val_samples, min(args.val_samples, len(val_samples)))

    stats = bridge_stats(args.model)
    print(f"loading {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model.language_model.config.use_cache = False
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for module_name in ("vision_backbone", "projector"):
        for parameter in getattr(model, module_name).parameters():
            parameter.requires_grad = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
        ),
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    parameter_count = count_parameters(model)
    print(
        f"trainable {parameter_count['trainable']} / {parameter_count['total']} "
        f"({parameter_count['trainable_ratio']:.4%})",
        flush=True,
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    config = {
        "base_model": str(args.model),
        "train_trajectories": len(read_episode_ids(args.train_csv)),
        "train_samples": len(train_samples),
        "val_trajectories": len(read_episode_ids(args.val_csv)),
        "val_samples": len(val_samples),
        "max_steps": args.max_steps,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "effective_batch_size": args.grad_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_targets": LORA_TARGETS,
        "precision": "base weights and forward bfloat16; trainable LoRA weights float32",
        "task_keywords": list(TASK_KEYWORDS),
        "min_xyz": args.min_xyz,
        "min_rotation": args.min_rotation,
        "train_filter": train_filter,
        "val_filter": val_filter,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_delta": args.early_stop_delta,
        "normalization": "base checkpoint bridge_orig q01/q99, gripper unmasked",
        "final_rule": "lowest validation action-token loss, early stop when val loss stops improving; SimplerEnv is not used",
        "parameters": parameter_count,
        "seed": args.seed,
    }
    (args.output_dir / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    best_val = float("inf")
    best_dir = args.output_dir / "best_adapter"
    stall = 0
    log_file = args.log_path.open("w", encoding="utf-8")
    order = list(range(len(train_samples)))
    rng.shuffle(order)
    cursor = 0
    running_loss = 0.0
    started = time.time()

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(args.grad_accumulation_steps):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            sample = train_samples[order[cursor]]
            cursor += 1
            batch = build_example(sample, processor, action_tokenizer, stats, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = output.loss / args.grad_accumulation_steps
            loss.backward()
            accumulated += float(output.loss.item())
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
        warmup_scale = min(1.0, step / max(args.warmup_steps, 1))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * warmup_scale
        optimizer.step()
        running_loss += accumulated / args.grad_accumulation_steps
        record = {
            "step": step,
            "train_loss": accumulated / args.grad_accumulation_steps,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": round(time.time() - started, 1),
        }
        if step % args.val_interval == 0 or step == args.max_steps:
            val_loss = evaluate_loss(model, val_probe, processor, action_tokenizer, stats, args.device)
            record["val_loss"] = val_loss
            record["mean_train_loss"] = running_loss / args.val_interval
            running_loss = 0.0
            if val_loss < best_val - args.early_stop_delta:
                best_val = val_loss
                stall = 0
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                record["saved_best"] = True
            else:
                stall += 1
                record["stall"] = stall
            print(
                f"step {step} train {record['train_loss']:.4f} val {val_loss:.4f} best {best_val:.4f} stall {stall}",
                flush=True,
            )
            if stall >= args.early_stop_patience:
                print(f"early stop at step {step}", flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                log_file.close()
                print(f"best validation loss {best_val:.4f} at {best_dir}", flush=True)
                del model
                torch.cuda.empty_cache()
                merge_adapter(args.model, best_dir, args.merged_dir, args.device)
                return
        elif step % 10 == 0:
            print(f"step {step} train {record['train_loss']:.4f}", flush=True)
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    log_file.close()
    print(f"best validation loss {best_val:.4f} at {best_dir}", flush=True)
    del model
    torch.cuda.empty_cache()
    merge_adapter(args.model, best_dir, args.merged_dir, args.device)


if __name__ == "__main__":
    main()
