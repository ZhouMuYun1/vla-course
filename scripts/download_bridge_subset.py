"""Download a deterministic, disk-bounded subset of official BridgeData V2 RLDS."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import shutil
import urllib.request


BASE_URL = (
    "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/"
    "tfds/bridge_dataset/1.0.0/"
)
SHARD_RE = re.compile(
    r'href="(bridge_dataset-(train|val)\.tfrecord-(\d+)-of-(\d+))"'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-shards", type=int, default=20)
    parser.add_argument("--val-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-bytes", type=int, default=3_000_000_000)
    return parser.parse_args()


def read_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def download(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_size:
        return
    partial = destination.with_name(destination.name + ".partial")
    with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    if partial.stat().st_size != expected_size:
        raise IOError(
            f"size mismatch for {destination.name}: "
            f"{partial.stat().st_size} != {expected_size}"
        )
    partial.replace(destination)


def choose_shards(index_html: str, split: str, count: int, rng: random.Random):
    matches = [
        (name, int(index), int(total))
        for name, item_split, index, total in SHARD_RE.findall(index_html)
        if item_split == split
    ]
    if count > len(matches):
        raise ValueError(f"requested {count} {split} shards, only {len(matches)} exist")
    return sorted(rng.sample(matches, count), key=lambda item: item[1])


def extend_selection(
    catalog: list[tuple[str, int, int]],
    existing_records: list[dict],
    target_count: int,
    seed: int,
    split: str,
) -> list[tuple[str, int, int]]:
    """Preserve an existing subset and append deterministic unseen shards."""
    existing = [
        (row["remote_name"], int(row["remote_index"]), int(row["remote_total"]))
        for row in existing_records
    ]
    if target_count < len(existing):
        raise ValueError(
            f"cannot shrink existing {split} subset from {len(existing)} to {target_count}"
        )
    existing_names = {name for name, _, _ in existing}
    remaining = [row for row in catalog if row[0] not in existing_names]
    needed = target_count - len(existing)
    if needed > len(remaining):
        raise ValueError(f"requested {needed} extra {split} shards, only {len(remaining)} remain")
    rng = random.Random(f"bridge-v2:{seed}:{split}:{len(existing)}:{target_count}")
    return existing + sorted(rng.sample(remaining, needed), key=lambda item: item[1])


def main() -> None:
    args = parse_args()
    version_dir = args.output_root / "bridge_dataset" / "1.0.0"
    version_dir.mkdir(parents=True, exist_ok=True)

    index_html = read_url(BASE_URL).decode("utf-8")
    catalog = {
        split: [
            (name, int(index), int(total))
            for name, item_split, index, total in SHARD_RE.findall(index_html)
            if item_split == split
        ]
        for split in ("train", "val")
    }
    manifest_path = version_dir / "subset_manifest.json"
    existing_manifest = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(existing_manifest["selection_seed"]) != args.seed:
            raise ValueError("existing subset uses a different selection seed")
        selections = {
            "train": extend_selection(
                catalog["train"],
                existing_manifest["splits"]["train"],
                args.train_shards,
                args.seed,
                "train",
            ),
            "val": extend_selection(
                catalog["val"],
                existing_manifest["splits"]["val"],
                args.val_shards,
                args.seed,
                "val",
            ),
        }
    else:
        rng = random.Random(args.seed)
        selections = {
            "train": choose_shards(index_html, "train", args.train_shards, rng),
            "val": choose_shards(index_html, "val", args.val_shards, rng),
        }

    selected_sizes = {}
    existing_sizes = {
        (split, row["remote_name"]): int(row["bytes"])
        for split, rows in (existing_manifest or {}).get("splits", {}).items()
        for row in rows
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(remote_size, BASE_URL + name): (split, name)
            for split, rows in selections.items()
            for name, _, _ in rows
            if (split, name) not in existing_sizes
        }
        for future in as_completed(futures):
            split, name = futures[future]
            selected_sizes[(split, name)] = future.result()
    selected_sizes.update(existing_sizes)
    total_bytes = sum(selected_sizes.values())
    if total_bytes > args.max_bytes:
        raise RuntimeError(
            f"selected shards need {total_bytes:,} bytes, over budget {args.max_bytes:,}"
        )

    original_info = json.loads(read_url(BASE_URL + "dataset_info.json"))
    (version_dir / "features.json").write_bytes(read_url(BASE_URL + "features.json"))
    (version_dir / "dataset_info.original.json").write_text(
        json.dumps(original_info, indent=2), encoding="utf-8"
    )

    manifest = {
        "source": BASE_URL,
        "selection_seed": args.seed,
        "byte_budget": args.max_bytes,
        "selected_bytes": total_bytes,
        "splits": {},
    }
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for split, selected in selections.items():
            local_total = len(selected)
            manifest["splits"][split] = []
            old_records = {
                row["remote_name"]: row
                for row in (existing_manifest or {}).get("splits", {}).get(split, [])
            }
            for local_index, (remote_name, remote_index, remote_total) in enumerate(selected):
                local_name = (
                    f"bridge_dataset-{split}.tfrecord-"
                    f"{local_index:05d}-of-{local_total:05d}"
                )
                expected_size = selected_sizes[(split, remote_name)]
                destination = version_dir / local_name
                record = {
                    "remote_name": remote_name,
                    "remote_index": remote_index,
                    "remote_total": remote_total,
                    "local_name": local_name,
                    "bytes": expected_size,
                }
                manifest["splits"][split].append(record)
                if remote_name in old_records:
                    old_name = old_records[remote_name]["local_name"]
                    old_path = version_dir / old_name
                    if not old_path.exists():
                        candidates = list(
                            version_dir.glob(
                                f"bridge_dataset-{split}.tfrecord-{local_index:05d}-of-*"
                            )
                        )
                        if len(candidates) != 1:
                            raise FileNotFoundError(f"cannot locate existing shard {remote_name}")
                        old_path = candidates[0]
                    if old_path != destination:
                        old_path.rename(destination)
                    if destination.stat().st_size != expected_size:
                        raise IOError(f"existing shard size mismatch: {destination}")
                else:
                    jobs.append(
                        pool.submit(download, BASE_URL + remote_name, destination, expected_size)
                    )
        for future in as_completed(jobs):
            future.result()

    split_info = {item["name"]: item for item in original_info["splits"]}
    patched_splits = []
    for split, selected in selections.items():
        source = dict(split_info[split])
        original_lengths = source["shardLengths"]
        source["shardLengths"] = [str(original_lengths[index]) for _, index, _ in selected]
        source["numBytes"] = str(
            sum(row["bytes"] for row in manifest["splits"][split])
        )
        patched_splits.append(source)
    patched_info = dict(original_info)
    patched_info["splits"] = patched_splits
    patched_info["description"] = (
        "Deterministic shard subset of the official BridgeData V2 RLDS release. "
        "See subset_manifest.json for source shard mapping."
    )
    (version_dir / "dataset_info.json").write_text(
        json.dumps(patched_info, indent=2), encoding="utf-8"
    )
    (version_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for statistics_path in version_dir.glob("dataset_statistics_*.json"):
        statistics_path.unlink()
    print(json.dumps({
        "dataset_dir": str(version_dir),
        "train_shards": args.train_shards,
        "val_shards": args.val_shards,
        "selected_bytes": total_bytes,
        "selected_gib": round(total_bytes / 1024**3, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
