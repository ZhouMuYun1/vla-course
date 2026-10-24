"""Download extra Bridge train shards and list keyword trajectories.

Shards already used by data/bridge_course_subset are skipped. Only official
train shards are downloaded, so the existing validation split stays unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_bridge_subset import BASE_URL, SHARD_RE, read_url


REPO_ROOT = Path(__file__).resolve().parents[1]
KEYWORDS = ("spoon", "towel", "eggplant", "basket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "bridge_keyword_extra")
    parser.add_argument(
        "--existing-manifest",
        type=Path,
        default=REPO_ROOT / "data" / "bridge_course_subset" / "bridge_dataset" / "1.0.0" / "subset_manifest.json",
    )
    parser.add_argument("--train-shards", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "artifacts" / "openvla_bridge_split" / "train_extra_keywords.csv",
    )
    return parser.parse_args()


def choose_new_train_shards(index_html: str, used_names: set[str], count: int, seed: int):
    catalog = [
        (name, int(index), int(total))
        for name, split, index, total in SHARD_RE.findall(index_html)
        if split == "train" and name not in used_names
    ]
    if count > len(catalog):
        raise ValueError(f"requested {count} new train shards, only {len(catalog)} remain")
    rng = random.Random(f"bridge-keyword-extra:{seed}:{count}")
    return sorted(rng.sample(catalog, count), key=lambda item: item[1])


def download_shards(version_dir: Path, selected: list[tuple[str, int, int]], workers: int) -> list[dict]:
    records = []
    jobs = []
    for local_index, (remote_name, remote_index, remote_total) in enumerate(selected):
        local_name = f"bridge_dataset-train.tfrecord-{local_index:05d}-of-{len(selected):05d}"
        records.append({
            "remote_name": remote_name,
            "remote_index": remote_index,
            "remote_total": remote_total,
            "local_name": local_name,
        })
        destination = version_dir / local_name
        jobs.append((BASE_URL + remote_name, destination, remote_name))

    print("checking shard sizes", flush=True)
    sizes = {}
    for url, _destination, remote_name in jobs:
        request_url = url
        import urllib.request
        request = urllib.request.Request(request_url, method="HEAD")
        with urllib.request.urlopen(request, timeout=120) as response:
            sizes[remote_name] = int(response.headers["Content-Length"])
    total_bytes = sum(sizes.values())
    done_bytes = 0
    finished = 0
    lock = threading.Lock()

    def show():
        frac = done_bytes / total_bytes if total_bytes else 1
        width = 30
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r[{bar}] {frac * 100:5.1f}%  {done_bytes / 1024 ** 3:.2f}/{total_bytes / 1024 ** 3:.2f} GiB  {finished}/{len(jobs)} shards",
            end="",
            flush=True,
        )

    pending = []
    for record, (url, destination, remote_name) in zip(records, jobs):
        record["bytes"] = sizes[remote_name]
        if destination.exists() and destination.stat().st_size == sizes[remote_name]:
            done_bytes += sizes[remote_name]
            finished += 1
        else:
            pending.append((url, destination, sizes[remote_name]))
    show()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import urllib.request

    def download_one(url: str, destination: Path, size: int) -> None:
        nonlocal done_bytes, finished
        partial = destination.with_name(destination.name + ".partial")
        with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                with lock:
                    done_bytes += len(chunk)
                    show()
        if partial.stat().st_size != size:
            raise IOError(f"size mismatch for {destination.name}")
        partial.replace(destination)
        with lock:
            finished += 1
            show()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, url, destination, size) for url, destination, size in pending]
        for future in as_completed(futures):
            future.result()
    print("\nALL_SHARDS_DONE", flush=True)
    return records


def write_tfds_metadata(version_dir: Path, records: list[dict], seed: int) -> None:
    original_info = json.loads(read_url(BASE_URL + "dataset_info.json"))
    (version_dir / "features.json").write_bytes(read_url(BASE_URL + "features.json"))
    split_info = {item["name"]: item for item in original_info["splits"]}
    source = dict(split_info["train"])
    lengths = source["shardLengths"]
    source["shardLengths"] = [str(lengths[record["remote_index"]]) for record in records]
    source["numBytes"] = str(sum(record["bytes"] for record in records))
    patched = dict(original_info)
    patched["splits"] = [source]
    patched["description"] = "Extra Bridge train shards for spoon/towel/eggplant/basket. See subset_manifest.json."
    (version_dir / "dataset_info.json").write_text(json.dumps(patched, indent=2), encoding="utf-8")
    manifest = {
        "source": BASE_URL,
        "selection_seed": seed,
        "selected_bytes": sum(record["bytes"] for record in records),
        "splits": {"train": records},
    }
    (version_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def scan_keywords(data_dir: Path, csv_path: Path) -> None:
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    import numpy as np
    import tensorflow_datasets as tfds

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bridge_filters import classify_instruction

    def to_text(value) -> str:
        item = np.asarray(value).reshape(-1)[0]
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)

    builder = tfds.builder("bridge_dataset", data_dir=str(data_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    rows = []
    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        steps = list(episode["steps"])
        if not steps:
            continue
        instruction = to_text(steps[0]["language_instruction"])
        if not any(keyword in instruction.lower() for keyword in KEYWORDS):
            continue
        rows.append({
            "split": "train",
            "episode_index": episode_index,
            "category": classify_instruction(instruction),
            "instruction": instruction,
            "steps": len(steps),
        })
        if len(rows) % 20 == 0:
            print(f"keyword trajectories {len(rows)}", flush=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "episode_index", "category", "instruction", "steps"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} keyword trajectories, {sum(row['steps'] for row in rows)} steps -> {csv_path}", flush=True)


def main() -> None:
    args = parse_args()
    version_dir = args.output_root / "bridge_dataset" / "1.0.0"
    version_dir.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.existing_manifest.read_text())
    used = {row["remote_name"] for row in existing["splits"]["train"]}
    index_html = read_url(BASE_URL).decode("utf-8")
    selected = choose_new_train_shards(index_html, used, args.train_shards, args.seed)
    print(f"selected {len(selected)} new train shards", flush=True)
    records = download_shards(version_dir, selected, args.workers)
    write_tfds_metadata(version_dir, records, args.seed)
    scan_keywords(args.output_root, args.csv)


if __name__ == "__main__":
    main()
