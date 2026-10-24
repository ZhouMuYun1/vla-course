"""Combine formal Base/Final episode logs into course result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ("widowx_spoon_on_towel", "widowx_put_eggplant_in_basket")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/formal"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    summary_rows = []
    for model in ("base", "final"):
        rates = []
        for task in TASKS:
            path = args.results_dir / model / task / "episodes.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for record in records:
                episodes.append({
                    "model": model,
                    "task": task,
                    "episode": record["episode"],
                    "env_seed": record["env_seed"],
                    "model_seed": record["model_seed"],
                    "success": int(record["success"]),
                    "control_steps": record["control_steps"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "video": record["video"],
                })
            successes = sum(row["success"] for row in records)
            rate = successes / len(records)
            rates.append(rate)
            summary_rows.append({
                "model": model,
                "task": task,
                "successes": successes,
                "episodes": len(records),
                "success_rate": rate,
            })
        summary_rows.append({
            "model": model,
            "task": "B-SimplerScore",
            "successes": "",
            "episodes": "",
            "success_rate": sum(rates) / len(rates),
        })

    for filename, rows in (("episodes.csv", episodes), ("results.csv", summary_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
