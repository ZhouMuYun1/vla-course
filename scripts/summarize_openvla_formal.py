"""Build the B-class results table from formal Base and Final episode logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ("widowx_spoon_on_towel", "widowx_put_eggplant_in_basket")


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rate(successes: int, total: int) -> float:
    return 100.0 * successes / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("results/b_formal/base/episodes.csv"))
    parser.add_argument("--final", type=Path, default=Path("results/b_formal/final/episodes.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b_formal"))
    args = parser.parse_args()
    rows = load_rows(args.base) + load_rows(args.final)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.output_dir / "episodes.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for model in ("base", "final"):
        per_task = {}
        for task in TASKS:
            selected = [row for row in rows if row["model"] == model and row["task"] == task]
            successes = sum(row["success"] == "True" for row in selected)
            per_task[task] = {"successes": successes, "episodes": len(selected), "rate": rate(successes, len(selected))}
        score = (per_task[TASKS[0]]["rate"] + per_task[TASKS[1]]["rate"]) / 2
        summary_rows.append({"model": model, "tasks": per_task, "b_simpler_score": score})
    base_score = summary_rows[0]["b_simpler_score"]
    final_score = summary_rows[1]["b_simpler_score"]
    payload = {
        "models": summary_rows,
        "delta_percentage_points": {
            task: summary_rows[1]["tasks"][task]["rate"] - summary_rows[0]["tasks"][task]["rate"] for task in TASKS
        },
        "b_simpler_score_delta": final_score - base_score,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "module", "t1_successes", "t1_episodes", "t2_successes", "t2_episodes", "b_simpler_score"])
        modules = {"base": "none", "final": "lora_q_proj_v_proj"}
        for item in summary_rows:
            writer.writerow(
                [
                    item["model"],
                    modules[item["model"]],
                    item["tasks"][TASKS[0]]["successes"],
                    item["tasks"][TASKS[0]]["episodes"],
                    item["tasks"][TASKS[1]]["successes"],
                    item["tasks"][TASKS[1]]["episodes"],
                    f"{item['b_simpler_score']:.1f}",
                ]
            )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
