"""Report total/frozen/trainable parameters for the Task-B head-only setup."""

from __future__ import annotations

import argparse
import json

from flax.traverse_util import flatten_dict
import numpy as np

from octo.model.octo_model import OctoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=300000)
    args = parser.parse_args()
    model = OctoModel.load_pretrained(args.checkpoint, step=args.checkpoint_step)
    flat = flatten_dict(model.params, sep=".")
    rows = []
    for name, value in flat.items():
        count = int(np.prod(value.shape))
        frozen = name.startswith("octo_transformer.")
        rows.append((name, count, frozen))
    total = sum(count for _, count, _ in rows)
    frozen = sum(count for _, count, is_frozen in rows if is_frozen)
    output = {
        "mode": "head_only",
        "frozen_pattern": "octo_transformer.*",
        "total_parameters": total,
        "frozen_parameters": frozen,
        "trainable_parameters": total - frozen,
        "trainable_percent": round((total - frozen) / total * 100, 4),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
