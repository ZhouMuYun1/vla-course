#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$PROJECT_DIR")}"
DATA_DIR="${DATA_DIR:-$WORKSPACE_DIR/data/bridge_course_subset}"

cd "$PROJECT_DIR"
python scripts/download_bridge_subset.py \
  --output-root "$DATA_DIR" \
  --train-shards 60 \
  --val-shards 12 \
  --seed 42 \
  --workers 4 \
  --max-bytes 9000000000

python scripts/inspect_bridge_subset.py \
  --data-dir "$DATA_DIR" \
  --output-dir artifacts/data_report
