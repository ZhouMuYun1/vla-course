#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$PROJECT_DIR")}"
OCTO_DIR="${OCTO_DIR:-$WORKSPACE_DIR/octo}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE_DIR/models/octo-small-1.5}"
DATA_DIR="${DATA_DIR:-$WORKSPACE_DIR/data/bridge_course_subset}"
SAVE_DIR="${SAVE_DIR:-$WORKSPACE_DIR/checkpoints}"

export PYTHONPATH="$PROJECT_DIR:$OCTO_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline

mkdir -p "$PROJECT_DIR/logs" "$SAVE_DIR"
cd "$PROJECT_DIR"

python "$OCTO_DIR/scripts/finetune.py" \
  --name task_b_bridge_head \
  --config="$PROJECT_DIR/config/finetune_octo_bridge.py:head_only,language_conditioned" \
  --config.pretrained_path="$MODEL_DIR" \
  --config.pretrained_step=300000 \
  --config.dataset_kwargs.data_dir="$DATA_DIR" \
  --config.save_dir="$SAVE_DIR" \
  "$@" 2>&1 | tee "$PROJECT_DIR/logs/train_head_only.log"
