#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$PROJECT_DIR")}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$WORKSPACE_DIR/models/octo-small-1.5}"
FINAL_MODEL_DIR="${FINAL_MODEL_DIR:?Set FINAL_MODEL_DIR to the selected checkpoint directory}"
FINAL_STEP="${FINAL_STEP:?Set FINAL_STEP to the selected checkpoint step}"
EVAL_STATES="${EVAL_STATES:-$PROJECT_DIR/config/eval_states_b.json}"

run_model() {
  local model_name="$1"
  local checkpoint="$2"
  local step="$3"
  for task in widowx_spoon_on_towel widowx_put_eggplant_in_basket; do
    bash "$PROJECT_DIR/scripts/run_octo_simpler.sh" \
      --model-name "$model_name" \
      --checkpoint "$checkpoint" \
      --checkpoint-step "$step" \
      --task "$task" \
      --episodes 10 \
      --model-seed 0 \
      --eval-states "$EVAL_STATES" \
      --max-episode-steps 60 \
      --control-freq 5 \
      --output-dir "$PROJECT_DIR/results/formal/${model_name}/${task}"
  done
}

run_model base "$BASE_MODEL_DIR" 300000
run_model final "$FINAL_MODEL_DIR" "$FINAL_STEP"
