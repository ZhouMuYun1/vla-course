#!/usr/bin/env bash
# B-class OpenVLA run: LoRA finetune on the Bridge subset, then 10 episodes
# of Base and Final on both WidowX tasks. Formal seeds are 1100-1109 and
# 2200-2209 and are not read during training.
set -euo pipefail

cd /home/lizhenkai/vlalora
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_CPP_MIN_LOG_LEVEL=2

OPENVLA="/home/lizhenkai/anaconda3/envs/openvla/bin/python"
SIMPLER="/home/lizhenkai/anaconda3/envs/simpler_env/bin/python"

"$OPENVLA" scripts/finetune_openvla_bridge.py

run_eval() {
  local model="$1"
  local output="$2"
  "$OPENVLA" scripts/serve_openvla.py --model "$model" --port 8000 &
  local pid=$!
  for _ in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8000 >/dev/null; then
      break
    fi
    sleep 2
  done
  "$SIMPLER" scripts/evaluate_openvla_widowx.py --episodes 10 --output-dir "$output"
  kill "$pid" || true
  wait "$pid" || true
}

run_eval models/openvla-7b-bridge-lora-v2 results/b_eval_v2/final

"$OPENVLA" - << 'PY'
import json
from pathlib import Path

def load(path):
    payload = json.loads(Path(path).read_text())
    summary = payload.get("summary", payload)
    return summary

base = load("results/b_eval/base/summary.json")
final = load("results/b_eval_v2/final/summary.json")
print(json.dumps({"base": base, "final": final}, indent=2))
PY
