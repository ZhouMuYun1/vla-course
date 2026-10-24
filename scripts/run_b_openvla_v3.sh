#!/usr/bin/env bash
# Fine-tune on the original keyword trajectories plus the extra Bridge shards,
# then evaluate the merged model on the fixed WidowX seeds.
set -euo pipefail

cd /home/lizhenkai/vlalora
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_CPP_MIN_LOG_LEVEL=2

OPENVLA="/home/lizhenkai/anaconda3/envs/openvla/bin/python"
SIMPLER="/home/lizhenkai/anaconda3/envs/simpler_env/bin/python"

"$OPENVLA" scripts/finetune_openvla_bridge.py \
  --output-dir checkpoints/openvla_bridge_lora_v3 \
  --merged-dir models/openvla-7b-bridge-lora-v3 \
  --log-path logs/openvla_bridge_lora_v3.jsonl \
  --max-steps 1200 \
  --val-interval 50

"$OPENVLA" scripts/serve_openvla.py --model models/openvla-7b-bridge-lora-v3 --port 8000 &
server_pid=$!
for _ in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000 >/dev/null; then
    break
  fi
  sleep 2
done
"$SIMPLER" scripts/evaluate_openvla_widowx.py --episodes 10 --output-dir results/b_eval_v3/final
kill "$server_pid" || true
wait "$server_pid" || true
