#!/usr/bin/env bash
# Re-evaluate Base and LoRA v1/v2/v3 with the current WidowX script.
# All four use config/eval_states_b_formal.json. Training is not repeated.
set -euo pipefail

cd /home/lizhenkai/vlalora
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_CPP_MIN_LOG_LEVEL=2

OPENVLA="/home/lizhenkai/anaconda3/envs/openvla/bin/python"
SIMPLER="/home/lizhenkai/anaconda3/envs/simpler_env/bin/python"
CONFIG="config/eval_states_b_formal.json"
SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

run_eval() {
  local model="$1"
  local label="$2"
  local output="$3"
  if [[ -f "${output}/summary.json" ]]; then
    local episodes
    episodes="$("$OPENVLA" -c "import json,sys; print(json.load(open(sys.argv[1]))['summary']['episodes'])" "${output}/summary.json")"
    if [[ "${episodes}" == "20" ]]; then
      echo "skip ${label}: ${output}/summary.json already has 20 episodes"
      return 0
    fi
  fi
  stop_server
  echo "start ${label} $(date '+%F %T') model=${model}"
  "$OPENVLA" scripts/serve_openvla.py --model "$model" --port 8000 &
  SERVER_PID=$!
  local ready=0
  for _ in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000 >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "server exited before becoming ready for ${label}" >&2
      exit 1
    fi
    sleep 2
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "server failed to start for ${label}" >&2
    exit 1
  fi
  "$SIMPLER" scripts/evaluate_openvla_widowx.py \
    --episodes 10 \
    --model-label "${label}" \
    --eval-config "${CONFIG}" \
    --output-dir "${output}"
  stop_server
  echo "done ${label} $(date '+%F %T')"
}

run_eval models/openvla-7b base results/b_unified/base
run_eval models/openvla-7b-bridge-lora v1 results/b_unified/v1
run_eval models/openvla-7b-bridge-lora-v2 v2 results/b_unified/v2
run_eval models/openvla-7b-bridge-lora-v3 v3 results/b_unified/v3
echo "UNIFIED_DONE $(date '+%F %T')"
