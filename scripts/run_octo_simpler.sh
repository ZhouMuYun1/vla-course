#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=""
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-root}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$PROJECT_DIR")}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-$WORKSPACE_DIR/SimplerEnv}"
OCTO_DIR="${OCTO_DIR:-$WORKSPACE_DIR/octo}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE_DIR/models/octo-small-1.5}"
PYTHON_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_PYTHON_LIB="$PYTHON_SITE_PACKAGES/nvidia"

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$PROJECT_DIR/config/nvidia_icd.json}"
export LD_LIBRARY_PATH="$NVIDIA_PYTHON_LIB/cudnn/lib:$NVIDIA_PYTHON_LIB/cublas/lib:$NVIDIA_PYTHON_LIB/cuda_nvrtc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SIMPLERENV_DIR:$OCTO_DIR:${PYTHONPATH:-}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

cd "$PROJECT_DIR"

if [[ $# -eq 0 ]]; then
  set -- \
    --checkpoint "$MODEL_DIR" \
    --checkpoint-step 300000 \
    --task widowx_spoon_on_towel \
    --episodes 1 \
    --seed 0 \
    --output-dir "$PROJECT_DIR/results/smoke_spoon"
fi

python scripts/evaluate_octo_simpler.py "$@"
