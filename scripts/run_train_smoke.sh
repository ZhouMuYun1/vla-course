#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Verify RLDS loading, filtering, transforms, optimizer updates and checkpointing.
# Formal validation visualization remains in the full training run.
bash "$SCRIPT_DIR/run_finetune.sh" \
  --config.num_steps=2 \
  --config.batch_size=2 \
  --config.shuffle_buffer_size=64 \
  --config.optimizer.learning_rate.warmup_steps=1 \
  --config.optimizer.learning_rate.decay_steps=2 \
  --config.log_interval=1 \
  --config.eval_interval=999 \
  --config.save_interval=2 \
  --config.frame_transform_threads=2 \
  "$@"
