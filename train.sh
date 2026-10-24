#!/usr/bin/env bash
# Root-level train entry point (spec 9.3).
# Wrapper around the actual training script.
set -euo pipefail
cd "$(dirname "$0")"
exec bash scripts/run_b_openvla_v3.sh "$@"
