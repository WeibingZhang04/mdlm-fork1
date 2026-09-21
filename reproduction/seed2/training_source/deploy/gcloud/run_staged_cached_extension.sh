#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT_ROOT=/mnt/contextual-forest/staged-debug-20260907
readonly CODE_DIR="${1:?supply the clean committed checkout}"
cd "${CODE_DIR}"
export PYTHONUNBUFFERED=1
/mnt/contextual-forest/venv/bin/python scripts/run_staged_cached_overfit.py \
  --source-run-dir "${EXPERIMENT_ROOT}/overfit32-v1" \
  --source-results-sha256 26ff20d7e5e4fce89e4a2f98eeaffbf979ec304cf31be34afecf0a379b8ff784 \
  --output-dir "${EXPERIMENT_ROOT}/overfit32-2000-v1" \
  --steps 2000 --eval-every 100 --replay-tolerance 0.0001 --device cuda \
  > "${EXPERIMENT_ROOT}/logs/overfit32-2000-v1.log" 2>&1
