#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT_ROOT=/mnt/contextual-forest/staged-debug-20260907
readonly CODE_DIR="${1:?supply the authenticated analysis checkout}"
readonly RELEASE=/mnt/contextual-forest/releases/contextual-forest-adapter-cf8b808-20260831T131607Z
cd "${CODE_DIR}"
export HF_HOME=/mnt/contextual-forest/hf-home
export HUGGINGFACE_HUB_CACHE=/mnt/contextual-forest/huggingface/hub
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

# The completed development-only seal selects both pair heads at update 900
# and the unary head at 1000. Each file is supplied once; the evaluator checks
# its bytes, head identity and selected arm before opening reserved test tokens.
/mnt/contextual-forest/venv/bin/python scripts/evaluate_staged_fresh_selected.py \
  --selection "${EXPERIMENT_ROOT}/fresh-selection-v1/selection.json" \
  --expected-selection-sha256 f05a08df0de9bd81e7b782eae44bd2672f8290288137fe6de77b9fe6f6f68ad0 \
  --data-manifest "${EXPERIMENT_ROOT}/fresh-data-v1/manifest.json" \
  --expected-data-manifest-sha256 70d989988c6318a52960791667804fde30e794f324628186df8d010438439753 \
  --test-jsonl "${EXPERIMENT_ROOT}/fresh-data-v1/test.jsonl" \
  --expected-test-jsonl-sha256 2e22dae402fe61c6febf2a0d48c8161c1c8bb44a757061730c11205d2b6bffbb \
  --expectations "${RELEASE}/expectations/production-expectations-v2.json" \
  --expected-expectations-sha256 a978c17befc2788b0b773eee7cd608ac0200379947a63a0ba8699a0b9818c956 \
  --backbone-checkpoint "${RELEASE}/inputs/mdlm-owt-backbone-schema-v2.pt" \
  --selected-checkpoint "${EXPERIMENT_ROOT}/fresh-pilot-resume-v1/checkpoints/step-000900.pt" \
  --selected-checkpoint "${EXPERIMENT_ROOT}/fresh-pilot-resume-v1/checkpoints/step-001000.pt" \
  --output-dir "${EXPERIMENT_ROOT}/fresh-test-v1" \
  > "${EXPERIMENT_ROOT}/logs/fresh-test-v1.log" 2>&1
