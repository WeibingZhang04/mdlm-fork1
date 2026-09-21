#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT_ROOT=/mnt/contextual-forest/staged-debug-20260907
readonly CODE_DIR="${1:?supply the clean committed checkout}"
readonly RUN_MODE="${2:?supply smoke or pilot}"
readonly RELEASE=/mnt/contextual-forest/releases/contextual-forest-adapter-cf8b808-20260831T131607Z
case "${RUN_MODE}" in
  smoke) train_count=2; dev_count=2; updates=2; warmup=1; interval=1 ;;
  pilot) train_count=2048; dev_count=64; updates=1000; warmup=100; interval=100 ;;
  *) exit 2 ;;
esac
cd "${CODE_DIR}"
export HF_HOME=/mnt/contextual-forest/hf-home
export HUGGINGFACE_HUB_CACHE=/mnt/contextual-forest/huggingface/hub
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
/mnt/contextual-forest/venv/bin/python scripts/run_staged_fresh_training.py \
  --train-jsonl "${EXPERIMENT_ROOT}/fresh-data-v1/train.jsonl" \
  --dev-jsonl "${EXPERIMENT_ROOT}/fresh-data-v1/dev.jsonl" \
  --expected-train-sha256 1d387c4b4e90e2a7a994eb70521335c45ef2e1574aae8de9108dc944f1ee7dd6 \
  --expected-dev-sha256 68390ef4a9e2f5be6df70657ea70e186957ae68450e4f680392eff84fc36802a \
  --backbone-checkpoint "${RELEASE}/inputs/mdlm-owt-backbone-schema-v2.pt" \
  --expectations "${RELEASE}/expectations/production-expectations-v2.json" \
  --expected-expectations-sha256 a978c17befc2788b0b773eee7cd608ac0200379947a63a0ba8699a0b9818c956 \
  --output-dir "${EXPERIMENT_ROOT}/fresh-${RUN_MODE}-v1" \
  --override model.length=128 \
  --train-examples "${train_count}" --dev-examples "${dev_count}" --length 128 \
  --steps "${updates}" --warmup-steps "${warmup}" --eval-every "${interval}" \
  --batch-size 4 --backbone-batch-size 1 --learning-rate .0003 \
  --mask-rates .25 .5 .75 .9 --rank 16 --directional-rank 8 --unary-rank 16 \
  --top-k 64 --factor-init-std .25 --seed 1 --device cuda \
  > "${EXPERIMENT_ROOT}/logs/fresh-${RUN_MODE}-v1.log" 2>&1
