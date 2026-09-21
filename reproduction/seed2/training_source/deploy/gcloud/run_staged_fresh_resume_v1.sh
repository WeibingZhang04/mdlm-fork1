#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT_ROOT=/mnt/contextual-forest/staged-debug-20260907
readonly CODE_DIR="${1:?supply the unchanged training checkout}"
readonly RELEASE=/mnt/contextual-forest/releases/contextual-forest-adapter-cf8b808-20260831T131607Z
readonly SOURCE_RUN="${EXPERIMENT_ROOT}/fresh-pilot-v1"
readonly RESUMED_RUN="${EXPERIMENT_ROOT}/fresh-pilot-resume-v1"
[[ ! -e "${RESUMED_RUN}" ]]
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
  --output-dir "${RESUMED_RUN}" --override model.length=128 \
  --resume-checkpoint "${SOURCE_RUN}/checkpoints/step-000800.pt" \
  --expected-resume-sha256 5eb5fc41d8b62770b09b8de3fac9fec681f43ef91eaefd603ecc9c357376a4a4 \
  --train-examples 2048 --dev-examples 64 --length 128 \
  --steps 1000 --warmup-steps 100 --eval-every 100 \
  --batch-size 4 --backbone-batch-size 1 --learning-rate .0003 \
  --mask-rates .25 .5 .75 .9 --rank 16 --directional-rank 8 --unary-rank 16 \
  --top-k 64 --factor-init-std .25 --seed 1 --device cuda \
  > "${EXPERIMENT_ROOT}/logs/fresh-pilot-resume-v1.log" 2>&1 &
training_pid=$!

# The unchanged runner creates its output directory exclusively, then builds
# development caches before saving checkpoint 800. Copy only immutable earlier
# checkpoints so the new final ledger covers the complete development grid.
# All prefix hashes are checked against the rehydrated records by the sealer.
while [[ ! -d "${RESUMED_RUN}" ]]; do
  if ! kill -0 "${training_pid}" 2>/dev/null; then
    wait "${training_pid}"
    exit 1
  fi
  sleep 1
done
mkdir -p "${RESUMED_RUN}/checkpoints"
for step in 000000 000100 000200 000300 000400 000500 000600 000700; do
  cp -n "${SOURCE_RUN}/checkpoints/step-${step}.pt" "${RESUMED_RUN}/checkpoints/step-${step}.pt"
done
wait "${training_pid}"
