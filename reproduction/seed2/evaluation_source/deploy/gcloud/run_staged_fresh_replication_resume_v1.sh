#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT_ROOT=/mnt/contextual-forest/staged-debug-20260907
readonly CODE_DIR="${1:?supply the unchanged training checkout}"
readonly RESUME_STEP="${2:?supply the verified saved update}"
readonly RESUME_SHA="${3:?supply the saved checkpoint SHA256}"
readonly QUEUE_MODE="${4:-both}"
case "${QUEUE_MODE}" in both|seed2-only) ;; *) exit 2 ;; esac
[[ "${RESUME_STEP}" =~ ^[0-9]+$ && "${RESUME_SHA}" =~ ^[0-9a-f]{64}$ ]]
(( RESUME_STEP >= 0 && RESUME_STEP <= 1000 && RESUME_STEP % 100 == 0 ))
readonly RELEASE=/mnt/contextual-forest/releases/contextual-forest-adapter-cf8b808-20260831T131607Z
readonly SOURCE_RUN="${EXPERIMENT_ROOT}/fresh-replication-seed2-retry1-v1"
readonly RESUMED_RUN="${EXPERIMENT_ROOT}/fresh-replication-seed2-resume-v1"
[[ ! -e "${RESUMED_RUN}" && ! -e "${EXPERIMENT_ROOT}/fresh-replication-seed3-retry1-v1" ]]
printf -v resume_filename 'step-%06d.pt' "${RESUME_STEP}"
cd "${CODE_DIR}"
export HF_HOME=/mnt/contextual-forest/hf-home
export HUGGINGFACE_HUB_CACHE=/mnt/contextual-forest/huggingface/hub
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
common=(
  --train-jsonl "${EXPERIMENT_ROOT}/fresh-data-v1/train.jsonl"
  --dev-jsonl "${EXPERIMENT_ROOT}/fresh-data-v1/dev.jsonl"
  --expected-train-sha256 1d387c4b4e90e2a7a994eb70521335c45ef2e1574aae8de9108dc944f1ee7dd6
  --expected-dev-sha256 68390ef4a9e2f5be6df70657ea70e186957ae68450e4f680392eff84fc36802a
  --backbone-checkpoint "${RELEASE}/inputs/mdlm-owt-backbone-schema-v2.pt"
  --expectations "${RELEASE}/expectations/production-expectations-v2.json"
  --expected-expectations-sha256 a978c17befc2788b0b773eee7cd608ac0200379947a63a0ba8699a0b9818c956
  --override model.length=128 --train-examples 2048 --dev-examples 64 --length 128
  --steps 1000 --warmup-steps 100 --eval-every 100 --batch-size 4 --backbone-batch-size 1
  --learning-rate .0003 --mask-rates .25 .5 .75 .9 --rank 16 --directional-rank 8
  --unary-rank 16 --top-k 64 --factor-init-std .25 --device cuda
)
/mnt/contextual-forest/venv/bin/python scripts/run_staged_fresh_training.py "${common[@]}" \
  --seed 2 --output-dir "${RESUMED_RUN}" \
  --resume-checkpoint "${SOURCE_RUN}/checkpoints/${resume_filename}" \
  --expected-resume-sha256 "${RESUME_SHA}" \
  > "${EXPERIMENT_ROOT}/logs/fresh-replication-seed2-resume-v1.log" 2>&1 &
training_pid=$!
while [[ ! -d "${RESUMED_RUN}" ]]; do
  if ! kill -0 "${training_pid}" 2>/dev/null; then
    wait "${training_pid}"
    exit 1
  fi
  sleep 1
done
mkdir -p "${RESUMED_RUN}/checkpoints"
for ((step=0; step<RESUME_STEP; step+=100)); do
  printf -v prefix_filename 'step-%06d.pt' "${step}"
  cp -n "${SOURCE_RUN}/checkpoints/${prefix_filename}" "${RESUMED_RUN}/checkpoints/${prefix_filename}"
done
wait "${training_pid}"
[[ "${QUEUE_MODE}" == both ]] || exit 0
/mnt/contextual-forest/venv/bin/python scripts/run_staged_fresh_training.py "${common[@]}" \
  --seed 3 --output-dir "${EXPERIMENT_ROOT}/fresh-replication-seed3-retry1-v1" \
  > "${EXPERIMENT_ROOT}/logs/fresh-replication-seed3-retry1-v1.log" 2>&1
