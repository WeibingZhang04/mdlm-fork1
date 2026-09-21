#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=/mnt/contextual-forest/staged-debug-20260907
readonly REPO="${EXPERIMENT}/code"
readonly PYTHON=/mnt/contextual-forest/venv/bin/python
readonly RELEASE=/mnt/contextual-forest/releases/contextual-forest-adapter-cf8b808-20260831T131607Z
readonly CHECKPOINT="${RELEASE}/inputs/mdlm-owt-backbone-schema-v2.pt"
readonly EXPECTATIONS="${RELEASE}/expectations/production-expectations-v2.json"
readonly EXPECTATIONS_SHA=a978c17befc2788b0b773eee7cd608ac0200379947a63a0ba8699a0b9818c956
export HF_HOME=/mnt/contextual-forest/hf-home
export HUGGINGFACE_HUB_CACHE=/mnt/contextual-forest/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "${REPO}"
mkdir -p "${EXPERIMENT}/logs"

"${PYTHON}" scripts/audit_staged_backbone_batch.py \
  --jsonl "${EXPERIMENT}/data/debug-fit.jsonl" \
  --backbone-checkpoint "${CHECKPOINT}" --expectations "${EXPECTATIONS}" \
  --expected-expectations-sha256 "${EXPECTATIONS_SHA}" \
  --output "${EXPERIMENT}/backbone-batch-audit.json" \
  --examples 3 --length 128 --mask-rate .5 --seed 1 --top-k 64 --device cuda \
  > "${EXPERIMENT}/logs/backbone-batch-audit.log" 2>&1

"${PYTHON}" scripts/run_staged_real_overfit.py \
  --train-jsonl "${EXPERIMENT}/data/debug-fit.jsonl" \
  --dev-jsonl "${EXPERIMENT}/data/dev.jsonl" \
  --backbone-checkpoint "${CHECKPOINT}" --expectations "${EXPECTATIONS}" \
  --expected-expectations-sha256 "${EXPECTATIONS_SHA}" \
  --output-dir "${EXPERIMENT}/overfit32-v1" \
  --examples 32 --dev-examples 32 --length 128 --mask-rate .5 \
  --steps 500 --batch-size 4 --learning-rate .003 --eval-every 50 \
  --rank 16 --directional-rank 8 --unary-rank 16 --top-k 64 \
  --factor-init-std .25 --warmup-steps 100 --seed 1 --device cuda \
  > "${EXPERIMENT}/logs/overfit32-v1.log" 2>&1

for arm in fixed_dynamic dynamic_dynamic; do
  adapter_dir="/mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/runs/export--${arm}--s001--k128/attempts/attempt-0001"
  adapter_sha="$(sha256sum "${adapter_dir}/adapter.safetensors" | cut -d' ' -f1)"
  manifest_sha="$(sha256sum "${adapter_dir}/adapter-manifest.json" | cut -d' ' -f1)"
  "${PYTHON}" scripts/run_staged_dependence_eval.py \
    --checkpoint "${CHECKPOINT}" --expectations "${EXPECTATIONS}" \
    --expectations-sha256 "${EXPECTATIONS_SHA}" \
    --adapter "${adapter_dir}/adapter.safetensors" --adapter-sha256 "${adapter_sha}" \
    --adapter-manifest "${adapter_dir}/adapter-manifest.json" \
    --adapter-manifest-sha256 "${manifest_sha}" \
    --input-jsonl "${EXPERIMENT}/data/dependence.jsonl" \
    --input-sha256 c9ff3569e04b2c5234553548b9e5bd35f55625df9c9039333e2594012a02aaf6 \
    --output-jsonl "${EXPERIMENT}/dependence-${arm}.jsonl" \
    --summary "${EXPERIMENT}/dependence-${arm}-summary.json" \
    --length 128 --max-per-split 32 --mask-rates .25,.5,.75,.9 \
    --lambda-grid 0,.25,.5,.75,1 --seed 20260907 --device cuda \
    > "${EXPERIMENT}/logs/dependence-${arm}.log" 2>&1
done
