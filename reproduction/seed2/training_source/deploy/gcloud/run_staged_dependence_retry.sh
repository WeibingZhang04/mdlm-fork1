#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/mnt/contextual-forest/staged-debug-20260907
readonly RELEASE=/mnt/contextual-forest/releases/contextual-forest-adapter-cf8b808-20260831T131607Z
export HF_HOME=/mnt/contextual-forest/hf-home
export HUGGINGFACE_HUB_CACHE=/mnt/contextual-forest/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "${ROOT}/eval-code-ae731e4"
for arm in fixed_dynamic dynamic_dynamic; do
  adapter_dir="/mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/runs/export--${arm}--s001--k128/attempts/attempt-0001"
  adapter_sha="$(sha256sum "${adapter_dir}/adapter.safetensors" | cut -d' ' -f1)"
  manifest_sha="$(sha256sum "${adapter_dir}/adapter-manifest.json" | cut -d' ' -f1)"
  /mnt/contextual-forest/venv/bin/python scripts/run_staged_dependence_eval.py \
    --checkpoint "${RELEASE}/inputs/mdlm-owt-backbone-schema-v2.pt" \
    --expectations "${RELEASE}/expectations/production-expectations-v2.json" \
    --expectations-sha256 a978c17befc2788b0b773eee7cd608ac0200379947a63a0ba8699a0b9818c956 \
    --adapter "${adapter_dir}/adapter.safetensors" --adapter-sha256 "${adapter_sha}" \
    --adapter-manifest "${adapter_dir}/adapter-manifest.json" \
    --adapter-manifest-sha256 "${manifest_sha}" \
    --input-jsonl "${ROOT}/data/dependence.jsonl" \
    --input-sha256 c9ff3569e04b2c5234553548b9e5bd35f55625df9c9039333e2594012a02aaf6 \
    --output-jsonl "${ROOT}/dependence-${arm}.jsonl" \
    --summary "${ROOT}/dependence-${arm}-summary.json" \
    --length 128 --max-per-split 32 --mask-rates .25,.5,.75,.9 \
    --lambda-grid 0,.25,.5,.75,1 --seed 20260907 --device cuda \
    > "${ROOT}/logs/dependence-${arm}-retry1.log" 2>&1
done
