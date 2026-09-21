#!/bin/bash
# Resource/node/mail settings are supplied explicitly by the submission tool.
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:?}"
export HF_HUB_CACHE="${CCF_CACHE_ROOT:?}/huggingface"
export TOKENIZERS_PARALLELISM=false
EXTRA_ARGS=()
if [[ "${CCF_VERIFY_ONLY:-0}" == 1 ]]; then
  EXTRA_ARGS+=(--verify-only)
fi
srun --ntasks=1 python -u scripts/evaluate_ccf_confirmation.py \
  --selection "${CCF_CODE_ROOT}/selection.json" \
  --selection-sha256 "${CCF_SELECTION_SHA:?}" \
  --index "${SLURM_ARRAY_TASK_ID:?}" \
  --output-root "${CCF_OUTPUT_ROOT:?}" "${EXTRA_ARGS[@]}"
