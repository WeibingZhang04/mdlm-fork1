#!/bin/bash
# Array 0-4: 16 transitions; 5-9: 32. Within each: MDLM, SS, FD, DF, DD.
#SBATCH --job-name=ccf-lowsteps20
#SBATCH --array=0-9%2
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS
set -euo pipefail
# Activate the Python environment before submitting; supply site options to sbatch.
CCF_CODE_ROOT="${CCF_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CCF_CACHE_ROOT:?Set CCF_CACHE_ROOT to your checkpoints/cache/runs directory}"
export CCF_CODE_ROOT CCF_CACHE_ROOT
cd "${CCF_CODE_ROOT:?}"
export HF_HUB_CACHE=${CCF_CACHE_ROOT}/huggingface
export TOKENIZERS_PARALLELISM=false
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?}
CELL_INDEX=$((TASK_INDEX % 5))
if (( TASK_INDEX < 5 )); then
  SAMPLE_STEPS=16
else
  SAMPLE_STEPS=32
fi
srun --ntasks=1 python -u scripts/evaluate_ccf_selected_five.py \
  --suite basic_7k --index "$CELL_INDEX" --num-samples 20 \
  --sampling-steps "$SAMPLE_STEPS" \
  --output-root "${CCF_OUTPUT_ROOT:?}/steps_${SAMPLE_STEPS}"
