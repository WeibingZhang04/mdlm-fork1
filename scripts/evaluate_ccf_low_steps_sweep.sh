#!/bin/bash
# Each adjacent task pair evaluates one checkpoint at 16 then 32 transitions.
#SBATCH --job-name=ccf-lowsteps-sweep
#SBATCH --array=0-103%2
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
CELL_INDEX=$((TASK_INDEX / 2))
SAMPLE_STEPS=$((16 * (1 + TASK_INDEX % 2)))
srun --ntasks=1 python -u scripts/evaluate_ccf_selected_five.py \
  --suite low_steps_sweep --index "$CELL_INDEX" --num-samples 20 \
  --sampling-steps "$SAMPLE_STEPS" \
  --output-root "${CCF_OUTPUT_ROOT:?}/steps_${SAMPLE_STEPS}"
