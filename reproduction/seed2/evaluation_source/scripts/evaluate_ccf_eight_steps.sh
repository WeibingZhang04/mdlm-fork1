#!/bin/bash
# Task 0: MDLM. Tasks 1–56: all CCF checkpoints, ascending training step.
# Submit with afterany dependency on the current 16/32-step sweep.
#SBATCH --job-name=ccf-eight-steps
#SBATCH --array=0-56%2
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --nodelist=watgpu508
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:?}"
export HF_HUB_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
export TOKENIZERS_PARALLELISM=false
srun --ntasks=1 python -u scripts/evaluate_ccf_selected_five.py \
  --suite full_checkpoint_sweep --index "${SLURM_ARRAY_TASK_ID:?}" \
  --num-samples 20 --sampling-steps 8 \
  --output-root "${CCF_OUTPUT_ROOT:?}/steps_8"
