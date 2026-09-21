#!/bin/bash
#SBATCH --job-name=ccf-selected-five
#SBATCH --time=00:45:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --exclude=watgpu108,watgpu1008,watgpu1109,watgpu608,watgpu908
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:?}"
export HF_HUB_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
export TOKENIZERS_PARALLELISM=false
srun --ntasks=1 python -u scripts/evaluate_ccf_selected_five.py \
  --index "${SLURM_ARRAY_TASK_ID:?}" --output-root "${CCF_OUTPUT_ROOT:?}"
