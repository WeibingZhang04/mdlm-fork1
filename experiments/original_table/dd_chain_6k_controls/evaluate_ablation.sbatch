#!/bin/bash
#SBATCH --job-name=dd-chain-off
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
STUDY="${1:?}"
cd /u401/n23zhang/clean_tree_mdlm/mdlm-fork1
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
printf 'Job=%s task=%s host=%s python=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$(command -v python)"
export SLURM_EXPORT_ENV=ALL
srun --ntasks=1 python -u "$STUDY/ablate_chain.py" --study "$STUDY" --index "$SLURM_ARRAY_TASK_ID"
