#!/bin/bash
#SBATCH --job-name=original-ccf-train
#SBATCH --array=0-7%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
STUDY="${1:?Pass the prepared study directory}"
srun --ntasks=1 python -u "$SLURM_SUBMIT_DIR/experiments/original_table/run.py" train --study "$STUDY" --index "${SLURM_ARRAY_TASK_ID:?}"
