#!/bin/bash
#SBATCH --job-name=original-ccf-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
STUDY="${1:?Pass the prepared study directory}"
SUITE="${2:?Pass pilot or confirmation}"
srun --ntasks=1 python -u "$STUDY/suite/run.py" evaluate --study "$STUDY" --suite "$SUITE" --index "${SLURM_ARRAY_TASK_ID:?}"
