#!/bin/bash
#SBATCH --job-name=ccf-fd-dd-6k
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
STUDY="${1:?Pass the prepared study directory}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
printf 'Job=%s task=%s host=%s python=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$(command -v python)"
export SLURM_EXPORT_ENV=ALL
srun --ntasks=1 python -u experiments/original_table/run_fd_dd_6k.py train --study "$STUDY" --index "${SLURM_ARRAY_TASK_ID:?}"
