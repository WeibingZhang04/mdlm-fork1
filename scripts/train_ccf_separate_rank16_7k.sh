#!/bin/bash
#SBATCH --job-name=ccf-separate-r16-7k
#SBATCH --array=0-1%2
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

# New capacity screen; same settings as separate rank8 except rank itself.
# When using speed fix v1, submit after successful GPU verification.
set -euo pipefail
# Activate the Python environment before submitting; supply site options to sbatch.
CCF_CODE_ROOT="${CCF_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CCF_CACHE_ROOT:?Set CCF_CACHE_ROOT to your checkpoints/cache/runs directory}"
export CCF_CODE_ROOT CCF_CACHE_ROOT
cd "${CCF_CODE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
export CCF_FACTOR_RANK=16
exec bash scripts/run_ccf_separate_7k.sh
