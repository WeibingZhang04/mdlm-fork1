#!/bin/bash
#SBATCH --job-name=ccf-separate-r8-7k
#SBATCH --array=0-1%2
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=watgpu1008,watgpu1109,watgpu608,watgpu908
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-separate-r8-7k-%A_%a.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-separate-r8-7k-%A_%a.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

# Two fresh matched arms: 0=fixed_dynamic, 1=dynamic_dynamic.
set -euo pipefail
cd "${CCF_CODE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
exec bash scripts/run_ccf_separate_7k.sh
