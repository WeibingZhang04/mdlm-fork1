#!/bin/bash
# Run with: bash scripts/sample_diagose_all.sh
# This submits the array and puts all outputs in one UTC-dated folder.
# Initial comparison: generate + GPT-2 score, then held-out denoising per model.
#SBATCH --job-name=chain-sampling
#SBATCH --array=0-4%2
#SBATCH --time=05:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --nodelist="watgpu108"
#SBATCH --mail-user="n23zhang"
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_dir="${CHAIN_SAMPLE_RUN_DIR:-/u401/n23zhang/rework-data/runs/$(date -u +%Y_%m_%d_%H_%M_%S_UTC)}"
  mkdir -p "$run_dir"
  sbatch \
    --output="$run_dir/slurm_%A_%a.out" \
    --error="$run_dir/slurm_%A_%a.err" \
    --export=ALL,CHAIN_SAMPLE_RUN_DIR="$run_dir" \
    "$(realpath "$0")"
  printf 'All sampling outputs: %s\n' "$run_dir"
  exit 0
fi