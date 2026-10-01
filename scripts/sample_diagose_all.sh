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

: "${CHAIN_SAMPLE_RUN_DIR:?Submit with bash scripts/sample_diagose_all.sh}"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm-crf
export REWORK_DATA=/u401/n23zhang/rework-data
export HF_HOME="$REWORK_DATA/hf-cache"
export PYTHONNOUSERSITE=1
cd /u401/n23zhang/crf-rework/mdlm-fork1

gpu_name="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
if [[ "$gpu_name" != "NVIDIA RTX 6000 Ada Generation" ]]; then
  printf 'Expected an RTX 6000 Ada GPU; got %s\n' "$gpu_name" >&2
  exit 1
fi

case "$SLURM_ARRAY_TASK_ID" in
  0) name=base; mode_args=(--mode backbone) ;;
  1) name=count; mode_args=(--mode count --counts "$REWORK_DATA/checkpoints/owt-counts.pt" --count-mode pmi --strength 0.1) ;;
  2) name=contextual; mode_args=(--mode contextual --head "$REWORK_DATA/runs/contextual/best.pt" --k 64) ;;
  3) name=global; mode_args=(--mode global --head "$REWORK_DATA/runs/global/best.pt" --k 64) ;;
  4) name=independent; mode_args=(--mode independent --head "$REWORK_DATA/runs/independent-ada-108/best.pt" --k 64) ;;
  *) printf 'Unknown array task ID: %s\n' "$SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

output_dir="$CHAIN_SAMPLE_RUN_DIR/eval-$name"
log_file="$CHAIN_SAMPLE_RUN_DIR/${name}_sample.log"
printf 'Task %s: %s on %s; output=%s; log=%s\n' "$SLURM_ARRAY_TASK_ID" "$name" "$gpu_name" "$output_dir" "$log_file"

python scripts/evaluate_chain_crf.py \
  --backbone-checkpoint "$REWORK_DATA/checkpoints/mdlm-owt.pt" \
  --output "$output_dir" \
  "${mode_args[@]}" \
  --length 256 --steps 16 --samples 256 \
  --score-gpt2 \
  > "$log_file" 2>&1

# CHAIN_CRF.md: held-out denoising diagnostics. Uses the default 32 dev examples.
# Apply the documented contextual command to each model with its own mode/head.
diagnostic_dir="$CHAIN_SAMPLE_RUN_DIR/dev-$name"
diagnostic_log="$CHAIN_SAMPLE_RUN_DIR/${name}_denoise.log"
printf 'Denoising %s; output=%s; log=%s\n' "$name" "$diagnostic_dir" "$diagnostic_log"

python scripts/evaluate_chain_crf.py \
  --backbone-checkpoint "$REWORK_DATA/checkpoints/mdlm-owt.pt" \
  --output "$diagnostic_dir" \
  "${mode_args[@]}" \
  --dev-data "$REWORK_DATA/data/chain-owt/dev.pt" \
  --length 256 --denoise-only \
  > "$diagnostic_log" 2>&1
