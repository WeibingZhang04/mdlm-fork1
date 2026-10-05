#!/bin/bash
# Copy of sample_diagose_all.sh for the approved preliminary 1024 comparison.
# Run with: bash scripts/sample_diagose_preliminary_30k.sh
# Frozen MDLM + counts + preserved global/contextual step-30000 checkpoints.
# Same metrics: generation, GPT-2-large scoring, then 32-row dev denoising.
#SBATCH --job-name=chain-prelim-30k
#SBATCH --partition=ALL
#SBATCH --array=0-3%2
#SBATCH --time=05:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=watgpu108

set -euo pipefail
if (( $# != 0 )); then
  printf 'Usage: bash scripts/sample_diagose_preliminary_30k.sh\n' >&2
  exit 2
fi
repo=/u401/n23zhang/crf-rework/mdlm-fork1
base=/u401/n23zhang/rework-data
training="$base/runs/draft-full-scale-training/train_2026_10_01_05_46_13_UTC"
snapshots="$base/runs/draft-full-scale-training/preliminary-checkpoints_2026_10_01_14_29_45_UTC"
dev_data="$base/data/draft-full-scale-training/owt-1024-2026-10-01_050424_UTC/dev.pt"
backbone="$base/checkpoints/mdlm-owt.pt"
counts="$training/counts/owt-counts.pt"

# Verify the copies, never point sampling at a live best.pt or last.pt.
for input in "$backbone" "$counts" "$dev_data" "$snapshots/manifest.json" \
             "$snapshots/global-step-30000.pt" "$snapshots/contextual-step-30000.pt"; do
  [[ -f "$input" && -r "$input" && -O "$input" ]] || { printf 'Missing or unowned input: %s\n' "$input" >&2; exit 1; }
done
printf '%s  %s\n' \
  f321ca07cb674fcf3d686a376e5ac46752fc8a91aaf3c1c42e738fa68c6ad518 "$snapshots/global-step-30000.pt" \
  e53c78a85aaeeeea39a77dd5673c4bd31dc93e83f9798a8d8f18a84cc4a08749 "$snapshots/contextual-step-30000.pt" \
  | sha256sum --check --status

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_dir="${CHAIN_SAMPLE_RUN_DIR:-$base/runs/draft-full-scale-training/preliminary_30k_$(date -u +%Y_%m_%d_%H_%M_%S_UTC)}"
  case "$run_dir" in "$base/runs/draft-full-scale-training/"*) ;; *) printf 'Use your draft training run directory.\n' >&2; exit 1 ;; esac
  mkdir -p "$(dirname "$run_dir")"
  mkdir "$run_dir" # Refuse to overwrite an existing experiment.
  cp "$0" "$run_dir/sample_diagose_preliminary_30k.sh"
  cp "$snapshots/manifest.json" "$run_dir/checkpoint-snapshots.json"
  sha256sum "$backbone" "$counts" "$dev_data" \
    "$snapshots/global-step-30000.pt" "$snapshots/contextual-step-30000.pt" \
    > "$run_dir/inputs.sha256"
  job="$(sbatch --parsable \
    --output="$run_dir/slurm_%A_%a.out" --error="$run_dir/slurm_%A_%a.err" \
    --export=ALL,CHAIN_SAMPLE_RUN_DIR="$run_dir" \
    "$run_dir/sample_diagose_preliminary_30k.sh")"
  printf '%s\n' "$job" > "$run_dir/submitted-job.txt"
  printf 'Preliminary sampling array: %s\nAll outputs: %s\n' "$job" "$run_dir"
  exit 0
fi

: "${CHAIN_SAMPLE_RUN_DIR:?Submit with bash scripts/sample_diagose_preliminary_30k.sh}"
sha256sum --check --status "$CHAIN_SAMPLE_RUN_DIR/inputs.sha256"
set +u
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm-crf
set -u
export REWORK_DATA="$base" HF_HOME="$base/hf-cache" PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$repo"

gpu_name="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
if [[ "$gpu_name" != 'NVIDIA RTX 6000 Ada Generation' ]]; then
  printf 'Expected an RTX 6000 Ada GPU; got %s\n' "$gpu_name" >&2
  exit 1
fi
case "$SLURM_ARRAY_TASK_ID" in
  0) name=baseline; mode_args=(--mode backbone) ;;
  1) name=count; mode_args=(--mode count --counts "$counts" --count-mode pmi --strength 0.1) ;;
  2) name=global-30k; mode_args=(--mode global --head "$snapshots/global-step-30000.pt") ;;
  3) name=contextual-30k; mode_args=(--mode contextual --head "$snapshots/contextual-step-30000.pt") ;;
  *) printf 'Unknown array task ID: %s\n' "$SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
common=(--backbone-checkpoint "$backbone" --cache-dir "$HF_HOME/hub"
  --length 1024 --k 64 --batch-size 1 --temperature 1 --sampling joint --inference dense)

# Matched random reveal schedule, as in the original script; not native ddpm_cache.
# Shared preliminary draw IDs 1000000..1000199, separate from validation draws.
printf 'Generating %s on %s: 200 samples, length 1024, 16 reveal steps.\n' "$name" "$gpu_name"
python -u scripts/evaluate_chain_crf.py "${common[@]}" "${mode_args[@]}" \
  --output "$CHAIN_SAMPLE_RUN_DIR/eval-$name" \
  --steps 16 --samples 200 --sample-offset 1000000 --warmup 1 --score-gpt2 \
  > "$CHAIN_SAMPLE_RUN_DIR/${name}_sample.log" 2>&1

# Preliminary diagnostics use the same first 32 dev rows and fixed mask draws.
# The reserved final-evaluation population remains unused.
printf 'Development denoising: %s, 32 examples.\n' "$name"
python -u scripts/evaluate_chain_crf.py "${common[@]}" "${mode_args[@]}" \
  --output "$CHAIN_SAMPLE_RUN_DIR/dev-$name" \
  --dev-data "$dev_data" --dev-examples 32 --denoise-only \
  > "$CHAIN_SAMPLE_RUN_DIR/${name}_denoise.log" 2>&1
