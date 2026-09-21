#!/bin/bash
#SBATCH --job-name=mdlm-runner-control
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm-runner-control-%j.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm-runner-control-%j.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

# Submit: sbatch scripts/evaluate_mdlm_factorized_runner.sh
# One GPU, one condition. No training and no checkpoint export.
# Same runner, converted backbone, seeds, length and transition budget as CCF.
# The existing runner requires an adapter at initialization; factorized mode
# bypasses its predictions entirely. Do not set independent_mode=true: that
# would change the adapter identity and is a different control.
# This runner uses uncached ddpm, not original main.py's ddpm_cache.
# No node restriction; add sbatch --nodelist=watgpu408 if needed.
set -euo pipefail
source activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

CACHE_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
HF_CACHE="$CACHE_ROOT/huggingface"
ADAPTER_DIR="$CACHE_ROOT/runs/stale/four_arm_s001_k128_job1541029/exported_adapters"
ADAPTER="$ADAPTER_DIR/static_static.safetensors"
MANIFEST="$ADAPTER_DIR/static_static.manifest.json"
ADAPTER_SHA=26ea46f17db3873b3607dc1f026506f08ef148c3e9f61531e2b4e9eb7b58b4e1
MANIFEST_SHA=39253ccbba30cd7cf7c456b4977920016cc22c07c9241396c6fcf50a54a62450
NUM_SAMPLES="${NUM_SAMPLES:-10}"
if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SAMPLES must be a positive integer."
  exit 2
fi
JOB_TAG="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$CACHE_ROOT/runs/mdlm_factorized_runner_l1024_t1000_$JOB_TAG"
GEN_DIR="$RUN_ROOT/generation"
export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

for REQUIRED in "$BACKBONE" "$ADAPTER" "$MANIFEST" scripts/run_generation_pilot.py scripts/score_text_files.py; do
  if [[ ! -f "$REQUIRED" ]]; then
    echo "Required file missing: $REQUIRED"
    exit 2
  fi
done
printf '%s  %s\n' "$ADAPTER_SHA" "$ADAPTER" "$MANIFEST_SHA" "$MANIFEST" | sha256sum --check
BACKBONE_SHA=$(sha256sum "$BACKBONE" | awk '{print $1}')
# Exclusive creation: never overwrite an earlier run, even a partial one.
mkdir "$RUN_ROOT"
printf '%s  %s\n' "$BACKBONE_SHA" "$BACKBONE" "$ADAPTER_SHA" "$ADAPTER" "$MANIFEST_SHA" "$MANIFEST" > "$RUN_ROOT/input-sha256.txt"
git rev-parse HEAD > "$RUN_ROOT/git-commit.txt"
git status --short > "$RUN_ROOT/git-status.txt"
sha256sum scripts/run_generation_pilot.py scripts/score_text_files.py \
  evaluation/generation_harness.py evaluation/generation_metrics.py \
  diffusion.py models/dit.py noise_schedule.py > "$RUN_ROOT/code-sha256.txt"

echo "Condition: MDLM-only predictions inside the CCF runner (factorized)"
echo "Samples: $NUM_SAMPLES; length: 1024; transitions: 1000; batch: 1"
echo "Output: $RUN_ROOT"

srun --ntasks=1 python -u scripts/run_generation_pilot.py \
  --backbone-checkpoint "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA" \
  --adapter "$ADAPTER" \
  --adapter-sha256 "$ADAPTER_SHA" \
  --adapter-manifest "$MANIFEST" \
  --adapter-manifest-sha256 "$MANIFEST_SHA" \
  --output-dir "$GEN_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --sequence-length 1024 \
  --batch-size 1 \
  --base-seed 91001 \
  --modes factorized \
  --nfe-budgets 1001 \
  --device cuda \
  --model-config contextual-forest-small \
  --data-config train_openwebtext_pinned \
  --allow-dirty \
  --override "data.cache_dir=$HF_CACHE" \
  --override model.structured_decoder.top_k=128 \
  --override model.structured_decoder.topology_mode=fixed \
  --override model.structured_decoder.factor_mode=fixed \
  --override model.structured_decoder.independent_mode=false \
  --override model.structured_decoder.training.topology_weight=0.0 \
  --override "checkpointing.save_dir=$RUN_ROOT"

# The generation process has exited before GPT2 is loaded.
# Preserve the existing scorer and its EOS policy for comparison.
srun --ntasks=1 python -u scripts/score_text_files.py "$GEN_DIR/samples.jsonl" \
  --model gpt2-large \
  --revision 32b71b12589c2f8d625668d2335a01cac3249519 \
  --device cuda --batch-size 1 --max-length 1024 --dtype float32 \
  --output "$RUN_ROOT/gpt2-scores.tsv"

echo "Completed factorized-runner control"
echo "Samples: $GEN_DIR/samples.jsonl"
echo "Scores: $RUN_ROOT/gpt2-scores.tsv"

