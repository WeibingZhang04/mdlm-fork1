#!/bin/bash
#SBATCH --job-name=ccf-step500
#SBATCH --time=20:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%1
#SBATCH --nodelist=watgpu408
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-step500-%A_%a.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-step500-%A_%a.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

# Export best.ckpt only if it is step 500, generate, then score each arm.
# Submit: sbatch scripts/evaluate_four_ccf_step500.sh
# Matches the prior CCF run: 10 samples, length 1024, batch 1,
# base seed 91001, 1000 transitions plus the final cleanup allowance.
# Array concurrency is ONE GPU. No training or checkpoint overwriting.
# Keep the same Python code/scorer used for step 1000 for direct comparison.
set -euo pipefail
source activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

CACHE_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache
ARCHIVE="$CACHE_ROOT/runs/stale/four_arm_s001_k128_job1541029"
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
HF_CACHE="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ ! "$IDX" =~ ^[0-3]$ ]]; then
  echo "Provide an array index 0 through 3 (or a positional argument)."
  exit 2
fi
NUM_SAMPLES="${NUM_SAMPLES:-10}"
if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SAMPLES must be a positive integer."
  exit 2
fi
JOB_TAG="${SLURM_ARRAY_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$CACHE_ROOT/runs/ccf_step500_l1024_t1000_$JOB_TAG"

ARMS=(static_static fixed_dynamic dynamic_fixed dynamic_dynamic)
TOPOLOGIES=(fixed fixed dynamic dynamic)
FACTORS=(fixed dynamic fixed dynamic)
WEIGHTS=(0.0 0.0 0.1 0.1)
ARM="${ARMS[$IDX]}"
CHECKPOINT="$ARCHIVE/$ARM/checkpoints/best.ckpt"
ARM_DIR="$RUN_ROOT/$ARM"
ADAPTER="$ARM_DIR/$ARM.safetensors"
MANIFEST="$ARM_DIR/$ARM.manifest.json"
GEN_DIR="$ARM_DIR/generation"

for REQUIRED in "$CHECKPOINT" "$BACKBONE" scripts/export_structured_adapter.py scripts/run_generation_pilot.py scripts/score_text_files.py; do
  if [[ ! -f "$REQUIRED" ]]; then
    echo "Required file missing: $REQUIRED"
    exit 2
  fi
done
mkdir -p "$RUN_ROOT"
# Exclusive creation prevents accidental reuse, including partially failed runs.
mkdir "$ARM_DIR"

echo "Arm: $ARM"
echo "Input checkpoint: $CHECKPOINT"
echo "Output: $ARM_DIR"

# Record the bytes available now; this is not a historical authenticity claim.
CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
BACKBONE_SHA=$(sha256sum "$BACKBONE" | awk '{print $1}')
printf '%s  %s\n' "$CHECKPOINT_SHA" "$CHECKPOINT" "$BACKBONE_SHA" "$BACKBONE" > "$ARM_DIR/input-sha256.txt"
git rev-parse HEAD > "$ARM_DIR/git-commit.txt"
git status --short > "$ARM_DIR/git-status.txt"
sha256sum scripts/export_structured_adapter.py scripts/run_generation_pilot.py \
  scripts/score_text_files.py evaluation/generation_harness.py \
  evaluation/generation_metrics.py diffusion.py models/dit.py \
  models/structured_decoder.py structured_objective.py structured_utils.py \
  > "$ARM_DIR/code-sha256.txt"

# The exporter verifies the stored global_step and checkpoint configuration.
# Fail instead of substituting last.ckpt if best.ckpt is not step 500.
srun --ntasks=1 python -u scripts/export_structured_adapter.py \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-global-step 500 \
  --output "$ADAPTER" \
  --manifest "$MANIFEST" \
  --control-identity "$ARM" \
  --topology-mode "${TOPOLOGIES[$IDX]}" \
  --factor-mode "${FACTORS[$IDX]}" \
  --candidate-k 128 \
  --independent-mode false \
  --topology-weight "${WEIGHTS[$IDX]}" \
  > "$ARM_DIR/export-report.json"

ADAPTER_SHA=$(sha256sum "$ADAPTER" | awk '{print $1}')
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
printf '%s  %s\n' "$ADAPTER_SHA" "$ADAPTER" "$MANIFEST_SHA" "$MANIFEST" > "$ARM_DIR/adapter-sha256.txt"

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
  --modes structured_joint \
  --nfe-budgets 1001 \
  --device cuda \
  --model-config contextual-forest-small \
  --data-config train_openwebtext_pinned \
  --allow-dirty \
  --override "data.cache_dir=$HF_CACHE" \
  --override model.structured_decoder.top_k=128 \
  --override "model.structured_decoder.topology_mode=${TOPOLOGIES[$IDX]}" \
  --override "model.structured_decoder.factor_mode=${FACTORS[$IDX]}" \
  --override model.structured_decoder.independent_mode=false \
  --override "model.structured_decoder.training.topology_weight=${WEIGHTS[$IDX]}" \
  --override "checkpointing.save_dir=$RUN_ROOT"

# Generation exits before GPT2 loads, freeing its GPU memory.
# Score every generated record directly, using the previous scorer/settings.
# The table path identifies the arm; its condition field may say structured_joint.
srun --ntasks=1 python -u scripts/score_text_files.py "$GEN_DIR/samples.jsonl" \
  --model gpt2-large \
  --revision 32b71b12589c2f8d625668d2335a01cac3249519 \
  --device cuda --batch-size 1 --max-length 1024 \
  --output "$ARM_DIR/gpt2-scores.tsv"

echo "Completed $ARM at training step 500"
echo "Samples: $GEN_DIR/samples.jsonl"
echo "Scores: $ARM_DIR/gpt2-scores.tsv"

