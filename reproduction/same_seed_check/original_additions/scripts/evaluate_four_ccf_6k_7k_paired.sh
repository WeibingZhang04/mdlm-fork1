#!/bin/bash
#SBATCH --job-name=ccf-paired6k7k
#SBATCH --time=20:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%2
#SBATCH --exclude=watgpu1008,watgpu1109,watgpu608
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-paired6k7k-%A_%a.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-paired6k7k-%A_%a.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

# Export and evaluate all four arms at updates 6000 and 7000.
# Tasks 0-3 use step 6000; tasks 4-7 use step 7000.
# Two paired samples per task by default; preserve per-seed GPT-2 scores.
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

CACHE_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache
ARCHIVE="$CACHE_ROOT/runs/four_arm_continue_s001_k128_6k_to10k_job1542883"
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
HF_CACHE="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ ! "$IDX" =~ ^[0-7]$ ]]; then
  echo "Provide an array index 0 through 7 (or a positional argument)."
  exit 2
fi
STEPS=(6000 7000)
STEP="${STEPS[$((IDX / 4))]}"
IDX=$((IDX % 4))
if [[ "$STEP" == 6000 ]]; then
  ARCHIVE="$CACHE_ROOT/runs/four_arm_continue_s001_k128_3k_to6k_job1542808"
fi
NUM_SAMPLES="${NUM_SAMPLES:-2}"
if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SAMPLES must be a positive integer."
  exit 2
fi
JOB_TAG="${SLURM_ARRAY_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$CACHE_ROOT/runs/ccf_paired6k7k_l1024_t1000_$JOB_TAG/step$STEP"

ARMS=(static_static fixed_dynamic dynamic_fixed dynamic_dynamic)
TOPOLOGIES=(fixed fixed dynamic dynamic)
FACTORS=(fixed dynamic fixed dynamic)
WEIGHTS=(0.0 0.0 0.1 0.1)
ARM="${ARMS[$IDX]}"
CHECKPOINT="$ARCHIVE/$ARM/checkpoints/0-$STEP.ckpt"
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
# Verify the exact requested update before generating.
srun --ntasks=1 python -u scripts/export_structured_adapter.py \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-global-step "$STEP" \
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
  --reference-lm gpt2-large \
  --reference-lm-revision 32b71b12589c2f8d625668d2335a01cac3249519 \
  --reference-lm-device cuda \
  --reference-lm-batch-size 1 \
  --reference-lm-max-length 1024 \
  --reference-lm-dtype float32 \
  --override "data.cache_dir=$HF_CACHE" \
  --override model.structured_decoder.top_k=128 \
  --override "model.structured_decoder.topology_mode=${TOPOLOGIES[$IDX]}" \
  --override "model.structured_decoder.factor_mode=${FACTORS[$IDX]}" \
  --override model.structured_decoder.independent_mode=false \
  --override "model.structured_decoder.training.topology_weight=${WEIGHTS[$IDX]}" \
  --override "checkpointing.save_dir=$RUN_ROOT"

echo "Completed $ARM at training step $STEP"
echo "Samples and per-seed GPT-2 scores: $GEN_DIR/samples.jsonl"
echo "Aggregate GPT-2 score and repetition: $GEN_DIR/summary.json"
