#!/bin/bash
#SBATCH --job-name=ccf-film-ckpt-ppl
#SBATCH --array=0-2%1
#SBATCH --time=02:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=watgpu1008,watgpu1109,watgpu608,watgpu908
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-film-ckpt-ppl-%A_%a.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-film-ckpt-ppl-%A_%a.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

# One paired sample at 2k, 4k and 6k. Reuse the existing 7k result.
# Submit with afterok dependencies on priority experiments #1 and #2.
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ ! "$IDX" =~ ^[0-2]$ ]]; then
  echo "Expected index 0, 1, or 2 (steps 2000, 4000, 6000)"
  exit 2
fi
STEPS=(2000 4000 6000)
STEP="${STEPS[$IDX]}"
CACHE_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache
ARCHIVE="$CACHE_ROOT/runs/ccf_dd_film_mlp128_s001_step7000_job1543361/training/checkpoints"
CHECKPOINT="$ARCHIVE/0-$STEP.ckpt"
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
HF_CACHE="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
RUN_ROOT="$CACHE_ROOT/runs/ccf_film_checkpoints_job${SLURM_ARRAY_JOB_ID:?Submit as Slurm array}/step$STEP"
test -f "$CHECKPOINT"
test -f "$BACKBONE"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.run.lock"
flock -n 9 || { echo "Another process owns $RUN_ROOT"; exit 2; }
if [[ -L "$RUN_ROOT/generation" && -s "$RUN_ROOT/generation/samples.jsonl" && -s "$RUN_ROOT/generation/summary.json" ]]; then
  echo "Step $STEP evaluation already complete"
  exit 0
fi
ATTEMPT_DIR=$(mktemp -d "$RUN_ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
ADAPTER="$ATTEMPT_DIR/dynamic_dynamic.safetensors"
MANIFEST="$ATTEMPT_DIR/dynamic_dynamic.manifest.json"
GEN_DIR="$ATTEMPT_DIR/generation"
git rev-parse HEAD > "$ATTEMPT_DIR/git-commit.txt"
git status --short > "$ATTEMPT_DIR/git-status.txt"
sha256sum diffusion.py models/structured_decoder.py structured_objective.py \
  structured_utils.py scripts/export_structured_adapter.py \
  scripts/run_generation_pilot.py scripts/evaluate_ccf_film_checkpoints.sh \
  evaluation/generation_harness.py evaluation/generation_metrics.py \
  > "$ATTEMPT_DIR/code-sha256.txt"
CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
BACKBONE_SHA=$(sha256sum "$BACKBONE" | awk '{print $1}')
printf '%s  %s\n%s  %s\n' "$CHECKPOINT_SHA" "$CHECKPOINT" \
  "$BACKBONE_SHA" "$BACKBONE" > "$ATTEMPT_DIR/input-sha256.txt"

srun --ntasks=1 python -u scripts/export_structured_adapter.py \
  --checkpoint "$CHECKPOINT" --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-global-step "$STEP" --output "$ADAPTER" --manifest "$MANIFEST" \
  --control-identity dynamic_dynamic --topology-mode dynamic --factor-mode dynamic \
  --candidate-k 128 --independent-mode false --topology-weight 0.1 \
  > "$ATTEMPT_DIR/export-report.json"
ADAPTER_SHA=$(sha256sum "$ADAPTER" | awk '{print $1}')
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')

srun --ntasks=1 python -u scripts/run_generation_pilot.py \
  --backbone-checkpoint "$BACKBONE" --backbone-sha256 "$BACKBONE_SHA" \
  --adapter "$ADAPTER" --adapter-sha256 "$ADAPTER_SHA" \
  --adapter-manifest "$MANIFEST" --adapter-manifest-sha256 "$MANIFEST_SHA" \
  --output-dir "$GEN_DIR" --num-samples 1 --sequence-length 1024 \
  --batch-size 1 --base-seed 91001 --modes structured_joint --nfe-budgets 1001 \
  --device cuda --model-config contextual-forest-small \
  --data-config train_openwebtext_pinned --allow-dirty \
  --reference-lm gpt2-large \
  --reference-lm-revision 32b71b12589c2f8d625668d2335a01cac3249519 \
  --reference-lm-device cuda --reference-lm-batch-size 1 \
  --reference-lm-max-length 1024 --reference-lm-dtype float32 \
  --override "data.cache_dir=$HF_CACHE" \
  --override model.structured_decoder.top_k=128 \
  --override model.structured_decoder.rank=16 \
  --override ++model.structured_decoder.factor_embedding_mode=shared \
  --override ++model.structured_decoder.factor_conditioner_hidden_dim=128 \
  --override model.structured_decoder.topology_mode=dynamic \
  --override model.structured_decoder.factor_mode=dynamic \
  --override model.structured_decoder.independent_mode=false \
  --override model.structured_decoder.training.topology_weight=0.1 \
  --override "checkpointing.save_dir=$RUN_ROOT"
ln -s "$GEN_DIR" "$RUN_ROOT/generation"
echo "Completed FiLM checkpoint $STEP: $RUN_ROOT/generation/samples.jsonl"
