#!/bin/bash
#SBATCH --job-name=ccf-lrdecay-10k-ppl
#SBATCH --array=0-3%2
#SBATCH --time=03:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=watgpu1008,watgpu1109,watgpu608,watgpu908
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-lrdecay-10k-ppl-%A_%a.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-lrdecay-10k-ppl-%A_%a.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

# Four LR-decay arms; seeds 91001 and 91002, one sequence per seed.
# Training checkpoints are read-only. Each job/arm/attempt has isolated outputs.
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ ! "$IDX" =~ ^[0-3]$ ]]; then
  echo "Expected arm index 0 through 3"
  exit 2
fi
ARMS=(static_static fixed_dynamic dynamic_fixed dynamic_dynamic)
TOPOLOGIES=(fixed fixed dynamic dynamic)
FACTORS=(fixed dynamic fixed dynamic)
WEIGHTS=(0.0 0.0 0.1 0.1)
ARM="${ARMS[$IDX]}"
CACHE_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache
ARCHIVE="$CACHE_ROOT/runs/four_arm_continue_s001_k128_shared_r16_6k_to10k_lrdecay_job1543563"
CHECKPOINT="$ARCHIVE/$ARM/checkpoints/0-10000.ckpt"
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
HF_CACHE="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
RUN_ROOT="$CACHE_ROOT/runs/ccf_lrdecay_10k_eval_job${SLURM_ARRAY_JOB_ID:?Submit as Slurm array}/$ARM"
test -f "$CHECKPOINT"
test -f "$BACKBONE"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.run.lock"
flock -n 9 || { echo "Another process owns $RUN_ROOT"; exit 2; }
if [[ -L "$RUN_ROOT/generation" && -s "$RUN_ROOT/generation/samples.jsonl" && -s "$RUN_ROOT/generation/summary.json" ]]; then
  echo "$ARM evaluation already complete"
  exit 0
fi
ATTEMPT_DIR=$(mktemp -d "$RUN_ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
ADAPTER="$ATTEMPT_DIR/$ARM.safetensors"
MANIFEST="$ATTEMPT_DIR/$ARM.manifest.json"
GEN_DIR="$ATTEMPT_DIR/generation"
echo "Arm: $ARM; checkpoint: $CHECKPOINT; output: $RUN_ROOT"
git rev-parse HEAD > "$ATTEMPT_DIR/git-commit.txt"
git status --short > "$ATTEMPT_DIR/git-status.txt"
sha256sum diffusion.py models/structured_decoder.py structured_objective.py \
  structured_utils.py scripts/export_structured_adapter.py \
  scripts/run_generation_pilot.py scripts/evaluate_ccf_lr_decay_10k.sh \
  evaluation/generation_harness.py evaluation/generation_metrics.py \
  > "$ATTEMPT_DIR/code-sha256.txt"
CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
BACKBONE_SHA=$(sha256sum "$BACKBONE" | awk '{print $1}')
printf '%s  %s\n%s  %s\n' "$CHECKPOINT_SHA" "$CHECKPOINT" \
  "$BACKBONE_SHA" "$BACKBONE" > "$ATTEMPT_DIR/input-sha256.txt"

srun --ntasks=1 python -u scripts/export_structured_adapter.py \
  --checkpoint "$CHECKPOINT" --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-global-step 10000 --output "$ADAPTER" --manifest "$MANIFEST" \
  --control-identity "$ARM" --topology-mode "${TOPOLOGIES[$IDX]}" \
  --factor-mode "${FACTORS[$IDX]}" --candidate-k 128 \
  --independent-mode false --topology-weight "${WEIGHTS[$IDX]}" \
  > "$ATTEMPT_DIR/export-report.json"
ADAPTER_SHA=$(sha256sum "$ADAPTER" | awk '{print $1}')
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')

srun --ntasks=1 python -u scripts/run_generation_pilot.py \
  --backbone-checkpoint "$BACKBONE" --backbone-sha256 "$BACKBONE_SHA" \
  --adapter "$ADAPTER" --adapter-sha256 "$ADAPTER_SHA" \
  --adapter-manifest "$MANIFEST" --adapter-manifest-sha256 "$MANIFEST_SHA" \
  --output-dir "$GEN_DIR" --num-samples 2 --sequence-length 1024 \
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
  --override ++model.structured_decoder.factor_conditioner_hidden_dim=0 \
  --override "model.structured_decoder.topology_mode=${TOPOLOGIES[$IDX]}" \
  --override "model.structured_decoder.factor_mode=${FACTORS[$IDX]}" \
  --override model.structured_decoder.independent_mode=false \
  --override "model.structured_decoder.training.topology_weight=${WEIGHTS[$IDX]}" \
  --override "checkpointing.save_dir=$RUN_ROOT"
ln -s "$GEN_DIR" "$RUN_ROOT/generation"
echo "Completed $ARM: $RUN_ROOT/generation/samples.jsonl"
