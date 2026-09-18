#!/bin/bash
#SBATCH --job-name=ccf-train4-3k
#SBATCH --array=0-3%2
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

# Continue the matched four-arm CCF seed-1 run from update 1000 to update
# 3000.  Validation and durable checkpoints occur at 1500, 2000, 2500,
# and 3000.  Each arm resumes its full Lightning checkpoint, preserving the
# optimizer, scheduler, RNG, and loop state used by the original run.

set -euo pipefail
# Activate the Python environment before submitting; supply site options to sbatch.
CCF_CODE_ROOT="${CCF_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CCF_CACHE_ROOT:?Set CCF_CACHE_ROOT to your checkpoints/cache/runs directory}"
export CCF_CODE_ROOT CCF_CACHE_ROOT

cd "$CCF_CODE_ROOT"

CACHE_ROOT=${CCF_CACHE_ROOT}
SOURCE_ROOT="$CACHE_ROOT/runs/stale/four_arm_s001_k128_run-baseline-1k"
BACKBONE="$CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
HF_CACHE="$CACHE_ROOT/huggingface"
TRAIN_SEED=1

IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ ! "$IDX" =~ ^[0-3]$ ]]; then
  echo "Arm index must be 0, 1, 2, or 3; found: ${IDX:-<empty>}"
  exit 2
fi

if [[ -n "${RUN_TAG:-}" ]]; then
  JOB_TAG="$RUN_TAG"
elif [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_TAG="job${SLURM_ARRAY_JOB_ID}"
else
  JOB_TAG="manual-$(date -u +%Y%m%dT%H%M%SZ)"
fi
RUN_ROOT="$CACHE_ROOT/runs/four_arm_continue_s001_k128_to3k_${JOB_TAG}"
EXPORT_DIR="$RUN_ROOT/exported_adapters"

ARMS=(static_static fixed_dynamic dynamic_fixed dynamic_dynamic)
TOPOLOGIES=(fixed fixed dynamic dynamic)
FACTORS=(fixed dynamic fixed dynamic)
TOPOLOGY_WEIGHTS=(0.0 0.0 0.1 0.1)

ARM="${ARMS[$IDX]}"
TOPOLOGY="${TOPOLOGIES[$IDX]}"
FACTOR="${FACTORS[$IDX]}"
TOPOLOGY_WEIGHT="${TOPOLOGY_WEIGHTS[$IDX]}"
SOURCE_CHECKPOINT="$SOURCE_ROOT/$ARM/checkpoints/0-1000.ckpt"
RUN_DIR="$RUN_ROOT/$ARM"
ADAPTER="$EXPORT_DIR/$ARM.safetensors"
MANIFEST="$EXPORT_DIR/$ARM.manifest.json"
RESUME_CHECKPOINT="$SOURCE_CHECKPOINT"
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"

export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

for REQUIRED in "$SOURCE_CHECKPOINT" "$BACKBONE"; do
  if [[ ! -f "$REQUIRED" ]]; then
    echo "Required file missing: $REQUIRED"
    exit 2
  fi
done
if [[ -e "$ADAPTER" || -e "$MANIFEST" ]]; then
  echo "Refusing to overwrite a completed export for $ARM under $RUN_ROOT"
  exit 2
fi
if [[ -f "$RUN_DIR/checkpoints/last.ckpt" ]]; then
  RESUME_CHECKPOINT="$RUN_DIR/checkpoints/last.ckpt"
  echo "Restart $RESTART_COUNT: resuming the interrupted arm from $RESUME_CHECKPOINT"
elif [[ -d "$RUN_DIR" ]]; then
  echo "Restart $RESTART_COUNT: no continuation checkpoint exists; restarting this arm from update 1000"
fi
mkdir -p "$RUN_DIR" "$EXPORT_DIR"

SOURCE_CHECKPOINT_SHA256=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')
RESUME_CHECKPOINT_SHA256=$(sha256sum "$RESUME_CHECKPOINT" | awk '{print $1}')
BACKBONE_SHA256=$(sha256sum "$BACKBONE" | awk '{print $1}')
printf '%s  %s\n%s  %s\n%s  %s\n' \
  "$SOURCE_CHECKPOINT_SHA256" "$SOURCE_CHECKPOINT" \
  "$RESUME_CHECKPOINT_SHA256" "$RESUME_CHECKPOINT" \
  "$BACKBONE_SHA256" "$BACKBONE" \
  > "$RUN_DIR/resume-input-$RESTART_COUNT-sha256.txt"

echo "Arm: $ARM"
echo "Resuming: $RESUME_CHECKPOINT"
echo "Output: $RUN_DIR"

srun --ntasks=1 python -u main.py \
  mode=train \
  data=train_openwebtext_pinned \
  "data.cache_dir=$HF_CACHE" \
  seed="$TRAIN_SEED" \
  backbone=dit \
  parameterization=subs \
  model=contextual-forest-small \
  model.length=1024 \
  model.structured_decoder.top_k=128 \
  "model.structured_decoder.topology_mode=$TOPOLOGY" \
  "model.structured_decoder.factor_mode=$FACTOR" \
  model.structured_decoder.independent_mode=false \
  model.structured_decoder.training.backbone_mode=frozen \
  model.structured_decoder.training.require_pretrained_backbone=true \
  model.structured_decoder.training.strict_backbone_checkpoint=true \
  "model.structured_decoder.training.backbone_checkpoint=$BACKBONE" \
  model.structured_decoder.training.use_ema_backbone=false \
  model.structured_decoder.training.deterministic_backbone=true \
  model.structured_decoder.training.backbone_lr_multiplier=0.0 \
  model.structured_decoder.training.head_lr=0.0003 \
  model.structured_decoder.training.structured_nll_weight=1.0 \
  model.structured_decoder.training.factorized_aux_weight=0.0 \
  "model.structured_decoder.training.topology_weight=$TOPOLOGY_WEIGHT" \
  training.antithetic_sampling=true \
  training.importance_sampling=false \
  training.sampling_eps=0.001 \
  training.change_of_variables=false \
  training.ema=0 \
  optim.lr=0.0003 \
  optim.weight_decay=0 \
  lr_scheduler.num_warmup_steps=50 \
  trainer.max_steps=3000 \
  trainer.val_check_interval=500 \
  trainer.limit_val_batches=32 \
  trainer.num_sanity_val_steps=0 \
  trainer.devices=1 \
  trainer.precision=bf16 \
  loader.global_batch_size=4 \
  loader.eval_global_batch_size=4 \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  loader.num_workers=0 \
  strategy.find_unused_parameters=true \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=500 \
  checkpointing.resume_from_ckpt=true \
  "checkpointing.resume_ckpt_path=$RESUME_CHECKPOINT" \
  "checkpointing.save_dir=$RUN_DIR" \
  "hydra.run.dir=$RUN_DIR/hydra" \
  wandb=null

CHECKPOINT="$RUN_DIR/checkpoints/last.ckpt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training finished without the expected checkpoint: $CHECKPOINT"
  exit 2
fi
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')

python -u scripts/export_structured_adapter.py \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --expected-global-step 3000 \
  --output "$ADAPTER" \
  --manifest "$MANIFEST" \
  --control-identity "$ARM" \
  --topology-mode "$TOPOLOGY" \
  --factor-mode "$FACTOR" \
  --candidate-k 128 \
  --independent-mode false \
  --topology-weight "$TOPOLOGY_WEIGHT" \
  > "$EXPORT_DIR/$ARM.export-report.json"

sha256sum "$ADAPTER" > "$ADAPTER.sha256"
sha256sum "$MANIFEST" > "$MANIFEST.sha256"

echo "Completed continuation through update 3000: $ARM"
echo "Checkpoints: $RUN_DIR/checkpoints"
echo "Adapter: $ADAPTER"
