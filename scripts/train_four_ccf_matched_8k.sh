#!/bin/bash
#SBATCH --job-name=ccf-train4-8k
#SBATCH --array=0-3%2
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

# Train the four CCF controls with paired settings.  The array admits at most
# two jobs at once, and each arm uses one GPU.  The released MDLM backbone is
# loaded identically and frozen in every arm; only the CCF adapter is trained.
#
# Default: training seed 1.  To run another complete four-arm seed:
#   sbatch --export=ALL,TRAIN_SEED=2,RUN_TAG=seed2 \
#     scripts/train_four_ccf_matched_8k.sh

set -euo pipefail

# Activate the Python environment before submitting this script.
CCF_CODE_ROOT="${CCF_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CCF_CACHE_ROOT:?Set CCF_CACHE_ROOT to your local checkpoints/cache/runs directory}"
export CCF_CODE_ROOT CCF_CACHE_ROOT
cd "$CCF_CODE_ROOT"

BACKBONE=${CCF_CACHE_ROOT}/checkpoints/mdlm-owt-backbone.pt
HF_CACHE=${CCF_CACHE_ROOT}/huggingface
TRAIN_SEED="${TRAIN_SEED:-1}"

if [[ ! "$TRAIN_SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_SEED must be a positive integer; found: $TRAIN_SEED"
  exit 2
fi
if [[ ! -f "$BACKBONE" ]]; then
  echo "Backbone not found: $BACKBONE"
  exit 2
fi

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
SEED_TAG=$(printf '%03d' "$TRAIN_SEED")
RUN_ROOT="${CCF_CACHE_ROOT}/runs/four_arm_train_s${SEED_TAG}_k128_${JOB_TAG}"

ARMS=(static_static fixed_dynamic dynamic_fixed dynamic_dynamic)
TOPOLOGIES=(fixed fixed dynamic dynamic)
FACTORS=(fixed dynamic fixed dynamic)
TOPOLOGY_WEIGHTS=(0.0 0.0 0.1 0.1)

ARM="${ARMS[$IDX]}"
TOPOLOGY="${TOPOLOGIES[$IDX]}"
FACTOR="${FACTORS[$IDX]}"
TOPOLOGY_WEIGHT="${TOPOLOGY_WEIGHTS[$IDX]}"
RUN_DIR="$RUN_ROOT/$ARM"

export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite an existing output for $ARM under $RUN_ROOT"
  exit 2
fi
mkdir -p "$RUN_DIR"

BACKBONE_SHA256=$(sha256sum "$BACKBONE" | awk '{print $1}')
echo "Arm: $ARM"
echo "Training seed: $TRAIN_SEED"
echo "Backbone: $BACKBONE"
echo "Backbone SHA256: $BACKBONE_SHA256"
echo "Run directory: $RUN_DIR"

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
  trainer.max_steps=8000 \
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
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=$RUN_DIR" \
  "hydra.run.dir=$RUN_DIR/hydra" \
  wandb=null

CHECKPOINT="$RUN_DIR/checkpoints/last.ckpt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training finished without the expected checkpoint: $CHECKPOINT"
  exit 2
fi
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')

echo "Completed 8,000 training steps: $ARM"
echo "Checkpoint: $CHECKPOINT"
echo "Checkpoint SHA256: $CHECKPOINT_SHA256"
