#!/bin/bash
#SBATCH --job-name=ccf-s1-k128
#SBATCH --array=0-3%2
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist="watgpu308"
#SBATCH --open-mode=append
#SBATCH --output=ccf-%A_%a-%j.out
#SBATCH --error=ccf-%A_%a-%j.err
#SBATCH --mail-user="n23zhang@uwaterloo.ca"
#SBATCH --mail-type=ALL


source activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

export CCF_BACKBONE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/checkpoints/mdlm-owt-backbone.pt
export CCF_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
export CCF_RUN_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/four_arm_s001_k128

ARMS=(static_static fixed_dynamic dynamic_fixed dynamic_dynamic)
TOPOLOGIES=(fixed fixed dynamic dynamic)
FACTORS=(fixed dynamic fixed dynamic)
WEIGHTS=(0.0 0.0 0.1 0.1)

IDX="$SLURM_ARRAY_TASK_ID"
ARM="${ARMS[$IDX]}"
RUN_DIR="$CCF_RUN_ROOT/$ARM"

if [[ -e "$RUN_DIR" ]]; then
  echo "Output already exists: $RUN_DIR"
  exit 2
fi

UNUSED=()
if [[ "$ARM" != "dynamic_dynamic" ]]; then
  UNUSED=(strategy.find_unused_parameters=true)
fi

srun python -u main.py \
  data=train_openwebtext_pinned data.cache_dir="$CCF_CACHE" \
  seed=1 model=contextual-forest-small \
  trainer.max_steps=1000 trainer.val_check_interval=500 \
  trainer.limit_val_batches=32 trainer.num_sanity_val_steps=0 \
  trainer.devices=1 trainer.precision=bf16 \
  loader.global_batch_size=4 loader.eval_global_batch_size=4 \
  loader.batch_size=4 loader.eval_batch_size=4 loader.num_workers=0 \
  model.structured_decoder.top_k=128 \
  model.structured_decoder.topology_mode="${TOPOLOGIES[$IDX]}" \
  model.structured_decoder.factor_mode="${FACTORS[$IDX]}" \
  model.structured_decoder.independent_mode=false \
  model.structured_decoder.training.topology_weight="${WEIGHTS[$IDX]}" \
  model.structured_decoder.training.backbone_checkpoint="$CCF_BACKBONE" \
  model.structured_decoder.training.use_ema_backbone=false \
  model.structured_decoder.training.head_lr=0.0003 \
  model.structured_decoder.training.factorized_aux_weight=0.0 \
  lr_scheduler.num_warmup_steps=50 training.ema=0 \
  eval.generate_samples=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=500 \
  checkpointing.resume_from_ckpt=false \
  checkpointing.save_dir="$RUN_DIR" hydra.run.dir="$RUN_DIR" \
  wandb=null "${UNUSED[@]}"