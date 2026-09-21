#!/bin/bash
#SBATCH --job-name=mdlm-ordinary
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist="watgpu308"

#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm-ordinary-%j.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm-ordinary-%j.err
#SBATCH --mail-user="n23zhang@uwaterloo.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

source activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

export HF_HUB_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
export TOKENIZERS_PARALLELISM=false

JOB_TAG="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ordinary_mdlm_${JOB_TAG}"
mkdir -p "$OUT_ROOT"

for SEED in 1 2 3; do
  RUN_DIR="$OUT_ROOT/seed_${SEED}"
  mkdir -p "$RUN_DIR"

  echo "=== ORDINARY MDLM SEED $SEED ==="
  python -u main.py \
    mode=sample_eval \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    data=openwebtext-split \
    model.length=1024 \
    sampling.predictor=ddpm_cache \
    sampling.steps=10000 \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=1 \
    backbone=hf_dit \
    seed="$SEED" \
    checkpointing.save_dir="$RUN_DIR" \
    hydra.run.dir="$RUN_DIR/hydra" \
    2>&1 | tee "$RUN_DIR/output.txt"
done

echo "Saved all three runs under: $OUT_ROOT"
