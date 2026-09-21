#!/bin/bash
#SBATCH --job-name=mdlm-hf-1k
#SBATCH --time=04:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist="watgpu308"
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm-hf-1k-%j.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm-hf-1k-%j.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL

# Generate ten adapter-free samples directly from the released Hugging Face
# MDLM checkpoint.  main.py only prints the final sample batch, so this script
# makes ten one-batch calls and preserves every sample in its own output.txt.

set -euo pipefail

source activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

HF_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
NUM_SAMPLES="${NUM_SAMPLES:-10}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}}"
OUT_ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/original_mdlm_hf_l1024_t1000_${RUN_TAG}"

if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SAMPLES must be a positive integer; found: $NUM_SAMPLES"
  exit 2
fi
if [[ -e "$OUT_ROOT" ]]; then
  echo "Output already exists: $OUT_ROOT"
  exit 2
fi
mkdir -p "$OUT_ROOT"

export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

for ((SEED = 1; SEED <= NUM_SAMPLES; SEED++)); do
  RUN_DIR="$OUT_ROOT/seed_${SEED}"
  mkdir -p "$RUN_DIR"

  echo "=== HUGGING FACE MDLM SEED $SEED/$NUM_SAMPLES ==="
  python -u main.py \
    mode=sample_eval \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    data=openwebtext-split \
    "data.cache_dir=$HF_CACHE" \
    model.length=1024 \
    sampling.predictor=ddpm_cache \
    sampling.steps=1000 \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=1 \
    eval.compute_generative_perplexity=false \
    backbone=hf_dit \
    seed="$SEED" \
    checkpointing.resume_from_ckpt=false \
    "checkpointing.save_dir=$RUN_DIR" \
    "hydra.run.dir=$RUN_DIR/hydra" \
    2>&1 | tee "$RUN_DIR/output.txt"
done

echo "Saved all $NUM_SAMPLES Hugging Face MDLM samples under: $OUT_ROOT"
echo "Score them with:"
echo "python scripts/score_text_files.py $OUT_ROOT/seed_*/output.txt --device cuda --output $OUT_ROOT/gpt2-generation-scores.tsv"
