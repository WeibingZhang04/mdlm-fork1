#!/bin/bash
# Evaluate the released, adapter-free MDLM on OpenWebText using the repo's
# diffusion-NELBO/ELBO-bound metric (the kind of PPL reported in Table 2).
# This does NOT score generated text files.

# Usage:
#   bash scripts/eval_original_mdlm_owt_ppl.sh [output-directory]

set -euo pipefail

#source activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

HF_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
OUTPUT_DIR="${1:-/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/original_mdlm_owt_ppl}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output already exists: $OUTPUT_DIR"
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

python -u main.py \
  mode=ppl_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  "data.cache_dir=$HF_CACHE" \
  model=small \
  model.length=1024 \
  parameterization=subs \
  backbone=hf_dit \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  trainer.devices=1 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=$OUTPUT_DIR" \
  "hydra.run.dir=$OUTPUT_DIR/hydra" \
  wandb=null \
  2>&1 | tee "$OUTPUT_DIR/output.txt"

echo "Full Table-2-style evaluation log: $OUTPUT_DIR/output.txt"
