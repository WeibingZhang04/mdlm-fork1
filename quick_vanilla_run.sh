
set -eo pipefail
SECONDS=0

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm

REPRO_ROOT=/u401/n23zhang/rework-data/original_reproduction
REPRO_CODE=/u401/n23zhang/bd3lm_mdlm_original_reproduction/mdlm
REPRO_RUN="$REPRO_ROOT/runs/mdlm_$(date -u +%Y_%m_%d_%H_%M_%S_UTC)_$$"

export HF_HOME="$REPRO_ROOT/cache/mdlm/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$REPRO_ROOT/cache/mdlm/torch"

mkdir -p "$REPRO_RUN" "$REPRO_ROOT/data/mdlm-official"
cd "$REPRO_CODE"

printf 'Code:         %s\n' "$PWD"
printf 'Job ID:       %s\n' "${SLURM_JOB_ID:-unset}"
printf 'Visible GPUs: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'Model cache:  %s\n' "$HF_HUB_CACHE"
printf 'Data:         %s\n' "$REPRO_ROOT/data/mdlm-official"
printf 'Log:          %s\n' "$REPRO_RUN/run.log"

date -u > timestart.txt

python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  parameterization=subs \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=32 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=20 \
  sampling.semi_ar=False \
  backbone=hf_dit \
  hydra.run.dir="$REPRO_RUN" \
  checkpointing.save_dir="$REPRO_RUN" \
  > "$REPRO_RUN/run.log" 2>&1

date -u > timeend.txt
elapsed_seconds=$SECONDS
printf 'Total elapsed: %s seconds\n' "$elapsed_seconds"
