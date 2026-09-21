#!/bin/bash
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
export CCF_CAMPAIGN_ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_replication15k_20260921.wmjngfco"
export CCF_CACHE_ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache"
export HF_HUB_CACHE="$CCF_CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONPATH="$CCF_CAMPAIGN_ROOT/helpers:$CCF_CAMPAIGN_ROOT/code:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
cd "$CCF_CAMPAIGN_ROOT/code"
export CCF_LEGACY_STUDY=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_replication15k_20260921.wmjngfco/legacy_bf16_seed2/study_qv4ipesi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_replication15k_20260921.wmjngfco/legacy_bf16_seed2/study_qv4ipesi:/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_replication15k_20260921.wmjngfco/helpers:${PYTHONPATH:-}
cd /u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_replication15k_20260921.wmjngfco/legacy_bf16_seed2/study_qv4ipesi
python -u /u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_replication15k_20260921.wmjngfco/legacy_bf16_seed2/study_qv4ipesi/train_arm.py "$SLURM_ARRAY_TASK_ID"
