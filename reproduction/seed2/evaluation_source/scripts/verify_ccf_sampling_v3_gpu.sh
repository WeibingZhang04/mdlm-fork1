#!/bin/bash
#SBATCH --job-name=ccf-v3-speed-test
#SBATCH --array=0-2%3
#SBATCH --time=00:45:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=watgpu108,watgpu1008,watgpu1109,watgpu608,watgpu908
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-v3-speed-%A_%a.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-v3-speed-%A_%a.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:?}"
export HF_HUB_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
export TOKENIZERS_PARALLELISM=false
MODES=(unchecked unchecked_batched unchecked_batched_roots_kruskal)
MODE="${MODES[${SLURM_ARRAY_TASK_ID:?}]}"
ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_v3_speed_job${SLURM_ARRAY_JOB_ID}/$MODE"
mkdir -p "$ROOT"
ATTEMPT=$(mktemp -d "$ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
printf '%s\n' "${CCF_LOCAL_TEST_COMMIT:?}" > "$ATTEMPT/local-test-commit.txt"
sha256sum structured_utils.py structured_objective.py models/structured_decoder.py diffusion.py \
  evaluation/generation_harness.py scripts/audit_ccf_sampling_v3.py \
  scripts/verify_ccf_sampling_v3_gpu.py > "$ATTEMPT/code-sha256.txt"
srun --ntasks=1 python -u scripts/verify_ccf_sampling_v3_gpu.py --mode "$MODE" --output-dir "$ATTEMPT/results"
