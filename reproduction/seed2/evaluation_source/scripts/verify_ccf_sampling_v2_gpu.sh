#!/bin/bash
#SBATCH --job-name=ccf-speed-v2
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=watgpu1008,watgpu1109,watgpu608,watgpu908
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-speed-v2-%j.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-speed-v2-%j.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:?}"
export HF_HUB_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
RUN_ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_speed_v2_job${SLURM_JOB_ID:?}"
mkdir -p "$RUN_ROOT"
ATTEMPT=$(mktemp -d "$RUN_ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
printf '%s\n' "${CCF_LOCAL_FIX_COMMIT:?}" > "$ATTEMPT/local-fix-commit.txt"
sha256sum structured_utils.py structured_objective.py \
  scripts/verify_ccf_sampling_v2.py tests/test_ccf_sampling_optimization.py \
  > "$ATTEMPT/code-sha256.txt"
srun --ntasks=1 python -m unittest tests.test_ccf_sampling_optimization -v \
  > "$ATTEMPT/tests.log" 2>&1
srun --ntasks=1 python -u scripts/verify_ccf_sampling_v2.py --device cuda \
  --v1-utils /u401/n23zhang/mdlm_data/tree_mdlm_cache/code/ccf_sampling_speed_v1.ihSeMi/structured_utils.py \
  --real-step-check > "$ATTEMPT/report.json"
echo "V2 verification passed. Results: $ATTEMPT"
