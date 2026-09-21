#!/bin/bash
#SBATCH --job-name=ccf-speed-verify
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=watgpu1008,watgpu1109,watgpu608,watgpu908
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-speed-verify-%j.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-speed-verify-%j.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS

set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export HF_HUB_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
export TOKENIZERS_PARALLELISM=false
RUN_ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_speed_verify_job${SLURM_JOB_ID:?}"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.run.lock"
flock -n 9 || exit 2
ATTEMPT=$(mktemp -d "$RUN_ROOT/attempt-${SLURM_RESTART_COUNT:-0}.XXXXXX")
git rev-parse HEAD > "$ATTEMPT/git-commit.txt"
printf '%s\n' "${CCF_LOCAL_FIX_COMMIT:?Provide local fix commit provenance}" > "$ATTEMPT/local-fix-commit.txt"
sha256sum structured_objective.py structured_utils.py evaluation/generation_harness.py \
  scripts/verify_ccf_sampling_optimization.py > "$ATTEMPT/code-sha256.txt"
srun --ntasks=1 python -u scripts/verify_ccf_sampling_optimization.py \
  --device cuda --benchmark --real-step-check > "$ATTEMPT/report.log"
echo "All exact token/RNG checks passed. Report: $ATTEMPT/report.log"
