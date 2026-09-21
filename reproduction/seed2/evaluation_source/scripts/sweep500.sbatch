#!/bin/bash
#SBATCH --job-name=ccf500
#SBATCH --time=00:45:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=watgpu108,watgpu1008,watgpu1109,watgpu608,watgpu908
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
cd "${CCF_CODE_ROOT:?}"
export HF_HUB_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
sha256sum --check sweep-source-sha256.txt
PHASE="${1:?}"
GROUP="${2:-baseline}"
if [[ "$PHASE" == verify && "${SLURM_ARRAY_TASK_ID:?}" == 0 ]]; then
  srun --ntasks=1 python -m pytest -q -p no:cacheprovider \
    tests/test_ccf_sampling_optimization.py tests/test_ccf_sampling_v3_experiments.py \
    tests/test_ccf_sampling_v4_experiments.py tests/test_generation_harness.py \
    tests/test_generation_metrics.py
fi
if [[ "$PHASE" == generate ]]; then
  srun --ntasks=1 python scripts/check_sweep500_gate.py "${CCF_OUTPUT_ROOT:?}"
fi
srun --ntasks=1 python -u scripts/sweep500.py "$PHASE" --group "$GROUP" \
  --index "${SLURM_ARRAY_TASK_ID:-0}" --output-root "${CCF_OUTPUT_ROOT:?}"
