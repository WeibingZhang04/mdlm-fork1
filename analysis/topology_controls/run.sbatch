#!/bin/bash
# DO NOT touch other people's files. DO NOT touch other people's jobs.
# Do not interfere with other people's processes.
#SBATCH --job-name=dd-topology
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ "$(id -un)" == n23zhang ]] || exit 2
cd /u401/n23zhang/clean_tree_mdlm/mdlm-fork1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" SLURM_EXPORT_ENV=ALL
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
case "${1:?mode}" in
 gate) srun --ntasks=1 python -u analysis/topology_controls/run.py gate --manifest "$2" ;;
 generate) srun --ntasks=1 python -u analysis/topology_controls/run.py generate --manifest "$2" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
 analyze) srun --ntasks=1 python -u analysis/topology_controls/report.py --manifest "$2" ;;
 *) exit 2 ;;
esac
