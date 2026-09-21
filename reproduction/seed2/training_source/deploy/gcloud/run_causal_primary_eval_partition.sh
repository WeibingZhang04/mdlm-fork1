#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PARTITION_INDEX NUM_PARTITIONS" >&2
  exit 2
fi

readonly partition_index="$1"
readonly num_partitions="$2"
readonly plan_dir='/mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-primary-from-d8671a0-v1'
readonly repo_dir='/mnt/contextual-forest/mdlm-causal-a574aca'
readonly python_bin='/mnt/contextual-forest/venv/bin/python'

if ! [[ "${partition_index}" =~ ^[0-9]+$ \
    && "${num_partitions}" =~ ^[1-9][0-9]*$ ]] \
    || (( partition_index >= num_partitions )); then
  echo 'partition index must be an integer in [0, NUM_PARTITIONS)' >&2
  exit 2
fi

cd "${repo_dir}"
mapfile -t eval_jobs < <(
  jq -r '.job_ids[] | select(startswith("eval--"))' \
    "${plan_dir}/compiled-plan.json"
)

selected=0
for index in "${!eval_jobs[@]}"; do
  if (( index % num_partitions != partition_index )); then
    continue
  fi
  selected=$((selected + 1))
  "${python_bin}" scripts/run_compiled_job.py \
    --plan-dir "${plan_dir}" \
    --job-id "${eval_jobs[index]}"
done

printf 'causal-primary eval partition %d/%d complete: %d jobs at %s\n' \
  "${partition_index}" "${num_partitions}" "${selected}" \
  "$(date --iso-8601=seconds)"
