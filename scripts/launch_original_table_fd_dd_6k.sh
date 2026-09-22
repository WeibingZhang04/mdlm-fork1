#!/usr/bin/env bash
# Submit Basic FD/DD through 6k, then 100-sample evaluation at 8/16/32 steps.
# All executable experiment code stays in this checkout. Only data goes in STUDY.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [[ "$(git branch --show-current)" != original_table_base_crf-recovery ]]; then
  echo "Switch to original_table_base_crf-recovery before launching." >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --prepare-only ) ]]; then
  echo "Usage: bash scripts/launch_original_table_fd_dd_6k.sh [--prepare-only]" >&2
  exit 2
fi

: "${CCF_CACHE_ROOT:=/u401/n23zhang/mdlm_data/tree_mdlm_cache}"
: "${CCF_MAIL_USER:=n23zhang@uwaterloo.ca}"
# Both 1558233 tasks failed at Slurm step launch on this node.
# Override with another list, or explicitly set empty once the cluster is fixed.
: "${CCF_EXCLUDE_NODES=watgpu608}"
# Preparation uses only the standard library. Home is noexec on the login node,
# so do not use the home-installed Conda interpreter for this metadata operation.
: "${CCF_PREPARE_PYTHON:=/usr/bin/python3}"
if [[ -z "${CCF_STUDY:-}" ]]; then
  mkdir -p "$CCF_CACHE_ROOT/runs"
  CCF_STUDY="$(mktemp -d "$CCF_CACHE_ROOT/runs/fd_dd_exact_seed1_6k.XXXXXX")/study"
fi
export CCF_CACHE_ROOT CCF_MAIL_USER CCF_STUDY
"$CCF_PREPARE_PYTHON" "$REPO/experiments/original_table/run_fd_dd_6k.py" prepare \
  --cache "$CCF_CACHE_ROOT" --study "$CCF_STUDY"
if [[ "${1:-}" == --prepare-only ]]; then
  printf 'Prepared only; no jobs submitted. Study: %s\n' "$CCF_STUDY"
  exit 0
fi

SBATCH_OPTIONS=(--parsable --chdir="$REPO" --export=ALL
  --mail-user="$CCF_MAIL_USER" --mail-type=ALL)
if [[ -n "$CCF_EXCLUDE_NODES" ]]; then
  SBATCH_OPTIONS+=(--exclude="$CCF_EXCLUDE_NODES")
fi
TRAIN_JOB="$(sbatch "${SBATCH_OPTIONS[@]}" \
  --output="$CCF_STUDY/logs/train-%A_%a.out" \
  --error="$CCF_STUDY/logs/train-%A_%a.err" \
  "$REPO/experiments/original_table/train_fd_dd_6k.sbatch" "$CCF_STUDY")"
TRAIN_JOB="${TRAIN_JOB%%;*}"
printf '%s\n' "$TRAIN_JOB" > "$CCF_STUDY/train-job-id.txt"
printf 'Study: %s\nTraining job: %s\n' "$CCF_STUDY" "$TRAIN_JOB"

EVAL_JOB="$(sbatch "${SBATCH_OPTIONS[@]}" --array=0-5%2 --dependency="afterok:$TRAIN_JOB" \
  --output="$CCF_STUDY/logs/eval-%A_%a.out" \
  --error="$CCF_STUDY/logs/eval-%A_%a.err" \
  "$REPO/experiments/original_table/evaluate.sbatch" "$CCF_STUDY" confirmation)"
EVAL_JOB="${EVAL_JOB%%;*}"
printf '%s\n' "$EVAL_JOB" > "$CCF_STUDY/eval-job-id.txt"
printf 'Evaluation job: %s\n' "$EVAL_JOB"
