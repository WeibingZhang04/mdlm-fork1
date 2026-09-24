#!/usr/bin/env bash
# Train Basic DD continuously from 0 to 6k with active-chain proposals, then
# run 100 samples at 8/16/32 steps with opt-in edge-source diagnostics and
# BF16 rotary caches.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [[ "$(git branch --show-current)" != original_table_base_crf-recovery ]]; then
  echo "Switch to original_table_base_crf-recovery before launching." >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --prepare-only ) ]]; then
  echo "Usage: bash scripts/launch_dd_chain_6k.sh [--prepare-only]" >&2
  exit 2
fi

: "${CCF_CACHE_ROOT:=/u401/n23zhang/mdlm_data/tree_mdlm_cache}"
: "${CCF_MAIL_USER:=n23zhang@uwaterloo.ca}"
: "${CCF_EXCLUDE_NODES=watgpu608,watgpu1008,watgpu1109}"
: "${CCF_PREPARE_PYTHON:=/usr/bin/python3}"
export CCF_ROTARY_CACHE_PRECISION=bf16
if [[ -z "${CCF_STUDY:-}" ]]; then
  mkdir -p "$CCF_CACHE_ROOT/runs"
  CCF_STUDY="$(mktemp -d "$CCF_CACHE_ROOT/runs/dd_chain_seed1_6k.XXXXXX")/study"
fi
export CCF_CACHE_ROOT CCF_MAIL_USER CCF_STUDY
"$CCF_PREPARE_PYTHON" \
  "$REPO/experiments/original_table/run_dd_chain_6k.py" prepare \
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
  --output="$CCF_STUDY/logs/train-%j.out" \
  --error="$CCF_STUDY/logs/train-%j.err" \
  "$REPO/experiments/original_table/train_dd_chain_6k.sbatch" "$CCF_STUDY")"
TRAIN_JOB="${TRAIN_JOB%%;*}"
printf '%s\n' "$TRAIN_JOB" > "$CCF_STUDY/train-job-id.txt"

EVAL_JOB="$(sbatch "${SBATCH_OPTIONS[@]}" --array=0-2%2 \
  --dependency="afterok:$TRAIN_JOB" \
  --output="$CCF_STUDY/logs/eval-%A_%a.out" \
  --error="$CCF_STUDY/logs/eval-%A_%a.err" \
  "$REPO/experiments/original_table/evaluate.sbatch" \
  "$CCF_STUDY" confirmation)"
EVAL_JOB="${EVAL_JOB%%;*}"
printf '%s\n' "$EVAL_JOB" > "$CCF_STUDY/eval-job-id.txt"
printf 'Study: %s\nTraining job: %s\nEvaluation job: %s\n' \
  "$CCF_STUDY" "$TRAIN_JOB" "$EVAL_JOB"
