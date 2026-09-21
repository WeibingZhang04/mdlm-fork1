#!/usr/bin/env python3
"""Run and persist the non-neural Gate-1 structural sanity benchmark."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from synthetic.g1_benchmark import (  # noqa: E402
  evaluate_frozen_gate,
  run_benchmark,
)


def _git_sha() -> str:
  try:
    return subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return 'unknown'


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
  parser.add_argument('--train-samples-per-context', type=int, default=4096)
  parser.add_argument('--eval-samples-per-model', type=int, default=20000)
  parser.add_argument('--alpha', type=float, default=0.25)
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  if args.train_samples_per_context <= 0 or args.eval_samples_per_model <= 0:
    raise ValueError('sample counts must be positive')
  start = dt.datetime.now(dt.timezone.utc)
  records = run_benchmark(
    seeds=args.seeds,
    train_samples_per_context=args.train_samples_per_context,
    eval_samples_per_model=args.eval_samples_per_model,
    alpha=args.alpha)
  gate = evaluate_frozen_gate(records, seeds=args.seeds)
  end = dt.datetime.now(dt.timezone.utc)

  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  record_dicts = [record.as_dict() for record in records]
  (output_dir / 'records.json').write_text(
    json.dumps(record_dicts, indent=2, sort_keys=True) + '\n')
  with (output_dir / 'records.csv').open('w', newline='') as handle:
    fieldnames = [field.name for field in dataclasses.fields(records[0])]
    writer = csv.DictWriter(
      handle, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    for record in record_dicts:
      row = dict(record)
      row['predicted_edges'] = json.dumps(row['predicted_edges'])
      row['true_edges'] = json.dumps(row['true_edges'])
      writer.writerow(row)
  (output_dir / 'gate.json').write_text(
    json.dumps(gate, indent=2, sort_keys=True) + '\n')
  manifest = {
    'benchmark': 'g1_table_fit_structural_sanity',
    'scientific_scope': gate['scientific_scope'],
    'git_sha': _git_sha(),
    'command': sys.argv,
    'seeds': args.seeds,
    'train_samples_per_context': args.train_samples_per_context,
    'eval_samples_per_model': args.eval_samples_per_model,
    'alpha': args.alpha,
    'hostname': platform.node(),
    'platform': platform.platform(),
    'python': platform.python_version(),
    'numpy': np.__version__,
    'start_time_utc': start.isoformat(),
    'end_time_utc': end.isoformat(),
    'environment': {
      key: os.environ[key]
      for key in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID')
      if key in os.environ
    },
  }
  (output_dir / 'manifest.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  print(json.dumps(gate, indent=2, sort_keys=True))
  return 0 if gate['passed'] else 2


if __name__ == '__main__':
  raise SystemExit(main())
