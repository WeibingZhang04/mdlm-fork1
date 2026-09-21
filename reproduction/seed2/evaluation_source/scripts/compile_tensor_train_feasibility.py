#!/usr/bin/env python3
"""Compile the frozen Tensor-Train OWT feasibility protocol into six jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.tensor_train_baseline import (  # noqa: E402
  compile_plan,
  write_plan,
)


DEFAULT_PROTOCOL = (
  REPO_ROOT / 'configs/experiment/tensor-train-owt-feasibility-v1.yaml')


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      'Compile the immutable six-job Tensor-Train OWT feasibility matrix.'))
  parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
  parser.add_argument('--source-root', type=Path, required=True)
  parser.add_argument(
    '--checkpoint-root', type=Path, required=True,
    help='Root containing owt/marginal.pt and owt/ttd_4_marg.pt.')
  parser.add_argument('--artifact-root', type=Path, required=True)
  parser.add_argument(
    '--cache-root', type=Path, required=True,
    help='Prefetched offline Hugging Face cache root prepared by the harness.')
  parser.add_argument('--output-dir', type=Path, required=True)
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = _parse_args(argv)
  plan, jobs = compile_plan(
    args.protocol,
    source_root=args.source_root,
    checkpoint_root=args.checkpoint_root,
    artifact_root=args.artifact_root,
    harness_repo_root=REPO_ROOT,
    cache_root=args.cache_root)
  plan_path = write_plan(plan, jobs, args.output_dir)
  print(json.dumps({
    'event': 'tensor_train_feasibility_plan_compiled',
    'plan_id': plan['plan_id'],
    'plan_path': str(plan_path),
    'num_jobs': len(jobs),
    'job_ids': list(jobs),
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
