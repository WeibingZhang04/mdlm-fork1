#!/usr/bin/env python3
"""Verify and aggregate a complete generation-pilot shard union."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.generation_shard_aggregation import (  # noqa: E402
  aggregate_generation_shards,
)


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      'Fail-closed verification and aggregation of every atomic generation '
      'pilot shard.'))
  parser.add_argument(
    '--shard', action='append', type=Path, required=True,
    help='Shard directory or exact manifest.json; repeat once per shard.')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--baseline-mode', default='factorized')
  parser.add_argument('--bootstrap-resamples', type=int, default=20_000)
  parser.add_argument('--bootstrap-seed', type=int, default=91017)
  parser.add_argument('--bootstrap-confidence', type=float, default=0.95)
  return parser.parse_args(argv)


def _atomic_write(path: Path, content: str) -> None:
  path = path.expanduser().resolve()
  if path.exists():
    raise FileExistsError(f'refusing to overwrite aggregate artifact {path}')
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    temporary.write_text(content)
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def main(argv=None) -> int:
  args = _parse_args(argv)
  result = aggregate_generation_shards(
    args.shard,
    baseline_mode=args.baseline_mode,
    bootstrap_resamples=args.bootstrap_resamples,
    bootstrap_seed=args.bootstrap_seed,
    bootstrap_confidence=args.bootstrap_confidence)
  output = args.output.expanduser().resolve()
  _atomic_write(output, json.dumps(result, indent=2, sort_keys=True) + '\n')
  print(json.dumps({
    'event': 'generation_shard_union_verified',
    'output': str(output),
    'num_shards': result['coverage']['num_shards'],
    'num_paired_draws': result['coverage']['global_num_paired_draws'],
    'num_unique_prompts': result['coverage']['num_unique_prompts'],
    'num_records': result['coverage']['verified_output_records'],
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

