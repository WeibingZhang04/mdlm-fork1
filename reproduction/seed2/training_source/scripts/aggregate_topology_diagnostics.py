#!/usr/bin/env python3
"""Aggregate or replay fail-closed contextual-forest topology evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.topology_diagnostics import (  # noqa: E402
  aggregate_plan,
  verify_replay,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--protocol', type=Path, required=True,
    help='Frozen topology-diagnostics protocol JSON.')
  parser.add_argument(
    '--plan-dir', type=Path, required=True,
    help='Trusted compiled plan containing completed topology eval jobs.')
  destination = parser.add_mutually_exclusive_group(required=True)
  destination.add_argument(
    '--output', type=Path,
    help='Fresh output path for deterministic aggregate JSON.')
  destination.add_argument(
    '--verify-analysis', type=Path,
    help='Existing aggregate to verify by replaying every raw record.')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.verify_analysis is not None:
    analysis = verify_replay(
      analysis_path=args.verify_analysis,
      protocol_path=args.protocol,
      plan_dir=args.plan_dir)
    print(json.dumps({
      'status': 'verified',
      'analysis_sha256': analysis['analysis_sha256'],
      'num_records': analysis['grid_validation']['num_records'],
    }, sort_keys=True))
    return

  analysis = aggregate_plan(
    protocol_path=args.protocol, plan_dir=args.plan_dir)
  output = args.output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.exists():
    raise FileExistsError(
      f'{output} already exists; use --verify-analysis or a fresh path')
  with output.open('x', encoding='utf-8') as handle:
    handle.write(json.dumps(
      analysis, indent=2, sort_keys=True, allow_nan=False) + '\n')
  print(json.dumps({
    'status': 'written',
    'output': str(output),
    'analysis_sha256': analysis['analysis_sha256'],
    'num_records': analysis['grid_validation']['num_records'],
  }, sort_keys=True))


if __name__ == '__main__':
  main()
