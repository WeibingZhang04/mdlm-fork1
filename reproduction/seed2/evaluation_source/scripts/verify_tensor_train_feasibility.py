#!/usr/bin/env python3
"""Replay and optionally persist the complete Tensor-Train feasibility matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.tensor_train_baseline import (  # noqa: E402
  verify_complete_matrix,
)


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Fail-closed replay of all six Tensor-Train feasibility runs.')
  parser.add_argument('--plan', type=Path, required=True)
  parser.add_argument('--output', type=Path)
  return parser.parse_args(argv)


def _write_exclusive(path: Path, content: str) -> None:
  path = path.expanduser().resolve(strict=False)
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  with os.fdopen(descriptor, 'w') as handle:
    handle.write(content)
    handle.flush()
    os.fsync(handle.fileno())


def main(argv=None) -> int:
  args = _parse_args(argv)
  result = verify_complete_matrix(args.plan)
  serialized = json.dumps(result, indent=2, sort_keys=True) + '\n'
  if args.output is not None:
    _write_exclusive(args.output, serialized)
  print(serialized, end='')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
