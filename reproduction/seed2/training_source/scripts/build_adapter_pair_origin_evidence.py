#!/usr/bin/env python3
"""Build immutable origin evidence for a paired generation-adapter export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.adapter_pair_origin import (  # noqa: E402
  build_adapter_pair_origin_evidence,
  load_and_validate_adapter_pair_origin_evidence,
  write_adapter_pair_origin_evidence,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan-dir', type=Path, required=True)
  parser.add_argument('--suite', required=True)
  parser.add_argument('--candidate-k', type=int, required=True)
  parser.add_argument('--train-seed', type=int, required=True)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  payload = build_adapter_pair_origin_evidence(
    args.plan_dir,
    suite=args.suite,
    candidate_k=args.candidate_k,
    train_seed=args.train_seed,
  )
  file_sha256 = write_adapter_pair_origin_evidence(args.output, payload)
  validated = load_and_validate_adapter_pair_origin_evidence(
    args.output,
    expected_evidence_sha256=file_sha256,
    expected_plan_sha256=payload['source']['compiled_plan_sha256'],
    expected_suite=args.suite,
    expected_candidate_k=args.candidate_k,
    expected_train_seed=args.train_seed,
  )
  print(json.dumps({
    'output': str(args.output.expanduser().resolve()),
    'file_sha256': file_sha256,
    'evidence_sha256': validated['evidence_sha256'],
    'compiled_plan_sha256': validated['source']['compiled_plan_sha256'],
    'candidate_k': validated['source']['candidate_k'],
    'train_seed': validated['source']['train_seed'],
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
