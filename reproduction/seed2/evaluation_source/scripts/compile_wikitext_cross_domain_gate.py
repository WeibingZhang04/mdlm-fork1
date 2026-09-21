#!/usr/bin/env python3
"""Compile a reviewed, hash-bound WikiText authorization gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.generation_analysis_artifacts import (  # noqa: E402
  compile_reviewed_wikitext_gate,
  validate_reviewed_wikitext_gate,
  write_reviewed_wikitext_gate,
)


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dynamic-union', type=Path, required=True)
  parser.add_argument('--static-union', type=Path, required=True)
  parser.add_argument('--paired-comparison', type=Path, required=True)
  parser.add_argument('--decision', choices=('proceed', 'hold'), required=True)
  parser.add_argument('--review-statement', required=True)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = _parse_args(argv)
  payload = compile_reviewed_wikitext_gate(
    args.dynamic_union,
    args.static_union,
    args.paired_comparison,
    decision=args.decision,
    review_statement=args.review_statement)
  sha256 = write_reviewed_wikitext_gate(args.output, payload)
  validate_reviewed_wikitext_gate(
    args.output, expected_sha256=sha256, require_proceed=False)
  print(json.dumps({
    'event': 'reviewed_wikitext_cross_domain_gate_compiled',
    'decision': args.decision,
    'output': str(args.output.expanduser().resolve()),
    'sha256': sha256,
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
