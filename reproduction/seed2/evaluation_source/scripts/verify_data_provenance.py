#!/usr/bin/env python3
"""Fail closed when a submission dataset provenance artifact is incomplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import data_provenance  # noqa: E402


def verify(path: Path) -> dict:
  payload = data_provenance.read_manifest(str(path))
  specification = payload.get('specification')
  if not isinstance(specification, dict):
    raise ValueError(f'{path} has no specification object')
  payload = data_provenance.validate_manifest(
    payload, expected_specification=specification)
  observed = payload.get('observed')
  if not isinstance(observed, dict):
    raise ValueError(f'{path} has no observed object')

  expected_rows = specification['source_num_rows']
  observed_rows = observed.get('source_num_rows')
  if observed_rows is not None and observed_rows != expected_rows:
    raise ValueError(
      f'{path} source row count mismatch: expected {expected_rows}, '
      f'observed {observed_rows}')
  processed = observed.get('processed_num_sequences')
  if processed is not None and processed <= 0:
    raise ValueError(f'{path} records no processed sequences')

  proof = specification.get('disjoint_window_proof')
  if proof is not None:
    recomputed = data_provenance.disjoint_window_proof(
      dataset_name_or_path=proof['dataset_name_or_path'],
      dataset_config_name=proof['dataset_config_name'],
      split=proof['split'],
      revision=proof['revision'],
      source_num_rows=proof['source_num_rows'],
      train_window=proof['train_window'],
      heldout_window=proof['heldout_window'])
    if recomputed != proof:
      raise ValueError(f'{path} has an invalid disjoint-window proof')

  return {
    'path': str(path.resolve()),
    'manifest_sha256': payload['manifest_sha256'],
    'specification_sha256': payload['specification_sha256'],
    'dataset': specification['logical_dataset_name'],
    'revision': specification['source_revision'],
    'source_window': specification['source_window'],
    'processed_num_sequences': processed,
    'disjoint_window_proof_sha256': (
      proof['proof_sha256'] if proof is not None else None),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('manifests', type=Path, nargs='+')
  args = parser.parse_args()
  summaries = [verify(path) for path in args.manifests]
  print(json.dumps(summaries, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
