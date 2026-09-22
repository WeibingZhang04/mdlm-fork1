"""Fail-closed dataset provenance for submission experiments.

The normal Hugging Face cache key is an implementation detail and is not a
scientific provenance record.  This module defines a small, canonical record
that pins every mutable input used by the multi-corpus evaluation protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import fsspec


PROVENANCE_SCHEMA_VERSION = 1
_COMMIT_RE = re.compile(r'^[0-9a-f]{40}$')


def require_commit_revision(value: Any, *, field: str) -> str:
  """Return an exact Hub commit or reject a mutable/missing revision."""
  if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
    raise ValueError(
      f'{field} must be an exact 40-character lowercase commit SHA; '
      f'got {value!r}')
  return value


def normalize_window(
    value: Sequence[int] | None,
    *,
    field: str,
    source_num_rows: int | None = None) -> tuple[int, int] | None:
  """Validate a half-open row window and return it as ``(start, stop)``."""
  if value is None:
    return None
  if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
      or len(value) != 2):
    raise ValueError(f'{field} must be a two-element half-open row window')
  start, stop = value
  if any(not isinstance(item, int) or isinstance(item, bool)
         for item in (start, stop)):
    raise ValueError(f'{field} endpoints must be integers')
  if start < 0 or stop <= start:
    raise ValueError(
      f'{field} must satisfy 0 <= start < stop; got [{start}, {stop})')
  if source_num_rows is not None and stop > source_num_rows:
    raise ValueError(
      f'{field} stop {stop} exceeds pinned source size {source_num_rows}')
  return int(start), int(stop)


def disjoint_window_proof(
    *,
    dataset_name_or_path: str,
    dataset_config_name: str | None,
    split: str,
    revision: str,
    source_num_rows: int,
    train_window: Sequence[int],
    heldout_window: Sequence[int]) -> dict[str, Any]:
  """Construct a cryptographic proof that two pinned row intervals do not overlap."""
  revision = require_commit_revision(revision, field='dataset revision')
  train = normalize_window(
    train_window, field='train_window', source_num_rows=source_num_rows)
  heldout = normalize_window(
    heldout_window, field='heldout_window',
    source_num_rows=source_num_rows)
  assert train is not None and heldout is not None
  overlap = max(train[0], heldout[0]) < min(train[1], heldout[1])
  if overlap:
    raise ValueError(
      f'train_window [{train[0]}, {train[1]}) overlaps heldout_window '
      f'[{heldout[0]}, {heldout[1]})')
  statement = {
    'dataset_name_or_path': dataset_name_or_path,
    'dataset_config_name': dataset_config_name,
    'split': split,
    'revision': revision,
    'source_num_rows': source_num_rows,
    'train_window': list(train),
    'heldout_window': list(heldout),
    'interval_semantics': 'zero_based_half_open_huggingface_row_indices',
    'overlap_num_rows': 0,
  }
  statement['proof_sha256'] = canonical_sha256(statement)
  return statement


def canonical_json(payload: Mapping[str, Any]) -> str:
  return json.dumps(
    dict(payload), sort_keys=True, separators=(',', ':'), allow_nan=False)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
  return hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()


def cache_key(specification: Mapping[str, Any]) -> str:
  """Content-address a processed dataset by all scientific inputs."""
  return canonical_sha256(specification)[:20]


def build_manifest(
    *, specification: Mapping[str, Any],
    observed: Mapping[str, Any]) -> dict[str, Any]:
  payload = {
    'schema_version': PROVENANCE_SCHEMA_VERSION,
    'artifact': 'pinned_text_dataset_provenance',
    'specification': dict(specification),
    'observed': dict(observed),
  }
  payload['specification_sha256'] = canonical_sha256(
    payload['specification'])
  payload['manifest_sha256'] = canonical_sha256(payload)
  return payload


def validate_manifest(
    payload: Mapping[str, Any],
    *, expected_specification: Mapping[str, Any]) -> dict[str, Any]:
  payload = dict(payload)
  if payload.get('schema_version') != PROVENANCE_SCHEMA_VERSION:
    raise ValueError('unsupported dataset provenance schema version')
  if payload.get('artifact') != 'pinned_text_dataset_provenance':
    raise ValueError('invalid dataset provenance artifact type')
  expected_hash = canonical_sha256(expected_specification)
  if payload.get('specification_sha256') != expected_hash:
    raise ValueError(
      'dataset provenance does not match the requested pinned specification')
  if payload.get('specification') != dict(expected_specification):
    raise ValueError('dataset provenance specification is non-canonical')
  manifest_hash = payload.pop('manifest_sha256', None)
  if manifest_hash != canonical_sha256(payload):
    raise ValueError('dataset provenance manifest hash mismatch')
  payload['manifest_sha256'] = manifest_hash
  return payload


def read_manifest(path: str) -> dict[str, Any]:
  with fsspec.open(path, 'r') as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise ValueError(f'{path} must contain a JSON object')
  return payload


def write_manifest(path: str, payload: Mapping[str, Any]) -> str:
  """Write strict JSON; any write error is deliberately allowed to abort the run."""
  serialized = json.dumps(
    dict(payload), indent=2, sort_keys=True, allow_nan=False) + '\n'
  with fsspec.open(path, 'w', auto_mkdir=True) as handle:
    handle.write(serialized)
  return path
