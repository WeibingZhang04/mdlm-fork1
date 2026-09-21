"""Cryptographic provenance for paired structured validation runs.

The digest deliberately contains no token values or filesystem paths.  It is
an ordered commitment to the clean inputs, attention/active masks, sampled
times, and forward corruptions seen by one validation pass.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any, Mapping, Sequence

import fsspec
import numpy as np
import torch


PAIRING_DIGEST_FILENAME = 'validation_pairing_digest.json'
PAIRING_DIGEST_SCHEMA_VERSION = 1
_DIGEST_DOMAIN = (
  b'contextual-coupling-forest/structured-validation-pairing/v1\0')
_COMBINED_DIGEST_DOMAIN = (
  b'contextual-coupling-forest/combined-validation-pairing/v1\0')


def _update_length_prefixed(hasher, payload: bytes) -> None:
  hasher.update(struct.pack('<Q', len(payload)))
  hasher.update(payload)


def _canonical_array(
    tensor: torch.Tensor,
    *,
    kind: str,
    name: str) -> np.ndarray:
  value = torch.as_tensor(tensor).detach().cpu()
  if kind == 'integer':
    return np.ascontiguousarray(
      value.to(torch.int64).numpy().astype('<i8', copy=False))
  if kind == 'mask':
    if value.numel() and not bool(
        torch.logical_or(value == 0, value == 1).all().item()):
      raise ValueError(f'{name} must contain only boolean/0/1 values')
    return np.ascontiguousarray(
      value.to(torch.uint8).numpy().astype('u1', copy=False))
  if kind == 'float':
    value = value.to(torch.float64)
    if value.numel() and not bool(torch.isfinite(value).all().item()):
      raise ValueError(f'{name} must contain only finite values')
    return np.ascontiguousarray(
      value.numpy().astype('<f8', copy=False))
  raise ValueError(f'unknown canonical tensor kind: {kind}')


def _update_tensor(
    hasher,
    *,
    name: str,
    tensor: torch.Tensor,
    kind: str) -> None:
  array = _canonical_array(tensor, kind=kind, name=name)
  _update_length_prefixed(hasher, name.encode('utf-8'))
  hasher.update(struct.pack('<Q', array.ndim))
  for dimension in array.shape:
    hasher.update(struct.pack('<Q', int(dimension)))
  _update_length_prefixed(hasher, array.tobytes(order='C'))


class StructuredValidationPairingDigest:
  """Incrementally hash one rank's validation stream in batch order."""

  def __init__(self):
    self._hasher = hashlib.sha256()
    self._hasher.update(_DIGEST_DOMAIN)
    self.num_batches = 0
    self.num_examples = 0
    self.num_token_slots = 0
    self.num_active_tokens = 0

  def update(
      self,
      *,
      clean_input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      sampled_times: torch.Tensor,
      corrupted_input_ids: torch.Tensor,
      active_mask: torch.Tensor) -> None:
    clean_input_ids = torch.as_tensor(clean_input_ids)
    attention_mask = torch.as_tensor(attention_mask)
    sampled_times = torch.as_tensor(sampled_times)
    corrupted_input_ids = torch.as_tensor(corrupted_input_ids)
    active_mask = torch.as_tensor(active_mask)
    if clean_input_ids.ndim != 2:
      raise ValueError('clean_input_ids must have shape [batch, length]')
    if corrupted_input_ids.shape != clean_input_ids.shape:
      raise ValueError(
        'corrupted_input_ids must match clean_input_ids shape')
    if attention_mask.shape != clean_input_ids.shape:
      raise ValueError('attention_mask must match clean_input_ids shape')
    if active_mask.shape != clean_input_ids.shape:
      raise ValueError('active_mask must match clean_input_ids shape')
    if sampled_times.numel() != clean_input_ids.shape[0]:
      raise ValueError('sampled_times must contain one value per example')
    sampled_times = sampled_times.reshape(clean_input_ids.shape[0])

    self._hasher.update(struct.pack('<Q', self.num_batches))
    for name, tensor, kind in (
        ('clean_input_ids', clean_input_ids, 'integer'),
        ('attention_mask', attention_mask, 'mask'),
        ('sampled_times', sampled_times, 'float'),
        ('corrupted_input_ids', corrupted_input_ids, 'integer'),
        ('active_mask', active_mask, 'mask')):
      _update_tensor(
        self._hasher, name=name, tensor=tensor, kind=kind)

    self.num_batches += 1
    self.num_examples += int(clean_input_ids.shape[0])
    self.num_token_slots += int(clean_input_ids.numel())
    self.num_active_tokens += int(
      active_mask.to(torch.int64).sum().item())

  def rank_record(self, rank: int) -> dict[str, Any]:
    if rank < 0:
      raise ValueError('rank must be non-negative')
    return {
      'rank': int(rank),
      'sha256': self._hasher.hexdigest(),
      'num_batches': self.num_batches,
      'num_examples': self.num_examples,
      'num_token_slots': self.num_token_slots,
      'num_active_tokens': self.num_active_tokens,
    }


def combine_rank_records(
    rank_records: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    step: int,
    sanity_checking: bool) -> dict[str, Any]:
  """Combine rank-local commitments in ascending rank order."""
  if not rank_records:
    raise ValueError('at least one rank digest is required')
  records = [dict(record) for record in rank_records]
  records.sort(key=lambda record: int(record['rank']))
  expected_ranks = list(range(len(records)))
  observed_ranks = [int(record['rank']) for record in records]
  if observed_ranks != expected_ranks:
    raise ValueError(
      f'rank digests must cover contiguous ranks {expected_ranks}, got '
      f'{observed_ranks}')
  for record in records:
    digest = record.get('sha256')
    if (not isinstance(digest, str) or len(digest) != 64
        or any(character not in '0123456789abcdef' for character in digest)):
      raise ValueError('rank sha256 must be 64 lowercase hexadecimal digits')
    for field in (
        'num_batches', 'num_examples', 'num_token_slots',
        'num_active_tokens'):
      value = record.get(field)
      if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f'rank {field} must be a non-negative integer')

  if len(records) == 1:
    combined_digest = records[0]['sha256']
    digest_scope = 'single_rank_ordered_validation_stream'
  else:
    hasher = hashlib.sha256()
    hasher.update(_COMBINED_DIGEST_DOMAIN)
    for record in records:
      canonical_record = json.dumps(
        record, sort_keys=True, separators=(',', ':')).encode('utf-8')
      _update_length_prefixed(hasher, canonical_record)
    combined_digest = hasher.hexdigest()
    digest_scope = 'rank_ordered_commitment_to_ordered_rank_streams'

  return {
    'schema_version': PAIRING_DIGEST_SCHEMA_VERSION,
    'artifact': 'structured_validation_pairing_digest',
    'algorithm': 'sha256',
    'sha256': combined_digest,
    'digest_scope': digest_scope,
    'canonical_fields': [
      'clean_input_ids:int64-le',
      'attention_mask:uint8',
      'sampled_times:float64-le',
      'corrupted_input_ids:int64-le',
      'active_mask:uint8',
    ],
    'epoch': int(epoch),
    'step': int(step),
    'sanity_checking': bool(sanity_checking),
    'world_size': len(records),
    'num_batches': sum(record['num_batches'] for record in records),
    'num_examples': sum(record['num_examples'] for record in records),
    'num_token_slots': sum(
      record['num_token_slots'] for record in records),
    'num_active_tokens': sum(
      record['num_active_tokens'] for record in records),
    'rank_streams': records,
  }


def write_pairing_digest(
    save_dir: str,
    payload: Mapping[str, Any]) -> str:
  """Write a sanitized digest artifact directly under the run directory."""
  if not save_dir:
    raise ValueError('save_dir must be non-empty')
  path = save_dir.rstrip('/') + '/' + PAIRING_DIGEST_FILENAME
  serialized = json.dumps(
    dict(payload), indent=2, sort_keys=True, allow_nan=False) + '\n'
  with fsspec.open(path, 'w', auto_mkdir=True) as handle:
    handle.write(serialized)
  return path
