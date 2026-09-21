"""Deterministic prompt construction from pinned document-local windows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import torch


PROMPT_POLICY_ID = 'document-local-contiguous-span-v1'


def _scalar_int(value: Any, *, name: str) -> int:
  if torch.is_tensor(value):
    if value.numel() != 1:
      raise ValueError(f'{name} must be scalar')
    value = value.item()
  if isinstance(value, bool):
    raise TypeError(f'{name} must be an integer, not bool')
  result = int(value)
  if result < 0:
    raise ValueError(f'{name} must be non-negative')
  return result


def _token_ids(value: Any) -> list[int]:
  if torch.is_tensor(value):
    if value.ndim != 1:
      raise ValueError('input_ids must be one-dimensional')
    value = value.detach().cpu().tolist()
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    raise TypeError('input_ids must be a token sequence')
  result = [int(token) for token in value]
  if any(token < 0 for token in result):
    raise ValueError('input_ids must be non-negative')
  return result


def _document_hash(value: Any) -> str:
  result = str(value)
  if (len(result) != 64
      or any(character not in '0123456789abcdef' for character in result)):
    raise ValueError('source_document_sha256 must be lowercase SHA256')
  return result


def deterministic_span_start(
    *,
    dataset_id: str,
    document_sha256: str,
    chunk_index: int,
    sequence_length: int,
    span_length: int,
    selection_seed: int,
) -> int:
  """Select a content-independent span wholly inside BOS/EOS boundaries."""
  if not dataset_id:
    raise ValueError('dataset_id must be non-empty')
  if sequence_length < 3:
    raise ValueError('sequence_length must leave BOS, payload, and EOS')
  if span_length <= 0 or span_length > sequence_length - 2:
    raise ValueError('span_length must lie in [1, sequence_length - 2]')
  if selection_seed < 0:
    raise ValueError('selection_seed must be non-negative')
  maximum_start = sequence_length - 1 - span_length
  number_of_starts = maximum_start
  commitment = json.dumps({
    'policy_id': PROMPT_POLICY_ID,
    'dataset_id': dataset_id,
    'document_sha256': document_sha256,
    'chunk_index': int(chunk_index),
    'sequence_length': int(sequence_length),
    'span_length': int(span_length),
    'selection_seed': int(selection_seed),
  }, sort_keys=True, separators=(',', ':')).encode('utf-8')
  offset = int(hashlib.sha256(commitment).hexdigest()[:16], 16)
  return 1 + offset % number_of_starts


def prompt_from_validation_record(
    record: Mapping[str, Any],
    *,
    dataset_id: str,
    span_length: int,
    selection_seed: int,
) -> dict[str, Any]:
  """Convert one document-local token window into the harness JSON schema."""
  required = (
    'input_ids', 'source_document_index', 'source_document_sha256',
    'source_chunk_index')
  missing = [name for name in required if name not in record]
  if missing:
    raise ValueError(
      f'validation record lacks document-local fields: {missing}')
  tokens = _token_ids(record['input_ids'])
  document_index = _scalar_int(
    record['source_document_index'], name='source_document_index')
  chunk_index = _scalar_int(
    record['source_chunk_index'], name='source_chunk_index')
  document_sha256 = _document_hash(record['source_document_sha256'])
  start = deterministic_span_start(
    dataset_id=dataset_id,
    document_sha256=document_sha256,
    chunk_index=chunk_index,
    sequence_length=len(tokens),
    span_length=span_length,
    selection_seed=selection_seed)
  stop = start + span_length
  active_mask = [start <= index < stop for index in range(len(tokens))]
  prompt_id = (
    f'{dataset_id}/document-{document_index:09d}/'
    f'chunk-{chunk_index:05d}/span-{span_length:04d}')
  return {
    'id': prompt_id,
    'input_ids': tokens,
    'active_mask': active_mask,
    'reference_token_ids': list(tokens),
    'metadata': {
      'prompt_policy_id': PROMPT_POLICY_ID,
      'dataset_id': dataset_id,
      'source_document_index': document_index,
      'source_document_sha256': document_sha256,
      'source_chunk_index': chunk_index,
      'sequence_length': len(tokens),
      'span_start': start,
      'span_stop': stop,
      'span_length': span_length,
      'selection_seed': selection_seed,
    },
  }


def build_infilling_prompts(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    span_length: int,
    selection_seed: int,
    num_prompts: int,
) -> list[dict[str, Any]]:
  """Take the first pinned windows and reject duplicate prompt identities."""
  if num_prompts <= 0:
    raise ValueError('num_prompts must be positive')
  prompts = []
  seen_ids = set()
  for record in records:
    prompt = prompt_from_validation_record(
      record,
      dataset_id=dataset_id,
      span_length=span_length,
      selection_seed=selection_seed)
    if prompt['id'] in seen_ids:
      raise ValueError(f'duplicate document-local prompt id {prompt["id"]}')
    seen_ids.add(prompt['id'])
    prompts.append(prompt)
    if len(prompts) == num_prompts:
      break
  if len(prompts) != num_prompts:
    raise ValueError(
      f'validation dataset yielded {len(prompts)} prompts; '
      f'{num_prompts} were required')
  sequence_lengths = {len(prompt['input_ids']) for prompt in prompts}
  if len(sequence_lengths) != 1:
    raise ValueError('validation prompts do not have one fixed sequence length')
  return prompts


def serialize_prompt_jsonl(prompts: Sequence[Mapping[str, Any]]) -> bytes:
  """Canonical UTF-8 JSONL representation used for file hashing."""
  if not prompts:
    raise ValueError('cannot serialize an empty prompt set')
  return ''.join(
    json.dumps(
      dict(prompt), sort_keys=True, separators=(',', ':'), allow_nan=False)
    + '\n'
    for prompt in prompts).encode('utf-8')
