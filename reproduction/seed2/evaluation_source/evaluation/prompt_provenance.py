"""Fail-closed validation for pinned document-local infilling prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import data_provenance
from evaluation.infilling_prompts import (
  PROMPT_POLICY_ID,
  deterministic_span_start,
)


PROMPT_MANIFEST_SCHEMA_VERSION = 2
PROMPT_ARTIFACT = 'pinned_document_local_infilling_prompts'


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(chunk_size), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
  result = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f'duplicate JSON object key: {key!r}')
    result[key] = value
  return result


def _reject_nonfinite(value: str) -> None:
  raise ValueError(f'non-finite JSON number: {value}')


def _load_json(path: Path) -> Any:
  try:
    return json.loads(
      path.read_text(), object_pairs_hook=_reject_duplicate_keys,
      parse_constant=_reject_nonfinite)
  except json.JSONDecodeError as error:
    raise ValueError(f'invalid JSON in {path}: {error}') from error


def _lower_hex(value: object, length: int, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != length
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(
      f'{context} must be {length} lowercase hexadecimal digits')
  return value


def _positive_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f'{context} must be a positive integer')
  return value


def _nonnegative_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f'{context} must be a non-negative integer')
  return value


def _nonempty_string(value: object, *, context: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f'{context} must be a non-empty string')
  return value


def _strict_fields(
    payload: Mapping[str, Any], expected: set[str], *, context: str,
) -> None:
  if set(payload) != expected:
    raise ValueError(
      f'{context} schema mismatch: missing={sorted(expected - set(payload))}, '
      f'unknown={sorted(set(payload) - expected)}')


def _validate_runtime_provenance(
    path: Path,
    *,
    expected_sha256: str,
    data_config: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  actual_sha256 = sha256_file(path)
  if actual_sha256 != expected_sha256:
    raise ValueError(
      f'prompt runtime provenance SHA256 mismatch: expected '
      f'{expected_sha256}, found {actual_sha256}')
  payload = _load_json(path)
  if not isinstance(payload, Mapping):
    raise TypeError('prompt runtime provenance must be a JSON object')
  required = {
    'schema_version', 'artifact', 'specification', 'observed',
    'specification_sha256', 'manifest_sha256',
  }
  _strict_fields(payload, required, context='prompt runtime provenance')
  if (payload['schema_version'] != data_provenance.PROVENANCE_SCHEMA_VERSION
      or payload['artifact'] != 'pinned_text_dataset_provenance'):
    raise ValueError('unsupported prompt runtime provenance identity')
  body = dict(payload)
  committed_manifest_sha = body.pop('manifest_sha256')
  if committed_manifest_sha != data_provenance.canonical_sha256(body):
    raise ValueError('prompt runtime provenance self-hash mismatch')
  specification = payload['specification']
  if not isinstance(specification, Mapping):
    raise TypeError('prompt runtime provenance specification must be an object')
  if payload['specification_sha256'] != \
      data_provenance.canonical_sha256(specification):
    raise ValueError('prompt runtime provenance specification hash mismatch')
  expected_semantics = {
    'logical_dataset_name': data_config['logical_validation_dataset'],
    'source_revision': data_config['dataset_revision'],
    'tokenizer_name_or_path': data_config['tokenizer_name_or_path'],
    'tokenizer_revision': data_config['tokenizer_revision'],
    'block_size': policy['sequence_length'],
  }
  for field, expected in expected_semantics.items():
    if specification.get(field) != expected:
      raise ValueError(
        f'prompt runtime provenance {field} differs from the prompt manifest')
  if specification.get('document_boundary_mode') not in {
      'source_document', 'wikitext_articles'}:
    raise ValueError(
      'prompt runtime provenance does not preserve document boundaries')
  return {
    'sha256': actual_sha256,
    'specification_sha256': payload['specification_sha256'],
    'manifest_sha256': committed_manifest_sha,
  }


def _validate_prompt_records(
    prompt_path: Path,
    *,
    dataset_id: str,
    policy: Mapping[str, Any],
    expected_count: int,
) -> None:
  sequence_length = policy['sequence_length']
  span_length = policy['span_length']
  selection_seed = policy['selection_seed']
  seen_ids = set()
  count = 0
  with prompt_path.open() as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        raise ValueError(f'{prompt_path}:{line_number} is blank')
      try:
        record = json.loads(
          line, object_pairs_hook=_reject_duplicate_keys,
          parse_constant=_reject_nonfinite)
      except json.JSONDecodeError as error:
        raise ValueError(
          f'invalid JSON in {prompt_path}:{line_number}: {error}') from error
      if not isinstance(record, Mapping):
        raise TypeError(f'{prompt_path}:{line_number} must be a JSON object')
      _strict_fields(
        record,
        {'id', 'input_ids', 'active_mask', 'reference_token_ids', 'metadata'},
        context=f'{prompt_path}:{line_number}')
      prompt_id = _nonempty_string(
        record['id'], context=f'{prompt_path}:{line_number}.id')
      if prompt_id in seen_ids:
        raise ValueError(f'duplicate prompt id {prompt_id!r}')
      seen_ids.add(prompt_id)
      inputs = record['input_ids']
      active = record['active_mask']
      reference = record['reference_token_ids']
      if (not isinstance(inputs, list) or not isinstance(active, list)
          or not isinstance(reference, list)
          or len(inputs) != sequence_length
          or len(active) != sequence_length
          or len(reference) != sequence_length):
        raise ValueError(
          f'{prompt_path}:{line_number} token arrays must have the committed '
          'sequence length')
      if (any(not isinstance(token, int) or isinstance(token, bool)
              or token < 0 for token in inputs)
          or inputs != reference
          or any(type(value) is not bool for value in active)):
        raise ValueError(
          f'{prompt_path}:{line_number} has invalid token/reference arrays')
      metadata = record['metadata']
      if not isinstance(metadata, Mapping):
        raise TypeError(f'{prompt_path}:{line_number}.metadata must be an object')
      metadata_fields = {
        'prompt_policy_id', 'dataset_id', 'source_document_index',
        'source_document_sha256', 'source_chunk_index', 'sequence_length',
        'span_start', 'span_stop', 'span_length', 'selection_seed',
      }
      _strict_fields(
        metadata, metadata_fields,
        context=f'{prompt_path}:{line_number}.metadata')
      if (metadata['prompt_policy_id'] != PROMPT_POLICY_ID
          or metadata['dataset_id'] != dataset_id
          or metadata['sequence_length'] != sequence_length
          or metadata['span_length'] != span_length
          or metadata['selection_seed'] != selection_seed):
        raise ValueError(
          f'{prompt_path}:{line_number} metadata differs from prompt policy')
      document_index = _nonnegative_int(
        metadata['source_document_index'], context='source_document_index')
      chunk_index = _nonnegative_int(
        metadata['source_chunk_index'], context='source_chunk_index')
      document_sha = _lower_hex(
        metadata['source_document_sha256'], 64,
        context='source_document_sha256')
      start = deterministic_span_start(
        dataset_id=dataset_id,
        document_sha256=document_sha,
        chunk_index=chunk_index,
        sequence_length=sequence_length,
        span_length=span_length,
        selection_seed=selection_seed)
      stop = start + span_length
      expected_active = [start <= index < stop for index in range(sequence_length)]
      expected_id = (
        f'{dataset_id}/document-{document_index:09d}/'
        f'chunk-{chunk_index:05d}/span-{span_length:04d}')
      if (metadata['span_start'] != start or metadata['span_stop'] != stop
          or active != expected_active or prompt_id != expected_id):
        raise ValueError(
          f'{prompt_path}:{line_number} is not the deterministic committed span')
      count += 1
  if count != expected_count:
    raise ValueError(
      f'prompt JSONL contains {count} records; manifest requires '
      f'{expected_count}')


def validate_prompt_bundle(
    prompt_path: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_data_config: str | None = None,
    expected_sequence_length: int | None = None,
) -> dict[str, Any]:
  """Validate prompt bytes, builder manifest, data config, and source proof."""
  prompt_path = prompt_path.expanduser().resolve()
  manifest_path = manifest_path.expanduser().resolve()
  if not prompt_path.is_file():
    raise FileNotFoundError(prompt_path)
  if not manifest_path.is_file():
    raise FileNotFoundError(manifest_path)
  expected_manifest_sha256 = _lower_hex(
    expected_manifest_sha256.lower(), 64,
    context='expected prompt manifest SHA256')
  actual_manifest_sha256 = sha256_file(manifest_path)
  if actual_manifest_sha256 != expected_manifest_sha256:
    raise ValueError(
      f'prompt manifest SHA256 mismatch: expected {expected_manifest_sha256}, '
      f'found {actual_manifest_sha256}')
  manifest = _load_json(manifest_path)
  if not isinstance(manifest, Mapping):
    raise TypeError('prompt manifest must be a JSON object')
  required = {
    'schema_version', 'artifact', 'created_utc', 'command', 'repository',
    'data_config', 'runtime_provenance', 'policy', 'output',
    'model_weights_loaded',
  }
  _strict_fields(manifest, required, context='prompt manifest')
  if (manifest['schema_version'] != PROMPT_MANIFEST_SCHEMA_VERSION
      or manifest['artifact'] != PROMPT_ARTIFACT
      or manifest['model_weights_loaded'] is not False):
    raise ValueError('unsupported prompt manifest identity')
  repository = manifest['repository']
  if not isinstance(repository, Mapping):
    raise TypeError('prompt manifest repository must be an object')
  _strict_fields(repository, {'git_sha', 'clean'}, context='prompt repository')
  git_sha = _lower_hex(repository['git_sha'], 40, context='prompt git SHA')
  if repository['clean'] is not True:
    raise ValueError('prompt builder repository was not clean')

  data_config = manifest['data_config']
  if not isinstance(data_config, Mapping):
    raise TypeError('prompt data_config must be an object')
  data_fields = {
    'name', 'path', 'sha256', 'logical_validation_dataset',
    'dataset_revision', 'tokenizer_name_or_path', 'tokenizer_revision',
  }
  _strict_fields(data_config, data_fields, context='prompt data_config')
  data_name = _nonempty_string(data_config['name'], context='data config name')
  if expected_data_config is not None and data_name != expected_data_config:
    raise ValueError(
      f'prompt data config {data_name!r} differs from requested '
      f'{expected_data_config!r}')
  data_path = Path(_nonempty_string(
    data_config['path'], context='data config path')).expanduser().resolve()
  if not data_path.is_file():
    raise FileNotFoundError(data_path)
  data_sha = _lower_hex(
    data_config['sha256'], 64, context='data config SHA256')
  if sha256_file(data_path) != data_sha:
    raise ValueError('prompt data config bytes differ from the manifest')
  dataset_id = _nonempty_string(
    data_config['logical_validation_dataset'], context='logical dataset')
  dataset_revision = _lower_hex(
    data_config['dataset_revision'], 40, context='dataset revision')
  tokenizer_name = _nonempty_string(
    data_config['tokenizer_name_or_path'], context='tokenizer name')
  tokenizer_revision = _lower_hex(
    data_config['tokenizer_revision'], 40, context='tokenizer revision')

  policy = manifest['policy']
  if not isinstance(policy, Mapping):
    raise TypeError('prompt policy must be an object')
  policy_fields = {
    'policy_id', 'selection_seed', 'span_length', 'sequence_length',
    'record_selection', 'boundary_policy',
  }
  _strict_fields(policy, policy_fields, context='prompt policy')
  if (policy['policy_id'] != PROMPT_POLICY_ID
      or policy['record_selection'] != 'first_n_in_pinned_validation_order'
      or policy['boundary_policy'] != 'never_mask_first_or_last_token'):
    raise ValueError('unsupported prompt selection policy')
  selection_seed = _nonnegative_int(
    policy['selection_seed'], context='prompt selection_seed')
  span_length = _positive_int(
    policy['span_length'], context='prompt span_length')
  sequence_length = _positive_int(
    policy['sequence_length'], context='prompt sequence_length')
  if (span_length > sequence_length - 2
      or (expected_sequence_length is not None
          and sequence_length != expected_sequence_length)):
    raise ValueError('prompt policy sequence/span length is incompatible')

  output = manifest['output']
  if not isinstance(output, Mapping):
    raise TypeError('prompt output must be an object')
  _strict_fields(
    output, {'path', 'sha256', 'size_bytes', 'num_prompts'},
    context='prompt output')
  output_name = Path(_nonempty_string(
    output['path'], context='prompt output path')).name
  if output_name != prompt_path.name:
    raise ValueError('prompt JSONL basename differs from its manifest')
  prompt_sha = _lower_hex(output['sha256'], 64, context='prompt SHA256')
  if sha256_file(prompt_path) != prompt_sha:
    raise ValueError('prompt JSONL bytes differ from its manifest')
  prompt_size = _positive_int(output['size_bytes'], context='prompt size')
  if prompt_path.stat().st_size != prompt_size:
    raise ValueError('prompt JSONL size differs from its manifest')
  prompt_count = _positive_int(
    output['num_prompts'], context='prompt count')

  runtime = manifest['runtime_provenance']
  if not isinstance(runtime, Mapping):
    raise TypeError('prompt runtime_provenance must be an object')
  _strict_fields(runtime, {'path', 'sha256'}, context='runtime provenance')
  runtime_path = Path(_nonempty_string(
    runtime['path'], context='runtime provenance path')).expanduser().resolve()
  runtime_sha = _lower_hex(
    runtime['sha256'], 64, context='runtime provenance SHA256')
  runtime_identity = _validate_runtime_provenance(
    runtime_path, expected_sha256=runtime_sha,
    data_config=data_config, policy=policy)
  _validate_prompt_records(
    prompt_path, dataset_id=dataset_id, policy=policy,
    expected_count=prompt_count)

  return {
    'schema_version': PROMPT_MANIFEST_SCHEMA_VERSION,
    'artifact': PROMPT_ARTIFACT,
    'manifest_sha256': actual_manifest_sha256,
    'builder_git_sha': git_sha,
    'data_config': {
      'name': data_name,
      'sha256': data_sha,
      'logical_validation_dataset': dataset_id,
      'dataset_revision': dataset_revision,
      'tokenizer_name_or_path': tokenizer_name,
      'tokenizer_revision': tokenizer_revision,
    },
    'runtime_provenance': runtime_identity,
    'policy': {
      'policy_id': PROMPT_POLICY_ID,
      'selection_seed': selection_seed,
      'span_length': span_length,
      'sequence_length': sequence_length,
      'record_selection': policy['record_selection'],
      'boundary_policy': policy['boundary_policy'],
    },
    'output': {
      'sha256': prompt_sha,
      'size_bytes': prompt_size,
      'num_prompts': prompt_count,
    },
  }
