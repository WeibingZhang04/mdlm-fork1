"""Fail-closed decode/retokenize diagnostics for generation samples.

Generation quality scores that operate on decoded text can hide byte-level
tokenization failures.  This module audits the raw token IDs without changing
the primary token-space evaluation: it decodes each sequence, retokenizes the
result, and reports replacement characters and exact round-trip failures.

The implementation is intentionally tokenizer-agnostic.  Callers supply an
object with the small Hugging Face tokenizer surface used by ``decode`` and
``__call__``; the command-line wrapper is responsible for loading it.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ARTIFACT = 'generation_decode_integrity_audit'
SCHEMA_VERSION = 1
REPLACEMENT_CHARACTER = '\ufffd'
TOKEN_SCOPES = ('sample_token_ids', 'sample_active_token_ids')


def canonical_sha256(payload: Any) -> str:
  """Hash a JSON value under a stable UTF-8 serialization."""
  encoded = json.dumps(
    payload, sort_keys=True, separators=(',', ':'),
    ensure_ascii=False).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


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


def resolve_sample_paths(inputs: Sequence[Path | str]) -> list[Path]:
  """Resolve files and recursively discover ``samples.jsonl`` in directories."""
  if not inputs:
    raise ValueError('at least one samples.jsonl file or directory is required')
  discovered: dict[str, Path] = {}
  for raw_path in inputs:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f'decode-integrity input does not exist: {path}')
    if path.is_dir():
      candidates = sorted(path.rglob('samples.jsonl'))
      if not candidates:
        raise FileNotFoundError(
          f'no samples.jsonl files found under input directory: {path}')
    elif path.is_file():
      if path.suffix != '.jsonl':
        raise ValueError(f'decode-integrity input is not JSONL: {path}')
      candidates = [path]
    else:
      raise ValueError(f'decode-integrity input is not a regular path: {path}')
    for candidate in candidates:
      resolved = candidate.resolve()
      discovered[str(resolved)] = resolved
  return [discovered[key] for key in sorted(discovered)]


def _token_ids(value: Any, *, context: str) -> list[int]:
  if not isinstance(value, list):
    raise TypeError(f'{context} must be a JSON array of token IDs')
  if not value:
    raise ValueError(f'{context} must contain at least one token ID')
  result = []
  for index, token in enumerate(value):
    if type(token) is not int or token < 0:
      raise ValueError(
        f'{context}[{index}] must be a nonnegative integer token ID')
    result.append(token)
  return result


def _active_mask(value: Any, *, length: int, context: str) -> list[bool]:
  if not isinstance(value, list) or len(value) != length:
    raise ValueError(f'{context} must be a boolean array of length {length}')
  if not all(type(item) is bool for item in value):
    raise ValueError(f'{context} must contain only booleans')
  if not any(value):
    raise ValueError(f'{context} must select at least one token')
  return list(value)


def _encode_text(tokenizer, text: str) -> list[int]:
  encoded = tokenizer(
    text, add_special_tokens=False, return_attention_mask=False)
  values = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded.input_ids
  if hasattr(values, 'tolist'):
    values = values.tolist()
  if values and isinstance(values[0], (list, tuple)):
    if len(values) != 1:
      raise ValueError('tokenizer returned more than one row for one string')
    values = values[0]
  if not isinstance(values, (list, tuple)):
    raise TypeError('tokenizer input_ids must be a one-dimensional sequence')
  result = []
  for index, token in enumerate(values):
    if type(token) is not int or token < 0:
      raise ValueError(
        f'tokenizer input_ids[{index}] is not a nonnegative integer')
    result.append(token)
  return result


def _decode_tokens(tokenizer, token_ids: Sequence[int]) -> str:
  decoded = tokenizer.decode(
    list(token_ids), skip_special_tokens=False,
    clean_up_tokenization_spaces=False)
  if not isinstance(decoded, str):
    raise TypeError('tokenizer.decode must return a string')
  return decoded


def _first_mismatch(
    raw_token_ids: Sequence[int],
    retokenized_ids: Sequence[int],
) -> dict[str, int | None] | None:
  for index, (raw, retokenized) in enumerate(
      zip(raw_token_ids, retokenized_ids)):
    if raw != retokenized:
      return {
        'index': index,
        'raw_token_id': raw,
        'retokenized_token_id': retokenized,
      }
  if len(raw_token_ids) == len(retokenized_ids):
    return None
  index = min(len(raw_token_ids), len(retokenized_ids))
  return {
    'index': index,
    'raw_token_id': (
      raw_token_ids[index] if index < len(raw_token_ids) else None),
    'retokenized_token_id': (
      retokenized_ids[index] if index < len(retokenized_ids) else None),
  }


def audit_token_ids(tokenizer, token_ids: Sequence[int]) -> dict[str, Any]:
  """Audit one raw token sequence without storing its decoded text."""
  raw = list(token_ids)
  if not raw:
    raise ValueError('cannot audit an empty token sequence')
  decoded = _decode_tokens(tokenizer, raw)
  retokenized = _encode_text(tokenizer, decoded)
  replacement_positions = [
    index for index, character in enumerate(decoded)
    if character == REPLACEMENT_CHARACTER
  ]
  raw_count = len(raw)
  retokenized_count = len(retokenized)
  decoded_codepoints = len(decoded)
  exact_match = raw == retokenized
  return {
    'raw_token_ids_sha256': canonical_sha256(raw),
    'retokenized_token_ids_sha256': canonical_sha256(retokenized),
    'decoded_text_sha256': hashlib.sha256(decoded.encode('utf-8')).hexdigest(),
    'raw_token_count': raw_count,
    'retokenized_token_count': retokenized_count,
    'token_length_delta': retokenized_count - raw_count,
    'token_length_ratio': retokenized_count / raw_count,
    'decoded_codepoint_count': decoded_codepoints,
    'contains_replacement_character': bool(replacement_positions),
    'replacement_character_count': len(replacement_positions),
    'replacement_character_codepoint_rate': (
      len(replacement_positions) / decoded_codepoints
      if decoded_codepoints else None),
    'replacement_character_positions': replacement_positions,
    'roundtrip_exact_match': exact_match,
    'roundtrip_mismatch': not exact_match,
    'first_roundtrip_mismatch': _first_mismatch(raw, retokenized),
  }


def _optional_nonempty_string(
    value: Any, *, context: str,
) -> str | None:
  if value is None:
    return None
  if not isinstance(value, str) or not value:
    raise ValueError(f'{context} must be a non-empty string when present')
  return value


def _optional_nonnegative_int(
    value: Any, *, context: str,
) -> int | None:
  if value is None:
    return None
  if type(value) is not int or value < 0:
    raise ValueError(f'{context} must be a nonnegative integer when present')
  return value


def _record_dimensions(record: Mapping[str, Any], *, context: str) -> dict[str, Any]:
  metadata = record.get('prompt_metadata')
  if metadata is not None and not isinstance(metadata, Mapping):
    raise TypeError(f'{context}.prompt_metadata must be an object when present')
  metadata_dataset = metadata.get('dataset_id') if metadata is not None else None
  top_dataset = record.get('dataset_id')
  if (metadata_dataset is not None and top_dataset is not None
      and metadata_dataset != top_dataset):
    raise ValueError(f'{context} has conflicting dataset identifiers')
  dataset = _optional_nonempty_string(
    top_dataset if top_dataset is not None else metadata_dataset,
    context=f'{context}.dataset_id')
  mode = _optional_nonempty_string(
    record.get('sampling_mode'), context=f'{context}.sampling_mode')
  requested_nfe = _optional_nonnegative_int(
    record.get('requested_nfe_budget'),
    context=f'{context}.requested_nfe_budget')
  measured_nfe = _optional_nonnegative_int(
    record.get('measured_nfe'), context=f'{context}.measured_nfe')
  dimensions = {}
  if dataset is not None:
    dimensions['dataset_id'] = dataset
  if mode is not None:
    dimensions['sampling_mode'] = mode
  if requested_nfe is not None:
    dimensions['requested_nfe_budget'] = requested_nfe
  if measured_nfe is not None:
    dimensions['measured_nfe'] = measured_nfe
  return dimensions


def _diagnostic_key(record: Mapping[str, Any], *, context: str) -> dict[str, Any]:
  result = _record_dimensions(record, context=context)
  for field in ('pair_key', 'prompt_id'):
    value = _optional_nonempty_string(
      record.get(field), context=f'{context}.{field}')
    if value is not None:
      result[field] = value
  for field in ('sample_index', 'pair_seed'):
    value = _optional_nonnegative_int(
      record.get(field), context=f'{context}.{field}')
    if value is not None:
      result[field] = value
  return result


def _audit_record(
    record: Any,
    *,
    tokenizer,
    source_path: Path,
    source_sha256: str,
    line_number: int,
) -> dict[str, Any]:
  context = f'{source_path}:{line_number}'
  if not isinstance(record, Mapping):
    raise TypeError(f'{context} must contain a JSON object')
  missing = [field for field in TOKEN_SCOPES if field not in record]
  if missing:
    raise ValueError(
      f'{context} missing required raw token IDs: {", ".join(missing)}')
  full_ids = _token_ids(
    record['sample_token_ids'], context=f'{context}.sample_token_ids')
  active_ids = _token_ids(
    record['sample_active_token_ids'],
    context=f'{context}.sample_active_token_ids')
  if 'active_mask' not in record:
    raise ValueError(f'{context} missing active_mask for raw-ID validation')
  active_mask = _active_mask(
    record['active_mask'], length=len(full_ids),
    context=f'{context}.active_mask')
  expected_active = [
    token for token, active in zip(full_ids, active_mask) if active
  ]
  if active_ids != expected_active:
    raise ValueError(
      f'{context}.sample_active_token_ids is inconsistent with '
      'sample_token_ids and active_mask')
  key = _diagnostic_key(record, context=context)
  source = {
    'path': str(source_path),
    'sha256': source_sha256,
    'line_number': line_number,
    'record_sha256': canonical_sha256(record),
  }
  diagnostic = {
    'diagnostic_key': key,
    'diagnostic_key_sha256': canonical_sha256({
      'source_sha256': source_sha256,
      'line_number': line_number,
      'key': key,
    }),
    'source': source,
    'scopes': {
      'sample_token_ids': audit_token_ids(tokenizer, full_ids),
      'sample_active_token_ids': audit_token_ids(tokenizer, active_ids),
    },
  }
  diagnostic['diagnostic_sha256'] = canonical_sha256(diagnostic)
  return diagnostic


def _mean(values: Sequence[float | int]) -> float:
  if not values:
    raise ValueError('cannot compute an empty mean')
  return sum(values) / len(values)


def _summarize_scope(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  count = len(diagnostics)
  if not count:
    raise ValueError('cannot summarize an empty diagnostic group')
  replacement_count = sum(
    item['replacement_character_count'] for item in diagnostics)
  decoded_count = sum(item['decoded_codepoint_count'] for item in diagnostics)
  replacement_records = sum(
    int(item['contains_replacement_character']) for item in diagnostics)
  mismatch_records = sum(int(item['roundtrip_mismatch']) for item in diagnostics)
  raw_count = sum(item['raw_token_count'] for item in diagnostics)
  retokenized_count = sum(
    item['retokenized_token_count'] for item in diagnostics)
  return {
    'num_records': count,
    'raw_token_count': raw_count,
    'retokenized_token_count': retokenized_count,
    'records_with_replacement_character': replacement_records,
    'replacement_character_record_rate': replacement_records / count,
    'replacement_character_count': replacement_count,
    'decoded_codepoint_count': decoded_count,
    'replacement_character_codepoint_rate': (
      replacement_count / decoded_count if decoded_count else None),
    'roundtrip_exact_match_records': count - mismatch_records,
    'roundtrip_exact_match_rate': (count - mismatch_records) / count,
    'roundtrip_mismatch_records': mismatch_records,
    'roundtrip_mismatch_rate': mismatch_records / count,
    'token_length_delta_sum': retokenized_count - raw_count,
    'token_length_delta_mean': _mean([
      item['token_length_delta'] for item in diagnostics
    ]),
    'token_length_ratio_mean': _mean([
      item['token_length_ratio'] for item in diagnostics
    ]),
    'token_length_ratio_total': retokenized_count / raw_count,
  }


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  return {
    'num_records': len(records),
    'scopes': {
      scope: _summarize_scope([
        record['scopes'][scope] for record in records
      ])
      for scope in TOKEN_SCOPES
    },
  }


def _group_dimensions(diagnostic: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
  key = diagnostic['diagnostic_key']
  return tuple(
    (field, key[field])
    for field in ('dataset_id', 'sampling_mode', 'requested_nfe_budget')
    if field in key
  )


def audit_decode_integrity(
    inputs: Sequence[Path | str],
    *,
    tokenizer,
    tokenizer_identity: Mapping[str, Any],
) -> dict[str, Any]:
  """Audit one or more raw generation JSONL files deterministically."""
  if not isinstance(tokenizer_identity, Mapping) or not tokenizer_identity:
    raise ValueError('tokenizer_identity must be a non-empty object')
  paths = resolve_sample_paths(inputs)
  diagnostics = []
  input_provenance = []
  for path in paths:
    source_sha256 = sha256_file(path)
    record_count = 0
    with path.open(encoding='utf-8') as handle:
      for line_number, line in enumerate(handle, start=1):
        if not line.strip():
          raise ValueError(f'{path}:{line_number} is a blank JSONL record')
        try:
          record = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as error:
          raise ValueError(
            f'{path}:{line_number} is not valid unambiguous JSON: '
            f'{error}') from error
        diagnostics.append(_audit_record(
          record, tokenizer=tokenizer, source_path=path,
          source_sha256=source_sha256, line_number=line_number))
        record_count += 1
    if record_count == 0:
      raise ValueError(f'{path} contains no JSONL records')
    input_provenance.append({
      'path': str(path),
      'sha256': source_sha256,
      'size_bytes': path.stat().st_size,
      'num_records': record_count,
    })
  groups: dict[tuple[tuple[str, Any], ...], list[Mapping[str, Any]]] = defaultdict(list)
  for diagnostic in diagnostics:
    groups[_group_dimensions(diagnostic)].append(diagnostic)
  grouped_summary = []
  for dimensions in sorted(groups, key=lambda item: canonical_sha256(item)):
    summary = _summarize_records(groups[dimensions])
    summary['dimensions'] = dict(dimensions)
    grouped_summary.append(summary)
  result = {
    'schema_version': SCHEMA_VERSION,
    'artifact': ARTIFACT,
    'policy': {
      'token_scopes': list(TOKEN_SCOPES),
      'decode': {
        'skip_special_tokens': False,
        'clean_up_tokenization_spaces': False,
      },
      'retokenize': {'add_special_tokens': False},
      'replacement_character': REPLACEMENT_CHARACTER,
      'primary_metrics_unchanged': True,
    },
    'tokenizer': dict(tokenizer_identity),
    'inputs': input_provenance,
    'aggregate': _summarize_records(diagnostics),
    'groups': grouped_summary,
    'records': diagnostics,
  }
  result['audit_sha256'] = canonical_sha256(result)
  return result


def tokenizer_identity(
    tokenizer,
    *,
    name_or_path: str,
    requested_revision: str,
) -> dict[str, Any]:
  """Build a JSON-compatible tokenizer identity for the audit artifact."""
  if not name_or_path or not requested_revision:
    raise ValueError('tokenizer name/path and requested revision are required')
  requested_revision = requested_revision.lower()
  if (len(requested_revision) != 40
      or any(character not in '0123456789abcdef'
             for character in requested_revision)):
    raise ValueError(
      'tokenizer requested_revision must be an exact 40-character Git SHA')
  init_kwargs = getattr(tokenizer, 'init_kwargs', {})
  resolved_revision = (
    init_kwargs.get('_commit_hash') if isinstance(init_kwargs, Mapping)
    else None)
  if resolved_revision is not None:
    resolved_revision = str(resolved_revision).lower()
    if resolved_revision != requested_revision:
      raise ValueError(
        'resolved tokenizer revision differs from requested exact Git SHA')
  return {
    'name_or_path': name_or_path,
    'requested_revision': requested_revision,
    'resolved_revision': resolved_revision,
    'class': (
      f'{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}'),
    'vocab_size': getattr(tokenizer, 'vocab_size', None),
    'bos_token_id': getattr(tokenizer, 'bos_token_id', None),
    'eos_token_id': getattr(tokenizer, 'eos_token_id', None),
    'pad_token_id': getattr(tokenizer, 'pad_token_id', None),
    'mask_token_id': getattr(tokenizer, 'mask_token_id', None),
  }
