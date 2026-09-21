"""Fail-closed verification and aggregation for generation pilot shards.

The generation runner writes one atomic directory per Spot-safe shard.  This
module treats the JSONL records as the primary data and the shard manifests as
cryptographic commitments.  It refuses partial grids, identity drift, or
unpaired samples before recomputing metrics over the complete shard union.

Timing is intentionally descriptive.  Distinct sample batches are not timing
replicates, even when many batches are available, so this module never turns
them into an inferential confidence interval.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation.generation_metrics import (
  REFERENCE_LM_SEQUENCE_POLICY,
  paired_token_metrics,
  repetition_rate,
  summarize_token_metrics,
)
from evaluation.infilling_prompts import (
  PROMPT_POLICY_ID,
  deterministic_span_start,
)
from evaluation.prompt_provenance import (
  PROMPT_ARTIFACT,
  PROMPT_MANIFEST_SCHEMA_VERSION,
)


EXPERIMENT = 'paired_contextual_forest_generation_pilot'
PAIRING_DIGEST_ALGORITHM = 'sha256-canonical-json-v2-prompt-metadata'
SUPPORTED_SAMPLING_MODES = {
  'factorized', 'factorized_confidence_gated',
  'structured_marginal', 'structured_joint',
}
CONTROL_MODES = {
  'dynamic_dynamic': ('dynamic', 'dynamic'),
  'fixed_dynamic': ('fixed', 'dynamic'),
  'dynamic_fixed': ('dynamic', 'fixed'),
  'static_static': ('fixed', 'fixed'),
}
STRUCTURED_IDENTITY_FIELDS = {
  'control_identity', 'topology_mode', 'factor_mode', 'candidate_top_k',
  'independent_mode', 'topology_weight', 'head_semantics',
  'training_semantics',
}
HEAD_SEMANTIC_FIELDS = {
  'rank', 'time_embed_dim', 'topology_dim', 'local_window',
  'num_anchor_slots', 'contextual_neighbors', 'component_size_cap',
  'min_edge_score', 'fixed_edges', 'fixed_edge_path',
}
TRAINING_SEMANTIC_FIELDS = {
  'objective_name', 'factorized_aux_weight', 'topology_strategy',
  'topology_temperature', 'topology_minimum_choices',
  'topology_edge_weight', 'topology_anchor_weight',
  'topology_slot_weight', 'topology_on_validation',
}
RUNTIME_PACKAGE_FIELDS = {
  'numpy', 'safetensors', 'tokenizers', 'transformers',
}
# Reference-LM perplexity is redundantly stored beside mean NLL.  Permit only
# a near-machine-precision exp() compatibility check here; deterministic token
# metrics and summaries below are compared exactly.
REFERENCE_LM_EXP_REL_TOL = 1e-12
REFERENCE_LM_EXP_ABS_TOL = 1e-12
MANIFEST_FIELDS = {
  'schema_version', 'experiment', 'scientific_scope', 'command',
  'start_time_utc', 'end_time_utc', 'duration_seconds', 'host',
  'repository', 'artifacts', 'adapter_origin_evidence', 'prompts', 'pairing',
  'spot_interruption_policy', 'matrix', 'outputs', 'reference_lm',
}
RECORD_FIELDS = {
  'schema_version', 'sample_index', 'pair_key', 'pair_seed', 'prompt_id',
  'prompt_metadata', 'sampling_mode', 'requested_nfe_budget',
  'measured_nfe', 'batch_seed', 'initial_token_ids', 'active_mask',
  'reference_token_ids', 'sample_token_ids', 'sample_active_token_ids',
  'text', 'metrics', 'timing', 'global_pairing_digest',
  'shard_pairing_digest', 'num_shards', 'shard_index',
}
OPTIONAL_RECORD_FIELDS = {'reference_lm'}
TIMING_FIELDS = {
  'batch_seed', 'batch_size', 'requested_nfe_budget', 'measured_nfe',
  'wall_clock_seconds', 'active_tokens', 'active_tokens_per_second',
  'sequence_tokens_per_second', 'peak_memory_bytes',
  'unresolved_mask_tokens',
}
GROUP_SUMMARY_FIELDS = {
  'num_sequences', 'num_tokens', 'distinct_n', 'mean_repetition_rate',
  'reference', 'sampling_mode', 'requested_nfe_budget', 'pairing_digest',
  'num_batches', 'wall_clock_seconds', 'active_tokens_per_second',
  'peak_memory_bytes', 'measured_nfe_values', 'unresolved_mask_tokens',
  'input_pairing_digest',
}


def canonical_sha256(payload: Any) -> str:
  """Hash a JSON value with the serialization used by the generation runner."""
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


def _reject_nonfinite_json(value: str) -> None:
  raise ValueError(f'non-finite JSON number: {value}')


def _load_json(path: Path) -> Any:
  try:
    with path.open() as handle:
      return json.load(
        handle,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_json)
  except json.JSONDecodeError as error:
    raise ValueError(f'invalid JSON in {path}: {error}') from error


def _load_json_line(line: str, *, source: str) -> Any:
  try:
    return json.loads(
      line,
      object_pairs_hook=_reject_duplicate_keys,
      parse_constant=_reject_nonfinite_json)
  except json.JSONDecodeError as error:
    raise ValueError(f'invalid JSON in {source}: {error}') from error


def _lower_hex(value: object, length: int, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != length
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(
      f'{context} must be {length} lowercase hexadecimal digits')
  return value


def _nonnegative_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f'{context} must be a non-negative integer')
  return value


def _positive_int(value: object, *, context: str) -> int:
  result = _nonnegative_int(value, context=context)
  if result == 0:
    raise ValueError(f'{context} must be positive')
  return result


def _finite_float(
    value: object,
    *,
    context: str,
    minimum: float | None = None,
) -> float:
  if (not isinstance(value, (int, float)) or isinstance(value, bool)
      or not math.isfinite(float(value))):
    raise ValueError(f'{context} must be a finite number')
  result = float(value)
  if minimum is not None and result < minimum:
    raise ValueError(f'{context} must be >= {minimum}')
  return result


def _nonempty_string(value: object, *, context: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f'{context} must be a non-empty string')
  return value


def _strict_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
    optional: set[str] | None = None,
) -> None:
  optional = optional or set()
  observed = set(payload)
  missing = expected - observed
  unknown = observed - expected - optional
  if missing or unknown:
    raise ValueError(
      f'{context} schema mismatch: missing={sorted(missing)}, '
      f'unknown={sorted(unknown)}')


def _safe_output_path(
    manifest_path: Path,
    entry: Mapping[str, Any],
    *,
    context: str,
) -> Path:
  if not isinstance(entry, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  relative = Path(_nonempty_string(entry.get('path'), context=f'{context}.path'))
  if relative.is_absolute() or '..' in relative.parts:
    raise ValueError(f'{context}.path must remain inside the shard directory')
  shard_dir = manifest_path.parent.resolve()
  resolved = (shard_dir / relative).resolve()
  try:
    resolved.relative_to(shard_dir)
  except ValueError as error:
    raise ValueError(f'{context}.path escapes the shard directory') from error
  if not resolved.is_file():
    raise FileNotFoundError(resolved)
  return resolved


def _pair_identity(record: Mapping[str, Any]) -> dict[str, Any]:
  return {
    'sample_index': record['sample_index'],
    'pair_key': record['pair_key'],
    'pair_seed': record['pair_seed'],
    'prompt_id': record['prompt_id'],
    'initial_token_ids': record['initial_token_ids'],
    'active_mask': record['active_mask'],
    'reference_token_ids': record['reference_token_ids'],
    'prompt_metadata': record['prompt_metadata'],
  }


def pairing_digest(records: Iterable[Mapping[str, Any]]) -> str:
  ordered = sorted(records, key=lambda record: record['sample_index'])
  return canonical_sha256([_pair_identity(record) for record in ordered])


def _assert_equivalent(actual: Any, expected: Any, *, context: str) -> None:
  """Compare deterministic nested JSON exactly, including scalar types."""
  if isinstance(expected, Mapping):
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
      raise ValueError(f'{context} object keys differ from recomputed values')
    for key in expected:
      _assert_equivalent(
        actual[key], expected[key], context=f'{context}.{key}')
    return
  if isinstance(expected, list):
    if not isinstance(actual, list) or len(actual) != len(expected):
      raise ValueError(f'{context} list differs from recomputed values')
    for index, (actual_item, expected_item) in enumerate(
        zip(actual, expected)):
      _assert_equivalent(
        actual_item, expected_item, context=f'{context}[{index}]')
    return
  if actual != expected or type(actual) is not type(expected):
    raise ValueError(
      f'{context} differs from recomputed value: '
      f'{actual!r} versus {expected!r}')


def _validate_reference_lm_score(
    payload: object,
    *,
    context: str,
) -> dict[str, Any]:
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  expected = {
    'model_name_or_path', 'revision', 'sequence_policy', 'token_count',
    'mean_nll_nats', 'perplexity',
  }
  _strict_fields(payload, expected, context=context)
  result = dict(payload)
  _nonempty_string(
    result['model_name_or_path'], context=f'{context}.model_name_or_path')
  _lower_hex(result['revision'], 40, context=f'{context}.revision')
  _nonempty_string(
    result['sequence_policy'], context=f'{context}.sequence_policy')
  if result['sequence_policy'] != REFERENCE_LM_SEQUENCE_POLICY:
    raise ValueError(
      f'{context}.sequence_policy must equal '
      f'{REFERENCE_LM_SEQUENCE_POLICY!r}')
  count = _nonnegative_int(result['token_count'], context=f'{context}.token_count')
  if count == 0:
    if result['mean_nll_nats'] is not None or result['perplexity'] is not None:
      raise ValueError(f'{context} zero-token score must use null NLL/perplexity')
  else:
    mean_nll = _finite_float(
      result['mean_nll_nats'], context=f'{context}.mean_nll_nats',
      minimum=0.0)
    perplexity = _finite_float(
      result['perplexity'], context=f'{context}.perplexity', minimum=0.0)
    expected_perplexity = math.exp(min(mean_nll, 80.0))
    if not math.isclose(
        perplexity, expected_perplexity,
        rel_tol=REFERENCE_LM_EXP_REL_TOL,
        abs_tol=REFERENCE_LM_EXP_ABS_TOL):
      raise ValueError(f'{context}.perplexity is inconsistent with mean NLL')
  return result


def _validate_record(
    payload: object,
    *,
    source: str,
    pairing: Mapping[str, Any],
    prompt_identity: Mapping[str, Any],
    modes: Sequence[str],
    nfe_budgets: Sequence[int],
) -> tuple[dict[str, Any], int]:
  if not isinstance(payload, Mapping):
    raise TypeError(f'{source} must be a JSON object')
  _strict_fields(
    payload, RECORD_FIELDS, context=source, optional=OPTIONAL_RECORD_FIELDS)
  record = dict(payload)
  if record['schema_version'] != 1:
    raise ValueError(f'{source}.schema_version must equal 1')
  sample_index = _nonnegative_int(
    record['sample_index'], context=f'{source}.sample_index')
  if sample_index >= pairing['global_num_samples']:
    raise ValueError(f'{source}.sample_index exceeds global sample range')
  _nonempty_string(record['pair_key'], context=f'{source}.pair_key')
  pair_seed = _nonnegative_int(record['pair_seed'], context=f'{source}.pair_seed')
  if pair_seed != pairing['base_seed'] + sample_index:
    raise ValueError(f'{source}.pair_seed is not base_seed + sample_index')
  _nonempty_string(record['prompt_id'], context=f'{source}.prompt_id')
  if not isinstance(record['prompt_metadata'], Mapping):
    raise TypeError(f'{source}.prompt_metadata must be a JSON object')
  if prompt_identity['source'] == 'jsonl':
    metadata = record['prompt_metadata']
    expected_metadata_fields = {
      'prompt_policy_id', 'dataset_id', 'source_document_index',
      'source_document_sha256', 'source_chunk_index', 'sequence_length',
      'span_start', 'span_stop', 'span_length', 'selection_seed',
    }
    _strict_fields(
      metadata, expected_metadata_fields,
      context=f'{source}.prompt_metadata')
    policy = prompt_identity['bundle_identity']['policy']
    dataset_id = prompt_identity['bundle_identity']['data_config'][
      'logical_validation_dataset']
    if (metadata['prompt_policy_id'] != PROMPT_POLICY_ID
        or metadata['dataset_id'] != dataset_id
        or metadata['sequence_length'] != policy['sequence_length']
        or metadata['span_length'] != policy['span_length']
        or metadata['selection_seed'] != policy['selection_seed']):
      raise ValueError(f'{source}.prompt_metadata differs from prompt bundle')
    document_index = _nonnegative_int(
      metadata['source_document_index'],
      context=f'{source}.prompt_metadata.source_document_index')
    chunk_index = _nonnegative_int(
      metadata['source_chunk_index'],
      context=f'{source}.prompt_metadata.source_chunk_index')
    document_sha = _lower_hex(
      metadata['source_document_sha256'], 64,
      context=f'{source}.prompt_metadata.source_document_sha256')
    span_start = deterministic_span_start(
      dataset_id=dataset_id,
      document_sha256=document_sha,
      chunk_index=chunk_index,
      sequence_length=policy['sequence_length'],
      span_length=policy['span_length'],
      selection_seed=policy['selection_seed'])
    span_stop = span_start + policy['span_length']
    expected_prompt_id = (
      f'{dataset_id}/document-{document_index:09d}/'
      f'chunk-{chunk_index:05d}/span-{policy["span_length"]:04d}')
    if (metadata['span_start'] != span_start
        or metadata['span_stop'] != span_stop
        or record['prompt_id'] != expected_prompt_id):
      raise ValueError(f'{source} is not the committed deterministic prompt')
  if record['sampling_mode'] not in modes:
    raise ValueError(f'{source}.sampling_mode is outside the declared matrix')
  if record['requested_nfe_budget'] not in nfe_budgets:
    raise ValueError(
      f'{source}.requested_nfe_budget is outside the declared matrix')
  requested_nfe = _positive_int(
    record['requested_nfe_budget'],
    context=f'{source}.requested_nfe_budget')
  measured_nfe = _positive_int(
    record['measured_nfe'], context=f'{source}.measured_nfe')
  if measured_nfe > requested_nfe:
    raise ValueError(f'{source}.measured_nfe exceeds requested NFE')
  batch_seed = _nonnegative_int(
    record['batch_seed'], context=f'{source}.batch_seed')

  sequence_length = pairing['sequence_length']
  for field in (
      'initial_token_ids', 'active_mask', 'sample_token_ids',
      'sample_active_token_ids'):
    if not isinstance(record[field], list):
      raise TypeError(f'{source}.{field} must be a JSON array')
  for field in ('initial_token_ids', 'sample_token_ids'):
    if len(record[field]) != sequence_length:
      raise ValueError(f'{source}.{field} has the wrong sequence length')
    for index, value in enumerate(record[field]):
      _nonnegative_int(value, context=f'{source}.{field}[{index}]')
  if len(record['active_mask']) != sequence_length:
    raise ValueError(f'{source}.active_mask has the wrong sequence length')
  if not all(type(value) is bool for value in record['active_mask']):
    raise ValueError(f'{source}.active_mask must contain only booleans')
  if not any(record['active_mask']):
    raise ValueError(f'{source}.active_mask must select at least one token')
  if prompt_identity['source'] == 'jsonl':
    expected_active = [
      prompt_identity['bundle_identity']['policy']['span_length'] > 0
      and record['prompt_metadata']['span_start'] <= index
      < record['prompt_metadata']['span_stop']
      for index in range(sequence_length)
    ]
    if record['active_mask'] != expected_active:
      raise ValueError(f'{source}.active_mask differs from prompt provenance')
  reference = record['reference_token_ids']
  if reference is not None:
    if not isinstance(reference, list) or len(reference) != sequence_length:
      raise ValueError(
        f'{source}.reference_token_ids has the wrong sequence length')
    for index, value in enumerate(reference):
      _nonnegative_int(
        value, context=f'{source}.reference_token_ids[{index}]')
    if any(
        initial != target
        for initial, target, active in zip(
          record['initial_token_ids'], reference, record['active_mask'])
        if not active):
      raise ValueError(
        f'{source} observed prompt tokens differ from the reference')

  mask_values = {
    token for token, active in zip(
      record['initial_token_ids'], record['active_mask']) if active
  }
  if len(mask_values) != 1:
    raise ValueError(
      f'{source} active initial positions do not share one mask token')
  mask_token_id = next(iter(mask_values))
  if any(
      token == mask_token_id and not active
      for token, active in zip(
        record['initial_token_ids'], record['active_mask'])):
    raise ValueError(f'{source} mask token appears outside active positions')
  if mask_token_id in record['sample_token_ids']:
    raise ValueError(f'{source} contains unresolved mask tokens')
  if any(
      initial != sampled
      for initial, sampled, active in zip(
        record['initial_token_ids'], record['sample_token_ids'],
        record['active_mask'])
      if not active):
    raise ValueError(f'{source} modified observed prompt tokens')
  active_values = [
    token for token, active in zip(
      record['sample_token_ids'], record['active_mask']) if active
  ]
  if record['sample_active_token_ids'] != active_values:
    raise ValueError(f'{source}.sample_active_token_ids is inconsistent')
  if not isinstance(record['text'], str):
    raise TypeError(f'{source}.text must be a string')

  if not isinstance(record['metrics'], Mapping):
    raise TypeError(f'{source}.metrics must be a JSON object')
  expected_metrics: dict[str, Any] = {
    'repetition_rate': {
      str(n): repetition_rate(active_values, n=n) for n in (1, 2, 4)
    },
  }
  if reference is not None:
    reference_active = [
      token for token, active in zip(reference, record['active_mask'])
      if active
    ]
    expected_metrics.update(paired_token_metrics(active_values, reference_active))
  _assert_equivalent(
    record['metrics'], expected_metrics, context=f'{source}.metrics')

  timing = record['timing']
  if not isinstance(timing, Mapping):
    raise TypeError(f'{source}.timing must be a JSON object')
  _strict_fields(timing, TIMING_FIELDS, context=f'{source}.timing')
  if timing['batch_seed'] != batch_seed:
    raise ValueError(f'{source}.timing.batch_seed differs from the record')
  if timing['requested_nfe_budget'] != requested_nfe:
    raise ValueError(f'{source}.timing requested NFE differs from the record')
  if timing['measured_nfe'] != measured_nfe:
    raise ValueError(f'{source}.timing measured NFE differs from the record')
  _positive_int(timing['batch_size'], context=f'{source}.timing.batch_size')
  _positive_int(timing['active_tokens'], context=f'{source}.timing.active_tokens')
  elapsed = _finite_float(
    timing['wall_clock_seconds'],
    context=f'{source}.timing.wall_clock_seconds', minimum=0.0)
  if elapsed <= 0.0:
    raise ValueError(f'{source}.timing.wall_clock_seconds must be positive')
  for field in ('active_tokens_per_second', 'sequence_tokens_per_second'):
    _finite_float(
      timing[field], context=f'{source}.timing.{field}', minimum=0.0)
  peak_memory = timing['peak_memory_bytes']
  if peak_memory is not None:
    _nonnegative_int(peak_memory, context=f'{source}.timing.peak_memory_bytes')
  if timing['unresolved_mask_tokens'] != 0:
    raise ValueError(f'{source}.timing reports unresolved mask tokens')

  if record['global_pairing_digest'] != pairing['global_pairing_digest']:
    raise ValueError(f'{source}.global_pairing_digest mismatch')
  if record['shard_pairing_digest'] != pairing['shard_pairing_digest']:
    raise ValueError(f'{source}.shard_pairing_digest mismatch')
  if record['num_shards'] != pairing['num_shards']:
    raise ValueError(f'{source}.num_shards mismatch')
  if record['shard_index'] != pairing['shard_index']:
    raise ValueError(f'{source}.shard_index mismatch')
  if sample_index % pairing['num_shards'] != pairing['shard_index']:
    raise ValueError(f'{source}.sample_index is assigned to another shard')

  if 'reference_lm' in record:
    record['reference_lm'] = _validate_reference_lm_score(
      record['reference_lm'], context=f'{source}.reference_lm')
  return record, mask_token_id


def _summarize_reference_lm(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
  presence = {'reference_lm' in record for record in records}
  if presence == {False}:
    return None
  if presence != {True}:
    raise ValueError('reference-LM score presence differs across paired records')
  identities = {
    (record['reference_lm']['model_name_or_path'],
     record['reference_lm']['revision'],
     record['reference_lm']['sequence_policy'])
    for record in records
  }
  if len(identities) != 1:
    raise ValueError('reference-LM identities differ across records')
  model, revision, sequence_policy = next(iter(identities))
  total_tokens = sum(record['reference_lm']['token_count'] for record in records)
  total_nll = math.fsum(
    record['reference_lm']['mean_nll_nats']
    * record['reference_lm']['token_count']
    for record in records
    if record['reference_lm']['mean_nll_nats'] is not None)
  mean_nll = total_nll / total_tokens if total_tokens else None
  return {
    'model_name_or_path': model,
    'revision': revision,
    'sequence_policy': sequence_policy,
    'num_scored_sequences': len(records),
    'num_scored_tokens': total_tokens,
    'mean_nll_nats': mean_nll,
    'perplexity': math.exp(min(mean_nll, 80.0)) if mean_nll is not None else None,
  }


def _batch_timings(
    records: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> list[dict[str, Any]]:
  by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
  for record in records:
    by_seed[record['batch_seed']].append(record)
  timings = []
  # Preserve first appearance: this exactly matches summarize_group's batch
  # accumulation order for deterministic shard-summary verification.
  for seed, batch_records in by_seed.items():
    first = dict(batch_records[0]['timing'])
    for record in batch_records[1:]:
      _assert_equivalent(
        record['timing'], first,
        context=f'{context}.batch[{seed}].timing')
    if len(batch_records) != first['batch_size']:
      raise ValueError(
        f'{context}.batch[{seed}] record count differs from timing.batch_size')
    active_tokens = sum(sum(record['active_mask']) for record in batch_records)
    if active_tokens != first['active_tokens']:
      raise ValueError(
        f'{context}.batch[{seed}] active-token count is inconsistent')
    sequence_tokens = sum(len(record['sample_token_ids']) for record in batch_records)
    expected_active_rate = active_tokens / first['wall_clock_seconds']
    expected_sequence_rate = sequence_tokens / first['wall_clock_seconds']
    if not math.isclose(
        first['active_tokens_per_second'], expected_active_rate,
        rel_tol=1e-9, abs_tol=1e-9):
      raise ValueError(
        f'{context}.batch[{seed}] active-token throughput is inconsistent')
    if not math.isclose(
        first['sequence_tokens_per_second'], expected_sequence_rate,
        rel_tol=1e-9, abs_tol=1e-9):
      raise ValueError(
        f'{context}.batch[{seed}] sequence-token throughput is inconsistent')
    timings.append(first)
  return timings


def _raw_group_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  if not records:
    raise ValueError(f'{context} is empty')
  ordered = sorted(records, key=lambda record: record['sample_index'])
  generated = [record['sample_active_token_ids'] for record in ordered]
  reference_presence = {
    record['reference_token_ids'] is not None for record in ordered
  }
  if len(reference_presence) != 1:
    raise ValueError(f'{context} mixes referenced and unreferenced prompts')
  reference = None
  if reference_presence == {True}:
    reference = [
      [
        token for token, active in zip(
          record['reference_token_ids'], record['active_mask']) if active
      ]
      for record in ordered
    ]
  summary = summarize_token_metrics(generated, reference=reference)
  timings = _batch_timings(ordered, context=context)
  elapsed = sum(float(item['wall_clock_seconds']) for item in timings)
  active_tokens = sum(item['active_tokens'] for item in timings)
  summary.update({
    'sampling_mode': ordered[0]['sampling_mode'],
    'requested_nfe_budget': ordered[0]['requested_nfe_budget'],
    'pairing_digest': pairing_digest(ordered),
    'num_batches': len(timings),
    'wall_clock_seconds': elapsed,
    'active_tokens_per_second': active_tokens / elapsed if elapsed else None,
    'peak_memory_bytes': max(
      (item['peak_memory_bytes'] for item in timings
       if item['peak_memory_bytes'] is not None),
      default=None),
    'measured_nfe_values': sorted({
      item['measured_nfe'] for item in timings
    }),
    'unresolved_mask_tokens': sum(
      item['unresolved_mask_tokens'] for item in timings),
  })
  return summary, timings


def _validate_group_summary(
    claimed: object,
    records: Sequence[Mapping[str, Any]],
    *,
    shard_digest: str,
    context: str,
) -> None:
  if not isinstance(claimed, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  _strict_fields(
    claimed, GROUP_SUMMARY_FIELDS, context=context,
    optional={'reference_lm'})
  recomputed, unused_timings = _raw_group_summary(records, context=context)
  del unused_timings
  recomputed['input_pairing_digest'] = shard_digest
  if 'reference_lm' in claimed:
    recomputed['reference_lm'] = _summarize_reference_lm(records)
  _assert_equivalent(claimed, recomputed, context=context)


def _normalize_prompt_bundle_identity(
    payload: object,
    *,
    context: str,
) -> dict[str, Any]:
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  expected = {
    'schema_version', 'artifact', 'manifest_sha256', 'builder_git_sha',
    'data_config', 'runtime_provenance', 'policy', 'output',
  }
  _strict_fields(payload, expected, context=context)
  if (payload['schema_version'] != PROMPT_MANIFEST_SCHEMA_VERSION
      or payload['artifact'] != PROMPT_ARTIFACT):
    raise ValueError(f'{context} has an unsupported identity')
  manifest_sha = _lower_hex(
    payload['manifest_sha256'], 64,
    context=f'{context}.manifest_sha256')
  builder_sha = _lower_hex(
    payload['builder_git_sha'], 40,
    context=f'{context}.builder_git_sha')

  data_config = payload['data_config']
  if not isinstance(data_config, Mapping):
    raise TypeError(f'{context}.data_config must be an object')
  data_fields = {
    'name', 'sha256', 'logical_validation_dataset', 'dataset_revision',
    'tokenizer_name_or_path', 'tokenizer_revision',
  }
  _strict_fields(data_config, data_fields, context=f'{context}.data_config')
  normalized_data = {
    'name': _nonempty_string(
      data_config['name'], context=f'{context}.data_config.name'),
    'sha256': _lower_hex(
      data_config['sha256'], 64, context=f'{context}.data_config.sha256'),
    'logical_validation_dataset': _nonempty_string(
      data_config['logical_validation_dataset'],
      context=f'{context}.data_config.logical_validation_dataset'),
    'dataset_revision': _lower_hex(
      data_config['dataset_revision'], 40,
      context=f'{context}.data_config.dataset_revision'),
    'tokenizer_name_or_path': _nonempty_string(
      data_config['tokenizer_name_or_path'],
      context=f'{context}.data_config.tokenizer_name_or_path'),
    'tokenizer_revision': _lower_hex(
      data_config['tokenizer_revision'], 40,
      context=f'{context}.data_config.tokenizer_revision'),
  }

  runtime = payload['runtime_provenance']
  if not isinstance(runtime, Mapping):
    raise TypeError(f'{context}.runtime_provenance must be an object')
  runtime_fields = {'sha256', 'specification_sha256', 'manifest_sha256'}
  _strict_fields(
    runtime, runtime_fields, context=f'{context}.runtime_provenance')
  normalized_runtime = {
    field: _lower_hex(
      runtime[field], 64,
      context=f'{context}.runtime_provenance.{field}')
    for field in sorted(runtime_fields)
  }

  policy = payload['policy']
  if not isinstance(policy, Mapping):
    raise TypeError(f'{context}.policy must be an object')
  policy_fields = {
    'policy_id', 'selection_seed', 'span_length', 'sequence_length',
    'record_selection', 'boundary_policy',
  }
  _strict_fields(policy, policy_fields, context=f'{context}.policy')
  if (policy['policy_id'] != PROMPT_POLICY_ID
      or policy['record_selection'] != 'first_n_in_pinned_validation_order'
      or policy['boundary_policy'] != 'never_mask_first_or_last_token'):
    raise ValueError(f'{context}.policy is unsupported')
  normalized_policy = {
    'policy_id': PROMPT_POLICY_ID,
    'selection_seed': _nonnegative_int(
      policy['selection_seed'], context=f'{context}.policy.selection_seed'),
    'span_length': _positive_int(
      policy['span_length'], context=f'{context}.policy.span_length'),
    'sequence_length': _positive_int(
      policy['sequence_length'], context=f'{context}.policy.sequence_length'),
    'record_selection': policy['record_selection'],
    'boundary_policy': policy['boundary_policy'],
  }
  if normalized_policy['span_length'] > normalized_policy['sequence_length'] - 2:
    raise ValueError(f'{context}.policy span is incompatible with sequence')

  output = payload['output']
  if not isinstance(output, Mapping):
    raise TypeError(f'{context}.output must be an object')
  _strict_fields(
    output, {'sha256', 'size_bytes', 'num_prompts'},
    context=f'{context}.output')
  normalized_output = {
    'sha256': _lower_hex(
      output['sha256'], 64, context=f'{context}.output.sha256'),
    'size_bytes': _positive_int(
      output['size_bytes'], context=f'{context}.output.size_bytes'),
    'num_prompts': _positive_int(
      output['num_prompts'], context=f'{context}.output.num_prompts'),
  }
  return {
    'schema_version': PROMPT_MANIFEST_SCHEMA_VERSION,
    'artifact': PROMPT_ARTIFACT,
    'manifest_sha256': manifest_sha,
    'builder_git_sha': builder_sha,
    'data_config': normalized_data,
    'runtime_provenance': normalized_runtime,
    'policy': normalized_policy,
    'output': normalized_output,
  }


def _normalize_prompt_identity(prompts: object, *, context: str) -> dict[str, Any]:
  if not isinstance(prompts, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  source = _nonempty_string(prompts['source'], context=f'{context}.source')
  if source == 'jsonl':
    _strict_fields(
      prompts,
      {
        'source', 'path', 'sha256', 'num_prompt_records', 'manifest_path',
        'manifest_sha256', 'bundle_identity',
      },
      context=context)
    _nonempty_string(prompts['path'], context=f'{context}.path')
    _nonempty_string(
      prompts['manifest_path'], context=f'{context}.manifest_path')
    count = _positive_int(
      prompts['num_prompt_records'], context=f'{context}.num_prompt_records')
    digest = _lower_hex(
      prompts['sha256'], 64, context=f'{context}.sha256')
    manifest_sha = _lower_hex(
      prompts['manifest_sha256'], 64,
      context=f'{context}.manifest_sha256')
    bundle = _normalize_prompt_bundle_identity(
      prompts['bundle_identity'], context=f'{context}.bundle_identity')
    if (bundle['manifest_sha256'] != manifest_sha
        or bundle['output']['sha256'] != digest
        or bundle['output']['num_prompts'] != count):
      raise ValueError(f'{context} differs from its prompt bundle identity')
    return {
      'source': source,
      'sha256': digest,
      'num_prompt_records': count,
      'manifest_sha256': manifest_sha,
      'bundle_identity': bundle,
    }
  elif source == 'generated_unconditional_prompt':
    _strict_fields(
      prompts, {'source', 'path', 'sha256', 'num_prompt_records'},
      context=context)
    count = _positive_int(
      prompts['num_prompt_records'], context=f'{context}.num_prompt_records')
    if prompts['sha256'] is not None or prompts['path'] is not None or count != 1:
      raise ValueError(f'{context} has invalid generated prompt provenance')
    return {
      'source': source,
      'sha256': None,
      'num_prompt_records': count,
    }
  else:
    raise ValueError(f'{context}.source is unsupported: {source!r}')


def _normalize_structured_adapter_identity(
    payload: object,
    *,
    context: str,
) -> dict[str, Any]:
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  _strict_fields(payload, STRUCTURED_IDENTITY_FIELDS, context=context)
  control = _nonempty_string(
    payload['control_identity'], context=f'{context}.control_identity')
  expected_modes = CONTROL_MODES.get(control)
  observed_modes = (payload['topology_mode'], payload['factor_mode'])
  if expected_modes is None or observed_modes != expected_modes:
    raise ValueError(
      f'{context} control/mode identity is inconsistent: '
      f'{control!r}, {observed_modes!r}')
  candidate_k = _positive_int(
    payload['candidate_top_k'], context=f'{context}.candidate_top_k')
  if not isinstance(payload['independent_mode'], bool):
    raise ValueError(f'{context}.independent_mode must be boolean')
  topology_weight = _finite_float(
    payload['topology_weight'], context=f'{context}.topology_weight',
    minimum=0.0)

  head = payload['head_semantics']
  if not isinstance(head, Mapping):
    raise TypeError(f'{context}.head_semantics must be an object')
  _strict_fields(head, HEAD_SEMANTIC_FIELDS, context=f'{context}.head_semantics')
  for field in (
      'rank', 'time_embed_dim', 'topology_dim', 'local_window',
      'num_anchor_slots', 'contextual_neighbors', 'component_size_cap'):
    _positive_int(head[field], context=f'{context}.head_semantics.{field}')
  if head['min_edge_score'] is not None:
    _finite_float(
      head['min_edge_score'], context=f'{context}.head_semantics.min_edge_score')
  fixed_edges = head['fixed_edges']
  if fixed_edges is not None:
    if not isinstance(fixed_edges, list):
      raise TypeError(f'{context}.head_semantics.fixed_edges must be an array')
    for index, edge in enumerate(fixed_edges):
      if (not isinstance(edge, list) or len(edge) != 2
          or any(not isinstance(value, int) or isinstance(value, bool)
                 or value < 0 for value in edge)):
        raise ValueError(
          f'{context}.head_semantics.fixed_edges[{index}] is invalid')
  fixed_edge_path = head['fixed_edge_path']
  if fixed_edge_path is not None and not isinstance(fixed_edge_path, str):
    raise TypeError(f'{context}.head_semantics.fixed_edge_path must be a string')

  training = payload['training_semantics']
  if not isinstance(training, Mapping):
    raise TypeError(f'{context}.training_semantics must be an object')
  _strict_fields(
    training, TRAINING_SEMANTIC_FIELDS,
    context=f'{context}.training_semantics')
  for field in (
      'factorized_aux_weight', 'topology_temperature',
      'topology_edge_weight', 'topology_anchor_weight',
      'topology_slot_weight'):
    _finite_float(
      training[field], context=f'{context}.training_semantics.{field}')
  _positive_int(
    training['topology_minimum_choices'],
    context=f'{context}.training_semantics.topology_minimum_choices')
  for field in ('objective_name', 'topology_strategy'):
    _nonempty_string(
      training[field], context=f'{context}.training_semantics.{field}')
  if not isinstance(training['topology_on_validation'], bool):
    raise ValueError(
      f'{context}.training_semantics.topology_on_validation must be boolean')
  return {
    'control_identity': control,
    'topology_mode': observed_modes[0],
    'factor_mode': observed_modes[1],
    'candidate_top_k': candidate_k,
    'independent_mode': payload['independent_mode'],
    'topology_weight': topology_weight,
    'head_semantics': dict(head),
    'training_semantics': dict(training),
  }


def _normalize_artifact_identity(
    payload: object,
    *,
    context: str,
    require_adapter_manifest: bool = False,
) -> dict[str, Any]:
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  expected = {'path', 'sha256', 'size_bytes'}
  if require_adapter_manifest:
    expected.update({
      'manifest_path', 'manifest_sha256', 'identity_sha256',
      'semantic_identity',
    })
  _strict_fields(payload, expected, context=context)
  _nonempty_string(payload['path'], context=f'{context}.path')
  digest = _lower_hex(payload['sha256'], 64, context=f'{context}.sha256')
  size = _positive_int(payload['size_bytes'], context=f'{context}.size_bytes')
  result = {'sha256': digest, 'size_bytes': size}
  if require_adapter_manifest:
    manifest_path = _nonempty_string(
      payload['manifest_path'], context=f'{context}.manifest_path').strip()
    if not manifest_path:
      raise ValueError(f'{context}.manifest_path must not be whitespace')
    semantic_identity = _normalize_structured_adapter_identity(
      payload['semantic_identity'], context=f'{context}.semantic_identity')
    identity_sha256 = _lower_hex(
      payload['identity_sha256'], 64,
      context=f'{context}.identity_sha256')
    if canonical_sha256(semantic_identity) != identity_sha256:
      raise ValueError(
        f'{context}.identity_sha256 does not commit to semantic_identity')
    result.update({
      'manifest_sha256': _lower_hex(
        payload['manifest_sha256'], 64,
        context=f'{context}.manifest_sha256'),
      'identity_sha256': identity_sha256,
      'semantic_identity': semantic_identity,
    })
  return result


def _normalize_adapter_origin_binding(
    payload: object,
    *,
    context: str,
) -> dict[str, Any]:
  """Validate the runner's self-hashed train/export/adapter binding."""
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  expected_fields = {
    'schema_version', 'artifact', 'evidence_file', 'source', 'arm',
    'adapter', 'plan_export', 'binding_sha256',
  }
  _strict_fields(payload, expected_fields, context=context)
  if (payload['schema_version'] != 1
      or payload['artifact'] !=
      'contextual_forest_generation_adapter_origin_binding'):
    raise ValueError(f'{context} has an unsupported identity')
  binding_sha256 = _lower_hex(
    payload['binding_sha256'], 64, context=f'{context}.binding_sha256')
  body = {
    field: payload[field] for field in expected_fields
    if field != 'binding_sha256'
  }
  if canonical_sha256(body) != binding_sha256:
    raise ValueError(f'{context}.binding_sha256 does not commit to its body')

  evidence_file = payload['evidence_file']
  if not isinstance(evidence_file, Mapping):
    raise TypeError(f'{context}.evidence_file must be a JSON object')
  _strict_fields(
    evidence_file, {'path', 'sha256', 'evidence_sha256'},
    context=f'{context}.evidence_file')
  evidence_path = Path(_nonempty_string(
    evidence_file['path'], context=f'{context}.evidence_file.path'))
  if not evidence_path.is_absolute():
    raise ValueError(f'{context}.evidence_file.path must be absolute')
  for field in ('sha256', 'evidence_sha256'):
    _lower_hex(
      evidence_file[field], 64,
      context=f'{context}.evidence_file.{field}')

  source = payload['source']
  source_fields = {
    'compiled_plan_path', 'compiled_plan_sha256', 'plan_id', 'protocol_id',
    'source_manifest_sha256', 'repository', 'suite', 'candidate_k',
    'train_seed', 'legacy_plan_schema',
  }
  if not isinstance(source, Mapping):
    raise TypeError(f'{context}.source must be a JSON object')
  _strict_fields(source, source_fields, context=f'{context}.source')
  plan_path = Path(_nonempty_string(
    source['compiled_plan_path'],
    context=f'{context}.source.compiled_plan_path'))
  if not plan_path.is_absolute() or plan_path.name != 'compiled-plan.json':
    raise ValueError(
      f'{context}.source.compiled_plan_path must be an absolute plan path')
  for field in ('compiled_plan_sha256', 'plan_id', 'source_manifest_sha256'):
    _lower_hex(source[field], 64, context=f'{context}.source.{field}')
  for field in ('protocol_id', 'suite'):
    _nonempty_string(source[field], context=f'{context}.source.{field}')
  candidate_k = _positive_int(
    source['candidate_k'], context=f'{context}.source.candidate_k')
  _positive_int(source['train_seed'], context=f'{context}.source.train_seed')
  if source['legacy_plan_schema'] is not False:
    raise ValueError(
      f'{context} requires a schema-v2 plan with schema-v4 exports')
  repository = source['repository']
  if (not isinstance(repository, Mapping)
      or set(repository) != {'sha', 'clean'}
      or repository['clean'] is not True):
    raise ValueError(f'{context}.source.repository must be exactly clean')
  _lower_hex(
    repository['sha'], 40, context=f'{context}.source.repository.sha')

  arm = payload['arm']
  if arm not in {'dynamic_dynamic', 'static_static'}:
    raise ValueError(f'{context}.arm is not a paper comparison arm')
  adapter = payload['adapter']
  adapter_fields = {
    'path', 'sha256', 'manifest_path', 'manifest_sha256',
    'structured_decoder_identity', 'structured_decoder_identity_sha256',
    'source_checkpoint_sha256', 'source_checkpoint_global_step',
    'released_backbone',
  }
  if not isinstance(adapter, Mapping):
    raise TypeError(f'{context}.adapter must be a JSON object')
  _strict_fields(adapter, adapter_fields, context=f'{context}.adapter')
  for field in ('path', 'manifest_path'):
    artifact_path = Path(_nonempty_string(
      adapter[field], context=f'{context}.adapter.{field}'))
    if not artifact_path.is_absolute():
      raise ValueError(f'{context}.adapter.{field} must be absolute')
  for field in (
      'sha256', 'manifest_sha256', 'structured_decoder_identity_sha256',
      'source_checkpoint_sha256'):
    _lower_hex(adapter[field], 64, context=f'{context}.adapter.{field}')
  _nonnegative_int(
    adapter['source_checkpoint_global_step'],
    context=f'{context}.adapter.source_checkpoint_global_step')
  semantic_identity = _normalize_structured_adapter_identity(
    adapter['structured_decoder_identity'],
    context=f'{context}.adapter.structured_decoder_identity')
  _assert_equivalent(
    adapter['structured_decoder_identity'], semantic_identity,
    context=f'{context}.adapter.structured_decoder_identity')
  if canonical_sha256(semantic_identity) != \
      adapter['structured_decoder_identity_sha256']:
    raise ValueError(
      f'{context}.adapter identity SHA256 does not commit to its semantics')
  if (semantic_identity['control_identity'] != arm
      or semantic_identity['candidate_top_k'] != candidate_k):
    raise ValueError(f'{context}.adapter semantics differ from source/arm')

  released = adapter['released_backbone']
  released_fields = {
    'repository', 'revision', 'source_sha256', 'source_size_bytes',
    'tensor_count',
  }
  if not isinstance(released, Mapping):
    raise TypeError(f'{context}.adapter.released_backbone must be an object')
  _strict_fields(
    released, released_fields,
    context=f'{context}.adapter.released_backbone')
  for field in ('repository', 'revision'):
    _nonempty_string(
      released[field],
      context=f'{context}.adapter.released_backbone.{field}')
  _lower_hex(
    released['source_sha256'], 64,
    context=f'{context}.adapter.released_backbone.source_sha256')
  for field in ('source_size_bytes', 'tensor_count'):
    _positive_int(
      released[field],
      context=f'{context}.adapter.released_backbone.{field}')

  plan_export = payload['plan_export']
  plan_export_fields = {
    'train_job_id', 'train_job_spec_sha256', 'train_job_execution_sha256',
    'train_success_marker_sha256', 'checkpoint_sha256',
    'training_data_provenance_sha256',
    'training_validation_data_provenance_sha256', 'export_job_id',
    'export_job_spec_sha256', 'export_job_execution_sha256',
    'export_success_marker_sha256', 'adapter_sha256',
    'adapter_manifest_sha256',
  }
  if not isinstance(plan_export, Mapping):
    raise TypeError(f'{context}.plan_export must be a JSON object')
  _strict_fields(
    plan_export, plan_export_fields, context=f'{context}.plan_export')
  for field in ('train_job_id', 'export_job_id'):
    _nonempty_string(
      plan_export[field], context=f'{context}.plan_export.{field}')
  for field in plan_export_fields - {'train_job_id', 'export_job_id'}:
    _lower_hex(
      plan_export[field], 64, context=f'{context}.plan_export.{field}')
  if (plan_export['adapter_sha256'] != adapter['sha256']
      or plan_export['adapter_manifest_sha256'] != adapter['manifest_sha256']
      or plan_export['checkpoint_sha256'] !=
      adapter['source_checkpoint_sha256']):
    raise ValueError(f'{context}.plan_export differs from adapter provenance')
  return dict(payload)


def _normalize_runtime_identity(
    payload: object,
    *,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Return fail-closed quality and timing execution identities."""
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  expected = {
    'hostname', 'platform', 'python', 'torch', 'cuda_runtime', 'device',
    'gpu', 'parameter_dtypes', 'precision_policy', 'packages',
  }
  _strict_fields(payload, expected, context=context)
  _nonempty_string(payload['hostname'], context=f'{context}.hostname')
  platform_identity = _nonempty_string(
    payload['platform'], context=f'{context}.platform').strip()
  python = _nonempty_string(
    payload['python'], context=f'{context}.python').strip()
  torch_version = _nonempty_string(
    payload['torch'], context=f'{context}.torch').strip()
  device = _nonempty_string(
    payload['device'], context=f'{context}.device').strip().lower()
  device_type = device.split(':', 1)[0]
  if device_type not in {'cpu', 'cuda', 'mps'}:
    raise ValueError(f'{context}.device has unsupported type {device_type!r}')

  cuda_runtime = payload['cuda_runtime']
  if cuda_runtime is not None:
    cuda_runtime = _nonempty_string(
      cuda_runtime, context=f'{context}.cuda_runtime').strip()
  if device_type == 'cuda' and cuda_runtime is None:
    raise ValueError(f'{context}.cuda_runtime is required for CUDA execution')

  dtypes = payload['parameter_dtypes']
  if (not isinstance(dtypes, list) or not dtypes
      or any(not isinstance(value, str) or not value.strip()
             for value in dtypes)):
    raise ValueError(
      f'{context}.parameter_dtypes must be a non-empty string array')
  normalized_dtypes = sorted(value.strip() for value in dtypes)
  if len(set(normalized_dtypes)) != len(normalized_dtypes):
    raise ValueError(f'{context}.parameter_dtypes contains duplicates')
  if dtypes != normalized_dtypes:
    raise ValueError(f'{context}.parameter_dtypes must be sorted')
  precision_policy = _nonempty_string(
    payload['precision_policy'], context=f'{context}.precision_policy').strip()
  packages = payload['packages']
  if not isinstance(packages, Mapping) or set(packages) != \
      RUNTIME_PACKAGE_FIELDS:
    raise ValueError(
      f'{context}.packages must contain exactly '
      f'{sorted(RUNTIME_PACKAGE_FIELDS)}')
  normalized_packages = {
    name: _nonempty_string(
      packages[name], context=f'{context}.packages.{name}').strip()
    for name in sorted(RUNTIME_PACKAGE_FIELDS)
  }
  if any(not value for value in normalized_packages.values()):
    raise ValueError(f'{context}.packages versions must not be whitespace')

  gpu = payload['gpu']
  if device_type == 'cuda':
    gpu = _nonempty_string(gpu, context=f'{context}.gpu').strip()
  elif gpu is not None:
    raise ValueError(f'{context}.gpu must be null for non-CUDA execution')
  quality_identity = {
    'platform': platform_identity,
    'python': python,
    'torch': torch_version,
    'packages': normalized_packages,
    'cuda_runtime': cuda_runtime,
    'device_type': device_type,
    'parameter_dtypes': normalized_dtypes,
    'precision_policy': precision_policy,
  }
  timing_identity = {
    'device_type': device_type,
    'gpu': gpu,
  }
  return quality_identity, timing_identity


def _reference_lm_identity(payload: object, *, context: str) -> dict[str, Any] | None:
  if payload is None:
    return None
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must be null or a JSON object')
  expected_fields = {
    'model_name_or_path', 'revision', 'sequence_policy', 'runtime_identity',
    'num_scored_sequences', 'num_scored_tokens', 'mean_nll_nats',
    'perplexity',
  }
  _strict_fields(payload, expected_fields, context=context)
  model = _nonempty_string(
    payload.get('model_name_or_path'),
    context=f'{context}.model_name_or_path')
  revision = _lower_hex(payload.get('revision'), 40, context=f'{context}.revision')
  sequence_policy = _nonempty_string(
    payload.get('sequence_policy'), context=f'{context}.sequence_policy')
  if sequence_policy != REFERENCE_LM_SEQUENCE_POLICY:
    raise ValueError(
      f'{context}.sequence_policy must equal '
      f'{REFERENCE_LM_SEQUENCE_POLICY!r}')
  runtime = payload['runtime_identity']
  if not isinstance(runtime, Mapping):
    raise TypeError(f'{context}.runtime_identity must be a JSON object')
  runtime_fields = {
    'schema_version', 'model_name_or_path', 'model_revision', 'model_class',
    'model_config_class', 'tokenizer_name_or_path', 'tokenizer_revision',
    'tokenizer_class', 'tokenizer_vocab_size', 'tokenizer_bos_token_id',
    'tokenizer_eos_token_id', 'tokenizer_pad_token_id',
    'tokenizer_padding_side', 'tokenizer_truncation_side',
    'tokenization_policy', 'sequence_policy', 'add_special_tokens',
    'batch_size', 'max_length', 'requested_dtype', 'parameter_dtypes',
    'precision_policy', 'device', 'python', 'torch', 'cuda_runtime',
    'transformers', 'tokenizers',
  }
  _strict_fields(
    runtime, runtime_fields, context=f'{context}.runtime_identity')
  if runtime['schema_version'] != 1:
    raise ValueError(f'{context}.runtime_identity schema is unsupported')
  if (runtime['model_name_or_path'] != model
      or runtime['model_revision'] != revision
      or runtime['tokenizer_revision'] != revision
      or runtime['sequence_policy'] != sequence_policy):
    raise ValueError(f'{context}.runtime_identity differs from model identity')
  for field in (
      'model_class', 'model_config_class', 'tokenizer_name_or_path',
      'tokenizer_class', 'tokenizer_padding_side',
      'tokenizer_truncation_side', 'tokenization_policy', 'requested_dtype',
      'precision_policy', 'device', 'python', 'torch', 'transformers',
      'tokenizers'):
    _nonempty_string(
      runtime[field], context=f'{context}.runtime_identity.{field}')
  for field in ('tokenizer_vocab_size', 'batch_size', 'max_length'):
    _positive_int(
      runtime[field], context=f'{context}.runtime_identity.{field}')
  for field in (
      'tokenizer_bos_token_id', 'tokenizer_eos_token_id',
      'tokenizer_pad_token_id'):
    _nonnegative_int(
      runtime[field], context=f'{context}.runtime_identity.{field}')
  if runtime['add_special_tokens'] is not True:
    raise ValueError(
      f'{context}.runtime_identity.add_special_tokens must be true')
  parameter_dtypes = runtime['parameter_dtypes']
  if (not isinstance(parameter_dtypes, list) or not parameter_dtypes
      or parameter_dtypes != sorted(parameter_dtypes)
      or len(set(parameter_dtypes)) != len(parameter_dtypes)
      or any(not isinstance(value, str) or not value.strip()
             for value in parameter_dtypes)):
    raise ValueError(
      f'{context}.runtime_identity.parameter_dtypes is invalid')
  device_type = runtime['device'].split(':', 1)[0].lower()
  if device_type not in {'cpu', 'cuda', 'mps'}:
    raise ValueError(f'{context}.runtime_identity.device is unsupported')
  cuda_runtime = runtime['cuda_runtime']
  if device_type == 'cuda':
    _nonempty_string(
      cuda_runtime, context=f'{context}.runtime_identity.cuda_runtime')
  elif cuda_runtime is not None:
    raise ValueError(
      f'{context}.runtime_identity.cuda_runtime must be null off CUDA')
  return {
    'model_name_or_path': model,
    'revision': revision,
    'sequence_policy': sequence_policy,
    'runtime_identity': dict(runtime),
  }


def _validate_manifest_header(
    manifest: object,
    *,
    path: Path,
) -> dict[str, Any]:
  if not isinstance(manifest, Mapping):
    raise TypeError(f'{path} must contain a JSON object')
  _strict_fields(manifest, MANIFEST_FIELDS, context=str(path))
  result = dict(manifest)
  if result['schema_version'] != 1 or result['experiment'] != EXPERIMENT:
    raise ValueError(f'{path} has an unsupported manifest identity')

  repository = result['repository']
  if not isinstance(repository, Mapping):
    raise TypeError(f'{path}.repository must be a JSON object')
  if repository.get('dirty') is not False:
    raise ValueError(f'{path} was not produced from a clean repository')
  _lower_hex(repository.get('git_sha'), 40, context=f'{path}.repository.git_sha')
  if repository.get('status_porcelain') != []:
    raise ValueError(f'{path}.repository status is not empty')
  if repository.get('untracked_files') != []:
    raise ValueError(f'{path}.repository records untracked files')

  (result['_runtime_identity'],
   result['_timing_hardware_identity']) = _normalize_runtime_identity(
     result['host'], context=f'{path}.host')

  artifacts = result['artifacts']
  if not isinstance(artifacts, Mapping) or set(artifacts) != {
      'backbone_checkpoint', 'structured_adapter'}:
    raise ValueError(f'{path}.artifacts has an invalid schema')
  result['_artifact_identity'] = {
    'backbone_checkpoint': _normalize_artifact_identity(
      artifacts['backbone_checkpoint'],
      context=f'{path}.artifacts.backbone_checkpoint'),
    'structured_adapter': _normalize_artifact_identity(
      artifacts['structured_adapter'],
      context=f'{path}.artifacts.structured_adapter',
      require_adapter_manifest=True),
  }
  result['_adapter_origin_identity'] = _normalize_adapter_origin_binding(
    result['adapter_origin_evidence'],
    context=f'{path}.adapter_origin_evidence')
  if (result['_adapter_origin_identity']['adapter']['sha256'] !=
      result['_artifact_identity']['structured_adapter']['sha256']
      or result['_adapter_origin_identity']['adapter']['manifest_sha256'] !=
      result['_artifact_identity']['structured_adapter']['manifest_sha256']
      or result['_adapter_origin_identity']['adapter'][
        'structured_decoder_identity_sha256'] !=
      result['_artifact_identity']['structured_adapter']['identity_sha256']):
    raise ValueError(
      f'{path}.adapter_origin_evidence differs from the loaded adapter')
  result['_prompt_identity'] = _normalize_prompt_identity(
    result['prompts'], context=f'{path}.prompts')

  pairing = result['pairing']
  if not isinstance(pairing, Mapping):
    raise TypeError(f'{path}.pairing must be a JSON object')
  expected_pairing_fields = {
    'digest_algorithm', 'global_pairing_digest', 'shard_pairing_digest',
    'base_seed', 'batch_size', 'global_num_samples', 'shard_num_samples',
    'num_shards', 'shard_index', 'sequence_length',
  }
  _strict_fields(pairing, expected_pairing_fields, context=f'{path}.pairing')
  if pairing['digest_algorithm'] != PAIRING_DIGEST_ALGORITHM:
    raise ValueError(f'{path}.pairing uses an unsupported digest algorithm')
  for field in ('global_pairing_digest', 'shard_pairing_digest'):
    _lower_hex(pairing[field], 64, context=f'{path}.pairing.{field}')
  _nonnegative_int(pairing['base_seed'], context=f'{path}.pairing.base_seed')
  for field in (
      'batch_size', 'global_num_samples', 'shard_num_samples', 'num_shards',
      'sequence_length'):
    _positive_int(pairing[field], context=f'{path}.pairing.{field}')
  shard_index = _nonnegative_int(
    pairing['shard_index'], context=f'{path}.pairing.shard_index')
  if shard_index >= pairing['num_shards']:
    raise ValueError(f'{path}.pairing.shard_index lies outside the shard range')
  if (result['_prompt_identity']['source'] == 'jsonl'
      and result['_prompt_identity']['bundle_identity']['policy'][
        'sequence_length'] != pairing['sequence_length']):
    raise ValueError(
      f'{path}.pairing.sequence_length differs from prompt provenance')

  matrix = result['matrix']
  if not isinstance(matrix, Mapping):
    raise TypeError(f'{path}.matrix must be a JSON object')
  _strict_fields(
    matrix, {'sampling_modes', 'nfe_budgets', 'num_output_records'},
    context=f'{path}.matrix')
  modes = matrix['sampling_modes']
  budgets = matrix['nfe_budgets']
  if (not isinstance(modes, list) or not modes
      or any(not isinstance(mode, str) or not mode for mode in modes)
      or len(set(modes)) != len(modes)):
    raise ValueError(f'{path}.matrix.sampling_modes must be unique strings')
  if not set(modes).issubset(SUPPORTED_SAMPLING_MODES):
    raise ValueError(f'{path}.matrix.sampling_modes contains an unknown mode')
  if (not isinstance(budgets, list) or not budgets
      or any(not isinstance(value, int) or isinstance(value, bool)
             or value < 2 for value in budgets)
      or len(set(budgets)) != len(budgets)):
    raise ValueError(f'{path}.matrix.nfe_budgets must be unique integers >= 2')
  declared_records = _positive_int(
    matrix['num_output_records'],
    context=f'{path}.matrix.num_output_records')
  expected_records = pairing['shard_num_samples'] * len(modes) * len(budgets)
  if declared_records != expected_records:
    raise ValueError(f'{path}.matrix record count is not the full Cartesian grid')

  outputs = result['outputs']
  if not isinstance(outputs, Mapping) or set(outputs) != {
      'samples_jsonl', 'summary_json', 'resolved_config'}:
    raise ValueError(f'{path}.outputs has an invalid schema')
  sample_entry = outputs['samples_jsonl']
  summary_entry = outputs['summary_json']
  config_entry = outputs['resolved_config']
  if not isinstance(sample_entry, Mapping):
    raise TypeError(f'{path}.outputs.samples_jsonl must be an object')
  _strict_fields(
    sample_entry, {'path', 'sha256', 'num_records'},
    context=f'{path}.outputs.samples_jsonl')
  for context, entry in (
      (f'{path}.outputs.summary_json', summary_entry),
      (f'{path}.outputs.resolved_config', config_entry)):
    if not isinstance(entry, Mapping):
      raise TypeError(f'{context} must be an object')
    _strict_fields(entry, {'path', 'sha256'}, context=context)
  for name, entry in outputs.items():
    _lower_hex(entry['sha256'], 64, context=f'{path}.outputs.{name}.sha256')
  output_records = _positive_int(
    sample_entry['num_records'],
    context=f'{path}.outputs.samples_jsonl.num_records')
  if output_records != declared_records:
    raise ValueError(f'{path}.outputs.samples_jsonl record count mismatch')

  result['_reference_lm_identity'] = _reference_lm_identity(
    result['reference_lm'], context=f'{path}.reference_lm')
  if result['_reference_lm_identity'] is not None:
    reference_runtime = result['_reference_lm_identity']['runtime_identity']
    host_runtime = result['_runtime_identity']
    expected_shared_runtime = {
      'python': host_runtime['python'],
      'torch': host_runtime['torch'],
      'cuda_runtime': host_runtime['cuda_runtime'],
      'transformers': host_runtime['packages']['transformers'],
      'tokenizers': host_runtime['packages']['tokenizers'],
      'device_type': host_runtime['device_type'],
    }
    observed_shared_runtime = {
      'python': reference_runtime['python'],
      'torch': reference_runtime['torch'],
      'cuda_runtime': reference_runtime['cuda_runtime'],
      'transformers': reference_runtime['transformers'],
      'tokenizers': reference_runtime['tokenizers'],
      'device_type': reference_runtime['device'].split(':', 1)[0].lower(),
    }
    if observed_shared_runtime != expected_shared_runtime:
      raise ValueError(
        f'{path}.reference_lm runtime differs from the generation host')
  return result


def _validate_summary(
    payload: object,
    *,
    manifest: Mapping[str, Any],
    groups: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    path: Path,
) -> None:
  if not isinstance(payload, Mapping):
    raise TypeError(f'{path} must contain a JSON object')
  expected_fields = {
    'schema_version', 'experiment', 'global_pairing_digest',
    'input_pairing_digest', 'global_num_paired_samples',
    'num_paired_samples', 'shard_index', 'num_shards', 'groups',
    'reference_lm',
  }
  _strict_fields(payload, expected_fields, context=str(path))
  pairing = manifest['pairing']
  expected_identity = {
    'schema_version': 1,
    'experiment': EXPERIMENT,
    'global_pairing_digest': pairing['global_pairing_digest'],
    'input_pairing_digest': pairing['shard_pairing_digest'],
    'global_num_paired_samples': pairing['global_num_samples'],
    'num_paired_samples': pairing['shard_num_samples'],
    'shard_index': pairing['shard_index'],
    'num_shards': pairing['num_shards'],
  }
  for field, expected in expected_identity.items():
    _assert_equivalent(
      payload[field], expected, context=f'{path}.{field}')
  claimed_groups = payload['groups']
  if not isinstance(claimed_groups, list):
    raise TypeError(f'{path}.groups must be a JSON array')
  by_group = {}
  for index, group in enumerate(claimed_groups):
    if not isinstance(group, Mapping):
      raise TypeError(f'{path}.groups[{index}] must be a JSON object')
    key = (group.get('sampling_mode'), group.get('requested_nfe_budget'))
    if key in by_group:
      raise ValueError(f'{path} contains a duplicate group summary {key}')
    by_group[key] = group
  if set(by_group) != set(groups):
    raise ValueError(f'{path} group matrix differs from sample records')
  for key, records in groups.items():
    _validate_group_summary(
      by_group[key], records,
      shard_digest=pairing['shard_pairing_digest'],
      context=f'{path}.groups[{key[0]},{key[1]}]')
  reference_summary = _summarize_reference_lm(
    [record for records in groups.values() for record in records])
  if reference_summary is None:
    if payload['reference_lm'] is not None:
      raise ValueError(f'{path}.reference_lm should be null')
  else:
    # The top-level runner includes a device field not present in group
    # summaries.  Validate the statistical values and immutable identity,
    # while treating the execution device as descriptive provenance.
    claimed_reference = payload['reference_lm']
    if not isinstance(claimed_reference, Mapping):
      raise TypeError(f'{path}.reference_lm must be an object')
    for field, expected in reference_summary.items():
      _assert_equivalent(
        claimed_reference.get(field), expected,
        context=f'{path}.reference_lm.{field}')


def _resolve_manifest(raw_path: Path) -> Path:
  path = raw_path.expanduser().resolve()
  if path.is_dir():
    path = path / 'manifest.json'
  if not path.is_file():
    raise FileNotFoundError(path)
  if path.name != 'manifest.json':
    raise ValueError(f'expected a shard directory or manifest.json, got {path}')
  return path


def load_generation_shard(raw_path: Path) -> dict[str, Any]:
  """Load and fully verify one atomic generation shard directory."""
  manifest_path = _resolve_manifest(raw_path)
  manifest = _validate_manifest_header(
    _load_json(manifest_path), path=manifest_path)
  outputs = manifest['outputs']
  sample_path = _safe_output_path(
    manifest_path, outputs['samples_jsonl'],
    context=f'{manifest_path}.outputs.samples_jsonl')
  summary_path = _safe_output_path(
    manifest_path, outputs['summary_json'],
    context=f'{manifest_path}.outputs.summary_json')
  config_path = _safe_output_path(
    manifest_path, outputs['resolved_config'],
    context=f'{manifest_path}.outputs.resolved_config')
  for name, path in (
      ('samples_jsonl', sample_path), ('summary_json', summary_path),
      ('resolved_config', config_path)):
    if sha256_file(path) != outputs[name]['sha256']:
      raise ValueError(f'{path} SHA256 differs from the manifest')

  pairing = manifest['pairing']
  records = []
  mask_token_ids = set()
  with sample_path.open() as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        raise ValueError(f'{sample_path}:{line_number} is blank')
      record, mask_token_id = _validate_record(
        _load_json_line(
          line, source=f'{sample_path}:{line_number}'),
        source=f'{sample_path}:{line_number}', pairing=pairing,
        prompt_identity=manifest['_prompt_identity'],
        modes=manifest['matrix']['sampling_modes'],
        nfe_budgets=manifest['matrix']['nfe_budgets'])
      records.append(record)
      mask_token_ids.add(mask_token_id)
  declared_records = manifest['matrix']['num_output_records']
  if len(records) != declared_records:
    raise ValueError(
      f'{sample_path} has {len(records)} records; expected {declared_records}')
  if len(mask_token_ids) != 1:
    raise ValueError(f'{sample_path} uses inconsistent mask-token identities')

  groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
  duplicate_keys = set()
  observed_keys = set()
  for record in records:
    key = (
      record['sampling_mode'], record['requested_nfe_budget'],
      record['sample_index'])
    if key in observed_keys:
      duplicate_keys.add(key)
    observed_keys.add(key)
    groups[key[:2]].append(record)
  if duplicate_keys:
    raise ValueError(f'{sample_path} contains duplicate matrix/sample keys')
  expected_group_keys = {
    (mode, budget)
    for mode in manifest['matrix']['sampling_modes']
    for budget in manifest['matrix']['nfe_budgets']
  }
  if set(groups) != expected_group_keys:
    raise ValueError(f'{sample_path} does not contain the declared group grid')
  expected_indices = [
    index for index in range(pairing['global_num_samples'])
    if index % pairing['num_shards'] == pairing['shard_index']
  ]
  if len(expected_indices) != pairing['shard_num_samples']:
    raise ValueError(
      f'{manifest_path}.pairing.shard_num_samples is inconsistent with '
      'modulo shard assignment')
  pair_identity_by_index = {}
  pair_key_by_index = {}
  for key, group_records in groups.items():
    actual_indices = [record['sample_index'] for record in group_records]
    if actual_indices != expected_indices:
      raise ValueError(
        f'{sample_path} group {key} does not follow complete ascending '
        'modulo-assigned shard coverage')
    digest = pairing_digest(group_records)
    if digest != pairing['shard_pairing_digest']:
      raise ValueError(f'{sample_path} group {key} pairing digest mismatch')
    for record in group_records:
      index = record['sample_index']
      identity = _pair_identity(record)
      if index in pair_identity_by_index:
        _assert_equivalent(
          identity, pair_identity_by_index[index],
          context=f'{sample_path}.pair_identity[{index}]')
      else:
        pair_identity_by_index[index] = identity
        pair_key_by_index[index] = record['pair_key']
  if len(set(pair_key_by_index.values())) != len(pair_key_by_index):
    raise ValueError(
      f'{sample_path} reuses a pair_key for distinct paired sample draws')

  # Reconstruct the runner's exact ascending shard batches. Merely trusting
  # records that agree with one another would accept a consistently forged
  # seed across all mode/NFE arms.
  expected_batches = [
    expected_indices[offset:offset + pairing['batch_size']]
    for offset in range(0, len(expected_indices), pairing['batch_size'])
  ]
  for key, group_records in groups.items():
    record_by_index = {
      record['sample_index']: record for record in group_records
    }
    for batch_number, batch_indices in enumerate(expected_batches):
      batch_records = [record_by_index[index] for index in batch_indices]
      seed_payload = [
        {
          'pair_key': record['pair_key'],
          'pair_seed': record['pair_seed'],
        }
        for record in batch_records
      ]
      expected_seed = (
        int(canonical_sha256(seed_payload)[:16], 16) % (2 ** 63 - 1))
      for record in batch_records:
        if record['batch_seed'] != expected_seed:
          raise ValueError(
            f'{sample_path} group {key} batch {batch_number} has a batch_seed '
            'that does not commit to its ordered pair_key/pair_seed values')
        if record['timing']['batch_size'] != len(batch_indices):
          raise ValueError(
            f'{sample_path} group {key} batch {batch_number} has an invalid '
            'timing.batch_size (including final partial batch)')

  summary = _load_json(summary_path)
  _validate_summary(
    summary, manifest=manifest, groups=groups, path=summary_path)
  _assert_equivalent(
    manifest['reference_lm'], summary['reference_lm'],
    context=f'{manifest_path}.reference_lm')
  return {
    'manifest_path': manifest_path,
    'manifest_sha256': sha256_file(manifest_path),
    'samples_path': sample_path,
    'summary_path': summary_path,
    'config_path': config_path,
    'manifest': manifest,
    'records': records,
    'groups': dict(groups),
    'pair_identity_by_index': pair_identity_by_index,
    'mask_token_id': next(iter(mask_token_ids)),
  }


def _linear_percentile(sorted_values: Sequence[float], probability: float) -> float:
  if not sorted_values:
    raise ValueError('cannot compute a percentile of an empty sample')
  position = (len(sorted_values) - 1) * probability
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return float(sorted_values[lower])
  weight = position - lower
  return float(
    sorted_values[lower] * (1.0 - weight)
    + sorted_values[upper] * weight)


def _percentile_ci(
    bootstrap_values: list[float],
    *,
    confidence_level: float,
) -> tuple[float, float]:
  bootstrap_values.sort()
  tail = (1.0 - confidence_level) / 2.0
  return (
    _linear_percentile(bootstrap_values, tail),
    _linear_percentile(bootstrap_values, 1.0 - tail),
  )


def paired_bootstrap_intervals(
    values_by_sample: Mapping[int, float],
    prompt_by_sample: Mapping[int, str],
    *,
    num_resamples: int = 20_000,
    rng_seed: int = 91017,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
  """Paired draw and prompt-cluster percentile bootstrap intervals.

  ``values_by_sample`` should already contain paired treatment-minus-baseline
  differences.  Prompt clustering uses equal weight per prompt: it averages
  repeated draws inside a prompt before resampling prompt means.
  """
  if not values_by_sample:
    raise ValueError('paired bootstrap requires at least one paired draw')
  if set(values_by_sample) != set(prompt_by_sample):
    raise ValueError('paired bootstrap sample and prompt keys differ')
  if num_resamples <= 0:
    raise ValueError('bootstrap num_resamples must be positive')
  if not 0.0 < confidence_level < 1.0:
    raise ValueError('bootstrap confidence_level must lie in (0,1)')
  if any(not math.isfinite(float(value)) for value in values_by_sample.values()):
    raise ValueError('bootstrap values must be finite')

  sample_indices = sorted(values_by_sample)
  sample_values = [float(values_by_sample[index]) for index in sample_indices]
  by_prompt: dict[str, list[float]] = defaultdict(list)
  for index in sample_indices:
    prompt = _nonempty_string(
      prompt_by_sample[index], context=f'prompt_by_sample[{index}]')
    by_prompt[prompt].append(float(values_by_sample[index]))
  prompt_means = [
    math.fsum(by_prompt[prompt]) / len(by_prompt[prompt])
    for prompt in sorted(by_prompt)
  ]

  def bootstrap_means(values: Sequence[float], seed: int) -> list[float]:
    # Bound temporary index arrays while keeping a 20k x 256 pilot fast.
    rng = np.random.default_rng(seed)
    value_array = np.asarray(values, dtype=np.float64)
    chunks = []
    remaining = num_resamples
    while remaining:
      chunk_size = min(remaining, 2048)
      indices = rng.integers(
        0, len(value_array), size=(chunk_size, len(value_array)))
      chunks.append(value_array[indices].mean(axis=1))
      remaining -= chunk_size
    return np.concatenate(chunks).tolist()

  sample_bootstrap = bootstrap_means(sample_values, rng_seed)
  prompt_bootstrap = bootstrap_means(prompt_means, rng_seed + 1)
  sample_lower, sample_upper = _percentile_ci(
    sample_bootstrap, confidence_level=confidence_level)
  prompt_lower, prompt_upper = _percentile_ci(
    prompt_bootstrap, confidence_level=confidence_level)
  return {
    'direction': 'treatment_minus_baseline',
    'paired_draws': {
      'method': 'paired_sample_draw_bootstrap_percentile',
      'estimand': 'equal-weight mean across paired sample draws',
      'point_estimate': math.fsum(sample_values) / len(sample_values),
      'num_paired_draws': len(sample_values),
      'num_resamples': num_resamples,
      'rng': 'NumPy Generator (PCG64)',
      'rng_seed': rng_seed,
      'confidence_level': confidence_level,
      'ci_lower': sample_lower,
      'ci_upper': sample_upper,
    },
    'prompt_clusters': {
      'method': 'paired_prompt_cluster_bootstrap_percentile',
      'estimand': (
        'equal-weight mean across prompt means after averaging repeated '
        'paired draws within prompt_id'),
      'point_estimate': math.fsum(prompt_means) / len(prompt_means),
      'num_prompt_clusters': len(prompt_means),
      'num_paired_draws': len(sample_values),
      'draws_per_prompt': {
        prompt: len(by_prompt[prompt]) for prompt in sorted(by_prompt)
      },
      'num_resamples': num_resamples,
      'rng': 'NumPy Generator (PCG64)',
      'rng_seed': rng_seed + 1,
      'confidence_level': confidence_level,
      'ci_lower': prompt_lower,
      'ci_upper': prompt_upper,
      'degenerate_single_cluster': len(prompt_means) == 1,
    },
  }


def _record_endpoints(record: Mapping[str, Any]) -> dict[str, float]:
  result = {
    f'repetition_rate_{n}gram': float(record['metrics']['repetition_rate'][str(n)])
    for n in (1, 2, 4)
  }
  if record['reference_token_ids'] is not None:
    result.update({
      'reference_token_accuracy': float(
        record['metrics']['reference_token_accuracy']),
      'reference_exact_match': float(
        bool(record['metrics']['reference_exact_match'])),
    })
  if 'reference_lm' in record:
    score = record['reference_lm']
    if score['mean_nll_nats'] is not None:
      result['reference_lm_mean_nll_nats'] = float(score['mean_nll_nats'])
  return result


def _comparison(
    baseline_records: Sequence[Mapping[str, Any]],
    treatment_records: Sequence[Mapping[str, Any]],
    *,
    comparison_kind: str,
    num_resamples: int,
    rng_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
  baseline = {record['sample_index']: record for record in baseline_records}
  treatment = {record['sample_index']: record for record in treatment_records}
  if set(baseline) != set(treatment):
    raise ValueError('comparison groups do not contain identical paired draws')
  endpoints_by_sample = {}
  prompt_by_sample = {}
  for index in sorted(baseline):
    if _pair_identity(baseline[index]) != _pair_identity(treatment[index]):
      raise ValueError(f'comparison pair identity differs for sample {index}')
    baseline_endpoints = _record_endpoints(baseline[index])
    treatment_endpoints = _record_endpoints(treatment[index])
    if set(baseline_endpoints) != set(treatment_endpoints):
      raise ValueError(f'comparison endpoint availability differs for sample {index}')
    endpoints_by_sample[index] = {
      endpoint: treatment_endpoints[endpoint] - baseline_endpoints[endpoint]
      for endpoint in baseline_endpoints
    }
    prompt_by_sample[index] = baseline[index]['prompt_id']
  endpoint_sets = {
    frozenset(values) for values in endpoints_by_sample.values()}
  if len(endpoint_sets) != 1:
    raise ValueError('comparison endpoint availability differs across samples')
  endpoint_names = set(next(iter(endpoint_sets)))
  intervals = {}
  for offset, endpoint in enumerate(sorted(endpoint_names)):
    intervals[endpoint] = paired_bootstrap_intervals(
      {
        index: endpoints_by_sample[index][endpoint]
        for index in endpoints_by_sample
      },
      prompt_by_sample,
      num_resamples=num_resamples,
      rng_seed=rng_seed + 2 * offset,
      confidence_level=confidence_level)
  first_baseline = baseline_records[0]
  first_treatment = treatment_records[0]
  return {
    'comparison_kind': comparison_kind,
    'baseline': {
      'sampling_mode': first_baseline['sampling_mode'],
      'requested_nfe_budget': first_baseline['requested_nfe_budget'],
    },
    'treatment': {
      'sampling_mode': first_treatment['sampling_mode'],
      'requested_nfe_budget': first_treatment['requested_nfe_budget'],
    },
    'num_paired_draws': len(baseline),
    'num_prompt_clusters': len(set(prompt_by_sample.values())),
    'endpoints': intervals,
  }


def _descriptive_timing(
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  wall = [float(item['wall_clock_seconds']) for item in timings]
  active_rates = [float(item['active_tokens_per_second']) for item in timings]
  sequence_rates = [
    float(item['sequence_tokens_per_second']) for item in timings
  ]

  def describe(values: Sequence[float]) -> dict[str, float]:
    return {
      'mean': math.fsum(values) / len(values),
      'median': statistics.median(values),
      'minimum': min(values),
      'maximum': max(values),
    }

  return {
    'inferential_status': 'descriptive_only',
    'reason': (
      'each unique sample batch has one timing measurement; different '
      'sample batches are not repeated timing trials'),
    'num_unique_sample_batches': len(timings),
    'repeated_measurements_per_identical_batch': False,
    'wall_clock_seconds_per_batch': describe(wall),
    'active_tokens_per_second_per_batch': describe(active_rates),
    'sequence_tokens_per_second_per_batch': describe(sequence_rates),
    'total_wall_clock_seconds': math.fsum(wall),
    'aggregate_active_tokens_per_second': (
      sum(item['active_tokens'] for item in timings) / math.fsum(wall)),
    'peak_memory_bytes_maximum': max(
      (item['peak_memory_bytes'] for item in timings
       if item['peak_memory_bytes'] is not None),
      default=None),
  }


def aggregate_generation_shards(
    shard_paths: Sequence[Path],
    *,
    baseline_mode: str = 'factorized',
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 91017,
    bootstrap_confidence: float = 0.95,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
  """Verify a complete shard union and return a recomputed result payload."""
  if not shard_paths:
    raise ValueError('at least one generation shard is required')
  if bootstrap_resamples <= 0:
    raise ValueError('bootstrap_resamples must be positive')
  if (not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool)
      or bootstrap_seed < 0):
    raise ValueError('bootstrap_seed must be a non-negative integer')
  if not 0.0 < bootstrap_confidence < 1.0:
    raise ValueError('bootstrap_confidence must lie in (0,1)')
  resolved = [_resolve_manifest(Path(path)) for path in shard_paths]
  if len(set(resolved)) != len(resolved):
    raise ValueError('duplicate shard path')
  shards = [load_generation_shard(path) for path in resolved]
  first_manifest = shards[0]['manifest']
  first_pairing = first_manifest['pairing']
  modes = first_manifest['matrix']['sampling_modes']
  budgets = first_manifest['matrix']['nfe_budgets']
  if baseline_mode not in modes:
    raise ValueError(f'baseline mode {baseline_mode!r} is not in the matrix')

  invariant_identity = {
    'repository': first_manifest['repository'],
    'artifacts': first_manifest['_artifact_identity'],
    'adapter_origin_evidence': first_manifest['_adapter_origin_identity'],
    'prompts': first_manifest['_prompt_identity'],
    'resolved_config_sha256': first_manifest['outputs']['resolved_config']['sha256'],
    'reference_lm': first_manifest['_reference_lm_identity'],
    'runtime': first_manifest['_runtime_identity'],
    'global_pairing_digest': first_pairing['global_pairing_digest'],
    'base_seed': first_pairing['base_seed'],
    'batch_size': first_pairing['batch_size'],
    'global_num_samples': first_pairing['global_num_samples'],
    'num_shards': first_pairing['num_shards'],
    'sequence_length': first_pairing['sequence_length'],
    'sampling_modes': modes,
    'nfe_budgets': budgets,
  }
  timing_hardware_identity = first_manifest['_timing_hardware_identity']
  indices = []
  for shard in shards:
    manifest = shard['manifest']
    pairing = manifest['pairing']
    observed_identity = {
      'repository': manifest['repository'],
      'artifacts': manifest['_artifact_identity'],
      'adapter_origin_evidence': manifest['_adapter_origin_identity'],
      'prompts': manifest['_prompt_identity'],
      'resolved_config_sha256': manifest['outputs']['resolved_config']['sha256'],
      'reference_lm': manifest['_reference_lm_identity'],
      'runtime': manifest['_runtime_identity'],
      'global_pairing_digest': pairing['global_pairing_digest'],
      'base_seed': pairing['base_seed'],
      'batch_size': pairing['batch_size'],
      'global_num_samples': pairing['global_num_samples'],
      'num_shards': pairing['num_shards'],
      'sequence_length': pairing['sequence_length'],
      'sampling_modes': manifest['matrix']['sampling_modes'],
      'nfe_budgets': manifest['matrix']['nfe_budgets'],
    }
    _assert_equivalent(
      observed_identity, invariant_identity,
      context=f'{shard["manifest_path"]}.cross_shard_identity')
    if manifest['_timing_hardware_identity'] != timing_hardware_identity:
      raise ValueError(
        f'{shard["manifest_path"]}.cross_shard_timing_identity differs; '
        'refusing to pool timing across different GPU identities')
    indices.append(pairing['shard_index'])
  if len(set(indices)) != len(indices):
    raise ValueError('duplicate generation shard index')
  expected_shards = list(range(first_pairing['num_shards']))
  if sorted(indices) != expected_shards:
    raise ValueError(
      f'incomplete shard coverage: observed={sorted(indices)}, '
      f'expected={expected_shards}')

  global_pairs = {}
  pair_keys = {}
  prompt_identity = {}
  mask_token_ids = set()
  union_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
  for shard in sorted(
      shards, key=lambda item: item['manifest']['pairing']['shard_index']):
    mask_token_ids.add(shard['mask_token_id'])
    for index, identity in shard['pair_identity_by_index'].items():
      if index in global_pairs:
        raise ValueError(f'paired sample index {index} occurs in multiple shards')
      global_pairs[index] = identity
      pair_keys[index] = identity['pair_key']
      prompt_id = identity['prompt_id']
      identity_without_draw = {
        'initial_token_ids': identity['initial_token_ids'],
        'active_mask': identity['active_mask'],
        'reference_token_ids': identity['reference_token_ids'],
        'prompt_metadata': identity['prompt_metadata'],
      }
      if prompt_id in prompt_identity:
        _assert_equivalent(
          identity_without_draw, prompt_identity[prompt_id],
          context=f'prompt_identity[{prompt_id!r}]')
      else:
        prompt_identity[prompt_id] = identity_without_draw
    for key, records in shard['groups'].items():
      union_groups[key].extend(records)
  if len(mask_token_ids) != 1:
    raise ValueError('mask-token identity differs across shards')
  expected_indices = list(range(first_pairing['global_num_samples']))
  if sorted(global_pairs) != expected_indices:
    raise ValueError('paired sample union is not exactly indices [0,N)')
  if len(set(pair_keys.values())) != len(pair_keys):
    raise ValueError('pair_key values are not unique across paired sample draws')
  global_digest = canonical_sha256([
    _pair_identity(global_pairs[index]) for index in expected_indices
  ])
  if global_digest != first_pairing['global_pairing_digest']:
    raise ValueError('global pairing digest does not match the shard union')

  prompt_record_count = first_manifest['_prompt_identity']['num_prompt_records']
  if len(prompt_identity) > prompt_record_count:
    raise ValueError('observed prompt IDs exceed the prompt provenance count')
  if (first_pairing['global_num_samples'] >= prompt_record_count
      and len(prompt_identity) != prompt_record_count):
    raise ValueError(
      'observed unique prompt count differs from prompt provenance count')

  expected_groups = {
    (mode, budget) for mode in modes for budget in budgets
  }
  if set(union_groups) != expected_groups:
    raise ValueError('unioned group matrix is incomplete')
  recomputed_groups = []
  records_by_group = {}
  canonical_union_records = []
  for mode in modes:
    for budget in budgets:
      key = (mode, budget)
      records = sorted(
        union_groups[key], key=lambda record: record['sample_index'])
      if [record['sample_index'] for record in records] != expected_indices:
        raise ValueError(f'unioned group {key} has incomplete sample coverage')
      if pairing_digest(records) != global_digest:
        raise ValueError(f'unioned group {key} has inconsistent sample pairing')
      summary, timings = _raw_group_summary(
        records, context=f'union.groups[{mode},{budget}]')
      summary.update({
        'records_sha256': canonical_sha256(records),
        'timing': _descriptive_timing(timings),
        'reference_lm': _summarize_reference_lm(records),
      })
      recomputed_groups.append(summary)
      records_by_group[key] = records
      canonical_union_records.extend(records)

  comparisons = []
  comparison_index = 0
  ordered_modes = [baseline_mode] + [
    mode for mode in modes if mode != baseline_mode
  ]
  for budget in budgets:
    for baseline_index, comparison_baseline_mode in enumerate(ordered_modes):
      for treatment_mode in ordered_modes[baseline_index + 1:]:
        comparisons.append(_comparison(
          records_by_group[(comparison_baseline_mode, budget)],
          records_by_group[(treatment_mode, budget)],
          comparison_kind='sampling_mode_at_fixed_nfe',
          num_resamples=bootstrap_resamples,
          rng_seed=bootstrap_seed + 1000 * comparison_index,
          confidence_level=bootstrap_confidence))
        comparison_index += 1
  lowest_budget = min(budgets)
  for mode in modes:
    for treatment_budget in sorted(budgets):
      if treatment_budget == lowest_budget:
        continue
      comparisons.append(_comparison(
        records_by_group[(mode, lowest_budget)],
        records_by_group[(mode, treatment_budget)],
        comparison_kind='nfe_budget_within_sampling_mode',
        num_resamples=bootstrap_resamples,
        rng_seed=bootstrap_seed + 1000 * comparison_index,
        confidence_level=bootstrap_confidence))
      comparison_index += 1

  prompt_draw_counts = Counter(
    identity['prompt_id'] for identity in global_pairs.values())
  expected_record_count = (
    first_pairing['global_num_samples'] * len(modes) * len(budgets))
  if len(canonical_union_records) != expected_record_count:
    raise AssertionError('internal union record-count invariant failed')
  created = timestamp_utc or dt.datetime.now(dt.timezone.utc).isoformat()
  return {
    'schema_version': 1,
    'artifact': 'verified_generation_shard_union',
    'experiment': EXPERIMENT,
    'created_utc': created,
    'scope_note': (
      'Generation quality summaries and paired bootstrap intervals; not a '
      'diffusion ELBO or likelihood estimate. Timing is descriptive unless '
      'the collection protocol adds repeated identical timing trials.'),
    'identity': invariant_identity,
    'coverage': {
      'num_shards': first_pairing['num_shards'],
      'shard_indices': sorted(indices),
      'global_num_paired_draws': first_pairing['global_num_samples'],
      'num_unique_prompts': len(prompt_identity),
      'paired_draws_per_prompt': {
        prompt: prompt_draw_counts[prompt]
        for prompt in sorted(prompt_draw_counts)
      },
      'num_sampling_modes': len(modes),
      'num_nfe_budgets': len(budgets),
      'num_groups': len(expected_groups),
      'expected_output_records': expected_record_count,
      'verified_output_records': len(canonical_union_records),
      'global_pairing_digest': global_digest,
      'record_digest_algorithm': 'sha256-canonical-json-array-v1',
      'canonical_union_records_sha256': canonical_sha256(
        canonical_union_records),
    },
    'input_shards': [
      {
        'shard_index': shard['manifest']['pairing']['shard_index'],
        'manifest_path': str(shard['manifest_path']),
        'manifest_sha256': shard['manifest_sha256'],
        'samples_sha256': shard['manifest']['outputs']['samples_jsonl']['sha256'],
        'num_records': shard['manifest']['matrix']['num_output_records'],
      }
      for shard in sorted(
        shards, key=lambda item: item['manifest']['pairing']['shard_index'])
    ],
    'groups': recomputed_groups,
    'comparisons': comparisons,
    'bootstrap': {
      'paired_draw_resampling': True,
      'prompt_cluster_resampling': True,
      'repeated_prompt_draws_are_clustered_by': 'prompt_id',
      'pairing_scope': (
        'intervals condition on the frozen shard and batch layout; the '
        'generation harness seeds each batch from its ordered paired draws'),
      'num_resamples': bootstrap_resamples,
      'base_rng_seed': bootstrap_seed,
      'confidence_level': bootstrap_confidence,
    },
    'timing_policy': {
      'status': 'descriptive_only',
      'hardware_identity': timing_hardware_identity,
      'reason': (
        'the generation harness records one timing measurement per unique '
        'sample batch, not repeated trials of identical work'),
    },
  }
