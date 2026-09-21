"""Fail-closed, replayable topology diagnostics for contextual forests.

The model-facing artifact is deliberately just a canonical edge list plus
source and intervention commitments.  All paper-facing quantities are
recomputed here on CPU.  This keeps the analysis independent of model code
and makes a reported result replayable from the committed JSONL records.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ARTIFACT = 'contextual_forest_topology_diagnostics_protocol'
RECORD_ARTIFACT = 'contextual_forest_topology_diagnostic_record'
MANIFEST_ARTIFACT = 'contextual_forest_topology_record_manifest'
ANALYSIS_ARTIFACT = 'contextual_forest_topology_diagnostics_analysis'
SOURCE_SELECTION_ARTIFACT = 'contextual_forest_topology_source_selection'
SOURCE_INTEGRITY_ARTIFACT = 'contextual_forest_topology_source_integrity'
GPU_EXCLUSIVITY_ARTIFACT = 'contextual_forest_topology_gpu_exclusivity'
GPU_EXCLUSIVITY_POLICY = (
  'nonblocking_flock_and_continuous_pid_monitor_v1')
SUBMISSION_GPU_LOCK = Path(
  '/mnt/contextual-forest/experiments/.submission-gpu.lock')
GPU_MONITOR_INTERVAL_SECONDS = 1.0
SCHEMA_VERSION = 2

INTERVENTIONS = (
  'learned',
  'matched_permuted',
  'fixed_time',
  'zero_time',
  'timestep_shuffled',
)
PROTOCOL_FIELDS = {
  'schema_version', 'artifact', 'protocol_id', 'protocol_status',
  'candidate_top_k',
  'scientific_scope', 'corruption_seeds', 'time_points', 'interventions',
  'time_parameterization', 'topology_head_time_transform',
  'natural_order_chain', 'nonlocal_edge_threshold',
  'component_depth_root', 'component_size_cap', 'completeness',
  'corruption_policy', 'determinism', 'evaluator_source_path',
  'intervention_locus', 'require_nonempty_learned_forest',
  'source_selection',
}
SOURCE_BINDING_FIELDS = {
  'schema_version', 'artifact', 'job_id', 'compiled_plan_sha256',
  'plan_id', 'job_spec_sha256', 'job_execution_sha256',
  'repository_sha', 'repository_clean', 'adapter_sha256',
  'adapter_export_manifest_sha256', 'data_config_sha256',
  'dataset_provenance_sha256', 'evaluator_source_sha256',
  'arm', 'dataset', 'train_seed', 'source_selection_sha256',
}
RECORD_FIELDS = {
  'schema_version', 'artifact', 'record_id', 'protocol_id',
  'protocol_sha256', 'source_binding_sha256', 'job_id', 'dataset',
  'dataset_revision', 'train_seed', 'source_unit_id', 'document_id',
  'document_sha256', 'selection_index', 'chunk_index',
  'clean_example_sha256',
  'sequence_length',
  'corruption_seed', 'base_noise_sha256', 'corrupted_tokens_sha256',
  'attention_mask_sha256', 'active_mask_sha256',
  'corruption_context_sha256',
  'requested_time_index', 'requested_time', 'effective_time',
  'intervention', 'intervention_metadata', 'active_nodes',
  'selected_edges',
}
INTERVENTION_METADATA_FIELDS = {
  'reference_record_id', 'intervention_seed', 'time_donor_record_id',
  'time_donor_index', 'node_permutation',
}
MANIFEST_FIELDS = {
  'schema_version', 'artifact', 'protocol_id', 'protocol_sha256',
  'source_binding', 'source_binding_sha256', 'record_files',
  'num_records', 'manifest_sha256',
}
RECORD_FILE_FIELDS = {'path', 'sha256', 'num_records'}
SOURCE_DESCRIPTOR_FIELDS = {
  'dataset', 'dataset_revision', 'selection_index', 'source_unit_id',
  'document_id', 'document_sha256', 'chunk_index',
  'clean_example_sha256', 'sequence_length',
}
SOURCE_SELECTION_FIELDS = {
  'schema_version', 'artifact', 'protocol_id', 'protocol_sha256',
  'dataset', 'dataset_revision', 'tokenizer_revision', 'selection_policy',
  'entries', 'selection_sha256', 'manifest_sha256',
}
CORRUPTION_POLICY_FIELDS = {
  'forward_process', 'base_noise', 'time_coupling', 'active_nodes',
  'active_mask_nesting', 'context_commitment', 'mask_threshold',
}
ANALYSIS_FIELDS = {
  'schema_version', 'artifact', 'scientific_scope', 'protocol_id',
  'protocol_sha256', 'source_manifests', 'source_integrity',
  'input_commitment_sha256', 'grid_validation', 'metric_definitions',
  'observation_metrics', 'observation_metrics_sha256',
  'learned_topology_summary', 'learned_topology_by_dataset',
  'by_intervention_and_requested_time',
  'learned_edge_stability', 'intervention_comparisons', 'analysis_sha256',
}
SOURCE_INTEGRITY_FIELDS = {
  'schema_version', 'artifact', 'protocol_id', 'protocol_sha256',
  'compiled_plan_sha256', 'plan_id', 'source_manifest_sha256',
  'repository_sha', 'repository_clean', 'validated_job_ids', 'jobs',
  'dependencies', 'commitment_sha256',
}
GPU_EXCLUSIVITY_FIELDS = {
  'schema_version', 'artifact', 'job_id', 'required', 'policy', 'lock_path',
  'lock_acquired', 'monitor_interval_seconds', 'monitor_samples',
  'preflight_other_compute_pids', 'postflight_other_compute_pids',
  'foreign_pid_observations', 'monitor_errors',
}


def canonical_json(value: Any) -> str:
  """Return the one serialization used by all topology commitments."""
  return json.dumps(
    value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    allow_nan=False)


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(chunk_size), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
  result = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f'duplicate JSON object key: {key!r}')
    result[key] = value
  return result


def _reject_nonfinite_json(value: str) -> None:
  raise ValueError(f'non-finite JSON number: {value}')


def load_json(path: Path) -> Any:
  try:
    with path.open(encoding='utf-8') as handle:
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


def _strict_fields(
    value: object,
    fields: set[str],
    *,
    context: str,
) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise TypeError(f'{context} must be a JSON object')
  missing = fields - set(value)
  unknown = set(value) - fields
  if missing or unknown:
    raise ValueError(
      f'{context} schema mismatch: missing={sorted(missing)}, '
      f'unknown={sorted(unknown)}')
  return value


def _nonempty_string(value: object, *, context: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f'{context} must be a non-empty string')
  return value


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
    maximum: float | None = None,
) -> float:
  if (not isinstance(value, (int, float)) or isinstance(value, bool)
      or not math.isfinite(float(value))):
    raise ValueError(f'{context} must be a finite number')
  result = float(value)
  if minimum is not None and result < minimum:
    raise ValueError(f'{context} must be >= {minimum}')
  if maximum is not None and result > maximum:
    raise ValueError(f'{context} must be <= {maximum}')
  return result


def validate_protocol(value: object) -> dict[str, Any]:
  payload = dict(_strict_fields(
    value, PROTOCOL_FIELDS, context='topology protocol'))
  if payload['schema_version'] != SCHEMA_VERSION \
      or payload['artifact'] != PROTOCOL_ARTIFACT:
    raise ValueError('invalid topology protocol identity')
  _nonempty_string(payload['protocol_id'], context='protocol_id')
  if payload['protocol_status'] != 'frozen_before_topology_results':
    raise ValueError(
      'protocol_status must equal frozen_before_topology_results')
  _nonempty_string(payload['scientific_scope'], context='scientific_scope')
  if payload['determinism'] != 'torch_eval_no_grad_dropout_disabled':
    raise ValueError('unsupported topology determinism policy')
  if payload['intervention_locus'] != (
      'structured_decoder_topology_time_input_only_'
      'reuse_backbone_hidden_unary_v1'):
    raise ValueError('unsupported topology intervention locus')
  evaluator_source = Path(_nonempty_string(
    payload['evaluator_source_path'], context='evaluator_source_path'))
  if (evaluator_source.is_absolute() or '..' in evaluator_source.parts
      or evaluator_source.as_posix() != payload['evaluator_source_path']):
    raise ValueError('evaluator_source_path must be a safe repository path')
  corruption_policy = _strict_fields(
    payload['corruption_policy'], CORRUPTION_POLICY_FIELDS,
    context='corruption_policy')
  expected_corruption_policy = {
    'forward_process': 'absorbing_mask_diffusion',
    'base_noise': (
      'sha256_uint53_per_position_v1_one_private_vector_per_'
      'source_unit_x_corruption_seed'),
    'time_coupling': (
      'reuse_identical_base_noise_across_time_grid_and_train_seeds'),
    'active_nodes': 'mask_token_and_attention_mask',
    'active_mask_nesting': 'active_sets_nondecreasing_with_time',
    'context_commitment': 'canonical_corruption_hash_bundle_v1',
    'mask_threshold': 'uint53_lt_floor_requested_probability_times_2pow53',
  }
  if dict(corruption_policy) != expected_corruption_policy:
    raise ValueError('unsupported topology corruption policy')
  payload['candidate_top_k'] = _positive_int(
    payload['candidate_top_k'], context='candidate_top_k')
  if payload['time_parameterization'] != 'absorbing_mask_probability':
    raise ValueError('unsupported topology time parameterization')
  if payload['topology_head_time_transform'] != \
      'negative_log1p_one_minus_probability_v1':
    raise ValueError('unsupported topology-head time transform')
  seeds = payload['corruption_seeds']
  if (not isinstance(seeds, list) or len(seeds) < 2
      or any(_nonnegative_int(seed, context='corruption_seed') != seed
             for seed in seeds)
      or seeds != sorted(set(seeds))):
    raise ValueError(
      'corruption_seeds must contain at least two unique increasing integers')
  times = payload['time_points']
  if not isinstance(times, list) or len(times) < 2:
    raise ValueError('time_points must contain at least two values')
  normalized_times = []
  for value in times:
    normalized = _finite_float(
      value, context='time_point', minimum=0.0, maximum=1.0)
    if normalized >= 1.0:
      raise ValueError('time_points must be strictly below one')
    normalized_times.append(normalized)
  if normalized_times != sorted(set(normalized_times)):
    raise ValueError('time_points must be unique and increasing')
  payload['time_points'] = normalized_times
  interventions = _strict_fields(
    payload['interventions'], set(INTERVENTIONS),
    context='protocol.interventions')
  _strict_fields(
    interventions['learned'], {'effective_time'},
    context='interventions.learned')
  if interventions['learned']['effective_time'] != 'requested':
    raise ValueError('learned effective_time must equal requested')
  matched = _strict_fields(
    interventions['matched_permuted'], {
      'algorithm', 'effective_time', 'permutation_seed',
      'minimum_pooled_edge_set_changed_fraction'},
    context='interventions.matched_permuted')
  if matched['algorithm'] != 'sha256_sort_active_nodes_v1':
    raise ValueError('unsupported matched node-permutation algorithm')
  if matched['effective_time'] != 'requested':
    raise ValueError('matched_permuted effective_time must equal requested')
  _nonnegative_int(
    matched['permutation_seed'], context='matched permutation_seed')
  _finite_float(
    matched['minimum_pooled_edge_set_changed_fraction'],
    context='matched minimum changed fraction', minimum=0.0, maximum=1.0)
  fixed = _strict_fields(
    interventions['fixed_time'], {'effective_time'},
    context='interventions.fixed_time')
  fixed_time = _finite_float(
    fixed['effective_time'], context='fixed effective_time',
    minimum=0.0, maximum=1.0)
  if fixed_time >= 1.0:
    raise ValueError('fixed effective_time must be strictly below one')
  zero = _strict_fields(
    interventions['zero_time'], {'effective_time'},
    context='interventions.zero_time')
  if _finite_float(
      zero['effective_time'], context='zero effective_time',
      minimum=0.0, maximum=1.0) != 0.0:
    raise ValueError('zero_time effective_time must be 0')
  shuffled = _strict_fields(
    interventions['timestep_shuffled'], {
      'algorithm', 'effective_time', 'shuffle_seed'},
    context='interventions.timestep_shuffled')
  if shuffled['algorithm'] != 'sha256_sort_rotate_time_grid_v1':
    raise ValueError('unsupported timestep shuffle algorithm')
  if shuffled['effective_time'] != 'deterministic_permutation_of_time_grid':
    raise ValueError('invalid timestep_shuffled effective_time policy')
  _nonnegative_int(shuffled['shuffle_seed'], context='shuffle_seed')
  if payload['natural_order_chain'] != 'consecutive_active_positions':
    raise ValueError('unsupported natural_order_chain definition')
  threshold = _nonnegative_int(
    payload['nonlocal_edge_threshold'], context='nonlocal_edge_threshold')
  if threshold == 0:
    raise ValueError('nonlocal_edge_threshold must be positive')
  if payload['component_depth_root'] != 'minimum_active_position':
    raise ValueError('unsupported component_depth_root definition')
  payload['component_size_cap'] = _positive_int(
    payload['component_size_cap'], context='component_size_cap')
  if payload['completeness'] != (
      'exact_source_unit_x_corruption_seed_x_time_point_x_intervention_grid'):
    raise ValueError('unsupported completeness policy')
  if not isinstance(payload['require_nonempty_learned_forest'], bool):
    raise ValueError('require_nonempty_learned_forest must be boolean')
  selection = _strict_fields(
    payload['source_selection'], {
      'arm', 'bundling', 'datasets', 'train_seeds',
      'require_identical_source_units_across_train_seeds',
      'source_unit_order'},
    context='source_selection')
  _nonempty_string(selection['arm'], context='source_selection.arm')
  if selection['bundling'] != 'one_bundle_per_dataset_x_train_seed':
    raise ValueError('unsupported source-selection bundling policy')
  if selection['source_unit_order'] != (
      'first_n_pinned_document_local_eval_order'):
    raise ValueError('unsupported source-unit ordering policy')
  if selection['require_identical_source_units_across_train_seeds'] is not True:
    raise ValueError(
      'source units must be identical across training seeds')
  train_seeds = selection['train_seeds']
  if (not isinstance(train_seeds, list) or not train_seeds
      or any(_nonnegative_int(seed, context='source train_seed') != seed
             for seed in train_seeds)
      or train_seeds != sorted(set(train_seeds))):
    raise ValueError(
      'source_selection.train_seeds must be unique and increasing')
  datasets = selection['datasets']
  if not isinstance(datasets, Mapping) or not datasets:
    raise ValueError('source_selection.datasets must be a non-empty object')
  for dataset, specification in datasets.items():
    _nonempty_string(dataset, context='source dataset')
    specification = _strict_fields(
      specification, {
        'num_source_units', 'data_config_path', 'dataset_revision',
        'tokenizer_revision'},
      context=f'source_selection.datasets.{dataset}')
    _positive_int(
      specification['num_source_units'],
      context=f'source_selection.datasets.{dataset}.num_source_units')
    data_config_path = Path(_nonempty_string(
      specification['data_config_path'],
      context=f'source_selection.datasets.{dataset}.data_config_path'))
    if (data_config_path.is_absolute() or '..' in data_config_path.parts
        or data_config_path.as_posix() != specification['data_config_path']):
      raise ValueError(f'{dataset} data_config_path is not repository-relative')
    _lower_hex(
      specification['dataset_revision'], 40,
      context=f'source_selection.datasets.{dataset}.dataset_revision')
    _lower_hex(
      specification['tokenizer_revision'], 40,
      context=f'source_selection.datasets.{dataset}.tokenizer_revision')
  return payload


def read_protocol(path: Path) -> tuple[dict[str, Any], str]:
  payload = validate_protocol(load_json(path))
  return payload, canonical_sha256(payload)


def _validate_trusted_protocol_path(
    path: Path, protocol: Mapping[str, Any],
) -> None:
  expected = (
    REPO_ROOT / 'configs' / 'evaluation'
    / f'{protocol["protocol_id"]}.json').resolve()
  if path.expanduser().resolve() != expected:
    raise ValueError(
      'authoritative topology aggregation requires the repository-trusted '
      f'protocol path {expected}')


def validate_compiled_topology_plan_lineage(
    plan: Mapping[str, Any],
    *,
    jobs: Mapping[str, Mapping[str, Any]],
    plan_dir: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
  """Rehash the derived plan's frozen protocol and authenticated parent."""
  plan_dir = plan_dir.expanduser().resolve()
  protocol_path = protocol_path.expanduser().resolve()
  if plan.get('selected_suites') != ['topology_diagnostics']:
    raise ValueError('authoritative topology plan must use its dedicated suite')
  if Path(plan.get('compiled_plan_dir', '')).expanduser().resolve() != plan_dir:
    raise ValueError('topology plan directory commitment drifted')
  topology = dict(_strict_fields(
    plan.get('topology_protocol'), {
      'path', 'protocol_id', 'canonical_sha256', 'file_sha256',
      'protocol_status'},
    context='compiled topology protocol binding'))
  if (Path(topology['path']).expanduser().resolve() != protocol_path
      or topology['protocol_id'] != protocol['protocol_id']
      or topology['canonical_sha256'] != protocol_sha256
      or topology['file_sha256'] != sha256_file(protocol_path)
      or topology['protocol_status'] != protocol['protocol_status']
      ):
    raise ValueError('compiled topology protocol binding drifted')
  source = dict(_strict_fields(
    plan.get('source_compiled_plan'), {'path', 'sha256', 'plan_id'},
    context='source compiled-plan binding'))
  source_path = Path(source['path']).expanduser().resolve()
  if source_path == plan_dir / 'compiled-plan.json' or not source_path.is_file():
    raise ValueError('source compiled plan is missing or self-referential')
  if sha256_file(source_path) != source['sha256']:
    raise ValueError('source compiled plan hash drifted')
  source_plan = load_json(source_path)
  promotion = plan.get('promotion_evidence')
  if (not isinstance(promotion, Mapping)
      or set(promotion) != {'candidate_k_128_confirmation'}):
    raise ValueError('topology plan lacks exact K=128 promotion evidence')
  promotion_entry = dict(_strict_fields(
    promotion['candidate_k_128_confirmation'], {
      'path', 'sha256', 'source_suite', 'route_name',
      'canonical_decision_sha256', 'source_compiled_plan_sha256'},
    context='topology K=128 promotion evidence'))
  promotion_path = Path(promotion_entry['path']).expanduser().resolve()
  if (not promotion_path.is_file()
      or sha256_file(promotion_path) != promotion_entry['sha256']
      or promotion_entry['source_suite'] != 'candidate_k_128_pilot'
      or promotion_entry['route_name'] != 'confirmation'):
    raise ValueError('topology K=128 promotion evidence drifted')
  if (not isinstance(source_plan, Mapping)
      or source_plan.get('plan_id') != source['plan_id']
      or source_plan.get('source_manifest_sha256')
      != plan.get('source_manifest_sha256')
      or source_plan.get('repository') != plan.get('repository')
      or source_plan.get('promotion_evidence', {}).get(
        'candidate_k_128_confirmation') != promotion_entry):
    raise ValueError('source compiled plan identity differs from derived plan')

  # The derived plan lives outside Git, so its internal job-spec hashes are
  # not an independent authenticity boundary. Re-run the repository compiler
  # from the hash-bound parent and compare every plan field and job byte-for-
  # byte. This binds copied train/export jobs to the promoted parent and binds
  # each eval job to the reviewed emitter argv, dependency, inputs, and output
  # contract.
  from scripts.compile_topology_diagnostics import (  # noqa: PLC0415
    derive_topology_plan,
  )
  expected_plan, expected_jobs, expected_dir = derive_topology_plan(
    source_plan_dir=source_path.parent,
    output_dir=plan_dir,
    protocol_path=protocol_path)
  if expected_dir != plan_dir:
    raise AssertionError('canonical topology compiler output path drifted')
  if set(jobs) != set(expected_jobs):
    raise ValueError(
      'compiled topology job set differs from canonical compiler reconstruction')
  for job_id in expected_jobs:
    if canonical_json(jobs[job_id]) != canonical_json(expected_jobs[job_id]):
      raise ValueError(
        f'compiled topology job {job_id!r} differs from canonical compiler '
        'reconstruction')
  if canonical_json(plan) != canonical_json(expected_plan):
    raise ValueError(
      'compiled topology plan differs from canonical compiler reconstruction')
  return source


def validate_source_binding(value: object) -> dict[str, Any]:
  payload = dict(_strict_fields(
    value, SOURCE_BINDING_FIELDS, context='source_binding'))
  if payload['schema_version'] != SCHEMA_VERSION \
      or payload['artifact'] != 'contextual_forest_topology_source_binding':
    raise ValueError('invalid topology source binding identity')
  _nonempty_string(payload['job_id'], context='source_binding.job_id')
  _nonempty_string(payload['arm'], context='source_binding.arm')
  _nonempty_string(payload['dataset'], context='source_binding.dataset')
  payload['train_seed'] = _nonnegative_int(
    payload['train_seed'], context='source_binding.train_seed')
  _lower_hex(
    payload['compiled_plan_sha256'], 64,
    context='source_binding.compiled_plan_sha256')
  _lower_hex(payload['plan_id'], 64, context='source_binding.plan_id')
  _lower_hex(
    payload['job_spec_sha256'], 64,
    context='source_binding.job_spec_sha256')
  _lower_hex(
    payload['job_execution_sha256'], 64,
    context='source_binding.job_execution_sha256')
  _lower_hex(
    payload['repository_sha'], 40,
    context='source_binding.repository_sha')
  if payload['repository_clean'] is not True:
    raise ValueError('topology evidence requires a clean repository checkout')
  for field in (
      'adapter_sha256', 'adapter_export_manifest_sha256',
      'data_config_sha256', 'dataset_provenance_sha256',
      'evaluator_source_sha256', 'source_selection_sha256'):
    _lower_hex(payload[field], 64, context=f'source_binding.{field}')
  return payload


def record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
  """Return the immutable experimental coordinates behind a record ID."""
  return {
    'protocol_id': record['protocol_id'],
    'protocol_sha256': record['protocol_sha256'],
    'source_binding_sha256': record['source_binding_sha256'],
    'job_id': record['job_id'],
    'dataset': record['dataset'],
    'dataset_revision': record['dataset_revision'],
    'train_seed': record['train_seed'],
    'selection_index': record['selection_index'],
    'source_unit_id': record['source_unit_id'],
    'clean_example_sha256': record['clean_example_sha256'],
    'corruption_seed': record['corruption_seed'],
    'requested_time_index': record['requested_time_index'],
    'intervention': record['intervention'],
  }


def record_id_for(record: Mapping[str, Any]) -> str:
  return canonical_sha256(record_identity(record))


def active_mask_sha256_for(
    *, sequence_length: int, active_nodes: Sequence[int],
) -> str:
  """Commit the exact boolean active mask without storing a dense vector."""
  return canonical_sha256({
    'sequence_length': sequence_length,
    'active_nodes': list(active_nodes),
  })


def corruption_context_sha256_for(record: Mapping[str, Any]) -> str:
  """Commit every corruption input needed to pair topology interventions."""
  return canonical_sha256({
    'clean_example_sha256': record['clean_example_sha256'],
    'corruption_seed': record['corruption_seed'],
    'requested_time_index': record['requested_time_index'],
    'requested_time': record['requested_time'],
    'base_noise_sha256': record['base_noise_sha256'],
    'corrupted_tokens_sha256': record['corrupted_tokens_sha256'],
    'attention_mask_sha256': record['attention_mask_sha256'],
    'active_mask_sha256': record['active_mask_sha256'],
  })


_UINT53_SCALE = 1 << 53
BASE_NOISE_ALGORITHM = 'sha256_uint53_per_position_v1'


def topology_head_time_for_probability(probability: float) -> float:
  """Map an absorbing-mask probability to the head's sigma conditioning."""
  probability = _finite_float(
    probability, context='absorbing-mask probability',
    minimum=0.0, maximum=1.0)
  if probability >= 1.0:
    raise ValueError('absorbing-mask probability must be strictly below one')
  return -math.log1p(-probability)


def sequence_sha256(values: Sequence[Any], *, dtype: str) -> str:
  """Commit a one-dimensional model input in a platform-neutral encoding."""
  if dtype not in {'bool', 'int64'}:
    raise ValueError(f'unsupported committed sequence dtype: {dtype}')
  if dtype == 'bool':
    normalized = [bool(value) for value in values]
  else:
    normalized = [
      _nonnegative_int(value, context='committed token') for value in values]
  return canonical_sha256({
    'dtype': dtype,
    'shape': [len(normalized)],
    'values': normalized,
  })


def clean_example_sha256_for(
    input_ids: Sequence[int], attention_mask: Sequence[bool],
) -> str:
  if len(input_ids) != len(attention_mask) or not input_ids:
    raise ValueError('clean input and attention mask lengths must match')
  return canonical_sha256({
    'input_ids_sha256': sequence_sha256(input_ids, dtype='int64'),
    'attention_mask_sha256': sequence_sha256(
      attention_mask, dtype='bool'),
  })


def deterministic_base_noise_uint53(
    *,
    source_descriptor: Mapping[str, Any],
    corruption_seed: int,
    sequence_length: int,
) -> tuple[list[int], str]:
  """Return the frozen per-position private noise vector and commitment.

  Integer uniforms avoid device, dtype, and PyTorch-generator drift. The
  source descriptor contains only committed identifiers and content hashes;
  training seed and adapter identity are intentionally absent.
  """
  descriptor = dict(_strict_fields(
    source_descriptor, SOURCE_DESCRIPTOR_FIELDS,
    context='base-noise source descriptor'))
  corruption_seed = _nonnegative_int(
    corruption_seed, context='base-noise corruption_seed')
  sequence_length = _positive_int(
    sequence_length, context='base-noise sequence_length')
  if descriptor['sequence_length'] != sequence_length:
    raise ValueError('base-noise descriptor sequence length differs')
  domain = canonical_json({
    'algorithm': BASE_NOISE_ALGORITHM,
    'source_descriptor': descriptor,
    'corruption_seed': corruption_seed,
  }).encode('utf-8')
  values = []
  for position in range(sequence_length):
    digest = hashlib.sha256(
      domain + b'|' + str(position).encode('ascii')).digest()
    values.append(int.from_bytes(digest[:8], 'big') % _UINT53_SCALE)
  commitment = canonical_sha256({
    'algorithm': BASE_NOISE_ALGORITHM,
    'source_descriptor': descriptor,
    'corruption_seed': corruption_seed,
    'uint53_values': values,
  })
  return values, commitment


def absorbing_mask_corruption(
    *,
    clean_tokens: Sequence[int],
    attention_mask: Sequence[bool],
    base_noise_uint53: Sequence[int],
    requested_probability: float,
    mask_token_id: int,
) -> tuple[list[int], list[int]]:
  """Apply the exact nested absorbing corruption frozen by the protocol."""
  if not (
      len(clean_tokens) == len(attention_mask) == len(base_noise_uint53)):
    raise ValueError('corruption inputs must have identical lengths')
  probability = _finite_float(
    requested_probability, context='requested_probability',
    minimum=0.0, maximum=1.0)
  if probability >= 1.0:
    raise ValueError('requested_probability must be strictly below one')
  mask_token_id = _nonnegative_int(mask_token_id, context='mask_token_id')
  threshold = math.floor(probability * _UINT53_SCALE)
  corrupted = [
    mask_token_id
    if bool(attention) and _nonnegative_int(
      noise, context='base-noise value') < threshold
    else _nonnegative_int(token, context='clean token')
    for token, attention, noise in zip(
      clean_tokens, attention_mask, base_noise_uint53)
  ]
  active_nodes = [
    index for index, (token, attention) in enumerate(
      zip(corrupted, attention_mask))
    if bool(attention) and token == mask_token_id
  ]
  return corrupted, active_nodes


def _validate_nodes(value: object, *, context: str) -> list[int]:
  if not isinstance(value, list) or not value:
    raise ValueError(f'{context} must be a non-empty list')
  nodes = [
    _nonnegative_int(node, context=f'{context}[{index}]')
    for index, node in enumerate(value)
  ]
  if nodes != sorted(set(nodes)):
    raise ValueError(f'{context} must be unique and increasing')
  return nodes


def _validate_edges(
    value: object,
    *,
    active_nodes: Sequence[int],
    component_size_cap: int,
    context: str,
) -> list[list[int]]:
  if not isinstance(value, list):
    raise TypeError(f'{context} must be a list')
  active = set(active_nodes)
  edges = []
  for index, edge in enumerate(value):
    if (not isinstance(edge, list) or len(edge) != 2
        or any(not isinstance(node, int) or isinstance(node, bool)
               for node in edge)):
      raise ValueError(f'{context}[{index}] must be an integer pair')
    left, right = edge
    if left >= right:
      raise ValueError(
        f'{context}[{index}] endpoints must be canonical with left < right')
    if left not in active or right not in active:
      raise ValueError(f'{context}[{index}] references an inactive node')
    edges.append([left, right])
  if edges != sorted(edges) or len(edges) != len({tuple(edge) for edge in edges}):
    raise ValueError(f'{context} must be unique and lexicographically sorted')
  components = _forest_components(active_nodes, edges, context=context)
  oversized = [
    len(component) for component in components
    if len(component) > component_size_cap
  ]
  if oversized:
    raise ValueError(
      f'{context} exceeds component_size_cap={component_size_cap}: '
      f'{oversized}')
  return edges


def _validate_metadata(value: object, *, context: str) -> dict[str, Any]:
  payload = dict(_strict_fields(
    value, INTERVENTION_METADATA_FIELDS, context=context))
  for field in ('reference_record_id', 'time_donor_record_id'):
    if payload[field] is not None:
      _lower_hex(payload[field], 64, context=f'{context}.{field}')
  for field in ('intervention_seed', 'time_donor_index'):
    if payload[field] is not None:
      _nonnegative_int(payload[field], context=f'{context}.{field}')
  permutation = payload['node_permutation']
  if permutation is not None:
    if not isinstance(permutation, list):
      raise TypeError(f'{context}.node_permutation must be a list or null')
    normalized = []
    for index, pair in enumerate(permutation):
      if (not isinstance(pair, list) or len(pair) != 2
          or any(not isinstance(node, int) or isinstance(node, bool)
                 for node in pair)):
        raise ValueError(
          f'{context}.node_permutation[{index}] must be an integer pair')
      normalized.append(list(pair))
    if normalized != sorted(normalized):
      raise ValueError(
        f'{context}.node_permutation must be sorted by source node')
  return payload


def validate_record(
    value: object,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    source_binding: Mapping[str, Any],
    source_binding_sha256: str,
    context: str = 'record',
) -> dict[str, Any]:
  record = dict(_strict_fields(value, RECORD_FIELDS, context=context))
  if record['schema_version'] != SCHEMA_VERSION \
      or record['artifact'] != RECORD_ARTIFACT:
    raise ValueError(f'{context} has an invalid identity')
  if record['protocol_id'] != protocol['protocol_id'] \
      or record['protocol_sha256'] != protocol_sha256:
    raise ValueError(f'{context} is bound to a different protocol')
  if record['source_binding_sha256'] != source_binding_sha256 \
      or record['job_id'] != source_binding['job_id']:
    raise ValueError(f'{context} is bound to a different source job')
  for field in ('dataset', 'source_unit_id', 'document_id'):
    _nonempty_string(record[field], context=f'{context}.{field}')
  _lower_hex(
    record['dataset_revision'], 40,
    context=f'{context}.dataset_revision')
  record['train_seed'] = _nonnegative_int(
    record['train_seed'], context=f'{context}.train_seed')
  _lower_hex(
    record['document_sha256'], 64,
    context=f'{context}.document_sha256')
  record['selection_index'] = _nonnegative_int(
    record['selection_index'], context=f'{context}.selection_index')
  record['chunk_index'] = _nonnegative_int(
    record['chunk_index'], context=f'{context}.chunk_index')
  _lower_hex(
    record['clean_example_sha256'], 64,
    context=f'{context}.clean_example_sha256')
  record['sequence_length'] = _positive_int(
    record['sequence_length'], context=f'{context}.sequence_length')
  record['corruption_seed'] = _nonnegative_int(
    record['corruption_seed'], context=f'{context}.corruption_seed')
  for field in (
      'base_noise_sha256', 'corrupted_tokens_sha256',
      'attention_mask_sha256', 'active_mask_sha256',
      'corruption_context_sha256'):
    _lower_hex(record[field], 64, context=f'{context}.{field}')
  time_index = _nonnegative_int(
    record['requested_time_index'],
    context=f'{context}.requested_time_index')
  if time_index >= len(protocol['time_points']):
    raise ValueError(f'{context}.requested_time_index is outside the grid')
  requested = _finite_float(
    record['requested_time'], context=f'{context}.requested_time',
    minimum=0.0, maximum=1.0)
  if requested != protocol['time_points'][time_index]:
    raise ValueError(f'{context}.requested_time differs from the protocol')
  record['requested_time'] = requested
  record['effective_time'] = _finite_float(
    record['effective_time'], context=f'{context}.effective_time',
    minimum=0.0, maximum=1.0)
  if record['intervention'] not in INTERVENTIONS:
    raise ValueError(f'{context}.intervention is unsupported')
  record['intervention_metadata'] = _validate_metadata(
    record['intervention_metadata'],
    context=f'{context}.intervention_metadata')
  record['active_nodes'] = _validate_nodes(
    record['active_nodes'], context=f'{context}.active_nodes')
  if record['active_nodes'][-1] >= record['sequence_length']:
    raise ValueError(f'{context}.active_nodes exceed sequence_length')
  if record['active_mask_sha256'] != active_mask_sha256_for(
      sequence_length=record['sequence_length'],
      active_nodes=record['active_nodes']):
    raise ValueError(f'{context}.active_mask_sha256 differs from active_nodes')
  if record['corruption_context_sha256'] != \
      corruption_context_sha256_for(record):
    raise ValueError(
      f'{context}.corruption_context_sha256 differs from its inputs')
  record['selected_edges'] = _validate_edges(
    record['selected_edges'], active_nodes=record['active_nodes'],
    component_size_cap=protocol['component_size_cap'],
    context=f'{context}.selected_edges')
  _lower_hex(record['record_id'], 64, context=f'{context}.record_id')
  if record['record_id'] != record_id_for(record):
    raise ValueError(f'{context}.record_id differs from its identity hash')
  return record


def _forest_components(
    nodes: Sequence[int],
    edges: Sequence[Sequence[int]],
    *,
    context: str,
) -> list[list[int]]:
  parent = {node: node for node in nodes}

  def find(node: int) -> int:
    while parent[node] != node:
      parent[node] = parent[parent[node]]
      node = parent[node]
    return node

  for left, right in edges:
    root_left = find(left)
    root_right = find(right)
    if root_left == root_right:
      raise ValueError(f'{context} contains a cycle')
    parent[root_right] = root_left
  components: dict[int, list[int]] = defaultdict(list)
  for node in nodes:
    components[find(node)].append(node)
  return sorted(
    (sorted(component) for component in components.values()),
    key=lambda component: component[0])


def _component_depths(
    components: Sequence[Sequence[int]],
    edges: Sequence[Sequence[int]],
) -> tuple[list[int], list[int]]:
  adjacency: dict[int, list[int]] = defaultdict(list)
  for left, right in edges:
    adjacency[left].append(right)
    adjacency[right].append(left)
  rooted_depths = []
  diameters = []
  for component in components:
    root = min(component)

    def distances(start: int) -> dict[int, int]:
      result = {start: 0}
      queue = deque([start])
      while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
          if neighbor not in result:
            result[neighbor] = result[node] + 1
            queue.append(neighbor)
      return result

    root_distances = distances(root)
    if len(root_distances) != len(component):
      raise RuntimeError('component traversal failed')
    rooted_depths.append(max(root_distances.values(), default=0))
    farthest = max(root_distances, key=lambda node: (root_distances[node], node))
    diameters.append(max(distances(farthest).values(), default=0))
  return rooted_depths, diameters


def topology_metrics(
    record: Mapping[str, Any],
    *,
    nonlocal_threshold: int,
) -> dict[str, Any]:
  nodes = record['active_nodes']
  edges = record['selected_edges']
  edge_set = {tuple(edge) for edge in edges}
  chain = {
    tuple(sorted((left, right)))
    for left, right in zip(nodes[:-1], nodes[1:])
  }
  overlap = len(edge_set & chain)
  distances = [right - left for left, right in edges]
  components = _forest_components(nodes, edges, context='selected_edges')
  depths, diameters = _component_depths(components, edges)
  return {
    'record_id': record['record_id'],
    'dataset': record['dataset'],
    'train_seed': record['train_seed'],
    'source_unit_id': record['source_unit_id'],
    'corruption_seed': record['corruption_seed'],
    'requested_time_index': record['requested_time_index'],
    'requested_time': record['requested_time'],
    'effective_time': record['effective_time'],
    'intervention': record['intervention'],
    'active_node_count': len(nodes),
    'edge_count': len(edges),
    'natural_chain_edge_count': len(chain),
    'natural_chain_overlap_count': overlap,
    'nonlocal_edge_count': sum(
      distance > nonlocal_threshold for distance in distances),
    'edge_distance_histogram': _histogram(distances),
    'component_sizes': [len(component) for component in components],
    'minimum_position_rooted_depths': depths,
    'component_diameters': diameters,
  }


def _histogram(values: Iterable[int]) -> dict[str, int]:
  counts = Counter(values)
  return {str(value): counts[value] for value in sorted(counts)}


def _time_shuffle_mapping(
    *,
    protocol_id: str,
    source_group_key: Sequence[Any],
    num_times: int,
    seed: int,
) -> dict[int, int]:
  """Return the frozen hash-sort-and-rotate non-identity permutation."""
  keyed = []
  group_digest = canonical_sha256(list(source_group_key))
  for index in range(num_times):
    digest = hashlib.sha256(
      f'{protocol_id}|{seed}|{group_digest}|{index}'.encode('utf-8')).digest()
    keyed.append((digest, index))
  order = [index for _, index in sorted(keyed)]
  return {
    index: order[(position + 1) % len(order)]
    for position, index in enumerate(order)
  }


def _matched_node_permutation(
    active_nodes: Sequence[int],
    *,
    seed: int,
) -> list[list[int]]:
  """Reproduce the predeclared contextual-forest topology control."""
  if len(active_nodes) < 2:
    return [[node, node] for node in active_nodes]
  fingerprint = ','.join(str(node) for node in active_nodes)
  keyed = []
  for node in active_nodes:
    digest = hashlib.sha256(
      f'contextual-forest-topology-control-v1|{seed}|'
      f'{fingerprint}|{node}'.encode('ascii')).digest()
    keyed.append((digest, node))
  targets = [node for _, node in sorted(keyed)]
  if targets == list(active_nodes):
    targets = targets[1:] + targets[:1]
  return [[source, target] for source, target in zip(active_nodes, targets)]


def _canonical_output_edges(output: Any, batch_index: int) -> list[list[int]]:
  edge_index = output.edge_index[batch_index].detach().cpu()
  edge_mask = output.edge_mask[batch_index].detach().cpu()
  edges = [
    sorted(edge_index[index].tolist())
    for index in edge_mask.nonzero(as_tuple=False).flatten().tolist()]
  return sorted(edges)


def evaluate_topology_head_interventions(
    *,
    head: Any,
    hidden_states: Any,
    unary_logits: Any,
    requested_time: Any,
    active_mask: Any,
    requested_time_indices: Sequence[int],
    source_descriptors: Sequence[Mapping[str, Any]],
    corruption_seeds: Sequence[int],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
  """Run every topology intervention while reusing one backbone encoding.

  This is the model-facing emitter primitive. It intentionally accepts
  already-computed hidden states and unary logits, so fixed/zero/shuffled time
  cannot accidentally rerun or perturb the language-model backbone.
  """
  import torch  # noqa: PLC0415

  protocol = validate_protocol(protocol)
  if getattr(head, 'training', True):
    raise ValueError('topology intervention emitter requires head.eval()')
  if (hidden_states.shape[:2] != unary_logits.shape[:2]
      or active_mask.shape != hidden_states.shape[:2]
      or active_mask.dtype != torch.bool):
    raise ValueError('topology emitter tensors disagree on batch/sequence')
  batch_size = hidden_states.shape[0]
  if not (
      len(requested_time_indices) == len(source_descriptors)
      == len(corruption_seeds) == batch_size):
    raise ValueError('topology emitter metadata does not match batch size')
  if requested_time.numel() != batch_size:
    raise ValueError('requested_time must contain one value per example')
  requested = requested_time.reshape(batch_size)
  expected_requested = requested.new_tensor([
    protocol['time_points'][index] for index in requested_time_indices])
  if not torch.equal(requested, expected_requested):
    raise ValueError('requested_time differs from frozen probability grid')

  def head_time(probabilities: Any) -> Any:
    return -torch.log1p(-probabilities)

  def run(effective_time: Any) -> Any:
    with torch.no_grad():
      return head(
        hidden_states=hidden_states,
        unary_logits=unary_logits,
        timestep=effective_time,
        active_mask=active_mask,
        topology_mode='dynamic')

  learned_output = run(head_time(requested))
  fixed_value = protocol['interventions']['fixed_time']['effective_time']
  fixed_time = head_time(requested.new_full((batch_size,), fixed_value))
  zero_time = head_time(requested.new_zeros(batch_size))
  donor_indices = []
  shuffled_values = []
  for descriptor, corruption_seed, requested_index in zip(
      source_descriptors, corruption_seeds, requested_time_indices):
    mapping = _time_shuffle_mapping(
      protocol_id=protocol['protocol_id'],
      source_group_key=(dict(descriptor), corruption_seed),
      num_times=len(protocol['time_points']),
      seed=protocol['interventions']['timestep_shuffled']['shuffle_seed'])
    donor_index = mapping[requested_index]
    donor_indices.append(donor_index)
    shuffled_values.append(protocol['time_points'][donor_index])
  shuffled_time = head_time(requested.new_tensor(shuffled_values))
  fixed_output = run(fixed_time)
  zero_output = run(zero_time)
  shuffled_output = run(shuffled_time)

  emitted = []
  for batch_index in range(batch_size):
    # Persist the canonical JSON-grid value, not a float32 tensor round-trip.
    # Values such as 0.1 and 0.9 are not exactly representable in float32 and
    # must still satisfy the protocol's exact requested/effective-time pairing.
    requested_value = protocol['time_points'][
      requested_time_indices[batch_index]]
    active_nodes = active_mask[batch_index].nonzero(
      as_tuple=False).flatten().detach().cpu().tolist()
    learned_edges = _canonical_output_edges(learned_output, batch_index)
    permutation = _matched_node_permutation(
      active_nodes,
      seed=protocol['interventions']['matched_permuted']['permutation_seed'])
    node_mapping = dict(permutation)
    permuted_edges = sorted([
      sorted((node_mapping[left], node_mapping[right]))
      for left, right in learned_edges])
    fixed_edges = _canonical_output_edges(fixed_output, batch_index)
    if (protocol['time_points'][requested_time_indices[batch_index]]
        == fixed_value and fixed_edges != learned_edges):
      raise ValueError(
        'deterministic fixed-time anchor did not reproduce learned topology')
    emitted.append({
      'learned': {
        'effective_time': requested_value,
        'selected_edges': learned_edges,
        'node_permutation': None,
        'time_donor_index': None,
      },
      'matched_permuted': {
        'effective_time': requested_value,
        'selected_edges': permuted_edges,
        'node_permutation': permutation,
        'time_donor_index': None,
      },
      'fixed_time': {
        'effective_time': fixed_value,
        'selected_edges': fixed_edges,
        'node_permutation': None,
        'time_donor_index': None,
      },
      'zero_time': {
        'effective_time': 0.0,
        'selected_edges': _canonical_output_edges(
          zero_output, batch_index),
        'node_permutation': None,
        'time_donor_index': None,
      },
      'timestep_shuffled': {
        'effective_time': shuffled_values[batch_index],
        'selected_edges': _canonical_output_edges(
          shuffled_output, batch_index),
        'node_permutation': None,
        'time_donor_index': donor_indices[batch_index],
      },
    })
  return emitted


SOURCE_EMITTER_FIELDS = SOURCE_DESCRIPTOR_FIELDS | {
  'input_ids', 'attention_mask',
}


def _one_dimensional_list(value: Any, *, context: str) -> list[Any]:
  if hasattr(value, 'detach'):
    value = value.detach().cpu().tolist()
  elif hasattr(value, 'tolist'):
    value = value.tolist()
  if not isinstance(value, list):
    raise TypeError(f'{context} must be a one-dimensional sequence')
  if any(isinstance(item, (list, tuple, Mapping)) for item in value):
    raise ValueError(f'{context} must be one-dimensional')
  return value


def source_units_from_ordered_dataset(
    dataset_object: Any,
    *,
    protocol: Mapping[str, Any],
    dataset: str,
) -> list[dict[str, Any]]:
  """Select exactly the first N pinned document-local evaluation windows."""
  protocol = validate_protocol(protocol)
  if dataset not in protocol['source_selection']['datasets']:
    raise ValueError('dataset is outside the topology source selection')
  specification = protocol['source_selection']['datasets'][dataset]
  count = specification['num_source_units']
  try:
    available = len(dataset_object)
  except TypeError as error:
    raise ValueError(
      'ordered topology selection requires a finite map-style dataset') \
      from error
  if available < count:
    raise ValueError(
      f'pinned dataset contains {available} windows; expected at least {count}')
  result = []
  for selection_index in range(count):
    row = dataset_object[selection_index]
    if not isinstance(row, Mapping):
      raise TypeError('pinned dataset row must be a mapping')
    required = {
      'input_ids', 'attention_mask', 'source_document_index',
      'source_document_sha256', 'source_chunk_index'}
    missing = required - set(row)
    if missing:
      raise ValueError(
        f'pinned document-local row lacks metadata: {sorted(missing)}')
    input_ids = _one_dimensional_list(
      row['input_ids'], context='source input_ids')
    attention_mask = _one_dimensional_list(
      row['attention_mask'], context='source attention_mask')
    if len(input_ids) != len(attention_mask) or not input_ids:
      raise ValueError('pinned source input and attention lengths differ')
    document_index = int(row['source_document_index'])
    chunk_index = int(row['source_chunk_index'])
    if document_index < 0 or chunk_index < 0:
      raise ValueError('pinned source indices must be non-negative')
    document_sha = str(row['source_document_sha256'])
    _lower_hex(document_sha, 64, context='source document SHA256')
    document_id = f'{dataset}:{document_index}'
    source_unit_id = f'{document_id}:chunk-{chunk_index}'
    normalized_tokens = [int(token) for token in input_ids]
    normalized_attention = [bool(value) for value in attention_mask]
    result.append({
      'dataset': dataset,
      'dataset_revision': specification['dataset_revision'],
      'selection_index': selection_index,
      'source_unit_id': source_unit_id,
      'document_id': document_id,
      'document_sha256': document_sha,
      'chunk_index': chunk_index,
      'clean_example_sha256': clean_example_sha256_for(
        normalized_tokens, normalized_attention),
      'sequence_length': len(normalized_tokens),
      'input_ids': normalized_tokens,
      'attention_mask': normalized_attention,
    })
  return result


def _validate_emitter_source_unit(
    value: object,
    *,
    protocol: Mapping[str, Any],
    dataset: str,
    expected_index: int,
) -> dict[str, Any]:
  source = dict(_strict_fields(
    value, SOURCE_EMITTER_FIELDS,
    context=f'topology source unit {expected_index}'))
  specification = protocol['source_selection']['datasets'][dataset]
  if (source['dataset'] != dataset
      or source['dataset_revision'] != specification['dataset_revision']
      or source['selection_index'] != expected_index):
    raise ValueError('topology source unit differs from ordered selection')
  for field in ('source_unit_id', 'document_id'):
    _nonempty_string(source[field], context=f'source unit.{field}')
  for field in ('document_sha256', 'clean_example_sha256'):
    _lower_hex(source[field], 64, context=f'source unit.{field}')
  source['chunk_index'] = _nonnegative_int(
    source['chunk_index'], context='source unit.chunk_index')
  source['sequence_length'] = _positive_int(
    source['sequence_length'], context='source unit.sequence_length')
  input_ids = source['input_ids']
  attention_mask = source['attention_mask']
  if (not isinstance(input_ids, list) or not isinstance(attention_mask, list)
      or len(input_ids) != source['sequence_length']
      or len(attention_mask) != source['sequence_length']):
    raise ValueError('source unit tensors differ from sequence_length')
  source['input_ids'] = [
    _nonnegative_int(token, context='source unit token')
    for token in input_ids]
  if any(not isinstance(value, (bool, int)) for value in attention_mask):
    raise ValueError('source unit attention mask must be boolean')
  source['attention_mask'] = [bool(value) for value in attention_mask]
  if not any(source['attention_mask']):
    raise ValueError('source unit attention mask cannot be empty')
  if source['clean_example_sha256'] != clean_example_sha256_for(
      source['input_ids'], source['attention_mask']):
    raise ValueError('source unit clean-example commitment differs')
  return source


def emit_topology_records(
    *,
    model: Any,
    source_units: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    device: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
  """Execute the complete model-facing grid for one dataset/adapter job.

  The backbone runs exactly once per requested corruption batch. All four
  model-evaluated interventions reuse its hidden states and unary logits;
  matched permutation is derived from the learned edge list without a model
  call. Returned records are ready for ``write_record_bundle``.
  """
  import torch  # noqa: PLC0415

  protocol = validate_protocol(protocol)
  binding = validate_source_binding(source_binding)
  binding_sha = canonical_sha256(binding)
  dataset = binding['dataset']
  if dataset not in protocol['source_selection']['datasets']:
    raise ValueError('emitter dataset is outside the frozen protocol')
  expected_sources = protocol['source_selection']['datasets'][dataset][
    'num_source_units']
  if len(source_units) != expected_sources:
    raise ValueError(
      f'emitter received {len(source_units)} sources; expected '
      f'{expected_sources}')
  batch_size = _positive_int(batch_size, context='emitter batch_size')
  sources = [
    _validate_emitter_source_unit(
      source, protocol=protocol, dataset=dataset, expected_index=index)
    for index, source in enumerate(source_units)
  ]
  if getattr(model, 'training', True):
    raise ValueError('topology emitter requires model.eval()')
  head = getattr(model, 'structured_head', None)
  if head is None or getattr(head, 'training', True):
    raise ValueError('topology emitter requires a structured head in eval mode')
  if getattr(head, 'top_k', None) != protocol['candidate_top_k']:
    raise ValueError('runtime topology head candidate K differs from protocol')
  if getattr(head, 'component_size_cap', None) != \
      protocol['component_size_cap']:
    raise ValueError('runtime topology component cap differs from protocol')
  mask_token_id = _nonnegative_int(
    getattr(model, 'mask_index', None), context='model.mask_index')

  descriptors = [{
    field: source[field] for field in SOURCE_DESCRIPTOR_FIELDS
  } for source in sources]
  attention_hashes = [
    sequence_sha256(source['attention_mask'], dtype='bool')
    for source in sources]
  base_noise: dict[tuple[int, int], tuple[list[int], str]] = {}
  for source_index, descriptor in enumerate(descriptors):
    for corruption_seed in protocol['corruption_seeds']:
      base_noise[(source_index, corruption_seed)] = \
        deterministic_base_noise_uint53(
          source_descriptor=descriptor,
          corruption_seed=corruption_seed,
          sequence_length=descriptor['sequence_length'])

  cells: dict[tuple[int, int, int], dict[str, Any]] = {}
  for corruption_seed in protocol['corruption_seeds']:
    for time_index, requested_probability in enumerate(
        protocol['time_points']):
      for start in range(0, len(sources), batch_size):
        stop = min(start + batch_size, len(sources))
        corrupted_batch = []
        active_batch = []
        for source_index in range(start, stop):
          source = sources[source_index]
          noise_values, noise_sha = base_noise[
            (source_index, corruption_seed)]
          corrupted, active_nodes = absorbing_mask_corruption(
            clean_tokens=source['input_ids'],
            attention_mask=source['attention_mask'],
            base_noise_uint53=noise_values,
            requested_probability=requested_probability,
            mask_token_id=mask_token_id)
          if not active_nodes:
            raise ValueError(
              'frozen corruption produced no active node; protocol cannot '
              'emit a nonempty learned forest')
          corrupted_batch.append(corrupted)
          active_batch.append(active_nodes)
          cells[(source_index, corruption_seed, time_index)] = {
            'base_noise_sha256': noise_sha,
            'corrupted_tokens_sha256': sequence_sha256(
              corrupted, dtype='int64'),
            'active_nodes': active_nodes,
          }
        sequence_lengths = {
          len(tokens) for tokens in corrupted_batch}
        if len(sequence_lengths) != 1:
          raise ValueError('one emitter batch cannot mix sequence lengths')
        tokens = torch.as_tensor(
          corrupted_batch, dtype=torch.long, device=device)
        attention = torch.as_tensor([
          sources[index]['attention_mask'] for index in range(start, stop)
        ], dtype=torch.bool, device=device)
        active_mask = tokens.eq(mask_token_id) & attention
        requested = torch.full(
          (stop - start,), requested_probability,
          dtype=torch.float32, device=device)
        head_conditioning = -torch.log1p(-requested)
        with torch.no_grad():
          hidden_states, unary_logits = model._structured_backbone_output(
            tokens=tokens,
            conditioning=head_conditioning[:, None],
            force_no_grad=True)
        interventions = evaluate_topology_head_interventions(
          head=head,
          hidden_states=hidden_states,
          unary_logits=unary_logits,
          requested_time=requested,
          active_mask=active_mask,
          requested_time_indices=[time_index] * (stop - start),
          source_descriptors=descriptors[start:stop],
          corruption_seeds=[corruption_seed] * (stop - start),
          protocol=protocol)
        for local_index, intervention_results in enumerate(interventions):
          source_index = start + local_index
          cell = cells[(source_index, corruption_seed, time_index)]
          observed_active = active_mask[local_index].nonzero(
            as_tuple=False).flatten().detach().cpu().tolist()
          if observed_active != cell['active_nodes']:
            raise RuntimeError('device corruption active mask drifted')
          cell['interventions'] = intervention_results

  protocol_sha = canonical_sha256(protocol)
  learned_records: dict[tuple[int, int, int], dict[str, Any]] = {}

  def record_base(source_index: int, corruption_seed: int,
                  time_index: int) -> dict[str, Any]:
    source = sources[source_index]
    cell = cells[(source_index, corruption_seed, time_index)]
    active_nodes = cell['active_nodes']
    result = {
      'schema_version': SCHEMA_VERSION,
      'artifact': RECORD_ARTIFACT,
      'record_id': '0' * 64,
      'protocol_id': protocol['protocol_id'],
      'protocol_sha256': protocol_sha,
      'source_binding_sha256': binding_sha,
      'job_id': binding['job_id'],
      'dataset': source['dataset'],
      'dataset_revision': source['dataset_revision'],
      'train_seed': binding['train_seed'],
      'source_unit_id': source['source_unit_id'],
      'document_id': source['document_id'],
      'document_sha256': source['document_sha256'],
      'selection_index': source['selection_index'],
      'chunk_index': source['chunk_index'],
      'clean_example_sha256': source['clean_example_sha256'],
      'sequence_length': source['sequence_length'],
      'corruption_seed': corruption_seed,
      'base_noise_sha256': cell['base_noise_sha256'],
      'corrupted_tokens_sha256': cell['corrupted_tokens_sha256'],
      'attention_mask_sha256': attention_hashes[source_index],
      'active_mask_sha256': active_mask_sha256_for(
        sequence_length=source['sequence_length'],
        active_nodes=active_nodes),
      'corruption_context_sha256': '0' * 64,
      'requested_time_index': time_index,
      'requested_time': protocol['time_points'][time_index],
      'effective_time': protocol['time_points'][time_index],
      'intervention': 'learned',
      'intervention_metadata': {
        'reference_record_id': None,
        'intervention_seed': None,
        'time_donor_record_id': None,
        'time_donor_index': None,
        'node_permutation': None,
      },
      'active_nodes': list(active_nodes),
      'selected_edges': [],
    }
    result['corruption_context_sha256'] = corruption_context_sha256_for(
      result)
    return result

  for key, cell in cells.items():
    learned = record_base(*key)
    learned['selected_edges'] = cell['interventions']['learned'][
      'selected_edges']
    learned['record_id'] = record_id_for(learned)
    learned_records[key] = learned

  records = []
  for key in sorted(cells):
    source_index, corruption_seed, time_index = key
    learned = learned_records[key]
    records.append(learned)
    for intervention in INTERVENTIONS[1:]:
      result = cells[key]['interventions'][intervention]
      record = record_base(*key)
      record['intervention'] = intervention
      record['effective_time'] = result['effective_time']
      metadata = record['intervention_metadata']
      metadata['reference_record_id'] = learned['record_id']
      if intervention == 'matched_permuted':
        metadata['intervention_seed'] = protocol['interventions'][
          intervention]['permutation_seed']
        metadata['node_permutation'] = result['node_permutation']
      elif intervention == 'timestep_shuffled':
        donor_index = result['time_donor_index']
        metadata['intervention_seed'] = protocol['interventions'][
          intervention]['shuffle_seed']
        metadata['time_donor_index'] = donor_index
        metadata['time_donor_record_id'] = learned_records[
          (source_index, corruption_seed, donor_index)]['record_id']
      record['selected_edges'] = result['selected_edges']
      record['record_id'] = record_id_for(record)
      records.append(record)
  _validate_bundle_source_selection(
    records, source_binding=binding, protocol=protocol)
  validate_complete_grid(records, protocol=protocol)
  return records


def _source_unit_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
  return (
    record['source_binding_sha256'], record['dataset'],
    record['dataset_revision'], record['train_seed'],
    record['selection_index'],
    record['source_unit_id'], record['document_id'],
    record['document_sha256'], record['chunk_index'],
    record['clean_example_sha256'], record['sequence_length'],
  )


def _source_descriptor(record: Mapping[str, Any]) -> dict[str, Any]:
  return {
    'dataset': record['dataset'],
    'dataset_revision': record['dataset_revision'],
    'selection_index': record['selection_index'],
    'source_unit_id': record['source_unit_id'],
    'document_id': record['document_id'],
    'document_sha256': record['document_sha256'],
    'chunk_index': record['chunk_index'],
    'clean_example_sha256': record['clean_example_sha256'],
    'sequence_length': record['sequence_length'],
  }


def source_selection_sha256(records: Sequence[Mapping[str, Any]]) -> str:
  descriptors: dict[int, dict[str, Any]] = {}
  for record in records:
    descriptor = _source_descriptor(record)
    index = descriptor['selection_index']
    previous = descriptors.get(index)
    if previous is not None and previous != descriptor:
      raise ValueError(
        f'selection index {index} maps to multiple source descriptors')
    descriptors[index] = descriptor
  if sorted(descriptors) != list(range(len(descriptors))):
    raise ValueError('source selection indices must be contiguous from zero')
  return canonical_sha256([
    descriptors[index] for index in range(len(descriptors))
  ])


def validate_source_selection_manifest(
    value: object,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    dataset: str,
) -> dict[str, Any]:
  payload = dict(_strict_fields(
    value, SOURCE_SELECTION_FIELDS, context='source-selection manifest'))
  committed = payload.pop('manifest_sha256')
  _lower_hex(committed, 64, context='source-selection manifest_sha256')
  if committed != canonical_sha256(payload):
    raise ValueError('source-selection manifest self-hash mismatch')
  payload['manifest_sha256'] = committed
  if (payload['schema_version'] != SCHEMA_VERSION
      or payload['artifact'] != SOURCE_SELECTION_ARTIFACT
      or payload['protocol_id'] != protocol['protocol_id']
      or payload['protocol_sha256'] != protocol_sha256
      or payload['dataset'] != dataset):
    raise ValueError('invalid source-selection manifest identity')
  specification = protocol['source_selection']['datasets'][dataset]
  if (payload['dataset_revision'] != specification['dataset_revision']
      or payload['tokenizer_revision']
      != specification['tokenizer_revision']
      or payload['selection_policy']
      != protocol['source_selection']['source_unit_order']):
    raise ValueError('source-selection manifest differs from the protocol')
  entries = payload['entries']
  if not isinstance(entries, list):
    raise TypeError('source-selection entries must be a list')
  expected_count = specification['num_source_units']
  if len(entries) != expected_count:
    raise ValueError(
      f'source-selection manifest has {len(entries)} entries; expected '
      f'{expected_count}')
  normalized = []
  for index, entry in enumerate(entries):
    descriptor = dict(_strict_fields(
      entry, SOURCE_DESCRIPTOR_FIELDS,
      context=f'source-selection entries[{index}]'))
    if descriptor['selection_index'] != index:
      raise ValueError(
        'source-selection entries must remain in first-N evaluation order')
    if (descriptor['dataset'] != dataset
        or descriptor['dataset_revision']
        != specification['dataset_revision']):
      raise ValueError('source-selection entry dataset identity differs')
    for field in ('source_unit_id', 'document_id'):
      _nonempty_string(
        descriptor[field], context=f'source-selection entry.{field}')
    for field in ('document_sha256', 'clean_example_sha256'):
      _lower_hex(
        descriptor[field], 64,
        context=f'source-selection entry.{field}')
    descriptor['chunk_index'] = _nonnegative_int(
      descriptor['chunk_index'], context='source-selection entry.chunk_index')
    descriptor['sequence_length'] = _positive_int(
      descriptor['sequence_length'],
      context='source-selection entry.sequence_length')
    normalized.append(descriptor)
  if payload['selection_sha256'] != canonical_sha256(normalized):
    raise ValueError('source-selection digest differs from ordered entries')
  payload['entries'] = normalized
  return payload


def write_source_selection_manifest(
    *,
    path: Path,
    protocol: Mapping[str, Any],
    dataset: str,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
  """Write the pre-result ordered first-N source commitment once."""
  protocol = validate_protocol(protocol)
  protocol_sha = canonical_sha256(protocol)
  specification = protocol['source_selection']['datasets'][dataset]
  normalized_entries = [dict(entry) for entry in entries]
  body = {
    'schema_version': SCHEMA_VERSION,
    'artifact': SOURCE_SELECTION_ARTIFACT,
    'protocol_id': protocol['protocol_id'],
    'protocol_sha256': protocol_sha,
    'dataset': dataset,
    'dataset_revision': specification['dataset_revision'],
    'tokenizer_revision': specification['tokenizer_revision'],
    'selection_policy': protocol['source_selection']['source_unit_order'],
    'entries': normalized_entries,
    'selection_sha256': canonical_sha256(normalized_entries),
  }
  manifest = {**body, 'manifest_sha256': canonical_sha256(body)}
  validate_source_selection_manifest(
    manifest, protocol=protocol, protocol_sha256=protocol_sha,
    dataset=dataset)
  path = path.expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('x', encoding='utf-8') as handle:
    handle.write(json.dumps(
      manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
  return path


def _validate_bundle_source_selection(
    records: Sequence[Mapping[str, Any]],
    *,
    source_binding: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
  if not records:
    raise ValueError('topology source bundle cannot be empty')
  selection = protocol['source_selection']
  if source_binding['arm'] != selection['arm']:
    raise ValueError('source bundle arm differs from the frozen selection')
  dataset = source_binding['dataset']
  train_seed = source_binding['train_seed']
  if dataset not in selection['datasets']:
    raise ValueError('source bundle dataset is outside the frozen selection')
  if train_seed not in selection['train_seeds']:
    raise ValueError('source bundle train seed is outside the frozen selection')
  dataset_specification = selection['datasets'][dataset]
  for record in records:
    if record['dataset'] != dataset or record['train_seed'] != train_seed:
      raise ValueError(
        'source bundle mixes datasets or training seeds')
    if record['dataset_revision'] != dataset_specification['dataset_revision']:
      raise ValueError('source record dataset revision differs from protocol')
  descriptors = {
    canonical_json(_source_descriptor(record)) for record in records
  }
  expected_count = dataset_specification['num_source_units']
  if len(descriptors) != expected_count:
    raise ValueError(
      f'source bundle has {len(descriptors)} source units; frozen '
      f'selection requires {expected_count}')
  observed_indices = sorted({
    record['selection_index'] for record in records})
  if observed_indices != list(range(expected_count)):
    raise ValueError(
      'source bundle does not preserve exact first-N selection indices')
  observed_selection_sha = source_selection_sha256(records)
  if source_binding['source_selection_sha256'] != observed_selection_sha:
    raise ValueError('source-selection SHA256 differs from raw records')


def _validate_source_bundle_grid(
    bundles: Sequence[tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]],
    *,
    protocol: Mapping[str, Any],
) -> None:
  selection = protocol['source_selection']
  expected = {
    (dataset, train_seed)
    for dataset in selection['datasets']
    for train_seed in selection['train_seeds']
  }
  observed = {}
  for records, manifest in bundles:
    binding = manifest['source_binding']
    _validate_bundle_source_selection(
      records, source_binding=binding, protocol=protocol)
    key = (binding['dataset'], binding['train_seed'])
    if key in observed:
      raise ValueError(f'duplicate source bundle for {key}')
    observed[key] = (records, binding)
  if set(observed) != expected:
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    raise ValueError(
      f'topology source-bundle grid is incomplete: '
      f'missing={missing}, extra={extra}')

  plan_identities = {
    (
      binding['compiled_plan_sha256'], binding['plan_id'],
      binding['repository_sha'])
    for _, binding in observed.values()
  }
  if len(plan_identities) != 1:
    raise ValueError(
      'topology source bundles do not share one plan and repository commit')
  evaluator_hashes = {
    binding['evaluator_source_sha256'] for _, binding in observed.values()}
  if len(evaluator_hashes) != 1:
    raise ValueError('topology evaluator source differs across bundles')
  for train_seed in selection['train_seeds']:
    checkpoint_hashes = {
      observed[(dataset, train_seed)][1]['adapter_sha256']
      for dataset in selection['datasets']
    }
    if len(checkpoint_hashes) != 1:
      raise ValueError(
        f'datasets disagree on the adapter checkpoint for seed {train_seed}')
    adapter_manifest_hashes = {
      observed[(dataset, train_seed)][1][
        'adapter_export_manifest_sha256']
      for dataset in selection['datasets']
    }
    if len(adapter_manifest_hashes) != 1:
      raise ValueError(
        f'datasets disagree on the adapter export for seed {train_seed}')
  for dataset in selection['datasets']:
    provenance_hashes = {
      observed[(dataset, train_seed)][1]['dataset_provenance_sha256']
      for train_seed in selection['train_seeds']
    }
    data_config_hashes = {
      observed[(dataset, train_seed)][1]['data_config_sha256']
      for train_seed in selection['train_seeds']
    }
    if len(provenance_hashes) != 1 or len(data_config_hashes) != 1:
      raise ValueError(
        f'{dataset} data provenance differs across training seeds')
  if selection['require_identical_source_units_across_train_seeds']:
    for dataset in selection['datasets']:
      descriptor_lists = []
      for train_seed in selection['train_seeds']:
        records = observed[(dataset, train_seed)][0]
        descriptors = {
          record['selection_index']: _source_descriptor(record)
          for record in records}
        descriptor_lists.append([
          descriptors[index] for index in sorted(descriptors)])
      if any(current != descriptor_lists[0]
             for current in descriptor_lists[1:]):
        raise ValueError(
          f'{dataset} source units differ across training seeds')

  # The exact same corruption is replayed for every trained adapter. This is
  # stricter than merely reusing an integer seed and prevents loader/RNG drift.
  corruption_by_coordinate: dict[tuple[Any, ...], tuple[Any, ...]] = {}
  base_noise_by_coordinate: dict[tuple[Any, ...], str] = {}
  base_noise_by_source: dict[tuple[Any, ...], dict[int, str]] = defaultdict(dict)
  attention_by_source: dict[tuple[Any, ...], str] = {}
  learned_trajectories: dict[tuple[Any, ...], dict[int, set[int]]] = \
    defaultdict(dict)
  for records, _ in observed.values():
    for record in records:
      source = _source_descriptor(record)
      cross_seed_source = tuple(
        source[field] for field in sorted(SOURCE_DESCRIPTOR_FIELDS))
      previous_attention = attention_by_source.setdefault(
        cross_seed_source, record['attention_mask_sha256'])
      if previous_attention != record['attention_mask_sha256']:
        raise ValueError('attention-mask commitment differs for one source')
      base_key = (*cross_seed_source, record['corruption_seed'])
      previous_noise = base_noise_by_coordinate.setdefault(
        base_key, record['base_noise_sha256'])
      if previous_noise != record['base_noise_sha256']:
        raise ValueError(
          'base-noise commitment differs across times or training seeds')
      seed_noises = base_noise_by_source[cross_seed_source]
      previous_seed_noise = seed_noises.setdefault(
        record['corruption_seed'], record['base_noise_sha256'])
      if previous_seed_noise != record['base_noise_sha256']:
        raise ValueError('base-noise commitment differs within corruption seed')
      coordinate = (*base_key, record['requested_time_index'])
      signature = (
        record['corrupted_tokens_sha256'],
        record['attention_mask_sha256'], record['active_mask_sha256'],
        tuple(record['active_nodes']), record['corruption_context_sha256'])
      previous = corruption_by_coordinate.setdefault(coordinate, signature)
      if previous != signature:
        raise ValueError(
          'paired corruption context differs across training seeds')
      if record['intervention'] == 'learned':
        learned_trajectories[
          (*_source_unit_key(record), record['corruption_seed'])][
            record['requested_time_index']] = set(record['active_nodes'])
  for seed_noises in base_noise_by_source.values():
    if set(seed_noises) != set(protocol['corruption_seeds']):
      raise ValueError('base-noise grid is incomplete')
    if len(set(seed_noises.values())) != len(seed_noises):
      raise ValueError(
        'independent corruption seeds reused one base-noise commitment')
  for trajectory in learned_trajectories.values():
    previous_nodes: set[int] = set()
    for time_index in range(len(protocol['time_points'])):
      current_nodes = trajectory[time_index]
      if not previous_nodes.issubset(current_nodes):
        raise ValueError(
          'active masks are not nested under shared absorbing-noise draw')
      previous_nodes = current_nodes


def _record_grid_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
  return (
    *_source_unit_key(record), record['corruption_seed'],
    record['requested_time_index'], record['intervention'],
  )


def _edge_set(record: Mapping[str, Any]) -> set[tuple[int, int]]:
  return {tuple(edge) for edge in record['selected_edges']}


def _validate_intervention_pair(
    *,
    record: Mapping[str, Any],
    learned: Mapping[str, Any],
    records_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
    protocol: Mapping[str, Any],
    source_key: tuple[Any, ...],
) -> tuple[int, int]:
  intervention = record['intervention']
  metadata = record['intervention_metadata']
  if record['clean_example_sha256'] != learned['clean_example_sha256']:
    raise ValueError('intervention clean-example commitment differs')
  if (record['active_nodes'] != learned['active_nodes']
      or record['corruption_context_sha256']
      != learned['corruption_context_sha256']):
    raise ValueError(
      f'{intervention} must use the paired learned corruption context')
  expected_reference = None if intervention == 'learned' \
    else learned['record_id']
  if metadata['reference_record_id'] != expected_reference:
    raise ValueError(f'{intervention} has an invalid learned reference')

  null_fields = ('time_donor_record_id', 'time_donor_index')
  if intervention == 'learned':
    if (record['effective_time'] != record['requested_time']
        or metadata['intervention_seed'] is not None
        or metadata['node_permutation'] is not None
        or any(metadata[field] is not None for field in null_fields)):
      raise ValueError('learned intervention metadata is not neutral')
    return 0, 0
  if intervention == 'matched_permuted':
    expected_seed = protocol['interventions'][intervention][
      'permutation_seed']
    if (record['effective_time'] != record['requested_time']
        or metadata['intervention_seed'] != expected_seed
        or any(metadata[field] is not None for field in null_fields)):
      raise ValueError('matched_permuted metadata differs from the protocol')
    permutation = metadata['node_permutation']
    if permutation is None:
      raise ValueError('matched_permuted requires a node permutation')
    nodes = learned['active_nodes']
    expected_permutation = _matched_node_permutation(
      nodes, seed=expected_seed)
    if permutation != expected_permutation:
      raise ValueError(
        'matched_permuted node mapping differs from the frozen algorithm')
    sources = [pair[0] for pair in permutation]
    targets = [pair[1] for pair in permutation]
    if sources != nodes or sorted(targets) != nodes:
      raise ValueError(
        'matched_permuted node mapping must be a bijection on active nodes')
    mapping = dict(permutation)
    expected_edges = sorted([
      sorted((mapping[left], mapping[right]))
      for left, right in learned['selected_edges']
    ])
    if expected_edges != record['selected_edges']:
      raise ValueError(
        'matched_permuted edges do not equal the committed node mapping')
    learned_edges = _edge_set(learned)
    changed = len(learned_edges - _edge_set(record))
    return changed, len(learned_edges)
  if metadata['node_permutation'] is not None:
    raise ValueError(f'{intervention} cannot declare a node permutation')
  if intervention == 'fixed_time':
    expected = protocol['interventions'][intervention]['effective_time']
    if (record['effective_time'] != expected
        or metadata['intervention_seed'] is not None
        or any(metadata[field] is not None for field in null_fields)):
      raise ValueError('fixed_time metadata differs from the protocol')
    if (record['requested_time'] == expected
        and record['selected_edges'] != learned['selected_edges']):
      raise ValueError(
        'fixed_time must exactly reproduce learned edges at its anchor time')
    return 0, 0
  if intervention == 'zero_time':
    if (record['effective_time'] != 0.0
        or metadata['intervention_seed'] is not None
        or any(metadata[field] is not None for field in null_fields)):
      raise ValueError('zero_time metadata differs from the protocol')
    return 0, 0
  if intervention != 'timestep_shuffled':
    raise RuntimeError(f'unhandled intervention {intervention}')
  expected_seed = protocol['interventions'][intervention]['shuffle_seed']
  if metadata['intervention_seed'] != expected_seed:
    raise ValueError('timestep_shuffled seed differs from the protocol')
  # Do not include job, checkpoint, training seed, or post-run commitments:
  # this intervention must be known before model execution and paired across
  # independently trained adapters.
  shuffle_group = (
    _source_descriptor(record), record['corruption_seed'])
  mapping = _time_shuffle_mapping(
    protocol_id=protocol['protocol_id'],
    source_group_key=shuffle_group,
    num_times=len(protocol['time_points']),
    seed=expected_seed)
  donor_index = mapping[record['requested_time_index']]
  donor_key = (
    *source_key, record['corruption_seed'], donor_index, 'learned')
  donor = records_by_key[donor_key]
  if (metadata['time_donor_index'] != donor_index
      or metadata['time_donor_record_id'] != donor['record_id']
      or record['effective_time'] != donor['requested_time']):
    raise ValueError(
      'timestep_shuffled donor differs from the deterministic permutation')
  return 0, 0


def validate_complete_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
  if not records:
    raise ValueError('topology analysis requires at least one record')
  records_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
  records_by_id = {}
  source_keys = set()
  suffixes_by_source = defaultdict(set)
  for record in records:
    key = _record_grid_key(record)
    if key in records_by_key:
      raise ValueError(f'duplicate topology grid cell: {key}')
    if record['record_id'] in records_by_id:
      raise ValueError(f'duplicate topology record ID: {record["record_id"]}')
    records_by_key[key] = record
    records_by_id[record['record_id']] = record
    source_key = _source_unit_key(record)
    source_keys.add(source_key)
    suffixes_by_source[source_key].add(key[-3:])
  expected_suffixes = {
    (seed, time_index, intervention)
    for seed in protocol['corruption_seeds']
    for time_index in range(len(protocol['time_points']))
    for intervention in INTERVENTIONS
  }
  changed_edges = 0
  total_edges = 0
  for source_key in sorted(source_keys):
    observed_suffixes = suffixes_by_source[source_key]
    if observed_suffixes != expected_suffixes:
      missing = sorted(expected_suffixes - observed_suffixes)
      extra = sorted(observed_suffixes - expected_suffixes)
      raise ValueError(
        f'incomplete topology grid for source unit {source_key}: '
        f'missing={missing}, extra={extra}')
    for corruption_seed, time_index, intervention in sorted(expected_suffixes):
      record = records_by_key[
        (*source_key, corruption_seed, time_index, intervention)]
      learned = records_by_key[
        (*source_key, corruption_seed, time_index, 'learned')]
      if (protocol['require_nonempty_learned_forest']
          and not learned['selected_edges']):
        raise ValueError('learned topology must select at least one edge')
      changed, denominator = _validate_intervention_pair(
        record=record,
        learned=learned,
        records_by_key=records_by_key,
        protocol=protocol,
        source_key=source_key)
      changed_edges += changed
      total_edges += denominator
  if total_edges <= 0:
    raise ValueError('matched permutation gate has no learned edges')
  changed_fraction = changed_edges / total_edges
  minimum = protocol['interventions']['matched_permuted'][
    'minimum_pooled_edge_set_changed_fraction']
  if changed_fraction < minimum:
    raise ValueError(
      f'matched permutation changed {changed_fraction:.6f} of learned '
      f'edges; protocol requires at least {minimum:.6f}')
  return {
    'num_source_units': len(source_keys),
    'num_records': len(records),
    'expected_records_per_source_unit': len(expected_suffixes),
    'matched_permuted_edge_set_changed': changed_edges,
    'matched_permuted_learned_edges': total_edges,
    'matched_permuted_pooled_edge_set_changed_fraction': changed_fraction,
  }


def _safe_bundle_file(
    manifest_path: Path,
    relative_value: object,
    *,
    context: str,
) -> Path:
  relative = Path(_nonempty_string(relative_value, context=context))
  if relative.is_absolute() or '..' in relative.parts:
    raise ValueError(f'{context} must stay within the bundle directory')
  base = manifest_path.parent.resolve()
  result = (base / relative).resolve()
  try:
    result.relative_to(base)
  except ValueError as error:
    raise ValueError(f'{context} escapes the bundle directory') from error
  if not result.is_file():
    raise FileNotFoundError(result)
  return result


def load_record_bundle(
    manifest_path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  manifest_path = manifest_path.expanduser().resolve()
  payload = dict(_strict_fields(
    load_json(manifest_path), MANIFEST_FIELDS,
    context=f'manifest {manifest_path}'))
  committed = payload.pop('manifest_sha256')
  _lower_hex(committed, 64, context=f'{manifest_path}.manifest_sha256')
  if committed != canonical_sha256(payload):
    raise ValueError(f'topology manifest hash mismatch: {manifest_path}')
  payload['manifest_sha256'] = committed
  if payload['schema_version'] != SCHEMA_VERSION \
      or payload['artifact'] != MANIFEST_ARTIFACT:
    raise ValueError(f'invalid topology manifest identity: {manifest_path}')
  if payload['protocol_id'] != protocol['protocol_id'] \
      or payload['protocol_sha256'] != protocol_sha256:
    raise ValueError(f'{manifest_path} is bound to a different protocol')
  binding = validate_source_binding(payload['source_binding'])
  binding_sha = _lower_hex(
    payload['source_binding_sha256'], 64,
    context=f'{manifest_path}.source_binding_sha256')
  if binding_sha != canonical_sha256(binding):
    raise ValueError(f'source binding hash mismatch: {manifest_path}')
  entries = payload['record_files']
  if not isinstance(entries, list) or not entries:
    raise ValueError(f'{manifest_path} must contain record files')
  rows = []
  seen_paths = set()
  for file_index, entry_value in enumerate(entries):
    entry = _strict_fields(
      entry_value, RECORD_FILE_FIELDS,
      context=f'{manifest_path}.record_files[{file_index}]')
    path = _safe_bundle_file(
      manifest_path, entry['path'], context='record file path')
    if path in seen_paths:
      raise ValueError(f'duplicate record file: {path}')
    seen_paths.add(path)
    expected_sha = _lower_hex(
      entry['sha256'], 64, context=f'{path}.sha256')
    if sha256_file(path) != expected_sha:
      raise ValueError(f'topology record SHA256 mismatch: {path}')
    expected_count = _nonnegative_int(
      entry['num_records'], context=f'{path}.num_records')
    file_rows = []
    with path.open(encoding='utf-8') as handle:
      for line_number, line in enumerate(handle, start=1):
        if not line.strip():
          raise ValueError(f'{path}:{line_number} is blank')
        row = validate_record(
          _load_json_line(line, source=f'{path}:{line_number}'),
          protocol=protocol,
          protocol_sha256=protocol_sha256,
          source_binding=binding,
          source_binding_sha256=binding_sha,
          context=f'{path}:{line_number}')
        file_rows.append(row)
    if len(file_rows) != expected_count:
      raise ValueError(f'{path} record count differs from its manifest')
    rows.extend(file_rows)
  if _nonnegative_int(
      payload['num_records'], context=f'{manifest_path}.num_records') \
      != len(rows):
    raise ValueError(f'{manifest_path} aggregate record count differs')
  _validate_bundle_source_selection(
    rows, source_binding=binding, protocol=protocol)
  return rows, payload


def write_record_bundle(
    *,
    output_dir: Path,
    protocol: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Path:
  """Write one immutable, single-file evaluator bundle.

  Existing outputs are never overwritten. Spot retries must use a fresh
  directory so a failed attempt remains auditable.
  """
  protocol = validate_protocol(protocol)
  protocol_sha = canonical_sha256(protocol)
  binding = validate_source_binding(source_binding)
  binding_sha = canonical_sha256(binding)
  validated = [
    validate_record(
      record,
      protocol=protocol,
      protocol_sha256=protocol_sha,
      source_binding=binding,
      source_binding_sha256=binding_sha,
      context=f'record[{index}]')
    for index, record in enumerate(records)
  ]
  _validate_bundle_source_selection(
    validated, source_binding=binding, protocol=protocol)
  validate_complete_grid(validated, protocol=protocol)
  output_dir = output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  record_path = output_dir / 'topology_records.jsonl'
  manifest_path = output_dir / 'topology_records.manifest.json'
  if record_path.exists() or manifest_path.exists():
    raise FileExistsError(
      'topology bundle paths already exist; use a fresh attempt directory')
  ordered = sorted(validated, key=lambda record: record['record_id'])
  with record_path.open('x', encoding='utf-8') as handle:
    for record in ordered:
      handle.write(canonical_json(record) + '\n')
  body = {
    'schema_version': SCHEMA_VERSION,
    'artifact': MANIFEST_ARTIFACT,
    'protocol_id': protocol['protocol_id'],
    'protocol_sha256': protocol_sha,
    'source_binding': binding,
    'source_binding_sha256': binding_sha,
    'record_files': [{
      'path': record_path.name,
      'sha256': sha256_file(record_path),
      'num_records': len(ordered),
    }],
    'num_records': len(ordered),
  }
  manifest = {**body, 'manifest_sha256': canonical_sha256(body)}
  with manifest_path.open('x', encoding='utf-8') as handle:
    handle.write(json.dumps(
      manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
  return manifest_path


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
  if not sorted_values:
    raise ValueError('cannot compute a quantile of an empty sequence')
  if len(sorted_values) == 1:
    return float(sorted_values[0])
  position = probability * (len(sorted_values) - 1)
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return float(sorted_values[lower])
  weight = position - lower
  return float(
    sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
  finite = [float(value) for value in values]
  if any(not math.isfinite(value) for value in finite):
    raise ValueError('diagnostic summary received a non-finite value')
  if not finite:
    return {
      'count': 0, 'mean': None, 'minimum': None, 'q25': None,
      'median': None, 'q75': None, 'q90': None, 'maximum': None,
    }
  ordered = sorted(finite)
  return {
    'count': len(ordered),
    'mean': statistics.fmean(ordered),
    'minimum': ordered[0],
    'q25': _quantile(ordered, 0.25),
    'median': _quantile(ordered, 0.5),
    'q75': _quantile(ordered, 0.75),
    'q90': _quantile(ordered, 0.9),
    'maximum': ordered[-1],
  }


def _jaccard(
    left: set[tuple[int, int]],
    right: set[tuple[int, int]],
) -> float | None:
  union = left | right
  if not union:
    return None
  return len(left & right) / len(union)


def _pair_jaccard(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, float | None]:
  left_edges = _edge_set(left)
  right_edges = _edge_set(right)
  shared_nodes = set(left['active_nodes']) & set(right['active_nodes'])
  left_shared = {
    edge for edge in left_edges if edge[0] in shared_nodes
    and edge[1] in shared_nodes
  }
  right_shared = {
    edge for edge in right_edges if edge[0] in shared_nodes
    and edge[1] in shared_nodes
  }
  active_union = set(left['active_nodes']) | set(right['active_nodes'])
  return {
    'all_edge_jaccard': _jaccard(left_edges, right_edges),
    'shared_node_edge_jaccard': _jaccard(left_shared, right_shared),
    'shared_active_node_fraction': (
      len(shared_nodes) / len(active_union) if active_union else None),
  }


def _jaccard_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  return {
    field: _distribution([
      item[field] for item in values if item[field] is not None
    ])
    for field in (
      'all_edge_jaccard', 'shared_node_edge_jaccard',
      'shared_active_node_fraction')
  }


def _stability_across_corruptions(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
  learned = [record for record in records if record['intervention'] == 'learned']
  groups = defaultdict(dict)
  for record in learned:
    groups[(*_source_unit_key(record), record['requested_time_index'])][
      record['corruption_seed']] = record
  by_pair = []
  all_values = []
  seeds = protocol['corruption_seeds']
  for left_index, left_seed in enumerate(seeds):
    for right_seed in seeds[left_index + 1:]:
      values = [
        _pair_jaccard(group[left_seed], group[right_seed])
        for group in groups.values()
      ]
      all_values.extend(values)
      by_pair.append({
        'corruption_seeds': [left_seed, right_seed],
        'num_pairs': len(values),
        **_jaccard_summary(values),
      })
  return {
    'scope': 'learned_edges_same_source_unit_and_requested_time',
    'empty_shared_edge_unions_are_excluded': True,
    'overall': _jaccard_summary(all_values),
    'by_corruption_seed_pair': by_pair,
  }


def _stability_across_times(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
  learned = [record for record in records if record['intervention'] == 'learned']
  groups = defaultdict(dict)
  for record in learned:
    groups[(*_source_unit_key(record), record['corruption_seed'])][
      record['requested_time_index']] = record
  by_pair = []
  all_values = []
  for left_index, left_time in enumerate(protocol['time_points']):
    for right_index in range(left_index + 1, len(protocol['time_points'])):
      right_time = protocol['time_points'][right_index]
      values = [
        _pair_jaccard(group[left_index], group[right_index])
        for group in groups.values()
      ]
      all_values.extend(values)
      by_pair.append({
        'requested_times': [left_time, right_time],
        'absolute_time_gap': right_time - left_time,
        'num_pairs': len(values),
        **_jaccard_summary(values),
      })
  return {
    'scope': 'learned_edges_same_source_unit_and_corruption_seed',
    'empty_shared_edge_unions_are_excluded': True,
    'overall': _jaccard_summary(all_values),
    'by_requested_time_pair': by_pair,
  }


def _stability_across_training_seeds(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
  learned = [record for record in records if record['intervention'] == 'learned']
  groups = defaultdict(dict)
  for record in learned:
    source_key = canonical_json(_source_descriptor(record))
    groups[(source_key, record['corruption_seed'],
            record['requested_time_index'])][record['train_seed']] = record
  by_pair = []
  all_values = []
  seeds = protocol['source_selection']['train_seeds']
  for left_index, left_seed in enumerate(seeds):
    for right_seed in seeds[left_index + 1:]:
      values = [
        _pair_jaccard(group[left_seed], group[right_seed])
        for group in groups.values()]
      all_values.extend(values)
      by_pair.append({
        'training_seeds': [left_seed, right_seed],
        'num_pairs': len(values),
        **_jaccard_summary(values),
      })
  return {
    'scope': (
      'learned_edges_same_source_corruption_and_requested_time'),
    'corruption_inputs_required_identical': True,
    'empty_shared_edge_unions_are_excluded': True,
    'overall': _jaccard_summary(all_values),
    'by_training_seed_pair': by_pair,
  }


def _topology_summary(
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  edge_count = sum(item['edge_count'] for item in metrics)
  chain_count = sum(item['natural_chain_edge_count'] for item in metrics)
  overlap_count = sum(
    item['natural_chain_overlap_count'] for item in metrics)
  nonlocal_count = sum(item['nonlocal_edge_count'] for item in metrics)
  distance_counts = Counter()
  component_sizes = []
  depths = []
  diameters = []
  for item in metrics:
    distance_counts.update({
      int(distance): count
      for distance, count in item['edge_distance_histogram'].items()
    })
    component_sizes.extend(item['component_sizes'])
    depths.extend(item['minimum_position_rooted_depths'])
    diameters.extend(item['component_diameters'])
  distances = [
    distance
    for distance, count in sorted(distance_counts.items())
    for _ in range(count)
  ]
  return {
    'num_observations': len(metrics),
    'active_nodes': _distribution([
      item['active_node_count'] for item in metrics]),
    'edges': _distribution([item['edge_count'] for item in metrics]),
    'pooled_natural_chain_precision': (
      overlap_count / edge_count if edge_count else None),
    'pooled_natural_chain_recall': (
      overlap_count / chain_count if chain_count else None),
    'pooled_nonlocal_edge_fraction': (
      nonlocal_count / edge_count if edge_count else None),
    'edge_distance': {
      'histogram': {
        str(distance): distance_counts[distance]
        for distance in sorted(distance_counts)
      },
      'distribution': _distribution(distances),
    },
    'component_size': _distribution(component_sizes),
    'minimum_position_rooted_depth': _distribution(depths),
    'component_diameter': _distribution(diameters),
  }


def _intervention_comparisons(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  by_key = {_record_grid_key(record): record for record in records}
  learned = [record for record in records if record['intervention'] == 'learned']
  result = {}
  for intervention in INTERVENTIONS[1:]:
    jaccards = []
    retained = 0
    learned_edges = 0
    intervention_edges = 0
    for reference in learned:
      candidate = by_key[(*_record_grid_key(reference)[:-1], intervention)]
      comparison = _pair_jaccard(reference, candidate)
      jaccards.append(comparison)
      reference_edges = _edge_set(reference)
      candidate_edges = _edge_set(candidate)
      retained += len(reference_edges & candidate_edges)
      learned_edges += len(reference_edges)
      intervention_edges += len(candidate_edges)
    result[intervention] = {
      'num_pairs': len(jaccards),
      'jaccard': _jaccard_summary(jaccards),
      'pooled_learned_edge_retention': (
        retained / learned_edges if learned_edges else None),
      'pooled_edge_set_changed_fraction': (
        1.0 - retained / learned_edges if learned_edges else None),
      'pooled_intervention_to_learned_edge_count_ratio': (
        intervention_edges / learned_edges if learned_edges else None),
    }
  return result


def _marker_output(
    marker: Mapping[str, Any], name: str,
) -> Mapping[str, Any]:
  matches = [item for item in marker['outputs'] if item['name'] == name]
  if len(matches) != 1:
    raise ValueError(
      f'completed topology job requires exactly one {name!r} output')
  return matches[0]


def _marker_output_path(
    marker: Mapping[str, Any], output: Mapping[str, Any],
) -> Path:
  run_dir = Path(marker['run_dir']).expanduser().resolve()
  path = (run_dir / output['relative_path']).resolve()
  try:
    path.relative_to(run_dir)
  except ValueError as error:
    raise ValueError('completed topology output escapes its run directory') \
      from error
  if not path.is_file() or sha256_file(path) != output['sha256']:
    raise ValueError(f'completed topology output hash drifted: {path}')
  return path


def validate_gpu_exclusivity_evidence(
    value: object,
    *,
    expected_job_id: str,
) -> dict[str, Any]:
  """Validate the fail-closed shared-GPU gate recorded by an eval job."""
  payload = dict(_strict_fields(
    value, GPU_EXCLUSIVITY_FIELDS, context='topology GPU exclusivity'))
  if (payload['schema_version'] != SCHEMA_VERSION
      or payload['artifact'] != GPU_EXCLUSIVITY_ARTIFACT
      or payload['job_id'] != expected_job_id):
    raise ValueError('invalid topology GPU-exclusivity identity')
  if (payload['required'] is not True
      or payload['policy'] != GPU_EXCLUSIVITY_POLICY
      or payload['lock_acquired'] is not True):
    raise ValueError('topology GPU-exclusivity policy was not enforced')
  lock_path = Path(_nonempty_string(
    payload['lock_path'], context='topology GPU lock path'))
  if lock_path.expanduser().resolve(strict=False) != SUBMISSION_GPU_LOCK:
    raise ValueError('topology GPU lock path differs from the submission lock')
  if _finite_float(
      payload['monitor_interval_seconds'],
      context='topology GPU monitor interval') != \
      GPU_MONITOR_INTERVAL_SECONDS:
    raise ValueError('topology GPU monitor interval must equal one second')
  if _positive_int(
      payload['monitor_samples'], context='topology GPU monitor samples') < 3:
    raise ValueError('topology GPU monitor did not sample pre/during/post run')
  for field in (
      'preflight_other_compute_pids', 'postflight_other_compute_pids',
      'foreign_pid_observations', 'monitor_errors'):
    if payload[field] != []:
      raise ValueError(
        f'topology GPU exclusivity observed a nonempty {field}')
  return payload


def _validate_dataset_provenance_file(
    path: Path,
    *,
    expected_sha256: str,
    dataset_specification: Mapping[str, Any],
) -> dict[str, Any]:
  if sha256_file(path) != expected_sha256:
    raise ValueError('dataset provenance file SHA256 mismatch')
  payload = load_json(path)
  expected_fields = {
    'schema_version', 'artifact', 'specification', 'observed',
    'specification_sha256', 'manifest_sha256'}
  payload = dict(_strict_fields(
    payload, expected_fields, context='dataset provenance'))
  body = dict(payload)
  committed = body.pop('manifest_sha256')
  if (payload['schema_version'] != 1
      or payload['artifact'] != 'pinned_text_dataset_provenance'
      or committed != canonical_sha256(body)):
    raise ValueError('invalid dataset provenance identity or self-hash')
  specification = payload['specification']
  if (not isinstance(specification, Mapping)
      or payload['specification_sha256']
      != canonical_sha256(specification)):
    raise ValueError('invalid dataset provenance specification commitment')
  for field, expected in (
      ('source_revision', dataset_specification['dataset_revision']),
      ('tokenizer_revision', dataset_specification['tokenizer_revision'])):
    if specification.get(field) != expected:
      raise ValueError(
        f'dataset provenance {field} differs from topology protocol')
  if specification.get('document_boundary_mode') not in {
      'source_document', 'wikitext_articles'}:
    raise ValueError(
      'topology dataset provenance is not document-local')
  return payload


def _validate_adapter_export(
    *,
    adapter_path: Path,
    manifest_path: Path,
    expected_adapter_sha256: str,
    expected_manifest_sha256: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
  """Rehash and structurally validate the exact adapter export pair."""
  if sha256_file(adapter_path) != expected_adapter_sha256:
    raise ValueError('adapter bytes differ from their topology binding')
  if sha256_file(manifest_path) != expected_manifest_sha256:
    raise ValueError('adapter manifest differs from its topology binding')
  payload = load_json(manifest_path)
  if not isinstance(payload, Mapping):
    raise TypeError('adapter manifest must be a JSON object')
  identity = payload.get('structured_decoder_identity')
  if not isinstance(identity, Mapping):
    raise ValueError('adapter manifest lacks structured decoder identity')
  # The exporter validates every safetensors header, tensor, and released
  # backbone commitment. Supplying the manifest identity back as the runtime
  # identity validates bytes against that identity; the frozen topology
  # semantics are then checked independently below.
  from scripts.export_structured_adapter import (  # pylint: disable=import-outside-toplevel
    load_and_validate_adapter_manifest,
  )
  validated = load_and_validate_adapter_manifest(
    manifest_path,
    adapter_path,
    expected_identity=identity,
    expected_adapter_sha256=expected_adapter_sha256,
    expected_manifest_sha256=expected_manifest_sha256)
  head_semantics = identity.get('head_semantics')
  if (identity.get('control_identity')
      != protocol['source_selection']['arm']
      or identity.get('topology_mode') != 'dynamic'
      or identity.get('factor_mode') != 'dynamic'
      or identity.get('candidate_top_k') != protocol['candidate_top_k']
      or identity.get('independent_mode') is not False
      or not isinstance(head_semantics, Mapping)
      or head_semantics.get('component_size_cap')
      != protocol['component_size_cap']):
    raise ValueError(
      'adapter export identity differs from frozen topology semantics')
  return validated


def _dependency_markers(
    job: Mapping[str, Any],
    *,
    jobs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
  from scripts.run_compiled_job import _validated_marker  # noqa: PLC0415

  result: dict[str, Mapping[str, Any]] = {}

  def visit(job_id: str) -> None:
    if job_id in result:
      return
    dependency = jobs[job_id]
    for nested in dependency['dependencies']:
      visit(nested)
    marker = _validated_marker(dependency, required=True)
    assert marker is not None
    result[job_id] = marker

  for dependency_id in job['dependencies']:
    visit(dependency_id)
  return result


def _topology_jobs(
    jobs: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
  selected = []
  for job in jobs.values():
    output_names = {
      item.get('name') for item in job.get('required_outputs', [])
      if isinstance(item, Mapping)}
    if 'topology_record_manifest' in output_names:
      selected.append(job)
  return sorted(selected, key=lambda item: item['job_id'])


def load_trusted_plan_bundles(
    *,
    plan_dir: Path,
    protocol_path: Path,
) -> tuple[
    dict[str, Any], str,
    list[tuple[list[dict[str, Any]], dict[str, Any]]],
    dict[str, Any],
]:
  """Load topology outputs through the repository-trusted compiled plan."""
  from scripts.run_compiled_job import (  # noqa: PLC0415
    SUCCESS_MARKER,
    _job_execution_digest,
    _load_plan,
    _validate_repository_checkout,
    _validated_marker,
  )

  protocol, protocol_sha = read_protocol(protocol_path)
  _validate_trusted_protocol_path(protocol_path, protocol)
  plan_dir = plan_dir.expanduser().resolve()
  plan_path = plan_dir / 'compiled-plan.json'
  plan, jobs = _load_plan(plan_dir)
  _validate_repository_checkout(plan, repo_root=REPO_ROOT)
  validate_compiled_topology_plan_lineage(
    plan,
    jobs=jobs,
    plan_dir=plan_dir,
    protocol_path=protocol_path,
    protocol=protocol,
    protocol_sha256=protocol_sha)
  plan_sha = sha256_file(plan_path)
  repository = plan['repository']
  selected_jobs = _topology_jobs(jobs)
  expected_grid = {
    (dataset, train_seed)
    for dataset in protocol['source_selection']['datasets']
    for train_seed in protocol['source_selection']['train_seeds']}
  observed_grid = {}
  bundles = []
  committed_jobs = {}
  committed_dependencies = {}
  validated_adapters: set[tuple[str, str]] = set()

  for job in selected_jobs:
    if job['kind'] != 'eval':
      raise ValueError('topology record manifests may only come from eval jobs')
    identity = job['identity']
    required_identity = {'control', 'dataset', 'train_seed', 'candidate_k'}
    if not required_identity.issubset(identity):
      raise ValueError('topology job identity is incomplete')
    cell = (identity['dataset'], identity['train_seed'])
    if cell in observed_grid:
      raise ValueError(f'duplicate topology job for {cell}')
    observed_grid[cell] = job['job_id']
    if (identity['control'] != protocol['source_selection']['arm']
        or identity['candidate_k'] != protocol['candidate_top_k']
        or cell not in expected_grid):
      raise ValueError('topology job is outside the frozen source grid')

    dependency_markers = _dependency_markers(job, jobs=jobs)
    for dependency_id, dependency_marker in dependency_markers.items():
      dependency_job = jobs[dependency_id]
      dependency_marker_path = (
        Path(dependency_job['artifact_dir']).resolve() / SUCCESS_MARKER)
      committed_dependencies[dependency_id] = {
        'job_spec_sha256': plan['job_spec_sha256'][dependency_id],
        'job_execution_sha256': _job_execution_digest(dependency_job),
        'success_marker_sha256': sha256_file(dependency_marker_path),
        'outputs': [dict(item) for item in dependency_marker['outputs']],
      }
    marker = _validated_marker(job, required=True)
    assert marker is not None
    outputs = {
      name: _marker_output(marker, name) for name in (
        'topology_record_manifest', 'topology_records',
        'topology_source_selection', 'dataset_provenance',
        'gpu_exclusivity')}
    paths = {
      name: _marker_output_path(marker, output)
      for name, output in outputs.items()}
    record_manifest_sha = outputs['topology_record_manifest']['sha256']
    records, record_manifest = load_record_bundle(
      paths['topology_record_manifest'], protocol=protocol,
      protocol_sha256=protocol_sha)
    if len(record_manifest['record_files']) != 1:
      raise ValueError('trusted topology jobs require one canonical record file')
    if (record_manifest['record_files'][0]['sha256']
        != outputs['topology_records']['sha256']
        or paths['topology_records']
        != (paths['topology_record_manifest'].parent
            / record_manifest['record_files'][0]['path']).resolve()):
      raise ValueError('success marker and topology manifest disagree on records')
    binding = record_manifest['source_binding']
    expected_binding = {
      'compiled_plan_sha256': plan_sha,
      'plan_id': plan['plan_id'],
      'job_spec_sha256': plan['job_spec_sha256'][job['job_id']],
      'job_execution_sha256': _job_execution_digest(job),
      'repository_sha': repository['sha'],
      'repository_clean': True,
      'job_id': job['job_id'],
      'arm': identity['control'],
      'dataset': identity['dataset'],
      'train_seed': identity['train_seed'],
    }
    mismatches = {
      field: {'expected': expected, 'observed': binding.get(field)}
      for field, expected in expected_binding.items()
      if binding.get(field) != expected}
    if mismatches:
      raise ValueError(
        f'topology source binding differs from compiled job: {mismatches}')

    dataset_specification = protocol['source_selection']['datasets'][
      identity['dataset']]
    data_config_path = (
      REPO_ROOT / dataset_specification['data_config_path']).resolve()
    if (not data_config_path.is_file()
        or binding['data_config_sha256'] != sha256_file(data_config_path)):
      raise ValueError('topology data-config commitment is not authentic')
    import yaml  # noqa: PLC0415
    data_config = yaml.safe_load(data_config_path.read_text())
    if (not isinstance(data_config, Mapping)
        or data_config.get('valid_revision')
        != dataset_specification['dataset_revision']
        or data_config.get('tokenizer_revision')
        != dataset_specification['tokenizer_revision']
        or data_config.get('valid_document_boundary_mode') not in {
          'source_document', 'wikitext_articles'}):
      raise ValueError('topology data config differs from frozen semantics')
    if (binding['dataset_provenance_sha256']
        != outputs['dataset_provenance']['sha256']):
      raise ValueError('topology dataset-provenance commitment drifted')
    _validate_dataset_provenance_file(
      paths['dataset_provenance'],
      expected_sha256=binding['dataset_provenance_sha256'],
      dataset_specification=dataset_specification)
    validate_gpu_exclusivity_evidence(
      load_json(paths['gpu_exclusivity']), expected_job_id=job['job_id'])
    evaluator_source_path = (
      REPO_ROOT / protocol['evaluator_source_path']).resolve()
    if (not evaluator_source_path.is_file()
        or binding['evaluator_source_sha256']
        != sha256_file(evaluator_source_path)):
      raise ValueError('topology evaluator source commitment is not authentic')

    selection = validate_source_selection_manifest(
      load_json(paths['topology_source_selection']),
      protocol=protocol, protocol_sha256=protocol_sha,
      dataset=identity['dataset'])
    if (binding['source_selection_sha256']
        != selection['selection_sha256']
        or source_selection_sha256(records)
        != selection['selection_sha256']):
      raise ValueError('ordered topology source selection drifted')
    record_descriptors = {
      record['selection_index']: _source_descriptor(record)
      for record in records}
    if [record_descriptors[index] for index in sorted(record_descriptors)] \
        != selection['entries']:
      raise ValueError('topology records differ from ordered source manifest')

    adapter_candidates = []
    for dependency_id, dependency_marker in dependency_markers.items():
      names = {item['name'] for item in dependency_marker['outputs']}
      if {'adapter', 'adapter_manifest'}.issubset(names):
        dependency = jobs[dependency_id]
        if (dependency['identity'].get('control') == identity['control']
            and dependency['identity'].get('train_seed')
            == identity['train_seed']):
          adapter_candidates.append((dependency, dependency_marker))
    if len(adapter_candidates) != 1:
      raise ValueError(
        'topology job must have exactly one matching adapter export dependency')
    adapter_job, adapter_marker = adapter_candidates[0]
    adapter_output = _marker_output(adapter_marker, 'adapter')
    adapter_manifest_output = _marker_output(
      adapter_marker, 'adapter_manifest')
    adapter_path = _marker_output_path(adapter_marker, adapter_output)
    adapter_manifest_path = _marker_output_path(
      adapter_marker, adapter_manifest_output)
    if (binding['adapter_sha256'] != adapter_output['sha256']
        or binding['adapter_export_manifest_sha256']
        != adapter_manifest_output['sha256']):
      raise ValueError('topology adapter export binding drifted')
    adapter_key = (
      binding['adapter_sha256'], binding['adapter_export_manifest_sha256'])
    if adapter_key not in validated_adapters:
      _validate_adapter_export(
        adapter_path=adapter_path, manifest_path=adapter_manifest_path,
        expected_adapter_sha256=binding['adapter_sha256'],
        expected_manifest_sha256=(
          binding['adapter_export_manifest_sha256']),
        protocol=protocol)
      validated_adapters.add(adapter_key)

    marker_path = Path(job['artifact_dir']).resolve() / SUCCESS_MARKER
    committed_jobs[job['job_id']] = {
      'job_spec_sha256': plan['job_spec_sha256'][job['job_id']],
      'job_execution_sha256': _job_execution_digest(job),
      'success_marker_sha256': sha256_file(marker_path),
      'outputs': [dict(item) for item in marker['outputs']],
      'topology_record_manifest_sha256': record_manifest_sha,
      'gpu_exclusivity_sha256': outputs['gpu_exclusivity']['sha256'],
      'adapter_job_id': adapter_job['job_id'],
      'adapter_sha256': binding['adapter_sha256'],
      'adapter_export_manifest_sha256': (
        binding['adapter_export_manifest_sha256']),
    }
    bundles.append((records, record_manifest))

  if set(observed_grid) != expected_grid:
    raise ValueError(
      'compiled topology job grid is incomplete: '
      f'missing={sorted(expected_grid - set(observed_grid))}, '
      f'extra={sorted(set(observed_grid) - expected_grid)}')
  integrity_body = {
    'schema_version': SCHEMA_VERSION,
    'artifact': SOURCE_INTEGRITY_ARTIFACT,
    'protocol_id': protocol['protocol_id'],
    'protocol_sha256': protocol_sha,
    'compiled_plan_sha256': plan_sha,
    'plan_id': plan['plan_id'],
    'source_manifest_sha256': plan['source_manifest_sha256'],
    'repository_sha': repository['sha'],
    'repository_clean': repository['dirty'] is False,
    'validated_job_ids': sorted(committed_jobs),
    'jobs': {job_id: committed_jobs[job_id]
             for job_id in sorted(committed_jobs)},
    'dependencies': {
      job_id: committed_dependencies[job_id]
      for job_id in sorted(committed_dependencies)},
  }
  source_integrity = {
    **integrity_body,
    'commitment_sha256': canonical_sha256(integrity_body),
  }
  return protocol, protocol_sha, bundles, source_integrity


def build_analysis(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    bundles: Sequence[tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]],
    source_integrity: Mapping[str, Any],
) -> dict[str, Any]:
  protocol = validate_protocol(protocol)
  if canonical_sha256(protocol) != protocol_sha256:
    raise ValueError('provided protocol SHA256 does not match the protocol')
  if not bundles:
    raise ValueError('at least one topology bundle is required')
  integrity = dict(_strict_fields(
    source_integrity, SOURCE_INTEGRITY_FIELDS,
    context='source_integrity'))
  committed_integrity = integrity.pop('commitment_sha256', None)
  _lower_hex(
    committed_integrity, 64,
    context='source_integrity.commitment_sha256')
  if (committed_integrity != canonical_sha256(integrity)
      or integrity.get('schema_version') != SCHEMA_VERSION
      or integrity.get('artifact') != SOURCE_INTEGRITY_ARTIFACT
      or integrity.get('protocol_id') != protocol['protocol_id']
      or integrity.get('protocol_sha256') != protocol_sha256
      or integrity.get('repository_clean') is not True):
    raise ValueError('invalid topology source-integrity commitment')
  source_integrity = {
    **integrity, 'commitment_sha256': committed_integrity}
  _validate_source_bundle_grid(bundles, protocol=protocol)
  records = []
  sources = []
  seen_manifests = set()
  seen_bindings = set()
  for bundle_records, manifest in bundles:
    manifest_sha = manifest['manifest_sha256']
    binding_sha = manifest['source_binding_sha256']
    if manifest_sha in seen_manifests:
      raise ValueError(f'duplicate topology manifest {manifest_sha}')
    if binding_sha in seen_bindings:
      raise ValueError(f'duplicate topology source binding {binding_sha}')
    seen_manifests.add(manifest_sha)
    seen_bindings.add(binding_sha)
    records.extend(bundle_records)
    sources.append({
      'manifest_sha256': manifest_sha,
      'source_binding_sha256': binding_sha,
      'source_binding': manifest['source_binding'],
      'num_records': manifest['num_records'],
      'record_file_sha256': sorted(
        entry['sha256'] for entry in manifest['record_files']),
    })
  grid = validate_complete_grid(records, protocol=protocol)
  observation_metrics = [
    topology_metrics(
      record, nonlocal_threshold=protocol['nonlocal_edge_threshold'])
    for record in sorted(records, key=lambda item: item['record_id'])
  ]
  grouped = defaultdict(list)
  for item in observation_metrics:
    grouped[(item['intervention'], item['requested_time_index'])].append(item)
  by_intervention_and_time = []
  for intervention in INTERVENTIONS:
    for time_index, requested_time in enumerate(protocol['time_points']):
      by_intervention_and_time.append({
        'intervention': intervention,
        'requested_time_index': time_index,
        'requested_time': requested_time,
        'summary': _topology_summary(grouped[(intervention, time_index)]),
      })
  body = {
    'schema_version': SCHEMA_VERSION,
    'artifact': ANALYSIS_ARTIFACT,
    'scientific_scope': protocol['scientific_scope'],
    'protocol_id': protocol['protocol_id'],
    'protocol_sha256': protocol_sha256,
    'source_manifests': sorted(
      sources, key=lambda item: item['manifest_sha256']),
    'source_integrity': source_integrity,
    'input_commitment_sha256': canonical_sha256({
      'record_manifests': sorted(seen_manifests),
      'source_integrity': committed_integrity,
    }),
    'grid_validation': grid,
    'metric_definitions': {
      'natural_order_chain': protocol['natural_order_chain'],
      'nonlocal_edge': (
        f'absolute_position_distance_gt_'
        f'{protocol["nonlocal_edge_threshold"]}'),
      'component_depth_root': protocol['component_depth_root'],
      'all_edge_jaccard': 'intersection_over_union_of_canonical_edge_sets',
      'shared_node_edge_jaccard': (
        'intersection_over_union_after_restricting_both_edge_sets_to_'
        'shared_active_nodes'),
      'empty_jaccard_policy': 'exclude_pairs_with_empty_edge_union',
    },
    'observation_metrics': observation_metrics,
    'observation_metrics_sha256': canonical_sha256(observation_metrics),
    'learned_topology_summary': _topology_summary([
      item for item in observation_metrics
      if item['intervention'] == 'learned']),
    'learned_topology_by_dataset': [
      {
        'dataset': dataset,
        'summary': _topology_summary([
          item for item in observation_metrics
          if item['intervention'] == 'learned'
          and item['dataset'] == dataset]),
      }
      for dataset in sorted(protocol['source_selection']['datasets'])
    ],
    'by_intervention_and_requested_time': by_intervention_and_time,
    'learned_edge_stability': {
      'across_corruption_seeds': _stability_across_corruptions(
        records, protocol),
      'across_requested_times': _stability_across_times(records, protocol),
      'across_training_seeds': _stability_across_training_seeds(
        records, protocol),
    },
    'intervention_comparisons': _intervention_comparisons(records),
  }
  return {**body, 'analysis_sha256': canonical_sha256(body)}


def aggregate_plan(
    *, plan_dir: Path, protocol_path: Path,
) -> dict[str, Any]:
  protocol, protocol_sha, bundles, source_integrity = \
    load_trusted_plan_bundles(
      plan_dir=plan_dir, protocol_path=protocol_path)
  return build_analysis(
    protocol=protocol, protocol_sha256=protocol_sha, bundles=bundles,
    source_integrity=source_integrity)


def validate_analysis(value: object) -> dict[str, Any]:
  payload = dict(_strict_fields(
    value, ANALYSIS_FIELDS, context='topology analysis'))
  committed = payload.pop('analysis_sha256', None)
  _lower_hex(committed, 64, context='analysis_sha256')
  if committed != canonical_sha256(payload):
    raise ValueError('topology analysis SHA256 mismatch')
  if payload.get('schema_version') != SCHEMA_VERSION \
      or payload.get('artifact') != ANALYSIS_ARTIFACT:
    raise ValueError('invalid topology analysis identity')
  if payload.get('observation_metrics_sha256') != canonical_sha256(
      payload.get('observation_metrics')):
    raise ValueError('topology observation-metrics commitment mismatch')
  payload['analysis_sha256'] = committed
  return payload


def verify_replay(
    *,
    analysis_path: Path,
    protocol_path: Path,
    plan_dir: Path,
) -> dict[str, Any]:
  observed = validate_analysis(load_json(analysis_path))
  expected = aggregate_plan(
    plan_dir=plan_dir, protocol_path=protocol_path)
  if observed != expected:
    raise ValueError(
      'topology analysis differs from deterministic raw-record replay')
  return observed
