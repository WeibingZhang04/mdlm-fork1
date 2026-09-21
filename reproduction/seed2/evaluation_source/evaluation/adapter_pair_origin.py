"""Cryptographic origin evidence for the paired generation adapters.

The generation study compares the learned ``dynamic_dynamic`` adapter with its
paired ``static_static`` control.  Adapter and manifest hashes alone do not
prove that those files came from the frozen experiment plan.  This module
walks backward through the completed export and training jobs, validates every
success marker and output again, and emits one immutable evidence document for
the pair.

Both accepted plan formats go through the authoritative analysis loaders: the
single policy-pinned schema-v1 K=64 pilot and normal schema-v2 plans (including
the K=128 candidate pilot).  Evidence validation replays the complete walk; it
does not trust the evidence document merely because its self-hash is valid.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts.aggregate_hierarchical_document_eval import (
  _load_plan_for_analysis,
  _validated_analysis_marker,
)
from scripts.compile_experiment_matrix import sha256_file
from scripts.export_structured_adapter import (
  ADAPTER_PREFIX,
  RELEASE_TENSOR_COUNT,
  RELEASED_BACKBONE_IDENTITY,
  load_adapter_state,
  load_and_validate_adapter_manifest,
  safetensors_metadata_from_bytes,
)
from scripts.run_compiled_job import (
  SUCCESS_MARKER,
  _job_execution_digest,
)


SCHEMA_VERSION = 1
ARTIFACT_ROLE = 'contextual_forest_adapter_pair_origin_evidence'
GENERATION_BINDING_ARTIFACT_ROLE = (
  'contextual_forest_generation_adapter_origin_binding')
ARMS = ('dynamic_dynamic', 'static_static')
TRAIN_OUTPUTS = {
  'checkpoint',
  'training_data_provenance',
  'training_validation_data_provenance',
}
EXPORT_OUTPUTS = {'adapter', 'adapter_manifest'}
SOURCE_FIELDS = {
  'compiled_plan_path', 'compiled_plan_sha256', 'plan_id', 'protocol_id',
  'source_manifest_sha256', 'repository', 'suite', 'candidate_k',
  'train_seed', 'legacy_plan_schema',
}
TOP_LEVEL_FIELDS = {
  'schema_version', 'artifact', 'created_utc', 'source', 'arms',
  'evidence_sha256',
}


def _reject_constant(value: str) -> None:
  raise ValueError(f'non-finite JSON number is forbidden: {value}')


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f'duplicate JSON key is forbidden: {key!r}')
    result[key] = value
  return result


def _strict_json_bytes(payload: bytes, *, source: Path) -> dict[str, Any]:
  try:
    value = json.loads(
      payload,
      parse_constant=_reject_constant,
      object_pairs_hook=_reject_duplicate_pairs,
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f'{source} is not strict JSON') from error
  if not isinstance(value, dict):
    raise TypeError(f'{source} must contain a JSON object')
  return value


def _canonical_json(value: object) -> str:
  try:
    return json.dumps(
      value, sort_keys=True, separators=(',', ':'), allow_nan=False)
  except (TypeError, ValueError) as error:
    raise ValueError('adapter-origin evidence is not canonical JSON data') \
      from error


def _canonical_sha256(value: object) -> str:
  return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _lower_hex(value: object, length: int, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != length
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(
      f'{context} must be {length} lowercase hexadecimal digits')
  return value


def _positive_int(value: object, *, context: str) -> int:
  if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
    raise ValueError(f'{context} must be a positive integer')
  return value


def _nonnegative_int(value: object, *, context: str) -> int:
  if (not isinstance(value, int) or isinstance(value, bool) or value < 0):
    raise ValueError(f'{context} must be a non-negative integer')
  return value


def _finite_float(value: object, *, context: str) -> float:
  if (not isinstance(value, (int, float)) or isinstance(value, bool)
      or not math.isfinite(float(value))):
    raise ValueError(f'{context} must be finite')
  return float(value)


def _utc_timestamp(value: str | None) -> str:
  if value is None:
    return dt.datetime.now(dt.timezone.utc).isoformat()
  if not isinstance(value, str) or not value:
    raise ValueError('created_utc must be a non-empty ISO-8601 timestamp')
  try:
    parsed = dt.datetime.fromisoformat(value)
  except ValueError as error:
    raise ValueError('created_utc must be an ISO-8601 timestamp') from error
  if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
    raise ValueError('created_utc must carry an explicit UTC offset')
  return value


def _output_index(
    marker: Mapping[str, Any],
    *,
    expected_names: set[str],
    context: str,
) -> dict[str, dict[str, Any]]:
  raw_outputs = marker.get('outputs')
  if not isinstance(raw_outputs, list):
    raise ValueError(f'{context} marker outputs must be a list')
  outputs: dict[str, dict[str, Any]] = {}
  run_dir = Path(marker['run_dir']).expanduser().resolve()
  for raw in raw_outputs:
    if not isinstance(raw, Mapping) or set(raw) != {
        'name', 'relative_path', 'size_bytes', 'sha256'}:
      raise ValueError(f'{context} marker has an invalid output record')
    name = raw['name']
    if not isinstance(name, str) or name in outputs:
      raise ValueError(f'{context} marker repeats an output name')
    relative_path = raw['relative_path']
    if (not isinstance(relative_path, str)
        or Path(relative_path).is_absolute()
        or '..' in Path(relative_path).parts):
      raise ValueError(f'{context} marker has an unsafe output path')
    size_bytes = _positive_int(
      raw['size_bytes'], context=f'{context}.{name}.size_bytes')
    digest = _lower_hex(
      raw['sha256'], 64, context=f'{context}.{name}.sha256')
    path = (run_dir / relative_path).resolve()
    try:
      path.relative_to(run_dir)
    except ValueError as error:
      raise ValueError(f'{context} output escapes its run directory') \
        from error
    outputs[name] = {
      'path': str(path),
      'relative_path': relative_path,
      'size_bytes': size_bytes,
      'sha256': digest,
    }
  if set(outputs) != expected_names:
    raise ValueError(
      f'{context} outputs differ from the frozen contract: '
      f'expected={sorted(expected_names)}, found={sorted(outputs)}')
  return outputs


def _flag_value(argv: object, flag: str, *, context: str) -> str:
  if (not isinstance(argv, list)
      or any(not isinstance(token, str) or not token for token in argv)):
    raise ValueError(f'{context} argv must contain non-empty strings')
  positions = [index for index, token in enumerate(argv) if token == flag]
  if len(positions) != 1 or positions[0] + 1 >= len(argv):
    raise ValueError(f'{context} must contain exactly one {flag}')
  return argv[positions[0] + 1]


def _validate_legacy_adapter_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    adapter_path: Path,
    expected_adapter_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
  """Validate the exact schema emitted by the pinned legacy K=64 plan."""
  expected_fields = {
    'artifact_role', 'schema_version', 'format', 'adapter_file',
    'adapter_sha256', 'adapter_size_bytes', 'adapter_tensor_count',
    'adapter_parameter_count', 'adapter_tensor_bytes',
    'adapter_namespace_in_source', 'adapter_namespace_in_file',
    'tensor_schema', 'source_checkpoint_sha256',
    'source_checkpoint_size_bytes', 'source_checkpoint_global_step',
    'source_state_dict_tensor_count',
    'omitted_frozen_backbone_tensor_count', 'ema_available', 'ema_used',
    'required_loader', 'required_loader_strict', 'released_backbone',
  }
  if set(payload) != expected_fields:
    raise ValueError(
      'legacy structured adapter manifest schema mismatch: '
      f'missing={sorted(expected_fields - set(payload))}, '
      f'unknown={sorted(set(payload) - expected_fields)}')
  result = dict(payload)
  if (result['artifact_role'] != 'contextual_forest_structured_adapter'
      or result['schema_version'] != 1
      or result['format'] != 'safetensors'):
    raise ValueError('unsupported legacy structured adapter identity')
  if result['adapter_file'] != adapter_path.name:
    raise ValueError('legacy adapter filename differs from its manifest')
  if sha256_file(manifest_path) != expected_manifest_sha256:
    raise ValueError('legacy adapter manifest SHA256 mismatch')
  actual_adapter_sha256 = sha256_file(adapter_path)
  if actual_adapter_sha256 != expected_adapter_sha256:
    raise ValueError('legacy structured adapter SHA256 mismatch')
  if _lower_hex(
      result['adapter_sha256'], 64,
      context='legacy manifest adapter SHA256') != actual_adapter_sha256:
    raise ValueError('legacy adapter manifest differs from adapter bytes')
  if _positive_int(
      result['adapter_size_bytes'],
      context='legacy adapter_size_bytes') != adapter_path.stat().st_size:
    raise ValueError('legacy adapter size differs from its manifest')

  adapter_state = load_adapter_state(
    adapter_path, expected_sha256=actual_adapter_sha256)
  tensor_schema = {
    key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
    for key, value in sorted(adapter_state.items())
  }
  derived = {
    'adapter_tensor_count': len(adapter_state),
    'adapter_parameter_count': sum(
      value.numel() for value in adapter_state.values()),
    'adapter_tensor_bytes': sum(
      value.numel() * value.element_size()
      for value in adapter_state.values()),
    'adapter_namespace_in_source': f'{ADAPTER_PREFIX}*',
    'adapter_namespace_in_file': 'prefix-stripped',
    'tensor_schema': tensor_schema,
  }
  for field, expected in derived.items():
    if result[field] != expected:
      raise ValueError(
        f'legacy adapter manifest {field} differs from adapter bytes')
  metadata = safetensors_metadata_from_bytes(adapter_path.read_bytes())
  if metadata != {
      'artifact_role': 'contextual_forest_structured_head',
      'source_namespace': ADAPTER_PREFIX,
      'file_namespace': 'prefix-stripped'}:
    raise ValueError('legacy adapter safetensors metadata mismatch')

  _lower_hex(
    result['source_checkpoint_sha256'], 64,
    context='legacy source checkpoint SHA256')
  _positive_int(
    result['source_checkpoint_size_bytes'],
    context='legacy source checkpoint size')
  _nonnegative_int(
    result['source_checkpoint_global_step'],
    context='legacy source checkpoint global step')
  source_count = _positive_int(
    result['source_state_dict_tensor_count'],
    context='legacy source state tensor count')
  omitted_count = _positive_int(
    result['omitted_frozen_backbone_tensor_count'],
    context='legacy omitted backbone tensor count')
  if omitted_count != RELEASE_TENSOR_COUNT:
    raise ValueError('legacy adapter omitted an unexpected backbone')
  if source_count != omitted_count + len(adapter_state):
    raise ValueError('legacy source tensor count is inconsistent')
  if result['released_backbone'] != RELEASED_BACKBONE_IDENTITY:
    raise ValueError('legacy adapter does not bind the pinned release')
  if (result['ema_available'] is not False
      or result['ema_used'] is not False
      or result['required_loader_strict'] is not True
      or result['required_loader'] !=
      'scripts.export_structured_adapter.load_adapter_into_head'):
    raise ValueError('legacy adapter loader or EMA policy is invalid')
  return result


def _load_and_validate_adapter_origin_manifest(
    *,
    manifest_path: Path,
    adapter_path: Path,
    manifest_payload: Mapping[str, Any],
    legacy_plan: bool,
    expected_adapter_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
  schema_version = manifest_payload.get('schema_version')
  if schema_version == 1:
    if not legacy_plan:
      raise ValueError(
        'legacy adapter manifests are accepted only for the pinned legacy plan')
    return _validate_legacy_adapter_manifest(
      manifest_payload,
      manifest_path=manifest_path,
      adapter_path=adapter_path,
      expected_adapter_sha256=expected_adapter_sha256,
      expected_manifest_sha256=expected_manifest_sha256,
    )
  if schema_version != 4:
    raise ValueError('unsupported structured adapter manifest schema')
  manifest_identity = manifest_payload.get('structured_decoder_identity')
  if not isinstance(manifest_identity, Mapping):
    raise ValueError(f'{manifest_path} has no structured decoder identity')
  return load_and_validate_adapter_manifest(
    manifest_path,
    adapter_path,
    expected_identity=dict(manifest_identity),
    expected_adapter_sha256=expected_adapter_sha256,
    expected_manifest_sha256=expected_manifest_sha256,
  )


def _job_record(
    *,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    marker: Mapping[str, Any],
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
  marker_path = Path(job['artifact_dir']).expanduser().resolve() / SUCCESS_MARKER
  execution_sha256 = _job_execution_digest(job)
  if marker.get('job_execution_sha256') != execution_sha256:
    raise ValueError(f'{job["job_id"]} marker execution digest drifted')
  return {
    'job_id': job['job_id'],
    'job_spec_sha256': _lower_hex(
      plan['job_spec_sha256'][job['job_id']], 64,
      context=f'{job["job_id"]} job spec SHA256'),
    'job_execution_sha256': execution_sha256,
    'originating_plan_id': _lower_hex(
      marker['originating_plan_id'], 64,
      context=f'{job["job_id"]} originating plan ID'),
    'success_marker_path': str(marker_path),
    'success_marker_sha256': sha256_file(marker_path),
    'outputs': dict(outputs),
  }


def _validate_job_identity(
    job: Mapping[str, Any],
    *,
    expected_job_id: str,
    kind: str,
    arm: str,
    suite: str,
    candidate_k: int,
    train_seed: int,
) -> None:
  if job.get('job_id') != expected_job_id or job.get('kind') != kind:
    raise ValueError(f'compiled plan lacks canonical {kind} job {expected_job_id}')
  suites = job.get('suites')
  if not isinstance(suites, list) or suite not in suites:
    raise ValueError(f'{expected_job_id} does not belong to suite {suite!r}')
  identity = job.get('identity')
  if not isinstance(identity, Mapping):
    raise ValueError(f'{expected_job_id} has no identity mapping')
  expected = {
    'control': arm,
    'train_seed': train_seed,
    'candidate_k': candidate_k,
  }
  observed = {field: identity.get(field) for field in expected}
  if observed != expected:
    raise ValueError(
      f'{expected_job_id} identity differs from the requested adapter pair')
  if kind == 'train':
    _positive_int(identity.get('updates'), context=f'{expected_job_id}.updates')


def _arm_snapshot(
    *,
    plan: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    legacy: bool,
    suite: str,
    candidate_k: int,
    train_seed: int,
    arm: str,
) -> dict[str, Any]:
  suffix = f's{train_seed:03d}--k{candidate_k:03d}'
  train_id = f'train--{arm}--{suffix}'
  export_id = f'export--{arm}--{suffix}'
  if train_id not in jobs or export_id not in jobs:
    raise ValueError(f'compiled plan lacks the canonical {arm} adapter pair')
  train_job = jobs[train_id]
  export_job = jobs[export_id]
  _validate_job_identity(
    train_job, expected_job_id=train_id, kind='train', arm=arm,
    suite=suite, candidate_k=candidate_k, train_seed=train_seed)
  _validate_job_identity(
    export_job, expected_job_id=export_id, kind='export', arm=arm,
    suite=suite, candidate_k=candidate_k, train_seed=train_seed)
  if export_job.get('dependencies') != [train_id]:
    raise ValueError(f'{export_id} must depend only on {train_id}')

  train_marker = _validated_analysis_marker(
    train_job, legacy=legacy, required=True)
  export_marker = _validated_analysis_marker(
    export_job, legacy=legacy, required=True)
  assert train_marker is not None and export_marker is not None
  train_outputs = _output_index(
    train_marker, expected_names=TRAIN_OUTPUTS, context=train_id)
  export_outputs = _output_index(
    export_marker, expected_names=EXPORT_OUTPUTS, context=export_id)

  checkpoint = train_outputs['checkpoint']
  adapter = export_outputs['adapter']
  adapter_manifest = export_outputs['adapter_manifest']
  manifest_path = Path(adapter_manifest['path'])
  manifest_payload = _strict_json_bytes(
    manifest_path.read_bytes(), source=manifest_path)
  validated_manifest = _load_and_validate_adapter_origin_manifest(
    manifest_path=manifest_path,
    adapter_path=Path(adapter['path']),
    manifest_payload=manifest_payload,
    legacy_plan=legacy,
    expected_adapter_sha256=adapter['sha256'],
    expected_manifest_sha256=adapter_manifest['sha256'],
  )

  export_identity = export_job['identity']
  identity_sha256: str | None = None
  if validated_manifest['schema_version'] == 4:
    expected_modes = (
      ('dynamic', 'dynamic')
      if arm == 'dynamic_dynamic' else ('fixed', 'fixed'))
    expected_topology_weight = 0.1 if arm == 'dynamic_dynamic' else 0.0
    expected_manifest_identity = {
      'control_identity': arm,
      'topology_mode': expected_modes[0],
      'factor_mode': expected_modes[1],
      'candidate_top_k': candidate_k,
      'independent_mode': False,
      'topology_weight': expected_topology_weight,
    }
    job_export_identity = {
      'control_identity': arm,
      'topology_mode': export_identity.get('topology_mode'),
      'factor_mode': export_identity.get('factor_mode'),
      'candidate_top_k': candidate_k,
      'independent_mode': export_identity.get('independent_mode'),
      'topology_weight': _finite_float(
        export_identity.get('topology_weight'),
        context=f'{export_id}.topology_weight'),
    }
    if job_export_identity != expected_manifest_identity:
      raise ValueError(f'{export_id} has a non-canonical control identity')
    observed_manifest_identity = {
      field: validated_manifest['structured_decoder_identity'].get(field)
      for field in expected_manifest_identity
    }
    if observed_manifest_identity != expected_manifest_identity:
      raise ValueError(
        f'{arm} adapter manifest identity differs from its export job')
    identity_sha256 = _lower_hex(
      validated_manifest['structured_decoder_identity_sha256'], 64,
      context=f'{arm} structured decoder identity SHA256')

  if validated_manifest['source_checkpoint_sha256'] != checkpoint['sha256']:
    raise ValueError(
      f'{arm} adapter manifest source checkpoint differs from the train marker')
  if validated_manifest['source_checkpoint_size_bytes'] != \
      checkpoint['size_bytes']:
    raise ValueError(
      f'{arm} adapter manifest checkpoint size differs from the train marker')
  if validated_manifest['adapter_sha256'] != adapter['sha256']:
    raise ValueError(f'{arm} adapter manifest differs from the export marker')
  if validated_manifest['adapter_size_bytes'] != adapter['size_bytes']:
    raise ValueError(
      f'{arm} adapter manifest size differs from the export marker')

  updates = train_job['identity']['updates']
  global_step = validated_manifest['source_checkpoint_global_step']
  if global_step != updates:
    raise ValueError(
      f'{arm} adapter source checkpoint step differs from training updates')
  if _flag_value(
      export_marker['argv'], '--checkpoint', context=export_id) != \
      checkpoint['path']:
    raise ValueError(
      f'{arm} export marker did not consume the committed train checkpoint')
  if _flag_value(
      export_marker['argv'], '--expected-checkpoint-sha256',
      context=export_id) != checkpoint['sha256']:
    raise ValueError(
      f'{arm} export marker did not pin the committed checkpoint digest')
  if _flag_value(
      export_marker['argv'], '--expected-global-step',
      context=export_id) != str(global_step):
    raise ValueError(f'{arm} export marker global-step commitment differs')
  if _flag_value(export_marker['argv'], '--output', context=export_id) != \
      adapter['path']:
    raise ValueError(f'{arm} export marker output path differs')
  if _flag_value(export_marker['argv'], '--manifest', context=export_id) != \
      adapter_manifest['path']:
    raise ValueError(f'{arm} export marker manifest path differs')

  released_backbone = validated_manifest['released_backbone']
  if released_backbone != RELEASED_BACKBONE_IDENTITY:
    raise ValueError(f'{arm} adapter does not bind the pinned released backbone')

  return {
    'train': _job_record(
      plan=plan, job=train_job, marker=train_marker, outputs=train_outputs),
    'export': _job_record(
      plan=plan, job=export_job, marker=export_marker,
      outputs=export_outputs),
    'adapter_origin': {
      'structured_decoder_identity_sha256': identity_sha256,
      'source_checkpoint_sha256': checkpoint['sha256'],
      'source_checkpoint_global_step': global_step,
      'released_backbone': dict(released_backbone),
    },
  }


def _source_snapshot(
    plan_dir: Path,
    *,
    suite: str,
    candidate_k: int,
    train_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
  plan_dir = plan_dir.expanduser().resolve()
  candidate_k = _positive_int(candidate_k, context='candidate_k')
  train_seed = _positive_int(train_seed, context='train_seed')
  if not isinstance(suite, str) or not suite:
    raise ValueError('suite must be a non-empty string')
  # This is a provenance replay, not execution of the historical plan.  The
  # plan's clean source SHA remains committed below, while the generation
  # runner separately commits and checks its current clean builder SHA.
  plan, jobs, plan_sha256, legacy = _load_plan_for_analysis(
    plan_dir, require_current_repository_match=False)
  selected_suites = plan.get('selected_suites')
  if not isinstance(selected_suites, list) or suite not in selected_suites:
    raise ValueError(f'compiled plan does not select suite {suite!r}')
  repository = plan.get('repository')
  if (not isinstance(repository, Mapping)
      or repository.get('dirty') is not False):
    raise ValueError('compiled plan must bind a clean source repository')
  repository_sha = _lower_hex(
    repository.get('sha'), 40, context='compiled plan repository SHA')
  source = {
    'compiled_plan_path': str(plan_dir / 'compiled-plan.json'),
    'compiled_plan_sha256': _lower_hex(
      plan_sha256, 64, context='compiled plan SHA256'),
    'plan_id': _lower_hex(plan.get('plan_id'), 64, context='plan ID'),
    'protocol_id': plan.get('protocol_id'),
    'source_manifest_sha256': _lower_hex(
      plan.get('source_manifest_sha256'), 64,
      context='source manifest SHA256'),
    'repository': {'sha': repository_sha, 'clean': True},
    'suite': suite,
    'candidate_k': candidate_k,
    'train_seed': train_seed,
    'legacy_plan_schema': legacy,
  }
  if not isinstance(source['protocol_id'], str) or not source['protocol_id']:
    raise ValueError('compiled plan protocol_id must be non-empty')
  arms = {
    arm: _arm_snapshot(
      plan=plan, jobs=jobs, legacy=legacy, suite=suite,
      candidate_k=candidate_k, train_seed=train_seed, arm=arm)
    for arm in ARMS
  }
  if (arms['dynamic_dynamic']['adapter_origin']['released_backbone']
      != arms['static_static']['adapter_origin']['released_backbone']):
    raise ValueError('paired adapters bind different released backbones')
  return source, arms


def build_adapter_pair_origin_evidence(
    plan_dir: Path,
    *,
    suite: str,
    candidate_k: int,
    train_seed: int,
    created_utc: str | None = None,
) -> dict[str, Any]:
  """Build evidence from live, fully revalidated train/export artifacts."""
  source, arms = _source_snapshot(
    plan_dir, suite=suite, candidate_k=candidate_k, train_seed=train_seed)
  payload: dict[str, Any] = {
    'schema_version': SCHEMA_VERSION,
    'artifact': ARTIFACT_ROLE,
    'created_utc': _utc_timestamp(created_utc),
    'source': source,
    'arms': arms,
  }
  payload['evidence_sha256'] = _canonical_sha256(payload)
  return payload


def validate_adapter_pair_origin_evidence(
    payload: object,
    *,
    expected_plan_sha256: str | None = None,
    expected_suite: str | None = None,
    expected_candidate_k: int | None = None,
    expected_train_seed: int | None = None,
) -> dict[str, Any]:
  """Validate evidence and replay every plan, marker, and artifact check."""
  if not isinstance(payload, Mapping):
    raise TypeError('adapter-pair origin evidence must be a JSON object')
  result = dict(payload)
  if set(result) != TOP_LEVEL_FIELDS:
    raise ValueError(
      'adapter-pair origin evidence schema mismatch: '
      f'missing={sorted(TOP_LEVEL_FIELDS - set(result))}, '
      f'unknown={sorted(set(result) - TOP_LEVEL_FIELDS)}')
  if (result['schema_version'] != SCHEMA_VERSION
      or result['artifact'] != ARTIFACT_ROLE):
    raise ValueError('unsupported adapter-pair origin evidence identity')
  _utc_timestamp(result['created_utc'])
  claimed_digest = _lower_hex(
    result['evidence_sha256'], 64, context='evidence SHA256')
  body = {key: value for key, value in result.items()
          if key != 'evidence_sha256'}
  if claimed_digest != _canonical_sha256(body):
    raise ValueError('adapter-pair origin evidence self-hash mismatch')
  source = result['source']
  if not isinstance(source, Mapping) or set(source) != SOURCE_FIELDS:
    raise ValueError('adapter-pair origin source schema mismatch')
  if not isinstance(result['arms'], Mapping) \
      or set(result['arms']) != set(ARMS):
    raise ValueError('adapter-pair origin evidence must contain both arms')
  plan_path = Path(source['compiled_plan_path']).expanduser().resolve()
  if plan_path.name != 'compiled-plan.json':
    raise ValueError('source compiled_plan_path must name compiled-plan.json')
  observed_source, observed_arms = _source_snapshot(
    plan_path.parent,
    suite=source['suite'],
    candidate_k=source['candidate_k'],
    train_seed=source['train_seed'],
  )
  if source != observed_source or result['arms'] != observed_arms:
    raise ValueError('adapter-pair origin evidence differs from live sources')
  if expected_plan_sha256 is not None:
    expected_plan_sha256 = _lower_hex(
      expected_plan_sha256, 64, context='expected compiled plan SHA256')
    if source['compiled_plan_sha256'] != expected_plan_sha256:
      raise ValueError('adapter-pair evidence names an unexpected plan')
  if expected_suite is not None and source['suite'] != expected_suite:
    raise ValueError('adapter-pair evidence names an unexpected suite')
  if (expected_candidate_k is not None
      and source['candidate_k'] != expected_candidate_k):
    raise ValueError('adapter-pair evidence names an unexpected candidate K')
  if (expected_train_seed is not None
      and source['train_seed'] != expected_train_seed):
    raise ValueError('adapter-pair evidence names an unexpected train seed')
  return result


def load_and_validate_adapter_pair_origin_evidence(
    path: Path,
    *,
    expected_evidence_sha256: str | None = None,
    expected_plan_sha256: str | None = None,
    expected_suite: str | None = None,
    expected_candidate_k: int | None = None,
    expected_train_seed: int | None = None,
) -> dict[str, Any]:
  """Read strict JSON, optionally pin file bytes, and validate live origin."""
  path = path.expanduser().resolve()
  payload_bytes = path.read_bytes()
  if expected_evidence_sha256 is not None:
    normalized_expected = (
      expected_evidence_sha256.lower()
      if isinstance(expected_evidence_sha256, str)
      else expected_evidence_sha256)
    expected_evidence_sha256 = _lower_hex(
      normalized_expected, 64,
      context='expected evidence file SHA256')
    actual = hashlib.sha256(payload_bytes).hexdigest()
    if actual != expected_evidence_sha256:
      raise ValueError(
        f'adapter-pair evidence file SHA256 mismatch: expected '
        f'{expected_evidence_sha256}, found {actual}')
  payload = _strict_json_bytes(payload_bytes, source=path)
  return validate_adapter_pair_origin_evidence(
    payload,
    expected_plan_sha256=expected_plan_sha256,
    expected_suite=expected_suite,
    expected_candidate_k=expected_candidate_k,
    expected_train_seed=expected_train_seed,
  )


def bind_generation_arm_to_adapter_origin_evidence(
    evidence_path: Path,
    *,
    expected_evidence_sha256: str,
    arm: str,
    adapter_path: Path,
    expected_adapter_sha256: str,
    adapter_manifest_path: Path,
    expected_adapter_manifest_sha256: str,
    structured_decoder_identity: Mapping[str, Any],
) -> dict[str, Any]:
  """Bind one exact schema-v4 plan export to replayed pair evidence.

  This is the runner-facing integration boundary.  In particular, it does not
  accept a semantically equivalent re-export: both adapter and manifest bytes
  must equal the export outputs committed by the selected plan arm.
  """
  if arm not in ARMS:
    raise ValueError(f'unknown adapter-origin arm: {arm!r}')
  if not isinstance(structured_decoder_identity, Mapping):
    raise TypeError('structured_decoder_identity must be a mapping')
  semantic_identity = dict(structured_decoder_identity)
  evidence_path = evidence_path.expanduser().resolve()
  normalized_evidence_sha256 = _lower_hex(
    (expected_evidence_sha256.lower()
     if isinstance(expected_evidence_sha256, str)
     else expected_evidence_sha256),
    64,
    context='expected evidence file SHA256',
  )
  evidence = load_and_validate_adapter_pair_origin_evidence(
    evidence_path,
    expected_evidence_sha256=normalized_evidence_sha256,
  )

  adapter_path = adapter_path.expanduser().resolve()
  adapter_manifest_path = adapter_manifest_path.expanduser().resolve()
  normalized_adapter_sha256 = _lower_hex(
    (expected_adapter_sha256.lower()
     if isinstance(expected_adapter_sha256, str)
     else expected_adapter_sha256),
    64,
    context='expected generation adapter SHA256',
  )
  normalized_manifest_sha256 = _lower_hex(
    (expected_adapter_manifest_sha256.lower()
     if isinstance(expected_adapter_manifest_sha256, str)
     else expected_adapter_manifest_sha256),
    64,
    context='expected generation adapter manifest SHA256',
  )
  if not adapter_path.is_file():
    raise FileNotFoundError(adapter_path)
  if not adapter_manifest_path.is_file():
    raise FileNotFoundError(adapter_manifest_path)
  if sha256_file(adapter_path) != normalized_adapter_sha256:
    raise ValueError('generation adapter SHA256 differs from its CLI pin')
  if sha256_file(adapter_manifest_path) != normalized_manifest_sha256:
    raise ValueError(
      'generation adapter manifest SHA256 differs from its CLI pin')

  arm_evidence = evidence['arms'][arm]
  committed_adapter = arm_evidence['export']['outputs']['adapter']
  committed_manifest = arm_evidence['export']['outputs']['adapter_manifest']
  if normalized_adapter_sha256 != committed_adapter['sha256']:
    raise ValueError(
      'generation adapter is not the exact plan export committed by evidence')
  if normalized_manifest_sha256 != committed_manifest['sha256']:
    raise ValueError(
      'generation adapter manifest is not the exact plan export committed '
      'by evidence')

  raw_manifest = _strict_json_bytes(
    adapter_manifest_path.read_bytes(), source=adapter_manifest_path)
  if raw_manifest.get('schema_version') != 4:
    raise ValueError(
      'generation adapter origin binding requires a schema-v4 manifest')
  validated_manifest = load_and_validate_adapter_manifest(
    adapter_manifest_path,
    adapter_path,
    expected_identity=semantic_identity,
    expected_adapter_sha256=normalized_adapter_sha256,
    expected_manifest_sha256=normalized_manifest_sha256,
  )
  validated_identity = validated_manifest['structured_decoder_identity']
  if validated_identity != semantic_identity:
    raise ValueError(
      'loaded structured decoder identity differs from the adapter manifest')
  source = evidence['source']
  candidate_k = source['candidate_k']
  if (semantic_identity.get('control_identity') != arm
      or semantic_identity.get('candidate_top_k') != candidate_k):
    raise ValueError(
      'generation adapter semantic identity differs from evidence arm or K')
  identity_sha256 = _lower_hex(
    validated_manifest['structured_decoder_identity_sha256'], 64,
    context='generation structured decoder identity SHA256')
  if identity_sha256 != \
      arm_evidence['adapter_origin']['structured_decoder_identity_sha256']:
    raise ValueError(
      'generation adapter semantic identity differs from origin evidence')

  origin = arm_evidence['adapter_origin']
  for field in (
      'source_checkpoint_sha256', 'source_checkpoint_global_step',
      'released_backbone'):
    if validated_manifest[field] != origin[field]:
      raise ValueError(
        f'generation adapter {field} differs from origin evidence')

  train = arm_evidence['train']
  export = arm_evidence['export']
  binding: dict[str, Any] = {
    'schema_version': 1,
    'artifact': GENERATION_BINDING_ARTIFACT_ROLE,
    'evidence_file': {
      'path': str(evidence_path),
      'sha256': normalized_evidence_sha256,
      'evidence_sha256': evidence['evidence_sha256'],
    },
    'source': dict(source),
    'arm': arm,
    'adapter': {
      'path': str(adapter_path),
      'sha256': normalized_adapter_sha256,
      'manifest_path': str(adapter_manifest_path),
      'manifest_sha256': normalized_manifest_sha256,
      'structured_decoder_identity': semantic_identity,
      'structured_decoder_identity_sha256': identity_sha256,
      'source_checkpoint_sha256': origin['source_checkpoint_sha256'],
      'source_checkpoint_global_step': origin[
        'source_checkpoint_global_step'],
      'released_backbone': dict(origin['released_backbone']),
    },
    'plan_export': {
      'train_job_id': train['job_id'],
      'train_job_spec_sha256': train['job_spec_sha256'],
      'train_job_execution_sha256': train['job_execution_sha256'],
      'train_success_marker_sha256': train['success_marker_sha256'],
      'checkpoint_sha256': train['outputs']['checkpoint']['sha256'],
      'training_data_provenance_sha256': train['outputs'][
        'training_data_provenance']['sha256'],
      'training_validation_data_provenance_sha256': train['outputs'][
        'training_validation_data_provenance']['sha256'],
      'export_job_id': export['job_id'],
      'export_job_spec_sha256': export['job_spec_sha256'],
      'export_job_execution_sha256': export['job_execution_sha256'],
      'export_success_marker_sha256': export['success_marker_sha256'],
      'adapter_sha256': committed_adapter['sha256'],
      'adapter_manifest_sha256': committed_manifest['sha256'],
    },
  }
  binding['binding_sha256'] = _canonical_sha256(binding)
  return binding


def write_adapter_pair_origin_evidence(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
  """Write a new evidence file without replacing an existing commitment."""
  path = path.expanduser().resolve()
  if path.exists():
    raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  serialized = json.dumps(
    dict(payload), indent=2, sort_keys=True, allow_nan=False) + '\n'
  temporary = path.with_name(f'.{path.name}.tmp')
  if temporary.exists():
    raise FileExistsError(temporary)
  try:
    temporary.write_text(serialized)
    temporary.replace(path)
  finally:
    if temporary.exists():
      temporary.unlink()
  return sha256_file(path)
