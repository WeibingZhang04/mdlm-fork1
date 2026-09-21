#!/usr/bin/env python3
"""Apply the frozen pilot promotion policy to a hierarchical analysis.

The evaluator is deliberately fail closed. It accepts only the exact pilot,
analysis schema, compiled plan, condition grid, bootstrap configuration, and
bounded diagnostics committed by the separate promotion-policy artifact. A
valid analysis can independently route to the K=64 confirmation suite, the
diagnostic K=128 pilot, both, or neither.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
  sys.path.insert(0, str(_IMPORT_ROOT))

from scripts.compile_experiment_matrix import (  # noqa: E402
  DEFAULT_MANIFEST,
  REPO_ROOT,
  _canonical_json,
  load_and_validate_manifest,
  sha256_file,
)


DEFAULT_POLICY = (
  REPO_ROOT / 'configs/experiment'
  / 'contextual-forest-expansion-v1-promotion-policy.yaml')
STRICT_UTC_TIMESTAMP = re.compile(
  r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
  r'(?:\.\d{1,6})?(?:Z|\+00:00)$')


def canonical_sha256(value: object) -> str:
  """Hash a JSON value independently of whitespace and key order."""
  return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _exact_keys(
    value: object,
    expected: Iterable[str],
    *,
    context: str,
) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise TypeError(f'{context} must be a mapping')
  expected_set = set(expected)
  actual = set(value)
  missing = sorted(expected_set - actual)
  unknown = sorted(actual - expected_set)
  if missing or unknown:
    raise ValueError(
      f'{context} schema mismatch: missing={missing}, unknown={unknown}')
  return value


def _nonempty(value: object, *, context: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f'{context} must be a non-empty string')
  return value


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


def _finite(
    value: object,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
  if (not isinstance(value, (int, float)) or isinstance(value, bool)
      or not math.isfinite(float(value))):
    raise ValueError(f'{context} must be finite')
  result = float(value)
  if minimum is not None and result < minimum:
    raise ValueError(f'{context} must be >= {minimum}')
  if maximum is not None and result > maximum:
    raise ValueError(f'{context} must be <= {maximum}')
  return result


def _timestamp(value: object, *, context: str) -> dt.datetime:
  text = _nonempty(value, context=context)
  if not STRICT_UTC_TIMESTAMP.fullmatch(text):
    raise ValueError(
      f'{context} must be a complete ISO-8601 UTC timestamp ending in '
      "'Z' or '+00:00'")
  try:
    parsed = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
  except ValueError as error:
    raise ValueError(f'{context} must be an ISO-8601 UTC timestamp') from error
  if parsed.utcoffset() != dt.timedelta(0):
    raise ValueError(f'{context} must be UTC')
  return parsed


def _utc_now() -> str:
  return dt.datetime.now(dt.timezone.utc).isoformat()


def _strict_bool(value: object, *, context: str) -> bool:
  if not isinstance(value, bool):
    raise TypeError(f'{context} must be a boolean')
  return value


def _read_mapping(path: Path, *, context: str) -> dict[str, Any]:
  path = path.expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  with path.open() as handle:
    payload = yaml.safe_load(handle)
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must contain a mapping')
  return dict(payload)


def load_and_validate_policy(
    policy_path: Path = DEFAULT_POLICY,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  """Load the exact version-2 policy and bind it to the pilot manifest."""
  policy_path = policy_path.expanduser().resolve()
  manifest_path = manifest_path.expanduser().resolve()
  policy = _exact_keys(_read_mapping(policy_path, context='policy'), {
    'schema_version', 'artifact', 'policy_id', 'policy_status',
    'frozen_utc', 'protocol_id', 'source_manifest_sha256', 'source_suite',
    'analysis_contract', 'confirmation_gates',
    'k64_support_sufficiency', 'routing',
  }, context='policy')
  expected_identity = {
    'schema_version': 2,
    'artifact': 'experiment_promotion_policy',
    'policy_status': 'frozen_before_pilot_results',
  }
  for field, expected in expected_identity.items():
    if policy[field] != expected:
      raise ValueError(f'policy.{field} must equal {expected!r}')
  _nonempty(policy['policy_id'], context='policy.policy_id')
  _timestamp(policy['frozen_utc'], context='policy.frozen_utc')
  manifest_sha = sha256_file(manifest_path)
  if policy['source_manifest_sha256'] != manifest_sha:
    raise ValueError(
      'policy source_manifest_sha256 differs from the protocol manifest')
  _lower_hex(
    policy['source_manifest_sha256'], 64,
    context='policy.source_manifest_sha256')
  manifest = load_and_validate_manifest(manifest_path, repo_root=repo_root)
  if policy['protocol_id'] != manifest['protocol_id']:
    raise ValueError('policy protocol_id differs from the manifest')
  source_suite = _nonempty(
    policy['source_suite'], context='policy.source_suite')
  if source_suite not in manifest['suites']:
    raise ValueError(f'policy source suite is unknown: {source_suite}')
  suite = manifest['suites'][source_suite]

  contract = _exact_keys(policy['analysis_contract'], {
    'analysis_schema_version', 'analysis_artifact', 'objective',
    'comparison', 'baseline_arm', 'treatment_arm', 'source_plan_id',
    'source_compiled_plan_sha256', 'source_repository_sha',
    'source_repository_clean',
    'candidate_k', 'train_seeds', 'corruption_seeds', 'datasets',
    'mask_rates', 'expected_num_strata', 'bootstrap',
  }, context='policy.analysis_contract')
  if contract['analysis_schema_version'] != 2:
    raise ValueError('analysis_contract.analysis_schema_version must equal 2')
  if contract['analysis_artifact'] != \
      'hierarchical_conditional_denoising_analysis':
    raise ValueError('analysis_contract.analysis_artifact is unsupported')
  if contract['objective'] != \
      'conditional_denoising_nll_per_masked_token':
    raise ValueError('analysis_contract.objective is unsupported')
  _lower_hex(
    contract['source_plan_id'], 64,
    context='analysis_contract.source_plan_id')
  _lower_hex(
    contract['source_compiled_plan_sha256'], 64,
    context='analysis_contract.source_compiled_plan_sha256')
  _lower_hex(
    contract['source_repository_sha'], 40,
    context='analysis_contract.source_repository_sha')
  if contract['source_repository_clean'] is not True:
    raise ValueError(
      'analysis_contract.source_repository_clean must equal true')
  candidate_k = _positive_int(
    contract['candidate_k'], context='analysis_contract.candidate_k')
  if suite['candidate_ks'] != [candidate_k]:
    raise ValueError('policy candidate_k differs from the source suite')
  if contract['train_seeds'] != suite['train_seeds']:
    raise ValueError('policy train_seeds differ from the source suite')
  if contract['corruption_seeds'] != suite['corruption_seeds']:
    raise ValueError('policy corruption_seeds differ from the source suite')
  if contract['mask_rates'] != suite['mask_rates']:
    raise ValueError('policy mask_rates differ from the source suite')
  comparison_name = _nonempty(
    contract['comparison'], context='analysis_contract.comparison')
  comparison = manifest['evaluation']['comparisons'].get(comparison_name)
  if comparison is None:
    raise ValueError(f'policy comparison is unknown: {comparison_name}')
  if contract['baseline_arm'] != comparison['baseline'] \
      or contract['treatment_arm'] != comparison['treatment']:
    raise ValueError('policy arms differ from the manifest comparison')
  if not {contract['baseline_arm'], contract['treatment_arm']}.issubset(
      set(suite['controls'])):
    raise ValueError('policy arms are not both in the source suite')

  datasets = contract['datasets']
  if not isinstance(datasets, Mapping) or not datasets:
    raise ValueError('analysis_contract.datasets must be a non-empty mapping')
  manifest_datasets = {}
  for dataset_alias in suite['datasets']:
    data_config_name = manifest['datasets'][dataset_alias]['data_config']
    data_path = repo_root / 'configs/data' / f'{data_config_name}.yaml'
    data_config = _read_mapping(
      data_path, context=f'data config {data_config_name}')
    manifest_datasets[data_config['valid']] = {
      'revision': data_config['valid_revision']}
  if dict(datasets) != manifest_datasets:
    raise ValueError(
      'policy datasets/revisions differ from the pinned source suite')
  for dataset, raw_spec in datasets.items():
    _nonempty(dataset, context='analysis_contract dataset')
    spec = _exact_keys(
      raw_spec, {'revision'}, context=f'analysis_contract.datasets.{dataset}')
    _lower_hex(
      spec['revision'], 40,
      context=f'analysis_contract.datasets.{dataset}.revision')
  expected_strata = len(datasets) * len(contract['mask_rates'])
  if contract['expected_num_strata'] != expected_strata:
    raise ValueError(
      'analysis_contract.expected_num_strata does not match its grid')

  bootstrap = _exact_keys(contract['bootstrap'], {
    'method', 'num_resamples', 'rng_seed', 'confidence_level',
  }, context='analysis_contract.bootstrap')
  if bootstrap['method'] != 'hierarchical_paired_percentile_bootstrap':
    raise ValueError('unsupported promotion bootstrap method')
  if bootstrap['num_resamples'] != manifest['analysis']['bootstrap_resamples']:
    raise ValueError('policy bootstrap resamples differ from the manifest')
  if bootstrap['rng_seed'] != \
      manifest['analysis']['bootstrap_seed'] + candidate_k:
    raise ValueError('policy bootstrap RNG seed differs from the aggregate')
  if float(bootstrap['confidence_level']) != \
      float(manifest['analysis']['confidence_level']):
    raise ValueError('policy confidence level differs from the manifest')

  confirmation = _exact_keys(policy['confirmation_gates'], {
    'pooled', 'corpus_breadth', 'high_mask_robustness',
  }, context='policy.confirmation_gates')
  pooled = _exact_keys(confirmation['pooled'], {
    'min_mean_improvement', 'ci_lower_strictly_above',
  }, context='confirmation_gates.pooled')
  _finite(
    pooled['min_mean_improvement'],
    context='confirmation_gates.pooled.min_mean_improvement')
  _finite(
    pooled['ci_lower_strictly_above'],
    context='confirmation_gates.pooled.ci_lower_strictly_above')
  breadth = _exact_keys(confirmation['corpus_breadth'], {
    'min_positive_dataset_means',
  }, context='confirmation_gates.corpus_breadth')
  positive_datasets = _positive_int(
    breadth['min_positive_dataset_means'],
    context='corpus_breadth.min_positive_dataset_means')
  if positive_datasets > len(datasets):
    raise ValueError('corpus breadth threshold exceeds dataset count')
  high_mask = _exact_keys(confirmation['high_mask_robustness'], {
    'mask_rates', 'min_pooled_mean_improvement',
    'min_positive_dataset_means',
    'min_worst_dataset_mean_improvement',
  }, context='confirmation_gates.high_mask_robustness')
  if (not isinstance(high_mask['mask_rates'], list)
      or not high_mask['mask_rates']
      or len(set(high_mask['mask_rates'])) != len(high_mask['mask_rates'])
      or not set(high_mask['mask_rates']).issubset(
        set(contract['mask_rates']))):
    raise ValueError('high-mask rates must be a unique subset of mask_rates')
  for field in (
      'min_pooled_mean_improvement',
      'min_worst_dataset_mean_improvement'):
    _finite(high_mask[field], context=f'high_mask_robustness.{field}')
  high_positive = _positive_int(
    high_mask['min_positive_dataset_means'],
    context='high_mask_robustness.min_positive_dataset_means')
  if high_positive > len(datasets):
    raise ValueError('high-mask breadth threshold exceeds dataset count')

  support = _exact_keys(policy['k64_support_sufficiency'], {
    'arm', 'aggregation', 'min_overall_candidate_recall',
    'min_high_mask_dataset_candidate_recall',
    'min_overall_retained_unary_mass',
    'min_high_mask_dataset_retained_unary_mass',
  }, context='policy.k64_support_sufficiency')
  if support['arm'] != contract['treatment_arm']:
    raise ValueError('support sufficiency must use the treatment arm')
  if support['aggregation'] != 'equal_weight_analysis_strata':
    raise ValueError('unsupported support aggregation')
  for field in (
      'min_overall_candidate_recall',
      'min_high_mask_dataset_candidate_recall',
      'min_overall_retained_unary_mass',
      'min_high_mask_dataset_retained_unary_mass'):
    _finite(
      support[field], context=f'k64_support_sufficiency.{field}',
      minimum=0.0, maximum=1.0)

  routing = _exact_keys(policy['routing'], {
    'allow_parallel_routes', 'confirmation', 'k128',
  }, context='policy.routing')
  if not _strict_bool(
      routing['allow_parallel_routes'],
      context='routing.allow_parallel_routes'):
    raise ValueError('this policy requires independent parallel routes')
  expected_routes = {
    'confirmation': {
      'target_suite': 'confirmation',
      'requires': [
        'integrity_complete', 'pooled_mean', 'pooled_ci',
        'corpus_breadth', 'high_mask_pooled', 'high_mask_breadth',
        'high_mask_no_material_regression',
      ],
    },
    'k128': {
      'target_suite': 'candidate_k_128_pilot',
      'requires': ['integrity_complete', 'k64_support_limited'],
    },
  }
  for route_name, expected in expected_routes.items():
    route = _exact_keys(
      routing[route_name], {'target_suite', 'requires'},
      context=f'policy.routing.{route_name}')
    if dict(route) != expected:
      raise ValueError(
        f'policy.routing.{route_name} differs from the supported route')
    target = expected['target_suite']
    if manifest['suites'][target]['promotion_from'] != source_suite:
      raise ValueError(
        f'routing target {target} is not promoted from {source_suite}')
  return dict(policy)


def _validate_source_plan(
    policy: Mapping[str, Any],
    source_plan: object,
    *,
    source_plan_sha256: str,
) -> dict[str, Any]:
  """Validate the exact completed legacy pilot plan attested by the policy.

  The already-running pilot used the version-1 plan schema, whose plan ID did
  not include repository metadata.  We preserve that work only by binding the
  complete compiled-plan file hash and independently checking its clean source
  repository.  New plans use the revision-bound version-2 identity.
  """
  contract = policy['analysis_contract']
  source_plan_sha256 = _lower_hex(
    source_plan_sha256, 64, context='source compiled plan SHA256')
  if source_plan_sha256 != contract['source_compiled_plan_sha256']:
    raise ValueError('source compiled plan file SHA256 differs from policy')
  plan = _exact_keys(source_plan, {
    'schema_version', 'protocol_id', 'source_manifest_sha256',
    'artifact_root', 'selected_suites', 'promotion_evidence', 'plan_id',
    'manifest_protocol_status', 'scientific_scope', 'repository',
    'job_counts', 'num_jobs', 'job_ids', 'job_spec_sha256',
  }, context='source compiled plan')
  expected = {
    'schema_version': 1,
    'protocol_id': policy['protocol_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'selected_suites': [policy['source_suite']],
    'promotion_evidence': {},
    'plan_id': contract['source_plan_id'],
  }
  for field, value in expected.items():
    if plan[field] != value:
      raise ValueError(
        f'source compiled plan {field}={plan[field]!r}; expected {value!r}')
  repository = _exact_keys(
    plan['repository'], {'sha', 'dirty'},
    context='source compiled plan repository')
  if repository != {
      'sha': contract['source_repository_sha'],
      'dirty': contract['source_repository_clean'] is not True}:
    raise ValueError('source compiled plan repository differs from policy')
  job_ids = plan['job_ids']
  job_digests = plan['job_spec_sha256']
  if (not isinstance(job_ids, list) or not job_ids
      or len(job_ids) != len(set(job_ids))):
    raise ValueError('source compiled plan has invalid job IDs')
  if (not isinstance(job_digests, Mapping)
      or set(job_digests) != set(job_ids)):
    raise ValueError('source compiled plan has invalid job commitments')
  for job_id, digest in job_digests.items():
    _nonempty(job_id, context='source compiled plan job ID')
    _lower_hex(
      digest, 64, context=f'source compiled plan job digest {job_id}')
  if plan['num_jobs'] != len(job_ids):
    raise ValueError('source compiled plan num_jobs is inconsistent')
  return dict(plan)


def _validate_analysis_source_integrity(
    policy: Mapping[str, Any],
    payload: object,
) -> dict[str, Any]:
  """Validate the canonical marker/output commitment carried by analysis v2."""
  integrity = _exact_keys(payload, {
    'schema_version', 'source_compiled_plan_path',
    'source_compiled_plan_sha256', 'source_plan_id',
    'source_manifest_sha256', 'source_repository_sha',
    'source_repository_clean', 'validated_job_ids', 'jobs',
    'commitment_sha256',
  }, context='analysis.source_integrity')
  if integrity['schema_version'] != 1:
    raise ValueError('analysis.source_integrity.schema_version must equal 1')
  contract = policy['analysis_contract']
  expected = {
    'source_compiled_plan_sha256': contract['source_compiled_plan_sha256'],
    'source_plan_id': contract['source_plan_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'source_repository_sha': contract['source_repository_sha'],
    'source_repository_clean': True,
  }
  for field, value in expected.items():
    if integrity[field] != value:
      raise ValueError(
        f'analysis.source_integrity.{field} differs from the policy')
  source_plan_path = Path(_nonempty(
    integrity['source_compiled_plan_path'],
    context='analysis.source_integrity.source_compiled_plan_path'))
  if not source_plan_path.is_absolute() or source_plan_path.name != \
      'compiled-plan.json':
    raise ValueError(
      'analysis source compiled-plan path must be an absolute '
      'compiled-plan.json path')
  job_ids = integrity['validated_job_ids']
  jobs = integrity['jobs']
  if (not isinstance(job_ids, list) or not job_ids
      or job_ids != sorted(job_ids) or len(job_ids) != len(set(job_ids))):
    raise ValueError(
      'analysis.source_integrity.validated_job_ids must be sorted and unique')
  if not isinstance(jobs, Mapping) or set(jobs) != set(job_ids):
    raise ValueError('analysis.source_integrity.jobs differs from job IDs')
  for job_id in job_ids:
    _nonempty(job_id, context='analysis source-integrity job ID')
    job = _exact_keys(jobs[job_id], {
      'job_spec_sha256', 'job_execution_sha256', 'success_marker_path',
      'success_marker_sha256', 'outputs', 'scientific_output_sha256',
    }, context=f'analysis.source_integrity.jobs.{job_id}')
    for field in (
        'job_spec_sha256', 'job_execution_sha256',
        'success_marker_sha256'):
      _lower_hex(
        job[field], 64,
        context=f'analysis.source_integrity.jobs.{job_id}.{field}')
    marker_path = Path(_nonempty(
      job['success_marker_path'],
      context=(
        f'analysis.source_integrity.jobs.{job_id}.success_marker_path')))
    if not marker_path.is_absolute() or marker_path.name != '_job_success.json':
      raise ValueError(
        f'analysis source-integrity marker path is invalid for {job_id}')
    outputs = job['outputs']
    if not isinstance(outputs, list) or not outputs:
      raise ValueError(f'analysis source-integrity job {job_id} has no outputs')
    output_hashes = {}
    for index, raw_output in enumerate(outputs):
      output = _exact_keys(raw_output, {
        'name', 'relative_path', 'size_bytes', 'sha256',
      }, context=(
        f'analysis.source_integrity.jobs.{job_id}.outputs[{index}]'))
      name = _nonempty(
        output['name'], context='analysis source-integrity output name')
      if name in output_hashes:
        raise ValueError(
          f'analysis source-integrity job {job_id} repeats output {name}')
      relative = Path(_nonempty(
        output['relative_path'],
        context=f'analysis source-integrity output {name} path'))
      if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(
          f'analysis source-integrity output {name} has an unsafe path')
      _positive_int(
        output['size_bytes'],
        context=f'analysis source-integrity output {name} size')
      output_hashes[name] = _lower_hex(
        output['sha256'], 64,
        context=f'analysis source-integrity output {name} SHA256')
    scientific = job['scientific_output_sha256']
    if not isinstance(scientific, Mapping):
      raise TypeError(
        f'analysis source-integrity scientific outputs for {job_id} '
        'must be a mapping')
    supported_scientific = {
      'conditional_record_manifest', 'conditional_records',
      'dataset_provenance', 'pairing_digest'}
    if not set(scientific).issubset(supported_scientific):
      raise ValueError(
        f'analysis source-integrity job {job_id} has unknown scientific '
        'outputs')
    for name, digest in scientific.items():
      if output_hashes.get(name) != digest:
        raise ValueError(
          f'analysis source-integrity scientific output {job_id}/{name} '
          'differs from its output commitment')
  commitment = _lower_hex(
    integrity['commitment_sha256'], 64,
    context='analysis.source_integrity.commitment_sha256')
  body = dict(integrity)
  body.pop('commitment_sha256')
  if canonical_sha256(body) != commitment:
    raise ValueError('analysis source-integrity commitment SHA256 mismatch')
  return dict(integrity)


def _validate_analysis_contract(
    policy: Mapping[str, Any],
    analysis: object,
) -> tuple[dict[tuple[str, float], dict[str, Any]],
           dict[tuple[str, float], dict[str, Any]]]:
  analysis = _exact_keys(analysis, {
    'schema_version', 'artifact', 'created_utc', 'protocol_id', 'suite',
    'comparison', 'arms', 'objective', 'scope_note', 'by_candidate_k',
    'diagnostics', 'compiled_plan', 'source_integrity',
  }, context='analysis')
  contract = policy['analysis_contract']
  expected_identity = {
    'schema_version': contract['analysis_schema_version'],
    'artifact': contract['analysis_artifact'],
    'protocol_id': policy['protocol_id'],
    'suite': policy['source_suite'],
    'comparison': contract['comparison'],
    'objective': contract['objective'],
  }
  for field, expected in expected_identity.items():
    if analysis[field] != expected:
      raise ValueError(
        f'analysis.{field}={analysis[field]!r}; expected {expected!r}')
  if not isinstance(analysis['scope_note'], str) \
      or 'no diffusion ELBO' not in analysis['scope_note']:
    raise ValueError('analysis scope note does not preserve objective scope')
  analysis_created = _timestamp(
    analysis['created_utc'], context='analysis.created_utc')
  if analysis_created < _timestamp(
      policy['frozen_utc'], context='policy.frozen_utc'):
    raise ValueError('analysis predates the frozen promotion policy')
  arms = _exact_keys(
    analysis['arms'], {'baseline', 'treatment'}, context='analysis.arms')
  if dict(arms) != {
      'baseline': contract['baseline_arm'],
      'treatment': contract['treatment_arm']}:
    raise ValueError('analysis arms differ from the policy')
  compiled = _exact_keys(analysis['compiled_plan'], {
    'plan_id', 'source_manifest_sha256', 'source_compiled_plan_sha256',
    'source_repository_sha', 'source_repository_clean',
    'job_artifact_commitment_sha256',
  }, context='analysis.compiled_plan')
  if compiled['plan_id'] != contract['source_plan_id']:
    raise ValueError('analysis compiled plan ID differs from the policy')
  if compiled['source_manifest_sha256'] != policy['source_manifest_sha256']:
    raise ValueError('analysis compiled manifest hash differs from the policy')
  if compiled['source_compiled_plan_sha256'] != \
      contract['source_compiled_plan_sha256']:
    raise ValueError('analysis compiled-plan file hash differs from the policy')
  if compiled['source_repository_sha'] != contract['source_repository_sha'] \
      or compiled['source_repository_clean'] is not True:
    raise ValueError('analysis source repository differs from the policy')
  _lower_hex(
    compiled['job_artifact_commitment_sha256'], 64,
    context='analysis.compiled_plan.job_artifact_commitment_sha256')
  integrity = _validate_analysis_source_integrity(
    policy, analysis['source_integrity'])
  if compiled['job_artifact_commitment_sha256'] != \
      integrity['commitment_sha256']:
    raise ValueError(
      'analysis compiled-plan artifact commitment differs from '
      'source_integrity')

  k_text = str(contract['candidate_k'])
  by_k = _exact_keys(
    analysis['by_candidate_k'], {k_text},
    context='analysis.by_candidate_k')
  bootstrap = _exact_keys(by_k[k_text], {
    'method', 'improvement_definition', 'nesting', 'num_train_seeds',
    'num_strata', 'num_resamples', 'rng', 'rng_seed', 'confidence_level',
    'pooled', 'conditions',
  }, context=f'analysis.by_candidate_k.{k_text}')
  expected_bootstrap = contract['bootstrap']
  expected_values = {
    'method': expected_bootstrap['method'],
    'improvement_definition': 'baseline conditional NLL minus treatment',
    'nesting': [
      'average corruption replications within source document',
      'resample training seeds with replacement',
      'resample source documents within sampled training seed',
      'equal-weight frozen dataset x mask-rate strata',
    ],
    'num_train_seeds': len(contract['train_seeds']),
    'num_strata': contract['expected_num_strata'],
    'num_resamples': expected_bootstrap['num_resamples'],
    'rng': 'NumPy Generator(PCG64)',
    'rng_seed': expected_bootstrap['rng_seed'],
    'confidence_level': expected_bootstrap['confidence_level'],
  }
  for field, expected in expected_values.items():
    if bootstrap[field] != expected:
      raise ValueError(
        f'analysis bootstrap {field}={bootstrap[field]!r}; '
        f'expected {expected!r}')
  pooled = _exact_keys(
    bootstrap['pooled'], {'mean_improvement', 'ci_lower', 'ci_upper'},
    context='analysis bootstrap pooled')
  pooled_values = {
    field: _finite(value, context=f'analysis.pooled.{field}')
    for field, value in pooled.items()}
  if pooled_values['ci_lower'] > pooled_values['ci_upper']:
    raise ValueError('analysis pooled confidence interval is reversed')

  expected_pairs = {
    (dataset, float(mask_rate))
    for dataset in contract['datasets']
    for mask_rate in contract['mask_rates']}
  raw_conditions = bootstrap['conditions']
  if not isinstance(raw_conditions, Mapping):
    raise TypeError('analysis conditions must be a mapping')
  conditions = {}
  expected_condition_keys = set()
  for dataset, mask_rate in sorted(expected_pairs):
    key = (
      f'{dataset}|mask={mask_rate:.6f}|k={contract["candidate_k"]}')
    expected_condition_keys.add(key)
    if key not in raw_conditions:
      raise ValueError(f'analysis is missing condition {key}')
    row = _exact_keys(raw_conditions[key], {
      'dataset', 'dataset_revision', 'mask_rate', 'candidate_k',
      'mean_improvement', 'ci_lower', 'ci_upper',
    }, context=f'analysis condition {key}')
    expected_revision = contract['datasets'][dataset]['revision']
    identity = {
      'dataset': dataset,
      'dataset_revision': expected_revision,
      'mask_rate': mask_rate,
      'candidate_k': contract['candidate_k'],
    }
    for field, expected in identity.items():
      observed = row[field]
      if isinstance(expected, float):
        matches = isinstance(observed, (int, float)) \
          and math.isclose(float(observed), expected, abs_tol=1e-12)
      else:
        matches = observed == expected
      if not matches:
        raise ValueError(
          f'analysis condition {key} has wrong {field}: {observed!r}')
    values = {
      field: _finite(row[field], context=f'analysis condition {key}.{field}')
      for field in ('mean_improvement', 'ci_lower', 'ci_upper')}
    if values['ci_lower'] > values['ci_upper']:
      raise ValueError(f'analysis condition {key} interval is reversed')
    conditions[(dataset, mask_rate)] = {**dict(row), **values}
  if set(raw_conditions) != expected_condition_keys:
    raise ValueError('analysis condition grid has unknown or duplicate cells')
  condition_pooled_mean = math.fsum(
    row['mean_improvement'] for row in conditions.values()) / len(conditions)
  if not math.isclose(
      pooled_values['mean_improvement'], condition_pooled_mean,
      rel_tol=0.0, abs_tol=1e-12):
    raise ValueError(
      'analysis pooled mean is inconsistent with equal-weight conditions')

  raw_diagnostics = analysis['diagnostics']
  if not isinstance(raw_diagnostics, Mapping):
    raise TypeError('analysis diagnostics must be a mapping')
  diagnostics = {}
  expected_diagnostic_keys = set()
  for dataset, mask_rate in sorted(expected_pairs):
    revision = contract['datasets'][dataset]['revision']
    key = str((dataset, revision, mask_rate, contract['candidate_k']))
    expected_diagnostic_keys.add(key)
    if key not in raw_diagnostics:
      raise ValueError(f'analysis is missing diagnostics for {key}')
    row = _exact_keys(raw_diagnostics[key], {
      'num_documents', 'num_train_seeds', 'num_corruption_seeds', 'arms',
    }, context=f'analysis diagnostics {key}')
    _positive_int(
      row['num_documents'], context=f'analysis diagnostics {key}.num_documents')
    if row['num_train_seeds'] != len(contract['train_seeds']):
      raise ValueError(f'analysis diagnostics {key} train-seed mismatch')
    if row['num_corruption_seeds'] != len(contract['corruption_seeds']):
      raise ValueError(f'analysis diagnostics {key} corruption-seed mismatch')
    arms = _exact_keys(
      row['arms'], {contract['baseline_arm'], contract['treatment_arm']},
      context=f'analysis diagnostics {key}.arms')
    validated_arms = {}
    for arm, raw_metrics in arms.items():
      metrics = _exact_keys(raw_metrics, {
        'conditional_nll_per_masked_token', 'candidate_recall',
        'retained_unary_mass',
      }, context=f'analysis diagnostics {key}.arms.{arm}')
      validated_arms[arm] = {
        'conditional_nll_per_masked_token': _finite(
          metrics['conditional_nll_per_masked_token'],
          context=f'analysis diagnostics {key}.{arm}.nll', minimum=0.0),
        'candidate_recall': _finite(
          metrics['candidate_recall'],
          context=f'analysis diagnostics {key}.{arm}.candidate_recall',
          minimum=0.0, maximum=1.0),
        'retained_unary_mass': _finite(
          metrics['retained_unary_mass'],
          context=f'analysis diagnostics {key}.{arm}.retained_unary_mass',
          minimum=0.0, maximum=1.0),
      }
    diagnostics[(dataset, mask_rate)] = {
      **dict(row), 'arms': validated_arms}
  if set(raw_diagnostics) != expected_diagnostic_keys:
    raise ValueError('analysis diagnostic grid has unknown or duplicate cells')
  return conditions, diagnostics


def _verify_authoritative_analysis(
    policy: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    source_plan: Mapping[str, Any],
    source_plan_path: Path,
    source_plan_sha256: str,
    manifest_path: Path,
) -> dict[str, Any]:
  """Reload source artifacts from the current checkout and recompute."""
  source_plan_path = source_plan_path.expanduser().resolve()
  if source_plan_path.name != 'compiled-plan.json' \
      or not source_plan_path.is_file():
    raise FileNotFoundError(
      f'exact source compiled-plan.json is unavailable: {source_plan_path}')
  if sha256_file(source_plan_path) != source_plan_sha256:
    raise ValueError('source compiled-plan file changed before verification')
  if _read_mapping(source_plan_path, context='source compiled plan') != \
      dict(source_plan):
    raise ValueError('source compiled-plan mapping differs from supplied plan')
  integrity = analysis['source_integrity']
  if Path(integrity['source_compiled_plan_path']).expanduser().resolve() != \
      source_plan_path:
    raise ValueError(
      'analysis source-integrity compiled-plan path differs from supplied plan')

  # Import lazily to keep the policy module usable as a standalone validator
  # while ensuring production routing always passes through the hardened raw-
  # artifact loader and exact bootstrap implementation.
  from scripts.aggregate_hierarchical_document_eval import (  # pylint: disable=import-outside-toplevel
    aggregate_records,
    bind_analysis_to_source,
    load_plan_records,
  )
  contract = policy['analysis_contract']
  records, context = load_plan_records(
    source_plan_path.parent,
    manifest_path=manifest_path,
    suite_name=policy['source_suite'],
    comparison_name=contract['comparison'],
    expected_legacy_plan_sha256=contract['source_compiled_plan_sha256'],
    expected_legacy_repository_sha=contract['source_repository_sha'])
  comparison = context['comparison']
  analysis_cfg = context['manifest']['analysis']
  recomputed = aggregate_records(
    records,
    baseline_arm=comparison['baseline'],
    treatment_arm=comparison['treatment'],
    protocol_id=policy['protocol_id'],
    suite_name=policy['source_suite'],
    comparison_name=contract['comparison'],
    num_resamples=contract['bootstrap']['num_resamples'],
    rng_seed=analysis_cfg['bootstrap_seed'],
    confidence_level=contract['bootstrap']['confidence_level'],
    timestamp_utc=analysis['created_utc'])
  recomputed = bind_analysis_to_source(recomputed, context)
  if recomputed != dict(analysis):
    raise ValueError(
      'analysis differs from deterministic recomputation of the '
      'marker-bound source records')
  return context['source_integrity']


def evaluate_pilot_analysis(
    policy: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    policy_sha256: str,
    analysis_sha256: str,
    source_plan: Mapping[str, Any],
    source_plan_path: Path,
    source_plan_sha256: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    created_utc: str | None = None,
) -> dict[str, Any]:
  """Evaluate a validated pilot aggregate and return a routing decision."""
  _lower_hex(policy_sha256, 64, context='policy_sha256')
  _lower_hex(analysis_sha256, 64, context='analysis_sha256')
  validated_source_plan = _validate_source_plan(
    policy, source_plan, source_plan_sha256=source_plan_sha256)
  conditions, diagnostics = _validate_analysis_contract(policy, analysis)
  verified_integrity = _verify_authoritative_analysis(
    policy,
    analysis,
    source_plan=source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=source_plan_sha256,
    manifest_path=manifest_path)
  decision_time = created_utc or _utc_now()
  parsed_decision_time = _timestamp(decision_time, context='created_utc')
  if parsed_decision_time < _timestamp(
      analysis['created_utc'], context='analysis.created_utc'):
    raise ValueError('promotion decision predates the analysis')

  contract = policy['analysis_contract']
  confirmation = policy['confirmation_gates']
  high_rates = {
    float(value)
    for value in confirmation['high_mask_robustness']['mask_rates']}
  datasets = list(contract['datasets'])
  rates = [float(value) for value in contract['mask_rates']]
  dataset_means = {
    dataset: math.fsum(
      conditions[(dataset, rate)]['mean_improvement'] for rate in rates)
    / len(rates)
    for dataset in datasets}
  high_mask_dataset_means = {
    dataset: math.fsum(
      conditions[(dataset, rate)]['mean_improvement']
      for rate in high_rates) / len(high_rates)
    for dataset in datasets}
  high_mask_pooled = math.fsum(
    high_mask_dataset_means.values()) / len(high_mask_dataset_means)
  positive_dataset_count = sum(
    value > 0.0 for value in dataset_means.values())
  positive_high_mask_dataset_count = sum(
    value > 0.0 for value in high_mask_dataset_means.values())
  worst_high_mask_dataset = min(high_mask_dataset_means.values())
  pooled = analysis['by_candidate_k'][str(contract['candidate_k'])]['pooled']

  pooled_cfg = confirmation['pooled']
  breadth_cfg = confirmation['corpus_breadth']
  high_cfg = confirmation['high_mask_robustness']
  gate_states = {
    'integrity_complete': True,
    'pooled_mean': (
      float(pooled['mean_improvement'])
      >= float(pooled_cfg['min_mean_improvement'])),
    'pooled_ci': (
      float(pooled['ci_lower'])
      > float(pooled_cfg['ci_lower_strictly_above'])),
    'corpus_breadth': (
      positive_dataset_count >= breadth_cfg['min_positive_dataset_means']),
    'high_mask_pooled': (
      high_mask_pooled >= high_cfg['min_pooled_mean_improvement']),
    'high_mask_breadth': (
      positive_high_mask_dataset_count
      >= high_cfg['min_positive_dataset_means']),
    'high_mask_no_material_regression': (
      worst_high_mask_dataset
      >= high_cfg['min_worst_dataset_mean_improvement']),
  }

  treatment = policy['k64_support_sufficiency']['arm']
  support_cfg = policy['k64_support_sufficiency']
  overall_candidate_recall = math.fsum(
    diagnostics[pair]['arms'][treatment]['candidate_recall']
    for pair in diagnostics) / len(diagnostics)
  overall_retained_mass = math.fsum(
    diagnostics[pair]['arms'][treatment]['retained_unary_mass']
    for pair in diagnostics) / len(diagnostics)
  high_candidate_by_dataset = {
    dataset: math.fsum(
      diagnostics[(dataset, rate)]['arms'][treatment]['candidate_recall']
      for rate in high_rates) / len(high_rates)
    for dataset in datasets}
  high_retained_by_dataset = {
    dataset: math.fsum(
      diagnostics[(dataset, rate)]['arms'][treatment]['retained_unary_mass']
      for rate in high_rates) / len(high_rates)
    for dataset in datasets}
  min_high_candidate = min(high_candidate_by_dataset.values())
  min_high_retained = min(high_retained_by_dataset.values())
  support_thresholds = {
    'overall_candidate_recall': (
      overall_candidate_recall
      >= support_cfg['min_overall_candidate_recall']),
    'high_mask_dataset_candidate_recall': (
      min_high_candidate
      >= support_cfg['min_high_mask_dataset_candidate_recall']),
    'overall_retained_unary_mass': (
      overall_retained_mass
      >= support_cfg['min_overall_retained_unary_mass']),
    'high_mask_dataset_retained_unary_mass': (
      min_high_retained
      >= support_cfg['min_high_mask_dataset_retained_unary_mass']),
  }
  support_sufficient = all(support_thresholds.values())
  gate_states['k64_support_sufficient'] = support_sufficient
  gate_states['k64_support_limited'] = not support_sufficient

  routes = {}
  for route_name in ('confirmation', 'k128'):
    route_cfg = policy['routing'][route_name]
    criteria = {
      name: gate_states[name] for name in route_cfg['requires']}
    routes[route_name] = {
      'target_suite': route_cfg['target_suite'],
      'promote': all(criteria.values()),
      'criteria': criteria,
    }
  confirmation_route = routes['confirmation']['promote']
  k128_route = routes['k128']['promote']
  if confirmation_route and k128_route:
    outcome = 'promote_confirmation_and_k128'
  elif confirmation_route:
    outcome = 'promote_confirmation_only'
  elif k128_route:
    outcome = 'promote_k128_only'
  else:
    outcome = 'stop_after_pilot'

  gates = {
    'pooled_mean': {
      'passed': gate_states['pooled_mean'],
      'observed': float(pooled['mean_improvement']),
      'operator': '>=',
      'threshold': float(pooled_cfg['min_mean_improvement']),
    },
    'pooled_ci': {
      'passed': gate_states['pooled_ci'],
      'observed': float(pooled['ci_lower']),
      'operator': '>',
      'threshold': float(pooled_cfg['ci_lower_strictly_above']),
    },
    'corpus_breadth': {
      'passed': gate_states['corpus_breadth'],
      'observed': positive_dataset_count,
      'operator': '>=',
      'threshold': breadth_cfg['min_positive_dataset_means'],
    },
    'high_mask_pooled': {
      'passed': gate_states['high_mask_pooled'],
      'observed': high_mask_pooled,
      'operator': '>=',
      'threshold': float(high_cfg['min_pooled_mean_improvement']),
    },
    'high_mask_breadth': {
      'passed': gate_states['high_mask_breadth'],
      'observed': positive_high_mask_dataset_count,
      'operator': '>=',
      'threshold': high_cfg['min_positive_dataset_means'],
    },
    'high_mask_no_material_regression': {
      'passed': gate_states['high_mask_no_material_regression'],
      'observed': worst_high_mask_dataset,
      'operator': '>=',
      'threshold': float(high_cfg['min_worst_dataset_mean_improvement']),
    },
    'k64_support_sufficient': {
      'passed': support_sufficient,
      'criteria': support_thresholds,
    },
    'k64_support_limited': {
      'passed': not support_sufficient,
      'definition': 'logical negation of k64_support_sufficient',
    },
  }
  return {
    'schema_version': 2,
    'artifact': 'experiment_promotion_routing_decision',
    'created_utc': decision_time,
    'protocol_id': policy['protocol_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'source_suite': policy['source_suite'],
    'policy': {
      'policy_id': policy['policy_id'],
      'policy_sha256': policy_sha256,
      'policy_status': policy['policy_status'],
      'frozen_utc': policy['frozen_utc'],
    },
    'analysis': {
      'analysis_sha256': analysis_sha256,
      'created_utc': analysis['created_utc'],
      'plan_id': analysis['compiled_plan']['plan_id'],
    },
    'compiled_plan': {
      'file_sha256': source_plan_sha256,
      'plan_id': validated_source_plan['plan_id'],
      'source_repository_sha': validated_source_plan['repository']['sha'],
      'source_repository_clean': (
        validated_source_plan['repository']['dirty'] is False),
      'job_spec_commitment_sha256': canonical_sha256(
        validated_source_plan['job_spec_sha256']),
      'job_artifact_commitment_sha256': verified_integrity[
        'commitment_sha256'],
    },
    'integrity': {
      'passed': True,
      'criteria': {
        'identity_and_manifest_bound': True,
        'compiled_plan_bound': True,
        'success_markers_and_outputs_bound': True,
        'analysis_recomputed_from_bound_records': True,
        'complete_factorial_grid': True,
        'bootstrap_contract_exact': True,
        'diagnostics_complete_finite_and_bounded': True,
      },
    },
    'measurements': {
      'pooled': {
        'mean_improvement': float(pooled['mean_improvement']),
        'ci_lower': float(pooled['ci_lower']),
        'ci_upper': float(pooled['ci_upper']),
      },
      'dataset_mean_improvements': dataset_means,
      'high_mask_dataset_mean_improvements': high_mask_dataset_means,
      'high_mask_pooled_mean_improvement': high_mask_pooled,
      'k64_support': {
        'overall_candidate_recall': overall_candidate_recall,
        'minimum_high_mask_dataset_candidate_recall': min_high_candidate,
        'overall_retained_unary_mass': overall_retained_mass,
        'minimum_high_mask_dataset_retained_unary_mass': min_high_retained,
        'high_mask_candidate_recall_by_dataset': high_candidate_by_dataset,
        'high_mask_retained_unary_mass_by_dataset': high_retained_by_dataset,
      },
    },
    'gates': gates,
    'routes': routes,
    'outcome': outcome,
    'compiler_evidence': {
      route_name: {
        'eligible': route['promote'],
        'target_suite': route['target_suite'],
        'filename': f'{route["target_suite"]}-promotion.json',
      }
      for route_name, route in routes.items()
    },
  }


def build_compiler_evidence(
    decision: Mapping[str, Any],
    route_name: str,
    *,
    analysis_path: Path,
    source_plan_path: Path,
    decision_path: Path,
) -> dict[str, Any]:
  """Translate one true route into revision-bound version-2 evidence."""
  if (decision.get('schema_version') != 2
      or decision.get('artifact') != 'experiment_promotion_routing_decision'):
    raise ValueError('invalid routing decision artifact')
  routes = decision.get('routes')
  if not isinstance(routes, Mapping) or route_name not in routes:
    raise ValueError(f'unknown promotion route: {route_name}')
  route = routes[route_name]
  if not isinstance(route, Mapping) or route.get('promote') is not True:
    raise ValueError(f'promotion route {route_name} is not eligible')
  route_criteria = route.get('criteria')
  if (not isinstance(route_criteria, Mapping) or not route_criteria
      or any(value is not True for value in route_criteria.values())):
    raise ValueError(f'promotion route {route_name} is not all true')
  policy_info = decision.get('policy')
  analysis_info = decision.get('analysis')
  if not isinstance(policy_info, Mapping) \
      or not isinstance(analysis_info, Mapping):
    raise ValueError('routing decision lacks policy/analysis commitments')
  policy_sha = _lower_hex(
    policy_info.get('policy_sha256'), 64,
    context='routing decision policy_sha256')
  analysis_sha = _lower_hex(
    analysis_info.get('analysis_sha256'), 64,
    context='routing decision analysis_sha256')
  compiled_plan = _exact_keys(decision.get('compiled_plan'), {
    'file_sha256', 'plan_id', 'source_repository_sha',
    'source_repository_clean', 'job_spec_commitment_sha256',
    'job_artifact_commitment_sha256',
  }, context='routing decision compiled_plan')
  source_plan_sha = _lower_hex(
    compiled_plan['file_sha256'], 64,
    context='routing decision source compiled plan SHA256')
  source_repo_sha = _lower_hex(
    compiled_plan['source_repository_sha'], 40,
    context='routing decision source repository SHA')
  if compiled_plan['source_repository_clean'] is not True:
    raise ValueError('routing decision source repository is not clean')
  plan_id = _lower_hex(
    compiled_plan['plan_id'], 64, context='routing decision plan ID')
  job_commitment = _lower_hex(
    compiled_plan['job_spec_commitment_sha256'], 64,
    context='routing decision job-spec commitment')
  artifact_commitment = _lower_hex(
    compiled_plan['job_artifact_commitment_sha256'], 64,
    context='routing decision job-artifact commitment')
  decision_sha = canonical_sha256(decision)
  resolved_paths = {
    'analysis_path': str(analysis_path.expanduser().resolve()),
    'source_compiled_plan_path': str(
      source_plan_path.expanduser().resolve()),
    'routing_decision_path': str(decision_path.expanduser().resolve()),
  }
  for field, value in resolved_paths.items():
    if not Path(value).is_absolute():
      raise ValueError(f'{field} must resolve to an absolute path')
  return {
    'schema_version': 2,
    'artifact': 'experiment_suite_promotion_decision',
    'protocol_id': decision['protocol_id'],
    'source_manifest_sha256': decision['source_manifest_sha256'],
    'source_suite': decision['source_suite'],
    'promoted_suite': route['target_suite'],
    'route_name': route_name,
    'decision': 'promote',
    'criteria': dict(route_criteria),
    'commitments': {
      'policy_sha256': policy_sha,
      'analysis_sha256': analysis_sha,
      'source_compiled_plan_sha256': source_plan_sha,
      'source_plan_id': plan_id,
      'source_repository_sha': source_repo_sha,
      'source_repository_clean': True,
      'source_job_spec_commitment_sha256': job_commitment,
      'source_job_artifact_commitment_sha256': artifact_commitment,
      'canonical_decision_sha256': decision_sha,
    },
    'artifacts': resolved_paths,
    'created_utc': decision['created_utc'],
  }


def _artifact_path(value: object, *, context: str) -> Path:
  text = _nonempty(value, context=context)
  path = Path(text).expanduser()
  if not path.is_absolute():
    raise ValueError(f'{context} must be absolute')
  path = path.resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  return path


def verify_compiler_evidence(
    evidence: object,
    *,
    evidence_path: Path,
    promoted_suite: str,
    manifest_path: Path,
    trusted_policy_path: Path = DEFAULT_POLICY,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  """Deterministically re-evaluate and verify promotion evidence.

  The compiler calls this verifier instead of trusting booleans supplied by an
  evidence JSON.  The exact policy is selected by the compiler's trusted
  registry; the evidence cannot choose a different policy.
  """
  evidence = _exact_keys(evidence, {
    'schema_version', 'artifact', 'protocol_id',
    'source_manifest_sha256', 'source_suite', 'promoted_suite',
    'route_name', 'decision', 'criteria', 'commitments', 'artifacts',
    'created_utc',
  }, context='promotion evidence')
  if evidence['schema_version'] != 2 \
      or evidence['artifact'] != 'experiment_suite_promotion_decision':
    raise ValueError('promotion evidence must use the supported v2 schema')
  if evidence['promoted_suite'] != promoted_suite:
    raise ValueError('promotion evidence targets a different suite')
  if evidence['decision'] != 'promote':
    raise ValueError('promotion evidence decision must equal promote')
  _timestamp(evidence['created_utc'], context='promotion evidence created_utc')

  commitments = _exact_keys(evidence['commitments'], {
    'policy_sha256', 'analysis_sha256', 'source_compiled_plan_sha256',
    'source_plan_id', 'source_repository_sha', 'source_repository_clean',
    'source_job_spec_commitment_sha256',
    'source_job_artifact_commitment_sha256', 'canonical_decision_sha256',
  }, context='promotion evidence commitments')
  for field in (
      'policy_sha256', 'analysis_sha256', 'source_compiled_plan_sha256',
      'source_plan_id', 'source_job_spec_commitment_sha256',
      'source_job_artifact_commitment_sha256',
      'canonical_decision_sha256'):
    _lower_hex(commitments[field], 64, context=f'commitments.{field}')
  _lower_hex(
    commitments['source_repository_sha'], 40,
    context='commitments.source_repository_sha')
  if commitments['source_repository_clean'] is not True:
    raise ValueError('promotion evidence source repository must be clean')

  artifacts = _exact_keys(evidence['artifacts'], {
    'analysis_path', 'source_compiled_plan_path', 'routing_decision_path',
  }, context='promotion evidence artifacts')
  analysis_path = _artifact_path(
    artifacts['analysis_path'], context='artifacts.analysis_path')
  source_plan_path = _artifact_path(
    artifacts['source_compiled_plan_path'],
    context='artifacts.source_compiled_plan_path')
  decision_path = _artifact_path(
    artifacts['routing_decision_path'],
    context='artifacts.routing_decision_path')
  if evidence_path.expanduser().resolve() in {
      analysis_path, source_plan_path, decision_path}:
    raise ValueError('promotion evidence cannot self-reference an artifact')

  trusted_policy_path = trusted_policy_path.expanduser().resolve()
  policy_sha = sha256_file(trusted_policy_path)
  if commitments['policy_sha256'] != policy_sha:
    raise ValueError('promotion evidence policy SHA differs from trusted policy')
  policy = load_and_validate_policy(
    trusted_policy_path,
    manifest_path=manifest_path,
    repo_root=repo_root)
  if evidence['protocol_id'] != policy['protocol_id'] \
      or evidence['source_manifest_sha256'] != \
      policy['source_manifest_sha256'] \
      or evidence['source_suite'] != policy['source_suite']:
    raise ValueError('promotion evidence protocol identity differs from policy')

  analysis_sha = sha256_file(analysis_path)
  source_plan_sha = sha256_file(source_plan_path)
  if commitments['analysis_sha256'] != analysis_sha:
    raise ValueError('promotion evidence analysis SHA256 mismatch')
  if commitments['source_compiled_plan_sha256'] != source_plan_sha:
    raise ValueError('promotion evidence source plan SHA256 mismatch')
  analysis = _read_mapping(analysis_path, context='analysis')
  source_plan = _read_mapping(source_plan_path, context='source compiled plan')
  decision = _read_mapping(decision_path, context='routing decision')
  if canonical_sha256(decision) != commitments['canonical_decision_sha256']:
    raise ValueError('promotion evidence canonical decision SHA256 mismatch')

  canonical_decision = evaluate_pilot_analysis(
    policy,
    analysis,
    policy_sha256=policy_sha,
    analysis_sha256=analysis_sha,
    source_plan=source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=source_plan_sha,
    manifest_path=manifest_path,
    created_utc=decision.get('created_utc'))
  if decision != canonical_decision:
    raise ValueError('routing decision differs from deterministic reevaluation')
  expected = build_compiler_evidence(
    canonical_decision,
    str(evidence['route_name']),
    analysis_path=analysis_path,
    source_plan_path=source_plan_path,
    decision_path=decision_path)
  if dict(evidence) != expected:
    raise ValueError(
      'promotion evidence differs from canonical reevaluated evidence')
  return expected


def _atomic_write_json(path: Path, value: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists():
    raise FileExistsError(path)
  temporary = path.with_name(f'.{path.name}.tmp')
  if temporary.exists():
    raise FileExistsError(temporary)
  temporary.write_text(json.dumps(
    value, indent=2, sort_keys=True, allow_nan=False) + '\n')
  temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--analysis', type=Path, required=True)
  parser.add_argument(
    '--source-plan', type=Path, required=True,
    help='exact compiled-plan.json used to produce the pilot analysis')
  parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
  parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument(
    '--compiler-evidence-dir', type=Path,
    help=(
      'write compiler-compatible evidence only for true promotion routes; '
      'existing files are never overwritten'))
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  policy_path = args.policy.expanduser().resolve()
  analysis_path = args.analysis.expanduser().resolve()
  source_plan_path = args.source_plan.expanduser().resolve()
  output = args.output.expanduser().resolve()
  if output.exists():
    raise FileExistsError(output)
  policy = load_and_validate_policy(
    policy_path, manifest_path=args.manifest)
  analysis = _read_mapping(analysis_path, context='analysis')
  source_plan = _read_mapping(
    source_plan_path, context='source compiled plan')
  decision = evaluate_pilot_analysis(
    policy,
    analysis,
    policy_sha256=sha256_file(policy_path),
    analysis_sha256=sha256_file(analysis_path),
    source_plan=source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=sha256_file(source_plan_path),
    manifest_path=args.manifest)

  evidence_payloads = {}
  evidence_paths = {}
  if args.compiler_evidence_dir is not None:
    evidence_dir = args.compiler_evidence_dir.expanduser().resolve()
    for route_name, route in decision['routes'].items():
      if route['promote']:
        path = evidence_dir / f'{route["target_suite"]}-promotion.json'
        if path.exists():
          raise FileExistsError(path)
        evidence_paths[route_name] = path
        evidence_payloads[route_name] = build_compiler_evidence(
          decision,
          route_name,
          analysis_path=analysis_path,
          source_plan_path=source_plan_path,
          decision_path=output)
  _atomic_write_json(output, decision)
  for route_name, payload in evidence_payloads.items():
    _atomic_write_json(evidence_paths[route_name], payload)
  print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
  return 0 if any(route['promote'] for route in decision['routes'].values()) \
    else 2


if __name__ == '__main__':
  raise SystemExit(main())
