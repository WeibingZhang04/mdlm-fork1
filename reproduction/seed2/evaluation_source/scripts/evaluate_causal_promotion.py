#!/usr/bin/env python3
"""Evaluate causal-smoke technical validity and emit primary-route evidence.

Promotion is based only on artifact integrity, factorial completeness,
pairing, finite/ordered statistics, algebraic identities within frozen
tolerances, support-grid validity, and topology non-degeneracy. No NLL sign or
effect-size threshold is consulted anywhere in this evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_causal_denoising_eval import (  # noqa: E402
  _contrasts,
  build_analysis,
)
from scripts.compile_experiment_matrix import (  # noqa: E402
  _git_metadata,
  build_jobs,
  load_and_validate_manifest,
  sha256_file,
)
from scripts.evaluate_experiment_promotion import (  # noqa: E402
  _artifact_path,
  _exact_keys,
  _lower_hex,
  _read_mapping as _read_yaml_mapping,
  _timestamp,
  _utc_now,
  _validate_analysis_source_integrity,
  build_compiler_evidence,
  canonical_sha256,
)
from scripts.finalize_candidate_k_policy import (  # noqa: E402
  _exclusive_write_json,
)
from scripts.finalize_causal_smoke_policy import (  # noqa: E402
  DEFAULT_MANIFEST,
  DEFAULT_TEMPLATE,
  _validate_template,
)
from scripts.run_compiled_job import SUCCESS_MARKER  # noqa: E402


def _read_mapping(path: Path, *, context: str) -> dict[str, Any]:
  """Read JSON with its native number grammar and YAML only for YAML files.

  PyYAML's YAML-1.1 resolver treats JSON exponent forms such as ``1e-10`` as
  strings.  Causal policies freeze scientific tolerances, so silently changing
  their types on a JSON round trip would make replay depend on serialization.
  """
  path = path.expanduser().resolve()
  if path.suffix.lower() not in {'.yaml', '.yml'}:
    if not path.is_file():
      raise FileNotFoundError(path)
    with path.open() as handle:
      payload = json.load(handle)
    if not isinstance(payload, Mapping):
      raise TypeError(f'{context} must contain a mapping')
    return dict(payload)
  return _read_yaml_mapping(path, context=context)


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


def _positive_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f'{context} must be a positive integer')
  return value


def _validate_source_plan(
    policy: Mapping[str, Any],
    source_plan: object,
    *,
    source_plan_path: Path,
    source_plan_sha256: str,
    manifest_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  plan = _exact_keys(source_plan, {
    'schema_version', 'protocol_id', 'source_manifest_sha256', 'repository',
    'artifact_root', 'selected_suites', 'promotion_evidence', 'plan_id',
    'manifest_protocol_status', 'scientific_scope', 'job_counts',
    'num_jobs', 'job_ids', 'job_spec_sha256',
  }, context='causal-smoke source compiled plan')
  contract = policy['analysis_contract']
  expected = {
    'schema_version': 2,
    'protocol_id': policy['protocol_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'selected_suites': [policy['source_suite']],
    'promotion_evidence': {},
    'plan_id': contract['source_plan_id'],
  }
  for field, value in expected.items():
    if plan[field] != value:
      raise ValueError(f'causal-smoke source plan {field} differs from policy')
  if source_plan_sha256 != contract['source_compiled_plan_sha256']:
    raise ValueError('causal-smoke source plan bytes differ from policy')
  repository = _exact_keys(
    plan['repository'], {'sha', 'dirty'}, context='source plan repository')
  if dict(repository) != {
      'sha': contract['source_repository_sha'], 'dirty': False}:
    raise ValueError('causal-smoke source repository differs from policy')
  checkout = _git_metadata(repo_root)
  if checkout != dict(repository):
    raise ValueError(
      'causal-smoke authoritative replay requires the exact clean source '
      f'repository checkout: expected {dict(repository)}, found {checkout}')
  identity = {
    'protocol_id': plan['protocol_id'],
    'source_manifest_sha256': plan['source_manifest_sha256'],
    'repository': dict(repository),
    'artifact_root': plan['artifact_root'],
    'selected_suites': plan['selected_suites'],
    'promotion_evidence': plan['promotion_evidence'],
  }
  if canonical_sha256(identity) != plan['plan_id']:
    raise ValueError('causal-smoke source plan ID is not canonical')
  source_info = policy['source_plan']
  if (source_plan_path.expanduser().resolve()
      != Path(source_info['path']).expanduser().resolve()
      or source_info['file_sha256'] != source_plan_sha256
      or source_info['plan_id'] != plan['plan_id']
      or source_info['source_repository_sha'] != repository['sha']
      or source_info['source_repository_clean'] is not True
      or source_info['promotion_evidence'] != {}
      or source_info['job_spec_commitment_sha256']
      != canonical_sha256(dict(plan['job_spec_sha256']))):
    raise ValueError('causal-smoke source_plan policy block differs from plan')

  manifest = load_and_validate_manifest(manifest_path, repo_root=repo_root)
  expected_jobs = build_jobs(
    manifest,
    selected_suites=[policy['source_suite']],
    artifact_root=Path(plan['artifact_root']).expanduser().resolve(),
    source_manifest_sha256=plan['source_manifest_sha256'],
    source_repository_sha=repository['sha'],
    plan_id=plan['plan_id'])
  expected_counts = {}
  for job in expected_jobs.values():
    expected_counts[job['kind']] = expected_counts.get(job['kind'], 0) + 1
  expected_digests = {
    job_id: canonical_sha256(job) for job_id, job in expected_jobs.items()}
  if (plan['job_ids'] != list(expected_jobs)
      or plan['job_spec_sha256'] != expected_digests
      or plan['num_jobs'] != len(expected_jobs)
      or plan['job_counts'] != dict(sorted(expected_counts.items()))
      or plan['manifest_protocol_status'] != manifest['protocol_status']
      or plan['scientific_scope'] != manifest['scientific_scope']):
    raise ValueError('causal-smoke source plan factorial commitments differ')
  plan_dir = source_plan_path.expanduser().resolve().parent
  for job_id, expected_job in expected_jobs.items():
    job = _read_mapping(
      plan_dir / 'jobs' / f'{job_id}.json', context=f'job {job_id}')
    if job != expected_job:
      raise ValueError(f'causal-smoke source job {job_id} differs')
  return dict(plan)


def load_and_validate_causal_policy(
    policy_path: Path,
    *,
    template_path: Path = DEFAULT_TEMPLATE,
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  policy_path = policy_path.expanduser().resolve()
  template_path = template_path.expanduser().resolve()
  manifest_path = manifest_path.expanduser().resolve()
  template = _validate_template(
    _read_mapping(template_path, context='causal-smoke policy template'),
    manifest_path=manifest_path,
    repo_root=repo_root)
  policy = _exact_keys(_read_mapping(policy_path, context='causal policy'), {
    'schema_version', 'artifact', 'policy_id', 'policy_status', 'template',
    'frozen_utc', 'protocol_id', 'source_manifest_sha256', 'source_suite',
    'analysis_contract', 'source_plan', 'freeze_attestation',
    'technical_gates', 'routing',
  }, context='causal-smoke policy')
  expected_identity = {
    'schema_version': 1,
    'artifact': 'causal_smoke_promotion_policy',
    'policy_id': template['policy_id'],
    'policy_status': 'frozen_before_source_suite_results',
    'protocol_id': template['protocol_id'],
    'source_manifest_sha256': template['source_manifest_sha256'],
    'source_suite': template['source_suite'],
  }
  for field, value in expected_identity.items():
    if policy[field] != value:
      raise ValueError(f'causal-smoke policy {field} differs from template')
  template_info = _exact_keys(policy['template'], {
    'path', 'sha256', 'frozen_utc',
  }, context='causal-smoke policy template identity')
  if (template_info['sha256'] != sha256_file(template_path)
      or template_info['frozen_utc'] != template['template_frozen_utc']):
    raise ValueError('causal-smoke policy does not bind trusted template')
  frozen = _timestamp(policy['frozen_utc'], context='policy frozen_utc')
  if frozen < _timestamp(
      template['template_frozen_utc'], context='template frozen_utc'):
    raise ValueError('causal-smoke concrete policy predates template')
  if (policy['technical_gates'] != template['technical_gates']
      or policy['routing'] != template['routing']):
    raise ValueError('causal-smoke technical gates differ from template')
  contract = _exact_keys(policy['analysis_contract'], {
    *template['analysis_contract'].keys(),
    'source_plan_id', 'source_compiled_plan_sha256',
    'source_repository_sha', 'source_repository_clean',
  }, context='causal-smoke policy analysis contract')
  for field, value in template['analysis_contract'].items():
    if contract[field] != value:
      raise ValueError(
        f'causal-smoke analysis contract {field} differs from template')
  _lower_hex(contract['source_plan_id'], 64, context='source plan ID')
  _lower_hex(
    contract['source_compiled_plan_sha256'], 64,
    context='source compiled plan SHA256')
  _lower_hex(
    contract['source_repository_sha'], 40, context='source repository SHA')
  if contract['source_repository_clean'] is not True:
    raise ValueError('causal-smoke source repository must be clean')
  source_info = _exact_keys(policy['source_plan'], {
    'path', 'file_sha256', 'plan_id', 'source_repository_sha',
    'source_repository_clean', 'promotion_evidence',
    'job_spec_commitment_sha256',
  }, context='causal-smoke policy source_plan')
  plan_path = Path(source_info['path']).expanduser().resolve()
  if not plan_path.is_file() or plan_path.name != 'compiled-plan.json':
    raise FileNotFoundError('causal-smoke policy source plan is unavailable')
  plan = _read_mapping(plan_path, context='causal-smoke source plan')
  validated_plan = _validate_source_plan(
    policy,
    plan,
    source_plan_path=plan_path,
    source_plan_sha256=sha256_file(plan_path),
    manifest_path=manifest_path,
    repo_root=repo_root)
  attestation = _exact_keys(policy['freeze_attestation'], {
    'status', 'num_jobs_checked', 'job_ids_sha256', 'artifact_dirs_sha256',
  }, context='causal-smoke freeze attestation')
  if (attestation['status']
      != 'no_source_suite_artifact_directory_was_nonempty'
      or attestation['num_jobs_checked'] != validated_plan['num_jobs']
      or attestation['job_ids_sha256']
      != canonical_sha256(validated_plan['job_ids'])):
    raise ValueError('causal-smoke freeze attestation differs from source plan')
  artifact_dirs = []
  for job_id in validated_plan['job_ids']:
    job = _read_mapping(
      plan_path.parent / 'jobs' / f'{job_id}.json', context=f'job {job_id}')
    artifact_dirs.append(str(Path(job['artifact_dir']).expanduser().resolve()))
  if attestation['artifact_dirs_sha256'] != \
      canonical_sha256(sorted(artifact_dirs)):
    raise ValueError('causal-smoke artifact-directory commitment differs')
  return dict(policy)


def _verify_temporal_order(
    policy: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    *,
    source_plan_path: Path,
    analysis_created_utc: str,
) -> None:
  frozen = _timestamp(policy['frozen_utc'], context='policy frozen_utc')
  analysis_created = _timestamp(
    analysis_created_utc, context='analysis created_utc')
  latest_end = frozen
  for job_id in source_plan['job_ids']:
    job = _read_mapping(
      source_plan_path.parent / 'jobs' / f'{job_id}.json',
      context=f'job {job_id}')
    marker = _read_mapping(
      Path(job['artifact_dir']).expanduser().resolve() / SUCCESS_MARKER,
      context=f'success marker {job_id}')
    started = _timestamp(
      marker.get('start_time_utc'), context=f'{job_id} start_time_utc')
    ended = _timestamp(
      marker.get('end_time_utc'), context=f'{job_id} end_time_utc')
    if started <= frozen:
      raise ValueError(
        f'causal-smoke policy was not frozen before job {job_id} started')
    if ended < started:
      raise ValueError(f'causal-smoke job {job_id} ends before it starts')
    latest_end = max(latest_end, ended)
  if analysis_created < latest_end:
    raise ValueError('causal-smoke analysis predates source job completion')


def _validate_bootstrap_block(
    payload: object,
    *,
    context: str,
    contract: Mapping[str, Any],
    rng_seed: int,
    value_bounds: tuple[float, float] | None = None,
) -> None:
  block = _exact_keys(payload, {
    'method', 'estimand', 'nesting', 'top_level_resampling_unit',
    'num_adapter_seeds', 'num_strata', 'num_resamples', 'rng', 'rng_seed',
    'confidence_level', 'pooled', 'conditions',
  }, context=context)
  bootstrap = contract['bootstrap']
  expected = {
    'method': bootstrap['method'],
    'nesting': [
      'average corruption replications within source document',
      'resample adapter training seeds with replacement',
      'resample source documents within sampled adapter seed',
      'equal-weight frozen dataset x mask-rate strata',
    ],
    'top_level_resampling_unit': 'adapter_training_seed',
    'num_adapter_seeds': len(contract['train_seeds']),
    'num_strata': contract['expected_num_strata'],
    'num_resamples': bootstrap['num_resamples'],
    'rng': 'NumPy Generator(PCG64)',
    'rng_seed': rng_seed,
    'confidence_level': bootstrap['confidence_level'],
  }
  for field, expected_value in expected.items():
    if block[field] != expected_value:
      raise ValueError(f'{context}.{field} differs from policy')
  if not isinstance(block['estimand'], str) or not block['estimand']:
    raise ValueError(f'{context}.estimand must be non-empty')
  expected_cells = {
    (dataset, spec['revision'], float(rate), contract['candidate_k'])
    for dataset, spec in contract['datasets'].items()
    for rate in contract['mask_rates']}
  condition_cells = set()
  for key, raw in block['conditions'].items():
    row = _exact_keys(raw, {
      'dataset', 'dataset_revision', 'mask_rate', 'adapter_candidate_k',
      'estimate', 'ci_lower', 'ci_upper',
    }, context=f'{context}.conditions.{key}')
    cell = (
      row['dataset'], row['dataset_revision'], float(row['mask_rate']),
      row['adapter_candidate_k'])
    if cell in condition_cells:
      raise ValueError(f'{context} repeats condition {cell}')
    condition_cells.add(cell)
    lower_bound, upper_bound = (
      value_bounds if value_bounds is not None else (None, None))
    estimate = _finite(
      row['estimate'], context=f'{context}.{key}.estimate',
      minimum=lower_bound, maximum=upper_bound)
    lower = _finite(
      row['ci_lower'], context=f'{context}.{key}.ci_lower',
      minimum=lower_bound, maximum=upper_bound)
    upper = _finite(
      row['ci_upper'], context=f'{context}.{key}.ci_upper',
      minimum=lower_bound, maximum=upper_bound)
    del estimate
    if lower > upper:
      raise ValueError(f'{context}.{key} confidence interval is reversed')
  if (condition_cells != expected_cells
      or len(block['conditions']) != len(expected_cells)):
    raise ValueError(f'{context} condition grid is incomplete')
  pooled = _exact_keys(
    block['pooled'], {'estimate', 'ci_lower', 'ci_upper'},
    context=f'{context}.pooled')
  lower_bound, upper_bound = (
    value_bounds if value_bounds is not None else (None, None))
  _finite(
    pooled['estimate'], context=f'{context}.pooled.estimate',
    minimum=lower_bound, maximum=upper_bound)
  lower = _finite(
    pooled['ci_lower'], context=f'{context}.pooled.ci_lower',
    minimum=lower_bound, maximum=upper_bound)
  upper = _finite(
    pooled['ci_upper'], context=f'{context}.pooled.ci_upper',
    minimum=lower_bound, maximum=upper_bound)
  if lower > upper:
    raise ValueError(f'{context}.pooled confidence interval is reversed')


def _validate_analysis_contract(
    policy: Mapping[str, Any],
    analysis: object,
    *,
    source_plan: Mapping[str, Any],
) -> dict[str, bool]:
  analysis = _exact_keys(analysis, {
    'schema_version', 'artifact', 'created_utc', 'protocol_id', 'suite',
    'objective', 'scope_note', 'source_views', 'contrasts',
    'candidate_support', 'topology_permutation_diagnostic',
    'technical_diagnostics', 'compiled_plan', 'source_integrity',
    'analysis_sha256',
  }, context='causal-smoke analysis')
  contract = policy['analysis_contract']
  expected_identity = {
    'schema_version': contract['analysis_schema_version'],
    'artifact': contract['analysis_artifact'],
    'protocol_id': policy['protocol_id'],
    'suite': policy['source_suite'],
    'objective': contract['objective'],
  }
  for field, expected in expected_identity.items():
    if analysis[field] != expected:
      raise ValueError(f'causal-smoke analysis {field} differs from policy')
  if (not isinstance(analysis['scope_note'], str)
      or 'no diffusion ELBO' not in analysis['scope_note']):
    raise ValueError('causal-smoke analysis scope note is invalid')
  _timestamp(analysis['created_utc'], context='analysis created_utc')
  internal_sha = _lower_hex(
    analysis['analysis_sha256'], 64, context='analysis.analysis_sha256')
  body = dict(analysis)
  body.pop('analysis_sha256')
  if canonical_sha256(body) != internal_sha:
    raise ValueError('causal-smoke internal analysis SHA256 mismatch')

  compiled = _exact_keys(analysis['compiled_plan'], {
    'plan_id', 'source_manifest_sha256', 'source_compiled_plan_sha256',
    'source_repository_sha', 'source_repository_clean',
    'job_artifact_commitment_sha256',
  }, context='causal-smoke analysis compiled_plan')
  expected_compiled = {
    'plan_id': contract['source_plan_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'source_compiled_plan_sha256': contract['source_compiled_plan_sha256'],
    'source_repository_sha': contract['source_repository_sha'],
    'source_repository_clean': True,
  }
  for field, expected in expected_compiled.items():
    if compiled[field] != expected:
      raise ValueError(f'causal-smoke analysis compiled {field} differs')
  integrity = _validate_analysis_source_integrity(
    policy, analysis['source_integrity'])
  if (compiled['job_artifact_commitment_sha256']
      != integrity['commitment_sha256']):
    raise ValueError('causal-smoke analysis artifact commitment differs')
  if integrity['validated_job_ids'] != sorted(source_plan['job_ids']):
    raise ValueError('causal-smoke analysis does not bind every source job')

  source_views = _exact_keys(
    analysis['source_views'], set(contract['expected_source_views']),
    context='causal-smoke analysis source_views')
  for name, raw in source_views.items():
    view = _exact_keys(raw, {
      'compiled_plan_sha256', 'source_manifest_sha256',
      'source_repository_sha', 'source_integrity_commitment_sha256',
    }, context=f'causal-smoke source view {name}')
    if (view['compiled_plan_sha256']
        != contract['source_compiled_plan_sha256']
        or view['source_manifest_sha256']
        != policy['source_manifest_sha256']
        or view['source_repository_sha']
        != contract['source_repository_sha']):
      raise ValueError(f'causal-smoke source view {name} identity differs')
    _lower_hex(
      view['source_integrity_commitment_sha256'], 64,
      context=f'causal-smoke source view {name} commitment')

  expected_contrasts = dict(_contrasts())
  contrasts = _exact_keys(
    analysis['contrasts'], set(contract['expected_contrasts']),
    context='causal-smoke analysis contrasts')
  for index, name in enumerate(contract['expected_contrasts']):
    result = _exact_keys(contrasts[name], {
      'name', 'terms', 'analysis',
    }, context=f'causal-smoke contrast {name}')
    if result['name'] != name:
      raise ValueError(f'causal-smoke contrast {name} has wrong name')
    expected_terms = [
      {'arm': term.arm, 'metric': term.metric,
       'coefficient': term.coefficient}
      for term in expected_contrasts[name]]
    if result['terms'] != expected_terms:
      raise ValueError(f'causal-smoke contrast {name} terms differ')
    _validate_bootstrap_block(
      result['analysis'],
      context=f'causal-smoke contrast {name}',
      contract=contract,
      rng_seed=contract['bootstrap']['base_rng_seed'] + index * 101)

  support = _exact_keys(analysis['candidate_support'], {
    'arm', 'support_candidate_ks', 'by_candidate_k',
  }, context='causal-smoke candidate support')
  if (support['arm'] != 'dynamic_dynamic'
      or support['support_candidate_ks'] != contract['support_candidate_ks']):
    raise ValueError('causal-smoke candidate-support identity differs')
  by_support_k = _exact_keys(
    support['by_candidate_k'],
    {str(value) for value in contract['support_candidate_ks']},
    context='causal-smoke candidate support grid')
  for support_k in contract['support_candidate_ks']:
    metrics = _exact_keys(by_support_k[str(support_k)], {
      'candidate_recall', 'retained_unary_mass',
    }, context=f'causal-smoke support K={support_k}')
    for offset, metric in enumerate((
        'candidate_recall', 'retained_unary_mass')):
      _validate_bootstrap_block(
        metrics[metric],
        context=f'causal-smoke support K={support_k} {metric}',
        contract=contract,
        rng_seed=(contract['bootstrap']['base_rng_seed'] + 10_000
                  + support_k * 10 + offset),
        value_bounds=(0.0, 1.0))

  technical = _exact_keys(analysis['technical_diagnostics'], {
    'expected_arms', 'observed_arms', 'num_records', 'finite_statistics',
    'pairing', 'no_edge_identity', 'candidate_support',
    'topology_structure',
  }, context='causal-smoke technical diagnostics')
  if (technical['expected_arms'] != contract['controls']
      or technical['observed_arms'] != sorted(contract['controls'])):
    raise ValueError('causal-smoke technical-diagnostic arms differ')
  _positive_int(technical['num_records'], context='causal-smoke num_records')
  finite = _exact_keys(
    technical['finite_statistics'], {'passed'},
    context='causal-smoke finite diagnostics')
  pairing = _exact_keys(technical['pairing'], {
    'expected_train_seeds', 'observed_train_seeds',
    'num_conditions', 'mismatched_conditions', 'num_paired_units',
    'expected_arm_train_cells_per_unit', 'incomplete_paired_units',
    'masked_token_mismatched_units', 'passed',
  }, context='causal-smoke pairing diagnostics')
  expected_pairing_conditions = (
    len(contract['corruption_seeds']) * contract['expected_num_strata'])
  if (pairing['expected_train_seeds'] != contract['train_seeds']
      or pairing['observed_train_seeds'] != contract['train_seeds']
      or pairing['num_conditions'] != expected_pairing_conditions
      or pairing['expected_arm_train_cells_per_unit']
      != len(contract['controls']) * len(contract['train_seeds'])):
    raise ValueError('causal-smoke pairing diagnostic grid differs')
  _positive_int(
    pairing['num_paired_units'], context='causal-smoke paired units')
  for field in (
      'mismatched_conditions', 'incomplete_paired_units',
      'masked_token_mismatched_units'):
    value = pairing[field]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      raise ValueError(f'causal-smoke pairing {field} must be non-negative')
  pairing_passed = all(pairing[field] == 0 for field in (
    'mismatched_conditions', 'incomplete_paired_units',
    'masked_token_mismatched_units'))
  if pairing['passed'] is not pairing_passed:
    raise ValueError('causal-smoke pairing pass flag is inconsistent')
  no_edge = _exact_keys(technical['no_edge_identity'], {
    'maximum_absolute_error', 'absolute_tolerance', 'passed',
  }, context='causal-smoke no-edge diagnostic')
  no_edge_error = _finite(
    no_edge['maximum_absolute_error'], context='no-edge maximum error',
    minimum=0.0)
  no_edge_tolerance = _finite(
    no_edge['absolute_tolerance'], context='no-edge tolerance', minimum=0.0)
  frozen_no_edge_tolerance = float(
    policy['technical_gates']['no_edge_absolute_tolerance'])
  if (no_edge_tolerance != frozen_no_edge_tolerance
      or no_edge['passed'] is not (
        no_edge_error <= frozen_no_edge_tolerance)):
    raise ValueError('causal-smoke no-edge pass flag is inconsistent')
  support_diag = _exact_keys(technical['candidate_support'], {
    'expected_candidate_ks', 'grid_complete',
    'monotonicity_absolute_tolerance', 'monotone_within_tolerance',
  }, context='causal-smoke support diagnostics')
  frozen_support_tolerance = float(
    policy['technical_gates']['support_monotonicity_absolute_tolerance'])
  if (support_diag['expected_candidate_ks']
      != contract['support_candidate_ks']
      or _finite(
        support_diag['monotonicity_absolute_tolerance'],
        context='candidate support monotonicity tolerance', minimum=0.0)
      != frozen_support_tolerance):
    raise ValueError('causal-smoke support diagnostic K grid differs')
  topology = _exact_keys(technical['topology_structure'], {
    'conditions', 'every_condition_nonempty', 'passed',
  }, context='causal-smoke topology structure')
  if not isinstance(topology['conditions'], Mapping):
    raise TypeError('causal-smoke topology conditions must be a mapping')
  expected_topology_cells = {
    (arm, dataset, spec['revision'], float(rate), contract['candidate_k'])
    for arm in contract['controls']
    for dataset, spec in contract['datasets'].items()
    for rate in contract['mask_rates']}
  observed_topology_cells = set()
  recomputed_nonempty = []
  for key, raw in topology['conditions'].items():
    row = _exact_keys(raw, {
      'arm', 'dataset', 'dataset_revision', 'mask_rate',
      'adapter_candidate_k', 'num_records', 'masked_tokens',
      'selected_edges', 'selected_edges_per_masked_token', 'nonempty',
    }, context=f'causal-smoke topology condition {key}')
    cell = (
      row['arm'], row['dataset'], row['dataset_revision'],
      float(row['mask_rate']), row['adapter_candidate_k'])
    observed_topology_cells.add(cell)
    _positive_int(row['num_records'], context=f'{key}.num_records')
    masked = _positive_int(row['masked_tokens'], context=f'{key}.masked_tokens')
    selected = row['selected_edges']
    if (not isinstance(selected, int) or isinstance(selected, bool)
        or selected < 0):
      raise ValueError(f'{key}.selected_edges must be non-negative')
    density = _finite(
      row['selected_edges_per_masked_token'], context=f'{key}.edge density',
      minimum=0.0)
    if not math.isclose(
        density, selected / masked, rel_tol=0.0, abs_tol=1e-15):
      raise ValueError(f'{key}.edge density is inconsistent')
    if row['nonempty'] is not (selected > 0):
      raise ValueError(f'{key}.nonempty is inconsistent')
    recomputed_nonempty.append(selected > 0)
  if (observed_topology_cells != expected_topology_cells
      or len(topology['conditions']) != len(expected_topology_cells)):
    raise ValueError('causal-smoke topology structural grid is incomplete')
  every_nonempty = bool(recomputed_nonempty) and all(recomputed_nonempty)
  if (topology['every_condition_nonempty'] is not every_nonempty
      or topology['passed'] is not every_nonempty):
    raise ValueError('causal-smoke topology structural pass flag differs')

  permutation = _exact_keys(analysis['topology_permutation_diagnostic'], {
    'arm', 'estimand', 'pooled', 'conditions', 'gate',
  }, context='causal-smoke topology permutation diagnostic')
  if (permutation['arm'] != 'dynamic_dynamic'
      or permutation['estimand']
      != 'edge_weighted_fraction_of_selected_edges_reassigned'):
    raise ValueError('causal-smoke topology permutation identity differs')
  gates_cfg = policy['technical_gates']
  pooled = _exact_keys(permutation['pooled'], {
    'num_records', 'degree_sequence_preserved_records',
    'component_sizes_preserved_records',
    'degree_sequence_preserved_every_record',
    'component_sizes_preserved_every_record',
    'selected_edges', 'changed_edges', 'changed_edge_fraction',
    'minimum_changed_edge_fraction', 'passed',
  }, context='causal-smoke pooled topology permutation')
  pooled_selected = _positive_int(
    pooled['selected_edges'], context='pooled selected edges')
  pooled_records = _positive_int(
    pooled['num_records'], context='pooled topology records')
  pooled_degree_records = pooled['degree_sequence_preserved_records']
  pooled_component_records = pooled['component_sizes_preserved_records']
  for field, value in (
      ('degree_sequence_preserved_records', pooled_degree_records),
      ('component_sizes_preserved_records', pooled_component_records)):
    if (not isinstance(value, int) or isinstance(value, bool)
        or not 0 <= value <= pooled_records):
      raise ValueError(f'pooled {field} is invalid')
  pooled_degree_passed = pooled_degree_records == pooled_records
  pooled_component_passed = pooled_component_records == pooled_records
  if (pooled['degree_sequence_preserved_every_record']
      is not pooled_degree_passed
      or pooled['component_sizes_preserved_every_record']
      is not pooled_component_passed):
    raise ValueError(
      'causal-smoke pooled topology-preservation flags differ')
  pooled_changed = pooled['changed_edges']
  if (not isinstance(pooled_changed, int) or isinstance(pooled_changed, bool)
      or not 0 <= pooled_changed <= pooled_selected):
    raise ValueError('pooled changed edges are invalid')
  pooled_fraction = _finite(
    pooled['changed_edge_fraction'], context='pooled changed fraction',
    minimum=0.0, maximum=1.0)
  pooled_threshold = float(
    gates_cfg['minimum_pooled_changed_edge_fraction'])
  if (not math.isclose(
        pooled_fraction, pooled_changed / pooled_selected,
        rel_tol=0.0, abs_tol=1e-15)
      or float(pooled['minimum_changed_edge_fraction']) != pooled_threshold
      or pooled['passed'] is not (pooled_fraction >= pooled_threshold)):
    raise ValueError('causal-smoke pooled topology permutation differs')
  if not isinstance(permutation['conditions'], Mapping) \
      or len(permutation['conditions']) != contract['expected_num_strata']:
    raise ValueError('causal-smoke topology permutation grid is incomplete')
  condition_passes = []
  observed_permutation_cells = set()
  condition_selected_total = 0
  condition_changed_total = 0
  condition_record_total = 0
  condition_degree_record_total = 0
  condition_component_record_total = 0
  condition_degree_passes = []
  condition_component_passes = []
  condition_threshold = float(
    gates_cfg['minimum_condition_changed_edge_fraction'])
  for key, raw in permutation['conditions'].items():
    row = _exact_keys(raw, {
      'dataset', 'dataset_revision', 'mask_rate', 'adapter_candidate_k',
      'num_records', 'degree_sequence_preserved_records',
      'component_sizes_preserved_records',
      'degree_sequence_preserved_every_record',
      'component_sizes_preserved_every_record',
      'selected_edges', 'changed_edges',
      'changed_edge_fraction', 'minimum_changed_edge_fraction', 'passed',
    }, context=f'causal-smoke permutation condition {key}')
    cell = (
      row['dataset'], row['dataset_revision'], float(row['mask_rate']),
      row['adapter_candidate_k'])
    observed_permutation_cells.add(cell)
    num_records = _positive_int(
      row['num_records'], context=f'{key}.num_records')
    degree_records = row['degree_sequence_preserved_records']
    component_records = row['component_sizes_preserved_records']
    for field, value in (
        ('degree_sequence_preserved_records', degree_records),
        ('component_sizes_preserved_records', component_records)):
      if (not isinstance(value, int) or isinstance(value, bool)
          or not 0 <= value <= num_records):
        raise ValueError(f'{key}.{field} is invalid')
    degree_passed = degree_records == num_records
    component_passed = component_records == num_records
    if (row['degree_sequence_preserved_every_record'] is not degree_passed
        or row['component_sizes_preserved_every_record']
        is not component_passed):
      raise ValueError(f'{key}.topology preservation flag is inconsistent')
    selected = _positive_int(
      row['selected_edges'], context=f'{key}.selected_edges')
    changed = row['changed_edges']
    if (not isinstance(changed, int) or isinstance(changed, bool)
        or not 0 <= changed <= selected):
      raise ValueError(f'{key}.changed_edges is invalid')
    fraction = _finite(
      row['changed_edge_fraction'], context=f'{key}.changed fraction',
      minimum=0.0, maximum=1.0)
    passed = fraction >= condition_threshold
    if (not math.isclose(
          fraction, changed / selected, rel_tol=0.0, abs_tol=1e-15)
        or float(row['minimum_changed_edge_fraction']) != condition_threshold
        or row['passed'] is not passed):
      raise ValueError(f'{key}.permutation pass flag is inconsistent')
    condition_passes.append(passed)
    condition_selected_total += selected
    condition_changed_total += changed
    condition_record_total += num_records
    condition_degree_record_total += degree_records
    condition_component_record_total += component_records
    condition_degree_passes.append(degree_passed)
    condition_component_passes.append(component_passed)
  expected_permutation_cells = {
    (dataset, spec['revision'], float(rate), contract['candidate_k'])
    for dataset, spec in contract['datasets'].items()
    for rate in contract['mask_rates']}
  if (observed_permutation_cells != expected_permutation_cells
      or len(permutation['conditions']) != len(expected_permutation_cells)):
    raise ValueError('causal-smoke topology permutation cells differ')
  if (pooled_selected != condition_selected_total
      or pooled_changed != condition_changed_total
      or pooled_records != condition_record_total
      or pooled_degree_records != condition_degree_record_total
      or pooled_component_records != condition_component_record_total):
    raise ValueError('causal-smoke pooled topology counts differ from cells')
  permutation_gate = _exact_keys(permutation['gate'], {
    'degree_sequence_preserved_every_record',
    'component_sizes_preserved_every_record',
    'pooled_fraction_passed', 'every_condition_fraction_passed', 'passed',
  }, context='causal-smoke topology permutation gate')
  expected_every = all(condition_passes)
  expected_degree = bool(condition_degree_passes) and all(
    condition_degree_passes)
  expected_components = bool(condition_component_passes) and all(
    condition_component_passes)
  expected_gate = (
    bool(pooled['passed']) and expected_every
    and expected_degree and expected_components)
  if (permutation_gate['degree_sequence_preserved_every_record']
      is not expected_degree
      or permutation_gate['component_sizes_preserved_every_record']
      is not expected_components
      or permutation_gate['pooled_fraction_passed'] is not pooled['passed']
      or permutation_gate['every_condition_fraction_passed'] is not expected_every
      or permutation_gate['passed'] is not expected_gate):
    raise ValueError('causal-smoke topology permutation gate is inconsistent')

  for name, value in (
      ('finite_statistics.passed', finite['passed']),
      ('pairing.passed', pairing['passed']),
      ('candidate_support.grid_complete', support_diag['grid_complete']),
      ('candidate_support.monotone_within_tolerance',
       support_diag['monotone_within_tolerance'])):
    if type(value) is not bool:
      raise TypeError(f'causal-smoke {name} must be boolean')
  return {
    'integrity_complete': True,
    'four_arm_factorial_complete': True,
    'pairing_complete': pairing['passed'],
    'finite_statistics': finite['passed'],
    'confidence_intervals_ordered': True,
    'no_edge_identity_within_frozen_tolerance': no_edge['passed'],
    'candidate_support_grid_complete': support_diag['grid_complete'],
    'candidate_support_monotone_within_frozen_tolerance': (
      support_diag['monotone_within_tolerance']),
    'nonempty_forest_every_condition': topology['passed'],
    'topology_degree_sequence_preserved': expected_degree,
    'topology_component_sizes_preserved': expected_components,
    'topology_permutation_gate': permutation_gate['passed'],
  }


def _verify_authoritative_analysis(
    policy: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    source_plan_path: Path,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
  contract = policy['analysis_contract']
  recomputed = build_analysis(
    plan_dir=source_plan_path.expanduser().resolve().parent,
    manifest_path=manifest_path,
    suite_name=policy['source_suite'],
    num_resamples=contract['bootstrap']['num_resamples'],
    rng_seed=contract['bootstrap']['base_rng_seed'],
    confidence_level=contract['bootstrap']['confidence_level'],
    timestamp_utc=analysis['created_utc'],
    technical_gates=policy['technical_gates'],
    repo_root=repo_root)
  if recomputed != dict(analysis):
    raise ValueError(
      'causal-smoke analysis differs from deterministic recomputation of '
      'marker-bound source records')
  return recomputed['source_integrity']


def evaluate_causal_analysis(
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
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  _lower_hex(policy_sha256, 64, context='policy SHA256')
  _lower_hex(analysis_sha256, 64, context='analysis SHA256')
  source_plan_path = source_plan_path.expanduser().resolve()
  validated_plan = _validate_source_plan(
    policy,
    source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=source_plan_sha256,
    manifest_path=manifest_path,
    repo_root=repo_root)
  gate_states = _validate_analysis_contract(
    policy, analysis, source_plan=validated_plan)
  verified_integrity = _verify_authoritative_analysis(
    policy,
    analysis,
    source_plan_path=source_plan_path,
    manifest_path=manifest_path,
    repo_root=repo_root)
  _verify_temporal_order(
    policy,
    validated_plan,
    source_plan_path=source_plan_path,
    analysis_created_utc=analysis['created_utc'])
  decision_time = created_utc or _utc_now()
  if _timestamp(decision_time, context='decision created_utc') < _timestamp(
      analysis['created_utc'], context='analysis created_utc'):
    raise ValueError('causal-smoke routing decision predates analysis')
  route_cfg = policy['routing']['primary']
  criteria = {name: gate_states[name] for name in route_cfg['requires']}
  promote = all(criteria.values())
  route = {
    'target_suite': route_cfg['target_suite'],
    'promote': promote,
    'criteria': criteria,
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
      'template_sha256': policy['template']['sha256'],
    },
    'analysis': {
      'analysis_sha256': analysis_sha256,
      'internal_analysis_sha256': analysis['analysis_sha256'],
      'created_utc': analysis['created_utc'],
      'plan_id': analysis['compiled_plan']['plan_id'],
    },
    'compiled_plan': {
      'file_sha256': source_plan_sha256,
      'plan_id': validated_plan['plan_id'],
      'source_repository_sha': validated_plan['repository']['sha'],
      'source_repository_clean': True,
      'job_spec_commitment_sha256': canonical_sha256(
        validated_plan['job_spec_sha256']),
      'job_artifact_commitment_sha256': verified_integrity[
        'commitment_sha256'],
    },
    'integrity': {
      'passed': True,
      'criteria': {
        'trusted_template_bound': True,
        'policy_precedes_every_source_job': True,
        'analysis_follows_every_source_job': True,
        'identity_manifest_and_repository_bound': True,
        'compiled_plan_and_all_job_specs_bound': True,
        'success_markers_and_outputs_bound': True,
        'analysis_recomputed_from_bound_records': True,
        'technical_only_routing_contract': True,
      },
    },
    'gates': {
      name: {
        'passed': state,
        'criterion_type': 'technical_validity_only',
      }
      for name, state in gate_states.items()
    },
    'routes': {'primary': route},
    'outcome': 'promote_causal_primary' if promote \
      else 'stop_for_technical_remediation',
    'compiler_evidence': {
      'primary': {
        'eligible': promote,
        'target_suite': route['target_suite'],
        'filename': f'{route["target_suite"]}-promotion.json',
      },
    },
  }


def build_causal_compiler_evidence(
    decision: Mapping[str, Any],
    route_name: str,
    *,
    policy_path: Path,
    analysis_path: Path,
    source_plan_path: Path,
    decision_path: Path,
) -> dict[str, Any]:
  evidence = build_compiler_evidence(
    decision,
    route_name,
    analysis_path=analysis_path,
    source_plan_path=source_plan_path,
    decision_path=decision_path)
  evidence['schema_version'] = 3
  evidence['commitments']['policy_template_sha256'] = decision['policy'][
    'template_sha256']
  evidence['artifacts']['policy_path'] = str(
    policy_path.expanduser().resolve())
  return evidence


def verify_causal_compiler_evidence(
    evidence: object,
    *,
    evidence_path: Path,
    promoted_suite: str,
    manifest_path: Path,
    trusted_template_path: Path = DEFAULT_TEMPLATE,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  evidence = _exact_keys(evidence, {
    'schema_version', 'artifact', 'protocol_id', 'source_manifest_sha256',
    'source_suite', 'promoted_suite', 'route_name', 'decision', 'criteria',
    'commitments', 'artifacts', 'created_utc',
  }, context='causal-smoke promotion evidence')
  if (evidence['schema_version'] != 3
      or evidence['artifact'] != 'experiment_suite_promotion_decision'
      or evidence['promoted_suite'] != promoted_suite
      or evidence['route_name'] != 'primary'
      or evidence['decision'] != 'promote'):
    raise ValueError('invalid causal-smoke promotion evidence identity')
  commitments = _exact_keys(evidence['commitments'], {
    'policy_sha256', 'policy_template_sha256', 'analysis_sha256',
    'source_compiled_plan_sha256', 'source_plan_id',
    'source_repository_sha', 'source_repository_clean',
    'source_job_spec_commitment_sha256',
    'source_job_artifact_commitment_sha256',
    'canonical_decision_sha256',
  }, context='causal-smoke evidence commitments')
  for field, value in commitments.items():
    if field == 'source_repository_clean':
      if value is not True:
        raise ValueError('causal-smoke evidence repository is not clean')
    elif field == 'source_repository_sha':
      _lower_hex(value, 40, context=field)
    else:
      _lower_hex(value, 64, context=field)
  artifacts = _exact_keys(evidence['artifacts'], {
    'policy_path', 'analysis_path', 'source_compiled_plan_path',
    'routing_decision_path',
  }, context='causal-smoke evidence artifacts')
  policy_path = _artifact_path(artifacts['policy_path'], context='policy_path')
  analysis_path = _artifact_path(
    artifacts['analysis_path'], context='analysis_path')
  source_plan_path = _artifact_path(
    artifacts['source_compiled_plan_path'], context='source_plan_path')
  decision_path = _artifact_path(
    artifacts['routing_decision_path'], context='decision_path')
  if evidence_path.expanduser().resolve() in {
      policy_path, analysis_path, source_plan_path, decision_path}:
    raise ValueError('causal-smoke promotion evidence self-references')
  if (commitments['policy_sha256'] != sha256_file(policy_path)
      or commitments['policy_template_sha256']
      != sha256_file(trusted_template_path)
      or commitments['analysis_sha256'] != sha256_file(analysis_path)
      or commitments['source_compiled_plan_sha256']
      != sha256_file(source_plan_path)):
    raise ValueError('causal-smoke evidence artifact SHA mismatch')
  policy = load_and_validate_causal_policy(
    policy_path,
    template_path=trusted_template_path,
    manifest_path=manifest_path,
    repo_root=repo_root)
  analysis = _read_mapping(analysis_path, context='causal-smoke analysis')
  source_plan = _read_mapping(source_plan_path, context='causal-smoke plan')
  decision = _read_mapping(decision_path, context='causal-smoke decision')
  if canonical_sha256(decision) != commitments['canonical_decision_sha256']:
    raise ValueError('causal-smoke routing decision hash mismatch')
  canonical = evaluate_causal_analysis(
    policy,
    analysis,
    policy_sha256=sha256_file(policy_path),
    analysis_sha256=sha256_file(analysis_path),
    source_plan=source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=sha256_file(source_plan_path),
    manifest_path=manifest_path,
    created_utc=decision.get('created_utc'),
    repo_root=repo_root)
  if decision != canonical:
    raise ValueError('causal-smoke decision differs from deterministic replay')
  expected = build_causal_compiler_evidence(
    canonical,
    str(evidence['route_name']),
    policy_path=policy_path,
    analysis_path=analysis_path,
    source_plan_path=source_plan_path,
    decision_path=decision_path)
  if dict(evidence) != expected:
    raise ValueError('causal-smoke promotion evidence differs from canonical')
  return expected


def _parse_args(argv: Sequence[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--policy', type=Path, required=True)
  parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
  parser.add_argument('--analysis', type=Path, required=True)
  parser.add_argument('--source-plan', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--compiler-evidence-dir', type=Path)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  policy_path = args.policy.expanduser().resolve()
  analysis_path = args.analysis.expanduser().resolve()
  source_plan_path = args.source_plan.expanduser().resolve()
  output = args.output.expanduser().resolve()
  if output.exists():
    raise FileExistsError(output)
  policy = load_and_validate_causal_policy(
    policy_path,
    template_path=args.template,
    manifest_path=args.manifest)
  analysis = _read_mapping(analysis_path, context='causal-smoke analysis')
  source_plan = _read_mapping(source_plan_path, context='causal-smoke plan')
  decision = evaluate_causal_analysis(
    policy,
    analysis,
    policy_sha256=sha256_file(policy_path),
    analysis_sha256=sha256_file(analysis_path),
    source_plan=source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=sha256_file(source_plan_path),
    manifest_path=args.manifest)
  evidence_payload = None
  evidence_path = None
  if (args.compiler_evidence_dir is not None
      and decision['routes']['primary']['promote']):
    evidence_path = (
      args.compiler_evidence_dir.expanduser().resolve()
      / f'{decision["routes"]["primary"]["target_suite"]}-promotion.json')
    if evidence_path.exists():
      raise FileExistsError(evidence_path)
    evidence_payload = build_causal_compiler_evidence(
      decision,
      'primary',
      policy_path=policy_path,
      analysis_path=analysis_path,
      source_plan_path=source_plan_path,
      decision_path=output)
  _exclusive_write_json(output, decision)
  if evidence_payload is not None:
    _exclusive_write_json(evidence_path, evidence_payload)
  print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
  return 0 if decision['routes']['primary']['promote'] else 2


if __name__ == '__main__':
  raise SystemExit(main())
