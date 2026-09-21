#!/usr/bin/env python3
"""Bind the frozen causal-smoke routing template to an unstarted plan.

The template is committed before the source plan exists. This finalizer fills
only repository/plan provenance and refuses to run after any source artifact
directory becomes non-empty. Scientific outcomes are intentionally absent
from both the template and the finalized policy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.compile_experiment_matrix import (  # noqa: E402
  _git_metadata,
  build_jobs,
  load_and_validate_manifest,
  sha256_file,
)
from scripts.finalize_candidate_k_policy import (  # noqa: E402
  _exact_keys,
  _exclusive_write_json,
  _lower_hex,
  _read_mapping,
  _timestamp,
  canonical_sha256,
)


DEFAULT_MANIFEST = (
  REPO_ROOT / 'configs/experiment'
  / 'contextual-forest-causal-evidence-v1.yaml')
DEFAULT_TEMPLATE = (
  REPO_ROOT / 'configs/experiment'
  / 'contextual-forest-causal-smoke-promotion-policy-template.yaml')
EXPECTED_CONTROLS = [
  'dynamic_dynamic', 'fixed_dynamic', 'dynamic_fixed', 'static_static']
EXPECTED_SOURCE_VIEWS = [
  'contextual_vs_static', 'contextual_topology_gain',
  'contextual_factor_gain']
EXPECTED_CONTRASTS = [
  'factorized_backbone_vs_contextual_joint',
  'singleton_product_vs_contextual_joint',
  'parameter_matched_no_edge_vs_contextual_joint',
  'matched_permuted_topology_vs_contextual_joint',
  'contextual_joint_vs_static_joint',
  'contextual_topology_gain_dynamic_factors',
  'contextual_topology_gain_fixed_factors',
  'contextual_factor_gain_dynamic_topology',
  'contextual_factor_gain_fixed_topology',
  'topology_by_factor_interaction',
]
TECHNICAL_GATE_NAMES = [
  'integrity_complete',
  'four_arm_factorial_complete',
  'pairing_complete',
  'finite_statistics',
  'confidence_intervals_ordered',
  'no_edge_identity_within_frozen_tolerance',
  'candidate_support_grid_complete',
  'candidate_support_monotone_within_frozen_tolerance',
  'nonempty_forest_every_condition',
  'topology_degree_sequence_preserved',
  'topology_component_sizes_preserved',
  'topology_permutation_gate',
]


def _validate_template(
    template: Mapping[str, Any],
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  template = _exact_keys(template, {
    'schema_version', 'artifact', 'policy_id', 'policy_status',
    'template_frozen_utc', 'protocol_id', 'source_manifest_sha256',
    'source_suite', 'analysis_contract', 'technical_gates', 'routing',
  }, context='causal-smoke policy template')
  expected_identity = {
    'schema_version': 1,
    'artifact': 'causal_smoke_promotion_policy_template',
    'policy_status': 'frozen_template_before_causal_smoke_plan_or_results',
    'source_suite': 'causal_smoke',
  }
  for field, expected in expected_identity.items():
    if template[field] != expected:
      raise ValueError(f'causal-smoke template {field} must equal {expected!r}')
  _timestamp(
    template['template_frozen_utc'], context='template.template_frozen_utc')
  manifest_path = manifest_path.expanduser().resolve()
  manifest = load_and_validate_manifest(manifest_path, repo_root=repo_root)
  if template['protocol_id'] != manifest['protocol_id']:
    raise ValueError('causal-smoke template protocol differs from manifest')
  if template['source_manifest_sha256'] != sha256_file(manifest_path):
    raise ValueError('causal-smoke template manifest hash differs from bytes')
  _lower_hex(
    template['source_manifest_sha256'], 64,
    context='template.source_manifest_sha256')
  suite = manifest['suites'][template['source_suite']]
  contract = _exact_keys(template['analysis_contract'], {
    'analysis_schema_version', 'analysis_artifact', 'objective', 'controls',
    'candidate_k', 'support_candidate_ks', 'train_seeds',
    'corruption_seeds', 'datasets', 'mask_rates', 'expected_num_strata',
    'expected_source_views', 'expected_contrasts', 'bootstrap',
  }, context='template.analysis_contract')
  expected_contract_identity = {
    'analysis_schema_version': 2,
    'analysis_artifact': 'contextual_forest_causal_denoising_analysis',
    'objective': 'paired_conditional_denoising_nll_per_masked_token',
    'controls': EXPECTED_CONTROLS,
    'candidate_k': manifest['evaluation']['primary_candidate_k'],
    'support_candidate_ks': manifest['evaluation']['candidate_ks'],
    'train_seeds': suite['train_seeds'],
    'corruption_seeds': suite['corruption_seeds'],
    'mask_rates': suite['mask_rates'],
    'expected_source_views': EXPECTED_SOURCE_VIEWS,
    'expected_contrasts': EXPECTED_CONTRASTS,
  }
  for field, expected in expected_contract_identity.items():
    if contract[field] != expected:
      raise ValueError(
        f'causal-smoke template {field} differs from frozen protocol')
  if suite['controls'] != EXPECTED_CONTROLS or suite['candidate_ks'] != [
      contract['candidate_k']]:
    raise ValueError('causal smoke must be the exact four-arm K=128 suite')
  expected_datasets = {}
  for alias in suite['datasets']:
    config_name = manifest['datasets'][alias]['data_config']
    config = _read_mapping(
      repo_root / 'configs/data' / f'{config_name}.yaml',
      context=f'data config {config_name}')
    expected_datasets[config['valid']] = {'revision': config['valid_revision']}
  if dict(contract['datasets']) != expected_datasets:
    raise ValueError('causal-smoke template datasets differ from source suite')
  if contract['expected_num_strata'] != (
      len(expected_datasets) * len(contract['mask_rates'])):
    raise ValueError('causal-smoke expected_num_strata is inconsistent')
  bootstrap = _exact_keys(contract['bootstrap'], {
    'method', 'num_resamples', 'base_rng_seed', 'confidence_level',
  }, context='template.analysis_contract.bootstrap')
  analysis_cfg = manifest['analysis']
  if (bootstrap['method'] != 'hierarchical_paired_percentile_bootstrap'
      or bootstrap['num_resamples'] != analysis_cfg['bootstrap_resamples']
      or bootstrap['base_rng_seed'] != analysis_cfg['bootstrap_seed']
      or float(bootstrap['confidence_level'])
      != float(analysis_cfg['confidence_level'])):
    raise ValueError('causal-smoke bootstrap differs from frozen analysis')

  gates = _exact_keys(template['technical_gates'], {
    'require_complete_source_artifact_integrity',
    'require_complete_four_arm_factorial',
    'require_identical_pairing_digests',
    'require_identical_masked_tokens_across_arms_and_training_seeds',
    'require_finite_statistics',
    'require_ordered_confidence_intervals',
    'require_no_edge_identity_within_frozen_tolerance',
    'no_edge_absolute_tolerance',
    'require_complete_candidate_support_grid',
    'require_candidate_support_monotone_within_frozen_tolerance',
    'support_monotonicity_absolute_tolerance',
    'require_nonempty_forest_every_condition',
    'require_degree_sequence_preserved_every_record',
    'require_component_sizes_preserved_every_record',
    'minimum_pooled_changed_edge_fraction',
    'minimum_condition_changed_edge_fraction',
  }, context='template.technical_gates')
  for field in gates:
    if field.startswith('require_') and gates[field] is not True:
      raise ValueError(f'causal-smoke technical gate {field} must remain true')
  for field in (
      'no_edge_absolute_tolerance',
      'support_monotonicity_absolute_tolerance'):
    value = gates[field]
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) or float(value) < 0.0):
      raise ValueError(
        f'causal-smoke technical tolerance {field} must be finite and '
        'non-negative')
  permutation = analysis_cfg['permutation_control_gate']
  if (float(gates['minimum_pooled_changed_edge_fraction'])
      != float(permutation['minimum_pooled_changed_edge_fraction'])
      or float(gates['minimum_condition_changed_edge_fraction'])
      != float(permutation['minimum_condition_changed_edge_fraction'])):
    raise ValueError('causal-smoke topology gate differs from manifest')
  expected_routing = {
    'primary': {
      'target_suite': 'causal_primary',
      'requires': TECHNICAL_GATE_NAMES,
    },
  }
  if dict(template['routing']) != expected_routing:
    raise ValueError('causal-smoke routing differs from technical-only route')
  if manifest['suites']['causal_primary']['promotion_from'] != \
      template['source_suite']:
    raise ValueError('causal primary is not gated on causal smoke')
  return dict(template)


def _validate_source_plan(
    template: Mapping[str, Any],
    plan_dir: Path,
    *,
    manifest_path: Path,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
  manifest = load_and_validate_manifest(manifest_path, repo_root=repo_root)
  plan_path = plan_dir / 'compiled-plan.json'
  plan = _exact_keys(_read_mapping(plan_path, context='compiled plan'), {
    'schema_version', 'protocol_id', 'source_manifest_sha256', 'repository',
    'artifact_root', 'selected_suites', 'promotion_evidence', 'plan_id',
    'manifest_protocol_status', 'scientific_scope', 'job_counts',
    'num_jobs', 'job_ids', 'job_spec_sha256',
  }, context='compiled plan')
  expected = {
    'schema_version': 2,
    'protocol_id': template['protocol_id'],
    'source_manifest_sha256': template['source_manifest_sha256'],
    'selected_suites': [template['source_suite']],
    'promotion_evidence': {},
  }
  for field, value in expected.items():
    if plan[field] != value:
      raise ValueError(f'causal-smoke compiled plan {field} differs')
  repository = _exact_keys(
    plan['repository'], {'sha', 'dirty'}, context='compiled plan repository')
  repository_sha = _lower_hex(
    repository['sha'], 40, context='compiled plan repository SHA')
  if repository['dirty'] is not False:
    raise ValueError('causal-smoke compiled plan repository was not clean')
  checkout = _git_metadata(repo_root)
  if checkout != {'sha': repository_sha, 'dirty': False}:
    raise ValueError(
      'policy finalization checkout differs from the clean compiled-plan '
      f'repository: expected {repository_sha}, found {checkout}')
  plan_identity = {
    'protocol_id': plan['protocol_id'],
    'source_manifest_sha256': plan['source_manifest_sha256'],
    'repository': dict(repository),
    'artifact_root': plan['artifact_root'],
    'selected_suites': plan['selected_suites'],
    'promotion_evidence': plan['promotion_evidence'],
  }
  plan_id = _lower_hex(plan['plan_id'], 64, context='compiled plan ID')
  if canonical_sha256(plan_identity) != plan_id:
    raise ValueError('causal-smoke compiled plan ID is not canonical')
  expected_jobs = build_jobs(
    manifest,
    selected_suites=[template['source_suite']],
    artifact_root=Path(plan['artifact_root']).expanduser().resolve(),
    source_manifest_sha256=plan['source_manifest_sha256'],
    source_repository_sha=repository_sha,
    plan_id=plan_id)
  counts = {}
  for job in expected_jobs.values():
    counts[job['kind']] = counts.get(job['kind'], 0) + 1
  if (plan['job_ids'] != list(expected_jobs)
      or plan['job_spec_sha256'] != {
        job_id: canonical_sha256(job)
        for job_id, job in expected_jobs.items()}
      or plan['num_jobs'] != len(expected_jobs)
      or plan['job_counts'] != dict(sorted(counts.items()))
      or plan['manifest_protocol_status'] != manifest['protocol_status']
      or plan['scientific_scope'] != manifest['scientific_scope']):
    raise ValueError(
      'compiled plan does not match the exact four-arm causal smoke factorial')
  artifact_dirs = []
  artifact_root = Path(plan['artifact_root']).expanduser().resolve()
  for job_id, expected_job in expected_jobs.items():
    job = _read_mapping(
      plan_dir / 'jobs' / f'{job_id}.json', context=f'job {job_id}')
    if job != expected_job:
      raise ValueError(
        f'job {job_id} differs from the exact four-arm causal smoke factorial')
    artifact_dir = Path(job['artifact_dir']).expanduser().resolve()
    try:
      artifact_dir.relative_to(artifact_root)
    except ValueError as error:
      raise ValueError(f'job {job_id} artifact directory escapes root') \
        from error
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
      raise RuntimeError(
        'cannot freeze policy after source-suite execution began: '
        f'{artifact_dir} is non-empty')
    artifact_dirs.append(str(artifact_dir))
  return dict(plan), expected_jobs, artifact_dirs


def finalize_policy(
    template_path: Path,
    plan_dir: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    frozen_utc: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  template_path = template_path.expanduser().resolve()
  manifest_path = manifest_path.expanduser().resolve()
  plan_dir = plan_dir.expanduser().resolve()
  template = _validate_template(
    _read_mapping(template_path, context='causal-smoke policy template'),
    manifest_path=manifest_path,
    repo_root=repo_root)
  plan, _, artifact_dirs = _validate_source_plan(
    template,
    plan_dir,
    manifest_path=manifest_path,
    repo_root=repo_root)
  plan_path = plan_dir / 'compiled-plan.json'
  freeze_time = frozen_utc or dt.datetime.now(dt.timezone.utc).isoformat()
  if _timestamp(freeze_time, context='frozen_utc') < _timestamp(
      template['template_frozen_utc'], context='template frozen_utc'):
    raise ValueError('frozen_utc cannot predate the policy template')
  contract = {
    **dict(template['analysis_contract']),
    'source_plan_id': plan['plan_id'],
    'source_compiled_plan_sha256': sha256_file(plan_path),
    'source_repository_sha': plan['repository']['sha'],
    'source_repository_clean': True,
  }
  return {
    'schema_version': 1,
    'artifact': 'causal_smoke_promotion_policy',
    'policy_id': template['policy_id'],
    'policy_status': 'frozen_before_source_suite_results',
    'template': {
      'path': str(template_path),
      'sha256': sha256_file(template_path),
      'frozen_utc': template['template_frozen_utc'],
    },
    'frozen_utc': freeze_time,
    'protocol_id': template['protocol_id'],
    'source_manifest_sha256': template['source_manifest_sha256'],
    'source_suite': template['source_suite'],
    'analysis_contract': contract,
    'source_plan': {
      'path': str(plan_path),
      'file_sha256': sha256_file(plan_path),
      'plan_id': plan['plan_id'],
      'source_repository_sha': plan['repository']['sha'],
      'source_repository_clean': True,
      'promotion_evidence': {},
      'job_spec_commitment_sha256': canonical_sha256(
        dict(plan['job_spec_sha256'])),
    },
    'freeze_attestation': {
      'status': 'no_source_suite_artifact_directory_was_nonempty',
      'num_jobs_checked': len(plan['job_ids']),
      'job_ids_sha256': canonical_sha256(plan['job_ids']),
      'artifact_dirs_sha256': canonical_sha256(sorted(artifact_dirs)),
    },
    'technical_gates': dict(template['technical_gates']),
    'routing': dict(template['routing']),
  }


def _parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
  parser.add_argument('--plan-dir', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = _parse_args(argv)
  policy = finalize_policy(
    args.template, args.plan_dir, manifest_path=args.manifest)
  _exclusive_write_json(args.output, policy)
  print(json.dumps({
    'event': 'causal_smoke_policy_frozen',
    'output': str(args.output.expanduser().resolve()),
    'policy_id': policy['policy_id'],
    'source_plan_id': policy['source_plan']['plan_id'],
    'num_jobs_checked': policy['freeze_attestation']['num_jobs_checked'],
    'frozen_utc': policy['frozen_utc'],
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
