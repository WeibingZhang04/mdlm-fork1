#!/usr/bin/env python3
"""Evaluate the frozen K=128 pilot and emit revision-bound route evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.compile_experiment_matrix import (  # noqa: E402
  DEFAULT_MANIFEST,
  SLUG_PATTERN,
  _git_metadata,
  load_and_validate_manifest,
  sha256_file,
)
from scripts.evaluate_experiment_promotion import (  # noqa: E402
  _artifact_path,
  _exact_keys,
  _lower_hex,
  _read_mapping,
  _timestamp,
  _utc_now,
  _validate_analysis_contract,
  _verify_authoritative_analysis,
  build_compiler_evidence,
  canonical_sha256,
  verify_compiler_evidence as verify_pilot_compiler_evidence,
)
from scripts.finalize_candidate_k_policy import (  # noqa: E402
  DEFAULT_TEMPLATE,
  _exclusive_write_json,
  _validate_template,
)
from scripts.run_compiled_job import SUCCESS_MARKER  # noqa: E402


def _validate_source_plan(
    policy: Mapping[str, Any],
    source_plan: object,
    *,
    source_plan_sha256: str,
) -> dict[str, Any]:
  plan = _exact_keys(source_plan, {
    'schema_version', 'protocol_id', 'source_manifest_sha256', 'repository',
    'artifact_root', 'selected_suites', 'promotion_evidence', 'plan_id',
    'manifest_protocol_status', 'scientific_scope', 'job_counts',
    'num_jobs', 'job_ids', 'job_spec_sha256',
  }, context='candidate-K source compiled plan')
  contract = policy['analysis_contract']
  expected = {
    'schema_version': 2,
    'protocol_id': policy['protocol_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'selected_suites': [policy['source_suite']],
    'plan_id': contract['source_plan_id'],
  }
  for field, value in expected.items():
    if plan[field] != value:
      raise ValueError(f'candidate-K source plan {field} differs from policy')
  if source_plan_sha256 != contract['source_compiled_plan_sha256']:
    raise ValueError('candidate-K source plan bytes differ from policy')
  repository = _exact_keys(
    plan['repository'], {'sha', 'dirty'}, context='source plan repository')
  if dict(repository) != {
      'sha': contract['source_repository_sha'], 'dirty': False}:
    raise ValueError('candidate-K source repository differs from policy')
  plan_identity = {
    'protocol_id': plan['protocol_id'],
    'source_manifest_sha256': plan['source_manifest_sha256'],
    'repository': dict(repository),
    'artifact_root': plan['artifact_root'],
    'selected_suites': plan['selected_suites'],
    'promotion_evidence': plan['promotion_evidence'],
  }
  if canonical_sha256(plan_identity) != plan['plan_id']:
    raise ValueError('candidate-K source plan ID is not canonical')
  job_ids = plan['job_ids']
  job_digests = plan['job_spec_sha256']
  if (not isinstance(job_ids, list) or not job_ids
      or len(job_ids) != len(set(job_ids))
      or not isinstance(job_digests, Mapping)
      or set(job_digests) != set(job_ids)
      or plan['num_jobs'] != len(job_ids)):
    raise ValueError('candidate-K source job commitments are invalid')
  for job_id, digest in job_digests.items():
    if not isinstance(job_id, str) or not SLUG_PATTERN.fullmatch(job_id):
      raise ValueError('candidate-K source job ID is invalid')
    _lower_hex(digest, 64, context=f'job digest {job_id}')
  source_info = policy['source_plan']
  if (source_info['file_sha256'] != source_plan_sha256
      or source_info['plan_id'] != plan['plan_id']
      or source_info['source_repository_sha'] != repository['sha']
      or source_info['source_repository_clean'] is not True
      or source_info['promotion_evidence'] != plan['promotion_evidence']
      or source_info['job_spec_commitment_sha256']
      != canonical_sha256(dict(job_digests))):
    raise ValueError('candidate-K source_plan policy block differs from plan')
  return dict(plan)


def load_and_validate_candidate_k_policy(
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
    _read_mapping(template_path, context='candidate-K policy template'),
    manifest_path=manifest_path,
    repo_root=repo_root)
  policy = _exact_keys(_read_mapping(policy_path, context='candidate-K policy'), {
    'schema_version', 'artifact', 'policy_id', 'policy_status', 'template',
    'frozen_utc', 'protocol_id', 'source_manifest_sha256', 'source_suite',
    'analysis_contract', 'source_plan', 'freeze_attestation',
    'confirmation_gates', 'k128_support_sufficiency', 'routing',
  }, context='candidate-K policy')
  expected_identity = {
    'schema_version': 1,
    'artifact': 'candidate_k_promotion_policy',
    'policy_id': template['policy_id'],
    'policy_status': 'frozen_before_source_suite_results',
    'protocol_id': template['protocol_id'],
    'source_manifest_sha256': template['source_manifest_sha256'],
    'source_suite': template['source_suite'],
  }
  for field, value in expected_identity.items():
    if policy[field] != value:
      raise ValueError(f'candidate-K policy {field} differs from template')
  template_identity = _exact_keys(policy['template'], {
    'path', 'sha256', 'frozen_utc',
  }, context='candidate-K policy template identity')
  if (template_identity['sha256'] != sha256_file(template_path)
      or template_identity['frozen_utc'] != template['template_frozen_utc']):
    raise ValueError('candidate-K policy does not bind the trusted template')
  _timestamp(policy['frozen_utc'], context='candidate-K policy frozen_utc')
  if _timestamp(policy['frozen_utc'], context='candidate-K policy frozen_utc') < \
      _timestamp(template['template_frozen_utc'], context='template frozen_utc'):
    raise ValueError('candidate-K concrete policy predates its template')
  if (policy['confirmation_gates'] != template['confirmation_gates']
      or policy['k128_support_sufficiency']
      != template['k128_support_sufficiency']
      or policy['routing'] != template['routing']):
    raise ValueError('candidate-K policy scientific gates differ from template')
  contract = _exact_keys(policy['analysis_contract'], {
    *template['analysis_contract'].keys(),
    'source_plan_id', 'source_compiled_plan_sha256',
    'source_repository_sha', 'source_repository_clean',
  }, context='candidate-K policy analysis contract')
  for field, value in template['analysis_contract'].items():
    if contract[field] != value:
      raise ValueError(
        f'candidate-K analysis contract {field} differs from template')
  _lower_hex(contract['source_plan_id'], 64, context='source plan ID')
  _lower_hex(
    contract['source_compiled_plan_sha256'], 64,
    context='source compiled plan SHA256')
  _lower_hex(
    contract['source_repository_sha'], 40,
    context='source repository SHA')
  if contract['source_repository_clean'] is not True:
    raise ValueError('candidate-K source repository must be clean')
  source_info = _exact_keys(policy['source_plan'], {
    'path', 'file_sha256', 'plan_id', 'source_repository_sha',
    'source_repository_clean', 'promotion_evidence',
    'job_spec_commitment_sha256',
  }, context='candidate-K policy source_plan')
  plan_path = Path(str(source_info['path'])).expanduser().resolve()
  if not plan_path.is_file() or plan_path.name != 'compiled-plan.json':
    raise FileNotFoundError('candidate-K policy source plan is unavailable')
  plan_sha = sha256_file(plan_path)
  source_plan = _read_mapping(plan_path, context='candidate-K source plan')
  validated_plan = _validate_source_plan(
    policy, source_plan, source_plan_sha256=plan_sha)

  attestation = _exact_keys(policy['freeze_attestation'], {
    'status', 'num_jobs_checked', 'job_ids_sha256',
    'artifact_dirs_sha256',
  }, context='candidate-K freeze attestation')
  if (attestation['status']
      != 'no_source_suite_artifact_directory_was_nonempty'
      or attestation['num_jobs_checked'] != validated_plan['num_jobs']
      or attestation['job_ids_sha256']
      != canonical_sha256(validated_plan['job_ids'])):
    raise ValueError('candidate-K freeze attestation differs from source plan')
  artifact_dirs = []
  plan_dir = plan_path.parent
  for job_id in validated_plan['job_ids']:
    job = _read_mapping(
      plan_dir / 'jobs' / f'{job_id}.json', context=f'job {job_id}')
    artifact_dirs.append(str(Path(job['artifact_dir']).expanduser().resolve()))
  if attestation['artifact_dirs_sha256'] != \
      canonical_sha256(sorted(artifact_dirs)):
    raise ValueError('candidate-K freeze artifact-directory commitment differs')

  # Revalidate the K64 decision that authorized creation of this source plan.
  promotion = _exact_keys(
    validated_plan['promotion_evidence'], {policy['source_suite']},
    context='candidate-K parent promotion evidence')
  parent = _exact_keys(promotion[policy['source_suite']], {
    'path', 'sha256', 'source_suite', 'route_name',
    'canonical_decision_sha256', 'source_compiled_plan_sha256',
  }, context='candidate-K parent promotion evidence record')
  if parent['source_suite'] != 'pilot' or parent['route_name'] != 'k128':
    raise ValueError('candidate-K parent evidence is not the K64 K128 route')
  for field in (
      'sha256', 'canonical_decision_sha256',
      'source_compiled_plan_sha256'):
    _lower_hex(parent[field], 64, context=f'parent evidence {field}')
  parent_path = Path(parent['path']).expanduser().resolve()
  if sha256_file(parent_path) != parent['sha256']:
    raise ValueError('candidate-K parent promotion evidence changed')
  verified_parent = verify_pilot_compiler_evidence(
    _read_mapping(parent_path, context='parent K64 promotion evidence'),
    evidence_path=parent_path,
    promoted_suite=policy['source_suite'],
    manifest_path=manifest_path,
    repo_root=repo_root)
  if (verified_parent['source_suite'] != parent['source_suite']
      or verified_parent['route_name'] != parent['route_name']
      or verified_parent['commitments']['canonical_decision_sha256']
      != parent['canonical_decision_sha256']
      or verified_parent['commitments']['source_compiled_plan_sha256']
      != parent['source_compiled_plan_sha256']):
    raise ValueError(
      'candidate-K source plan parent evidence commitments differ')
  load_and_validate_manifest(manifest_path, repo_root=repo_root)
  return dict(policy)


def _verify_policy_predates_source_jobs(
    policy: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    *,
    plan_dir: Path,
) -> None:
  frozen = _timestamp(policy['frozen_utc'], context='policy frozen_utc')
  for job_id in source_plan['job_ids']:
    job = _read_mapping(
      plan_dir / 'jobs' / f'{job_id}.json', context=f'job {job_id}')
    marker_path = Path(job['artifact_dir']).expanduser().resolve() / SUCCESS_MARKER
    marker = _read_mapping(marker_path, context=f'success marker {job_id}')
    started = _timestamp(
      marker.get('start_time_utc'), context=f'{job_id} start_time_utc')
    if started <= frozen:
      raise ValueError(
        f'candidate-K policy was not frozen before job {job_id} started')


def _require_clean_descendant_repository(
    source_repository_sha: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
  """Require replay code to be a clean descendant of the frozen source."""
  source_repository_sha = _lower_hex(
    source_repository_sha, 40, context='source repository SHA')
  repo_root = repo_root.expanduser().resolve()
  current = _git_metadata(repo_root)
  if current.get('dirty') is not False:
    raise ValueError(
      'historical candidate replay requires a clean current repository')
  current_sha = _lower_hex(
    current.get('sha'), 40, context='current repository SHA')
  check = subprocess.run(
    ['git', 'merge-base', '--is-ancestor', source_repository_sha, current_sha],
    cwd=repo_root,
    check=False,
    capture_output=True,
    text=True)
  if check.returncode != 0:
    raise ValueError(
      'historical candidate replay requires a current checkout descended '
      f'from source revision {source_repository_sha}')


def _evaluate_candidate_k_analysis_core(
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
    authoritative_verifier: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
  _lower_hex(policy_sha256, 64, context='policy_sha256')
  _lower_hex(analysis_sha256, 64, context='analysis_sha256')
  source_plan_path = source_plan_path.expanduser().resolve()
  if source_plan_path != Path(
      str(policy['source_plan']['path'])).expanduser().resolve():
    raise ValueError('candidate-K source plan path differs from policy')
  validated_plan = _validate_source_plan(
    policy, source_plan, source_plan_sha256=source_plan_sha256)
  conditions, diagnostics = _validate_analysis_contract(policy, analysis)
  verified_integrity = authoritative_verifier(
    policy,
    analysis,
    source_plan=source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=source_plan_sha256,
    manifest_path=manifest_path)
  _verify_policy_predates_source_jobs(
    policy, validated_plan, plan_dir=source_plan_path.parent)
  decision_time = created_utc or _utc_now()
  if _timestamp(decision_time, context='decision created_utc') < \
      _timestamp(analysis['created_utc'], context='analysis created_utc'):
    raise ValueError('candidate-K routing decision predates analysis')

  contract = policy['analysis_contract']
  confirmation = policy['confirmation_gates']
  datasets = list(contract['datasets'])
  rates = [float(rate) for rate in contract['mask_rates']]
  high_rates = {
    float(rate) for rate in confirmation['high_mask_robustness']['mask_rates']}
  dataset_means = {
    dataset: math.fsum(
      conditions[(dataset, rate)]['mean_improvement'] for rate in rates)
    / len(rates)
    for dataset in datasets}
  high_dataset_means = {
    dataset: math.fsum(
      conditions[(dataset, rate)]['mean_improvement'] for rate in high_rates)
    / len(high_rates)
    for dataset in datasets}
  high_pooled = math.fsum(high_dataset_means.values()) / len(datasets)
  pooled = analysis['by_candidate_k'][str(contract['candidate_k'])]['pooled']
  pooled_cfg = confirmation['pooled']
  breadth_cfg = confirmation['corpus_breadth']
  high_cfg = confirmation['high_mask_robustness']
  gate_states = {
    'integrity_complete': True,
    'pooled_mean': float(pooled['mean_improvement'])
      >= float(pooled_cfg['min_mean_improvement']),
    'pooled_ci': float(pooled['ci_lower'])
      > float(pooled_cfg['ci_lower_strictly_above']),
    'corpus_breadth': sum(value > 0.0 for value in dataset_means.values())
      >= breadth_cfg['min_positive_dataset_means'],
    'high_mask_pooled': high_pooled
      >= float(high_cfg['min_pooled_mean_improvement']),
    'high_mask_breadth': sum(
      value > 0.0 for value in high_dataset_means.values())
      >= high_cfg['min_positive_dataset_means'],
    'high_mask_no_material_regression': min(high_dataset_means.values())
      >= float(high_cfg['min_worst_dataset_mean_improvement']),
  }
  support_cfg = policy['k128_support_sufficiency']
  treatment = support_cfg['arm']
  overall_candidate = math.fsum(
    diagnostics[key]['arms'][treatment]['candidate_recall']
    for key in diagnostics) / len(diagnostics)
  overall_mass = math.fsum(
    diagnostics[key]['arms'][treatment]['retained_unary_mass']
    for key in diagnostics) / len(diagnostics)
  high_candidate = {
    dataset: math.fsum(
      diagnostics[(dataset, rate)]['arms'][treatment]['candidate_recall']
      for rate in high_rates) / len(high_rates)
    for dataset in datasets}
  high_mass = {
    dataset: math.fsum(
      diagnostics[(dataset, rate)]['arms'][treatment]['retained_unary_mass']
      for rate in high_rates) / len(high_rates)
    for dataset in datasets}
  support_criteria = {
    'overall_candidate_recall': overall_candidate
      >= support_cfg['min_overall_candidate_recall'],
    'high_mask_dataset_candidate_recall': min(high_candidate.values())
      >= support_cfg['min_high_mask_dataset_candidate_recall'],
    'overall_retained_unary_mass': overall_mass
      >= support_cfg['min_overall_retained_unary_mass'],
    'high_mask_dataset_retained_unary_mass': min(high_mass.values())
      >= support_cfg['min_high_mask_dataset_retained_unary_mass'],
  }
  support_sufficient = all(support_criteria.values())
  gate_states['k128_support_sufficient'] = support_sufficient
  gate_states['k128_support_limited'] = not support_sufficient
  routes = {}
  for route_name in ('confirmation', 'k256'):
    route = policy['routing'][route_name]
    criteria = {name: gate_states[name] for name in route['requires']}
    routes[route_name] = {
      'target_suite': route['target_suite'],
      'promote': all(criteria.values()),
      'criteria': criteria,
    }
  confirmation_route = routes['confirmation']['promote']
  k256_route = routes['k256']['promote']
  if confirmation_route and k256_route:
    outcome = 'promote_k128_confirmation_and_k256'
  elif confirmation_route:
    outcome = 'promote_k128_confirmation_only'
  elif k256_route:
    outcome = 'promote_k256_only'
  else:
    outcome = 'stop_after_k128_pilot'
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
      'observed': sum(value > 0.0 for value in dataset_means.values()),
      'operator': '>=',
      'threshold': breadth_cfg['min_positive_dataset_means'],
    },
    'high_mask_pooled': {
      'passed': gate_states['high_mask_pooled'],
      'observed': high_pooled,
      'operator': '>=',
      'threshold': float(high_cfg['min_pooled_mean_improvement']),
    },
    'high_mask_breadth': {
      'passed': gate_states['high_mask_breadth'],
      'observed': sum(value > 0.0 for value in high_dataset_means.values()),
      'operator': '>=',
      'threshold': high_cfg['min_positive_dataset_means'],
    },
    'high_mask_no_material_regression': {
      'passed': gate_states['high_mask_no_material_regression'],
      'observed': min(high_dataset_means.values()),
      'operator': '>=',
      'threshold': float(high_cfg['min_worst_dataset_mean_improvement']),
    },
    'k128_support_sufficient': {
      'passed': support_sufficient,
      'criteria': support_criteria,
    },
    'k128_support_limited': {
      'passed': not support_sufficient,
      'definition': 'logical negation of k128_support_sufficient',
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
      'template_sha256': policy['template']['sha256'],
    },
    'analysis': {
      'analysis_sha256': analysis_sha256,
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
        'trusted_template_committed_in_source_repository': True,
        'declared_policy_timestamp_precedes_every_source_job': True,
        'identity_and_manifest_bound': True,
        'compiled_plan_bound': True,
        'parent_promotion_revalidated': True,
        'success_markers_and_outputs_bound': True,
        'analysis_recomputed_from_bound_records': True,
        'complete_factorial_grid': True,
        'bootstrap_contract_exact': True,
      },
    },
    'measurements': {
      'pooled': {
        'mean_improvement': float(pooled['mean_improvement']),
        'ci_lower': float(pooled['ci_lower']),
        'ci_upper': float(pooled['ci_upper']),
      },
      'dataset_mean_improvements': dataset_means,
      'high_mask_dataset_mean_improvements': high_dataset_means,
      'high_mask_pooled_mean_improvement': high_pooled,
      'k128_support': {
        'overall_candidate_recall': overall_candidate,
        'minimum_high_mask_dataset_candidate_recall': min(
          high_candidate.values()),
        'overall_retained_unary_mass': overall_mass,
        'minimum_high_mask_dataset_retained_unary_mass': min(
          high_mass.values()),
        'high_mask_candidate_recall_by_dataset': high_candidate,
        'high_mask_retained_unary_mass_by_dataset': high_mass,
      },
    },
    'gates': gates,
    'routes': routes,
    'outcome': outcome,
    'compiler_evidence': {
      name: {
        'eligible': route['promote'],
        'target_suite': route['target_suite'],
        'filename': f'{route["target_suite"]}-promotion.json',
      }
      for name, route in routes.items()
    },
  }


def evaluate_candidate_k_analysis(
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
  """Evaluate candidate K only from its exact current source checkout."""
  return _evaluate_candidate_k_analysis_core(
    policy,
    analysis,
    policy_sha256=policy_sha256,
    analysis_sha256=analysis_sha256,
    source_plan=source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=source_plan_sha256,
    manifest_path=manifest_path,
    created_utc=created_utc,
    authoritative_verifier=_verify_authoritative_analysis)


def _verify_historical_candidate_authoritative_analysis(
    policy: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    source_plan: Mapping[str, Any],
    source_plan_path: Path,
    source_plan_sha256: str,
    manifest_path: Path,
    trusted_policy_path: Path,
    trusted_analysis_path: Path,
    trusted_template_path: Path = DEFAULT_TEMPLATE,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  """Authenticate and replay one historical candidate-K source plan.

  This is the only caller allowed to bypass current-checkout equality. It
  independently reloads the policy, analysis, and plan and validates them
  against the repository's trusted candidate template and manifest before it
  reaches the private record-loader core. No generic verifier receives an
  opt-out flag.
  """
  trusted_template_path = trusted_template_path.expanduser().resolve()
  manifest_path = manifest_path.expanduser().resolve()
  repo_root = repo_root.expanduser().resolve()
  if trusted_template_path != DEFAULT_TEMPLATE.resolve():
    raise ValueError('historical candidate replay requires DEFAULT_TEMPLATE')
  if manifest_path != DEFAULT_MANIFEST.resolve():
    raise ValueError('historical candidate replay requires DEFAULT_MANIFEST')
  if repo_root != REPO_ROOT.resolve():
    raise ValueError('historical candidate replay requires the trusted repo')

  trusted_policy_path = trusted_policy_path.expanduser().resolve()
  trusted_analysis_path = trusted_analysis_path.expanduser().resolve()
  source_plan_path = source_plan_path.expanduser().resolve()
  validated_policy = load_and_validate_candidate_k_policy(
    trusted_policy_path,
    template_path=DEFAULT_TEMPLATE,
    manifest_path=DEFAULT_MANIFEST,
    repo_root=REPO_ROOT)
  if validated_policy != dict(policy):
    raise ValueError(
      'candidate-K policy mapping differs from the trusted policy artifact')
  trusted_analysis = _read_mapping(
    trusted_analysis_path, context='trusted candidate-K analysis')
  if trusted_analysis != dict(analysis):
    raise ValueError(
      'candidate-K analysis mapping differs from the trusted artifact')
  trusted_plan = _read_mapping(
    source_plan_path, context='trusted candidate-K source plan')
  if trusted_plan != dict(source_plan):
    raise ValueError(
      'candidate-K source-plan mapping differs from the trusted artifact')
  if sha256_file(source_plan_path) != source_plan_sha256:
    raise ValueError('candidate-K source-plan bytes changed before replay')
  if source_plan_path != Path(
      str(validated_policy['source_plan']['path'])).expanduser().resolve():
    raise ValueError('candidate-K source-plan path differs from trusted policy')
  validated_plan = _validate_source_plan(
    validated_policy, trusted_plan, source_plan_sha256=source_plan_sha256)
  _validate_analysis_contract(validated_policy, trusted_analysis)
  _require_clean_descendant_repository(
    validated_policy['analysis_contract']['source_repository_sha'],
    repo_root=REPO_ROOT)
  _verify_policy_predates_source_jobs(
    validated_policy, validated_plan, plan_dir=source_plan_path.parent)

  from scripts.aggregate_hierarchical_document_eval import (  # pylint: disable=import-outside-toplevel
    _load_plan_records_core,
    aggregate_records,
    bind_analysis_to_source,
  )
  contract = validated_policy['analysis_contract']
  records, context = _load_plan_records_core(
    source_plan_path.parent,
    manifest_path=DEFAULT_MANIFEST,
    suite_name=validated_policy['source_suite'],
    comparison_name=contract['comparison'],
    expected_legacy_plan_sha256=contract['source_compiled_plan_sha256'],
    expected_legacy_repository_sha=contract['source_repository_sha'],
    require_current_repository_match=False,
    repo_root=REPO_ROOT)
  comparison = context['comparison']
  base_seed = context['manifest']['analysis']['bootstrap_seed']
  expected_seed = base_seed + contract['candidate_k']
  if expected_seed != contract['bootstrap']['rng_seed']:
    raise ValueError(
      'candidate-K bootstrap seed binding differs from manifest base + K')
  recomputed = aggregate_records(
    records,
    baseline_arm=comparison['baseline'],
    treatment_arm=comparison['treatment'],
    protocol_id=validated_policy['protocol_id'],
    suite_name=validated_policy['source_suite'],
    comparison_name=contract['comparison'],
    num_resamples=contract['bootstrap']['num_resamples'],
    # aggregate_records domain-separates candidate supports by adding K.
    rng_seed=base_seed,
    confidence_level=contract['bootstrap']['confidence_level'],
    timestamp_utc=trusted_analysis['created_utc'])
  recomputed = bind_analysis_to_source(recomputed, context)
  if recomputed != trusted_analysis:
    raise ValueError(
      'analysis differs from deterministic recomputation of the '
      'historical marker-bound source records')
  return context['source_integrity']


def _evaluate_trusted_historical_candidate_k_analysis(
    *,
    policy_path: Path,
    analysis_path: Path,
    source_plan_path: Path,
    trusted_template_path: Path = DEFAULT_TEMPLATE,
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = REPO_ROOT,
    created_utc: str | None = None,
) -> dict[str, Any]:
  """Evaluate an authenticated candidate-K artifact from a descendant tree."""
  policy_path = policy_path.expanduser().resolve()
  analysis_path = analysis_path.expanduser().resolve()
  source_plan_path = source_plan_path.expanduser().resolve()
  policy = _read_mapping(policy_path, context='candidate-K policy')
  analysis = _read_mapping(analysis_path, context='candidate-K analysis')
  source_plan = _read_mapping(source_plan_path, context='candidate-K plan')

  def historical_verifier(*args, **kwargs):
    return _verify_historical_candidate_authoritative_analysis(
      *args,
      **kwargs,
      trusted_policy_path=policy_path,
      trusted_analysis_path=analysis_path,
      trusted_template_path=trusted_template_path,
      repo_root=repo_root)

  return _evaluate_candidate_k_analysis_core(
    policy,
    analysis,
    policy_sha256=sha256_file(policy_path),
    analysis_sha256=sha256_file(analysis_path),
    source_plan=source_plan,
    source_plan_path=source_plan_path,
    source_plan_sha256=sha256_file(source_plan_path),
    manifest_path=manifest_path,
    created_utc=created_utc,
    authoritative_verifier=historical_verifier)


def build_candidate_compiler_evidence(
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


def verify_candidate_compiler_evidence(
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
  }, context='candidate-K promotion evidence')
  if (evidence['schema_version'] != 3
      or evidence['artifact'] != 'experiment_suite_promotion_decision'
      or evidence['promoted_suite'] != promoted_suite
      or evidence['decision'] != 'promote'):
    raise ValueError('invalid candidate-K promotion evidence identity')
  commitments = _exact_keys(evidence['commitments'], {
    'policy_sha256', 'policy_template_sha256', 'analysis_sha256',
    'source_compiled_plan_sha256', 'source_plan_id',
    'source_repository_sha', 'source_repository_clean',
    'source_job_spec_commitment_sha256',
    'source_job_artifact_commitment_sha256',
    'canonical_decision_sha256',
  }, context='candidate-K evidence commitments')
  for field, value in commitments.items():
    if field == 'source_repository_clean':
      if value is not True:
        raise ValueError('candidate-K evidence source repository is not clean')
    elif field == 'source_repository_sha':
      _lower_hex(value, 40, context=field)
    else:
      _lower_hex(value, 64, context=field)
  artifacts = _exact_keys(evidence['artifacts'], {
    'policy_path', 'analysis_path', 'source_compiled_plan_path',
    'routing_decision_path',
  }, context='candidate-K evidence artifacts')
  policy_path = _artifact_path(artifacts['policy_path'], context='policy_path')
  analysis_path = _artifact_path(
    artifacts['analysis_path'], context='analysis_path')
  source_plan_path = _artifact_path(
    artifacts['source_compiled_plan_path'], context='source_plan_path')
  decision_path = _artifact_path(
    artifacts['routing_decision_path'], context='decision_path')
  if evidence_path.expanduser().resolve() in {
      policy_path, analysis_path, source_plan_path, decision_path}:
    raise ValueError('candidate-K promotion evidence self-references')
  if (commitments['policy_sha256'] != sha256_file(policy_path)
      or commitments['policy_template_sha256']
      != sha256_file(trusted_template_path)
      or commitments['analysis_sha256'] != sha256_file(analysis_path)
      or commitments['source_compiled_plan_sha256']
      != sha256_file(source_plan_path)):
    raise ValueError('candidate-K evidence artifact SHA mismatch')
  decision = _read_mapping(decision_path, context='candidate-K decision')
  if canonical_sha256(decision) != commitments['canonical_decision_sha256']:
    raise ValueError('candidate-K routing decision hash mismatch')
  canonical_decision = _evaluate_trusted_historical_candidate_k_analysis(
    policy_path=policy_path,
    analysis_path=analysis_path,
    source_plan_path=source_plan_path,
    trusted_template_path=trusted_template_path,
    manifest_path=manifest_path,
    repo_root=repo_root,
    created_utc=decision.get('created_utc'))
  if decision != canonical_decision:
    raise ValueError('candidate-K decision differs from deterministic reevaluation')
  expected = build_candidate_compiler_evidence(
    canonical_decision,
    str(evidence['route_name']),
    policy_path=policy_path,
    analysis_path=analysis_path,
    source_plan_path=source_plan_path,
    decision_path=decision_path)
  if dict(evidence) != expected:
    raise ValueError('candidate-K promotion evidence differs from canonical')
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


def _validate_publication_destinations(
    output: Path,
    evidence_paths: Mapping[str, Path],
) -> None:
  """Reject decision/evidence aliases before the first publication write."""
  destinations = {
    'decision': output.expanduser().resolve(),
    **{
      f'evidence:{route_name}': path.expanduser().resolve()
      for route_name, path in evidence_paths.items()},
  }
  owners_by_path: dict[Path, list[str]] = {}
  for owner, path in destinations.items():
    owners_by_path.setdefault(path, []).append(owner)
  aliases = {
    str(path): sorted(owners)
    for path, owners in owners_by_path.items() if len(owners) > 1}
  if aliases:
    raise ValueError(
      f'decision and compiler-evidence destinations must be pairwise '
      f'distinct: {aliases}')


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  policy_path = args.policy.expanduser().resolve()
  analysis_path = args.analysis.expanduser().resolve()
  source_plan_path = args.source_plan.expanduser().resolve()
  output = args.output.expanduser().resolve()
  if output.exists():
    raise FileExistsError(output)
  decision = _evaluate_trusted_historical_candidate_k_analysis(
    policy_path=policy_path,
    analysis_path=analysis_path,
    source_plan_path=source_plan_path,
    trusted_template_path=args.template,
    manifest_path=args.manifest)
  evidence_paths = {}
  if args.compiler_evidence_dir is not None:
    evidence_dir = args.compiler_evidence_dir.expanduser().resolve()
    for route_name, route in decision['routes'].items():
      if route['promote']:
        evidence_paths[route_name] = (
          evidence_dir / f'{route["target_suite"]}-promotion.json')
  _validate_publication_destinations(output, evidence_paths)
  for path in [output, *evidence_paths.values()]:
    if path.exists():
      raise FileExistsError(path)
  evidence_payloads = {
    route_name: build_candidate_compiler_evidence(
      decision,
      route_name,
      policy_path=policy_path,
      analysis_path=analysis_path,
      source_plan_path=source_plan_path,
      decision_path=output)
    for route_name in evidence_paths}
  _exclusive_write_json(output, decision)
  for route_name, payload in evidence_payloads.items():
    _exclusive_write_json(evidence_paths[route_name], payload)
  print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
  return 0 if any(route['promote'] for route in decision['routes'].values()) \
    else 2


if __name__ == '__main__':
  raise SystemExit(main())
