#!/usr/bin/env python3
"""Bind a frozen candidate-K routing template to an unstarted compiled plan.

The source plan ID necessarily depends on the repository revision that creates
it.  This deterministic finalization step fills only plan/provenance fields;
all scientific gates live in the earlier frozen template.  It refuses any
plan whose artifact directories already contain run state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.compile_experiment_matrix import (  # noqa: E402
  DEFAULT_MANIFEST,
  _canonical_json,
  _git_metadata,
  build_jobs,
  load_and_validate_manifest,
  sha256_file,
)


DEFAULT_TEMPLATE = (
  REPO_ROOT / 'configs/experiment'
  / 'contextual-forest-k128-promotion-policy-template.yaml')
STRICT_UTC_TIMESTAMP = re.compile(
  r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
  r'(?:\.\d{1,6})?(?:Z|\+00:00)$')


def canonical_sha256(value: object) -> str:
  return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _exact_keys(value: object, expected: Iterable[str], *, context: str):
  if not isinstance(value, Mapping):
    raise TypeError(f'{context} must be a mapping')
  expected = set(expected)
  observed = set(value)
  if observed != expected:
    raise ValueError(
      f'{context} schema mismatch: missing={sorted(expected - observed)}, '
      f'unknown={sorted(observed - expected)}')
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


def _finite(value: object, *, context: str, minimum=None, maximum=None):
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
  if not isinstance(value, str) or not STRICT_UTC_TIMESTAMP.fullmatch(value):
    raise ValueError(
      f'{context} must be a complete ISO-8601 UTC timestamp ending in '
      "'Z' or '+00:00'")
  try:
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
  except ValueError as error:
    raise ValueError(f'{context} must be an ISO-8601 UTC timestamp') from error
  if parsed.utcoffset() != dt.timedelta(0):
    raise ValueError(f'{context} must be UTC')
  return parsed


def _validate_template(
    template: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  template = _exact_keys(template, {
    'schema_version', 'artifact', 'policy_id', 'policy_status',
    'template_frozen_utc', 'protocol_id', 'source_manifest_sha256',
    'source_suite', 'analysis_contract', 'confirmation_gates',
    'k128_support_sufficiency', 'routing',
  }, context='policy template')
  expected_identity = {
    'schema_version': 1,
    'artifact': 'candidate_k_promotion_policy_template',
    'policy_status': 'frozen_template_before_k128_plan_or_results',
    'source_suite': 'candidate_k_128_pilot',
  }
  for field, expected in expected_identity.items():
    if template[field] != expected:
      raise ValueError(f'policy template {field} must equal {expected!r}')
  _timestamp(
    template['template_frozen_utc'], context='template.template_frozen_utc')
  manifest = load_and_validate_manifest(manifest_path, repo_root=repo_root)
  if template['protocol_id'] != manifest['protocol_id']:
    raise ValueError('template protocol differs from manifest')
  if template['source_manifest_sha256'] != sha256_file(manifest_path):
    raise ValueError('template manifest hash differs from manifest bytes')
  _lower_hex(
    template['source_manifest_sha256'], 64,
    context='template.source_manifest_sha256')
  suite = manifest['suites'][template['source_suite']]
  contract = _exact_keys(template['analysis_contract'], {
    'analysis_schema_version', 'analysis_artifact', 'objective',
    'comparison', 'baseline_arm', 'treatment_arm', 'candidate_k',
    'train_seeds', 'corruption_seeds', 'datasets', 'mask_rates',
    'expected_num_strata', 'bootstrap',
  }, context='template.analysis_contract')
  if (contract['analysis_schema_version'] != 2
      or contract['analysis_artifact']
      != 'hierarchical_conditional_denoising_analysis'
      or contract['objective']
      != 'conditional_denoising_nll_per_masked_token'):
    raise ValueError('template analysis artifact/objective is unsupported')
  expected_grid = {
    'candidate_k': suite['candidate_ks'][0],
    'train_seeds': suite['train_seeds'],
    'corruption_seeds': suite['corruption_seeds'],
    'mask_rates': suite['mask_rates'],
  }
  for field, expected in expected_grid.items():
    if contract[field] != expected:
      raise ValueError(f'template {field} differs from source suite')
  comparison = manifest['evaluation']['comparisons'][contract['comparison']]
  if (contract['baseline_arm'] != comparison['baseline']
      or contract['treatment_arm'] != comparison['treatment']):
    raise ValueError('template comparison arms differ from manifest')
  expected_datasets = {}
  for alias in suite['datasets']:
    data_name = manifest['datasets'][alias]['data_config']
    data = _read_mapping(
      repo_root / 'configs/data' / f'{data_name}.yaml',
      context=f'data config {data_name}')
    expected_datasets[data['valid']] = {'revision': data['valid_revision']}
  if dict(contract['datasets']) != expected_datasets:
    raise ValueError('template datasets differ from source suite')
  if contract['expected_num_strata'] != (
      len(expected_datasets) * len(contract['mask_rates'])):
    raise ValueError('template expected_num_strata is inconsistent')
  bootstrap = _exact_keys(contract['bootstrap'], {
    'method', 'num_resamples', 'rng_seed', 'confidence_level',
  }, context='template.analysis_contract.bootstrap')
  if (bootstrap['method'] != 'hierarchical_paired_percentile_bootstrap'
      or bootstrap['num_resamples'] != manifest['analysis']['bootstrap_resamples']
      or bootstrap['rng_seed']
      != manifest['analysis']['bootstrap_seed'] + contract['candidate_k']
      or float(bootstrap['confidence_level'])
      != float(manifest['analysis']['confidence_level'])):
    raise ValueError('template bootstrap differs from aggregate contract')

  confirmation = _exact_keys(template['confirmation_gates'], {
    'pooled', 'corpus_breadth', 'high_mask_robustness',
  }, context='template.confirmation_gates')
  pooled = _exact_keys(confirmation['pooled'], {
    'min_mean_improvement', 'ci_lower_strictly_above',
  }, context='template.confirmation_gates.pooled')
  for field in pooled:
    _finite(pooled[field], context=f'template pooled {field}')
  breadth = _exact_keys(confirmation['corpus_breadth'], {
    'min_positive_dataset_means',
  }, context='template.confirmation_gates.corpus_breadth')
  if not 1 <= _positive_int(
      breadth['min_positive_dataset_means'], context='corpus breadth') \
      <= len(expected_datasets):
    raise ValueError('template corpus breadth exceeds dataset count')
  high = _exact_keys(confirmation['high_mask_robustness'], {
    'mask_rates', 'min_pooled_mean_improvement',
    'min_positive_dataset_means', 'min_worst_dataset_mean_improvement',
  }, context='template.confirmation_gates.high_mask_robustness')
  if (not isinstance(high['mask_rates'], list) or not high['mask_rates']
      or not set(high['mask_rates']).issubset(set(contract['mask_rates']))):
    raise ValueError('template high-mask rates are invalid')
  for field in ('min_pooled_mean_improvement',
                'min_worst_dataset_mean_improvement'):
    _finite(high[field], context=f'template high-mask {field}')
  if not 1 <= _positive_int(
      high['min_positive_dataset_means'], context='high-mask breadth') \
      <= len(expected_datasets):
    raise ValueError('template high-mask breadth exceeds dataset count')
  support = _exact_keys(template['k128_support_sufficiency'], {
    'arm', 'aggregation', 'min_overall_candidate_recall',
    'min_high_mask_dataset_candidate_recall',
    'min_overall_retained_unary_mass',
    'min_high_mask_dataset_retained_unary_mass',
  }, context='template.k128_support_sufficiency')
  if (support['arm'] != contract['treatment_arm']
      or support['aggregation'] != 'equal_weight_analysis_strata'):
    raise ValueError('template support aggregation is invalid')
  for field in (
      'min_overall_candidate_recall',
      'min_high_mask_dataset_candidate_recall',
      'min_overall_retained_unary_mass',
      'min_high_mask_dataset_retained_unary_mass'):
    _finite(support[field], context=f'template support {field}',
            minimum=0.0, maximum=1.0)
  expected_routing = {
    'allow_parallel_routes': True,
    'confirmation': {
      'target_suite': 'candidate_k_128_confirmation',
      'requires': [
        'integrity_complete', 'pooled_mean', 'pooled_ci',
        'corpus_breadth', 'high_mask_pooled', 'high_mask_breadth',
        'high_mask_no_material_regression',
      ],
    },
    'k256': {
      'target_suite': 'candidate_k_256_pilot',
      'requires': ['integrity_complete', 'k128_support_limited'],
    },
  }
  if dict(template['routing']) != expected_routing:
    raise ValueError('template routing differs from supported K128 routes')
  for route in ('confirmation', 'k256'):
    target = template['routing'][route]['target_suite']
    if manifest['suites'][target]['promotion_from'] != template['source_suite']:
      raise ValueError(f'route target {target} has wrong promotion parent')
  return dict(template)


def finalize_policy(
    template_path: Path,
    plan_dir: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    frozen_utc: str | None = None,
) -> dict[str, Any]:
  template_path = template_path.expanduser().resolve()
  manifest_path = manifest_path.expanduser().resolve()
  plan_dir = plan_dir.expanduser().resolve()
  template = _validate_template(
    _read_mapping(template_path, context='policy template'),
    manifest_path=manifest_path)
  manifest = load_and_validate_manifest(manifest_path, repo_root=REPO_ROOT)
  plan_path = plan_dir / 'compiled-plan.json'
  plan = _exact_keys(_read_mapping(plan_path, context='compiled plan'), {
    'schema_version', 'protocol_id', 'source_manifest_sha256', 'repository',
    'artifact_root', 'selected_suites', 'promotion_evidence', 'plan_id',
    'manifest_protocol_status', 'scientific_scope', 'job_counts',
    'num_jobs', 'job_ids', 'job_spec_sha256',
  }, context='compiled plan')
  expected_plan = {
    'schema_version': 2,
    'protocol_id': template['protocol_id'],
    'source_manifest_sha256': template['source_manifest_sha256'],
    'selected_suites': [template['source_suite']],
  }
  for field, expected in expected_plan.items():
    if plan[field] != expected:
      raise ValueError(f'compiled plan {field} differs from template')
  repository = _exact_keys(
    plan['repository'], {'sha', 'dirty'}, context='compiled plan repository')
  repository_sha = _lower_hex(
    repository['sha'], 40, context='compiled plan repository SHA')
  if repository['dirty'] is not False:
    raise ValueError('compiled plan repository was not clean')
  checkout = _git_metadata(REPO_ROOT)
  if checkout != {'sha': repository_sha, 'dirty': False}:
    raise ValueError(
      'policy finalization checkout differs from the clean compiled-plan '
      f'repository: expected {repository_sha}, found {checkout}')
  plan_id = _lower_hex(plan['plan_id'], 64, context='compiled plan ID')
  plan_identity = {
    'protocol_id': plan['protocol_id'],
    'source_manifest_sha256': plan['source_manifest_sha256'],
    'repository': dict(repository),
    'artifact_root': plan['artifact_root'],
    'selected_suites': plan['selected_suites'],
    'promotion_evidence': plan['promotion_evidence'],
  }
  if canonical_sha256(plan_identity) != plan_id:
    raise ValueError('compiled plan ID is not its canonical identity hash')
  evidence = _exact_keys(
    plan['promotion_evidence'], {template['source_suite']},
    context='compiled plan promotion evidence')
  parent = _exact_keys(evidence[template['source_suite']], {
    'path', 'sha256', 'source_suite', 'route_name',
    'canonical_decision_sha256', 'source_compiled_plan_sha256',
  }, context='compiled plan parent evidence')
  if parent['source_suite'] != 'pilot' or parent['route_name'] != 'k128':
    raise ValueError('K128 plan lacks the canonical pilot K128 route')
  for field in (
      'sha256', 'canonical_decision_sha256',
      'source_compiled_plan_sha256'):
    _lower_hex(parent[field], 64, context=f'parent evidence {field}')
  parent_path = Path(str(parent['path'])).expanduser().resolve()
  if not parent_path.is_file() or sha256_file(parent_path) != parent['sha256']:
    raise ValueError('compiled plan parent evidence file is unavailable or changed')
  from scripts.evaluate_experiment_promotion import (  # pylint: disable=import-outside-toplevel
    DEFAULT_POLICY,
    verify_compiler_evidence,
  )
  verified_parent = verify_compiler_evidence(
    _read_mapping(parent_path, context='parent promotion evidence'),
    evidence_path=parent_path,
    promoted_suite=template['source_suite'],
    manifest_path=manifest_path,
    trusted_policy_path=DEFAULT_POLICY,
    repo_root=REPO_ROOT)
  if (verified_parent['route_name'] != parent['route_name']
      or verified_parent['commitments']['canonical_decision_sha256']
      != parent['canonical_decision_sha256']
      or verified_parent['commitments']['source_compiled_plan_sha256']
      != parent['source_compiled_plan_sha256']):
    raise ValueError('compiled plan parent evidence commitments differ')

  expected_jobs = build_jobs(
    manifest,
    selected_suites=[template['source_suite']],
    artifact_root=Path(plan['artifact_root']).expanduser().resolve(),
    source_manifest_sha256=plan['source_manifest_sha256'],
    source_repository_sha=repository_sha,
    plan_id=plan_id)
  expected_counts: dict[str, int] = {}
  for job in expected_jobs.values():
    expected_counts[job['kind']] = expected_counts.get(job['kind'], 0) + 1
  job_ids = plan['job_ids']
  digests = plan['job_spec_sha256']
  if (job_ids != list(expected_jobs)
      or not isinstance(digests, Mapping)
      or dict(digests) != {
        job_id: canonical_sha256(job)
        for job_id, job in expected_jobs.items()}
      or plan['num_jobs'] != len(expected_jobs)
      or plan['job_counts'] != dict(sorted(expected_counts.items()))
      or plan['manifest_protocol_status'] != manifest['protocol_status']
      or plan['scientific_scope'] != manifest['scientific_scope']):
    raise ValueError(
      'compiled plan does not match the exact frozen K128 factorial')
  checked_artifact_dirs = []
  for job_id in job_ids:
    job_path = plan_dir / 'jobs' / f'{job_id}.json'
    job = _read_mapping(job_path, context=f'job {job_id}')
    if job != expected_jobs[job_id]:
      raise ValueError(
        f'job {job_id} differs from the exact frozen K128 factorial')
    artifact_dir = Path(str(job.get('artifact_dir'))).expanduser().resolve()
    artifact_root = Path(plan['artifact_root']).expanduser().resolve()
    try:
      artifact_dir.relative_to(artifact_root)
    except ValueError as error:
      raise ValueError(f'job {job_id} artifact directory escapes root') \
        from error
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
      raise RuntimeError(
        f'cannot freeze policy after source-suite execution began: '
        f'{artifact_dir} is non-empty')
    checked_artifact_dirs.append(str(artifact_dir))

  freeze_time = frozen_utc or dt.datetime.now(dt.timezone.utc).isoformat()
  parsed = _timestamp(freeze_time, context='frozen_utc')
  template_frozen = _timestamp(
    template['template_frozen_utc'], context='template.template_frozen_utc')
  if parsed < template_frozen:
    raise ValueError('frozen_utc cannot predate the policy template')
  contract = {
    **dict(template['analysis_contract']),
    'source_plan_id': plan_id,
    'source_compiled_plan_sha256': sha256_file(plan_path),
    'source_repository_sha': repository_sha,
    'source_repository_clean': True,
  }
  return {
    'schema_version': 1,
    'artifact': 'candidate_k_promotion_policy',
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
      'plan_id': plan_id,
      'source_repository_sha': repository_sha,
      'source_repository_clean': True,
      'promotion_evidence': dict(plan['promotion_evidence']),
      'job_spec_commitment_sha256': canonical_sha256(dict(digests)),
    },
    'freeze_attestation': {
      'status': 'no_source_suite_artifact_directory_was_nonempty',
      'num_jobs_checked': len(job_ids),
      'job_ids_sha256': canonical_sha256(job_ids),
      'artifact_dirs_sha256': canonical_sha256(sorted(checked_artifact_dirs)),
    },
    'confirmation_gates': dict(template['confirmation_gates']),
    'k128_support_sufficiency': dict(
      template['k128_support_sufficiency']),
    'routing': dict(template['routing']),
  }


def _exclusive_write_json(path: Path, payload: object) -> None:
  """Atomically publish complete JSON without ever replacing a destination."""
  path = path.expanduser().resolve()
  if path.exists():
    raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    descriptor = os.open(
      temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, 'w') as handle:
      handle.write(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
      handle.flush()
      os.fsync(handle.fileno())
    # A hard-link publish is atomic and fails with EEXIST if another process
    # creates the destination after the check above. Unlike os.replace, it can
    # never overwrite that competing artifact.
    os.link(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
  """Backward-compatible name for exclusive candidate-policy publication."""
  _exclusive_write_json(path, payload)


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
  _atomic_write(args.output, policy)
  print(json.dumps({
    'event': 'candidate_k_policy_frozen',
    'output': str(args.output.expanduser().resolve()),
    'policy_id': policy['policy_id'],
    'source_plan_id': policy['source_plan']['plan_id'],
    'num_jobs_checked': policy['freeze_attestation']['num_jobs_checked'],
    'frozen_utc': policy['frozen_utc'],
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
