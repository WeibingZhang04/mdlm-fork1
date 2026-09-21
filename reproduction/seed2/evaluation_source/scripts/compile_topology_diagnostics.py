#!/usr/bin/env python3
"""Derive a runnable topology plan from the authenticated K=128 plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.topology_diagnostics import (  # noqa: E402
  _validate_trusted_protocol_path,
  read_protocol,
  sha256_file,
)
from scripts.compile_experiment_matrix import (  # noqa: E402
  DEFAULT_MANIFEST,
  PLAN_SCHEMA_VERSION,
  _canonical_json,
  _job,
  derive_matrix_plan,
  write_plan,
)
from scripts.run_compiled_job import (  # noqa: E402
  _load_plan,
  _validate_repository_checkout,
)


DEFAULT_PROTOCOL = (
  REPO_ROOT / 'configs' / 'evaluation'
  / 'contextual-forest-topology-diagnostics-v1.json')
SOURCE_SUITE = 'candidate_k_128_confirmation'
TOPOLOGY_SUITE = 'topology_diagnostics'


def _sha256_canonical(value: Any) -> str:
  return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _validate_source_promotion(plan: Mapping[str, Any]) -> dict[str, Any]:
  selected = plan.get('selected_suites')
  if not isinstance(selected, list) or SOURCE_SUITE not in selected:
    raise ValueError(
      f'source plan must contain the promoted {SOURCE_SUITE} suite')
  promotion = plan.get('promotion_evidence')
  if not isinstance(promotion, Mapping) or SOURCE_SUITE not in promotion:
    raise ValueError('source K=128 confirmation plan lacks promotion evidence')
  evidence = promotion[SOURCE_SUITE]
  expected_fields = {
    'path', 'sha256', 'source_suite', 'route_name',
    'canonical_decision_sha256', 'source_compiled_plan_sha256'}
  if not isinstance(evidence, Mapping) or set(evidence) != expected_fields:
    raise ValueError('source K=128 promotion evidence schema drifted')
  path = Path(evidence['path']).expanduser().resolve()
  if (not path.is_file() or sha256_file(path) != evidence['sha256']
      or evidence['source_suite'] != 'candidate_k_128_pilot'
      or evidence['route_name'] != 'confirmation'):
    raise ValueError('source K=128 promotion evidence is not authentic')
  return dict(evidence)


def _validate_canonical_source_plan(
    plan: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
) -> None:
  """Rebuild the promoted K=128 parent through the trusted compiler."""
  if plan.get('selected_suites') != [SOURCE_SUITE]:
    raise ValueError(
      f'topology parent must contain exactly the {SOURCE_SUITE} suite')
  promotion = _validate_source_promotion(plan)
  artifact_root = Path(plan.get('artifact_root', '')).expanduser().resolve()
  expected_plan, expected_jobs, expected_root = derive_matrix_plan(
    DEFAULT_MANIFEST,
    selected_suites=[SOURCE_SUITE],
    artifact_root_override=artifact_root,
    repo_root=REPO_ROOT,
    promotion_evidence={SOURCE_SUITE: Path(promotion['path'])})
  if expected_root != artifact_root:
    raise AssertionError('canonical parent compiler artifact root drifted')
  if set(jobs) != set(expected_jobs):
    raise ValueError(
      'K=128 topology parent job set differs from canonical compiler '
      'reconstruction')
  for job_id in expected_jobs:
    if _canonical_json(jobs[job_id]) != _canonical_json(
        expected_jobs[job_id]):
      raise ValueError(
        f'K=128 topology parent job {job_id!r} differs from canonical '
        'compiler reconstruction')
  if _canonical_json(plan) != _canonical_json(expected_plan):
    raise ValueError(
      'K=128 topology parent plan differs from canonical compiler '
      'reconstruction')


def _source_job_pair(
    jobs: Mapping[str, Mapping[str, Any]],
    *,
    train_seed: int,
    candidate_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
  train_id = (
    f'train--dynamic_dynamic--s{train_seed:03d}--k{candidate_k:03d}')
  export_id = (
    f'export--dynamic_dynamic--s{train_seed:03d}--k{candidate_k:03d}')
  if train_id not in jobs or export_id not in jobs:
    raise ValueError(
      f'source plan lacks K={candidate_k} adapter jobs for seed {train_seed}')
  train = dict(jobs[train_id])
  export = dict(jobs[export_id])
  if (train['kind'] != 'train' or export['kind'] != 'export'
      or export['dependencies'] != [train_id]
      or train['identity'].get('control') != 'dynamic_dynamic'
      or export['identity'].get('control') != 'dynamic_dynamic'
      or train['identity'].get('train_seed') != train_seed
      or export['identity'].get('train_seed') != train_seed
      or train['identity'].get('candidate_k') != candidate_k
      or export['identity'].get('candidate_k') != candidate_k
      or export['identity'].get('topology_mode') != 'dynamic'
      or export['identity'].get('factor_mode') != 'dynamic'
      or export['identity'].get('independent_mode') is not False):
    raise ValueError('source adapter job semantics differ from topology protocol')
  export_outputs = {
    output.get('name'): output.get('pattern')
    for output in export['required_outputs']}
  if export_outputs != {
      'adapter': 'adapter.safetensors',
      'adapter_manifest': 'adapter-manifest.json'}:
    raise ValueError('source adapter output contract drifted')
  backbone_inputs = [
    item for item in train['external_inputs']
    if item.get('role') == 'released_backbone_wrapper']
  if len(backbone_inputs) != 1:
    raise ValueError('source training job lacks one released backbone')
  return train, export


def derive_topology_plan(
    *,
    source_plan_dir: Path,
    output_dir: Path | None = None,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
  """Return the one canonical derived plan without writing any artifacts."""
  protocol_path = protocol_path.expanduser().resolve()
  protocol, protocol_sha = read_protocol(protocol_path)
  _validate_trusted_protocol_path(protocol_path, protocol)
  source_plan_dir = source_plan_dir.expanduser().resolve()
  source_plan_path = source_plan_dir / 'compiled-plan.json'
  source_plan, source_jobs = _load_plan(source_plan_dir)
  _validate_repository_checkout(source_plan, repo_root=REPO_ROOT)
  if source_plan.get('source_manifest_sha256') != sha256_file(
      DEFAULT_MANIFEST):
    raise ValueError(
      'source plan was not compiled from the repository-trusted manifest')
  _validate_canonical_source_plan(source_plan, source_jobs)
  source_promotion = _validate_source_promotion(source_plan)
  artifact_root = Path(source_plan['artifact_root']).expanduser().resolve()
  if output_dir is None:
    output_dir = artifact_root / 'plans' / TOPOLOGY_SUITE
  output_dir = output_dir.expanduser().resolve()
  try:
    output_dir.relative_to(artifact_root)
  except ValueError as error:
    raise ValueError('topology plan directory must remain in artifact_root') \
      from error
  if output_dir == artifact_root:
    raise ValueError('topology plan directory cannot equal artifact_root')

  plan_identity = {
    'protocol_id': source_plan['protocol_id'],
    'source_manifest_sha256': source_plan['source_manifest_sha256'],
    'repository': dict(source_plan['repository']),
    'artifact_root': str(artifact_root),
    'selected_suites': [TOPOLOGY_SUITE],
    'promotion_evidence': {SOURCE_SUITE: source_promotion},
    'topology_protocol': {
      'path': str(protocol_path),
      'protocol_id': protocol['protocol_id'],
      'canonical_sha256': protocol_sha,
      'file_sha256': sha256_file(protocol_path),
      'protocol_status': protocol['protocol_status'],
    },
    'source_compiled_plan': {
      'path': str(source_plan_path),
      'sha256': sha256_file(source_plan_path),
      'plan_id': source_plan['plan_id'],
    },
    'compiled_plan_dir': str(output_dir),
  }
  plan_id = _sha256_canonical(plan_identity)
  jobs: dict[str, dict[str, Any]] = {}
  source_pairs = {}
  for train_seed in protocol['source_selection']['train_seeds']:
    train, export = _source_job_pair(
      source_jobs,
      train_seed=train_seed,
      candidate_k=protocol['candidate_top_k'])
    for source_job in (train, export):
      copied = copy.deepcopy(source_job)
      copied['plan_id'] = plan_id
      copied['suites'] = [TOPOLOGY_SUITE]
      jobs[copied['job_id']] = copied
    source_pairs[train_seed] = (train, export)

  for dataset in sorted(protocol['source_selection']['datasets']):
    specification = protocol['source_selection']['datasets'][dataset]
    for train_seed in protocol['source_selection']['train_seeds']:
      train, export = source_pairs[train_seed]
      export_id = export['job_id']
      job_id = (
        f'topology--dynamic_dynamic--s{train_seed:03d}--{dataset}'
        f'--k{protocol["candidate_top_k"]:03d}')
      artifact_dir = artifact_root / 'runs' / job_id
      backbone = next(
        item for item in train['external_inputs']
        if item['role'] == 'released_backbone_wrapper')
      argv = [
        '{python}', 'scripts/run_topology_diagnostics.py',
        '--protocol', str(protocol_path),
        '--plan-dir', str(output_dir),
        '--job-id', job_id,
        '--backbone-checkpoint', backbone['path'],
        '--backbone-sha256', backbone['sha256'],
        '--adapter', f'${{artifact:{export_id}:adapter.safetensors}}',
        '--adapter-sha256',
        f'${{sha256:{export_id}:adapter.safetensors}}',
        '--adapter-manifest',
        f'${{artifact:{export_id}:adapter-manifest.json}}',
        '--adapter-manifest-sha256',
        f'${{sha256:{export_id}:adapter-manifest.json}}',
        '--output-dir', '{artifact_dir}',
        '--batch-size', '4',
        '--device', 'cuda',
      ]
      job = _job(
        protocol_id=source_plan['protocol_id'],
        source_manifest_sha256=source_plan['source_manifest_sha256'],
        source_repository_sha=source_plan['repository']['sha'],
        plan_id=plan_id,
        job_id=job_id,
        kind='eval',
        artifact_dir=artifact_dir,
        suites=[TOPOLOGY_SUITE],
        dependencies=[export_id],
        identity={
          'diagnostic': 'topology',
          'topology_protocol_id': protocol['protocol_id'],
          'topology_protocol_sha256': protocol_sha,
          'control': protocol['source_selection']['arm'],
          'dataset': dataset,
          'data_config_path': specification['data_config_path'],
          'train_seed': train_seed,
          'candidate_k': protocol['candidate_top_k'],
          'num_source_units': specification['num_source_units'],
          'num_records': (
            specification['num_source_units']
            * len(protocol['corruption_seeds'])
            * len(protocol['time_points'])
            * len(protocol['interventions'])),
        },
        argv=argv,
        execution_mode='fresh_attempt',
        external_inputs=[dict(backbone)],
        required_outputs=[
          {'name': 'topology_records',
           'pattern': 'topology_records.jsonl', 'exactly_one': True},
          {'name': 'topology_record_manifest',
           'pattern': 'topology_records.manifest.json', 'exactly_one': True},
          {'name': 'topology_source_selection',
           'pattern': 'topology_source_selection.json', 'exactly_one': True},
          {'name': 'dataset_provenance',
           'pattern': 'data_provenance/valid-*.json', 'exactly_one': True},
          {'name': 'gpu_exclusivity',
           'pattern': 'gpu_exclusivity.json', 'exactly_one': True},
        ])
      jobs[job_id] = job

  jobs = dict(sorted(jobs.items()))
  counts: dict[str, int] = {}
  for job in jobs.values():
    counts[job['kind']] = counts.get(job['kind'], 0) + 1
  plan = {
    'schema_version': PLAN_SCHEMA_VERSION,
    **plan_identity,
    'plan_id': plan_id,
    'manifest_protocol_status': source_plan['manifest_protocol_status'],
    'scientific_scope': protocol['scientific_scope'],
    'job_counts': dict(sorted(counts.items())),
    'num_jobs': len(jobs),
    'job_ids': list(jobs),
    'job_spec_sha256': {
      job_id: _sha256_canonical(job) for job_id, job in jobs.items()},
  }
  return plan, jobs, output_dir


def compile_topology_plan(
    *,
    source_plan_dir: Path,
    output_dir: Path | None = None,
    protocol_path: Path = DEFAULT_PROTOCOL,
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
  """Derive and atomically write the canonical topology plan."""
  plan, jobs, resolved_output = derive_topology_plan(
    source_plan_dir=source_plan_dir,
    output_dir=output_dir,
    protocol_path=protocol_path)
  write_plan(resolved_output, plan, jobs, resume=resume)
  return plan, jobs, resolved_output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source-plan-dir', type=Path, required=True)
  parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
  parser.add_argument('--output-dir', type=Path)
  parser.add_argument('--resume', action='store_true')
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  plan, jobs, output = compile_topology_plan(
    source_plan_dir=args.source_plan_dir,
    output_dir=args.output_dir,
    protocol_path=args.protocol,
    resume=args.resume)
  print(json.dumps({
    'plan_id': plan['plan_id'],
    'output_dir': str(output),
    'job_counts': plan['job_counts'],
    'num_jobs': len(jobs),
    'topology_jobs': sum(
      job['identity'].get('diagnostic') == 'topology'
      for job in jobs.values()),
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
