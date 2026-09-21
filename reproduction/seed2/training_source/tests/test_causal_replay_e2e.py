import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from data_provenance import canonical_sha256
from scripts.aggregate_causal_denoising_eval import build_analysis
from scripts.compile_experiment_matrix import (
  _git_metadata,
  build_jobs,
  load_and_validate_manifest,
  sha256_file,
)
from scripts.evaluate_causal_promotion import (
  build_causal_compiler_evidence,
  evaluate_causal_analysis,
  load_and_validate_causal_policy,
  verify_causal_compiler_evidence,
)
from scripts.finalize_causal_smoke_policy import (
  DEFAULT_MANIFEST,
  DEFAULT_TEMPLATE,
  REPO_ROOT,
  _read_mapping,
  _validate_template,
  finalize_policy,
)
from scripts.run_compiled_job import (
  SUCCESS_MARKER,
  _job_execution_digest,
  _output_records,
)


def _write_json(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(
    payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _init_source_repo(root: Path) -> tuple[Path, str]:
  repo = root / 'source-repo'
  repo.mkdir()
  subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
  subprocess.run(
    ['git', 'config', 'user.email', 'causal-replay@example.invalid'],
    cwd=repo, check=True)
  subprocess.run(
    ['git', 'config', 'user.name', 'Causal Replay Fixture'],
    cwd=repo, check=True)
  os.symlink(REPO_ROOT / 'configs', repo / 'configs', target_is_directory=True)
  (repo / 'fixture-source.txt').write_text('causal replay source v1\n')
  subprocess.run(['git', 'add', 'configs', 'fixture-source.txt'], cwd=repo,
                 check=True)
  subprocess.run(
    ['git', 'commit', '-q', '-m', 'freeze causal replay source'],
    cwd=repo, check=True)
  metadata = _git_metadata(repo)
  if metadata['dirty'] is not False:
    raise AssertionError('synthetic source repository is not clean')
  return repo, metadata['sha']


def _compile_source_plan(root: Path, repo: Path, repository_sha: str):
  template = _validate_template(
    _read_mapping(DEFAULT_TEMPLATE, context='template'),
    manifest_path=DEFAULT_MANIFEST,
    repo_root=repo)
  manifest = load_and_validate_manifest(DEFAULT_MANIFEST, repo_root=repo)
  plan_dir = root / 'plan'
  artifact_root = root / 'artifacts'
  repository = {'sha': repository_sha, 'dirty': False}
  identity = {
    'protocol_id': template['protocol_id'],
    'source_manifest_sha256': template['source_manifest_sha256'],
    'repository': repository,
    'artifact_root': str(artifact_root.resolve()),
    'selected_suites': [template['source_suite']],
    'promotion_evidence': {},
  }
  plan_id = canonical_sha256(identity)
  jobs = build_jobs(
    manifest,
    selected_suites=[template['source_suite']],
    artifact_root=artifact_root.resolve(),
    source_manifest_sha256=template['source_manifest_sha256'],
    source_repository_sha=repository_sha,
    plan_id=plan_id)
  counts = {}
  for job in jobs.values():
    counts[job['kind']] = counts.get(job['kind'], 0) + 1
  plan = {
    'schema_version': 2,
    **identity,
    'plan_id': plan_id,
    'manifest_protocol_status': manifest['protocol_status'],
    'scientific_scope': manifest['scientific_scope'],
    'job_counts': dict(sorted(counts.items())),
    'num_jobs': len(jobs),
    'job_ids': list(jobs),
    'job_spec_sha256': {
      job_id: canonical_sha256(job) for job_id, job in jobs.items()},
  }
  for job_id, job in jobs.items():
    _write_json(plan_dir / 'jobs' / f'{job_id}.json', job)
  _write_json(plan_dir / 'compiled-plan.json', plan)
  policy = finalize_policy(
    DEFAULT_TEMPLATE,
    plan_dir,
    manifest_path=DEFAULT_MANIFEST,
    frozen_utc='2026-08-31T05:00:00+00:00',
    repo_root=repo)
  policy_path = plan_dir / 'promotion-policy.json'
  _write_json(policy_path, policy)
  return plan_dir, plan, jobs, policy, policy_path


def _dataset_provenance(data_config: dict, *, compiled_dataset: str):
  specification = {
    'logical_dataset_name': data_config['valid'],
    'dataset_name_or_path': data_config['valid_dataset_name_or_path'],
    'dataset_config_name': data_config['valid_dataset_config_name'],
    'source_split': data_config['valid_source_split'],
    'source_revision': data_config['valid_revision'],
    'source_num_rows': data_config['valid_expected_source_num_rows'],
    'source_window': data_config['valid_source_window'],
    'text_field': data_config['valid_text_field'],
    'document_boundary_mode': data_config['valid_document_boundary_mode'],
    'trust_remote_code': data_config['valid_trust_remote_code'],
    'tokenizer_name_or_path': data_config['tokenizer_name_or_path'],
    'tokenizer_revision': data_config['tokenizer_revision'],
  }
  if compiled_dataset == 'wikitext103':
    processed = 197
    document_rows = 60
  else:
    processed = 256
    document_rows = 256
  observed = {
    'source_num_rows': data_config['valid_expected_source_num_rows'],
    'window_num_rows': data_config['valid_expected_source_num_rows'],
    'document_num_rows_after_boundary_recovery': document_rows,
    'processed_num_sequences': processed,
    'raw_fingerprint': f'{compiled_dataset}-raw-fixture-v1',
    'window_fingerprint': f'{compiled_dataset}-window-fixture-v1',
    'processed_fingerprint': f'{compiled_dataset}-processed-fixture-v1',
  }
  body = {
    'schema_version': 1,
    'artifact': 'pinned_text_dataset_provenance',
    'specification': specification,
    'specification_sha256': canonical_sha256(specification),
    'observed': observed,
  }
  return {**body, 'manifest_sha256': canonical_sha256(body)}, processed


def _causal_rows(job: dict, data_config: dict, pairing_sha: str,
                 *, num_records: int):
  identity = job['identity']
  arm_offsets = {
    'dynamic_dynamic': 0.0,
    'fixed_dynamic': 0.2,
    'dynamic_fixed': 0.3,
    'static_static': 0.5,
  }
  masked_tokens = 4
  logical_dataset = data_config['valid']
  document_sha = hashlib.sha256(logical_dataset.encode()).hexdigest()
  for index in range(num_records):
    joint = 12.0 + arm_offsets[identity['control']]
    yield {
      'schema_version': 2,
      'protocol_id': job['protocol_id'],
      'job_id': job['job_id'],
      'arm': identity['control'],
      'train_seed': identity['train_seed'],
      'corruption_seed': identity['corruption_seed'],
      'dataset': logical_dataset,
      'dataset_revision': data_config['valid_revision'],
      'mask_rate': float(identity['mask_rate']),
      'candidate_k': identity['candidate_k'],
      'rank': 0,
      'batch_index': index // 4,
      'example_index': index % 4,
      'document_id': f'{logical_dataset}:0',
      'document_index': 0,
      'document_sha256': document_sha,
      'chunk_index': index,
      'nll_sum': joint,
      'masked_tokens': masked_tokens,
      'candidate_hits': 3,
      'retained_mass_sum': 3.5,
      'pairing_digest_sha256': pairing_sha,
      'structured_marginal_nll_sum': joint + 0.1,
      'factorized_backbone_nll_sum': 14.0,
      'parameter_matched_no_edge_nll_sum': 14.0,
      'matched_permuted_topology_nll_sum': joint + 0.2,
      'selected_edges': 3,
      'permuted_changed_edges': 3,
      'selected_degree_sequence': [1, 1, 2, 2],
      'permuted_degree_sequence': [1, 1, 2, 2],
      'selected_component_sizes': [4],
      'permuted_component_sizes': [4],
      'candidate_support': [
        {'candidate_k': 32, 'candidate_hits': 1,
         'retained_mass_sum': 1.0},
        {'candidate_k': 64, 'candidate_hits': 2,
         'retained_mass_sum': 2.0},
        {'candidate_k': 128, 'candidate_hits': 3,
         'retained_mass_sum': 3.0},
        {'candidate_k': 256, 'candidate_hits': 4,
         'retained_mass_sum': 4.0},
      ],
    }


def _populate_completed_artifacts(
    repo: Path,
    jobs: dict[str, dict],
    *,
    marker_start: str = '2026-08-31T06:00:00+00:00',
) -> None:
  manifest = load_and_validate_manifest(DEFAULT_MANIFEST, repo_root=repo)
  for job in jobs.values():
    artifact_dir = Path(job['artifact_dir'])
    run_dir = artifact_dir / 'attempts' / 'attempt-0001'
    run_dir.mkdir(parents=True, exist_ok=True)
    if job['kind'] == 'train':
      (run_dir / 'checkpoints').mkdir()
      (run_dir / 'checkpoints/last.ckpt').write_bytes(b'checkpoint\n')
      (run_dir / 'data_provenance').mkdir()
      (run_dir / 'data_provenance/train-fixture.json').write_text('{}\n')
      (run_dir / 'data_provenance/valid-fixture.json').write_text('{}\n')
    elif job['kind'] == 'export':
      (run_dir / 'adapter.safetensors').write_bytes(b'adapter\n')
      (run_dir / 'adapter-manifest.json').write_text('{}\n')
    else:
      identity = job['identity']
      data_config_name = manifest['datasets'][identity['dataset']]['data_config']
      with (repo / 'configs/data' / f'{data_config_name}.yaml').open() as handle:
        data_config = yaml.safe_load(handle)
      provenance, num_records = _dataset_provenance(
        data_config, compiled_dataset=identity['dataset'])
      pairing_sha = canonical_sha256({
        'corruption_seed': identity['corruption_seed'],
        'dataset': identity['dataset'],
        'mask_rate': float(identity['mask_rate']),
        'candidate_k': identity['candidate_k'],
      })
      pairing = {'sha256': pairing_sha, 'world_size': 1}
      _write_json(run_dir / 'validation_pairing_digest.json', pairing)
      provenance_path = run_dir / 'data_provenance/valid-fixture.json'
      _write_json(provenance_path, provenance)
      rows_path = run_dir / 'conditional_denoising_records.rank0.jsonl'
      rows = list(_causal_rows(
        job, data_config, pairing_sha, num_records=num_records))
      rows_path.write_text(''.join(
        json.dumps(row, sort_keys=True, separators=(',', ':'),
                   allow_nan=False) + '\n'
        for row in rows))
      metadata = {
        'protocol_id': manifest['protocol_id'],
        'job_id': job['job_id'],
        'arm': identity['control'],
        'train_seed': identity['train_seed'],
        'corruption_seed': identity['corruption_seed'],
        'dataset': data_config['valid'],
        'dataset_revision': data_config['valid_revision'],
        'mask_rate': float(identity['mask_rate']),
        'candidate_k': identity['candidate_k'],
      }
      record_manifest = {
        'schema_version': 2,
        'artifact': 'conditional_denoising_record_manifest',
        'metadata': metadata,
        'pairing_digest': pairing,
        'rank_files': [{
          'rank': 0,
          'path': rows_path.name,
          'sha256': sha256_file(rows_path),
          'num_records': num_records,
          'total_masked_tokens': num_records * 4,
          'pairing_digest_sha256': pairing_sha,
        }],
        'num_records': num_records,
        'total_masked_tokens': num_records * 4,
      }
      _write_json(
        run_dir / 'conditional_denoising_records.manifest.json',
        record_manifest)
    outputs = _output_records(run_dir, job['required_outputs'])
    marker = {
      'schema_version': 2,
      'artifact': 'compiled_experiment_job_success',
      'job_id': job['job_id'],
      'originating_plan_id': job['plan_id'],
      'source_repository_sha': job['source_repository_sha'],
      'job_execution_sha256': _job_execution_digest(job),
      'run_dir': str(run_dir.resolve()),
      'argv': ['synthetic-causal-replay'],
      'start_time_utc': marker_start,
      'end_time_utc': '2026-08-31T06:10:00+00:00',
      'outputs': outputs,
    }
    _write_json(artifact_dir / SUCCESS_MARKER, marker)


def _build_verified_bundle(root: Path):
  repo, repository_sha = _init_source_repo(root)
  plan_dir, plan, jobs, policy, policy_path = _compile_source_plan(
    root, repo, repository_sha)
  _populate_completed_artifacts(repo, jobs)
  analysis = build_analysis(
    plan_dir=plan_dir,
    manifest_path=DEFAULT_MANIFEST,
    suite_name='causal_smoke',
    timestamp_utc='2026-08-31T07:00:00+00:00',
    technical_gates=policy['technical_gates'],
    repo_root=repo)
  analysis_path = plan_dir / 'analysis.json'
  _write_json(analysis_path, analysis)
  validated_policy = load_and_validate_causal_policy(
    policy_path,
    manifest_path=DEFAULT_MANIFEST,
    repo_root=repo)
  plan_path = plan_dir / 'compiled-plan.json'
  decision = evaluate_causal_analysis(
    validated_policy,
    analysis,
    policy_sha256=sha256_file(policy_path),
    analysis_sha256=sha256_file(analysis_path),
    source_plan=plan,
    source_plan_path=plan_path,
    source_plan_sha256=sha256_file(plan_path),
    manifest_path=DEFAULT_MANIFEST,
    created_utc='2026-08-31T08:00:00+00:00',
    repo_root=repo)
  decision_path = plan_dir / 'routing-decision.json'
  _write_json(decision_path, decision)
  evidence = build_causal_compiler_evidence(
    decision,
    'primary',
    policy_path=policy_path,
    analysis_path=analysis_path,
    source_plan_path=plan_path,
    decision_path=decision_path)
  evidence_path = plan_dir / 'causal_primary-promotion.json'
  _write_json(evidence_path, evidence)
  return {
    'repo': repo,
    'plan_dir': plan_dir,
    'plan': plan,
    'jobs': jobs,
    'policy': validated_policy,
    'policy_path': policy_path,
    'analysis': analysis,
    'analysis_path': analysis_path,
    'decision': decision,
    'decision_path': decision_path,
    'evidence': evidence,
    'evidence_path': evidence_path,
  }


class CausalReplayEndToEndTest(unittest.TestCase):

  def test_wrong_clean_head_and_dirty_checkout_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      repo, repository_sha = _init_source_repo(root)
      _, _, _, _, policy_path = _compile_source_plan(
        root, repo, repository_sha)
      (repo / 'dirty.txt').write_text('uncommitted\n')
      with self.assertRaisesRegex(ValueError, 'exact clean source'):
        load_and_validate_causal_policy(
          policy_path, manifest_path=DEFAULT_MANIFEST, repo_root=repo)
      subprocess.run(['git', 'add', 'dirty.txt'], cwd=repo, check=True)
      subprocess.run(
        ['git', 'commit', '-q', '-m', 'move away from source head'],
        cwd=repo, check=True)
      with self.assertRaisesRegex(ValueError, 'exact clean source'):
        load_and_validate_causal_policy(
          policy_path, manifest_path=DEFAULT_MANIFEST, repo_root=repo)

  def test_full_replay_rejects_output_timing_and_evidence_tampering(self):
    with tempfile.TemporaryDirectory() as directory:
      bundle = _build_verified_bundle(Path(directory))
      verified = verify_causal_compiler_evidence(
        bundle['evidence'],
        evidence_path=bundle['evidence_path'],
        promoted_suite='causal_primary',
        manifest_path=DEFAULT_MANIFEST,
        trusted_template_path=DEFAULT_TEMPLATE,
        repo_root=bundle['repo'])
      self.assertEqual(verified, bundle['evidence'])

      tampered_evidence = copy.deepcopy(bundle['evidence'])
      first_criterion = next(iter(tampered_evidence['criteria']))
      tampered_evidence['criteria'][first_criterion] = False
      with self.assertRaisesRegex(ValueError, 'differs from canonical'):
        verify_causal_compiler_evidence(
          tampered_evidence,
          evidence_path=bundle['evidence_path'],
          promoted_suite='causal_primary',
          manifest_path=DEFAULT_MANIFEST,
          trusted_template_path=DEFAULT_TEMPLATE,
          repo_root=bundle['repo'])

      eval_job = next(
        job for job in bundle['jobs'].values() if job['kind'] == 'eval')
      marker_path = Path(eval_job['artifact_dir']) / SUCCESS_MARKER
      marker_bytes = marker_path.read_bytes()
      marker = json.loads(marker_bytes)
      records_output = next(
        output for output in marker['outputs']
        if output['name'] == 'conditional_records')
      records_path = Path(marker['run_dir']) / records_output['relative_path']
      records_bytes = records_path.read_bytes()
      records_path.write_bytes(records_bytes + b'{}\n')
      with self.assertRaisesRegex(ValueError, 'outputs drifted'):
        verify_causal_compiler_evidence(
          bundle['evidence'],
          evidence_path=bundle['evidence_path'],
          promoted_suite='causal_primary',
          manifest_path=DEFAULT_MANIFEST,
          trusted_template_path=DEFAULT_TEMPLATE,
          repo_root=bundle['repo'])
      records_path.write_bytes(records_bytes)

      marker['start_time_utc'] = '2026-08-31T04:59:00+00:00'
      _write_json(marker_path, marker)
      bad_timing_analysis = build_analysis(
        plan_dir=bundle['plan_dir'],
        manifest_path=DEFAULT_MANIFEST,
        suite_name='causal_smoke',
        timestamp_utc='2026-08-31T07:00:00+00:00',
        technical_gates=bundle['policy']['technical_gates'],
        repo_root=bundle['repo'])
      with self.assertRaisesRegex(ValueError, 'not frozen before'):
        evaluate_causal_analysis(
          bundle['policy'],
          bad_timing_analysis,
          policy_sha256=sha256_file(bundle['policy_path']),
          analysis_sha256=canonical_sha256(bad_timing_analysis),
          source_plan=bundle['plan'],
          source_plan_path=bundle['plan_dir'] / 'compiled-plan.json',
          source_plan_sha256=sha256_file(
            bundle['plan_dir'] / 'compiled-plan.json'),
          manifest_path=DEFAULT_MANIFEST,
          created_utc='2026-08-31T08:00:00+00:00',
          repo_root=bundle['repo'])
      marker_path.write_bytes(marker_bytes)

      (bundle['repo'] / 'new-head.txt').write_text('new clean head\n')
      subprocess.run(
        ['git', 'add', 'new-head.txt'], cwd=bundle['repo'], check=True)
      subprocess.run(
        ['git', 'commit', '-q', '-m', 'advance synthetic source head'],
        cwd=bundle['repo'], check=True)
      with self.assertRaisesRegex(ValueError, 'exact clean source'):
        verify_causal_compiler_evidence(
          bundle['evidence'],
          evidence_path=bundle['evidence_path'],
          promoted_suite='causal_primary',
          manifest_path=DEFAULT_MANIFEST,
          trusted_template_path=DEFAULT_TEMPLATE,
          repo_root=bundle['repo'])


if __name__ == '__main__':
  unittest.main()
