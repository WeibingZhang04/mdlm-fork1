import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.compile_experiment_matrix import (
  DEFAULT_MANIFEST,
  JOB_SCHEMA_VERSION,
  REPO_ROOT,
  TRUSTED_CAUSAL_PROMOTION_TEMPLATES,
  TRUSTED_CANDIDATE_K_PROMOTION_TEMPLATES,
  compile_matrix,
  derive_matrix_plan,
  sha256_file,
)
from scripts.run_compiled_job import _job_execution_digest, run_job


CAUSAL_MANIFEST = (
  REPO_ROOT / 'configs/experiment/contextual-forest-causal-evidence-v1.yaml')


class ExperimentMatrixTest(unittest.TestCase):

  def setUp(self):
    self.git_metadata = mock.patch(
      'scripts.compile_experiment_matrix._git_metadata',
      return_value={'sha': 'd' * 40, 'dirty': False})
    self.git_metadata.start()

  def tearDown(self):
    self.git_metadata.stop()

  def test_pilot_compiles_canonical_k64_jobs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      plan, jobs, output = compile_matrix(
        DEFAULT_MANIFEST,
        selected_suites=['pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'plan')

    self.assertEqual(plan['job_counts'], {
      'eval': 48, 'export': 2, 'train': 2})
    self.assertEqual(plan['num_jobs'], 52)
    self.assertEqual(output, (artifact_root / 'plan').resolve())
    train = jobs['train--dynamic_dynamic--s001--k064']
    self.assertIn('model.structured_decoder.top_k=64', train['argv'])
    self.assertEqual(train['execution_mode'], 'fresh_attempt')
    self.assertIn('checkpointing.resume_from_ckpt=false', train['argv'])
    self.assertFalse(any(
      token.startswith('checkpointing.resume_ckpt_path=')
      for token in train['argv']))
    self.assertIn(
      'checkpointing.save_dir={artifact_dir}', train['argv'])
    self.assertEqual(
      train['identity']['corruption_rng_policy'],
      'paired_private_torch_generator_v1_seeded_by_train_seed_epoch_rank')
    self.assertEqual(
      train['identity']['topology_teacher_rng_policy'],
      'domain_separated_private_torch_generator_v1_offset_4294967291')
    self.assertEqual(
      train['identity']['cross_control_pairing_policy'],
      'identical_corruption_stream_for_equal_train_seed_epoch_rank')
    self.assertEqual(
      {item['name'] for item in train['required_outputs']}, {
        'checkpoint', 'training_data_provenance',
        'training_validation_data_provenance'})
    training_provenance = next(
      item for item in train['required_outputs']
      if item['name'] == 'training_data_provenance')
    self.assertEqual(
      training_provenance['pattern'], 'data_provenance/train-*.json')
    evaluation = next(
      job for job in jobs.values() if job['kind'] == 'eval')
    export = jobs['export--dynamic_dynamic--s001--k064']
    self.assertEqual(export['identity'], {
      'control': 'dynamic_dynamic',
      'train_seed': 1,
      'candidate_k': 64,
      'topology_mode': 'dynamic',
      'factor_mode': 'dynamic',
      'independent_mode': False,
      'topology_weight': 0.1,
    })
    for flag, expected in (
        ('--control-identity', 'dynamic_dynamic'),
        ('--topology-mode', 'dynamic'),
        ('--factor-mode', 'dynamic'),
        ('--candidate-k', '64'),
        ('--independent-mode', 'false'),
        ('--topology-weight', '0.1')):
      index = export['argv'].index(flag)
      self.assertEqual(export['argv'][index + 1], expected)
    self.assertNotIn(
      'strategy.find_unused_parameters=true',
      jobs['train--dynamic_dynamic--s001--k064']['argv'])
    self.assertIn(
      'strategy.find_unused_parameters=true',
      jobs['train--static_static--s001--k064']['argv'])
    self.assertEqual(evaluation['dependencies'], [
      f'export--{evaluation["identity"]["control"]}--s001--k064'])
    self.assertEqual(evaluation['identity']['validation_batches'], 128)
    self.assertIn(
      'eval.conditional_records.enabled=true', evaluation['argv'])
    self.assertIn(
      f'eval.conditional_records.job_id={evaluation["job_id"]}',
      evaluation['argv'])
    self.assertIn(
      f'eval.adapter_manifest=${{artifact:'
      f'{evaluation["dependencies"][0]}:adapter-manifest.json}}',
      evaluation['argv'])
    self.assertIn(
      f'eval.adapter_manifest_sha256=${{sha256:'
      f'{evaluation["dependencies"][0]}:adapter-manifest.json}}',
      evaluation['argv'])
    self.assertEqual(plan['repository'], {
      'sha': 'd' * 40, 'dirty': False})
    self.assertTrue(all(
      job['source_repository_sha'] == 'd' * 40
      for job in jobs.values()))
    self.assertEqual(
      {item['name'] for item in evaluation['required_outputs']}, {
        'pairing_digest', 'conditional_records',
        'conditional_record_manifest', 'dataset_provenance'})
    self.assertTrue(
      str(evaluation['artifact_dir']).startswith(str(artifact_root.resolve())))

  def test_in_memory_derivation_does_not_write_or_replace_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      artifact_root.mkdir()
      sentinel = artifact_root / 'preexisting-artifact.txt'
      sentinel.write_text('preserve exactly\n')
      plan, jobs, observed_root = derive_matrix_plan(
        DEFAULT_MANIFEST,
        selected_suites=['pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root)

      self.assertEqual(observed_root, artifact_root.resolve())
      self.assertEqual(plan['num_jobs'], len(jobs))
      self.assertEqual(sentinel.read_text(), 'preserve exactly\n')
      self.assertEqual(
        sorted(path.relative_to(artifact_root).as_posix()
               for path in artifact_root.rglob('*')),
        ['preexisting-artifact.txt'])

  def test_causal_smoke_compiles_schema_v2_paired_metrics(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      plan, jobs, _ = compile_matrix(
        CAUSAL_MANIFEST,
        selected_suites=['causal_smoke'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'plan')

    self.assertEqual(plan['job_counts'], {
      'eval': 32, 'export': 4, 'train': 4})
    self.assertEqual({
      job['identity']['control'] for job in jobs.values()
      if job['kind'] == 'train'}, {
        'dynamic_dynamic', 'fixed_dynamic',
        'dynamic_fixed', 'static_static'})
    evaluation = next(job for job in jobs.values() if job['kind'] == 'eval')
    self.assertIn(
      'eval.conditional_records.schema_version=2', evaluation['argv'])
    self.assertIn(
      'eval.conditional_records.support_candidate_ks=[32,64,128,256]',
      evaluation['argv'])
    self.assertEqual(evaluation['identity']['candidate_k'], 128)

  def test_causal_primary_requires_technical_smoke_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with self.assertRaisesRegex(ValueError, 'is gated on causal_smoke'):
        compile_matrix(
          CAUSAL_MANIFEST,
          selected_suites=['causal_primary'],
          allowed_artifact_root=root,
          artifact_root_override=root / 'artifacts',
          output_dir=root / 'artifacts/plan')

  def test_causal_primary_uses_trusted_causal_verifier_and_reuses_smoke(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      _, smoke_jobs, _ = compile_matrix(
        CAUSAL_MANIFEST,
        selected_suites=['causal_smoke'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'smoke-plan')
      evidence_path = root / 'causal-smoke-promotion.json'
      evidence_path.write_text('{}')
      verified = {
        'source_suite': 'causal_smoke',
        'route_name': 'primary',
        'commitments': {
          'canonical_decision_sha256': 'a' * 64,
          'source_compiled_plan_sha256': 'b' * 64,
        },
      }
      with mock.patch(
          'scripts.evaluate_causal_promotion.verify_causal_compiler_evidence',
          return_value=verified) as causal_verifier, mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          'verify_candidate_compiler_evidence') as candidate_verifier, \
          mock.patch(
          'scripts.evaluate_experiment_promotion.verify_compiler_evidence'
      ) as pilot_verifier:
        plan, primary_jobs, _ = compile_matrix(
          CAUSAL_MANIFEST,
          selected_suites=['causal_primary'],
          allowed_artifact_root=root,
          artifact_root_override=artifact_root,
          output_dir=artifact_root / 'primary-plan',
          promotion_evidence={'causal_primary': evidence_path})

    pilot_verifier.assert_not_called()
    candidate_verifier.assert_not_called()
    causal_verifier.assert_called_once()
    call = causal_verifier.call_args
    self.assertEqual(call.args, ({},))
    self.assertEqual(
      call.kwargs['trusted_template_path'],
      TRUSTED_CAUSAL_PROMOTION_TEMPLATES[
        'contextual-forest-causal-evidence-v1']['causal_smoke'])
    self.assertEqual(
      plan['promotion_evidence']['causal_primary']['source_suite'],
      'causal_smoke')
    for job_id in (
        'train--dynamic_dynamic--s001--k128',
        'export--dynamic_dynamic--s001--k128',
        'train--static_static--s001--k128',
        'export--static_static--s001--k128'):
      self.assertEqual(
        _job_execution_digest(smoke_jobs[job_id]),
        _job_execution_digest(primary_jobs[job_id]))

  def test_gated_suite_requires_matching_promotion_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      with self.assertRaisesRegex(ValueError, 'is gated on pilot'):
        compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['candidate_k_128_pilot'],
          allowed_artifact_root=root,
          artifact_root_override=artifact_root,
          output_dir=artifact_root / 'plan')

      evidence = root / 'promotion.json'
      evidence.write_text(json.dumps({
        'schema_version': 1,
        'artifact': 'experiment_suite_promotion_decision',
        'protocol_id': 'contextual-forest-expansion-v1',
        'source_manifest_sha256': sha256_file(DEFAULT_MANIFEST),
        'source_suite': 'pilot',
        'promoted_suite': 'candidate_k_128_pilot',
        'decision': 'promote',
        'criteria': {
          'pairing_verified': True,
          'quality_gate_passed': True,
          'candidate_coverage_gate_passed': True,
        },
        'created_utc': '2026-08-30T00:00:00+00:00',
      }, sort_keys=True))
      with self.assertRaisesRegex(ValueError, 'schema mismatch'):
        compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['candidate_k_128_pilot'],
          allowed_artifact_root=root,
          artifact_root_override=artifact_root,
          output_dir=artifact_root / 'plan',
          promotion_evidence={'candidate_k_128_pilot': evidence})

  def test_candidate_k_source_uses_frozen_candidate_k_verifier(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      evidence_path = root / 'candidate-k-promotion.json'
      evidence_path.write_text('{}')
      verified = {
        'source_suite': 'candidate_k_128_pilot',
        'route_name': 'confirmation',
        'commitments': {
          'canonical_decision_sha256': 'a' * 64,
          'source_compiled_plan_sha256': 'b' * 64,
        },
      }
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          'verify_candidate_compiler_evidence',
          return_value=verified) as candidate_verifier, mock.patch(
          'scripts.evaluate_experiment_promotion.verify_compiler_evidence'
      ) as pilot_verifier:
        plan, _, _ = compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['candidate_k_128_confirmation'],
          allowed_artifact_root=root,
          artifact_root_override=artifact_root,
          output_dir=artifact_root / 'plan',
          promotion_evidence={
            'candidate_k_128_confirmation': evidence_path})

    pilot_verifier.assert_not_called()
    candidate_verifier.assert_called_once()
    call = candidate_verifier.call_args
    self.assertEqual(call.args, ({},))
    self.assertEqual(
      call.kwargs['trusted_template_path'],
      TRUSTED_CANDIDATE_K_PROMOTION_TEMPLATES[
        'contextual-forest-expansion-v1']['candidate_k_128_pilot'])
    self.assertEqual(
      plan['promotion_evidence']['candidate_k_128_confirmation'][
        'source_suite'],
      'candidate_k_128_pilot')

  def test_plan_identity_changes_with_clean_repository_revision(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      first, _, _ = compile_matrix(
        DEFAULT_MANIFEST,
        selected_suites=['pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'first')
      with mock.patch(
          'scripts.compile_experiment_matrix._git_metadata',
          return_value={'sha': 'e' * 40, 'dirty': False}):
        second, _, _ = compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['pilot'],
          allowed_artifact_root=root,
          artifact_root_override=artifact_root,
          output_dir=artifact_root / 'second')
    self.assertNotEqual(first['plan_id'], second['plan_id'])

  def test_dirty_repository_cannot_compile(self):
    with tempfile.TemporaryDirectory() as directory, mock.patch(
        'scripts.compile_experiment_matrix._git_metadata',
        return_value={'sha': 'd' * 40, 'dirty': True}):
      root = Path(directory)
      with self.assertRaisesRegex(ValueError, 'clean committed repository'):
        compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['pilot'],
          allowed_artifact_root=root,
          artifact_root_override=root / 'artifacts',
          output_dir=root / 'artifacts/plan')

  def test_resume_rejects_compiled_job_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      _, jobs, output = compile_matrix(
        DEFAULT_MANIFEST,
        selected_suites=['pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'plan')
      compile_matrix(
        DEFAULT_MANIFEST,
        selected_suites=['pilot'],
        allowed_artifact_root=root,
        artifact_root_override=artifact_root,
        output_dir=artifact_root / 'plan',
        resume=True)
      job_id = next(iter(jobs))
      job_path = output / 'jobs' / f'{job_id}.json'
      job_path.write_text(job_path.read_text() + ' ')
      with self.assertRaisesRegex(ValueError, 'drifted'):
        compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['pilot'],
          allowed_artifact_root=root,
          artifact_root_override=artifact_root,
          output_dir=artifact_root / 'plan',
          resume=True)

  def test_manifest_unknown_fields_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      payload = yaml.safe_load(DEFAULT_MANIFEST.read_text())
      payload['unreviewed_override'] = True
      manifest = root / 'manifest.yaml'
      manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
      with self.assertRaisesRegex(ValueError, 'unknown=.*unreviewed_override'):
        compile_matrix(
          manifest,
          selected_suites=['pilot'],
          allowed_artifact_root=root,
          artifact_root_override=root / 'artifacts',
          output_dir=root / 'artifacts/plan',
          repo_root=REPO_ROOT)

  def test_manifest_rejects_unpaired_training_rng_policy(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      payload = yaml.safe_load(DEFAULT_MANIFEST.read_text())
      payload['training']['cross_control_pairing_policy'] = 'unpaired'
      manifest = root / 'manifest.yaml'
      manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
      with self.assertRaisesRegex(
          ValueError, 'cross_control_pairing_policy must equal'):
        compile_matrix(
          manifest,
          selected_suites=['pilot'],
          allowed_artifact_root=root,
          artifact_root_override=root / 'artifacts',
          output_dir=root / 'artifacts/plan',
          repo_root=REPO_ROOT)


class CompiledJobRunnerTest(unittest.TestCase):

  def test_whole_plan_dry_run_does_not_require_dependency_outputs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      parent_dir = artifact_root / 'runs' / 'parent'
      child_dir = artifact_root / 'runs' / 'child'
      plan = {
        'artifact_root': str(artifact_root),
        'plan_id': 'a' * 64,
        'repository': {'sha': 'd' * 40, 'dirty': False},
      }
      parent = {
        'schema_version': JOB_SCHEMA_VERSION,
        'protocol_id': 'test-protocol',
        'source_manifest_sha256': 'b' * 64,
        'source_repository_sha': 'd' * 40,
        'plan_id': 'a' * 64,
        'job_id': 'parent',
        'kind': 'train',
        'artifact_dir': str(parent_dir),
        'suites': ['pilot'],
        'dependencies': [],
        'identity': {},
        'argv': ['{python}', '--output', '{artifact_dir}/model.bin'],
        'execution_mode': 'fresh_attempt',
        'external_inputs': [],
        'required_outputs': [{
          'name': 'model', 'pattern': 'model.bin', 'exactly_one': True}],
      }
      child = dict(
        parent,
        job_id='child',
        kind='export',
        artifact_dir=str(child_dir),
        dependencies=['parent'],
        argv=['{python}', '--input', '${parent:model:path}',
              '--sha256', '${parent:model:sha256}',
              '--output', '{artifact_dir}/adapter.bin'],
        required_outputs=[{
          'name': 'adapter', 'pattern': 'adapter.bin', 'exactly_one': True}],
      )
      jobs = {'parent': parent, 'child': child}
      self.assertEqual(
        run_job('parent', plan=plan, jobs=jobs, dry_run=True), 'dry-run')
      self.assertEqual(
        run_job('child', plan=plan, jobs=jobs, dry_run=True), 'dry-run')

  def test_success_marker_is_hash_verified_and_resumable(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      artifact_root = root / 'artifacts'
      run_root = artifact_root / 'runs' / 'fake-job'
      plan = {
        'artifact_root': str(artifact_root),
        'plan_id': 'a' * 64,
        'repository': {'sha': 'd' * 40, 'dirty': False},
      }
      job = {
        'schema_version': JOB_SCHEMA_VERSION,
        'protocol_id': 'test-protocol',
        'source_manifest_sha256': 'b' * 64,
        'source_repository_sha': 'd' * 40,
        'plan_id': 'a' * 64,
        'job_id': 'fake-job',
        'kind': 'eval',
        'artifact_dir': str(run_root),
        'suites': ['pilot'],
        'dependencies': [],
        'identity': {},
        'argv': ['{python}', '--output', '{artifact_dir}/result.json'],
        'execution_mode': 'fresh_attempt',
        'external_inputs': [],
        'required_outputs': [{
          'name': 'result', 'pattern': 'result.json', 'exactly_one': True}],
      }

      def fake_run(argv, *, cwd, check):
        self.assertEqual(cwd, REPO_ROOT)
        self.assertTrue(check)
        output = Path(argv[argv.index('--output') + 1])
        output.write_text('{"ok": true}\n')

      with mock.patch('scripts.run_compiled_job.subprocess.run', fake_run):
        first = run_job(
          'fake-job', plan=plan, jobs={'fake-job': job})
        second = run_job(
          'fake-job', plan=plan, jobs={'fake-job': job})
      self.assertEqual(first, 'completed')
      self.assertEqual(second, 'skipped')
      promoted_job = dict(
        job, plan_id='c' * 64, suites=['confirmation'])
      promoted_plan = dict(plan, plan_id='c' * 64)
      self.assertEqual(run_job(
        'fake-job', plan=promoted_plan,
        jobs={'fake-job': promoted_job}), 'skipped')
      marker = json.loads((run_root / '_job_success.json').read_text())
      output_record = marker['outputs'][0]
      result_path = Path(marker['run_dir']) / output_record['relative_path']
      self.assertEqual(
        output_record['sha256'], hashlib.sha256(result_path.read_bytes()).hexdigest())

      changed_revision_job = dict(
        promoted_job, source_repository_sha='e' * 40)
      with self.assertRaisesRegex(ValueError, 'does not match job spec'):
        run_job(
          'fake-job',
          plan={
            **promoted_plan,
            'repository': {'sha': 'e' * 40, 'dirty': False},
          },
          jobs={'fake-job': changed_revision_job})

      result_path.write_text('{"ok": false}\n')
      with self.assertRaisesRegex(ValueError, 'outputs drifted'):
        run_job('fake-job', plan=plan, jobs={'fake-job': job})


if __name__ == '__main__':
  unittest.main()
