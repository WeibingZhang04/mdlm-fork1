import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.aggregate_hierarchical_document_eval import (
  _load_plan_for_analysis,
)
from scripts.compile_experiment_matrix import (
  DEFAULT_MANIFEST,
  build_jobs,
  load_and_validate_manifest,
  sha256_file,
)
from scripts.evaluate_candidate_k_promotion import (
  _require_clean_descendant_repository,
  _verify_historical_candidate_authoritative_analysis,
  build_candidate_compiler_evidence,
  evaluate_candidate_k_analysis,
  main as candidate_promotion_main,
  verify_candidate_compiler_evidence,
)
from scripts.evaluate_experiment_promotion import canonical_sha256
from scripts.finalize_candidate_k_policy import (
  DEFAULT_TEMPLATE,
  _atomic_write,
  _read_mapping,
  _validate_template,
  finalize_policy,
)


def _write_json(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(
    payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _candidate_plan(root):
  template = _validate_template(
    _read_mapping(DEFAULT_TEMPLATE, context='template'),
    manifest_path=DEFAULT_MANIFEST)
  plan_dir = root / 'plan'
  artifact_root = root / 'artifacts'
  parent_path = root / 'parent-promotion.json'
  _write_json(parent_path, {})
  parent = {
    'path': str(parent_path.resolve()),
    'sha256': sha256_file(parent_path),
    'source_suite': 'pilot',
    'route_name': 'k128',
    'canonical_decision_sha256': 'b' * 64,
    'source_compiled_plan_sha256': 'c' * 64,
  }
  repository = {'sha': 'd' * 40, 'dirty': False}
  promotion_evidence = {'candidate_k_128_pilot': parent}
  identity = {
    'protocol_id': template['protocol_id'],
    'source_manifest_sha256': template['source_manifest_sha256'],
    'repository': repository,
    'artifact_root': str(artifact_root.resolve()),
    'selected_suites': ['candidate_k_128_pilot'],
    'promotion_evidence': promotion_evidence,
  }
  plan_id = canonical_sha256(identity)
  manifest = load_and_validate_manifest(DEFAULT_MANIFEST)
  jobs = build_jobs(
    manifest,
    selected_suites=['candidate_k_128_pilot'],
    artifact_root=artifact_root.resolve(),
    source_manifest_sha256=template['source_manifest_sha256'],
    source_repository_sha=repository['sha'],
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
  return plan_dir, plan, jobs, template


def _parent_verification(plan):
  parent = plan['promotion_evidence']['candidate_k_128_pilot']
  return {
    'source_suite': parent['source_suite'],
    'route_name': 'k128',
    'commitments': {
      'canonical_decision_sha256': parent['canonical_decision_sha256'],
      'source_compiled_plan_sha256': parent[
        'source_compiled_plan_sha256'],
    },
  }


def _policy_bundle(root):
  plan_dir, plan, jobs, template = _candidate_plan(root)
  with mock.patch(
      'scripts.evaluate_experiment_promotion.verify_compiler_evidence',
      return_value=_parent_verification(plan)), mock.patch(
      'scripts.finalize_candidate_k_policy._git_metadata',
      return_value=plan['repository']):
    policy = finalize_policy(
      DEFAULT_TEMPLATE,
      plan_dir,
      manifest_path=DEFAULT_MANIFEST,
      frozen_utc='2026-08-30T16:00:00+00:00')
  return policy, plan_dir / 'compiled-plan.json', plan, jobs, template


def _source_integrity(policy, plan_path):
  contract = policy['analysis_contract']
  output_sha = 'e' * 64
  body = {
    'schema_version': 1,
    'source_compiled_plan_path': str(plan_path.resolve()),
    'source_compiled_plan_sha256': contract['source_compiled_plan_sha256'],
    'source_plan_id': contract['source_plan_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'source_repository_sha': contract['source_repository_sha'],
    'source_repository_clean': True,
    'validated_job_ids': ['eval--fixture'],
    'jobs': {
      'eval--fixture': {
        'job_spec_sha256': 'a' * 64,
        'job_execution_sha256': 'b' * 64,
        'success_marker_path': str(
          (plan_path.parent / 'run' / '_job_success.json').resolve()),
        'success_marker_sha256': 'c' * 64,
        'outputs': [{
          'name': 'conditional_records',
          'relative_path': 'conditional-records.jsonl',
          'size_bytes': 1,
          'sha256': output_sha,
        }],
        'scientific_output_sha256': {
          'conditional_records': output_sha,
        },
      },
    },
  }
  return {**body, 'commitment_sha256': canonical_sha256(body)}


def _analysis(
    policy,
    plan_path,
    *,
    improvement=0.02,
    pooled_ci_lower=0.005,
    candidate_recall=0.80,
    retained_mass=0.80,
):
  contract = policy['analysis_contract']
  conditions = {}
  diagnostics = {}
  for dataset, dataset_spec in contract['datasets'].items():
    revision = dataset_spec['revision']
    for mask_rate in contract['mask_rates']:
      condition_key = (
        f'{dataset}|mask={mask_rate:.6f}|k={contract["candidate_k"]}')
      conditions[condition_key] = {
        'dataset': dataset,
        'dataset_revision': revision,
        'mask_rate': mask_rate,
        'candidate_k': contract['candidate_k'],
        'mean_improvement': improvement,
        'ci_lower': improvement - 0.01,
        'ci_upper': improvement + 0.01,
      }
      diagnostic_key = str((
        dataset, revision, mask_rate, contract['candidate_k']))
      diagnostics[diagnostic_key] = {
        'num_documents': 20,
        'num_train_seeds': len(contract['train_seeds']),
        'num_corruption_seeds': len(contract['corruption_seeds']),
        'arms': {
          contract['baseline_arm']: {
            'conditional_nll_per_masked_token': 3.0,
            'candidate_recall': candidate_recall,
            'retained_unary_mass': retained_mass,
          },
          contract['treatment_arm']: {
            'conditional_nll_per_masked_token': 3.0 - improvement,
            'candidate_recall': candidate_recall,
            'retained_unary_mass': retained_mass,
          },
        },
      }
  bootstrap = contract['bootstrap']
  integrity = _source_integrity(policy, plan_path)
  return {
    'schema_version': contract['analysis_schema_version'],
    'artifact': contract['analysis_artifact'],
    'created_utc': '2026-08-30T17:00:00+00:00',
    'protocol_id': policy['protocol_id'],
    'suite': policy['source_suite'],
    'comparison': contract['comparison'],
    'arms': {
      'baseline': contract['baseline_arm'],
      'treatment': contract['treatment_arm'],
    },
    'objective': contract['objective'],
    'scope_note': (
      'Conditional denoising only; no diffusion ELBO, likelihood, '
      'perplexity, or generation-quality quantity is inferred.'),
    'by_candidate_k': {
      str(contract['candidate_k']): {
        'method': bootstrap['method'],
        'improvement_definition': 'baseline conditional NLL minus treatment',
        'nesting': [
          'average corruption replications within source document',
          'resample training seeds with replacement',
          'resample source documents within sampled training seed',
          'equal-weight frozen dataset x mask-rate strata',
        ],
        'num_train_seeds': len(contract['train_seeds']),
        'num_strata': contract['expected_num_strata'],
        'num_resamples': bootstrap['num_resamples'],
        'rng': 'NumPy Generator(PCG64)',
        'rng_seed': bootstrap['rng_seed'],
        'confidence_level': bootstrap['confidence_level'],
        'pooled': {
          'mean_improvement': improvement,
          'ci_lower': pooled_ci_lower,
          'ci_upper': improvement + 0.01,
        },
        'conditions': conditions,
      },
    },
    'diagnostics': diagnostics,
    'compiled_plan': {
      'plan_id': contract['source_plan_id'],
      'source_manifest_sha256': policy['source_manifest_sha256'],
      'source_compiled_plan_sha256': contract[
        'source_compiled_plan_sha256'],
      'source_repository_sha': contract['source_repository_sha'],
      'source_repository_clean': True,
      'job_artifact_commitment_sha256': integrity['commitment_sha256'],
    },
    'source_integrity': integrity,
  }


class CandidateKPromotionTest(unittest.TestCase):

  def _evaluate(self, root, **analysis_kwargs):
    policy, plan_path, plan, _, _ = _policy_bundle(root)
    analysis = _analysis(policy, plan_path, **analysis_kwargs)
    with mock.patch(
        'scripts.evaluate_candidate_k_promotion.'
        '_verify_authoritative_analysis',
        return_value=analysis['source_integrity']) as verifier, mock.patch(
        'scripts.evaluate_candidate_k_promotion.'
        '_verify_policy_predates_source_jobs'):
      decision = evaluate_candidate_k_analysis(
        policy,
        analysis,
        policy_sha256='a' * 64,
        analysis_sha256='b' * 64,
        source_plan=plan,
        source_plan_path=plan_path,
        source_plan_sha256=sha256_file(plan_path),
        manifest_path=DEFAULT_MANIFEST,
        created_utc='2026-08-30T18:00:00+00:00')
    self.assertNotIn(
      'require_current_repository_match', verifier.call_args.kwargs)
    return decision

  def test_template_is_bound_to_current_manifest(self):
    template = _validate_template(
      _read_mapping(DEFAULT_TEMPLATE, context='template'),
      manifest_path=DEFAULT_MANIFEST)
    self.assertEqual(template['analysis_contract']['candidate_k'], 128)
    self.assertEqual(
      template['source_manifest_sha256'], sha256_file(DEFAULT_MANIFEST))

  def test_quality_positive_support_limited_promotes_both_routes(self):
    with tempfile.TemporaryDirectory() as directory:
      decision = self._evaluate(Path(directory))
    self.assertEqual(
      decision['outcome'], 'promote_k128_confirmation_and_k256')
    self.assertTrue(decision['routes']['confirmation']['promote'])
    self.assertTrue(decision['routes']['k256']['promote'])

  def test_quality_negative_support_limited_promotes_only_k256(self):
    with tempfile.TemporaryDirectory() as directory:
      decision = self._evaluate(
        Path(directory), improvement=-0.02, pooled_ci_lower=-0.04)
    self.assertEqual(decision['outcome'], 'promote_k256_only')
    self.assertFalse(decision['routes']['confirmation']['promote'])
    self.assertTrue(decision['routes']['k256']['promote'])

  def test_quality_positive_support_sufficient_promotes_confirmation_only(self):
    with tempfile.TemporaryDirectory() as directory:
      decision = self._evaluate(
        Path(directory), candidate_recall=0.995, retained_mass=0.97)
    self.assertEqual(decision['outcome'], 'promote_k128_confirmation_only')
    self.assertTrue(decision['routes']['confirmation']['promote'])
    self.assertFalse(decision['routes']['k256']['promote'])

  def test_finalizer_refuses_nonempty_source_artifact_directory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, jobs, _ = _candidate_plan(root)
      job = next(iter(jobs.values()))
      artifact_dir = Path(job['artifact_dir'])
      artifact_dir.mkdir(parents=True)
      (artifact_dir / 'started.txt').write_text('started\n')
      with mock.patch(
          'scripts.evaluate_experiment_promotion.verify_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value=plan['repository']), self.assertRaisesRegex(
          RuntimeError, 'execution began'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def test_finalizer_rejects_tampered_factorial_job(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, jobs, _ = _candidate_plan(root)
      job_id, job = next(iter(jobs.items()))
      job['identity']['candidate_k'] = 256
      _write_json(plan_dir / 'jobs' / f'{job_id}.json', job)
      with mock.patch(
          'scripts.evaluate_experiment_promotion.verify_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value=plan['repository']), self.assertRaisesRegex(
          ValueError, 'exact frozen K128 factorial'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def test_finalizer_refuses_timestamp_before_template(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, _, _ = _candidate_plan(root)
      with mock.patch(
          'scripts.evaluate_experiment_promotion.verify_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value=plan['repository']), self.assertRaisesRegex(
          ValueError, 'predate the policy template'):
        finalize_policy(
          DEFAULT_TEMPLATE,
          plan_dir,
          frozen_utc='2026-08-30T15:54:59+00:00')

  def test_finalizer_rejects_dirty_checkout(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, _, _ = _candidate_plan(root)
      with mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value={
            'sha': plan['repository']['sha'], 'dirty': True,
          }), self.assertRaisesRegex(ValueError, 'checkout differs'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def test_finalizer_rejects_different_checkout_revision(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, _, _, _ = _candidate_plan(root)
      with mock.patch(
          'scripts.finalize_candidate_k_policy._git_metadata',
          return_value={'sha': 'e' * 40, 'dirty': False}), \
          self.assertRaisesRegex(ValueError, 'checkout differs'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def _assert_publication_race_preserves_competing_artifact(self, name):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / name
      original_link = os.link

      def competing_publish(source, target):
        Path(target).write_text('competing artifact\n')
        return original_link(source, target)

      with mock.patch(
          'scripts.finalize_candidate_k_policy.os.link',
          side_effect=competing_publish), self.assertRaises(FileExistsError):
        _atomic_write(destination, {'ours': True})
      self.assertEqual(destination.read_text(), 'competing artifact\n')
      self.assertEqual(
        list(destination.parent.glob(f'.{destination.name}.tmp-*')), [])

  def test_candidate_decision_race_never_overwrites_destination(self):
    self._assert_publication_race_preserves_competing_artifact(
      'routing-decision.json')

  def test_candidate_evidence_race_never_overwrites_destination(self):
    self._assert_publication_race_preserves_competing_artifact(
      'candidate-k-promotion.json')

  def test_evaluator_rejects_source_plan_copy_at_different_path(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      policy, plan_path, plan, _, _ = _policy_bundle(root)
      copied = root / 'copied' / 'compiled-plan.json'
      _write_json(copied, plan)
      analysis = _analysis(policy, plan_path)
      with self.assertRaisesRegex(
          ValueError, 'path differs from policy'):
        evaluate_candidate_k_analysis(
          policy,
          analysis,
          policy_sha256='a' * 64,
          analysis_sha256='b' * 64,
          source_plan=plan,
          source_plan_path=copied,
          source_plan_sha256=sha256_file(copied))

  def test_exact_historical_candidate_plan_loads_only_with_explicit_replay(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, jobs, _ = _candidate_plan(root)
      before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*') if path.is_file()}
      descendant = {'sha': 'e' * 40, 'dirty': False}
      with mock.patch(
          'scripts.run_compiled_job._git_metadata',
          return_value=descendant), self.assertRaisesRegex(
            ValueError, 'checkout differs'):
        _load_plan_for_analysis(plan_dir)
      with mock.patch(
          'scripts.run_compiled_job._git_metadata',
          return_value=descendant):
        loaded_plan, loaded_jobs, observed_sha, legacy = \
          _load_plan_for_analysis(
            plan_dir, require_current_repository_match=False)
      expected_sha = sha256_file(plan_dir / 'compiled-plan.json')
      after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*') if path.is_file()}

    self.assertFalse(legacy)
    self.assertEqual(loaded_plan, plan)
    self.assertEqual(loaded_jobs, jobs)
    self.assertEqual(observed_sha, expected_sha)
    self.assertEqual(after, before)

  def test_trusted_historical_verifier_authenticates_before_relaxed_load(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      policy, plan_path, plan, _, _ = _policy_bundle(root)
      policy_path = root / 'candidate-policy.json'
      analysis_path = root / 'analysis.json'
      _write_json(policy_path, policy)
      analysis = _analysis(policy, plan_path)
      _write_json(analysis_path, analysis)
      context = {
        'comparison': {
          'baseline': policy['analysis_contract']['baseline_arm'],
          'treatment': policy['analysis_contract']['treatment_arm'],
        },
        'manifest': {'analysis': {'bootstrap_seed': 1701}},
        'source_integrity': analysis['source_integrity'],
      }
      before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*') if path.is_file()}
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          'verify_pilot_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_require_clean_descendant_repository'), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_verify_policy_predates_source_jobs'), mock.patch(
          'scripts.aggregate_hierarchical_document_eval.'
          '_load_plan_records_core',
          return_value=([], context)) as loader, mock.patch(
          'scripts.aggregate_hierarchical_document_eval.aggregate_records',
          return_value={'schema_version': 1}) as aggregate, mock.patch(
          'scripts.aggregate_hierarchical_document_eval.'
          'bind_analysis_to_source',
          return_value=analysis):
        integrity = _verify_historical_candidate_authoritative_analysis(
          policy,
          analysis,
          source_plan=plan,
          source_plan_path=plan_path,
          source_plan_sha256=sha256_file(plan_path),
          manifest_path=DEFAULT_MANIFEST,
          trusted_policy_path=policy_path,
          trusted_analysis_path=analysis_path)
      after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*') if path.is_file()}

      self.assertEqual(integrity, analysis['source_integrity'])
      self.assertEqual(after, before)
      self.assertIs(
        loader.call_args.kwargs['require_current_repository_match'], False)
      self.assertEqual(
        aggregate.call_args.kwargs['rng_seed'],
        1701)

  def test_historical_replay_requires_a_clean_descendant_checkout(self):
    with mock.patch(
        'scripts.evaluate_candidate_k_promotion._git_metadata',
        return_value={'sha': 'e' * 40, 'dirty': False}), mock.patch(
        'scripts.evaluate_candidate_k_promotion.subprocess.run') as run:
      run.return_value.returncode = 0
      _require_clean_descendant_repository('d' * 40)
    self.assertEqual(run.call_args.args[0], [
      'git', 'merge-base', '--is-ancestor', 'd' * 40, 'e' * 40])

    with mock.patch(
        'scripts.evaluate_candidate_k_promotion._git_metadata',
        return_value={'sha': 'e' * 40, 'dirty': True}), \
        self.assertRaisesRegex(ValueError, 'requires a clean'):
      _require_clean_descendant_repository('d' * 40)

    with mock.patch(
        'scripts.evaluate_candidate_k_promotion._git_metadata',
        return_value={'sha': 'e' * 40, 'dirty': False}), mock.patch(
        'scripts.evaluate_candidate_k_promotion.subprocess.run') as run, \
        self.assertRaisesRegex(ValueError, 'descended from source'):
      run.return_value.returncode = 1
      _require_clean_descendant_repository('d' * 40)

  def test_untrusted_candidate_policy_cannot_reach_relaxed_loader(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      policy, plan_path, plan, _, _ = _policy_bundle(root)
      policy_path = root / 'candidate-policy.json'
      analysis_path = root / 'analysis.json'
      _write_json(policy_path, policy)
      analysis = _analysis(policy, plan_path)
      _write_json(analysis_path, analysis)
      altered = json.loads(json.dumps(policy))
      altered['analysis_contract']['source_repository_sha'] = 'e' * 40
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          'verify_pilot_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.aggregate_hierarchical_document_eval.'
          '_load_plan_records_core') as loader, self.assertRaisesRegex(
            ValueError, 'differs from the trusted policy artifact'):
        _verify_historical_candidate_authoritative_analysis(
          altered,
          analysis,
          source_plan=plan,
          source_plan_path=plan_path,
          source_plan_sha256=sha256_file(plan_path),
          manifest_path=DEFAULT_MANIFEST,
          trusted_policy_path=policy_path,
          trusted_analysis_path=analysis_path)
      loader.assert_not_called()

      altered_plan = json.loads(json.dumps(plan))
      altered_plan['repository']['sha'] = 'e' * 40
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          'verify_pilot_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.aggregate_hierarchical_document_eval.'
          '_load_plan_records_core') as loader, self.assertRaisesRegex(
            ValueError, 'source-plan mapping differs'):
        _verify_historical_candidate_authoritative_analysis(
          policy,
          analysis,
          source_plan=altered_plan,
          source_plan_path=plan_path,
          source_plan_sha256=sha256_file(plan_path),
          manifest_path=DEFAULT_MANIFEST,
          trusted_policy_path=policy_path,
          trusted_analysis_path=analysis_path)
      loader.assert_not_called()

      untrusted_template = root / 'template.json'
      _write_json(untrusted_template, {})
      with mock.patch(
          'scripts.aggregate_hierarchical_document_eval.'
          '_load_plan_records_core') as loader, self.assertRaisesRegex(
            ValueError, 'requires DEFAULT_TEMPLATE'):
        _verify_historical_candidate_authoritative_analysis(
          policy,
          analysis,
          source_plan=plan,
          source_plan_path=plan_path,
          source_plan_sha256=sha256_file(plan_path),
          manifest_path=DEFAULT_MANIFEST,
          trusted_policy_path=policy_path,
          trusted_analysis_path=analysis_path,
          trusted_template_path=untrusted_template)
      loader.assert_not_called()

  def test_historical_replay_still_rejects_altered_plan_commitments(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, _, _, _ = _candidate_plan(root)
      job_path = next((plan_dir / 'jobs').glob('*.json'))
      job = json.loads(job_path.read_text())
      job['identity']['candidate_k'] = 256
      _write_json(job_path, job)
      with self.assertRaisesRegex(ValueError, 'differs from its compiled plan'):
        _load_plan_for_analysis(
          plan_dir, require_current_repository_match=False)

  def test_candidate_replay_rejects_source_repository_drift_first(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      policy, plan_path, plan, _, _ = _policy_bundle(root)
      analysis = _analysis(policy, plan_path)
      altered = json.loads(json.dumps(plan))
      altered['repository']['sha'] = 'e' * 40
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_verify_authoritative_analysis') as verifier, \
          self.assertRaisesRegex(ValueError, 'repository differs from policy'):
        evaluate_candidate_k_analysis(
          policy,
          analysis,
          policy_sha256='a' * 64,
          analysis_sha256='b' * 64,
          source_plan=altered,
          source_plan_path=plan_path,
          source_plan_sha256=sha256_file(plan_path))
      verifier.assert_not_called()

  def test_candidate_evidence_verification_is_pure_and_rejects_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      policy, plan_path, plan, _, _ = _policy_bundle(root)
      policy_path = root / 'candidate-policy.json'
      _write_json(policy_path, policy)
      analysis = _analysis(policy, plan_path)
      analysis_path = root / 'analysis.json'
      _write_json(analysis_path, analysis)
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_verify_authoritative_analysis',
          return_value=analysis['source_integrity']), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_verify_policy_predates_source_jobs'):
        decision = evaluate_candidate_k_analysis(
          policy,
          analysis,
          policy_sha256=sha256_file(policy_path),
          analysis_sha256=sha256_file(analysis_path),
          source_plan=plan,
          source_plan_path=plan_path,
          source_plan_sha256=sha256_file(plan_path),
          created_utc='2026-08-30T18:00:00+00:00')
      decision_path = root / 'decision.json'
      _write_json(decision_path, decision)
      evidence = build_candidate_compiler_evidence(
        decision,
        'confirmation',
        policy_path=policy_path,
        analysis_path=analysis_path,
        source_plan_path=plan_path,
        decision_path=decision_path)
      evidence_path = root / 'promotion.json'
      _write_json(evidence_path, evidence)
      before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*') if path.is_file()}
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          'verify_pilot_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_verify_historical_candidate_authoritative_analysis',
          return_value=analysis['source_integrity']), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_verify_policy_predates_source_jobs'):
        verified = verify_candidate_compiler_evidence(
          evidence,
          evidence_path=evidence_path,
          promoted_suite='candidate_k_128_confirmation',
          manifest_path=DEFAULT_MANIFEST)
      after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*') if path.is_file()}
      self.assertEqual(verified, evidence)
      self.assertEqual(after, before)

      altered = json.loads(json.dumps(evidence))
      altered['commitments']['source_repository_sha'] = 'e' * 40
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          'verify_pilot_compiler_evidence',
          return_value=_parent_verification(plan)), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_verify_historical_candidate_authoritative_analysis',
          return_value=analysis['source_integrity']), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_verify_policy_predates_source_jobs'), self.assertRaisesRegex(
            ValueError, 'differs from canonical'):
        verify_candidate_compiler_evidence(
          altered,
          evidence_path=evidence_path,
          promoted_suite='candidate_k_128_confirmation',
          manifest_path=DEFAULT_MANIFEST)

  def test_cli_rejects_publication_aliases_without_filesystem_changes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      evidence_dir = root / 'not-created'
      aliased_output = (
        evidence_dir / 'candidate_k_128_confirmation-promotion.json')
      decision = {
        'routes': {
          'confirmation': {
            'promote': True,
            'target_suite': 'candidate_k_128_confirmation',
          },
        },
      }
      before = list(root.rglob('*'))
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_evaluate_trusted_historical_candidate_k_analysis',
          return_value=decision), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_exclusive_write_json') as writer, self.assertRaisesRegex(
            ValueError, 'destinations must be pairwise distinct'):
        candidate_promotion_main([
          '--policy', str(root / 'policy.json'),
          '--analysis', str(root / 'analysis.json'),
          '--source-plan', str(root / 'compiled-plan.json'),
          '--output', str(aliased_output),
          '--compiler-evidence-dir', str(evidence_dir),
        ])
      writer.assert_not_called()
      self.assertEqual(list(root.rglob('*')), before)

      pairwise_decision = {
        'routes': {
          'first': {'promote': True, 'target_suite': 'same_suite'},
          'second': {'promote': True, 'target_suite': 'same_suite'},
        },
      }
      with mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_evaluate_trusted_historical_candidate_k_analysis',
          return_value=pairwise_decision), mock.patch(
          'scripts.evaluate_candidate_k_promotion.'
          '_exclusive_write_json') as writer, self.assertRaisesRegex(
            ValueError, 'destinations must be pairwise distinct'):
        candidate_promotion_main([
          '--policy', str(root / 'policy.json'),
          '--analysis', str(root / 'analysis.json'),
          '--source-plan', str(root / 'compiled-plan.json'),
          '--output', str(root / 'decision.json'),
          '--compiler-evidence-dir', str(evidence_dir),
        ])
      writer.assert_not_called()
      self.assertEqual(list(root.rglob('*')), before)


if __name__ == '__main__':
  unittest.main()
