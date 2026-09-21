import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.aggregate_hierarchical_document_eval import (
  PINNED_LEGACY_PLAN_SHA256,
  PINNED_LEGACY_REPOSITORY_SHA,
)
from scripts.compile_experiment_matrix import (
  DEFAULT_MANIFEST,
  compile_matrix,
  sha256_file,
)
from scripts.evaluate_experiment_promotion import (
  DEFAULT_POLICY,
  build_compiler_evidence,
  canonical_sha256,
  evaluate_pilot_analysis,
  load_and_validate_policy,
  verify_compiler_evidence,
  _verify_authoritative_analysis,
)


def _write_json(path, value):
  path.write_text(json.dumps(
    value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _source_plan(policy):
  contract = policy['analysis_contract']
  return {
    'schema_version': 1,
    'protocol_id': policy['protocol_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'artifact_root': '/mnt/contextual-forest',
    'selected_suites': [policy['source_suite']],
    'promotion_evidence': {},
    'plan_id': contract['source_plan_id'],
    'manifest_protocol_status': 'frozen_before_primary_results',
    'scientific_scope': 'test fixture',
    'repository': {
      'sha': contract['source_repository_sha'],
      'dirty': False,
    },
    'job_counts': {'eval': 1},
    'num_jobs': 1,
    'job_ids': ['eval--fixture'],
    'job_spec_sha256': {'eval--fixture': 'a' * 64},
  }


def _source_integrity(policy, *, plan_path=Path('/tmp/compiled-plan.json')):
  contract = policy['analysis_contract']
  output_sha = 'd' * 64
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
          (plan_path.parent / 'eval--fixture' / '_job_success.json').resolve()),
        'success_marker_sha256': 'c' * 64,
        'outputs': [{
          'name': 'conditional_records',
          'relative_path': 'records.jsonl',
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
    *,
    plan_path=Path('/tmp/compiled-plan.json'),
    improvement=0.02,
    pooled_ci_lower=0.005,
    candidate_recall=0.80,
    retained_mass=0.80,
):
  contract = policy['analysis_contract']
  candidate_k = contract['candidate_k']
  conditions = {}
  diagnostics = {}
  for dataset, dataset_spec in contract['datasets'].items():
    revision = dataset_spec['revision']
    for mask_rate in contract['mask_rates']:
      condition_key = (
        f'{dataset}|mask={mask_rate:.6f}|k={candidate_k}')
      conditions[condition_key] = {
        'dataset': dataset,
        'dataset_revision': revision,
        'mask_rate': mask_rate,
        'candidate_k': candidate_k,
        'mean_improvement': improvement,
        'ci_lower': improvement - 0.01,
        'ci_upper': improvement + 0.01,
      }
      diagnostic_key = str((dataset, revision, mask_rate, candidate_k))
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
  source_integrity = _source_integrity(policy, plan_path=plan_path)
  return {
    'schema_version': contract['analysis_schema_version'],
    'artifact': contract['analysis_artifact'],
    'created_utc': '2026-08-30T15:00:00+00:00',
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
      str(candidate_k): {
        'method': bootstrap['method'],
        'improvement_definition': (
          'baseline conditional NLL minus treatment'),
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
      'job_artifact_commitment_sha256': source_integrity[
        'commitment_sha256'],
    },
    'source_integrity': source_integrity,
  }


def _promotion_bundle(root, *, route_name='k128'):
  """Write a self-consistent v2 policy/analysis/plan/decision/evidence set."""
  base_policy = load_and_validate_policy(DEFAULT_POLICY)
  source_plan_path = root / 'compiled-plan.json'
  _write_json(source_plan_path, _source_plan(base_policy))

  policy_payload = yaml.safe_load(DEFAULT_POLICY.read_text())
  policy_payload['analysis_contract']['source_compiled_plan_sha256'] = \
    sha256_file(source_plan_path)
  policy_path = root / 'promotion-policy.yaml'
  policy_path.write_text(yaml.safe_dump(policy_payload, sort_keys=False))
  policy = load_and_validate_policy(policy_path)

  improvement = -0.02 if route_name == 'k128' else 0.02
  ci_lower = -0.04 if route_name == 'k128' else 0.005
  analysis = _analysis(
    policy, plan_path=source_plan_path,
    improvement=improvement, pooled_ci_lower=ci_lower)
  analysis_path = root / 'analysis.json'
  _write_json(analysis_path, analysis)
  with mock.patch(
      'scripts.evaluate_experiment_promotion._verify_authoritative_analysis',
      return_value=analysis['source_integrity']):
    decision = evaluate_pilot_analysis(
      policy,
      analysis,
      policy_sha256=sha256_file(policy_path),
      analysis_sha256=sha256_file(analysis_path),
      source_plan=_source_plan(policy),
      source_plan_path=source_plan_path,
      source_plan_sha256=sha256_file(source_plan_path),
      created_utc='2026-08-30T16:00:00+00:00')
  decision_path = root / 'decision.json'
  _write_json(decision_path, decision)
  evidence = build_compiler_evidence(
    decision,
    route_name,
    analysis_path=analysis_path,
    source_plan_path=source_plan_path,
    decision_path=decision_path)
  evidence_path = root / 'promotion.json'
  _write_json(evidence_path, evidence)
  return {
    'policy_path': policy_path,
    'analysis_path': analysis_path,
    'source_plan_path': source_plan_path,
    'decision_path': decision_path,
    'evidence_path': evidence_path,
    'evidence': evidence,
  }


class ExperimentPromotionTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.policy = load_and_validate_policy(DEFAULT_POLICY)
    cls.policy_sha = sha256_file(DEFAULT_POLICY)

  def _evaluate(self, analysis):
    with mock.patch(
        'scripts.evaluate_experiment_promotion._verify_authoritative_analysis',
        return_value=analysis['source_integrity']):
      return evaluate_pilot_analysis(
        self.policy,
        analysis,
        policy_sha256=self.policy_sha,
        analysis_sha256='b' * 64,
        source_plan=_source_plan(self.policy),
        source_plan_path=Path('/tmp/compiled-plan.json'),
        source_plan_sha256=self.policy['analysis_contract'][
          'source_compiled_plan_sha256'],
        created_utc='2026-08-30T16:00:00+00:00')

  def test_quality_positive_support_limited_routes_to_both_suites(self):
    decision = self._evaluate(_analysis(self.policy))

    self.assertEqual(
      decision['outcome'], 'promote_confirmation_and_k128')
    self.assertTrue(decision['routes']['confirmation']['promote'])
    self.assertTrue(decision['routes']['k128']['promote'])
    self.assertFalse(decision['gates']['k64_support_sufficient']['passed'])
    self.assertTrue(decision['gates']['k64_support_limited']['passed'])

  def test_support_limited_negative_pilot_is_k128_not_failed_promotion(self):
    decision = self._evaluate(_analysis(
      self.policy, improvement=-0.02, pooled_ci_lower=-0.04))

    self.assertEqual(decision['outcome'], 'promote_k128_only')
    self.assertFalse(decision['routes']['confirmation']['promote'])
    self.assertTrue(decision['routes']['k128']['promote'])

  def test_support_sufficient_quality_positive_routes_to_confirmation(self):
    decision = self._evaluate(_analysis(
      self.policy, candidate_recall=0.995, retained_mass=0.97))

    self.assertEqual(decision['outcome'], 'promote_confirmation_only')
    self.assertTrue(decision['routes']['confirmation']['promote'])
    self.assertFalse(decision['routes']['k128']['promote'])

  def test_support_sufficient_quality_negative_stops(self):
    decision = self._evaluate(_analysis(
      self.policy,
      improvement=-0.02,
      pooled_ci_lower=-0.04,
      candidate_recall=0.995,
      retained_mass=0.97))

    self.assertEqual(decision['outcome'], 'stop_after_pilot')
    self.assertFalse(any(
      route['promote'] for route in decision['routes'].values()))

  def test_pooled_signal_without_corpus_breadth_does_not_confirm(self):
    analysis = _analysis(
      self.policy, candidate_recall=0.995, retained_mass=0.97)
    conditions = analysis['by_candidate_k']['64']['conditions']
    negative_datasets = set(list(self.policy['analysis_contract']['datasets'])[:2])
    for condition in conditions.values():
      condition['mean_improvement'] = (
        -0.005 if condition['dataset'] in negative_datasets else 0.025)
    analysis['by_candidate_k']['64']['pooled']['mean_improvement'] = 0.01
    decision = self._evaluate(analysis)

    self.assertFalse(decision['gates']['corpus_breadth']['passed'])
    self.assertFalse(decision['gates']['high_mask_breadth']['passed'])
    self.assertFalse(decision['routes']['confirmation']['promote'])

  def test_missing_condition_fails_closed(self):
    analysis = _analysis(self.policy)
    conditions = analysis['by_candidate_k']['64']['conditions']
    conditions.pop(next(iter(conditions)))
    with self.assertRaisesRegex(ValueError, 'missing condition'):
      self._evaluate(analysis)

  def test_out_of_bounds_support_metric_fails_closed(self):
    analysis = _analysis(self.policy)
    diagnostic = next(iter(analysis['diagnostics'].values()))
    diagnostic['arms']['dynamic_dynamic']['candidate_recall'] = 1.01
    with self.assertRaisesRegex(ValueError, 'must be <= 1.0'):
      self._evaluate(analysis)

  def test_pooled_mean_must_match_equal_weight_conditions(self):
    analysis = _analysis(self.policy)
    analysis['by_candidate_k']['64']['pooled']['mean_improvement'] = 0.03
    with self.assertRaisesRegex(ValueError, 'pooled mean is inconsistent'):
      self._evaluate(analysis)

  def test_wrong_plan_or_manifest_fails_closed(self):
    analysis = _analysis(self.policy)
    analysis['compiled_plan']['plan_id'] = 'c' * 64
    with self.assertRaisesRegex(ValueError, 'plan ID differs'):
      self._evaluate(analysis)

  def test_unknown_policy_field_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'policy.yaml'
      payload = yaml.safe_load(DEFAULT_POLICY.read_text())
      payload['post_hoc_override'] = True
      path.write_text(yaml.safe_dump(payload, sort_keys=False))
      with self.assertRaisesRegex(ValueError, 'unknown=.*post_hoc_override'):
        load_and_validate_policy(path)

  def test_legacy_aggregator_allowlist_matches_trusted_policy(self):
    contract = self.policy['analysis_contract']
    self.assertEqual(
      PINNED_LEGACY_PLAN_SHA256,
      contract['source_compiled_plan_sha256'])
    self.assertEqual(
      PINNED_LEGACY_REPOSITORY_SHA,
      contract['source_repository_sha'])

  def test_k128_route_compiles_only_after_deterministic_reevaluation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      bundle = _promotion_bundle(root)
      evidence = bundle['evidence']
      self.assertEqual(
        evidence['promoted_suite'], 'candidate_k_128_pilot')
      self.assertEqual(set(evidence['criteria']), {
        'integrity_complete', 'k64_support_limited'})
      self.assertEqual(
        evidence['commitments']['canonical_decision_sha256'],
        canonical_sha256(json.loads(bundle['decision_path'].read_text())))
      with mock.patch.dict(
          'scripts.compile_experiment_matrix.TRUSTED_PROMOTION_POLICIES',
          {'contextual-forest-expansion-v1': bundle['policy_path']},
          clear=True), mock.patch(
          'scripts.compile_experiment_matrix._git_metadata',
          return_value={'sha': 'd' * 40, 'dirty': False}), mock.patch(
          'scripts.evaluate_experiment_promotion._verify_authoritative_analysis',
          return_value=json.loads(
            bundle['analysis_path'].read_text())['source_integrity']):
        plan, _, _ = compile_matrix(
          DEFAULT_MANIFEST,
          selected_suites=['candidate_k_128_pilot'],
          allowed_artifact_root=root,
          artifact_root_override=root / 'artifacts',
          output_dir=root / 'artifacts' / 'plan',
          promotion_evidence={
            'candidate_k_128_pilot': bundle['evidence_path']})
    self.assertIn('candidate_k_128_pilot', plan['promotion_evidence'])

  def _verify_bundle(self, root, bundle):
    analysis = json.loads(bundle['analysis_path'].read_text())
    with mock.patch(
        'scripts.evaluate_experiment_promotion._verify_authoritative_analysis',
        return_value=analysis['source_integrity']):
      return verify_compiler_evidence(
        json.loads(bundle['evidence_path'].read_text()),
        evidence_path=bundle['evidence_path'],
        promoted_suite='candidate_k_128_pilot',
        manifest_path=DEFAULT_MANIFEST,
        trusted_policy_path=bundle['policy_path'])

  def test_forged_criterion_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      bundle = _promotion_bundle(root)
      evidence = json.loads(bundle['evidence_path'].read_text())
      evidence['criteria']['invented_quality_gate'] = True
      _write_json(bundle['evidence_path'], evidence)
      with self.assertRaisesRegex(ValueError, 'canonical reevaluated evidence'):
        self._verify_bundle(root, bundle)

  def test_invalid_or_forged_timestamp_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      bundle = _promotion_bundle(root)
      evidence = json.loads(bundle['evidence_path'].read_text())
      evidence['created_utc'] = 'yesterday'
      _write_json(bundle['evidence_path'], evidence)
      with self.assertRaisesRegex(ValueError, 'ISO-8601 UTC timestamp'):
        self._verify_bundle(root, bundle)

  def test_forged_source_plan_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      bundle = _promotion_bundle(root)
      source_plan = json.loads(bundle['source_plan_path'].read_text())
      source_plan['repository']['sha'] = 'c' * 40
      _write_json(bundle['source_plan_path'], source_plan)
      with self.assertRaisesRegex(ValueError, 'source plan SHA256 mismatch'):
        self._verify_bundle(root, bundle)

  def test_forged_repository_commitment_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      bundle = _promotion_bundle(root)
      evidence = json.loads(bundle['evidence_path'].read_text())
      evidence['commitments']['source_repository_sha'] = 'c' * 40
      _write_json(bundle['evidence_path'], evidence)
      with self.assertRaisesRegex(ValueError, 'canonical reevaluated evidence'):
        self._verify_bundle(root, bundle)

  def test_hand_authored_analysis_is_rejected_against_recomputed_records(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      base_policy = load_and_validate_policy(DEFAULT_POLICY)
      plan_path = root / 'compiled-plan.json'
      _write_json(plan_path, _source_plan(base_policy))
      policy_payload = yaml.safe_load(DEFAULT_POLICY.read_text())
      policy_payload['analysis_contract']['source_compiled_plan_sha256'] = \
        sha256_file(plan_path)
      policy_path = root / 'policy.yaml'
      policy_path.write_text(yaml.safe_dump(policy_payload, sort_keys=False))
      policy = load_and_validate_policy(policy_path)
      authoritative = _analysis(
        policy, plan_path=plan_path, improvement=-0.02,
        pooled_ci_lower=-0.04)
      hand_authored = _analysis(
        policy, plan_path=plan_path, improvement=0.02,
        pooled_ci_lower=0.005)
      context = {
        'comparison': {
          'baseline': policy['analysis_contract']['baseline_arm'],
          'treatment': policy['analysis_contract']['treatment_arm'],
        },
        'manifest': {'analysis': {'bootstrap_seed': 1701}},
        'source_integrity': authoritative['source_integrity'],
      }
      with mock.patch(
          'scripts.aggregate_hierarchical_document_eval.load_plan_records',
          return_value=([], context)) as loader, mock.patch(
          'scripts.aggregate_hierarchical_document_eval.aggregate_records',
          return_value={'schema_version': 1}), mock.patch(
          'scripts.aggregate_hierarchical_document_eval.bind_analysis_to_source',
          return_value=authoritative), self.assertRaisesRegex(
            ValueError, 'differs from deterministic recomputation'):
        _verify_authoritative_analysis(
          policy,
          hand_authored,
          source_plan=_source_plan(policy),
          source_plan_path=plan_path,
          source_plan_sha256=sha256_file(plan_path),
          manifest_path=DEFAULT_MANIFEST)
      self.assertNotIn(
        'require_current_repository_match', loader.call_args.kwargs)

  def test_authoritative_verifier_has_no_repository_match_escape_hatch(self):
    with self.assertRaisesRegex(TypeError, 'unexpected keyword argument'):
      _verify_authoritative_analysis(
        {}, {}, source_plan={}, source_plan_path=Path('/tmp/missing.json'),
        source_plan_sha256='a' * 64, manifest_path=DEFAULT_MANIFEST,
        require_current_repository_match=0)

  def test_ineligible_route_cannot_emit_evidence(self):
    decision = self._evaluate(_analysis(
      self.policy, candidate_recall=0.995, retained_mass=0.97))
    with self.assertRaisesRegex(ValueError, 'not eligible'):
      build_compiler_evidence(
        decision,
        'k128',
        analysis_path=Path('/tmp/analysis.json'),
        source_plan_path=Path('/tmp/compiled-plan.json'),
        decision_path=Path('/tmp/decision.json'))


if __name__ == '__main__':
  unittest.main()
