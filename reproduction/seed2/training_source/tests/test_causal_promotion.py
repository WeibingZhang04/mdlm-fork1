import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.aggregate_causal_denoising_eval import (
  _contrasts,
  _merge_source_integrity,
)
from scripts.compile_experiment_matrix import (
  build_jobs,
  load_and_validate_manifest,
  sha256_file,
)
from scripts.evaluate_causal_promotion import evaluate_causal_analysis
from scripts.evaluate_experiment_promotion import canonical_sha256
from scripts.finalize_causal_smoke_policy import (
  DEFAULT_MANIFEST,
  DEFAULT_TEMPLATE,
  TECHNICAL_GATE_NAMES,
  _read_mapping,
  _validate_template,
  finalize_policy,
)


def _write_json(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(
    payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _causal_plan(root):
  template = _validate_template(
    _read_mapping(DEFAULT_TEMPLATE, context='template'),
    manifest_path=DEFAULT_MANIFEST)
  manifest = load_and_validate_manifest(DEFAULT_MANIFEST)
  plan_dir = root / 'plan'
  artifact_root = root / 'artifacts'
  repository = {'sha': 'd' * 40, 'dirty': False}
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


def _policy_bundle(root):
  plan_dir, plan, jobs, template = _causal_plan(root)
  with mock.patch(
      'scripts.finalize_causal_smoke_policy._git_metadata',
      return_value=plan['repository']):
    policy = finalize_policy(
      DEFAULT_TEMPLATE,
      plan_dir,
      manifest_path=DEFAULT_MANIFEST,
      frozen_utc='2026-08-31T05:00:00+00:00')
  return policy, plan_dir / 'compiled-plan.json', plan, jobs, template


def _source_integrity(policy, plan_path, plan):
  jobs = {}
  for job_id in sorted(plan['job_ids']):
    output_sha = canonical_sha256({'job_id': job_id})
    jobs[job_id] = {
      'job_spec_sha256': plan['job_spec_sha256'][job_id],
      'job_execution_sha256': 'b' * 64,
      'success_marker_path': str(
        (plan_path.parent / job_id / '_job_success.json').resolve()),
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
    }
  contract = policy['analysis_contract']
  body = {
    'schema_version': 1,
    'source_compiled_plan_path': str(plan_path.resolve()),
    'source_compiled_plan_sha256': contract['source_compiled_plan_sha256'],
    'source_plan_id': contract['source_plan_id'],
    'source_manifest_sha256': policy['source_manifest_sha256'],
    'source_repository_sha': contract['source_repository_sha'],
    'source_repository_clean': True,
    'validated_job_ids': sorted(plan['job_ids']),
    'jobs': jobs,
  }
  return {**body, 'commitment_sha256': canonical_sha256(body)}


def _bootstrap(contract, *, rng_seed, estimate, bounded=False):
  conditions = {}
  for dataset, dataset_spec in contract['datasets'].items():
    for mask_rate in contract['mask_rates']:
      key = f'{dataset}|mask={mask_rate:.6f}|adapter_k=128'
      lower = max(0.0, estimate - 0.01) if bounded else estimate - 0.01
      upper = min(1.0, estimate + 0.01) if bounded else estimate + 0.01
      conditions[key] = {
        'dataset': dataset,
        'dataset_revision': dataset_spec['revision'],
        'mask_rate': mask_rate,
        'adapter_candidate_k': contract['candidate_k'],
        'estimate': estimate,
        'ci_lower': lower,
        'ci_upper': upper,
      }
  lower = max(0.0, estimate - 0.01) if bounded else estimate - 0.01
  upper = min(1.0, estimate + 0.01) if bounded else estimate + 0.01
  return {
    'method': contract['bootstrap']['method'],
    'estimand': 'fixture estimand',
    'nesting': [
      'average corruption replications within source document',
      'resample adapter training seeds with replacement',
      'resample source documents within sampled adapter seed',
      'equal-weight frozen dataset x mask-rate strata',
    ],
    'top_level_resampling_unit': 'adapter_training_seed',
    'num_adapter_seeds': len(contract['train_seeds']),
    'num_strata': contract['expected_num_strata'],
    'num_resamples': contract['bootstrap']['num_resamples'],
    'rng': 'NumPy Generator(PCG64)',
    'rng_seed': rng_seed,
    'confidence_level': contract['bootstrap']['confidence_level'],
    'pooled': {
      'estimate': estimate,
      'ci_lower': lower,
      'ci_upper': upper,
    },
    'conditions': conditions,
  }


def _analysis(policy, plan_path, plan, *, contrast_estimate=-0.5,
              topology_gate=True):
  contract = policy['analysis_contract']
  integrity = _source_integrity(policy, plan_path, plan)
  contrasts = {}
  contrast_terms = dict(_contrasts())
  for index, name in enumerate(contract['expected_contrasts']):
    contrasts[name] = {
      'name': name,
      'terms': [{
        'arm': term.arm,
        'metric': term.metric,
        'coefficient': term.coefficient,
      } for term in contrast_terms[name]],
      'analysis': _bootstrap(
        contract,
        rng_seed=contract['bootstrap']['base_rng_seed'] + index * 101,
        estimate=contrast_estimate),
    }
  support = {}
  for candidate_k in contract['support_candidate_ks']:
    estimate = 0.5 + 0.4 * candidate_k / max(
      contract['support_candidate_ks'])
    support[str(candidate_k)] = {
      metric: _bootstrap(
        contract,
        rng_seed=(contract['bootstrap']['base_rng_seed'] + 10_000
                  + candidate_k * 10 + offset),
        estimate=estimate,
        bounded=True)
      for offset, metric in enumerate((
        'candidate_recall', 'retained_unary_mass'))
    }
  topology_conditions = {}
  for arm in contract['controls']:
    for dataset, dataset_spec in contract['datasets'].items():
      for mask_rate in contract['mask_rates']:
        key = f'{arm}|{dataset}|mask={mask_rate:.6f}|adapter_k=128'
        topology_conditions[key] = {
          'arm': arm,
          'dataset': dataset,
          'dataset_revision': dataset_spec['revision'],
          'mask_rate': mask_rate,
          'adapter_candidate_k': contract['candidate_k'],
          'num_records': 10,
          'masked_tokens': 100,
          'selected_edges': 50,
          'selected_edges_per_masked_token': 0.5,
          'nonempty': True,
        }
  fraction = 1.0 if topology_gate else 0.5
  changed = 50 if topology_gate else 25
  permutation_conditions = {}
  for dataset, dataset_spec in contract['datasets'].items():
    for mask_rate in contract['mask_rates']:
      key = f'{dataset}|mask={mask_rate:.6f}|adapter_k=128'
      permutation_conditions[key] = {
        'dataset': dataset,
        'dataset_revision': dataset_spec['revision'],
        'mask_rate': mask_rate,
        'adapter_candidate_k': contract['candidate_k'],
        'num_records': 10,
        'degree_sequence_preserved_records': 10,
        'component_sizes_preserved_records': 10,
        'degree_sequence_preserved_every_record': True,
        'component_sizes_preserved_every_record': True,
        'selected_edges': 50,
        'changed_edges': changed,
        'changed_edge_fraction': fraction,
        'minimum_changed_edge_fraction': 0.90,
        'passed': topology_gate,
      }
  body = {
    'schema_version': 2,
    'artifact': contract['analysis_artifact'],
    'created_utc': '2026-08-31T07:00:00+00:00',
    'protocol_id': policy['protocol_id'],
    'suite': policy['source_suite'],
    'objective': contract['objective'],
    'scope_note': (
      'Conditional denoising only; no diffusion ELBO, likelihood, '
      'perplexity, or generation-quality quantity is inferred.'),
    'source_views': {
      name: {
        'compiled_plan_sha256': contract['source_compiled_plan_sha256'],
        'source_manifest_sha256': policy['source_manifest_sha256'],
        'source_repository_sha': contract['source_repository_sha'],
        'source_integrity_commitment_sha256': 'f' * 64,
      } for name in contract['expected_source_views']
    },
    'contrasts': contrasts,
    'candidate_support': {
      'arm': 'dynamic_dynamic',
      'support_candidate_ks': contract['support_candidate_ks'],
      'by_candidate_k': support,
    },
    'topology_permutation_diagnostic': {
      'arm': 'dynamic_dynamic',
      'estimand': 'edge_weighted_fraction_of_selected_edges_reassigned',
      'pooled': {
        'num_records': 40,
        'degree_sequence_preserved_records': 40,
        'component_sizes_preserved_records': 40,
        'degree_sequence_preserved_every_record': True,
        'component_sizes_preserved_every_record': True,
        'selected_edges': 200,
        'changed_edges': changed * 4,
        'changed_edge_fraction': fraction,
        'minimum_changed_edge_fraction': 0.95,
        'passed': topology_gate,
      },
      'conditions': permutation_conditions,
      'gate': {
        'degree_sequence_preserved_every_record': True,
        'component_sizes_preserved_every_record': True,
        'pooled_fraction_passed': topology_gate,
        'every_condition_fraction_passed': topology_gate,
        'passed': topology_gate,
      },
    },
    'technical_diagnostics': {
      'expected_arms': contract['controls'],
      'observed_arms': sorted(contract['controls']),
      'num_records': 100,
      'finite_statistics': {'passed': True},
      'pairing': {
        'expected_train_seeds': contract['train_seeds'],
        'observed_train_seeds': contract['train_seeds'],
        'num_conditions': (
          len(contract['corruption_seeds'])
          * contract['expected_num_strata']),
        'mismatched_conditions': 0,
        'num_paired_units': 10,
        'expected_arm_train_cells_per_unit': (
          len(contract['controls']) * len(contract['train_seeds'])),
        'incomplete_paired_units': 0,
        'masked_token_mismatched_units': 0,
        'passed': True,
      },
      'no_edge_identity': {
        'maximum_absolute_error': 0.0,
        'absolute_tolerance': policy['technical_gates'][
          'no_edge_absolute_tolerance'],
        'passed': True,
      },
      'candidate_support': {
        'expected_candidate_ks': contract['support_candidate_ks'],
        'grid_complete': True,
        'monotonicity_absolute_tolerance': policy['technical_gates'][
          'support_monotonicity_absolute_tolerance'],
        'monotone_within_tolerance': True,
      },
      'topology_structure': {
        'conditions': topology_conditions,
        'every_condition_nonempty': True,
        'passed': True,
      },
    },
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
  return {**body, 'analysis_sha256': canonical_sha256(body)}


class CausalPromotionTest(unittest.TestCase):

  def test_source_view_integrity_union_must_cover_complete_plan(self):
    plan_path = Path('/tmp/causal-fixture/compiled-plan.json')
    plan = {
      'plan_id': 'a' * 64,
      'source_manifest_sha256': 'b' * 64,
      'repository': {'sha': 'c' * 40, 'dirty': False},
      'job_ids': ['job-a', 'job-b', 'job-c'],
    }

    def context(job_ids):
      jobs = {job_id: {'job_id': job_id} for job_id in job_ids}
      return {
        'plan': plan,
        'plan_sha256': 'd' * 64,
        'source_integrity': {
          'schema_version': 1,
          'source_compiled_plan_path': str(plan_path),
          'source_compiled_plan_sha256': 'd' * 64,
          'source_plan_id': plan['plan_id'],
          'source_manifest_sha256': plan['source_manifest_sha256'],
          'source_repository_sha': plan['repository']['sha'],
          'source_repository_clean': True,
          'validated_job_ids': sorted(job_ids),
          'jobs': jobs,
          'commitment_sha256': 'e' * 64,
        },
      }

    merged = _merge_source_integrity({
      'first': context(['job-a', 'job-b']),
      'second': context(['job-b', 'job-c']),
    })
    self.assertEqual(
      merged['validated_job_ids'], ['job-a', 'job-b', 'job-c'])
    self.assertEqual(set(merged['jobs']), set(plan['job_ids']))
    with self.assertRaisesRegex(ValueError, 'complete compiled plan'):
      _merge_source_integrity({'incomplete': context(['job-a', 'job-b'])})

  def test_template_is_four_arm_and_technical_only(self):
    template = _validate_template(
      _read_mapping(DEFAULT_TEMPLATE, context='template'),
      manifest_path=DEFAULT_MANIFEST)
    self.assertEqual(template['analysis_contract']['controls'], [
      'dynamic_dynamic', 'fixed_dynamic',
      'dynamic_fixed', 'static_static'])
    self.assertEqual(
      template['source_manifest_sha256'], sha256_file(DEFAULT_MANIFEST))
    serialized = json.dumps(template['technical_gates'], sort_keys=True)
    for forbidden in ('mean_improvement', 'nll_sign', 'ci_lower_above'):
      self.assertNotIn(forbidden, serialized)
    self.assertEqual(
      template['routing']['primary']['requires'], TECHNICAL_GATE_NAMES)
    self.assertEqual(
      template['technical_gates']['no_edge_absolute_tolerance'], 1e-10)
    self.assertEqual(
      template['technical_gates'][
        'support_monotonicity_absolute_tolerance'], 1e-6)
    self.assertIn(
      'no_edge_identity_within_frozen_tolerance', TECHNICAL_GATE_NAMES)
    self.assertIn(
      'topology_degree_sequence_preserved', TECHNICAL_GATE_NAMES)

  def test_finalizer_binds_exact_40_job_smoke_before_execution(self):
    with tempfile.TemporaryDirectory() as directory:
      policy, _, plan, _, _ = _policy_bundle(Path(directory))
    self.assertEqual(plan['job_counts'], {
      'eval': 32, 'export': 4, 'train': 4})
    self.assertEqual(policy['freeze_attestation']['num_jobs_checked'], 40)
    self.assertEqual(policy['source_plan']['promotion_evidence'], {})

  def test_finalizer_refuses_nonempty_source_artifact_directory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan_dir, plan, jobs, _ = _causal_plan(root)
      artifact_dir = Path(next(iter(jobs.values()))['artifact_dir'])
      artifact_dir.mkdir(parents=True)
      (artifact_dir / 'started.txt').write_text('started\n')
      with mock.patch(
          'scripts.finalize_causal_smoke_policy._git_metadata',
          return_value=plan['repository']), self.assertRaisesRegex(
          RuntimeError, 'execution began'):
        finalize_policy(DEFAULT_TEMPLATE, plan_dir)

  def _evaluate(self, root, **analysis_kwargs):
    policy, plan_path, plan, _, _ = _policy_bundle(root)
    analysis = _analysis(policy, plan_path, plan, **analysis_kwargs)
    with mock.patch(
        'scripts.evaluate_causal_promotion._git_metadata',
        return_value=plan['repository']), mock.patch(
        'scripts.evaluate_causal_promotion._verify_authoritative_analysis',
        return_value=analysis['source_integrity']), mock.patch(
        'scripts.evaluate_causal_promotion._verify_temporal_order'):
      return evaluate_causal_analysis(
        policy,
        analysis,
        policy_sha256='a' * 64,
        analysis_sha256='b' * 64,
        source_plan=plan,
        source_plan_path=plan_path,
        source_plan_sha256=sha256_file(plan_path),
        manifest_path=DEFAULT_MANIFEST,
        created_utc='2026-08-31T08:00:00+00:00')

  def test_negative_nll_contrasts_still_promote_when_technical_gates_pass(self):
    with tempfile.TemporaryDirectory() as directory:
      decision = self._evaluate(
        Path(directory), contrast_estimate=-100.0)
    self.assertEqual(decision['outcome'], 'promote_causal_primary')
    self.assertTrue(decision['routes']['primary']['promote'])
    self.assertTrue(all(
      decision['routes']['primary']['criteria'].values()))

  def test_topology_technical_failure_stops_primary_route(self):
    with tempfile.TemporaryDirectory() as directory:
      decision = self._evaluate(Path(directory), topology_gate=False)
    self.assertEqual(decision['outcome'], 'stop_for_technical_remediation')
    self.assertFalse(decision['routes']['primary']['promote'])
    self.assertFalse(
      decision['routes']['primary']['criteria'][
        'topology_permutation_gate'])


if __name__ == '__main__':
  unittest.main()
