import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.aggregate_hierarchical_document_eval import (
  _expected_record_count,
  _load_plan_for_analysis,
  _record_cross_cell_dataset_provenance,
  _validated_analysis_marker,
  aggregate_records,
  load_plan_records,
  validate_record,
)
from scripts.run_compiled_job import _job_digest, _job_execution_digest


def _digest(text):
  return hashlib.sha256(text.encode()).hexdigest()


def _records():
  rows = []
  for dataset in ('wiki-pinned', 'arxiv-pinned'):
    for mask_rate in (0.5, 0.9):
      for train_seed in (1, 2):
        for corruption_seed in (201, 202):
          pairing = _digest(
            f'{dataset}|{mask_rate}|{corruption_seed}|corruption')
          for arm in ('static_static', 'dynamic_dynamic'):
            for document_index in (10, 20):
              for chunk_index in (0, 1):
                masked_tokens = 4 + chunk_index
                baseline_nll = 2.0 + 0.1 * (document_index == 20)
                per_token = (
                  baseline_nll if arm == 'static_static'
                  else baseline_nll - 0.5)
                rows.append({
                  'schema_version': 1,
                  'protocol_id': 'contextual-forest-expansion-v1',
                  'job_id': (
                    f'{arm}-{train_seed}-{corruption_seed}-{dataset}'),
                  'arm': arm,
                  'train_seed': train_seed,
                  'corruption_seed': corruption_seed,
                  'dataset': dataset,
                  'dataset_revision': 'a' * 40,
                  'mask_rate': mask_rate,
                  'candidate_k': 64,
                  'rank': 0,
                  'batch_index': document_index,
                  'example_index': 0,
                  'document_id': f'{dataset}:{document_index}',
                  'document_index': document_index,
                  'document_sha256': _digest(
                    f'{dataset}:{document_index}'),
                  'chunk_index': chunk_index,
                  'nll_sum': per_token * masked_tokens,
                  'masked_tokens': masked_tokens,
                  'candidate_hits': masked_tokens - 1,
                  'retained_mass_sum': 0.8 * masked_tokens,
                  'pairing_digest_sha256': pairing,
                })
  return rows


def _wiki_provenance(processed_num_sequences=197):
  return {
    'specification': {
      'logical_dataset_name': 'wikitext103-pinned',
      'dataset_name_or_path': 'Salesforce/wikitext',
      'dataset_config_name': 'wikitext-103-raw-v1',
      'source_split': 'validation',
      'source_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
      'source_num_rows': 3760,
      'source_window': None,
      'document_boundary_mode': 'wikitext_articles',
    },
    'observed': {
      'source_num_rows': 3760,
      'window_num_rows': 3760,
      'document_num_rows_after_boundary_recovery': 60,
      'processed_num_sequences': processed_num_sequences,
      'raw_fingerprint': 'wiki-raw-fingerprint',
      'window_fingerprint': 'wiki-window-fingerprint',
      'processed_fingerprint': '866bd647e4df8268',
    },
    'specification_sha256': _digest('wiki-specification'),
  }


class HierarchicalDocumentEvaluationTest(unittest.TestCase):

  def test_public_record_loader_is_strict_and_has_no_escape_hatch(self):
    sentinel = RuntimeError('stop after checking forwarded arguments')
    with mock.patch(
        'scripts.aggregate_hierarchical_document_eval._load_plan_records_core',
        side_effect=sentinel) as loader:
      with self.assertRaisesRegex(RuntimeError, 'forwarded arguments'):
        load_plan_records(
          Path('/tmp/plan'), manifest_path=Path('/tmp/manifest.yaml'),
          suite_name='pilot', comparison_name='contextual-vs-static')
      self.assertIs(
        loader.call_args.kwargs['require_current_repository_match'], True)

    with self.assertRaisesRegex(TypeError, 'unexpected keyword argument'):
      load_plan_records(
        Path('/tmp/plan'), manifest_path=Path('/tmp/manifest.yaml'),
        suite_name='pilot', comparison_name='contextual-vs-static',
        require_current_repository_match=False)

  def _legacy_job(self, root):
    return {
      'schema_version': 1,
      'protocol_id': 'contextual-forest-expansion-v1',
      'source_manifest_sha256': 'a' * 64,
      'plan_id': 'b' * 64,
      'job_id': 'eval--fixture',
      'kind': 'eval',
      'artifact_dir': str(root / 'artifact'),
      'suites': ['pilot'],
      'dependencies': [],
      'identity': {},
      'argv': ['python', 'main.py'],
      'execution_mode': 'fresh_attempt',
      'external_inputs': [],
      'required_outputs': [{
        'name': 'conditional_records',
        'pattern': 'records.jsonl',
        'exactly_one': True,
      }],
    }

  def test_exact_policy_pinned_legacy_plan_loads_without_checkout_switch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      job = self._legacy_job(root)
      (root / 'jobs').mkdir()
      (root / 'jobs' / 'eval--fixture.json').write_text(json.dumps(job))
      plan = {
        'schema_version': 1,
        'protocol_id': job['protocol_id'],
        'source_manifest_sha256': job['source_manifest_sha256'],
        'artifact_root': str(root),
        'selected_suites': ['pilot'],
        'promotion_evidence': {},
        'plan_id': job['plan_id'],
        'manifest_protocol_status': 'frozen_before_primary_results',
        'scientific_scope': 'fixture',
        'repository': {'sha': 'c' * 40, 'dirty': False},
        'job_counts': {'eval': 1},
        'num_jobs': 1,
        'job_ids': [job['job_id']],
        'job_spec_sha256': {job['job_id']: _job_digest(job)},
      }
      plan_path = root / 'compiled-plan.json'
      plan_path.write_text(json.dumps(plan))
      plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
      loaded_plan, loaded_jobs, observed_sha, legacy = \
        _load_plan_for_analysis(
          root,
          expected_legacy_plan_sha256=plan_sha,
          expected_legacy_repository_sha='c' * 40)
      self.assertTrue(legacy)
      self.assertEqual(observed_sha, plan_sha)
      self.assertEqual(loaded_plan, plan)
      self.assertEqual(loaded_jobs[job['job_id']], job)

  def test_legacy_marker_requires_presence_and_unchanged_outputs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      job = self._legacy_job(root)
      with self.assertRaisesRegex(FileNotFoundError, 'incomplete'):
        _validated_analysis_marker(job, legacy=True, required=True)

      run_dir = root / 'artifact' / 'attempts' / 'attempt-0001'
      run_dir.mkdir(parents=True)
      output_path = run_dir / 'records.jsonl'
      output_path.write_text('{}\n')
      output_record = {
        'name': 'conditional_records',
        'relative_path': 'records.jsonl',
        'size_bytes': output_path.stat().st_size,
        'sha256': hashlib.sha256(output_path.read_bytes()).hexdigest(),
      }
      marker = {
        'schema_version': 1,
        'artifact': 'compiled_experiment_job_success',
        'job_id': job['job_id'],
        'originating_plan_id': job['plan_id'],
        'job_execution_sha256': _job_execution_digest(job),
        'run_dir': str(run_dir),
        'argv': job['argv'],
        'start_time_utc': '2026-08-30T00:00:00+00:00',
        'end_time_utc': '2026-08-30T00:01:00+00:00',
        'outputs': [output_record],
      }
      marker_path = root / 'artifact' / '_job_success.json'
      marker_path.write_text(json.dumps(marker))
      self.assertEqual(
        _validated_analysis_marker(job, legacy=True, required=True), marker)
      output_path.write_text('{"tampered":true}\n')
      with self.assertRaisesRegex(ValueError, 'outputs drifted'):
        _validated_analysis_marker(job, legacy=True, required=True)

  def test_averages_corruptions_and_chunks_before_hierarchical_bootstrap(self):
    result = aggregate_records(
      _records(),
      baseline_arm='static_static',
      treatment_arm='dynamic_dynamic',
      protocol_id='contextual-forest-expansion-v1',
      suite_name='test-suite',
      comparison_name='contextual-vs-static',
      num_resamples=500,
      rng_seed=77,
      confidence_level=0.95,
      timestamp_utc='2026-08-30T00:00:00+00:00')

    bootstrap = result['by_candidate_k']['64']
    self.assertAlmostEqual(bootstrap['pooled']['mean_improvement'], 0.5)
    self.assertAlmostEqual(bootstrap['pooled']['ci_lower'], 0.5)
    self.assertAlmostEqual(bootstrap['pooled']['ci_upper'], 0.5)
    self.assertEqual(bootstrap['num_train_seeds'], 2)
    self.assertEqual(bootstrap['num_strata'], 4)
    self.assertIn('average corruption replications', bootstrap['nesting'][0])
    self.assertIn('no diffusion ELBO', result['scope_note'])

  def test_candidate_k_base_seed_produces_frozen_final_seed(self):
    records = _records()
    for row in records:
      row['candidate_k'] = 128
    result = aggregate_records(
      records,
      baseline_arm='static_static',
      treatment_arm='dynamic_dynamic',
      protocol_id='contextual-forest-expansion-v1',
      suite_name='candidate_k_128_pilot',
      comparison_name='contextual-vs-static',
      num_resamples=10,
      rng_seed=1701,
      confidence_level=0.95,
      timestamp_utc='2026-08-30T00:00:00+00:00')
    self.assertEqual(result['by_candidate_k']['128']['rng_seed'], 1829)

  def test_rejects_pairing_digest_mismatch_across_training_seeds(self):
    rows = _records()
    rows[0]['pairing_digest_sha256'] = 'f' * 64
    with self.assertRaisesRegex(ValueError, 'pairing digests differ'):
      aggregate_records(
        rows,
        baseline_arm='static_static',
        treatment_arm='dynamic_dynamic',
        protocol_id='contextual-forest-expansion-v1',
        suite_name='test-suite',
        comparison_name='contextual-vs-static',
        num_resamples=10)

  def test_rejects_incomplete_document_factorial_cell(self):
    rows = _records()
    rows = [
      row for row in rows
      if not (
        row['arm'] == 'dynamic_dynamic'
        and row['train_seed'] == 2
        and row['corruption_seed'] == 202
        and row['dataset'] == 'wiki-pinned'
        and row['mask_rate'] == 0.5
        and row['document_index'] == 20)]
    with self.assertRaisesRegex(ValueError, 'window identities differ'):
      aggregate_records(
        rows,
        baseline_arm='static_static',
        treatment_arm='dynamic_dynamic',
        protocol_id='contextual-forest-expansion-v1',
        suite_name='test-suite',
        comparison_name='contextual-vs-static',
        num_resamples=10)

  def test_record_schema_rejects_zero_masked_tokens(self):
    row = _records()[0]
    row['masked_tokens'] = 0
    with self.assertRaisesRegex(ValueError, 'must be positive'):
      validate_record(row)

  def test_record_target_rejects_short_non_wiki_dataset(self):
    with self.assertRaisesRegex(
        ValueError, 'arxiv.*requires 512'):
      _expected_record_count(
        compiled_dataset='scientific_papers_arxiv',
        suite_name='pilot',
        target_windows=512,
        provenance={
          'specification': {},
          'observed': {'processed_num_sequences': 511},
        },
        provenance_path=Path('/tmp/arxiv-provenance.json'))

  def test_record_target_allows_only_exact_full_pinned_wikitext_split(self):
    self.assertEqual(
      _expected_record_count(
        compiled_dataset='wikitext103',
        suite_name='confirmation',
        target_windows=2000,
        provenance=_wiki_provenance(),
        provenance_path=Path('/tmp/wiki-provenance.json')),
      197)

    partial = _wiki_provenance()
    partial['specification']['source_window'] = [0, 100]
    partial['observed']['window_num_rows'] = 100
    with self.assertRaisesRegex(ValueError, 'only the exact fully consumed'):
      _expected_record_count(
        compiled_dataset='wikitext103',
        suite_name='pilot',
        target_windows=512,
        provenance=partial,
        provenance_path=Path('/tmp/wiki-partial-provenance.json'))

    for observed_field, bad_value in (
        ('processed_num_sequences', 1),
        ('document_num_rows_after_boundary_recovery', 1),
        ('document_num_rows_after_boundary_recovery', 59)):
      forged = _wiki_provenance()
      forged['observed'][observed_field] = bad_value
      with self.subTest(field=observed_field, value=bad_value), \
          self.assertRaisesRegex(ValueError, 'only the exact fully consumed'):
        _expected_record_count(
          compiled_dataset='wikitext103',
          suite_name='pilot',
          target_windows=512,
          provenance=forged,
          provenance_path=Path('/tmp/wiki-forged-provenance.json'))

  def test_rejects_cross_cell_fingerprint_drift_or_missing_fingerprint(self):
    commitments = {}
    first = _wiki_provenance()
    _record_cross_cell_dataset_provenance(
      commitments,
      compiled_dataset='wikitext103',
      provenance=first,
      provenance_path=Path('/tmp/wiki-cell-a.json'))
    drifted = _wiki_provenance()
    drifted['observed']['processed_fingerprint'] = 'different-fingerprint'
    with self.assertRaisesRegex(
        ValueError, 'preprocessing identity differs'):
      _record_cross_cell_dataset_provenance(
        commitments,
        compiled_dataset='wikitext103',
        provenance=drifted,
        provenance_path=Path('/tmp/wiki-cell-b.json'))

    missing = _wiki_provenance()
    missing['observed']['raw_fingerprint'] = None
    with self.assertRaisesRegex(ValueError, 'non-empty raw_fingerprint'):
      _record_cross_cell_dataset_provenance(
        {}, compiled_dataset='wikitext103', provenance=missing,
        provenance_path=Path('/tmp/wiki-missing-fingerprint.json'))


if __name__ == '__main__':
  unittest.main()
