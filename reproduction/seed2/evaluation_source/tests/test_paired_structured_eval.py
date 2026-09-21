import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.aggregate_paired_structured_eval import (
  RunSpec,
  aggregate_runs,
  find_metrics_csv,
  paired_bootstrap_mean_ci,
  read_last_complete_metrics,
)
from structured_pairing import (
  StructuredValidationPairingDigest,
  combine_rank_records,
  write_pairing_digest,
)


FIELDNAMES = [
  'epoch',
  'step',
  'trainer/loss',
  'val/structured/conditional_nll_per_masked_token',
  'val/structured/candidate_recall_epoch',
  'val/structured/retained_unary_mass',
  'val/structured/active_fraction_epoch',
]


def _write_metrics(
    run_dir: Path,
    *,
    nll: float,
    candidate_recall: float = 0.91,
    retained_mass: float = 0.72,
    active_fraction: float = 0.35,
    previous_nll: float | None = None) -> Path:
  metrics_path = run_dir / 'lightning_logs' / 'version_0' / 'metrics.csv'
  metrics_path.parent.mkdir(parents=True)
  rows = [{'epoch': 0, 'step': 3, 'trainer/loss': 4.2}]
  if previous_nll is not None:
    rows.extend([
      {
        'epoch': 0,
        'step': 4,
        'val/structured/conditional_nll_per_masked_token': previous_nll,
        'val/structured/candidate_recall_epoch': candidate_recall,
      },
      {
        'epoch': 0,
        'step': 4,
        'val/structured/retained_unary_mass': retained_mass,
        'val/structured/active_fraction_epoch': active_fraction,
      },
    ])
  # Mimic CSVLogger's sparse rows from separate logging calls at one event.
  rows.extend([
    {
      'epoch': 1,
      'step': 8,
      'val/structured/conditional_nll_per_masked_token': nll,
      'val/structured/candidate_recall_epoch': candidate_recall,
    },
    {
      'epoch': 1,
      'step': 8,
      'val/structured/retained_unary_mass': retained_mass,
      'val/structured/active_fraction_epoch': active_fraction,
    },
  ])
  with metrics_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
  return metrics_path


def _pairing_inputs(offset: int = 0) -> dict[str, torch.Tensor]:
  return {
    'clean_input_ids': torch.tensor([[1 + offset, 2, 3]]),
    'attention_mask': torch.tensor([[1, 1, 1]], dtype=torch.bool),
    'sampled_times': torch.tensor([0.25]),
    'corrupted_input_ids': torch.tensor([[1 + offset, 9, 9]]),
    'active_mask': torch.tensor([[0, 1, 1]], dtype=torch.bool),
  }


def _write_pairing_digest(
    run_dir: Path,
    *,
    offset: int = 0,
    epoch: int = 1,
    step: int = 8,
    sanity_checking: bool = False) -> dict:
  digest = StructuredValidationPairingDigest()
  digest.update(**_pairing_inputs(offset))
  payload = combine_rank_records(
    [digest.rank_record(0)], epoch=epoch, step=step,
    sanity_checking=sanity_checking)
  write_pairing_digest(str(run_dir), payload)
  return payload


class PairedStructuredEvaluationTest(unittest.TestCase):

  def test_reads_last_complete_sparse_lightning_event(self):
    with tempfile.TemporaryDirectory() as directory:
      metrics_path = _write_metrics(
        Path(directory), nll=1.25, previous_nll=2.75)
      result = read_last_complete_metrics(metrics_path)

    self.assertEqual(result['epoch'], 1)
    self.assertEqual(result['step'], 8)
    self.assertEqual(result['last_csv_row'], 6)
    self.assertAlmostEqual(
      result['conditional_nll_per_masked_token'], 1.25)
    self.assertAlmostEqual(result['candidate_recall'], 0.91)
    self.assertAlmostEqual(result['retained_mass'], 0.72)
    self.assertAlmostEqual(result['active_fraction'], 0.35)

  def test_aggregates_paired_improvements_and_hashes_checkpoint(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      runs = []
      for seed, baseline_nll, treatment_nll in (
          (9, 2.0, 1.6), (10, 2.2, 1.5)):
        baseline = root / f'baseline-{seed}'
        treatment = root / f'treatment-{seed}'
        _write_metrics(baseline, nll=baseline_nll)
        _write_metrics(treatment, nll=treatment_nll)
        runs.extend([
          RunSpec('factorized', seed, baseline),
          RunSpec('contextual', seed, treatment),
        ])
      checkpoint = root / 'backbone.pt'
      checkpoint.write_bytes(b'test-checkpoint')
      expected_metrics_hash = hashlib.sha256(
        (root / 'baseline-9/lightning_logs/version_0/metrics.csv').
        read_bytes()).hexdigest()
      manifest = aggregate_runs(
        runs,
        baseline_arm='factorized',
        treatment_arm='contextual',
        protocol_id='frozen-owt-v1',
        protocol_metadata={'split': 'validation'},
        checkpoints=[('backbone', checkpoint)],
        repo_root=root,
        source_path_root=root,
        timestamp_utc='2026-08-30T00:00:00+00:00')

    self.assertEqual(manifest['seeds'], [9, 10])
    self.assertEqual(manifest['num_pairs'], 2)
    self.assertAlmostEqual(
      manifest['mean_conditional_nll_improvement_per_masked_token'], 0.55)
    self.assertAlmostEqual(
      manifest['pairs'][0][
        'conditional_nll_improvement_per_masked_token'], 0.4)
    self.assertEqual(
      manifest['checkpoints'][0]['sha256'],
      hashlib.sha256(b'test-checkpoint').hexdigest())
    self.assertEqual(manifest['checkpoints'][0]['path'], 'backbone.pt')
    self.assertEqual(
      manifest['pairs'][0]['baseline']['run_path'], 'baseline-9')
    self.assertEqual(
      manifest['pairs'][0]['baseline']['metrics_csv_sha256'],
      expected_metrics_hash)
    self.assertEqual(
      manifest['paired_bootstrap']['method'],
      'paired_seed_bootstrap_percentile')
    self.assertEqual(manifest['paired_bootstrap']['num_resamples'], 20_000)
    self.assertEqual(manifest['paired_bootstrap']['rng_seed'], 1701)
    self.assertIsNone(manifest['repository']['git_sha'])
    self.assertIn('not computed or inferred', manifest['scope_note'])
    json.dumps(manifest)

  def test_rejects_pairing_metric_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      baseline = root / 'baseline'
      treatment = root / 'treatment'
      _write_metrics(baseline, nll=2.0, active_fraction=0.35)
      _write_metrics(treatment, nll=1.5, active_fraction=0.36)
      with self.assertRaisesRegex(ValueError, 'not paired for active_fraction'):
        aggregate_runs(
          [
            RunSpec('factorized', 9, baseline),
            RunSpec('contextual', 9, treatment),
          ],
          baseline_arm='factorized',
          treatment_arm='contextual',
          protocol_id='frozen-owt-v1',
          repo_root=root,
          pairing_tolerance=1e-4)

  def test_requires_identical_cryptographic_pairing_digest(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      baseline = root / 'baseline'
      treatment = root / 'treatment'
      _write_metrics(baseline, nll=2.0)
      _write_metrics(treatment, nll=1.5)
      expected = _write_pairing_digest(baseline)
      _write_pairing_digest(treatment)
      expected_digest_file_hash = hashlib.sha256(
        (baseline / 'validation_pairing_digest.json').
        read_bytes()).hexdigest()
      manifest = aggregate_runs(
        [
          RunSpec('factorized', 9, baseline),
          RunSpec('contextual', 9, treatment),
        ],
        baseline_arm='factorized',
        treatment_arm='contextual',
        protocol_id='frozen-owt-v1',
        repo_root=root,
        source_path_root=root,
        require_pairing_digest=True)

    pair = manifest['pairs'][0]
    self.assertEqual(pair['pairing_digest_sha256'], expected['sha256'])
    self.assertEqual(
      pair['baseline']['pairing_digest'],
      pair['treatment']['pairing_digest'])
    self.assertEqual(
      pair['baseline']['pairing_digest_path'],
      'baseline/validation_pairing_digest.json')
    self.assertEqual(
      pair['baseline']['pairing_digest_file_sha256'],
      expected_digest_file_hash)
    self.assertTrue(manifest['protocol']['pairing_digest_required'])

  def test_rejects_different_validation_commitments(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      baseline = root / 'baseline'
      treatment = root / 'treatment'
      _write_metrics(baseline, nll=2.0)
      _write_metrics(treatment, nll=1.5)
      _write_pairing_digest(baseline, offset=0)
      _write_pairing_digest(treatment, offset=1)
      with self.assertRaisesRegex(ValueError, 'pairing digests differ'):
        aggregate_runs(
          [
            RunSpec('factorized', 9, baseline),
            RunSpec('contextual', 9, treatment),
          ],
          baseline_arm='factorized',
          treatment_arm='contextual',
          protocol_id='frozen-owt-v1',
          repo_root=root,
          require_pairing_digest=True)

  def test_rejects_digest_for_different_validation_event(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      baseline = root / 'baseline'
      treatment = root / 'treatment'
      _write_metrics(baseline, nll=2.0)
      _write_metrics(treatment, nll=1.5)
      _write_pairing_digest(baseline, step=99)
      _write_pairing_digest(treatment, step=99)
      with self.assertRaisesRegex(
          ValueError, 'does not match selected metrics event'):
        aggregate_runs(
          [
            RunSpec('factorized', 9, baseline),
            RunSpec('contextual', 9, treatment),
          ],
          baseline_arm='factorized',
          treatment_arm='contextual',
          protocol_id='frozen-owt-v1',
          repo_root=root,
          require_pairing_digest=True)

  def test_rejects_ambiguous_run_directory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_metrics(root / 'first', nll=2.0)
      _write_metrics(root / 'second', nll=2.0)
      with self.assertRaisesRegex(ValueError, 'multiple metrics.csv'):
        find_metrics_csv(root)

  def test_paired_bootstrap_is_deterministic_and_configurable(self):
    first = paired_bootstrap_mean_ci(
      [0.2, 0.4, 0.9], num_resamples=1_000, rng_seed=1701,
      confidence_level=0.95)
    second = paired_bootstrap_mean_ci(
      [0.2, 0.4, 0.9], num_resamples=1_000, rng_seed=1701,
      confidence_level=0.95)

    self.assertEqual(first, second)
    self.assertEqual(first['num_seed_pairs'], 3)
    self.assertEqual(first['num_resamples'], 1_000)
    self.assertEqual(first['rng_seed'], 1701)
    self.assertAlmostEqual(first['ci_lower'], 0.2)
    self.assertAlmostEqual(first['ci_upper'], 0.9)

  def test_pairing_digest_is_ordered_and_covers_every_required_field(self):
    base = _pairing_inputs()

    def digest_for(batches):
      digest = StructuredValidationPairingDigest()
      for batch in batches:
        digest.update(**batch)
      return digest.rank_record(0)['sha256']

    reference = digest_for([base])
    for field in base:
      changed = {name: value.clone() for name, value in base.items()}
      if field == 'sampled_times':
        changed[field][0] += 0.125
      elif field in {'attention_mask', 'active_mask'}:
        changed[field][0, 0] = ~changed[field][0, 0]
      else:
        changed[field][0, 0] += 1
      self.assertNotEqual(
        reference, digest_for([changed]), msg=f'unhashed field: {field}')

    second = _pairing_inputs(offset=4)
    self.assertNotEqual(
      digest_for([base, second]), digest_for([second, base]))


if __name__ == '__main__':
  unittest.main()
