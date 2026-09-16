"""Independent checks of the paper revision's immutable scores and new contrasts.

The independent bootstrap below does not call the production pairing, metric,
or aggregation helpers. It uses saved dependence scores and resampling weights
rather than reconstructing dependence and indexing repeated observations.
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

from scripts import plot_joint_paper_revision as revision


def read_json(path):
  return json.loads(path.read_text())


def independent_contrasts(rows, rate, *, replicates, bootstrap_seed):
  """Recompute token-weighted paired contrasts with a crossed bootstrap."""
  seeds = sorted({row['training_seed'] for row in rows})
  documents = sorted({(row['dataset'], row['document_id']) for row in rows})
  rates = sorted({row['mask_rate'] for row in rows})
  arms = ('shared', 'directional', 'unary')
  records = {}
  for row in rows:
    key = (row['training_seed'], (row['dataset'], row['document_id']),
           row['mask_rate'], row['arm'])
    if key in records:
      raise AssertionError('duplicate raw score')
    records[key] = row
  assert len(records) == len(seeds) * len(documents) * len(rates) * len(arms)
  selected_rates = rates if rate is None else [rate]
  totals = np.zeros((len(seeds), len(documents), 5), dtype=np.float64)
  counts = np.zeros(totals.shape[:2], dtype=np.int64)
  max_score_identity_error = 0.
  for si, seed in enumerate(seeds):
    for di, document in enumerate(documents):
      for mask_rate in selected_rates:
        tied, separate, independent = [records[seed, document, mask_rate, arm] for arm in arms]
        for field in ('active_token_count', 'corruption_seed', 'mask_sha256',
                      'clean_token_sha256', 'candidate_ids_sha256'):
          assert tied[field] == separate[field] == independent[field]
        for row in (tied, separate, independent):
          assert row['joint_log_probability'] <= 0
          assert row['marginal_log_probability'] <= 0
          error = abs(row['joint_log_probability'] - row['marginal_log_probability']
                      - row['dependence_log_probability'])
          max_score_identity_error = max(max_score_identity_error, error)
        marginal = separate['marginal_log_probability'] - tied['marginal_log_probability']
        dependence = separate['dependence_log_probability'] - tied['dependence_log_probability']
        joint = separate['joint_log_probability'] - tied['joint_log_probability']
        totals[si, di] += [-tied['marginal_log_probability'],
                          -separate['marginal_log_probability'], marginal, dependence, joint]
        counts[si, di] += tied['active_token_count']
  assert max_score_identity_error < 1e-10
  point = np.sum(totals, axis=(0, 1)) / np.sum(counts)
  strata = [[index for index, document in enumerate(documents) if document[0] == dataset]
            for dataset in sorted({document[0] for document in documents})]
  rng = np.random.default_rng(bootstrap_seed)
  draws = []
  for _ in range(replicates):
    seed_draw = rng.integers(len(seeds), size=len(seeds))
    document_draw = np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                                    for indices in strata])
    seed_weights = np.bincount(seed_draw, minlength=len(seeds))
    document_weights = np.bincount(document_draw, minlength=len(documents))
    weights = np.outer(seed_weights, document_weights)
    denominator = np.sum(counts * weights)
    draws.append(np.einsum('sdm,sd->m', totals, weights) / denominator)
  bounds = np.percentile(np.asarray(draws), [2.5, 97.5], axis=0)
  return point, bounds, int(counts.sum()), max_score_identity_error


class JointPaperRevisionTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.base = revision.BASE
    root = cls.base / 'fresh-three-seeds'
    cls.records = [json.loads(line) for line in (root / 'test/test-records.jsonl').read_text().splitlines()]
    cls.selection = read_json(root / 'selection/selection.json')
    cls.report = read_json(root / 'test/results.json')
    cls.sidecar = read_json(revision.REPO_ROOT / 'artifacts/paper/expert-revision-v1/between-model-decomposition.json')
    cls.runs, _, _, cls.sources = revision.read_inputs(
      [cls.base / 'fresh-pilot/training', root / 'seed2/training', root / 'seed3/training'],
      root / 'test', root / 'selection/selection.json')

  def test_all_issued_input_bytes_are_pinned(self):
    for filename, digest in revision.PINS.items():
      with self.subTest(filename=filename):
        self.assertEqual(hashlib.sha256((self.base / filename).read_bytes()).hexdigest(), digest)
    self.assertEqual(len(self.records), 4608)

  def test_independent_raw_score_bootstrap_reproduces_all_new_contrasts(self):
    names = ['tied_marginal_product_nll', 'separate_marginal_product_nll',
             'marginal_improvement', 'dependence_improvement', 'joint_improvement']
    for rate in (None, .25, .5, .75, .9):
      with self.subTest(mask_rate=rate):
        expected = self.sidecar['overall'] if rate is None else self.sidecar['by_mask_rate'][str(rate)]
        point, bounds, count, error = independent_contrasts(
          self.records, rate, replicates=self.report['bootstrap']['replicates'],
          bootstrap_seed=self.report['bootstrap']['seed'])
        metrics = expected['metrics_nats_per_masked_token']
        np.testing.assert_allclose(point, [metrics[name]['estimate'] for name in names], rtol=0, atol=1e-14)
        # Weighting unique observations and summing repeated observations use
        # different floating-point addition orders (~1e-14 at NLL around 6.5).
        np.testing.assert_allclose(bounds.T, [metrics[name]['ci95'] for name in names], rtol=0, atol=2e-14)
        self.assertEqual(count, expected['active_tokens_per_arm'])
        self.assertLess(error, 1e-10)
        self.assertAlmostEqual(point[2] + point[3], point[4], places=13)
        self.assertAlmostEqual(point[0] - point[1], point[2], places=13)
        self.assertAlmostEqual(point[3] / point[4], expected['dependence_fraction_of_mean_joint_improvement'], places=11)

  def test_canonical_full_report_reproduces_exactly(self):
    actual = revision.summarize_test(self.records, self.selection,
      bootstrap_replicates=self.report['bootstrap']['replicates'],
      bootstrap_seed=self.report['bootstrap']['seed'])
    for name, value in actual.items():
      self.assertEqual(value, self.report[name], name)

  def test_changed_pinned_input_fails_before_output_creation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      first_pin = next(iter(revision.PINS))
      path = root / first_pin
      path.parent.mkdir(parents=True)
      path.write_bytes((self.base / first_pin).read_bytes() + b'\n')
      with mock.patch.object(revision, 'BASE', root):
        with self.assertRaisesRegex(ValueError, 'immutable artifact hash mismatch'):
          revision.main(['--figure-dir', str(root / 'figures'), '--artifact-dir', str(root / 'outputs')])
      self.assertFalse((root / 'figures').exists())
      self.assertFalse((root / 'outputs').exists())

  def test_decomposition_rejects_missing_rate_seed_and_arm(self):
    mutations = [
      ('mask-rate grid', [row for row in self.records if row['mask_rate'] != .25]),
      ('omit a selected training seed', [row for row in self.records if row['training_seed'] != 3]),
      ('missing a paired arm', self.records[1:]),
    ]
    for message, rows in mutations:
      with self.subTest(mutation=message), tempfile.TemporaryDirectory() as directory:
        with self.assertRaisesRegex(ValueError, message):
          revision.decomposition(rows, self.selection, self.report, self.sources, Path(directory))
        self.assertFalse((Path(directory) / 'between-model-decomposition.json').exists())

  def test_decomposition_rejects_a_changed_canonical_comparison(self):
    changed = copy.deepcopy(self.report)
    changed['overall']['metrics_nats_per_masked_token']['directional_vs_shared']['estimate'] += .0001
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaises(AssertionError):
        revision.decomposition(self.records, self.selection, changed, self.sources, Path(directory))
      self.assertFalse((Path(directory) / 'between-model-decomposition.json').exists())

  def test_mask_rate_plot_preserves_every_estimate_and_interval(self):
    with mock.patch.object(revision, 'save_figure') as save:
      revision.mask_rates(self.report, self.sources, Path('.'), Path('.'))
    fig = save.call_args.args[0]
    self.addCleanup(plt.close, fig)
    groups = [self.report['overall']] + [self.report['by_mask_rate'][str(rate)] for rate in (.25, .5, .75, .9)]
    for ax, metric_name in zip(fig.axes, ('directional_vs_unary', 'directional_dependence_gain')):
      self.assertEqual([label.get_text() for label in ax.get_yticklabels()],
                       ['Pooled', '25%', '50%', '75%', '90%'])
      for index, group in enumerate(groups):
        metric = group['metrics_nats_per_masked_token'][metric_name]
        np.testing.assert_array_equal(ax.lines[index].get_xdata(), [metric['estimate']])
        np.testing.assert_array_equal(ax.collections[index].get_segments()[0][:, 0], metric['ci95'])
    metadata = save.call_args.args[-1]
    self.assertEqual(metadata['test_overall'], self.report['overall'])
    self.assertEqual(metadata['test_by_mask_rate'], self.report['by_mask_rate'])

  def test_development_plot_preserves_all_curves_and_selected_checkpoints(self):
    with mock.patch.object(revision, 'save_figure') as save:
      revision.development(self.runs, self.selection, self.sources, Path('.'), Path('.'))
    fig = save.call_args.args[0]
    self.addCleanup(plt.close, fig)
    for (seed, run), ax in zip(sorted(self.runs.items()), fig.axes):
      selected = {row['arm']: row for row in self.selection['selected'] if row['training_seed'] == seed}
      for index, arm in enumerate(revision.ARMS):
        curve, marker = ax.lines[2 * index:2 * index + 2]
        np.testing.assert_array_equal(curve.get_xdata(), [row['step'] for row in run['evaluations']])
        np.testing.assert_array_equal(curve.get_ydata(),
          [row['arms'][arm]['joint_gain_nats_per_masked_token'] for row in run['evaluations']])
        point = next(row for row in run['evaluations'] if row['step'] == selected[arm]['checkpoint_step'])
        np.testing.assert_array_equal(marker.get_xdata(), [point['step']])
        np.testing.assert_array_equal(marker.get_ydata(), [point['arms'][arm]['joint_gain_nats_per_masked_token']])
    self.assertEqual(save.call_args.args[-1]['selected_checkpoints'], self.selection['selected'])


if __name__ == '__main__':
  unittest.main()
