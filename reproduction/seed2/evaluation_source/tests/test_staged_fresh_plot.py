"""Figures preserve selected-checkpoint identities and paired interval signs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.fresh_pair_statistics import ARMS, select_checkpoints, summarize_test
from scripts.plot_staged_fresh import draw, read_inputs
from tests.test_fresh_pair_statistics import dev_records, make_test_records, observation, sha


class FreshPlotTest(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.run_dir, self.test_dir = self.root / 'train', self.root / 'test'
    self.run_dir.mkdir()
    self.test_dir.mkdir()
    self.rows = dev_records(seeds=(1,))
    self.rows += [observation(arm, 1, 0, document, rate, gain=0)
                  for arm in ARMS for document in ('dev-0', 'dev-1') for rate in (.25, .5)]
    self.selection = select_checkpoints(self.rows, excluded_test_document_ids=['train-0'])
    self.report = summarize_test(make_test_records(self.selection), self.selection,
                                 bootstrap_replicates=100)
    self.run = {'completed': True, 'completed_steps': 20, 'target_steps': 20, 'evaluations': []}
    for step in (0, 10, 20):
      arms = {}
      for arm in ARMS:
        rows = [row for row in self.rows if row['checkpoint_step'] == step and row['arm'] == arm]
        gain = sum(row['joint_log_probability'] - row['backbone_log_probability'] for row in rows)
        arms[arm] = {'joint_gain_nats_per_masked_token': gain / sum(row['active_token_count'] for row in rows)}
      self.run['evaluations'].append({'step': step, 'checkpoint_sha256': sha((1, step)), 'arms': arms})
    self.selection_path = self.root / 'selection.json'
    self.write()

  def write(self):
    for path, data in ((self.run_dir / 'results.json', self.run),
                       (self.test_dir / 'results.json', self.report),
                       (self.selection_path, self.selection)):
      path.write_text(json.dumps(data))

  def read(self):
    return read_inputs(self.run_dir, self.test_dir, self.selection_path)

  def test_plot_carries_exact_selected_test_values_and_source_hashes(self):
    output = self.root / 'figure.pdf'
    sidecar = draw(self.run_dir, self.test_dir, self.selection_path, output)
    self.assertEqual(sidecar['test_overall'], self.report['overall'])
    self.assertEqual(sidecar['selected_checkpoints'], self.selection['selected'])
    self.assertEqual(len(sidecar['source_sha256']), 4)
    self.assertTrue(all(len(value) == 64 for value in sidecar['source_sha256'].values()))
    self.assertGreater(output.stat().st_size, 1000)
    self.assertGreater(output.with_suffix('.png').stat().st_size, 1000)

  def test_incomplete_or_unordered_training_is_rejected(self):
    original = copy.deepcopy(self.run)
    for change in ({'completed': False}, {'completed_steps': 10},
                   {'evaluations': original['evaluations'][1:]},
                   {'evaluations': list(reversed(original['evaluations']))}):
      self.run = {**original, **change}
      self.write()
      with self.assertRaises(ValueError):
        self.read()

  def test_mismatched_selection_checkpoint_and_intervals_are_rejected(self):
    self.report['selection_sha256'] = '0' * 64
    self.write()
    with self.assertRaisesRegex(ValueError, 'authenticated selection'):
      self.read()
    self.report['selection_sha256'] = self.selection['selection_sha256']
    self.run['evaluations'][1]['checkpoint_sha256'] = '0' * 64
    self.write()
    with self.assertRaisesRegex(ValueError, 'not in the development curve'):
      self.read()
    self.run['evaluations'][1]['checkpoint_sha256'] = sha((1, 10))
    metric = self.report['overall']['metrics_nats_per_masked_token']['directional_vs_unary']
    metric['ci95'] = [1, -1]
    self.write()
    with self.assertRaisesRegex(ValueError, 'invalid paired test interval'):
      self.read()


if __name__ == '__main__':
  unittest.main()
