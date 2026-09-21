"""Multi-seed figures retain the complete development grid and test binding."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.fresh_pair_statistics import ARMS, select_checkpoints, summarize_test
from scripts.plot_staged_fresh import sha as file_sha
from scripts.plot_staged_fresh_multiseed import draw, read_inputs
from tests.test_fresh_pair_statistics import dev_records, make_test_records, observation, sha


class FreshMultiseedPlotTest(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.rows = dev_records(seeds=(1, 2, 3))
    self.rows += [observation(arm, seed, 0, document, rate, gain=0)
                  for seed in (1, 2, 3) for arm in ARMS
                  for document in ('dev-0', 'dev-1') for rate in (.25, .5)]
    self.selection = select_checkpoints(self.rows, excluded_test_document_ids=['train-0'])
    self.report = summarize_test(make_test_records(self.selection), self.selection,
                                 bootstrap_replicates=100)
    self.selection_path = self.root / 'selection.json'
    self.selection_path.write_text(json.dumps(self.selection))
    self.test_dir = self.root / 'test'
    self.test_dir.mkdir()
    (self.test_dir / 'results.json').write_text(json.dumps(self.report))
    self.run_dirs = []
    for seed in (1, 2, 3):
      directory = self.root / f'seed{seed}'
      directory.mkdir()
      protocol = {'training_config': {'seed': seed}, 'arguments': {'steps': 20, 'eval_every': 10}}
      (directory / 'protocol.json').write_text(json.dumps(protocol))
      run = {'completed': True, 'completed_steps': 20, 'target_steps': 20,
             'protocol_sha256': file_sha(directory / 'protocol.json'), 'evaluations': []}
      for step in (0, 10, 20):
        arms = {}
        for arm in ARMS:
          rows = [row for row in self.rows if row['training_seed'] == seed
                  and row['checkpoint_step'] == step and row['arm'] == arm]
          gain = sum(row['joint_log_probability'] - row['backbone_log_probability'] for row in rows)
          arms[arm] = {'joint_gain_nats_per_masked_token': gain / sum(row['active_token_count'] for row in rows)}
        run['evaluations'].append({'step': step, 'checkpoint_sha256': sha((seed, step)), 'arms': arms})
      (directory / 'results.json').write_text(json.dumps(run))
      self.run_dirs.append(directory)

  def read(self, directories=None):
    return read_inputs(self.run_dirs if directories is None else directories,
                       self.test_dir, self.selection_path)

  def test_all_seed_curves_and_crossed_intervals_are_preserved(self):
    output = self.root / 'figure.pdf'
    sidecar = draw(list(reversed(self.run_dirs)), self.test_dir, self.selection_path, output)
    self.assertEqual(sidecar['training_seeds'], [1, 2, 3])
    self.assertEqual(sidecar['selected_checkpoints'], self.selection['selected'])
    self.assertEqual(sidecar['test_overall'], self.report['overall'])
    self.assertTrue(sidecar['bootstrap']['training_seed_uncertainty_estimated'])
    self.assertEqual(len(sidecar['source_sha256']), 9)
    self.assertGreater(output.stat().st_size, 1000)
    self.assertGreater(output.with_suffix('.png').stat().st_size, 1000)

  def test_missing_and_duplicate_seed_curves_are_rejected(self):
    with self.assertRaisesRegex(ValueError, 'omit a selected'):
      self.read(self.run_dirs[:2])
    with self.assertRaisesRegex(ValueError, 'duplicate or add'):
      self.read(self.run_dirs + self.run_dirs[:1])

  def test_complete_checkpoint_grid_is_required_for_each_seed(self):
    path = self.run_dirs[1] / 'results.json'
    original = json.loads(path.read_text())
    for field, value in [('completed', False), ('evaluations', original['evaluations'][::2])]:
      run = copy.deepcopy(original)
      run[field] = value
      path.write_text(json.dumps(run))
      with self.assertRaises(ValueError):
        self.read()

  def test_mixed_selection_and_unmodeled_seed_uncertainty_fail(self):
    path = self.test_dir / 'results.json'
    for field, value in [('selection_sha256', '0' * 64),
                         ('bootstrap', {'training_seed_uncertainty_estimated': False})]:
      report = copy.deepcopy(self.report)
      report[field] = value
      path.write_text(json.dumps(report))
      with self.assertRaises(ValueError):
        self.read()


if __name__ == '__main__':
  unittest.main()
