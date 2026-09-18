import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_ccf_confirmation import load_selection, confirmation_args
from scripts.evaluate_ccf_selected_five import generation_args

ROOT = Path(__file__).resolve().parents[1]


class ConfirmationTest(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            path.write_text(json.dumps(data))
            return load_selection(path, hashlib.sha256(path.read_bytes()).hexdigest())[1]

    def setUp(self):
        self.selection = json.loads(
            (ROOT / 'experiments/confirmation-4.json').read_text())

    def test_selection_covers_three_checkpoints_and_fresh_baseline_per_budget(self):
        cells = self.load(self.selection)
        self.assertEqual(len(cells), 19 * len(self.selection['sampling_budgets']))
        for budget in self.selection['sampling_budgets']:
            group = [c for c in cells if c['sampling_steps'] == budget]
            self.assertEqual(len(group), 19)
            self.assertEqual(sum(c['mode'] == 'factorized' for c in group), 1)
            for family in ('B', 'C', 'D'):
                for arm in ('fixed_dynamic', 'dynamic_dynamic'):
                    self.assertEqual(len({c['step'] for c in group if c['family'] == family and c['arm'] == arm}), 3)

    def test_only_sample_count_and_seed_differ_from_matching_pilot(self):
        for cell in self.load(self.selection):
            before = generation_args(dict(cell, num_samples=20), Path('/tmp/unused'), 'a', 'm')
            after = confirmation_args(cell, Path('/tmp/unused'), 'a', 'm')
            before[before.index('--num-samples') + 1] = '100'
            before[before.index('--base-seed') + 1] = '100001'
            self.assertEqual(before, after)
            self.assertEqual(after[after.index('--nfe-budgets') + 1], str(cell['sampling_steps'] + 1))

    def test_duplicate_missing_baseline_invalid_checkpoint_and_seed_are_rejected(self):
        duplicate = copy.deepcopy(self.selection)
        duplicate['cells'].append(duplicate['cells'][0])
        no_baseline = copy.deepcopy(self.selection)
        no_baseline['cells'] = [c for c in no_baseline['cells'] if c['family'] != 'A']
        bad_checkpoint = copy.deepcopy(self.selection)
        bad_checkpoint['cells'][1]['step'] = 8000
        reused_seed = copy.deepcopy(self.selection)
        reused_seed['base_seed'] = 91001
        for bad in (duplicate, no_baseline, bad_checkpoint, reused_seed):
            with self.subTest(bad=bad['cells'][0]):
                with self.assertRaises((ValueError, KeyError)):
                    self.load(bad)

    def test_selection_tampering_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_selection(
                ROOT / 'experiments/confirmation-4.json', '0' * 64)

    def test_completed_confirmation_selection_is_also_valid(self):
        selection = json.loads(
            (ROOT / 'experiments/confirmation-8-16-32.json').read_text())
        cells = self.load(selection)
        self.assertEqual(len(cells), 57)
        self.assertEqual({c['sampling_steps'] for c in cells}, {8, 16, 32})


if __name__ == '__main__':
    unittest.main()
