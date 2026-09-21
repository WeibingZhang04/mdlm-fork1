import math
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.report_staged_support_floor import main, support_floor


class StagedSupportFloorTest(unittest.TestCase):

  def test_exact_floor_and_absorbing_mask_exclusion(self):
    probabilities = torch.tensor([0.5, 0.3, 0.15, 0.05, 0.0], dtype=torch.float64)
    logits = probabilities.log().expand(1, 4, -1).clone()
    targets = torch.tensor([[0, 2, 3, 1]])
    active = torch.tensor([[True, True, True, False]])
    result = support_floor(logits, targets, active, top_k=2, mask_index=4)
    expected = -math.log(0.15 / 0.2) - math.log(0.05 / 0.2)
    self.assertAlmostEqual(result['nll_floor_sum'], expected, places=12)
    self.assertEqual(result['active_tokens'], 3)
    self.assertEqual(result['candidate_hits'], 1)
    self.assertEqual(result['residual_targets'], 2)
    self.assertEqual(result['per_position_nll_floor'][0][0], 0)
    self.assertEqual(result['per_position_nll_floor'][0][3], 0)
    self.assertLessEqual(result['nll_floor_sum'], result['backbone_nll_sum'])

  def test_full_finite_support_has_zero_floor(self):
    logits = torch.tensor([[[3.0, 2.0, 1.0, -torch.inf]]])
    result = support_floor(logits, torch.tensor([[2]]), torch.tensor([[True]]), 3, mask_index=3)
    self.assertEqual(result['nll_floor_sum'], 0)
    self.assertEqual(result['residual_targets'], 0)
    self.assertEqual(result['active_rows_with_boundary_ties'], 0)

  def test_boundary_ties_are_reported(self):
    logits = torch.tensor([[[2.0, 1.0, 1.0, 0.0, -torch.inf]]])
    result = support_floor(logits, torch.tensor([[1]]), torch.tensor([[True]]), 2, mask_index=4)
    self.assertEqual(result['active_rows_with_boundary_ties'], 1)
    self.assertEqual(result['targets_tied_at_boundary'], 1)

  def test_empty_active_mask_and_tiny_tail(self):
    logits = torch.tensor([[[0.0, -1000.0, -1001.0, -torch.inf]]], dtype=torch.float64)
    empty = support_floor(logits, torch.tensor([[1]]), torch.tensor([[False]]), 1, mask_index=3)
    self.assertEqual(empty['active_tokens'], 0)
    self.assertEqual(empty['nll_floor_sum'], 0)
    result = support_floor(logits, torch.tensor([[1]]), torch.tensor([[True]]), 1, mask_index=3)
    self.assertAlmostEqual(result['nll_floor_sum'], math.log1p(math.exp(-1.0)), places=12)

  def test_rejects_invalid_clean_mask_target_and_nonfinite_logits(self):
    logits = torch.tensor([[[1.0, 0.0, -torch.inf]]])
    with self.assertRaisesRegex(ValueError, 'clean target'):
      support_floor(logits, torch.tensor([[2]]), torch.tensor([[True]]), 1, mask_index=2)
    with self.assertRaisesRegex(ValueError, 'mask must have logit'):
      support_floor(torch.zeros_like(logits), torch.tensor([[0]]), torch.tensor([[True]]), 1, mask_index=2)
    logits[..., 0] = torch.nan
    with self.assertRaisesRegex(ValueError, 'NaN'):
      support_floor(logits, torch.tensor([[0]]), torch.tensor([[True]]), 1)

  def test_cli_reads_cache_and_writes_aggregate_without_modifying_records(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      cache = root / 'cache'
      cache.mkdir()
      for index in range(2):
        torch.save({'logits': torch.tensor([[2.0, 1.0, 0.0, -torch.inf]]),
                    'targets': torch.tensor([index + 1]),
                    'active': torch.tensor([True]), 'corrupted': torch.tensor([3])},
                   cache / f'example-{index:04d}.pt')
      before = {path.name: path.read_bytes() for path in cache.iterdir()}
      output = root / 'floor.json'
      main(['--cache-dir', str(cache), '--output', str(output),
            '--device', 'cpu', '--batch-size', '2', '--top-k', '1'])
      result = json.loads(output.read_text())
      self.assertEqual(result['examples'], 2)
      self.assertEqual(result['active_tokens'], 2)
      self.assertEqual(result['residual_targets'], 2)
      self.assertEqual(len(result['cache_files']), 2)
      self.assertAlmostEqual(result['nll_floor_per_masked_token'],
                             0.5 * (math.log1p(math.exp(-1)) + math.log1p(math.exp(1))))
      self.assertEqual(before, {path.name: path.read_bytes() for path in cache.iterdir()})


if __name__ == '__main__':
  unittest.main()
