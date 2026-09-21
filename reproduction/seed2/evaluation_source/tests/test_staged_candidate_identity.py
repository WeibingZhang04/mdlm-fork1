"""The audit reproduces both original top-K paths and detects tied swaps."""

import tempfile
from pathlib import Path
import unittest

import torch

from scripts.audit_staged_candidate_identity import (
  audit_batch, audit_split, compare_candidate_ids, original_pair_candidates,
  original_unary_candidates,
)


class StagedCandidateIdentityTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(948)
    # Quantization creates many exact ties, resembling cached BF16 outputs.
    self.logits = torch.randn(4, 20, 101).mul(2).round().float()
    self.logits[..., -1] = -torch.inf
    self.active = torch.rand(4, 20) < 0.6
    self.targets = torch.randint(100, (4, 20))

  def test_original_paths_match_direct_statements_with_partial_last_chunk(self):
    pair_values, pair_ids = original_pair_candidates(self.logits, 8)
    torch.testing.assert_close(pair_ids, self.logits.topk(8, dim=-1).indices)
    values, ids, selected, shapes = original_unary_candidates(self.logits, self.active, 8, 32)
    expected = torch.cat([
      self.logits.reshape(-1, 101)[indices].float().topk(8, dim=-1).indices
      for indices in selected.split(32)])
    torch.testing.assert_close(ids, expected)
    self.assertLess(shapes[-1][0], 32)
    report = audit_batch(self.logits, self.targets, self.active, top_k=8)
    self.assertEqual(report['ordered_id_differing_rows'], int((pair_ids[self.active] != ids).any(-1).sum()))
    self.assertEqual(report['pair_gold_hits'], int(pair_ids[self.active].eq(self.targets[self.active, None]).any(-1).sum()))
    self.assertGreater(report['active_rows_with_boundary_ties'], 0)

  def test_tied_swap_changes_gold_membership_but_not_retained_mass(self):
    logits = torch.tensor([[[5., 4., 4., 0.]]])
    targets = torch.tensor([[1]])
    active = torch.tensor([[True]])
    pair_ids = torch.tensor([[[0, 1]]])
    pair_values = logits.gather(-1, pair_ids)
    unary_ids = torch.tensor([[0, 2]])
    unary_values = logits[active].gather(-1, unary_ids)
    report = compare_candidate_ids(logits, targets, active, pair_values, pair_ids,
                                    unary_values, unary_ids)
    self.assertEqual(report['ordered_id_differing_slots'], 1)
    self.assertEqual(report['candidate_set_differing_rows'], 1)
    self.assertEqual(report['candidate_symmetric_difference_tokens'], 2)
    self.assertEqual(report['gold_membership_differing_rows'], 1)
    self.assertEqual(report['set_differences_not_at_cutoff_rows'], 0)
    self.assertEqual(report['retained_base_probability_mass_max_absolute_difference'], 0)
    self.assertGreater(report['pair_only_base_probability_mass_sum'], 0)
    self.assertEqual(report['pair_only_base_probability_mass_sum'], report['unary_only_base_probability_mass_sum'])
    self.assertEqual(report['differing_rows_first'][0]['pair_only_ids'], [1])

  def test_reordering_is_distinguished_from_a_set_change(self):
    logits = torch.tensor([[[4., 4., 1., 0.]]])
    pair_ids = torch.tensor([[[0, 1]]])
    unary_ids = torch.tensor([[1, 0]])
    active = torch.tensor([[True]])
    report = compare_candidate_ids(logits, torch.tensor([[0]]), active,
                                    logits.gather(-1, pair_ids), pair_ids,
                                    logits[active].gather(-1, unary_ids), unary_ids)
    self.assertEqual(report['ordered_id_differing_rows'], 1)
    self.assertEqual(report['candidate_set_differing_rows'], 0)
    self.assertEqual(report['gold_membership_differing_rows'], 0)

  def test_disk_cache_audit_aggregates_every_active_position(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      for index in range(len(self.logits)):
        torch.save({'logits': self.logits[index], 'targets': self.targets[index],
                    'active': self.active[index]}, path / f'example-{index:04d}.pt')
      summary = audit_split(path, device='cpu', batch_size=4, top_k=8,
                            position_chunk_size=32, max_detail_rows=20)
      self.assertEqual(summary['active_tokens'], int(self.active.sum()))
      self.assertEqual(summary['examples'], 4)
      self.assertEqual(len(summary['cache_files']), 4)
      self.assertEqual(summary['all_active_candidate_sets_match'], summary['candidate_set_differing_rows'] == 0)


if __name__ == '__main__':
  unittest.main()
