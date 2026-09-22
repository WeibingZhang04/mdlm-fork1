"""Focused tests for sparse dynamic edge proposals."""

import unittest

import torch

from models.structured_decoder import SparseEdgeProposer, _fixed_chain_edges


class SparseEdgeProposerTest(unittest.TestCase):

  def test_dynamic_proposals_include_fixed_chain_across_inactive_gaps(self):
    torch.manual_seed(7)
    active = torch.tensor([
      [True, False, False, True, False, True],
      [True, False, True, False, False, True],
      [False, True, False, False, False, True],
      [True, False, False, False, False, False],
      [False, False, False, False, False, False],
    ])
    proposer = SparseEdgeProposer(
      topology_dim=4,
      local_window=2,
      num_anchor_slots=1,
      contextual_neighbors=0)
    edge_index, edge_mask, *_ = proposer(
      torch.randn(5, 6, 4), active)
    fixed_index, fixed_mask = _fixed_chain_edges(
      active, component_size_cap=0)

    for batch_index in range(active.shape[0]):
      proposed = {
        tuple(edge)
        for edge in edge_index[batch_index, edge_mask[batch_index]].tolist()
      }
      fixed_chain = {
        tuple(edge)
        for edge in fixed_index[batch_index, fixed_mask[batch_index]].tolist()
      }
      self.assertTrue(fixed_chain.issubset(proposed))

    proposed_first = {
      tuple(edge)
      for edge in edge_index[0, edge_mask[0]].tolist()
    }
    self.assertIn((0, 3), proposed_first)
    self.assertIn((3, 5), proposed_first)


if __name__ == "__main__":
  unittest.main()
