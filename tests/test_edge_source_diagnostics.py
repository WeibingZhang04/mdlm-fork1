# Inference diagnostics for selected edge proposal sources.

import unittest

import torch

from evaluation.fixed_group_sampling import generate_fixed_groups
from models.structured_decoder import (
  EDGE_SOURCE_CHAIN,
  EDGE_SOURCE_CONTEXTUAL,
  EDGE_SOURCE_FIXED,
  EDGE_SOURCE_LOCAL,
  ContextualCouplingForestHead,
  selected_edge_source_counts,
)


class EdgeSourceDiagnosticsTest(unittest.TestCase):

  @staticmethod
  def _head():
    torch.manual_seed(19)
    return ContextualCouplingForestHead(
      hidden_size=6, vocab_size=3, top_k=3, rank=2,
      time_embed_dim=4, topology_dim=5, local_window=1,
      num_anchor_slots=1, contextual_neighbors=0,
      component_size_cap=0)

  def test_dynamic_counts_distinguish_local_and_chain(self):
    head = self._head()
    active = torch.tensor([[True, False, False, True, True, False]])
    output = head(
      torch.randn(1, 6, 6), torch.randn(1, 6, 3),
      torch.tensor([0.5]), active,
      collect_edge_source_diagnostics=True)
    counts = selected_edge_source_counts(output)
    self.assertEqual(counts.tolist(), [[1, 1, 0, 0]])
    source_by_edge = {
      tuple(edge): int(source)
      for edge, source in zip(
        output.edge_index[0, output.edge_mask[0]].tolist(),
        output.edge_source[0, output.edge_mask[0]].tolist())
    }
    self.assertEqual(source_by_edge[(0, 3)], EDGE_SOURCE_CHAIN)
    self.assertEqual(source_by_edge[(3, 4)], EDGE_SOURCE_LOCAL)

  def test_fixed_topology_is_reported_separately(self):
    head = self._head()
    active = torch.tensor([[True, False, True, False, True, False]])
    output = head(
      torch.randn(1, 6, 6), torch.randn(1, 6, 3),
      torch.tensor([0.5]), active, topology_mode='fixed',
      collect_edge_source_diagnostics=True)
    self.assertEqual(selected_edge_source_counts(output).tolist(), [[0, 0, 0, 2]])
    self.assertTrue(bool(
      output.edge_source[output.edge_mask].eq(EDGE_SOURCE_FIXED).all()))

  def test_generation_accumulates_per_step_source_counts(self):
    head = self._head().eval()

    def model(
        tokens, sigma, active, *,
        collect_edge_source_diagnostics=False):
      hidden = torch.zeros(*tokens.shape, 6)
      logits = torch.zeros(*tokens.shape, 3)
      return head(
        hidden, logits, sigma, active,
        collect_edge_source_diagnostics=(
          collect_edge_source_diagnostics)), logits

    result = generate_fixed_groups(
      model, torch.tensor([[3, 3, 3]]), mask_index=3,
      group_size=1, mode='marginal',
      reveal_order=torch.tensor([[1, 0, 2]]),
      sampling_generator=torch.Generator().manual_seed(23),
      collect_edge_source_diagnostics=True)
    self.assertEqual(result.edge_source_counts.tolist(), [2, 1, 0, 0])
    self.assertEqual(
      torch.stack([step.edge_source_counts.sum(0)
                   for step in result.steps]).sum(0).tolist(),
      result.edge_source_counts.tolist())
    for step in result.steps:
      self.assertEqual(
        int(step.edge_source_counts.sum()), int(step.edge_mask.sum()))

  def test_diagnostics_are_opt_in(self):
    head = self._head().eval()
    active = torch.tensor([[True, False, True]])
    hidden = torch.zeros(1, 3, 6)
    logits = torch.zeros(1, 3, 3)
    output = head(hidden, logits, torch.tensor([0.5]), active)
    self.assertIsNone(output.proposal_edge_source)
    self.assertIsNone(output.edge_source)
    with self.assertRaisesRegex(ValueError, 'were not enabled'):
      selected_edge_source_counts(output)

    def model(tokens, sigma, active_mask):
      return head(hidden, logits, sigma, active_mask), logits

    result = generate_fixed_groups(
      model, torch.tensor([[3, 0, 3]]), mask_index=3,
      group_size=1, mode='marginal',
      reveal_order=torch.tensor([[0, 1, 2]]))
    self.assertIsNone(result.edge_source_counts)
    self.assertTrue(all(step.edge_source is None for step in result.steps))

  def test_enabled_generation_rejects_uninstrumented_output(self):
    head = self._head().eval()

    def model(
        tokens, sigma, active, *,
        collect_edge_source_diagnostics=False):
      hidden = torch.zeros(*tokens.shape, 6)
      logits = torch.zeros(*tokens.shape, 3)
      return head(hidden, logits, sigma, active), logits

    with self.assertRaisesRegex(ValueError, 'were not enabled'):
      generate_fixed_groups(
        model, torch.tensor([[3, 3, 3]]), mask_index=3,
        group_size=1, mode='marginal', reveal_seeds=[7],
        collect_edge_source_diagnostics=True)


if __name__ == '__main__':
  unittest.main()
