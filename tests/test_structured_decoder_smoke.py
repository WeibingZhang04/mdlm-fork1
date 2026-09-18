"""Smoke tests for the contextual coupling-forest decoder."""

import unittest

import torch

from models.structured_decoder import ContextualCouplingForestHead


class StructuredDecoderSmokeTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(7)
    self.batch_size = 2
    self.sequence_length = 7
    self.hidden_size = 24
    self.vocab_size = 17
    self.top_k = 5
    self.rank = 4
    self.head = ContextualCouplingForestHead(
      hidden_size=self.hidden_size,
      vocab_size=self.vocab_size,
      top_k=self.top_k,
      rank=self.rank,
      time_embed_dim=12,
      topology_dim=16,
      local_window=2,
      num_anchor_slots=4,
      contextual_neighbors=2,
      component_size_cap=3)
    self.hidden = torch.randn(
      self.batch_size, self.sequence_length, self.hidden_size,
      requires_grad=True)
    self.logits = torch.randn(
      self.batch_size, self.sequence_length, self.vocab_size,
      requires_grad=True)
    self.active = torch.tensor([
      [True, True, True, True, True, False, False],
      [True, False, True, True, False, True, True],
    ])

  def decode(self, **overrides):
    return self.head(
      self.hidden, self.logits, torch.tensor([0.2, 0.8]),
      self.active, **overrides)

  def test_shapes_candidates_factors_and_gradients(self):
    output = self.decode()
    self.assertEqual(
      output.candidate_ids.shape,
      (self.batch_size, self.sequence_length, self.top_k))
    self.assertEqual(
      output.unary_log_potentials.shape,
      (self.batch_size, self.sequence_length, self.top_k + 1))
    self.assertEqual(
      output.edge_index.shape,
      (self.batch_size, self.sequence_length - 1, 2))
    self.assertEqual(
      output.pair_left_factors.shape,
      (self.batch_size, self.sequence_length - 1, self.top_k, self.rank))

    dense = output.materialize_pair_factors()
    self.assertEqual(
      dense.shape,
      (self.batch_size, self.sequence_length - 1,
       self.top_k + 1, self.top_k + 1))
    self.assertTrue(bool((dense > 0).all()))
    torch.testing.assert_close(
      dense[..., -1, :], torch.ones_like(dense[..., -1, :]))
    torch.testing.assert_close(
      dense[..., :, -1], torch.ones_like(dense[..., :, -1]))

    torch.testing.assert_close(
      output.candidate_ids,
      self.logits.topk(self.top_k, dim=-1).indices)
    tail = self.logits.detach().clone()
    tail.scatter_(-1, output.candidate_ids, -torch.inf)
    torch.testing.assert_close(
      output.residual_log_mass,
      torch.logsumexp(tail.float(), dim=-1),
      atol=2e-5,
      rtol=2e-5)

    finite_scores = output.edge_scores.masked_select(output.edge_mask)
    loss = (output.unary_log_potentials.sum()
            + output.pair_left_factors.sum()
            + output.pair_right_factors.sum()
            + finite_scores.sum())
    loss.backward()
    self.assertIsNotNone(self.hidden.grad)
    self.assertIsNotNone(self.logits.grad)
    self.assertGreater(float(self.hidden.grad.abs().sum()), 0.0)
    self.assertGreater(float(self.logits.grad.abs().sum()), 0.0)

  def test_independent_mode_neutralizes_pair_factors(self):
    output = self.decode(
      topology_mode='fixed', factor_mode='fixed', independent_mode=True)
    dense = output.materialize_pair_factors()
    torch.testing.assert_close(dense, torch.ones_like(dense), atol=1e-6, rtol=0)

  def test_fixed_topology_accepts_shared_and_batched_edges(self):
    shared_edges = torch.tensor([[2, 0], [3, 2]])
    shared = self.decode(fixed_edge_index=shared_edges)
    self.assertEqual(shared.topology_mode, 'fixed')
    self.assertEqual(shared.edge_mask.sum(dim=1).tolist(), [2, 2])
    expected = torch.tensor([[0, 2], [2, 3]])
    torch.testing.assert_close(shared.edge_index[0, :2], expected)
    torch.testing.assert_close(shared.edge_index[1, :2], expected)
    self.assertFalse(shared.edge_index.requires_grad)

    batched_edges = torch.tensor([
      [[1, 0], [2, 1], [-1, -1]],
      [[2, 0], [3, 2], [6, 5]],
    ])
    batched_mask = torch.tensor([
      [True, True, False],
      [True, True, True],
    ])
    batched = self.decode(
      topology_mode='fixed',
      fixed_edge_index=batched_edges,
      fixed_edge_mask=batched_mask)
    self.assertEqual(batched.edge_mask.sum(dim=1).tolist(), [2, 3])

  def test_invalid_fixed_topologies_are_rejected(self):
    cases = (
      (torch.tensor([[0, 0]]), 'self edge', None, 'fixed'),
      (torch.tensor([[0, 2], [2, 0]]), 'duplicate edge', None, 'fixed'),
      (torch.tensor([[0, 1], [1, 2], [2, 0]]), 'cycle', None, 'fixed'),
      (torch.tensor([[0, 7]]), 'out of range', None, 'fixed'),
      (torch.tensor([[0, 5]]), 'outside active_mask', None, 'fixed'),
      (torch.tensor([[0, 1], [1, 2], [2, 3]]),
       'exceeding component_size_cap', None, 'fixed'),
      (torch.tensor([[2, 0], [3, 2]]), 'incompatible', None, 'dynamic'),
    )
    for edges, message, mask, mode in cases:
      with self.subTest(message=message), self.assertRaisesRegex(
          (TypeError, ValueError), message):
        self.decode(
          topology_mode=mode,
          fixed_edge_index=edges,
          fixed_edge_mask=mask)

    with self.assertRaisesRegex(ValueError, 'requires fixed_edge_index'):
      self.decode(fixed_edge_mask=torch.tensor([True]))


if __name__ == '__main__':
  unittest.main()
