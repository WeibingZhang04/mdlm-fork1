"""Tiny real-data-objective smoke tests without Lightning dependencies."""

import unittest

import torch

from models.structured_decoder import ContextualCouplingForestHead
from structured_training import (
  STRUCTURED_OBJECTIVE_NAME,
  factorized_denoising_nll,
  gold_reveal_influence_topology_loss,
  sample_active_sources,
  structured_denoising_loss,
  validated_ema_shadow_parameters,
  validate_structured_objective_name,
  validate_structured_sampling_mode,
)


class StructuredTrainingTest(unittest.TestCase):

  def _batch(self):
    torch.manual_seed(101)
    batch_size, length, hidden_size, vocab_size = 3, 6, 12, 11
    head = ContextualCouplingForestHead(
      hidden_size=hidden_size, vocab_size=vocab_size,
      top_k=5, rank=3, time_embed_dim=8, topology_dim=10,
      local_window=2, num_anchor_slots=4, contextual_neighbors=2,
      component_size_cap=4)
    hidden = torch.randn(
      batch_size, length, hidden_size, requires_grad=True)
    logits = torch.randn(
      batch_size, length, vocab_size, requires_grad=True)
    clean = torch.randint(vocab_size, (batch_size, length))
    active = torch.tensor([
      [True, True, True, True, False, False],
      [True, False, True, True, True, False],
      [False, False, False, False, False, False],
    ])
    output = head(hidden, logits, torch.tensor([0.2, 0.5, 0.8]), active)
    return head, hidden, logits, clean, active, output

  def test_joint_nll_is_finite_and_backpropagates(self):
    head, hidden, logits, clean, active, output = self._batch()
    objective = structured_denoising_loss(
      output, logits, clean, active)
    self.assertTrue(bool(torch.isfinite(objective.loss)))
    self.assertEqual(int(objective.active_tokens), int(active.sum()))
    self.assertEqual(objective.distributed_nll.shape, active.shape)
    self.assertGreaterEqual(float(objective.candidate_recall), 0.0)
    self.assertLessEqual(float(objective.candidate_recall), 1.0)
    torch.testing.assert_close(
      objective.loss,
      objective.nll_sum / objective.active_tokens)
    torch.testing.assert_close(
      objective.candidate_recall,
      objective.candidate_hits / objective.active_tokens)
    torch.testing.assert_close(
      objective.retained_mass,
      objective.retained_mass_sum / objective.active_tokens)
    objective.loss.backward()
    self.assertGreater(float(hidden.grad.abs().sum()), 0.0)
    self.assertGreater(float(logits.grad.abs().sum()), 0.0)
    self.assertTrue(any(
      parameter.grad is not None for parameter in head.parameters()))

  def test_gold_reveal_teacher_trains_topology_without_head_target_input(self):
    head, hidden, logits, clean, active, output = self._batch()
    sources = sample_active_sources(
      active, generator=torch.Generator().manual_seed(3))
    self.assertEqual(int(sources[-1]), -1)
    revealed = logits.detach().clone()
    # Make influence nonuniform at all non-source active sites.
    for batch_index, source in enumerate(sources.tolist()):
      if source < 0:
        continue
      for node in range(clean.shape[1]):
        if active[batch_index, node] and node != source:
          revealed[batch_index, node, clean[batch_index, node]] += node + 1
    topology = gold_reveal_influence_topology_loss(
      output=output,
      base_unary_logits=logits.detach(),
      revealed_unary_logits=revealed,
      clean_tokens=clean,
      active_mask=active,
      source_positions=sources,
      temperature=0.5,
      minimum_choices=2)
    self.assertTrue(bool(torch.isfinite(topology.loss)))
    topology.loss.backward()
    self.assertIsNotNone(head.edge_proposer.edge_scorer[-1].weight.grad)
    self.assertIsNotNone(head.edge_proposer.anchor_projection.weight.grad)
    self.assertIsNotNone(head.edge_proposer.slot_projection.weight.grad)
    self.assertGreater(int(topology.edge_coverage_denominator), 0)
    self.assertGreater(int(topology.anchor_coverage_denominator), 0)
    self.assertGreater(int(topology.slot_coverage_denominator), 0)
    torch.testing.assert_close(
      topology.edge_coverage,
      topology.edge_coverage_numerator.float()
      / topology.edge_coverage_denominator.float())
    torch.testing.assert_close(
      topology.anchor_coverage,
      topology.anchor_coverage_numerator.float()
      / topology.anchor_coverage_denominator.float())
    torch.testing.assert_close(
      topology.slot_coverage,
      topology.slot_coverage_numerator.float()
      / topology.slot_coverage_denominator.float())
    # Candidates remain a pure function of the original unary logits.
    torch.testing.assert_close(
      output.candidate_ids, logits.topk(5, dim=-1).indices)

  def test_factorized_auxiliary_and_empty_mask_are_safe(self):
    _, _, logits, clean, active, output = self._batch()
    auxiliary = factorized_denoising_nll(logits, clean, active)
    self.assertTrue(bool(torch.isfinite(auxiliary)))
    empty = torch.zeros_like(active)
    empty_output = ContextualCouplingForestHead(
      hidden_size=12, vocab_size=11, top_k=5, rank=3,
      time_embed_dim=8, topology_dim=10, local_window=2,
      num_anchor_slots=4, contextual_neighbors=2,
      component_size_cap=4)(
        torch.randn(3, 6, 12), logits, torch.rand(3), empty)
    objective = structured_denoising_loss(
      empty_output, logits, clean, empty)
    self.assertEqual(float(objective.loss.detach()), 0.0)
    self.assertEqual(float(objective.candidate_recall), 1.0)

  def test_frozen_context_backbone_one_step_smoke(self):
    """Exercise the same base/reveal/head/update order used by Diffusion."""
    torch.manual_seed(211)
    mask_index, vocab_size, hidden_size = 10, 11, 12

    class TinyContextBackbone(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size)

      def forward(self, tokens):
        hidden = self.embedding(tokens)
        hidden = hidden + hidden.mean(dim=1, keepdim=True)
        return hidden, self.output(hidden)

    backbone = TinyContextBackbone().requires_grad_(False)
    head = ContextualCouplingForestHead(
      hidden_size=hidden_size, vocab_size=vocab_size,
      top_k=5, rank=3, time_embed_dim=8, topology_dim=10,
      local_window=2, num_anchor_slots=4, contextual_neighbors=2,
      component_size_cap=4)
    clean = torch.randint(0, mask_index, (3, 6))
    active = torch.tensor([
      [True, True, True, True, False, False],
      [True, True, False, True, True, False],
      [False, True, True, True, True, False],
    ])
    corrupted = torch.where(
      active, torch.full_like(clean, mask_index), clean)
    with torch.no_grad():
      hidden, logits = backbone(corrupted)
    logits[:, :, mask_index] = -torch.inf
    output = head(hidden, logits, torch.tensor([0.2, 0.5, 0.8]), active)
    denoising = structured_denoising_loss(
      output, logits, clean, active)

    sources = sample_active_sources(
      active, generator=torch.Generator().manual_seed(4))
    revealed = corrupted.clone()
    batch_index = torch.arange(clean.shape[0])
    revealed[batch_index, sources] = clean[batch_index, sources]
    with torch.no_grad():
      _, revealed_logits = backbone(revealed)
    revealed_logits[:, :, mask_index] = -torch.inf
    topology = gold_reveal_influence_topology_loss(
      output, logits, revealed_logits, clean, active, sources)

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    before = head.token_factor_embedding.weight.detach().clone()
    optimizer.zero_grad()
    (denoising.loss + 0.1 * topology.loss).backward()
    optimizer.step()
    self.assertTrue(all(
      parameter.grad is None for parameter in backbone.parameters()))
    self.assertFalse(torch.equal(
      before, head.token_factor_embedding.weight.detach()))

  def test_objective_and_sampling_labels_are_explicit(self):
    self.assertEqual(
      validate_structured_objective_name(STRUCTURED_OBJECTIVE_NAME),
      STRUCTURED_OBJECTIVE_NAME)
    for mode in (
        'factorized', 'structured_marginal', 'structured_joint'):
      self.assertEqual(validate_structured_sampling_mode(mode), mode)
    with self.assertRaisesRegex(ValueError, 'objective_name'):
      validate_structured_objective_name('elbo')
    with self.assertRaisesRegex(ValueError, 'sampling mode'):
      validate_structured_sampling_mode('jointish')

  def test_ema_shadows_fail_closed_on_missing_or_mismatched_state(self):
    expected = [
      torch.nn.Parameter(torch.zeros(2, 3)),
      torch.nn.Parameter(torch.zeros(4)),
    ]
    shadows = [torch.ones(2, 3), torch.ones(4), torch.ones(1)]
    validated = validated_ema_shadow_parameters(
      {'shadow_params': shadows}, expected, 'test checkpoint',
      allow_extra=True)
    self.assertEqual(len(validated), len(expected))
    with self.assertRaisesRegex(ValueError, 'no EMA state'):
      validated_ema_shadow_parameters(
        None, expected, 'test checkpoint')
    with self.assertRaisesRegex(ValueError, 'expected 2'):
      validated_ema_shadow_parameters(
        {'shadow_params': shadows[:1]}, expected, 'test checkpoint')
    with self.assertRaisesRegex(ValueError, 'shape'):
      validated_ema_shadow_parameters(
        {'shadow_params': [torch.ones(3, 2), torch.ones(4)]},
        expected, 'test checkpoint')


if __name__ == '__main__':
  unittest.main()
