import collections
import itertools
import unittest

import torch

from evaluation.causal_denoising import (
  candidate_support_sweep,
  causal_denoising_metrics,
  matched_permuted_forest_edges,
)
from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import (
  factorized_token_log_probability,
  infer_structured_distribution,
  structured_marginal_token_log_probability,
  structured_token_log_probability,
)


class CausalDenoisingTest(unittest.TestCase):

  def _problem(self, *, independent_mode=False):
    torch.manual_seed(41)
    head = ContextualCouplingForestHead(
      hidden_size=10, vocab_size=6, top_k=3, rank=3,
      time_embed_dim=8, topology_dim=9, local_window=2,
      num_anchor_slots=2, contextual_neighbors=1,
      component_size_cap=5, topology_mode='fixed', factor_mode='dynamic',
      independent_mode=independent_mode)
    hidden = torch.randn(2, 5, 10)
    logits = torch.randn(2, 5, 6)
    active = torch.tensor([
      [True, True, True, True, True],
      [True, False, True, True, True],
    ])
    timestep = torch.tensor([0.3, 0.7])
    clean = torch.tensor([
      [0, 1, 2, 3, 4],
      [5, 0, 1, 2, 3],
    ])
    output = head(hidden, logits, timestep, active)
    return head, hidden, logits, timestep, clean, active, output

  def test_product_of_singletons_is_normalized(self):
    head, hidden, logits, timestep, _, active, output = self._problem()
    del head, hidden, timestep
    # Use one short example so exhaustive enumeration remains tiny.
    active = active[:1, :3]
    logits = logits[:1, :3]
    short_head = ContextualCouplingForestHead(
      hidden_size=10, vocab_size=6, top_k=3, rank=3,
      time_embed_dim=8, topology_dim=9, local_window=2,
      num_anchor_slots=2, contextual_neighbors=1,
      component_size_cap=3, topology_mode='fixed')
    short_output = short_head(
      torch.randn(1, 3, 10), logits, torch.tensor([0.3]), active)
    inference = infer_structured_distribution(short_output, active)
    probabilities = []
    for tokens in itertools.product(range(6), repeat=3):
      probabilities.append(structured_marginal_token_log_probability(
        short_output, logits, torch.tensor([tokens]), active,
        inference).exp())
    torch.testing.assert_close(
      torch.cat(probabilities).sum(), torch.tensor(1.0),
      atol=3e-6, rtol=3e-6)

  def test_neutral_pair_head_equals_original_factorized_backbone(self):
    _, _, logits, _, clean, active, output = self._problem(
      independent_mode=True)
    joint = structured_token_log_probability(
      output, logits, clean, active)
    factorized = factorized_token_log_probability(logits, clean, active)
    torch.testing.assert_close(joint, factorized, atol=2e-5, rtol=2e-5)

  def test_candidate_support_sweep_matches_direct_softmax(self):
    _, _, logits, _, clean, active, _ = self._problem()
    sweep = candidate_support_sweep(logits, clean, active, [1, 3, 6])
    self.assertEqual(sweep.candidate_ks, (1, 3, 6))
    probabilities = logits.softmax(dim=-1)
    for column, candidate_k in enumerate(sweep.candidate_ks):
      ids = logits.topk(candidate_k, dim=-1).indices
      hits = (ids.eq(clean[:, :, None]).any(-1) & active).sum(-1)
      mass = torch.gather(probabilities, -1, ids).sum(-1)
      mass = mass.masked_fill(~active, 0.0).sum(-1)
      torch.testing.assert_close(sweep.candidate_hits[:, column], hits)
      torch.testing.assert_close(
        sweep.retained_mass_sum[:, column], mass, atol=1e-6, rtol=1e-6)

  def test_matched_permutation_preserves_degree_and_edge_count(self):
    edge_index = torch.tensor([[[0, 1], [1, 2], [1, 3], [0, 0]]])
    edge_mask = torch.tensor([[True, True, True, False]])
    active = torch.tensor([[True, True, True, True]])
    permuted, mask, changed = matched_permuted_forest_edges(
      edge_index, edge_mask, active, seed=17)
    torch.testing.assert_close(mask, edge_mask)
    self.assertGreater(int(changed.item()), 0)

    def degrees(edges):
      result = collections.Counter()
      for left, right in edges:
        result[int(left)] += 1
        result[int(right)] += 1
      return sorted(result.values())

    self.assertEqual(
      degrees(edge_index[0, edge_mask[0]].tolist()),
      degrees(permuted[0, mask[0]].tolist()))

  def test_causal_metrics_share_one_primary_distribution(self):
    head, hidden, logits, timestep, clean, active, output = self._problem()
    joint_nll = -structured_token_log_probability(
      output, logits, clean, active)
    metrics = causal_denoising_metrics(
      head=head,
      primary_output=output,
      hidden_states=hidden,
      unary_logits=logits,
      timestep=timestep,
      clean_tokens=clean,
      active_mask=active,
      candidate_ks=[1, 3, 6],
      topology_permutation_seed=29,
      structured_joint_nll_sum=joint_nll)
    torch.testing.assert_close(metrics.structured_joint_nll_sum, joint_nll)
    torch.testing.assert_close(
      metrics.parameter_matched_no_edge_nll_sum,
      metrics.factorized_backbone_nll_sum)
    self.assertEqual(metrics.candidate_support.candidate_ks, (1, 3, 6))
    self.assertEqual(metrics.matched_permuted_topology_nll_sum.shape, (2,))
    self.assertEqual(
      metrics.selected_degree_sequences,
      metrics.permuted_degree_sequences)
    self.assertEqual(
      metrics.selected_component_sizes,
      metrics.permuted_component_sizes)


if __name__ == '__main__':
  unittest.main()
