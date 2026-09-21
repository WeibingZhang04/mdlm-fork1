import itertools
import unittest

import torch

from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import (
  full_vocabulary_marginals,
  infer_structured_distribution,
  sample_structured_marginal_tokens,
  sample_structured_tokens,
  structured_token_log_probability,
)


class StructuredObjectiveTest(unittest.TestCase):

  def _problem(self, top_k=2):
    torch.manual_seed(19)
    head = ContextualCouplingForestHead(
      hidden_size=12,
      vocab_size=4,
      top_k=top_k,
      rank=3,
      time_embed_dim=8,
      topology_dim=10,
      local_window=2,
      num_anchor_slots=2,
      contextual_neighbors=1,
      component_size_cap=3,
      topology_mode='fixed')
    hidden = torch.randn(1, 3, 12)
    logits = torch.randn(1, 3, 4)
    active = torch.ones(1, 3, dtype=torch.bool)
    output = head(hidden, logits, torch.tensor([0.4]), active)
    return output, logits, active

  def test_full_support_token_probabilities_sum_to_one(self):
    output, logits, active = self._problem(top_k=2)
    inference = infer_structured_distribution(output, active)
    log_probabilities = []
    for sequence in itertools.product(range(4), repeat=3):
      tokens = torch.tensor([sequence])
      log_probabilities.append(structured_token_log_probability(
        output, logits, tokens, active, inference))
    probabilities = torch.cat(log_probabilities).exp()
    torch.testing.assert_close(
      probabilities.sum(), torch.tensor(1.0), atol=2e-6, rtol=2e-6)
    self.assertTrue(bool((probabilities > 0).all().item()))

  def test_expanded_marginals_match_joint_sample_frequencies(self):
    output, logits, active = self._problem(top_k=2)
    inference = infer_structured_distribution(output, active)
    marginals = full_vocabulary_marginals(
      output, logits, active, inference)
    samples = sample_structured_tokens(
      output, logits, active, num_samples=30000,
      generator=torch.Generator().manual_seed(71),
      inference=inference)[0]
    frequencies = torch.nn.functional.one_hot(
      samples, num_classes=4).float().mean(dim=0)
    torch.testing.assert_close(
      frequencies, marginals[0].float(), atol=0.015, rtol=0.0)

  def test_independent_marginal_sampler_uses_exact_node_marginals(self):
    output, logits, active = self._problem(top_k=2)
    inference = infer_structured_distribution(output, active)
    marginals = full_vocabulary_marginals(
      output, logits, active, inference)
    samples = sample_structured_marginal_tokens(
      output, logits, active, num_samples=30000,
      generator=torch.Generator().manual_seed(72),
      inference=inference)[0]
    frequencies = torch.nn.functional.one_hot(
      samples, num_classes=4).float().mean(dim=0)
    torch.testing.assert_close(
      frequencies, marginals[0].float(), atol=0.015, rtol=0.0)

  def test_inactive_nodes_cancel_from_likelihood(self):
    output, logits, _ = self._problem(top_k=4)
    active = torch.tensor([[True, False, True]])
    output = ContextualCouplingForestHead(
      hidden_size=12, vocab_size=4, top_k=4, rank=3,
      time_embed_dim=8, topology_dim=10, local_window=2,
      num_anchor_slots=2, contextual_neighbors=1,
      component_size_cap=3, topology_mode='fixed')(
        torch.randn(1, 3, 12), logits, torch.tensor([0.4]), active)
    first = torch.tensor([[0, 1, 2]])
    second = torch.tensor([[0, 3, 2]])
    first_log_prob = structured_token_log_probability(
      output, logits, first, active)
    second_log_prob = structured_token_log_probability(
      output, logits, second, active)
    torch.testing.assert_close(first_log_prob, second_log_prob)

  def test_extreme_endpoint_factors_have_finite_log_probability(self):
    output, logits, active = self._problem(top_k=4)
    with torch.no_grad():
      output.pair_left_factors.fill_(1e30)
      output.pair_right_factors.fill_(1e30)
    inference = infer_structured_distribution(
      output, active, backend='low_rank')
    log_probability = structured_token_log_probability(
      output, logits, torch.tensor([[0, 1, 2]]), active, inference)
    self.assertTrue(bool(torch.isfinite(inference.marginals.log_partition).all()))
    self.assertTrue(bool(torch.isfinite(log_probability).all()))


if __name__ == '__main__':
  unittest.main()
