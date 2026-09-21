import dataclasses
import itertools
import unittest
from unittest import mock

import torch

from evaluation.dependence_diagnostics import (
  decompose_structured_log_probability,
  sample_shrunk_structured_tokens,
)
from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import (
  full_vocabulary_marginals,
  infer_structured_distribution,
  structured_marginal_token_log_probability,
  structured_token_log_probability,
)


class DependenceDiagnosticsTest(unittest.TestCase):

  def _problem(self, *, top_k=2, active=None):
    torch.manual_seed(120)
    if active is None:
      active = torch.ones(1, 3, dtype=torch.bool)
    batch, length = active.shape
    head = ContextualCouplingForestHead(
      hidden_size=8, vocab_size=4, top_k=top_k, rank=2,
      time_embed_dim=4, topology_dim=6, local_window=2,
      num_anchor_slots=2, contextual_neighbors=1,
      component_size_cap=4, topology_mode='fixed')
    logits = torch.randn(batch, length, 4)
    with torch.no_grad():
      output = head(
        torch.randn(batch, length, 8), logits, torch.full((batch,), 0.4), active)
    # Strong nonuniform factors ensure that the edge test cannot pass by
    # confusing a neutral pair potential with a calibrated edge belief.
    output = dataclasses.replace(
      output,
      unary_log_potentials=output.unary_log_potentials.double(),
      pair_left_factors=(1.3 * torch.randn_like(output.pair_left_factors)).double().exp(),
      pair_right_factors=(1.3 * torch.randn_like(output.pair_right_factors)).double().exp())
    return output, logits, active

  def test_decomposition_matches_exact_joint_and_product_scores(self):
    output, logits, active = self._problem()
    for backend in ['dense', 'low_rank']:
      inference = infer_structured_distribution(output, active, backend)
      for assignment in itertools.product(range(4), repeat=3):
        tokens = torch.tensor([assignment])
        result = decompose_structured_log_probability(
          output, logits, tokens, active, inference)
        joint = structured_token_log_probability(
          output, logits, tokens, active, inference)
        marginal = structured_marginal_token_log_probability(
          output, logits, tokens, active, inference)
        torch.testing.assert_close(result.joint_log_probability, joint)
        torch.testing.assert_close(result.marginal_log_probability, marginal)
        torch.testing.assert_close(result.joint_gain, result.marginal_gain + result.dependence_log_probability)
        torch.testing.assert_close(result.shrinkage_log_probability(0.0), marginal)
        torch.testing.assert_close(result.shrinkage_log_probability(1.0), joint)

  def test_shrinkage_is_normalized_and_preserves_every_full_token_marginal(self):
    output, logits, active = self._problem()
    inference = infer_structured_distribution(output, active, 'low_rank')
    expected_marginals = full_vocabulary_marginals(output, logits, active, inference)[0]
    assignments = torch.tensor(list(itertools.product(range(4), repeat=3)))
    results = [decompose_structured_log_probability(
      output, logits, row.unsqueeze(0), active, inference) for row in assignments]
    # Edge-specific strengths are valid as well as the common-lambda path.
    strengths = [0.0, 0.2, 0.5, 0.9, 1.0, torch.tensor([0.1, 0.8])]
    for strength in strengths:
      probabilities = torch.cat([
        result.shrinkage_log_probability(strength) for result in results]).exp()
      torch.testing.assert_close(probabilities.sum(), probabilities.new_tensor(1.0), atol=2e-7, rtol=2e-7)
      for node in range(3):
        actual = torch.stack([
          probabilities[assignments[:, node] == token].sum() for token in range(4)])
        torch.testing.assert_close(actual, expected_marginals[node], atol=2e-7, rtol=2e-7)

  def test_selected_edge_ratios_match_dense_beliefs_with_reversed_edges(self):
    output, logits, active = self._problem(top_k=4)
    output = dataclasses.replace(
      output, edge_index=output.edge_index.flip(-1),
      pair_left_factors=output.pair_right_factors,
      pair_right_factors=output.pair_left_factors)
    dense = infer_structured_distribution(output, active, 'dense')
    sparse = infer_structured_distribution(output, active, 'low_rank')
    for tokens in [torch.tensor([[0, 1, 2]]), torch.tensor([[3, 2, 0]])]:
      dense_result = decompose_structured_log_probability(output, logits, tokens, active, dense)
      with mock.patch.object(output, 'materialize_pair_factors', side_effect=AssertionError('dense allocation')):
        sparse_result = decompose_structured_log_probability(output, logits, tokens, active, sparse)
      torch.testing.assert_close(sparse_result.edge_log_dependence, dense_result.edge_log_dependence)

  def test_residual_edges_can_have_dependence_despite_unit_pair_potential(self):
    output, logits, active = self._problem()
    inference = infer_structured_distribution(output, active, 'low_rank')
    omitted = []
    for node in range(3):
      omitted.append(next(token for token in range(4) if token not in output.candidate_ids[0, node]))
    result = decompose_structured_log_probability(output, logits, torch.tensor([omitted]), active, inference)
    self.assertTrue(bool((result.edge_log_dependence.abs() > 1e-4).any()))
    torch.testing.assert_close(
      result.joint_log_probability,
      structured_token_log_probability(output, logits, torch.tensor([omitted]), active, inference))

  def test_inactive_nodes_padded_edges_and_empty_active_set(self):
    active = torch.tensor([[True, False, True, True], [False, False, False, False]])
    output, logits, active = self._problem(top_k=4, active=active)
    inference = infer_structured_distribution(output, active, 'low_rank')
    first = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    second = torch.tensor([[0, 3, 2, 3], [3, 2, 1, 0]])
    for tokens in [first, second]:
      result = decompose_structured_log_probability(output, logits, tokens, active, inference)
      self.assertEqual(result.joint_log_probability[1].item(), 0.0)
      self.assertTrue(bool((result.node_log_probability[~active] == 0).all()))
      self.assertTrue(bool((result.edge_log_dependence[~output.edge_mask] == 0).all()))
      torch.testing.assert_close(result.active_token_count, torch.tensor([3, 0]))
      torch.testing.assert_close(result.joint_log_probability, structured_token_log_probability(output, logits, tokens, active, inference))

  def test_edgeless_distribution_and_strength_validation(self):
    output, logits, active = self._problem(active=torch.ones(1, 1, dtype=torch.bool))
    inference = infer_structured_distribution(output, active, 'low_rank')
    result = decompose_structured_log_probability(output, logits, torch.tensor([[0]]), active, inference)
    self.assertEqual(tuple(result.edge_log_dependence.shape), (1, 0))
    torch.testing.assert_close(result.shrinkage_log_probability(0.3), result.joint_log_probability)
    for invalid in [-0.1, 1.1, float('nan')]:
      with self.assertRaises(ValueError):
        result.shrinkage_log_probability(invalid)

  def test_rejects_inference_for_different_active_mask(self):
    output, logits, active = self._problem()
    inference = infer_structured_distribution(output, active, 'low_rank')
    changed = active.clone()
    changed[0, 0] = False
    with self.assertRaisesRegex(ValueError, 'different active_mask'):
      decompose_structured_log_probability(output, logits, torch.tensor([[0, 1, 2]]), changed, inference)

  def test_extreme_endpoint_factors_remain_finite(self):
    output, logits, active = self._problem(top_k=4)
    output.pair_left_factors.mul_(1e100)
    output.pair_right_factors.mul_(1e100)
    inference = infer_structured_distribution(output, active, 'low_rank')
    result = decompose_structured_log_probability(output, logits, torch.tensor([[0, 1, 2]]), active, inference)
    self.assertTrue(bool(torch.isfinite(result.edge_log_dependence).all()))
    torch.testing.assert_close(
      result.joint_log_probability,
      structured_token_log_probability(output, logits, torch.tensor([[0, 1, 2]]), active, inference),
      atol=1e-10, rtol=1e-10)

  def test_shrinkage_sampler_matches_enumerated_joint_and_all_marginals(self):
    output, logits, active = self._problem()
    inference = infer_structured_distribution(output, active, 'low_rank')
    marginals = full_vocabulary_marginals(output, logits, active, inference)[0]
    assignments = torch.tensor(list(itertools.product(range(4), repeat=3)))
    results = [decompose_structured_log_probability(
      output, logits, row.unsqueeze(0), active, inference) for row in assignments]
    for index, strength in enumerate([0.0, 0.4, 1.0, torch.tensor([0.0, 1.0])]):
      with mock.patch.object(output, 'materialize_pair_factors', side_effect=AssertionError('dense allocation')):
        samples = sample_shrunk_structured_tokens(
          output, logits, active, strength, num_samples=30000,
          generator=torch.Generator().manual_seed(620 + index))[0]
      actual_marginals = torch.nn.functional.one_hot(samples, num_classes=4).double().mean(0)
      torch.testing.assert_close(actual_marginals, marginals, atol=0.012, rtol=0)
      sample_ids = (samples * torch.tensor([16, 4, 1])).sum(-1)
      observed_joint = torch.bincount(sample_ids, minlength=64).double() / samples.shape[0]
      expected_joint = torch.cat([
        result.shrinkage_log_probability(strength) for result in results]).exp()
      torch.testing.assert_close(observed_joint, expected_joint, atol=0.008, rtol=0)

  def test_shrinkage_sampler_inactive_nodes_and_disabled_tail(self):
    active = torch.tensor([[True, False, True], [False, False, False]])
    output, logits, active = self._problem(top_k=4, active=active)
    samples = sample_shrunk_structured_tokens(
      output, logits, active, 0.5, num_samples=100,
      generator=torch.Generator().manual_seed(999))
    expected = output.candidate_ids[..., 0][:, None].expand(-1, 100, -1)
    inactive = (~active)[:, None].expand_as(samples)
    torch.testing.assert_close(samples[inactive], expected[inactive])
    self.assertEqual(tuple(samples.shape), (2, 100, 3))
    self.assertTrue(bool(((samples >= 0) & (samples < 4)).all()))


if __name__ == '__main__':
  unittest.main()
