"""Standalone frozen-unary head: exact distributions, gradients, and AB/BA."""

import itertools
import math
import unittest

import torch

from models.centered_forest import FrozenUnaryCenteredForestHead, centered_forest_log_probability
from models.contextual_unary import ContextualUnaryAdapter
from models.structured_decoder import StructuredDecoderOutput
from scripts.evaluate_staged_fresh_selected import score_head_float64
from structured_objective import (
  full_vocabulary_marginals, infer_structured_distribution,
  sample_structured_marginal_tokens, sample_structured_tokens, structured_token_log_probability,
)


class CenteredForestTest(unittest.TestCase):

  def setUp(self):
    torch.set_num_threads(1)
    torch.manual_seed(107)

  def problem(self, *, batch=1, length=3, vocab=4, top_k=2, dtype=torch.float32):
    unary = ContextualUnaryAdapter(5, vocab, rank=3, time_embed_dim=4,
                                    top_k=top_k, correction_domain='candidates').to(dtype=dtype)
    with torch.no_grad():
      unary.hidden_projection.weight.normal_(std=0.3)
      unary.hidden_projection.bias.normal_(std=0.2)
      unary.time_projection.weight.normal_(std=0.2)
      unary.token_embedding.weight.normal_(std=0.5)
    head = FrozenUnaryCenteredForestHead(unary, feature_dim=3, eta=0.95)
    hidden = torch.randn(batch, length, 5, dtype=dtype)
    logits = torch.randn(batch, length, vocab, dtype=dtype)
    sigma = torch.linspace(0.2, 0.7, batch, dtype=dtype)
    active = torch.ones(batch, length, dtype=torch.bool)
    return unary, head, hidden, logits, sigma, active

  @staticmethod
  def make_dependent(head):
    with torch.no_grad():
      head.right_token_embedding.weight.normal_(std=0.8)
      for projection in (head.left_hidden_projection, head.right_hidden_projection,
                         head.left_time_projection, head.right_time_projection):
        projection.weight.normal_(std=0.2)

  def test_initial_independence_matches_actual_selected_test_scorer_including_tail(self):
    unary, head, hidden, logits, sigma, active = self.problem(batch=2)
    active[1, 1] = False
    output = head(hidden, logits, sigma, active)
    self.assertIsInstance(output, StructuredDecoderOutput)
    self.assertEqual(output.pair_left_factors.shape, (2, 2, 2, 7))
    self.assertEqual(output.unary_log_potentials.dtype, torch.float64)
    torch.testing.assert_close(output.materialize_pair_factors(),
                               torch.ones(2, 2, 3, 3, dtype=torch.float64), atol=2e-15, rtol=0)
    ids = logits.topk(2, -1).indices
    targets = logits.argmin(-1)  # All active targets are omitted candidates.
    targets[0, 0] = ids[0, 0, 0]
    batch = dict(hidden=hidden, logits=logits, sigma=sigma, active=active, targets=targets)
    reference = score_head_float64(unary, batch)
    actual = centered_forest_log_probability(output, logits, targets, active)
    torch.testing.assert_close(actual, reference['joint_log_probability'], atol=3e-14, rtol=0)
    self.assertTrue(torch.equal(output.candidate_ids, reference['candidate_ids']))
    with torch.no_grad():
      expected = unary.candidate_lattice(hidden, logits, sigma, active).unary_log_potentials.double().log_softmax(-1)
    torch.testing.assert_close(output.unary_log_potentials, expected, atol=0, rtol=0)

  def test_exact_frozen_unary_marginals_after_nontrivial_coupling(self):
    unary, head, hidden, logits, sigma, active = self.problem(batch=2, length=4)
    active[1] = torch.tensor([True, False, True, False])
    self.make_dependent(head)
    output = head(hidden, logits, sigma, active)
    for backend in ('dense', 'low_rank'):
      inference = infer_structured_distribution(output, active, backend)
      expected = output.unary_log_potentials.exp()
      torch.testing.assert_close(inference.marginals.node_marginals[active], expected[active], atol=2e-14, rtol=0)
      expanded = full_vocabulary_marginals(output, logits, active, inference)
      reference = expected[..., -1:] * output.residual_log_probs(logits).exp()
      reference.scatter_add_(-1, output.candidate_ids, expected[..., :-1])
      torch.testing.assert_close(expanded[active], reference[active], atol=2e-14, rtol=0)
      torch.testing.assert_close(expanded[active].sum(-1), torch.ones_like(expanded[active].sum(-1)), atol=2e-14, rtol=0)
    self.assertFalse(torch.allclose(output.materialize_pair_factors()[0, 0], torch.ones(3, 3, dtype=torch.float64)))

  def test_full_vocabulary_enumeration_normalizes_and_matches_existing_scores(self):
    _, head, hidden, logits, sigma, active = self.problem(dtype=torch.float64)
    self.make_dependent(head)
    output = head(hidden, logits, sigma, active)
    inference = infer_structured_distribution(output, active, 'low_rank')
    torch.testing.assert_close(inference.marginals.log_partition, torch.zeros(1, dtype=torch.float64), atol=2e-14, rtol=0)
    assignments = list(itertools.product(range(4), repeat=3))
    probabilities = []
    for assignment in assignments:
      targets = torch.tensor([assignment])
      direct = centered_forest_log_probability(output, logits, targets)
      existing = structured_token_log_probability(output, logits, targets, active, inference)
      torch.testing.assert_close(direct, existing, atol=2e-14, rtol=0)
      probabilities.append(direct.exp()[0])
    probabilities = torch.stack(probabilities)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1., dtype=torch.float64), atol=2e-14, rtol=0)
    marginals = torch.zeros(3, 4, dtype=torch.float64)
    for assignment, probability in zip(assignments, probabilities):
      for node, token in enumerate(assignment):
        marginals[node, token] += probability
    torch.testing.assert_close(marginals, full_vocabulary_marginals(output, logits, active, inference)[0], atol=2e-14, rtol=0)

  def test_direct_and_partition_inference_gradients_agree(self):
    _, head, hidden, logits, sigma, active = self.problem(dtype=torch.float64)
    self.make_dependent(head)
    output = head(hidden, logits, sigma, active)
    targets = output.candidate_ids[..., 0]
    direct = centered_forest_log_probability(output, logits, targets).sum()
    inference = infer_structured_distribution(output, active, 'low_rank')
    existing = structured_token_log_probability(output, logits, targets, active, inference).sum()
    parameters = [p for p in head.parameters() if p.requires_grad]
    direct_grad = torch.autograd.grad(direct, parameters, retain_graph=True)
    inferred_grad = torch.autograd.grad(existing, parameters)
    for left, right in zip(direct_grad, inferred_grad):
      self.assertTrue(torch.isfinite(left).all())
      self.assertTrue(torch.isfinite(right).all())
      torch.testing.assert_close(left, right, atol=3e-14, rtol=1e-10)

  def test_copy_freezing_and_context_inputs_remain_untouched(self):
    unary, head, hidden, logits, sigma, active = self.problem()
    self.assertTrue(unary.training)
    state = {name: value.clone() for name, value in unary.state_dict().items()}
    inputs = [value.clone() for value in (hidden, logits, sigma, active)]
    for value in (hidden, logits, sigma):
      value.requires_grad_(True)
    head.train().requires_grad_(True)
    self.assertFalse(head._frozen_unary.training)
    self.assertTrue(all(not p.requires_grad for p in head._frozen_unary.parameters()))
    self.assertTrue(all(p.requires_grad for p in unary.parameters()))
    for original, frozen in zip(unary.parameters(), head._frozen_unary.parameters()):
      self.assertNotEqual(original.data_ptr(), frozen.data_ptr())
    output = head(hidden, logits, sigma, active)
    loss = -centered_forest_log_probability(output, logits, output.candidate_ids[..., 0]).sum()
    loss.backward()
    self.assertGreater(float(head.right_token_embedding.weight.grad.abs().sum()), 1e-8)
    for value in (hidden, logits, sigma):
      self.assertIsNone(value.grad)
    for p in (*unary.parameters(), *head._frozen_unary.parameters()):
      self.assertIsNone(p.grad)
    for name, value in unary.state_dict().items():
      torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    for actual, original in zip((hidden, logits, sigma, active), inputs):
      torch.testing.assert_close(actual, original, atol=0, rtol=0)
    hidden_size, vocab, d, t = head.hidden_size, head.vocab_size, head.feature_dim, head.time_embed_dim
    expected = 2 * vocab * d + 2 * d * (hidden_size + t + 1) + 2 * hidden_size + 2 * t * t + 2 * t
    self.assertEqual(head.parameter_count, expected)
    self.assertEqual(head.parameter_summary()['positive_factor_rank'], 2 * d + 1)
    head.requires_grad_(False)
    self.assertEqual(head.parameter_count, 0)
    head.requires_grad_(True)
    self.assertEqual(head.parameter_count, expected)

  def test_target_independent_candidate_policy_and_natural_active_chain(self):
    _, head, hidden, logits, sigma, active = self.problem(batch=3, length=5)
    active[:] = torch.tensor([[True, False, True, False, True],
                              [False, False, True, False, False],
                              [False, False, False, False, False]])
    output = head(hidden, logits, sigma, active)
    self.assertEqual(output.edge_index[0, output.edge_mask[0]].tolist(), [[0, 2], [2, 4]])
    self.assertEqual(output.edge_mask.sum(-1).tolist(), [2, 0, 0])
    expected_ids = logits.topk(head.top_k, -1).indices
    self.assertTrue(torch.equal(output.candidate_ids, expected_ids))
    targets = torch.randint(head.vocab_size, active.shape)
    first = centered_forest_log_probability(output, logits, targets)
    alternate = targets.clone()
    alternate[~active] = (alternate[~active] + 1) % head.vocab_size
    torch.testing.assert_close(first, centered_forest_log_probability(output, logits, alternate), atol=0, rtol=0)
    self.assertEqual(float(first[2].detach()), 0.0)
    self.make_dependent(head)
    changed = head(hidden + 0.2, logits, sigma + 0.3, active)
    self.assertTrue(torch.equal(changed.candidate_ids, expected_ids))
    self.assertTrue(torch.equal(changed.edge_index, output.edge_index))
    self.assertTrue(torch.equal(changed.edge_mask, output.edge_mask))
    with self.assertRaisesRegex(ValueError, 'forward mask'):
      centered_forest_log_probability(output, logits, targets, ~active)

  def test_zero_residual_mass_disabled_states_and_neutral_residual_interactions(self):
    _, head, hidden, logits, sigma, active = self.problem(batch=2, length=2, vocab=4, top_k=3)
    logits[0, :, 2:] = -torch.inf  # Includes a disabled explicit state and residual.
    logits[1, :, 3] = -torch.inf  # Only the residual is disabled.
    self.make_dependent(head)
    output = head(hidden, logits, sigma, active)
    self.assertTrue(torch.isneginf(output.base_tail_log_mass).all())
    fallback = output.residual_log_probs(logits).exp()
    torch.testing.assert_close(fallback.sum(-1), torch.ones(2, 2, dtype=torch.float64), atol=0, rtol=0)
    self.assertTrue((fallback.gather(-1, output.candidate_ids[..., :1]) == 1).all())
    dense = output.materialize_pair_factors()
    self.assertTrue((dense[..., -1, :] == 1).all())
    self.assertTrue((dense[..., :, -1] == 1).all())
    self.assertTrue((output.pair_left_factors > 0).all())
    self.assertTrue((output.pair_right_factors > 0).all())
    inference = infer_structured_distribution(output, active, 'low_rank')
    torch.testing.assert_close(inference.marginals.node_marginals, output.unary_log_potentials.exp(), atol=2e-14, rtol=0)
    torch.testing.assert_close(inference.marginals.log_partition, torch.zeros(2, dtype=torch.float64), atol=2e-14, rtol=0)
    targets = torch.zeros_like(active, dtype=torch.long)
    loss = -centered_forest_log_probability(output, logits, targets).mean()
    loss.backward()
    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
    impossible = torch.full_like(targets, 3)
    self.assertTrue(torch.isneginf(centered_forest_log_probability(output, logits, impossible)).all())
    expanded = full_vocabulary_marginals(output, logits, active, inference)
    self.assertTrue(torch.isfinite(expanded).all())
    self.assertTrue((expanded[..., 3] == 0).all())

  def test_top_k_entire_vocabulary_has_no_tail_or_nan(self):
    _, head, hidden, logits, sigma, active = self.problem(top_k=4)
    output = head(hidden, logits, sigma, active)
    self.assertTrue(torch.isneginf(output.base_tail_log_mass).all())
    inference = infer_structured_distribution(output, active, 'low_rank')
    expanded = full_vocabulary_marginals(output, logits, active, inference)
    torch.testing.assert_close(expanded.sum(-1), torch.ones(1, 3, dtype=torch.float64), atol=2e-14, rtol=0)
    self.assertTrue(torch.isfinite(expanded).all())

  def test_mixed_empty_and_nonempty_residuals_work_with_existing_samplers(self):
    _, head, hidden, logits, sigma, active = self.problem(batch=2, length=2)
    logits[0, 0, 2:] = -torch.inf
    logits[1, 1, 2:] = -torch.inf
    self.make_dependent(head)
    output = head(hidden, logits, sigma, active)
    self.assertEqual(torch.isneginf(output.base_tail_log_mass).tolist(), [[True, False], [False, True]])
    inference = infer_structured_distribution(output, active, 'low_rank')
    expected = full_vocabulary_marginals(output, logits, active, inference)
    self.assertTrue(torch.isfinite(expected).all())
    for sampler in (sample_structured_tokens, sample_structured_marginal_tokens):
      samples = sampler(output, logits, active, num_samples=20000,
                         generator=torch.Generator().manual_seed(13), inference=inference)
      self.assertTrue((samples[0, :, 0] < 2).all())
      self.assertTrue((samples[1, :, 1] < 2).all())
      for batch, node in itertools.product(range(2), repeat=2):
        empirical = torch.bincount(samples[batch, :, node], minlength=4).double() / samples.shape[1]
        torch.testing.assert_close(empirical, expected[batch, node], atol=0.015, rtol=0)

  def test_one_node_and_all_observed_have_zero_edge_gradient_without_failure(self):
    for length in (1, 3):
      _, head, hidden, logits, sigma, active = self.problem(batch=2, length=length)
      active[1] = False
      output = head(hidden, logits, sigma, active)
      score = centered_forest_log_probability(output, logits, logits.argmax(-1))
      self.assertEqual(float(score[1].detach()), 0)
      (-score.sum()).backward()
      self.assertTrue(torch.isfinite(head.right_token_embedding.weight.grad).all())
      if length == 1:
        self.assertEqual(output.pair_left_factors.shape[1], 0)
        self.assertEqual(float(head.right_token_embedding.weight.grad.abs().sum()), 0)

  def test_ambient_autocast_does_not_change_reference_or_coupling_scores(self):
    _, head, hidden, logits, sigma, active = self.problem()
    self.make_dependent(head)
    targets = logits.argmin(-1)
    reference = head(hidden, logits, sigma, active)
    reference_score = centered_forest_log_probability(reference, logits, targets)
    with torch.autocast('cpu', dtype=torch.bfloat16):
      output = head(hidden, logits, sigma, active)
      score = centered_forest_log_probability(output, logits, targets)
      dense = output.materialize_pair_factors()
      tail = output.residual_log_probs(logits)
    for actual, expected in ((output.unary_log_potentials, reference.unary_log_potentials),
                             (output.pair_left_factors, reference.pair_left_factors),
                             (output.pair_right_factors, reference.pair_right_factors),
                             (score, reference_score), (dense, reference.materialize_pair_factors()),
                             (tail, reference.residual_log_probs(logits))):
      self.assertEqual(actual.dtype, torch.float64)
      torch.testing.assert_close(actual, expected, atol=0, rtol=0)

  def test_invalid_inputs_are_rejected(self):
    unary, head, hidden, logits, sigma, active = self.problem()
    unary.correction_domain = 'full'
    with self.assertRaisesRegex(ValueError, 'candidate-only'):
      FrozenUnaryCenteredForestHead(unary)
    for changes in ({'eta': 1}, {'eta': 0}, {'feature_dim': 0}, {'embedding_init_std': 0}):
      unary.correction_domain = 'candidates'
      with self.assertRaises(ValueError):
        FrozenUnaryCenteredForestHead(unary, **changes)
    with self.assertRaisesRegex(ValueError, 'candidate policy'):
      head(hidden, logits.bfloat16(), sigma, active)
    with self.assertRaisesRegex(ValueError, 'nonempty logit support'):
      head(hidden, torch.full_like(logits, -torch.inf), sigma, active)
    with self.assertRaisesRegex(ValueError, 'nonempty logit support'):
      head(hidden, logits, sigma * torch.nan, active)

  def test_learns_ab_ba_with_fixed_half_half_marginals_and_existing_sampler(self):
    # eta=.95 caps valid mass at .975; >.95 demonstrates useful expressivity,
    # not an unrestricted binary copula or a claim about language-model data.
    for seed in (11, 23, 37):
      torch.manual_seed(seed)
      unary = ContextualUnaryAdapter(2, 3, rank=2, time_embed_dim=2, top_k=2)
      head = FrozenUnaryCenteredForestHead(unary, feature_dim=1, eta=0.95)
      hidden = torch.zeros(2, 2, 2)
      logits = torch.tensor([[[0., 0., -torch.inf]]]).expand(2, 2, 3).clone()
      sigma = torch.full((2,), 0.4)
      active = torch.ones(2, 2, dtype=torch.bool)
      targets = torch.tensor([[0, 1], [1, 0]])
      optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=0.08)
      for step in range(250):
        optimizer.zero_grad(set_to_none=True)
        output = head(hidden, logits, sigma, active)
        loss = -centered_forest_log_probability(output, logits, targets).mean()
        loss.backward()
        if step == 0:
          self.assertGreater(float(head.right_token_embedding.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
        optimizer.step()
      output = head(hidden, logits, sigma, active)
      inference = infer_structured_distribution(output, active, 'low_rank')
      mass = centered_forest_log_probability(output, logits, targets).exp().sum()
      self.assertGreater(float(mass.detach()), 0.95, f'seed={seed}')
      self.assertLessEqual(float(mass.detach()), (1 + head.eta) / 2 + 1e-14)
      expanded = full_vocabulary_marginals(output, logits, active, inference)
      torch.testing.assert_close(expanded, logits.double().softmax(-1), atol=2e-14, rtol=0)
      self.assertLessEqual(float(output.materialize_pair_factors().detach().log().max()), math.log1p(head.eta) + 1e-14)
      if seed == 11:
        samples = sample_structured_tokens(output, logits, active, num_samples=20000,
                                            generator=torch.Generator().manual_seed(17), inference=inference)
        valid = (samples[0, :, 0] != samples[0, :, 1]).double().mean()
        self.assertAlmostEqual(float(valid), float(mass.detach()), delta=0.01)
        self.assertAlmostEqual(float((samples[0, :, 0] == 0).double().mean()), 0.5, delta=0.015)


if __name__ == '__main__':
  unittest.main()
