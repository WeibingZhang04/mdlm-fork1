"""Deterministic CPU tests for the CRF recovery helpers.

These tests intentionally import only PyTorch and ``crf_utils`` so they do not
require FlashAttention, Lightning, a tokenizer download, or a GPU.
"""

import unittest

import torch
import torch.nn as nn

import crf_utils


class LinearWarmupWeightTest(unittest.TestCase):

  def test_schedule_hits_start_midpoint_and_endpoint(self):
    self.assertEqual(
      crf_utils.linear_warmup_weight(0, 0.0, 0.1, 10000), 0.0)
    self.assertAlmostEqual(
      crf_utils.linear_warmup_weight(5000, 0.0, 0.1, 10000),
      0.05)
    self.assertAlmostEqual(
      crf_utils.linear_warmup_weight(10000, 0.0, 0.1, 10000),
      0.1)
    self.assertAlmostEqual(
      crf_utils.linear_warmup_weight(50000, 0.0, 0.1, 10000),
      0.1)


class UnigramAuxiliaryLossTest(unittest.TestCase):

  def test_zero_initialised_head_receives_gradient(self):
    torch.manual_seed(7)
    batch, length, hidden_size, vocab_size = 2, 4, 3, 5
    mask_index = vocab_size - 1
    head = nn.Linear(hidden_size, vocab_size)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)

    hidden = torch.randn(batch, length, hidden_size)
    logits = head(hidden)
    xt = torch.full((batch, length), mask_index, dtype=torch.long)
    x0 = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    diffusion_weight = torch.tensor([1.0, 2.0])
    schedule_weight = crf_utils.linear_warmup_weight(
      5000, 0.0, 0.1, 10000)

    loss = schedule_weight * crf_utils.unigram_denoising_loss(
      logits, xt, x0, mask_index, diffusion_weight).mean()
    loss.backward()

    self.assertGreater(head.weight.grad.abs().sum().item(), 0.0)
    self.assertGreater(head.bias.grad.abs().sum().item(), 0.0)

  def test_unmasked_positions_have_zero_auxiliary_loss(self):
    logits = torch.zeros(1, 3, 4)
    mask_index = 3
    xt = torch.tensor([[3, 1, 3]])
    x0 = torch.tensor([[0, 1, 2]])
    losses = crf_utils.unigram_denoising_loss(
      logits, xt, x0, mask_index, token_weight=1.0)
    self.assertEqual(losses[0, 1].item(), 0.0)
    self.assertGreater(losses[0, 0].item(), 0.0)
    self.assertGreater(losses[0, 2].item(), 0.0)


class ConfidentRevealTest(unittest.TestCase):

  def test_reveals_exact_count_at_highest_confidences(self):
    mask_index = 3
    x = torch.tensor([
      [3, 3, 3, 3, 3, 3],
      [0, 3, 3, 3, 3, 1],
    ])
    probabilities = torch.zeros(2, 6, 4)
    probabilities[0, :, 0] = torch.tensor(
      [0.1, 0.9, 0.4, 0.8, 0.2, 0.7])
    probabilities[1, :, 0] = torch.tensor(
      [1.0, 0.2, 0.6, 0.9, 0.4, 1.0])

    reveal = crf_utils.confident_reveal_mask(
      x=x,
      probabilities=probabilities,
      mask_index=mask_index,
      move_chance_t=torch.ones(2, 1, 1),
      move_chance_s=torch.full((2, 1, 1), 0.5))

    # Six current masks -> reveal three; four -> reveal two.
    self.assertEqual(reveal[0].sum().item(), 3)
    self.assertEqual(reveal[1].sum().item(), 2)
    self.assertEqual(
      set(reveal[0].nonzero().flatten().tolist()), {1, 3, 5})
    self.assertEqual(
      set(reveal[1].nonzero().flatten().tolist()), {2, 3})
    self.assertFalse((reveal & (x != mask_index)).any().item())

  def test_integer_rounding_never_reveals_too_early(self):
    x = torch.full((1, 3), 2, dtype=torch.long)
    probabilities = torch.tensor([[
      [0.9, 0.1, 0.0],
      [0.8, 0.2, 0.0],
      [0.7, 0.3, 0.0],
    ]])
    reveal = crf_utils.confident_reveal_mask(
      x, probabilities, mask_index=2,
      move_chance_t=1.0, move_chance_s=0.9)
    # ceil(3 * 0.9) == 3, so this tiny step keeps all sites masked.
    self.assertEqual(reveal.sum().item(), 0)

  def test_sequential_reveals_one_only_for_unfinished_rows(self):
    mask_index = 2
    x = torch.tensor([
      [2, 2, 2, 2],
      [0, 1, 0, 1],
    ])
    probabilities = torch.zeros(2, 4, 3)
    probabilities[0, :, 0] = torch.tensor([0.1, 0.8, 0.4, 0.7])
    probabilities[1, :, 0] = 1.0
    reveal = crf_utils.sequential_reveal_mask(
      x, probabilities, mask_index)
    self.assertEqual(reveal[0].sum().item(), 1)
    self.assertTrue(reveal[0, 1].item())
    self.assertEqual(reveal[1].sum().item(), 0)


class ChainConstraintTest(unittest.TestCase):

  def test_invalid_observed_state_cannot_affect_neighbor(self):
    emission = torch.zeros(1, 2)
    transitions_a = torch.zeros(1, 2, 2, 2)
    transitions_b = transitions_a.clone()
    # Make the alternate state at observed position 1 attractive, then vary
    # only its outgoing edge. Without a hard constraint this changes pos 2.
    transitions_a[:, 0, :, 1] = 5.0
    transitions_b[:, 0, :, 1] = 5.0
    transitions_a[:, 1, 1, :] = torch.tensor([8.0, -8.0])
    transitions_b[:, 1, 1, :] = torch.tensor([-8.0, 8.0])
    unconstrained_a = crf_utils.forward_backward(
      emission, transitions_a)
    unconstrained_b = crf_utils.forward_backward(
      emission, transitions_b)
    self.assertFalse(torch.allclose(
      unconstrained_a[:, 2], unconstrained_b[:, 2]))

    # Position 1 is observed as candidate 0; all other states are free.
    candidate_valid = torch.tensor([[
      [True, True],
      [True, False],
      [True, True],
    ]])
    emission_a, constrained_a = crf_utils.constrain_chain_potentials(
      emission, transitions_a, candidate_valid)
    emission_b, constrained_b = crf_utils.constrain_chain_potentials(
      emission, transitions_b, candidate_valid)
    marginals_a = crf_utils.forward_backward(emission_a, constrained_a)
    marginals_b = crf_utils.forward_backward(emission_b, constrained_b)

    self.assertTrue(torch.allclose(
      marginals_a, marginals_b, atol=1e-6))
    self.assertGreater(marginals_a[0, 1, 0].item(), 0.999)
    self.assertLess(marginals_a[0, 1, 1].item(), 1e-6)


class ChunkedTransitionTest(unittest.TestCase):

  def test_query_chunking_matches_dense_full_vocab_normalization(self):
    torch.manual_seed(11)
    batch, queries, vocab, candidates = 2, 7, 13, 4
    excluded_index = vocab - 1
    logits = torch.randn(batch, queries, vocab)
    gather_indices = torch.randint(
      0, vocab - 1, (batch, queries, candidates))

    dense_logits = logits.clone()
    dense_logits[..., excluded_index] = -1e6
    dense = torch.gather(
      torch.log_softmax(dense_logits, dim=-1),
      -1,
      gather_indices)

    chunk_calls = []

    def logits_fn(start, end):
      chunk_calls.append((start, end))
      return logits[:, start:end, :].clone()

    chunked = crf_utils.chunked_normalized_gather(
      logits_fn=logits_fn,
      gather_indices=gather_indices,
      query_chunk_size=3,
      excluded_index=excluded_index)
    self.assertTrue(torch.allclose(dense, chunked, atol=1e-6))
    self.assertEqual(chunk_calls, [(0, 3), (3, 6), (6, 7)])


if __name__ == '__main__':
  unittest.main()
