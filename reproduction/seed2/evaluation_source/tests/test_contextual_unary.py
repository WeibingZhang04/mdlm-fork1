"""Active unary control: exact probabilities, support, gradients and masking."""

import copy
import io
import itertools
import unittest

import torch

from models.contextual_unary import ContextualUnaryAdapter
from models.structured_decoder import ContextualCouplingForestHead


class ContextualUnaryTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(928)
    self.hidden = torch.randn(2, 4, 7)
    self.base = torch.randn(2, 4, 11)
    self.time = torch.tensor([0.2, 0.8])
    self.targets = torch.randint(11, (2, 4))
    self.active = torch.tensor([
      [True, True, False, True], [True, False, True, False]])

  def head(self, mode='candidates', top_k=4):
    return ContextualUnaryAdapter(
      hidden_size=7, vocab_size=11, rank=3, time_embed_dim=6,
      top_k=top_k, correction_domain=mode)

  @staticmethod
  def activate(head):
    with torch.no_grad():
      head.hidden_projection.weight.normal_(std=0.7)
      head.hidden_projection.bias.normal_(std=0.3)
      head.time_projection.weight.normal_(std=0.6)
      head.token_embedding.weight.normal_(std=0.8)
    return head

  def test_initial_logits_preserve_base_exactly(self):
    for mode in ('full', 'candidates'):
      with self.subTest(mode=mode):
        head = self.head(mode)
        logits = head(self.hidden, self.base, self.time, self.active)
        torch.testing.assert_close(logits, self.base, atol=0, rtol=0)
        lp = head.target_log_probs(
          self.hidden, self.base, self.time, self.targets, self.active,
          position_chunk_size=2, vocab_chunk_size=3)
        expected = self.base.log_softmax(-1).gather(
          -1, self.targets[..., None]).squeeze(-1) * self.active
        torch.testing.assert_close(lp, expected, atol=5e-7, rtol=5e-7)

  def test_context_time_and_mask_affect_only_intended_rows(self):
    for mode in ('full', 'candidates'):
      with self.subTest(mode=mode):
        head = self.activate(self.head(mode))
        logits = head(self.hidden, self.base, self.time, self.active)
        torch.testing.assert_close(logits[~self.active], self.base[~self.active])
        self.assertGreater(float((logits[self.active] - self.base[self.active]).detach().abs().sum()), 0)
        changed = self.hidden.clone()
        changed[0, 0, 0] += 4
        changed_logits = head(changed, self.base, self.time, self.active)
        self.assertGreater(float((changed_logits[0, 0] - logits[0, 0]).detach().abs().sum()), 0)
        # An output row never reads another row's adapter input. The backbone
        # itself may be bidirectional, but the conditional output is unary.
        torch.testing.assert_close(changed_logits[:, 1:], logits[:, 1:])
        torch.testing.assert_close(changed_logits[1], logits[1])
        timed = head(self.hidden, self.base, self.time + 0.4, self.active)
        self.assertGreater(float((timed - logits).detach().abs().sum()), 0)

  def test_candidates_and_residual_match_full_vocabulary_distribution(self):
    head = self.activate(self.head())
    lattice = head.candidate_lattice(
      self.hidden, self.base, self.time, self.active, vocab_chunk_size=3)
    ids = self.base.topk(4, dim=-1).indices
    torch.testing.assert_close(lattice.candidate_ids, ids)
    logits = head(self.hidden, self.base, self.time, self.active)
    support = torch.zeros_like(self.base, dtype=torch.bool).scatter(-1, ids, True)
    torch.testing.assert_close(logits[~support], self.base[~support], atol=0, rtol=0)
    full_lp = logits.log_softmax(-1)
    torch.testing.assert_close(
      lattice.log_probs[..., :4], full_lp.gather(-1, ids), atol=5e-7, rtol=5e-7)
    expected_tail = torch.logsumexp(full_lp.masked_fill(support, -torch.inf), dim=-1)
    torch.testing.assert_close(lattice.log_probs[..., -1], expected_tail)
    tail_base = self.base.masked_fill(support, -torch.inf)
    expected_residual = torch.logsumexp(tail_base, dim=-1)
    torch.testing.assert_close(lattice.residual_log_mass, expected_residual)

  def test_chunked_target_probabilities_and_gradients_equal_dense(self):
    for mode in ('full', 'candidates'):
      with self.subTest(mode=mode):
        chunked = self.activate(self.head(mode)).double()
        dense = copy.deepcopy(chunked)
        hidden, base, time = self.hidden.double(), self.base.double(), self.time.double()
        logits = dense(hidden, base, time, self.active)
        dense_lp = logits.log_softmax(-1).gather(
          -1, self.targets[..., None]).squeeze(-1) * self.active
        lp = chunked.target_log_probs(
          hidden, base, time, self.targets, self.active,
          position_chunk_size=2, vocab_chunk_size=3)
        torch.testing.assert_close(lp, dense_lp, atol=1e-12, rtol=1e-12)
        (-lp.sum()).backward()
        (-dense_lp.sum()).backward()
        for (name, actual), (expected_name, expected) in zip(
            chunked.named_parameters(), dense.named_parameters()):
          self.assertEqual(name, expected_name)
          torch.testing.assert_close(actual.grad, expected.grad, atol=1e-11, rtol=1e-10)

  def test_all_parameters_receive_learning_signal_after_identity_step(self):
    for mode in ('full', 'candidates'):
      with self.subTest(mode=mode):
        head = self.head(mode)
        # Include all retained IDs across positions so the tiny vocabulary
        # embedding has several active rows in candidate mode.
        optimizer = torch.optim.SGD(head.parameters(), lr=0.5)
        first_loss = head.nll_loss(
          self.hidden, self.base, self.time, self.targets, self.active)
        first_loss.backward()
        self.assertGreater(float(head.hidden_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.time_projection.weight.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss = head.nll_loss(
          self.hidden, self.base, self.time, self.targets, self.active)
        loss.backward()
        for name, parameter in head.named_parameters():
          self.assertIsNotNone(parameter.grad, name)
          self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
          self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        self.assertLess(float(loss.detach()), float(first_loss.detach()))

  def test_distribution_factorizes_by_enumeration(self):
    head = self.activate(ContextualUnaryAdapter(
      hidden_size=7, vocab_size=3, rank=3, time_embed_dim=6,
      top_k=2, correction_domain='candidates'))
    hidden = self.hidden[:1, :2]
    base = self.base[:1, :2, :3]
    logp = head(hidden, base, self.time[:1]).log_softmax(-1)[0]
    joint = torch.stack([
      (logp[0, a] + logp[1, b]).exp() for a, b in itertools.product(range(3), repeat=2)
    ]).reshape(3, 3)
    torch.testing.assert_close(joint.sum(), torch.tensor(1.0))
    torch.testing.assert_close(joint.sum(dim=1), logp[0].exp())
    torch.testing.assert_close(joint.sum(dim=0), logp[1].exp())
    torch.testing.assert_close(joint, joint.sum(1)[:, None] * joint.sum(0)[None, :])

  def test_all_vocabulary_candidates_equal_full_correction_mode(self):
    head = self.activate(self.head('candidates', top_k=11))
    full = self.head('full', top_k=11)
    full.load_state_dict(head.state_dict())
    torch.testing.assert_close(
      head(self.hidden, self.base, self.time), full(self.hidden, self.base, self.time))
    lattice = head.candidate_lattice(self.hidden, self.base, self.time, vocab_chunk_size=3)
    self.assertTrue(bool(torch.isneginf(lattice.residual_log_mass).all()))
    self.assertFalse(bool(lattice.candidate_state_mask[..., -1].any()))
    lp = head.target_log_probs(
      self.hidden, self.base, self.time, self.targets, vocab_chunk_size=3)
    torch.testing.assert_close(lp, full.target_log_probs(
      self.hidden, self.base, self.time, self.targets, vocab_chunk_size=3))

  def test_tiny_residual_mass_is_preserved(self):
    head = self.head(top_k=1).double()
    base = torch.full_like(self.base.double(), -1000.0)
    base[..., 0] = 0.0
    lattice = head.candidate_lattice(self.hidden.double(), base, self.time.double(), vocab_chunk_size=3)
    expected = torch.full_like(base[..., 0], -1000.0 + torch.log(torch.tensor(10.0)).item())
    torch.testing.assert_close(lattice.residual_log_mass, expected, atol=1e-6, rtol=0)
    targets = torch.ones_like(self.targets)
    lp = head.target_log_probs(self.hidden.double(), base, self.time.double(), targets,
                              vocab_chunk_size=3)
    self.assertTrue(bool(torch.isfinite(lp).all()))
    torch.testing.assert_close(lp, torch.full_like(lp, -1000.0))

  def test_base_gradients_remain_finite_when_tail_chunks_are_empty(self):
    for top_k in (4, 11):
      with self.subTest(top_k=top_k):
        head = self.activate(self.head(top_k=top_k)).double()
        base = self.base.double().requires_grad_()
        dense_base = base.detach().clone().requires_grad_()
        logp = head.target_log_probs(
          self.hidden.double(), base, self.time.double(), self.targets,
          self.active, vocab_chunk_size=1)
        dense = head(self.hidden.double(), dense_base, self.time.double())
        expected = dense.log_softmax(-1).gather(-1, self.targets[..., None]).squeeze(-1)
        (-logp.sum()).backward()
        (-(expected * self.active).sum()).backward()
        self.assertTrue(bool(torch.isfinite(base.grad).all()))
        torch.testing.assert_close(base.grad, dense_base.grad, atol=1e-12, rtol=1e-12)

  def test_full_mode_excluded_only_chunks_have_finite_exact_gradients(self):
    for checkpoint_chunks in (False, True):
      for excluded_token in (0, 10):
        with self.subTest(checkpoint=checkpoint_chunks, excluded=excluded_token):
          chunked = self.activate(self.head('full')).double()
          dense = copy.deepcopy(chunked)
          base = self.base.double().clone()
          base[..., excluded_token] = -torch.inf
          base.requires_grad_()
          dense_base = base.detach().clone().requires_grad_()
          targets = self.targets.clone()
          targets[targets == excluded_token] = 1
          actual = chunked.target_log_probs(
            self.hidden.double(), base, self.time.double(), targets, self.active,
            vocab_chunk_size=1, checkpoint_chunks=checkpoint_chunks)
          expected = dense(self.hidden.double(), dense_base, self.time.double())
          expected = expected.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
          expected = torch.where(self.active, expected, torch.zeros_like(expected))
          torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
          (-actual.sum()).backward()
          (-expected.sum()).backward()
          for (name, parameter), (_, reference) in zip(chunked.named_parameters(), dense.named_parameters()):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-11, rtol=1e-10)
          self.assertTrue(bool(torch.isfinite(base.grad).all()))
          torch.testing.assert_close(base.grad, dense_base.grad, atol=1e-12, rtol=1e-12)

  def test_empty_mask_returns_differentiable_zero(self):
    head = self.head()
    active = torch.zeros_like(self.active)
    loss = head.nll_loss(self.hidden, self.base, self.time, self.targets, active)
    self.assertEqual(float(loss.detach()), 0)
    loss.backward()
    self.assertIsNotNone(head.hidden_projection.weight.grad)
    self.assertEqual(float(head.hidden_projection.weight.grad.abs().sum()), 0)

  def test_parameter_difference_matches_real_forest_pair_branch(self):
    head = self.head()
    forest = ContextualCouplingForestHead(
      hidden_size=7, vocab_size=11, rank=3, time_embed_dim=6,
      top_k=4, topology_dim=8, num_anchor_slots=2, contextual_neighbors=2)
    pair_modules = (forest.hidden_norm, forest.time_embedding,
                    forest.token_factor_embedding, forest.factor_hidden_projection,
                    forest.factor_time_projection)
    pair_parameters = sum(p.numel() for module in pair_modules for p in module.parameters())
    summary = head.parameter_summary()
    self.assertEqual(summary['trainable_parameters'], sum(p.numel() for p in head.parameters()))
    self.assertEqual(summary['forest_pair_branch_parameters'], pair_parameters)
    self.assertEqual(summary['difference_from_forest_pair_branch'], 3 * -(7 + 6 + 1))

  def test_save_load_and_timestep_shapes(self):
    head = self.activate(self.head())
    data = io.BytesIO()
    torch.save(head.state_dict(), data)
    data.seek(0)
    restored = self.head()
    restored.load_state_dict(torch.load(data, weights_only=True))
    torch.testing.assert_close(
      head(self.hidden, self.base, self.time), restored(self.hidden, self.base, self.time))
    scalar = torch.tensor(0.5)
    torch.testing.assert_close(
      head(self.hidden, self.base, scalar), head(self.hidden, self.base, scalar.expand(2, 1)))

  def test_invalid_shapes_and_modes_fail(self):
    head = self.head()
    with self.assertRaisesRegex(ValueError, 'active_mask'):
      head(self.hidden, self.base, self.time, self.active.float())
    with self.assertRaisesRegex(ValueError, 'chunk sizes'):
      head.target_log_probs(self.hidden, self.base, self.time, self.targets, vocab_chunk_size=0)
    with self.assertRaisesRegex(ValueError, 'valid vocabulary'):
      head.target_log_probs(self.hidden, self.base, self.time, self.targets + 11)
    with self.assertRaisesRegex(ValueError, 'candidate correction'):
      self.head('full').candidate_lattice(self.hidden, self.base, self.time)


if __name__ == '__main__':
  unittest.main()
