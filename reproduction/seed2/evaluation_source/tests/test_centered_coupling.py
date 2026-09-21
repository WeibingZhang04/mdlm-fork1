"""Exact marginals, neutral residuals, low-rank compatibility and AB/BA fit."""

import itertools
import math
import unittest

import torch

from evaluation.centered_coupling import centered_coupling
from structured_utils import forest_sum_product_low_rank, sample_forest_low_rank


class CenteredCouplingTest(unittest.TestCase):

  def setUp(self):
    torch.set_num_threads(1)
    torch.manual_seed(190)

  def problem(self, *, dtype=torch.float64):
    left = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.01, 0.69, 0.1, 0.2]], dtype=dtype)
    right = torch.tensor([[0.1, 0.5, 0.2, 0.2], [0.4, 0.1, 0.3, 0.2]], dtype=dtype)
    raw_left = torch.randn(2, 3, 4, dtype=dtype)
    raw_right = torch.randn(2, 3, 4, dtype=dtype)
    return left, right, raw_left, raw_right

  def test_positive_rank_and_exact_node_edge_marginals(self):
    left, right, raw_left, raw_right = self.problem()
    result = centered_coupling(left, right, raw_left, raw_right, eta=torch.tensor([0.8, 0.999]))
    self.assertEqual(result.left_factors.shape, (2, 4, 9))
    self.assertTrue((result.left_factors > 0).all())
    self.assertTrue((result.right_factors > 0).all())
    expected_ratio = 1 + result.eta[..., None, None] * (
      result.left_features @ result.right_features.transpose(-1, -2)) / raw_left.shape[-1]
    torch.testing.assert_close(result.density_ratio(), expected_ratio, atol=1e-14, rtol=1e-14)
    joint = result.joint()
    torch.testing.assert_close(joint.sum(-1), left, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(joint.sum(-2), right, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(joint.sum((-2, -1)), torch.ones(2, dtype=torch.float64), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close((left[..., None] * result.left_features).sum(-2),
                               torch.zeros(2, 4, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close((right[..., None] * result.right_features).sum(-2),
                               torch.zeros(2, 4, dtype=torch.float64), atol=1e-14, rtol=0)

  def test_residual_dot_products_are_one_but_endpoint_rows_need_not_be(self):
    result = centered_coupling(*self.problem(), eta=0.8)
    ratio = result.density_ratio()
    torch.testing.assert_close(ratio[..., -1, :], torch.ones_like(ratio[..., -1, :]), atol=1e-14, rtol=0)
    torch.testing.assert_close(ratio[..., :, -1], torch.ones_like(ratio[..., :, -1]), atol=1e-14, rtol=0)
    self.assertTrue((result.left_features[..., -1, :] == 0).all())
    self.assertTrue((result.right_features[..., -1, :] == 0).all())
    self.assertFalse(torch.equal(result.right_factors[..., -1, :], torch.ones_like(result.right_factors[..., -1, :])))

  def test_observed_low_rank_scores_match_dense_fp64(self):
    result = centered_coupling(*self.problem(), eta=0.91)
    joint = result.joint()
    for left, right in itertools.product(range(4), repeat=2):
      observed = result.log_probability(torch.tensor([left, left]), torch.tensor([right, right]))
      torch.testing.assert_close(observed, joint[:, left, right].log(), atol=1e-14, rtol=1e-14)
    ratio = result.density_ratio()
    self.assertTrue((ratio.log() <= math.log1p(0.91) + 1e-14).all())
    self.assertTrue((ratio >= 1 - 0.91 - 1e-14).all())

  def test_existing_residual_neutral_forest_inference_preserves_all_three_nodes(self):
    b = torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3], [0.1, 0.4, 0.5]], dtype=torch.float64)
    edges = torch.tensor([[[0, 1], [1, 2]]])
    coupling = centered_coupling(b[[0, 1]][None], b[[1, 2]][None],
                                 torch.randn(1, 2, 2, 3, dtype=torch.float64),
                                 torch.randn(1, 2, 2, 3, dtype=torch.float64), eta=0.99)
    # Existing APIs accept explicit factors only, since the residual row and
    # column are supplied by their own hardcoded neutral-interaction logic.
    left, right = coupling.left_factors[..., :-1, :], coupling.right_factors[..., :-1, :]
    inference = forest_sum_product_low_rank(b.log()[None], left, right, edges)
    torch.testing.assert_close(inference.node_log_marginals.exp()[0], b, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(inference.log_partition, torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0)
    ratios = coupling.density_ratio()[0]
    assignments = list(itertools.product(range(3), repeat=3))
    mass = torch.stack([b[0, a] * b[1, c] * b[2, d] * ratios[0, a, c] * ratios[1, c, d]
                        for a, c, d in assignments])
    torch.testing.assert_close(mass.sum(), torch.tensor(1., dtype=torch.float64), atol=1e-14, rtol=0)
    for edge_index, (source, target) in enumerate(edges[0].tolist()):
      actual = torch.zeros(3, 3, dtype=torch.float64)
      for state, probability in zip(assignments, mass):
        actual[state[source], state[target]] += probability
      torch.testing.assert_close(actual, coupling.joint()[0, edge_index], atol=1e-14, rtol=1e-14)

  def test_sampling_matches_dense_joint_and_is_reproducible(self):
    left, right, raw_left, raw_right = self.problem()
    coupling = centered_coupling(left, right, raw_left, raw_right, eta=0.95)
    samples = coupling.sample(60000, generator=torch.Generator().manual_seed(83))
    self.assertEqual(samples.shape, (2, 60000, 2))
    for batch in range(2):
      empirical = torch.bincount(samples[batch, :, 0] * 4 + samples[batch, :, 1], minlength=16).double().reshape(4, 4) / 60000
      torch.testing.assert_close(empirical, coupling.joint()[batch], atol=0.006, rtol=0)
    small = coupling.sample(10, generator=torch.Generator().manual_seed(13))
    repeated = coupling.sample(10, generator=torch.Generator().manual_seed(13))
    self.assertTrue(torch.equal(small, repeated))
    scalar = centered_coupling(left[0], right[0], raw_left[0], raw_right[0])
    self.assertEqual(scalar.sample(3).shape, (3, 2))
    edges = torch.tensor([[[0, 1]], [[0, 1]]])
    nodes = torch.stack((left, right), dim=1)
    existing = sample_forest_low_rank(nodes.log(), coupling.left_factors[:, None, :-1],
                                      coupling.right_factors[:, None, :-1], edges,
                                      num_samples=60000, generator=torch.Generator().manual_seed(29))
    for batch in range(2):
      empirical = torch.bincount(existing[batch, :, 0] * 4 + existing[batch, :, 1], minlength=16).double().reshape(4, 4) / 60000
      torch.testing.assert_close(empirical, coupling.joint()[batch], atol=0.006, rtol=0)

  def test_zero_explicit_mass_and_masked_states_have_finite_gradients(self):
    left = torch.tensor([[0., 0., 1.], [1., 0., 0.]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[0.4, 0.6, 0.], [0., 0., 1.]], dtype=torch.float64, requires_grad=True)
    raw_left = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    raw_right = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    result = centered_coupling(left, right, raw_left, raw_right, eta=0.999)
    torch.testing.assert_close(result.joint().sum(-1), left)
    torch.testing.assert_close(result.joint().sum(-2), right)
    self.assertTrue((result.left_features[0] == 0).all())
    self.assertTrue((result.right_features[1] == 0).all())
    loss = -result.log_probability(torch.tensor([2, 0]), torch.tensor([0, 2])).sum()
    loss.backward()
    for tensor in (left, right, raw_left, raw_right):
      self.assertTrue(torch.isfinite(tensor.grad).all())
    self.assertTrue(torch.isneginf(result.log_probability(torch.tensor([0, 1]), torch.tensor([0, 0]))).all())
    samples = result.sample(100, generator=torch.Generator().manual_seed(2))
    self.assertTrue((samples[0, :, 0] == 2).all())
    self.assertTrue((samples[1, :, 0] == 0).all())
    self.assertTrue((samples[1, :, 1] == 2).all())

  def test_existing_forest_zero_support_inference_gradients_and_sampling(self):
    b = torch.tensor([[0.5, 0.5, 0.], [0.4, 0.6, 0.], [0., 0., 1.]], dtype=torch.float64)
    raw_left = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    raw_right = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    result = centered_coupling(b[[0, 1]][None], b[[1, 2]][None], raw_left, raw_right, eta=0.999)
    left, right = result.left_factors[..., :-1, :], result.right_factors[..., :-1, :]
    edges = torch.tensor([[[0, 1], [1, 2]]])
    inference = forest_sum_product_low_rank(b.log()[None], left, right, edges, state_mask=(b > 0)[None])
    torch.testing.assert_close(inference.node_log_marginals.exp()[0], b, atol=1e-14, rtol=0)
    torch.testing.assert_close(inference.log_partition, torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0)
    loss = inference.log_partition.sum() - inference.node_log_marginals[0, 0, 0]
    loss.backward()
    self.assertTrue(torch.isfinite(raw_left.grad).all())
    self.assertTrue(torch.isfinite(raw_right.grad).all())
    samples = sample_forest_low_rank(b.log()[None], left.detach(), right.detach(), edges, 1000,
                                      state_mask=(b > 0)[None], generator=torch.Generator().manual_seed(37))
    self.assertTrue((samples[0, :, :2] != 2).all())
    self.assertTrue((samples[0, :, 2] == 2).all())

  def test_ambient_cpu_autocast_cannot_change_probability_arithmetic(self):
    left, right = torch.softmax(torch.randn(9), 0), torch.softmax(torch.randn(9), 0)
    raw_left = torch.randn(8, 5, requires_grad=True)
    raw_right = torch.randn(8, 5, requires_grad=True)
    reference = centered_coupling(left, right, raw_left, raw_right, eta=0.999)
    states = torch.tensor(2), torch.tensor(3)
    expected_log_probability = reference.log_probability(*states)
    expected_joint = reference.joint()
    expected_samples = reference.sample(1000, generator=torch.Generator().manual_seed(9))
    with torch.autocast('cpu', dtype=torch.bfloat16):
      result = centered_coupling(left, right, raw_left, raw_right, eta=0.999)
      ratio, joint = result.density_ratio(), result.joint()
      log_probability = result.log_probability(*states)
      samples = result.sample(1000, generator=torch.Generator().manual_seed(9))
      self.assertTrue(torch.is_autocast_enabled('cpu'))
    for tensor in (result.left_features, result.right_features, ratio, joint, log_probability):
      self.assertEqual(tensor.dtype, torch.float32)
    torch.testing.assert_close(result.left_features, reference.left_features, atol=0, rtol=0)
    torch.testing.assert_close(result.right_features, reference.right_features, atol=0, rtol=0)
    torch.testing.assert_close(joint, expected_joint, atol=0, rtol=0)
    torch.testing.assert_close(log_probability, expected_log_probability, atol=0, rtol=0)
    self.assertTrue(torch.equal(samples, expected_samples))
    torch.testing.assert_close(joint.sum(-1), left, atol=2e-7, rtol=0)
    torch.testing.assert_close(joint.sum(-2), right, atol=2e-7, rtol=0)
    (-log_probability).backward()
    self.assertTrue(torch.isfinite(raw_left.grad).all())
    self.assertTrue(torch.isfinite(raw_right.grad).all())

  def test_mixed_dtype_marginals_must_be_normalized_in_working_precision(self):
    rounded = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    exact = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float32)
    raw_left, raw_right = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64)
    self.assertEqual(float(rounded.sum()), 1.0)
    self.assertNotEqual(float(rounded.double().sum()), 1.0)
    for left, right in ((rounded, exact), (exact, rounded)):
      with self.assertRaisesRegex(ValueError, 'normalized in working precision'):
        centered_coupling(left, right, raw_left, raw_right)
    result = centered_coupling(exact, exact, raw_left, raw_right)
    self.assertEqual(result.left_factors.dtype, torch.float64)
    torch.testing.assert_close(result.left_marginal, exact.double(), atol=0, rtol=0)
    torch.testing.assert_close(result.joint().sum(-1), exact.double(), atol=1e-14, rtol=0)
    torch.testing.assert_close(result.joint().sum(-2), exact.double(), atol=1e-14, rtol=0)

  def test_saturated_features_remain_strictly_positive_and_gradient_safe(self):
    for dtype in (torch.float32, torch.float64):
      with self.subTest(dtype=dtype):
        b = torch.tensor([0.000001, 0.799999, 0.2], dtype=dtype)
        raw_left = torch.tensor([[1000., -1000.], [-1000., 1000.]], dtype=dtype, requires_grad=True)
        raw_right = (-raw_left.detach()).requires_grad_()
        result = centered_coupling(b, b, raw_left, raw_right, eta=0.999)
        self.assertTrue((result.left_factors > 0).all())
        self.assertTrue((result.right_factors > 0).all())
        self.assertTrue((result.left_features.abs() < 1).all())
        self.assertTrue((result.right_features.abs() < 1).all())
        (-result.log_probability(torch.tensor(0), torch.tensor(1))).backward()
        self.assertTrue(torch.isfinite(raw_left.grad).all())
        self.assertTrue(torch.isfinite(raw_right.grad).all())

  def test_independent_initialization_has_nonzero_right_gradient(self):
    b = torch.tensor([0.5, 0.5, 0.], dtype=torch.float64)
    raw_left = torch.tensor([[0.9], [-0.3]], dtype=torch.float64, requires_grad=True)
    raw_right = torch.zeros(2, 1, dtype=torch.float64, requires_grad=True)
    result = centered_coupling(b, b, raw_left, raw_right, eta=0.999)
    torch.testing.assert_close(result.joint(), b[:, None] * b[None, :], atol=1e-14, rtol=0)
    loss = -(result.log_probability(torch.tensor(0), torch.tensor(1))
             + result.log_probability(torch.tensor(1), torch.tensor(0))) / 2
    loss.backward()
    self.assertGreater(float(raw_right.grad.abs().sum()), 0.01)
    self.assertTrue(torch.isfinite(raw_right.grad).all())
    torch.testing.assert_close(raw_left.grad, torch.zeros_like(raw_left), atol=1e-14, rtol=0)

  def test_autograd_includes_centering_and_strength(self):
    left = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    right = torch.tensor([0.4, 0.5, 0.1], dtype=torch.float64)
    raw_left = torch.tensor([[0.3, -0.7], [0.9, 0.2]], dtype=torch.float64, requires_grad=True)
    raw_right = torch.tensor([[-0.4, 0.8], [0.1, -0.6]], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor(0.83, dtype=torch.float64, requires_grad=True)
    function = lambda a, b, strength: centered_coupling(left, right, a, b, strength).joint()
    self.assertTrue(torch.autograd.gradcheck(function, (raw_left, raw_right, eta)))

  def test_three_seed_abba_training_exceeds_95_percent_with_fixed_marginals(self):
    b = torch.tensor([0.5, 0.5, 0.], dtype=torch.float64)
    for seed in (1, 7, 19):
      with self.subTest(seed=seed):
        generator = torch.Generator().manual_seed(seed)
        raw_left = torch.randn(2, 1, generator=generator, dtype=torch.float64).requires_grad_()
        raw_right = torch.zeros(2, 1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([raw_left, raw_right], lr=0.12)
        for step in range(200):
          optimizer.zero_grad()
          result = centered_coupling(b, b, raw_left, raw_right, eta=0.999)
          loss = -(result.log_probability(torch.tensor(0), torch.tensor(1))
                   + result.log_probability(torch.tensor(1), torch.tensor(0))) / 2
          self.assertTrue(torch.isfinite(loss))
          loss.backward()
          self.assertTrue(torch.isfinite(raw_left.grad).all())
          self.assertTrue(torch.isfinite(raw_right.grad).all())
          optimizer.step()
        joint = centered_coupling(b, b, raw_left, raw_right, eta=0.999).joint()
        self.assertGreater(float((joint[0, 1] + joint[1, 0]).detach()), 0.95)
        torch.testing.assert_close(joint.sum(-1), b, atol=1e-14, rtol=0)
        torch.testing.assert_close(joint.sum(-2), b, atol=1e-14, rtol=0)

  def test_invalid_strength_shapes_and_probability_inputs_fail(self):
    args = self.problem()
    for eta in (0., 1., -0.1, 1.1, float('nan'), torch.ones(3) * 0.5):
      with self.subTest(eta=eta), self.assertRaises(ValueError):
        centered_coupling(*args, eta=eta)
    with self.assertRaisesRegex(ValueError, 'normalized'):
      centered_coupling(args[0] * 0.8, *args[1:])
    with self.assertRaisesRegex(ValueError, 'matching'):
      centered_coupling(args[0], args[1], args[2][..., :2], args[3])
    malformed = args[2].clone()
    malformed[0, 0, 0] = torch.nan
    with self.assertRaisesRegex(ValueError, 'finite'):
      centered_coupling(args[0], args[1], malformed, args[3])
    with self.assertRaisesRegex(ValueError, 'FP32 or FP64'):
      centered_coupling(args[0].bfloat16(), args[1].bfloat16(), args[2], args[3])
    with self.assertRaisesRegex(ValueError, 'underflows'):
      centered_coupling(*args, eta=torch.nextafter(torch.tensor(0., dtype=torch.float64),
                                                   torch.tensor(1., dtype=torch.float64)))


if __name__ == '__main__':
  unittest.main()
