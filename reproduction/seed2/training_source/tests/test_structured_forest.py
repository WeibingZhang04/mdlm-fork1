"""Gate-0 tests for exact globally normalised forest inference.

All correctness references use exhaustive FP64 enumeration.  The test module
depends only on PyTorch and the local ``structured_utils`` module.
"""

import unittest

import torch
import torch.nn.functional as F

import structured_utils


class Float64TestCase(unittest.TestCase):
  """Use FP64 within each reference test without leaking global state."""

  def setUp(self):
    self._previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

  def tearDown(self):
    torch.set_default_dtype(self._previous_default_dtype)


def _batched_problem():
  """Two padded forests: one tree and one two-component forest."""
  generator = torch.Generator().manual_seed(193)
  node = torch.randn(2, 4, 3, generator=generator, dtype=torch.float64)
  pair = 0.7 * torch.randn(
    2, 3, 3, 3, generator=generator, dtype=torch.float64)
  edges = torch.tensor([
    [[0, 1], [2, 1], [2, 3]],
    [[2, 0], [1, 3], [-1, -1]],
  ])
  edge_mask = torch.tensor([
    [True, True, True],
    [True, True, False],
  ])
  return node, pair, edges, edge_mask


class EnumerationAgreementTest(Float64TestCase):

  def test_batched_sum_product_matches_fp64_enumeration(self):
    node, pair, edges, edge_mask = _batched_problem()
    exact = structured_utils.forest_sum_product(
      node, pair, edges, edge_mask=edge_mask, max_components=2)
    reference = structured_utils.enumerate_forest_distribution(
      node, pair, edges, edge_mask=edge_mask, max_components=2)

    torch.testing.assert_close(
      exact.log_partition, reference.log_partition,
      atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
      exact.node_marginals, reference.node_marginals,
      atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
      exact.edge_marginals, reference.edge_marginals,
      atol=2e-12, rtol=2e-12)

    # Edge axes must retain the orientation supplied by edge_index.
    for batch_index in range(2):
      for edge_id in range(3):
        if not edge_mask[batch_index, edge_id]:
          self.assertEqual(
            exact.edge_marginals[batch_index, edge_id].sum().item(), 0.0)
          continue
        left, right = edges[batch_index, edge_id]
        torch.testing.assert_close(
          exact.edge_marginals[batch_index, edge_id].sum(dim=1),
          exact.node_marginals[batch_index, left],
          atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(
          exact.edge_marginals[batch_index, edge_id].sum(dim=0),
          exact.node_marginals[batch_index, right],
          atol=2e-12, rtol=2e-12)

  def test_large_additive_offsets_do_not_destabilise_marginals(self):
    node, pair, edges, edge_mask = _batched_problem()
    baseline = structured_utils.forest_sum_product(
      node, pair, edges, edge_mask=edge_mask)
    shifted_node = node + 10_000.0
    shifted_pair = pair - 3_000.0
    shifted = structured_utils.forest_sum_product(
      shifted_node, shifted_pair, edges, edge_mask=edge_mask)

    torch.testing.assert_close(
      baseline.node_marginals, shifted.node_marginals,
      atol=5e-11, rtol=5e-11)
    torch.testing.assert_close(
      baseline.edge_marginals, shifted.edge_marginals,
      atol=5e-11, rtol=5e-11)
    expected_shift = torch.tensor([
      4 * 10_000.0 - 3 * 3_000.0,
      4 * 10_000.0 - 2 * 3_000.0,
    ])
    torch.testing.assert_close(
      shifted.log_partition - baseline.log_partition,
      expected_shift, atol=2e-11, rtol=2e-15)

  def test_isolated_nodes_and_zero_edges(self):
    node = torch.tensor([[[0.2, -0.4], [1.3, -0.7]]])
    pair = torch.empty(1, 0, 2, 2)
    edges = torch.empty(0, 2, dtype=torch.long)
    exact = structured_utils.forest_sum_product(
      node, pair, edges, max_components=2)
    reference = structured_utils.enumerate_forest_distribution(
      node, pair, edges, max_components=2)
    torch.testing.assert_close(exact.log_partition, reference.log_partition)
    torch.testing.assert_close(exact.node_marginals, reference.node_marginals)
    self.assertEqual(exact.edge_marginals.shape, (1, 0, 2, 2))


class GradientIdentityTest(Float64TestCase):

  def test_log_partition_gradients_are_exact_marginals(self):
    generator = torch.Generator().manual_seed(991)
    node = torch.randn(
      1, 3, 2, generator=generator, dtype=torch.float64,
      requires_grad=True)
    pair = torch.randn(
      1, 2, 2, 2, generator=generator, dtype=torch.float64,
      requires_grad=True)
    edges = torch.tensor([[0, 2], [1, 2]])
    exact = structured_utils.forest_sum_product(node, pair, edges)
    node_gradient, pair_gradient = torch.autograd.grad(
      exact.log_partition.sum(), (node, pair))

    torch.testing.assert_close(
      node_gradient, exact.node_marginals, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
      pair_gradient, exact.edge_marginals, atol=2e-12, rtol=2e-12)

    reference = structured_utils.enumerate_forest_distribution(
      node, pair, edges)
    reference_node_gradient, reference_pair_gradient = torch.autograd.grad(
      reference.log_partition.sum(), (node, pair))
    torch.testing.assert_close(
      node_gradient, reference_node_gradient, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
      pair_gradient, reference_pair_gradient, atol=2e-12, rtol=2e-12)


class ClampAndTopologyTest(Float64TestCase):

  def test_clamped_observation_is_exact_in_marginals_and_samples(self):
    node = torch.tensor([[[0.2, -0.1], [0.4, 0.3], [-0.5, 0.8]]])
    pair = torch.tensor([[[[1.1, -0.7], [-0.3, 0.9]],
                          [[-0.4, 0.6], [0.7, -0.8]]]])
    edges = torch.tensor([[1, 0], [1, 2]])
    clamps = torch.tensor([[-1, 0, -1]])
    exact = structured_utils.forest_sum_product(
      node, pair, edges, clamped_states=clamps)
    reference = structured_utils.enumerate_forest_distribution(
      node, pair, edges, clamped_states=clamps)
    torch.testing.assert_close(
      exact.node_marginals, reference.node_marginals,
      atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
      exact.node_marginals[0, 1], torch.tensor([1.0, 0.0]))

    samples = structured_utils.sample_forest(
      node, pair, edges, 1000, clamped_states=clamps,
      generator=torch.Generator().manual_seed(8))
    self.assertTrue(bool((samples[:, :, 1] == 0).all().item()))

  def test_cycles_duplicates_and_component_overflow_are_rejected(self):
    node = torch.zeros(1, 4, 2)
    triangle_pair = torch.zeros(1, 3, 2, 2)
    with self.assertRaisesRegex(ValueError, 'cycle'):
      structured_utils.forest_sum_product(
        node, triangle_pair,
        torch.tensor([[0, 1], [1, 2], [2, 0]]))
    with self.assertRaisesRegex(ValueError, 'duplicate'):
      structured_utils.forest_sum_product(
        node, torch.zeros(1, 2, 2, 2),
        torch.tensor([[0, 1], [1, 0]]))
    with self.assertRaisesRegex(ValueError, 'exceeding cap'):
      structured_utils.forest_sum_product(
        node, torch.zeros(1, 1, 2, 2),
        torch.tensor([[0, 1]]), max_components=2)
    with self.assertRaisesRegex(ValueError, 'component.*exceeding cap'):
      structured_utils.forest_sum_product(
        node, torch.zeros(1, 2, 2, 2),
        torch.tensor([[0, 1], [1, 2]]), max_component_size=2)

  def test_pair_factors_must_be_strictly_positive(self):
    with self.assertRaisesRegex(ValueError, 'strictly positive'):
      structured_utils.positive_pair_factors_to_log(
        torch.tensor([[1.0, 0.0], [2.0, 3.0]]))
    with self.assertRaisesRegex(ValueError, 'strictly positive'):
      structured_utils.forest_sum_product(
        torch.zeros(1, 2, 2),
        torch.tensor([[[[0.0, -torch.inf], [0.0, 0.0]]]]),
        torch.tensor([[0, 1]]))


class JointSamplingTest(Float64TestCase):

  def test_ancestral_sample_frequencies_match_node_and_edge_marginals(self):
    node = torch.tensor([[[0.4, -0.3], [0.1, 0.2], [-0.2, 0.5]]])
    # Both edges strongly reward agreement, making an independent-marginal
    # sampler detectably wrong even when its one-node frequencies look right.
    pair = torch.tensor([[[[1.6, -1.1], [-0.9, 1.2]],
                          [[1.3, -0.8], [-1.0, 1.5]]]])
    edges = torch.tensor([[0, 1], [2, 1]])
    exact = structured_utils.forest_sum_product(node, pair, edges)
    samples = structured_utils.sample_forest(
      node, pair, edges, 60_000,
      generator=torch.Generator().manual_seed(41))[0]

    node_frequency = F.one_hot(samples, num_classes=2).double().mean(dim=0)
    torch.testing.assert_close(
      node_frequency, exact.node_marginals[0], atol=0.012, rtol=0.0)
    for edge_id, (left, right) in enumerate(edges.tolist()):
      left_one_hot = F.one_hot(samples[:, left], num_classes=2).double()
      right_one_hot = F.one_hot(samples[:, right], num_classes=2).double()
      edge_frequency = torch.einsum(
        'mi,mj->ij', left_one_hot, right_one_hot) / samples.shape[0]
      torch.testing.assert_close(
        edge_frequency, exact.edge_marginals[0, edge_id],
        atol=0.012, rtol=0.0)


class CandidateCompressionTest(Float64TestCase):

  def test_topk_plus_residual_preserves_unary_mass_and_forces_observation(self):
    logits = torch.tensor([[
      [0.0, 3.0, 2.0, -1.0, 1.0],
      [4.0, -2.0, 0.5, 1.0, 2.0],
    ]])
    observations = torch.tensor([[3, 4]])
    observed = torch.tensor([[True, True]])
    support = structured_utils.topk_residual_support(
      logits, top_k=2, forced_token_ids=observations,
      forced_mask=observed)

    torch.testing.assert_close(
      torch.logsumexp(support.node_log_potentials, dim=-1),
      torch.logsumexp(logits, dim=-1), atol=2e-12, rtol=2e-12)
    self.assertTrue(bool((support.token_ids == 3).any(dim=-1)[0, 0]))
    self.assertTrue(bool((support.token_ids == 4).any(dim=-1)[0, 1]))
    clamps = support.clamped_states(observations, observed)
    self.assertTrue(bool((clamps >= 0).all().item()))
    mapped = support.states_for_tokens(torch.tensor([[0, 1]]))
    self.assertTrue(bool((mapped == support.residual_index).all().item()))

    unforced = structured_utils.topk_residual_support(logits, top_k=2)
    with self.assertRaisesRegex(ValueError, 'force it into top-K'):
      unforced.clamped_states(observations, observed)

  def test_residual_pair_factors_are_exactly_neutral(self):
    raw = torch.arange(32, dtype=torch.float64).reshape(2, 4, 4) / 7.0
    neutral = structured_utils.neutralize_residual_pair_factors(raw)
    torch.testing.assert_close(neutral[..., -1, :], torch.zeros(2, 4))
    torch.testing.assert_close(neutral[..., :, -1], torch.zeros(2, 4))
    torch.testing.assert_close(neutral[..., :-1, :-1], raw[..., :-1, :-1])

  def test_low_rank_positive_factor_construction_is_not_log_rank(self):
    left = torch.tensor([
      [1.0, 0.3], [0.2, 1.1], [0.8, 0.7], [0.5, 1.3]])
    right = torch.tensor([
      [0.4, 1.2], [1.4, 0.2], [0.9, 0.8], [1.1, 0.6]])
    log_pair = structured_utils.low_rank_positive_pair_log_factors(
      left.log(), right.log())
    expected_positive_factor = left @ right.T
    torch.testing.assert_close(
      log_pair.exp(), expected_positive_factor,
      atol=2e-12, rtol=2e-12)
    self.assertLessEqual(
      int(torch.linalg.matrix_rank(log_pair.exp()).item()), 2)
    # The logarithm is a different object and is generally full rank.
    self.assertGreater(
      int(torch.linalg.matrix_rank(log_pair).item()), 2)


class LowRankProductionPathTest(Float64TestCase):

  def test_partitions_and_node_marginals_match_dense_and_enumeration(self):
    generator = torch.Generator().manual_seed(1701)
    node = torch.randn(2, 4, 4, generator=generator)
    left = torch.exp(0.6 * torch.randn(
      2, 3, 3, 2, generator=generator))
    right = torch.exp(0.6 * torch.randn(
      2, 3, 3, 2, generator=generator))
    edges = torch.tensor([
      [[0, 2], [2, 1], [1, 3]],
      [[3, 0], [1, 2], [-1, -1]],
    ])
    edge_mask = torch.tensor([
      [True, True, True],
      [True, True, False],
    ])
    state_mask = torch.ones_like(node, dtype=torch.bool)
    state_mask[0, 3, 3] = False
    state_mask[1, 0, 1] = False
    clamps = torch.tensor([
      [-1, 2, -1, -1],
      [-1, -1, 3, -1],
    ])

    low_rank = structured_utils.forest_sum_product_low_rank(
      node, left, right, edges, edge_mask=edge_mask,
      state_mask=state_mask, clamped_states=clamps,
      max_components=2, max_component_size=4)
    dense_factors = structured_utils.materialize_low_rank_pair_factors(
      left, right, edge_mask=edge_mask)
    dense = structured_utils.forest_sum_product(
      node, dense_factors.log(), edges, edge_mask=edge_mask,
      state_mask=state_mask, clamped_states=clamps,
      max_components=2, max_component_size=4)
    enumeration = structured_utils.enumerate_forest_distribution(
      node, dense_factors.log(), edges, edge_mask=edge_mask,
      state_mask=state_mask, clamped_states=clamps,
      max_components=2, max_component_size=4)

    torch.testing.assert_close(
      low_rank.log_partition, dense.log_partition,
      atol=4e-12, rtol=4e-12)
    torch.testing.assert_close(
      low_rank.node_marginals, dense.node_marginals,
      atol=4e-12, rtol=4e-12)
    torch.testing.assert_close(
      low_rank.log_partition, enumeration.log_partition,
      atol=4e-12, rtol=4e-12)
    torch.testing.assert_close(
      low_rank.node_marginals, enumeration.node_marginals,
      atol=4e-12, rtol=4e-12)

  def test_high_degree_cavities_match_dense_under_hard_constraints(self):
    generator = torch.Generator().manual_seed(707)
    node = torch.randn(1, 7, 4, generator=generator)
    left = torch.exp(0.5 * torch.randn(
      1, 6, 3, 4, generator=generator))
    right = torch.exp(0.5 * torch.randn(
      1, 6, 3, 4, generator=generator))
    # Alternate orientation around a degree-six center.
    edges = torch.tensor([
      [0, 1], [2, 0], [0, 3], [4, 0], [0, 5], [6, 0],
    ])
    state_mask = torch.ones_like(node, dtype=torch.bool)
    state_mask[0, 1, 0] = False
    state_mask[0, 4, 3] = False
    clamps = torch.tensor([[-1, -1, 2, -1, -1, 3, -1]])

    low_rank = structured_utils.forest_sum_product_low_rank(
      node, left, right, edges,
      state_mask=state_mask, clamped_states=clamps)
    dense = structured_utils.forest_sum_product(
      node,
      structured_utils.materialize_low_rank_pair_factors(
        left, right).log(),
      edges, state_mask=state_mask, clamped_states=clamps)
    torch.testing.assert_close(
      low_rank.log_partition, dense.log_partition,
      atol=8e-12, rtol=8e-12)
    torch.testing.assert_close(
      low_rank.node_marginals, dense.node_marginals,
      atol=8e-12, rtol=8e-12)

  def test_marginals_are_invariant_to_large_additive_child_offset(self):
    # A state-independent child offset must cancel from every marginal.  The
    # power-of-two shift keeps the integer unary differences exactly
    # representable while exposing cancellation in ``total - child`` schemes.
    node = torch.tensor([[
      [0.0, -1.0, 2.0, 1.0],
      [1.0, -2.0, 0.0, 2.0],
      [-1.0, 2.0, 1.0, 0.0],
      [2.0, 0.0, -2.0, 1.0],
    ]], dtype=torch.float64)
    left = torch.tensor([[
      [[1.3, 0.4], [0.2, 1.7], [0.9, 0.8]],
      [[0.7, 1.2], [1.6, 0.3], [0.5, 1.1]],
      [[1.4, 0.2], [0.6, 1.5], [1.0, 0.7]],
    ]], dtype=torch.float64)
    right = torch.tensor([[
      [[0.8, 1.1], [1.5, 0.2], [0.4, 1.3]],
      [[1.2, 0.5], [0.3, 1.4], [1.1, 0.9]],
      [[0.6, 1.6], [1.3, 0.4], [0.8, 1.0]],
    ]], dtype=torch.float64)
    edges = torch.tensor([[0, 1], [2, 0], [0, 3]])
    baseline = structured_utils.forest_sum_product_low_rank(
      node, left, right, edges)
    additive_offset = float(2 ** 50)
    shifted_node = node.clone()
    shifted_node[:, 1] += additive_offset
    shifted = structured_utils.forest_sum_product_low_rank(
      shifted_node, left, right, edges)

    torch.testing.assert_close(
      shifted.node_marginals, baseline.node_marginals,
      atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
      shifted.log_partition - additive_offset,
      baseline.log_partition, atol=0.3, rtol=0.0)

  def test_broom_schedule_stores_each_child_once(self):
    # At depth two this tree has 512 possible parents, one with 511 children.
    # Max-degree rectangular padding would allocate 512*511 slots there.
    length = 1024
    edges = torch.tensor(
      ([(0, node) for node in range(1, 513)]
       + [(1, node) for node in range(513, length)]),
      dtype=torch.long)
    edge_mask = torch.ones(edges.shape[0], dtype=torch.bool)
    canonical_edges, canonical_mask = structured_utils._canonical_topology(
      edges, edge_mask, batch_size=1, edge_count=edges.shape[0],
      device=torch.device('cpu'))
    topologies = structured_utils._build_topology(
      canonical_edges, canonical_mask, length, None, None)
    schedule = structured_utils._build_low_rank_level_schedule(
      topologies, canonical_edges, length, edges.shape[0])

    stored_child_slots = 0
    hypothetical_rectangular_slots = 0
    for buckets in schedule.child_degree_buckets[1:]:
      self.assertIsNotNone(buckets)
      stored_child_slots += sum(
        bucket.child_indices.numel() for bucket in buckets)
      parent_count = sum(bucket.parent_slots.numel() for bucket in buckets)
      maximum_degree = max(
        bucket.child_indices.shape[1] for bucket in buckets)
      hypothetical_rectangular_slots += parent_count * maximum_degree
    self.assertEqual(stored_child_slots, length - 1)
    self.assertGreater(hypothetical_rectangular_slots, 260_000)

  def test_partition_gradients_match_dense_materialisation(self):
    generator = torch.Generator().manual_seed(9917)
    node = torch.randn(
      1, 3, 3, generator=generator, requires_grad=True)
    left = torch.exp(0.4 * torch.randn(
      1, 3, 2, 3, generator=generator)).requires_grad_()
    right = torch.exp(0.4 * torch.randn(
      1, 3, 2, 3, generator=generator)).requires_grad_()
    edges = torch.tensor([[2, 0], [1, 2], [-1, -1]])
    edge_mask = torch.tensor([True, True, False])

    low_rank = structured_utils.forest_sum_product_low_rank(
      node, left, right, edges, edge_mask=edge_mask)
    low_rank_gradients = torch.autograd.grad(
      low_rank.log_partition.sum(), (node, left, right),
      retain_graph=True)
    dense_factors = structured_utils.materialize_low_rank_pair_factors(
      left, right, edge_mask=edge_mask)
    dense = structured_utils.forest_sum_product(
      node, dense_factors.log(), edges, edge_mask=edge_mask)
    dense_gradients = torch.autograd.grad(
      dense.log_partition.sum(), (node, left, right),
      retain_graph=True)
    enumeration = structured_utils.enumerate_forest_distribution(
      node, dense_factors.log(), edges, edge_mask=edge_mask)
    enumeration_gradients = torch.autograd.grad(
      enumeration.log_partition.sum(), (node, left, right))

    for actual, expected in zip(low_rank_gradients, dense_gradients):
      torch.testing.assert_close(
        actual, expected, atol=5e-12, rtol=5e-12)
    for actual, expected in zip(low_rank_gradients, enumeration_gradients):
      torch.testing.assert_close(
        actual, expected, atol=5e-12, rtol=5e-12)
    # Padded edges are neutral/ignored and receive exactly zero gradient.
    torch.testing.assert_close(
      low_rank_gradients[1][:, 2], torch.zeros_like(left[:, 2]))
    torch.testing.assert_close(
      low_rank_gradients[2][:, 2], torch.zeros_like(right[:, 2]))

  def test_depth_batched_mixed_forests_match_dense_gradients(self):
    generator = torch.Generator().manual_seed(5031)
    node = torch.randn(
      3, 8, 5, generator=generator, dtype=torch.float64,
      requires_grad=True)
    left = torch.exp(0.35 * torch.randn(
      3, 7, 4, 3, generator=generator, dtype=torch.float64)).requires_grad_()
    right = torch.exp(0.35 * torch.randn(
      3, 7, 4, 3, generator=generator, dtype=torch.float64)).requires_grad_()
    # Mix a deep chain, a high-degree tree, and a disconnected forest.  Edge
    # orientation alternates independently of the rooted traversal.
    edges = torch.tensor([
      [[1, 0], [1, 2], [3, 2], [3, 4], [5, 4], [5, 6], [7, 6]],
      [[1, 0], [0, 2], [3, 0], [0, 4], [5, 0], [0, 6], [7, 0]],
      [[0, 2], [3, 2], [1, 4], [4, 6], [7, 6], [-1, -1], [-1, -1]],
    ])
    edge_mask = torch.tensor([
      [True, True, True, True, True, True, True],
      [True, True, True, True, True, True, True],
      [True, True, True, True, True, False, False],
    ])
    state_mask = torch.ones_like(node, dtype=torch.bool)
    state_mask[0, 5, 1] = False
    state_mask[1, 0, 4] = False
    state_mask[2, 7, 0] = False
    clamps = torch.tensor([
      [-1, -1, 2, -1, -1, -1, -1, -1],
      [-1, 4, -1, -1, -1, -1, -1, -1],
      [-1, -1, -1, -1, 1, -1, -1, -1],
    ])

    low_rank = structured_utils.forest_sum_product_low_rank(
      node, left, right, edges, edge_mask=edge_mask,
      state_mask=state_mask, clamped_states=clamps)
    low_rank_gradients = torch.autograd.grad(
      low_rank.log_partition.sum(), (node, left, right), retain_graph=True)
    dense = structured_utils.forest_sum_product(
      node,
      structured_utils.materialize_low_rank_pair_factors(
        left, right, edge_mask=edge_mask).log(),
      edges, edge_mask=edge_mask, state_mask=state_mask,
      clamped_states=clamps)
    dense_gradients = torch.autograd.grad(
      dense.log_partition.sum(), (node, left, right))

    torch.testing.assert_close(
      low_rank.log_partition, dense.log_partition,
      atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
      low_rank.node_marginals, dense.node_marginals,
      atol=1e-11, rtol=1e-11)
    for actual, expected in zip(low_rank_gradients, dense_gradients):
      torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)

  def test_marginal_dependent_gradients_match_dense(self):
    generator = torch.Generator().manual_seed(7721)
    node = torch.randn(
      2, 5, 4, generator=generator, dtype=torch.float64,
      requires_grad=True)
    left = torch.exp(0.4 * torch.randn(
      2, 4, 3, 3, generator=generator,
      dtype=torch.float64)).requires_grad_()
    right = torch.exp(0.4 * torch.randn(
      2, 4, 3, 3, generator=generator,
      dtype=torch.float64)).requires_grad_()
    edges = torch.tensor([
      [[1, 0], [0, 2], [3, 0], [3, 4]],
      [[0, 2], [3, 2], [1, 4], [-1, -1]],
    ])
    edge_mask = torch.tensor([
      [True, True, True, True],
      [True, True, True, False],
    ])
    weights = torch.randn(
      2, 5, 4, generator=generator, dtype=torch.float64)

    low_rank = structured_utils.forest_sum_product_low_rank(
      node, left, right, edges, edge_mask=edge_mask)
    low_rank_loss = (low_rank.node_marginals * weights).sum()
    low_rank_gradients = torch.autograd.grad(
      low_rank_loss, (node, left, right), retain_graph=True)
    dense = structured_utils.forest_sum_product(
      node,
      structured_utils.materialize_low_rank_pair_factors(
        left, right, edge_mask=edge_mask).log(),
      edges, edge_mask=edge_mask)
    dense_loss = (dense.node_marginals * weights).sum()
    dense_gradients = torch.autograd.grad(
      dense_loss, (node, left, right))

    torch.testing.assert_close(
      low_rank_loss, dense_loss, atol=2e-11, rtol=2e-11)
    for actual, expected in zip(low_rank_gradients, dense_gradients):
      torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
  def test_depth_batched_cuda_forward_and_gradients_are_repeatable(self):
    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(992)
    base_node = torch.randn(
      4, 9, 6, generator=generator, device=device)
    base_left = torch.rand(
      4, 8, 5, 4, generator=generator, device=device) + 0.2
    base_right = torch.rand(
      4, 8, 5, 4, generator=generator, device=device) + 0.2
    edges = torch.tensor([
      [0, 1], [0, 2], [3, 0], [3, 4],
      [5, 4], [5, 6], [7, 5], [7, 8],
    ], device=device)
    weights = torch.randn(
      4, 9, 6, generator=generator, device=device)

    deterministic_was_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_was_enabled = (
      torch.is_deterministic_algorithms_warn_only_enabled())
    records = []
    try:
      torch.use_deterministic_algorithms(True)
      for _ in range(2):
        node = base_node.detach().clone().requires_grad_()
        left = base_left.detach().clone().requires_grad_()
        right = base_right.detach().clone().requires_grad_()
        result = structured_utils.forest_sum_product_low_rank(
          node, left, right, edges)
        loss = (
          result.log_partition.sum()
          + (result.node_marginals * weights).sum())
        gradients = torch.autograd.grad(loss, (node, left, right))
        torch.cuda.synchronize(device)
        records.append((
          result.log_partition.detach(),
          result.node_log_marginals.detach(),
          *(gradient.detach() for gradient in gradients)))
    finally:
      torch.use_deterministic_algorithms(
        deterministic_was_enabled, warn_only=warn_only_was_enabled)

    for first, second in zip(*records):
      self.assertTrue(torch.equal(first, second))

  def test_low_rank_joint_sampling_matches_dense_edge_marginals(self):
    node = torch.tensor([[
      [0.3, -0.2, -0.4],
      [-0.1, 0.5, -0.3],
      [0.2, -0.4, 0.1],
    ]])
    left = torch.tensor([[[
      [1.4, 0.2], [0.3, 1.2]],
      [[0.7, 1.1], [1.3, 0.2]],
    ]])
    right = torch.tensor([[[
      [1.1, 0.3], [0.2, 1.5]],
      [[1.4, 0.1], [0.2, 1.2]],
    ]])
    edges = torch.tensor([[0, 1], [2, 1]])
    dense_factors = structured_utils.materialize_low_rank_pair_factors(
      left, right)
    dense = structured_utils.forest_sum_product(
      node, dense_factors.log(), edges)
    samples = structured_utils.sample_forest_low_rank(
      node, left, right, edges, 70_000,
      generator=torch.Generator().manual_seed(443))[0]

    node_frequency = F.one_hot(samples, num_classes=3).double().mean(dim=0)
    torch.testing.assert_close(
      node_frequency, dense.node_marginals[0], atol=0.012, rtol=0.0)
    for edge_id, (first, second) in enumerate(edges.tolist()):
      first_one_hot = F.one_hot(
        samples[:, first], num_classes=3).double()
      second_one_hot = F.one_hot(
        samples[:, second], num_classes=3).double()
      frequency = torch.einsum(
        'mi,mj->ij', first_one_hot, second_one_hot) / samples.shape[0]
      torch.testing.assert_close(
        frequency, dense.edge_marginals[0, edge_id],
        atol=0.012, rtol=0.0)

    clamps = torch.tensor([[-1, 2, -1]])
    clamped_samples = structured_utils.sample_forest_low_rank(
      node, left, right, edges, 1000, clamped_states=clamps,
      generator=torch.Generator().manual_seed(444))
    self.assertTrue(bool((clamped_samples[:, :, 1] == 2).all().item()))

  def test_log_domain_path_handles_products_dense_float_cannot_materialise(self):
    node = torch.tensor([[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.1]]])
    left = torch.tensor([[[
      [1e200, 1e-200], [1e-180, 1e180],
    ]]])
    right = torch.tensor([[[
      [1e200, 1e-180], [1e-190, 1e190],
    ]]])
    edges = torch.tensor([[1, 0]])
    low_rank = structured_utils.forest_sum_product_low_rank(
      node, left, right, edges)

    explicit_log_factors = (
      structured_utils.low_rank_positive_pair_log_factors(
        left.log(), right.log()))
    stable_dense_log_factors = F.pad(
      explicit_log_factors, (0, 1, 0, 1), value=0.0)
    stable_dense = structured_utils.forest_sum_product(
      node, stable_dense_log_factors, edges)
    self.assertTrue(bool(torch.isfinite(low_rank.log_partition).all()))
    torch.testing.assert_close(
      low_rank.log_partition, stable_dense.log_partition,
      atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
      low_rank.node_marginals, stable_dense.node_marginals,
      atol=2e-12, rtol=2e-12)
    self.assertTrue(bool(torch.isinf(
      structured_utils.materialize_low_rank_pair_factors(
        left, right)[..., :-1, :-1]).any()))

  def test_shapes_and_positive_endpoint_contract_are_enforced(self):
    node = torch.zeros(1, 2, 3)
    left = torch.ones(1, 1, 2, 2)
    right = left.clone()
    edges = torch.tensor([[0, 1]])
    with self.assertRaisesRegex(ValueError, 'explicit endpoint states'):
      structured_utils.forest_sum_product_low_rank(
        torch.zeros(1, 2, 4), left, right, edges)
    left[0, 0, 0, 0] = 0.0
    with self.assertRaisesRegex(ValueError, 'strictly positive'):
      structured_utils.forest_sum_product_low_rank(
        node, left, right, edges)


class SeparableReverseMixtureTest(Float64TestCase):

  def test_reverse_marginals_and_samples_preserve_latent_mixture(self):
    latent = torch.tensor([[[0.75, 0.25], [0.35, 0.65]]])
    kernel_probabilities = torch.tensor([[
      [[0.90, 0.10, 0.00], [0.15, 0.25, 0.60]],
      [[0.70, 0.20, 0.10], [0.05, 0.15, 0.80]],
    ]])
    kernel_log = torch.where(
      kernel_probabilities > 0,
      kernel_probabilities.log(),
      torch.full_like(kernel_probabilities, -torch.inf))
    expected = torch.einsum('bnk,bnkq->bnq', latent, kernel_probabilities)
    actual = structured_utils.separable_reverse_mixture_marginals(
      latent, kernel_log)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    sample_count = 50_000
    latent_samples = torch.empty(1, sample_count, 2, dtype=torch.long)
    generator = torch.Generator().manual_seed(101)
    for node in range(2):
      latent_samples[0, :, node] = torch.multinomial(
        latent[0, node], sample_count, replacement=True,
        generator=generator)
    outputs = structured_utils.sample_separable_reverse_mixture(
      latent_samples, kernel_log, generator=generator)
    frequency = F.one_hot(outputs, num_classes=3).double().mean(dim=1)
    torch.testing.assert_close(frequency, expected, atol=0.012, rtol=0.0)


if __name__ == '__main__':
  unittest.main()
