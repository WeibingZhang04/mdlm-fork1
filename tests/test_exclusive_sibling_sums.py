"""Focused sibling-sum checks for the CCF walkthrough.

The reference directly sums other children with the same parent. It does not
use prefix/suffix scans or subtract a child's value from its group total.
Only tests are added; this does not certify the complete training objective.
Run from the restored repository with python -m pytest or unittest discovery.
"""
import random
import unittest

import torch

import structured_utils as su


def _fixture(groups, device='cpu'):
  """Build test buckets from explicit groups of original child indices."""
  children = [c for group in groups for c in group]
  assert sorted(children) == list(range(len(children)))
  parents = [None] * len(children)
  for parent, group in enumerate(groups):
    for child in group:
      parents[child] = parent
  buckets, bucket_order = [], []
  for degree in sorted({len(group) for group in groups}):
    selected = [p for p, group in enumerate(groups) if len(group) == degree]
    indices = [groups[p] for p in selected]
    buckets.append(su._ChildDegreeBucket(
        parent_slots=torch.tensor(selected, dtype=torch.long, device=device),
        child_indices=torch.tensor(indices, dtype=torch.long, device=device)
        .reshape(len(selected), degree)))
    bucket_order.extend(c for group in indices for c in group)
  inverse = torch.tensor([bucket_order.index(c) for c in range(len(children))],
                         dtype=torch.long, device=device)
  return tuple(buckets), inverse, parents


def _reference(values, parents):
  results = []
  for child, parent in enumerate(parents):
    others = [values[j] for j, p in enumerate(parents)
              if p == parent and j != child]
    results.append(torch.stack(others).sum(0) if others
                   else torch.zeros_like(values[child]))
  return torch.stack(results)


class ExclusiveSiblingSumsTest(unittest.TestCase):
  def _check(self, values, groups):
    buckets, inverse, parents = _fixture(groups, values.device)
    before = values.detach().clone()
    actual = su._exclusive_sibling_sums(values, buckets, inverse)
    torch.testing.assert_close(actual, _reference(values, parents), atol=0, rtol=0)
    torch.testing.assert_close(values, before, atol=0, rtol=0)
    self.assertEqual(actual.shape, values.shape)
    self.assertEqual(actual.dtype, values.dtype)
    self.assertEqual(actual.device, values.device)
    return actual, parents

  def test_walkthrough_numbers_and_child_inverse(self):
    # Child indices 0,1,2 correspond to nodes 2,3,5. Parents are nodes 1,4.
    values = torch.tensor([[10., 11.], [20., 21.], [70., 71.]])
    groups = [[0, 1], [2]]
    buckets, inverse, _ = _fixture(groups)
    self.assertEqual(inverse.tolist(), [1, 2, 0])
    actual, _ = self._check(values, groups)
    torch.testing.assert_close(actual, torch.tensor([[20., 21.], [10., 11.],
                                                    [0., 0.]]))

  def test_mixed_degrees_shuffled_children_and_multiple_parents_per_bucket(self):
    groups = [[], [4, 0, 6], [2], [5, 1, 3], []]
    values = torch.arange(7 * 4, dtype=torch.float64).reshape(7, 4)
    self._check(values, groups)

  def test_single_children_have_zero_sibling_contribution(self):
    values = torch.tensor([[7., -3.], [19., -8.], [-2., 0.]])
    actual, _ = self._check(values, [[], [2], [0], [], [1]])
    torch.testing.assert_close(actual, torch.zeros_like(values))
    # This standalone result can be constant (requires_grad=False). In the
    # caller the parent_base remains differentiable; no fake graph is needed.

  def test_shapes_and_floating_dtypes(self):
    groups = [[2, 0], [], [4], [1, 3]]
    for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
      for trailing_shape in ((), (3,), (2, 3)):
        with self.subTest(dtype=dtype, trailing_shape=trailing_shape):
          size = 1
          for dimension in trailing_shape:
            size *= dimension
          values = torch.arange(5 * size, dtype=dtype).reshape(5, *trailing_shape)
          self._check(values, groups)

  def test_random_groups_against_direct_sibling_reference(self):
    for seed in range(30):
      with self.subTest(seed=seed):
        rng = random.Random(seed)
        count = rng.randrange(1, 45)
        groups = [[] for _ in range(rng.randrange(1, 12))]
        shuffled = list(range(count))
        rng.shuffle(shuffled)
        for child in shuffled:
          groups[rng.randrange(len(groups))].append(child)
        # Small integer-valued floats make exact comparison legitimate despite
        # differing summation orders in the reference and production scans.
        values = torch.tensor([[rng.randrange(-30, 31) for _ in range(4)]
                               for _ in range(count)], dtype=torch.float64)
        self._check(values, groups)

  def test_jacobian_excludes_self_and_other_parents(self):
    groups = [[4, 0, 2], [], [3], [1, 5]]
    buckets, inverse, parents = _fixture(groups)
    values = torch.arange(12., dtype=torch.float64).reshape(6, 2).requires_grad_()
    jacobian = torch.autograd.functional.jacobian(
        lambda x: su._exclusive_sibling_sums(x, buckets, inverse), values)
    expected = torch.zeros_like(jacobian)
    for receiver in range(6):
      for contributor in range(6):
        if receiver != contributor and parents[receiver] == parents[contributor]:
          for state in range(2):
            expected[receiver, state, contributor, state] = 1
    torch.testing.assert_close(jacobian, expected, atol=0, rtol=0)

  def test_finite_difference_gradcheck(self):
    buckets, inverse, _ = _fixture([[3, 0, 2], [1], []])
    values = torch.tensor([[0.3, -1.1], [2.1, 0.9], [-0.7, 1.8], [0.2, -0.4]],
                          dtype=torch.float64, requires_grad=True)
    self.assertTrue(torch.autograd.gradcheck(
        lambda x: su._exclusive_sibling_sums(x, buckets, inverse), (values,)))

  def test_large_self_message_does_not_destroy_small_sibling_sum(self):
    for dtype in (torch.float32, torch.float64):
      with self.subTest(dtype=dtype):
        values = torch.tensor([[-1e20], [-2.], [-3.]], dtype=dtype)
        actual, _ = self._check(values, [[0, 1, 2]])
        self.assertEqual(actual[0, 0].item(), -5.)
        naive = values.sum(dim=0) - values[0]
        self.assertNotEqual(naive.item(), -5.)

  def test_negative_infinity_is_excluded_without_nan(self):
    values = torch.tensor([[-torch.inf, 2.], [3., -torch.inf], [5., 7.]],
                          dtype=torch.float64)
    actual, _ = self._check(values, [[0, 1, 2]])
    self.assertFalse(bool(torch.isnan(actual).any()))
    torch.testing.assert_close(actual, torch.tensor(
        [[8., -torch.inf], [-torch.inf, 9.], [-torch.inf, -torch.inf]],
        dtype=torch.float64))
    self._check(torch.full((3, 2), -torch.inf), [[0, 1, 2]])

  def test_real_schedule_keeps_batch_items_and_children_separate(self):
    tree = [(0, 1), (0, 4), (1, 2), (1, 3), (4, 5)]
    edges = torch.tensor([tree, [(b, a) for a, b in reversed(tree)]])
    mask = torch.ones((2, 5), dtype=torch.bool)
    topologies = su._build_topology(edges, mask, 6, None, None)
    schedule = su._build_low_rank_level_schedule(topologies, edges, 6, 5)
    children = schedule.node_ids[2].tolist()
    # Independent, explicit graph parents; no scheduler metadata in oracle.
    parent_of = {2: 1, 3: 1, 5: 4, 8: 7, 9: 7, 11: 10}
    parents = [parent_of[c] for c in children]
    values = torch.tensor([[c * 10., c * 10. + 1] for c in children])
    actual = su._exclusive_sibling_sums(
        values, schedule.child_degree_buckets[2], schedule.child_inverse_order[2])
    torch.testing.assert_close(actual, _reference(values, parents), atol=0, rtol=0)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
  def test_cuda_strict_determinism_values_and_gradients(self):
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
      torch.use_deterministic_algorithms(True)
      groups = [[], [4, 0, 6], [2], [5, 1, 3], []]
      previous = None
      for _ in range(2):
        values = torch.arange(21., dtype=torch.float64, device='cuda')
        values = values.reshape(7, 3).requires_grad_()
        actual, parents = self._check(values, groups)
        weights = torch.arange(21., dtype=torch.float64, device='cuda').reshape(7, 3)
        weights = weights + 1
        gradient = torch.autograd.grad((actual * weights).sum(), values)[0]
        # A contributor receives the loss weights of its OTHER siblings.
        expected_gradient = _reference(weights, parents)
        torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
        if previous is not None:
          self.assertTrue(torch.equal(actual, previous[0]))
          self.assertTrue(torch.equal(gradient, previous[1]))
        previous = (actual.detach().clone(), gradient.detach().clone())
    finally:
      torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


if __name__ == '__main__':
  unittest.main()
