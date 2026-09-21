"""Focused ordering tests for the restored low-rank forest level schedule.

Written for the CCF walkthrough on 2026-09-20. Expected relationships come
from a small independent BFS over raw edges, not the scheduler's own parent
or bucket records. Only PyTorch, unittest and structured_utils are required.

These tests cover metadata and the gathers/reductions that consume it. They
do not certify the complete structured NLL or fix the deferred residual-mass
backward issue in the model head.
"""

from collections import deque
from dataclasses import fields, is_dataclass
import random
import unittest

import torch

import structured_utils as su


TEACHING_EDGES = [(0, 1), (0, 4), (1, 2), (1, 3), (4, 5)]


def _pack(examples, num_nodes, padding=0, device='cpu'):
  """None entries and trailing padding are invalid slots, not real edges."""
  width = max(map(len, examples)) + padding
  edges = torch.empty(len(examples), width, 2, dtype=torch.long, device=device)
  edges[..., 0] = -51
  edges[..., 1] = num_nodes + 17
  mask = torch.zeros(len(examples), width, dtype=torch.bool, device=device)
  for batch, example in enumerate(examples):
    for slot, edge in enumerate(example):
      if edge is not None:
        edges[batch, slot] = torch.tensor(edge, device=device)
        mask[batch, slot] = True
  return edges, mask


def _reference_bfs(edges, mask, num_nodes):
  """Independent graph oracle; never read production topology metadata."""
  raw_edges, raw_mask = edges.cpu().tolist(), mask.cpu().tolist()
  width = edges.shape[1]
  nodes, roots = {}, []
  for batch, example in enumerate(raw_edges):
    neighbors = [[] for _ in range(num_nodes)]
    for slot, (left, right) in enumerate(example):
      if raw_mask[batch][slot]:
        neighbors[left].append((right, slot))
        neighbors[right].append((left, slot))
    seen, batch_roots = set(), []
    for root in range(num_nodes):
      if root in seen:
        continue
      root_id = batch * num_nodes + root
      batch_roots.append(root_id)
      nodes[root_id] = {'parent': -1, 'edge': -1, 'depth': 0}
      seen.add(root)
      queue = deque([root])
      while queue:
        node = queue.popleft()
        node_id = batch * num_nodes + node
        for neighbor, slot in neighbors[node]:
          if neighbor not in seen:
            child_id = batch * num_nodes + neighbor
            nodes[child_id] = {
                'parent': node_id, 'edge': batch * width + slot,
                'depth': nodes[node_id]['depth'] + 1}
            seen.add(neighbor)
            queue.append(neighbor)
    roots.append(batch_roots)
  return nodes, roots


def _schedule(edges, mask, num_nodes):
  topology = su._build_topology(edges, mask, num_nodes, None, None)
  return su._build_low_rank_level_schedule(
      topology, edges, num_nodes, edges.shape[1])


class LowRankLevelScheduleOrderingTest(unittest.TestCase):

  def _assert_schedule(self, edges, mask, num_nodes, schedule=None,
                       check_gradients=False):
    """Check identities, not just shapes or sorted permutations."""
    schedule = schedule or _schedule(edges, mask, num_nodes)
    reference, roots = _reference_bfs(edges, mask, num_nodes)
    batch_size, edge_count = edges.shape[:2]
    levels = [ids.tolist() for ids in schedule.node_ids]
    self.assertEqual(len(levels), 1 + max(n['depth'] for n in reference.values()))
    self.assertEqual(sorted(n for level in levels for n in level),
                     list(range(batch_size * num_nodes)))

    # All index tensors must be usable on the same device as the input.
    def inspect(value):
      if torch.is_tensor(value):
        self.assertEqual(value.device, edges.device)
        self.assertIn(value.dtype, (torch.long, torch.bool))
      elif is_dataclass(value):
        for field in fields(value):
          inspect(getattr(value, field.name))
      elif isinstance(value, (tuple, list)):
        for item in value:
          inspect(item)
    inspect(schedule)
    for name in ('parent_slots', 'edge_ids', 'child_is_left',
                 'child_degree_buckets', 'parent_inverse_order',
                 'child_inverse_order'):
      collection = getattr(schedule, name)
      self.assertEqual(len(collection), len(levels), name)
      self.assertIsNone(collection[0], name)

    raw_edges = edges.reshape(-1, 2).tolist()
    left = torch.arange(batch_size * edge_count * 6, dtype=torch.float64,
                        device=edges.device).reshape(-1, 2, 3) + 100
    right = -left - 7  # Distinct endpoint tables expose role swaps.
    for depth, node_ids in enumerate(levels):
      self.assertEqual(set(node_ids),
                       {n for n, r in reference.items() if r['depth'] == depth})
      if depth == 0:
        continue
      parents = levels[depth - 1]
      parent_slots = schedule.parent_slots[depth].tolist()
      self.assertEqual(len(parent_slots), len(node_ids))
      self.assertEqual(schedule.edge_ids[depth].numel(), len(node_ids))
      source, target = su._oriented_low_rank_factors(
          left, right, schedule.edge_ids[depth], schedule.child_is_left[depth])
      for child_slot, child_id in enumerate(node_ids):
        expected = reference[child_id]
        self.assertEqual(parents[parent_slots[child_slot]], expected['parent'])
        edge_id = int(schedule.edge_ids[depth][child_slot])
        self.assertEqual(edge_id, expected['edge'])
        self.assertTrue(bool(mask.reshape(-1)[edge_id]))
        self.assertEqual(edge_id // edge_count, child_id // num_nodes)
        child_is_left = raw_edges[edge_id][0] == child_id % num_nodes
        self.assertEqual(bool(schedule.child_is_left[depth][child_slot]),
                         child_is_left)
        torch.testing.assert_close(source[child_slot],
                                   (left if child_is_left else right)[edge_id])
        torch.testing.assert_close(target[child_slot],
                                   (right if child_is_left else left)[edge_id])

      buckets = schedule.child_degree_buckets[depth]
      parent_order, child_order, degrees = [], [], []
      for bucket in buckets:
        parent_count, degree = bucket.child_indices.shape
        self.assertEqual(tuple(bucket.parent_slots.shape), (parent_count,))
        degrees.append(degree)
        for parent_slot, child_slots in zip(bucket.parent_slots.tolist(),
                                             bucket.child_indices.tolist()):
          parent_order.append(parent_slot)
          child_order.extend(child_slots)
          expected_children = {i for i, n in enumerate(node_ids)
                               if reference[n]['parent'] == parents[parent_slot]}
          self.assertEqual(set(child_slots), expected_children)
          self.assertEqual(len(child_slots), len(expected_children))
      self.assertEqual(degrees, sorted(set(degrees)))
      self.assertEqual(sorted(parent_order), list(range(len(parents))))
      self.assertEqual(sorted(child_order), list(range(len(node_ids))))
      self.assertEqual([parent_order[i] for i in
                        schedule.parent_inverse_order[depth].tolist()],
                       list(range(len(parents))))
      self.assertEqual([child_order[i] for i in
                        schedule.child_inverse_order[depth].tolist()],
                       list(range(len(node_ids))))

      # Use distinct values for every node/state. References group by the
      # independently recovered parent ID, not by schedule.parent_slots.
      messages = torch.tensor([[n * 11 + s + 1 for s in range(3)]
                               for n in node_ids], dtype=torch.float64,
                              device=edges.device, requires_grad=check_gradients)
      totals = su._bucketed_group_sum(
          messages, buckets, schedule.parent_inverse_order[depth])
      siblings = su._exclusive_sibling_sums(
          messages, buckets, schedule.child_inverse_order[depth])
      children_of = {p: [i for i, n in enumerate(node_ids)
                         if reference[n]['parent'] == p] for p in parents}
      def direct_sum(indices):
        return sum((messages[i].detach() for i in indices),
                   torch.zeros(3, dtype=torch.float64, device=edges.device))
      expected_totals = torch.stack([direct_sum(children_of[p]) for p in parents])
      expected_siblings = torch.stack([
          direct_sum([j for j in children_of[reference[n]['parent']] if j != i])
          for i, n in enumerate(node_ids)])
      torch.testing.assert_close(totals, expected_totals, atol=0, rtol=0)
      torch.testing.assert_close(siblings, expected_siblings, atol=0, rtol=0)
      if check_gradients:
        # Distinct weights expose gradients assigned to the wrong parent or
        # child, even when an unweighted sum would conceal the permutation.
        parent_weights = torch.tensor([[p * 5 + s + 2 for s in range(3)]
                                       for p in parents], dtype=torch.float64,
                                      device=edges.device)
        child_weights = torch.tensor([[n * 7 + s + 3 for s in range(3)]
                                      for n in node_ids], dtype=torch.float64,
                                     device=edges.device)
        loss = (totals * parent_weights).sum() + (siblings * child_weights).sum()
        actual_grad = torch.autograd.grad(loss, messages)[0]
        expected_grad = []
        for i, node in enumerate(node_ids):
          parent = reference[node]['parent']
          g = parent_weights[parents.index(parent)].clone()
          for other in children_of[parent]:
            if other != i:
              g += child_weights[other]
          expected_grad.append(g)
        torch.testing.assert_close(actual_grad, torch.stack(expected_grad),
                                   atol=0, rtol=0)

    # Tag every node uniquely: restoring depth order must recover all B*L
    # nodes, not just a permutation of labels within one example.
    tags = torch.arange(batch_size * num_nodes * 3, dtype=torch.float64,
                        device=edges.device).reshape(-1, 3).requires_grad_(
                            check_gradients)
    reordered = torch.cat([tags.index_select(0, ids) for ids in schedule.node_ids])
    restored = reordered.index_select(0, schedule.inverse_node_order)
    torch.testing.assert_close(restored, tags, atol=0, rtol=0)
    if check_gradients:
      weights = tags.detach() + 1
      grad = torch.autograd.grad((restored * weights).sum(), tags)[0]
      torch.testing.assert_close(grad, weights, atol=0, rtol=0)

    # Sentinel root entries must contribute zero, not another example's root.
    root_ids = levels[0]
    sentinel = len(root_ids)
    max_roots = max(map(len, roots))
    self.assertEqual(tuple(schedule.roots_by_batch.shape), (batch_size, max_roots))
    for batch, own_roots in enumerate(roots):
      wanted = [root_ids.index(root) for root in own_roots]
      wanted += [sentinel] * (max_roots - len(wanted))
      self.assertEqual(schedule.roots_by_batch[batch].tolist(), wanted)
    root_values = torch.tensor([n + 1.0 for n in root_ids], dtype=torch.float64,
                               device=edges.device, requires_grad=check_gradients)
    padded = torch.cat((root_values, root_values.new_zeros(1)))
    batch_sums = padded[schedule.roots_by_batch].sum(dim=1)
    expected_sums = root_values.new_tensor([sum(n + 1 for n in r) for r in roots])
    torch.testing.assert_close(batch_sums, expected_sums, atol=0, rtol=0)
    if check_gradients:
      weights = root_values.new_tensor([b + 2 for b in range(batch_size)])
      grad = torch.autograd.grad((batch_sums * weights).sum(), root_values)[0]
      expected = root_values.new_tensor([n // num_nodes + 2 for n in root_ids])
      torch.testing.assert_close(grad, expected, atol=0, rtol=0)
    return schedule

  def test_six_node_walkthrough_has_the_exact_explained_arrays(self):
    edges, mask = _pack([TEACHING_EDGES], 6)
    s = self._assert_schedule(edges, mask, 6)
    as_lists = lambda seq: [None if t is None else t.tolist() for t in seq]
    self.assertEqual(as_lists(s.node_ids), [[0], [1, 4], [2, 3, 5]])
    self.assertEqual(as_lists(s.parent_slots), [None, [0, 0], [0, 0, 1]])
    self.assertEqual(as_lists(s.edge_ids), [None, [0, 1], [2, 3, 4]])
    self.assertEqual(as_lists(s.parent_inverse_order), [None, [0], [1, 0]])
    self.assertEqual(as_lists(s.child_inverse_order), [None, [0, 1], [1, 2, 0]])
    self.assertEqual(s.inverse_node_order.tolist(), [0, 1, 3, 4, 2, 5])
    self.assertEqual([(b.parent_slots.tolist(), b.child_indices.tolist())
                      for b in s.child_degree_buckets[2]],
                     [([1], [[2]]), ([0], [[0, 1]])])

  def test_non_self_inverse_permutation_and_zero_child_parents(self):
    # Root children 1,2,3,4 have respectively 2,3,1,0 children.
    # This produces >2-parent permutations; a simple swap alone is weak.
    tree = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6),
            (2, 7), (2, 8), (2, 9), (3, 10)]
    edges, mask = _pack([tree], 11)
    s = self._assert_schedule(edges, mask, 11, check_gradients=True)
    parent_order = [p for b in s.child_degree_buckets[2]
                    for p in b.parent_slots.tolist()]
    self.assertEqual(parent_order, [3, 2, 0, 1])
    self.assertEqual(s.parent_inverse_order[2].tolist(), [2, 3, 1, 0])
    self.assertNotEqual(parent_order, s.parent_inverse_order[2].tolist())
    self.assertEqual(tuple(s.child_degree_buckets[2][0].child_indices.shape),
                     (1, 0))

  def test_one_bucket_contains_several_parents_with_aligned_children(self):
    tree = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5),
            (2, 6), (2, 7), (3, 8), (3, 9)]
    edges, mask = _pack([tree], 10)
    s = self._assert_schedule(edges, mask, 10, check_gradients=True)
    bucket, = s.child_degree_buckets[2]
    self.assertEqual(bucket.parent_slots.tolist(), [0, 1, 2])
    self.assertEqual(bucket.child_indices.tolist(), [[0, 1], [2, 3], [4, 5]])

  def test_batches_do_not_mix_and_root_padding_is_neutral(self):
    examples = [TEACHING_EDGES, [(0, 2), None, (3, 4)], []]
    edges, mask = _pack(examples, 6, padding=2)
    s = self._assert_schedule(edges, mask, 6, check_gradients=True)
    sentinel = len(s.node_ids[0])
    self.assertEqual(int((s.roots_by_batch[0] == sentinel).sum()), 5)
    self.assertEqual(int((s.roots_by_batch[1] == sentinel).sum()), 2)
    self.assertEqual(int((s.roots_by_batch[2] == sentinel).sum()), 0)

  def test_shuffled_edges_reversed_roles_and_masked_holes(self):
    # Root 0 reaches 5, then smaller children: parent != lower endpoint.
    examples = [[None, (2, 5), (5, 0), None, (5, 1), (3, 2)],
                [(0, 5), None, (5, 2), (1, 5), None, (2, 3)]]
    edges, mask = _pack(examples, 6, padding=1)
    s = self._assert_schedule(edges, mask, 6, check_gradients=True)
    flags = torch.cat(s.child_is_left[1:]).tolist()
    self.assertIn(True, flags)
    self.assertIn(False, flags)

  def test_empty_edge_axis_and_single_node_examples(self):
    for count in (1, 5):
      with self.subTest(num_nodes=count):
        edges, mask = _pack([[], []], count)
        s = self._assert_schedule(edges, mask, count, check_gradients=True)
        self.assertEqual(len(s.node_ids), 1)
        self.assertEqual(s.roots_by_batch.numel(), 2 * count)

  def test_chain_star_and_broom(self):
    size = 18
    examples = [
        [(i, i + 1) for i in range(size - 1)],
        [(size - 1, i) for i in range(size - 1)],
        [(0, i) for i in range(1, 9)] + [(1, i) for i in range(9, size)],
    ]
    edges, mask = _pack(examples, size, padding=1)
    self._assert_schedule(edges, mask, size, check_gradients=True)

  def test_random_forests_against_independent_graph_oracle(self):
    # Fixed local RNG: no external property-testing dependency or global RNG
    # changes. Shuffle labels, edge slots, directions and add masked holes.
    for seed in range(24):
      with self.subTest(seed=seed):
        rng, count, examples = random.Random(seed), 20, []
        for _ in range(3):
          labels = list(range(count))
          rng.shuffle(labels)
          tree = []
          for i in range(1, count):
            if rng.random() < 0.8:
              edge = (labels[i], labels[rng.randrange(i)])
              tree.append(edge if rng.random() < 0.5 else edge[::-1])
          rng.shuffle(tree)
          for _ in range(3):
            tree.insert(rng.randrange(len(tree) + 1), None)
          examples.append(tree)
        edges, mask = _pack(examples, count, padding=2)
        self._assert_schedule(edges, mask, count)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
  def test_cuda_metadata_and_routing_gradients(self):
    edges, mask = _pack([TEACHING_EDGES, [(5, 0), (2, 5), None, (1, 5)]],
                       6, padding=1, device='cuda')
    self._assert_schedule(edges, mask, 6, check_gradients=True)


if __name__ == '__main__':
  unittest.main()
