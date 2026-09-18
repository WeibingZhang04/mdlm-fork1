"""Experimental level-parallel ancestral arithmetic with legacy-order noise.

No production defaults changed. Uses v3's PyTorch-attributed exponential race,
with identical separate per-node RNG calls before batching deterministic work.
"""
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from unittest import mock

import torch

import structured_utils as utils
from scripts.audit_ccf_sampling_v3 import experiment as v3_experiment

BEST_V3 = 'unchecked_batched_roots_kruskal'
MODES = ('root_draws', 'level_draws', 'tensor_traversal', 'buffered_noise', 'aligned_noise')


def noise_storage(count, samples, width, *, device, dtype, aligned=False):
  stride = samples * width
  if aligned:
    stride = ((stride + 63) // 64) * 64
  storage = torch.empty(count, stride, device=device, dtype=dtype)
  return storage[:, :samples * width].view(count, samples, width)


def verify_buffered_noise(device, aligned=False):
  """Diagnostic: compare every noise value, not just resulting sampled tokens."""
  count = 0
  for dtype in (torch.float32, torch.float64):
    for samples in (1, 3):
      for width in (5, 33, 129, 257):
        for seed in (91001, 91002):
          old_rng = torch.Generator(device=device).manual_seed(seed)
          new_rng = torch.Generator(device=device).manual_seed(seed)
          old = torch.stack([torch.empty(samples, width, device=device, dtype=dtype).exponential_(1, generator=old_rng)
                             for _ in range(17)])
          new = noise_storage(17, samples, width, device=device, dtype=dtype, aligned=aligned)
          for row in new.unbind():
            row.exponential_(1, generator=new_rng)
          assert torch.equal(old, new), 'Buffered noise values differ'
          assert torch.equal(old_rng.get_state(), new_rng.get_state()), 'Buffered noise RNG differs'
          count += 1
  return count


def draw_with_noise(logits, noise):
  shape = logits.shape
  probabilities = torch.softmax(logits.reshape(-1, shape[-1]), dim=-1).reshape(shape)
  return torch.div(probabilities, noise).argmax(dim=-1)


def grouped_pair_rows(states, source, target):
  """Same arithmetic as _low_rank_pair_rows, with a leading edge axis."""
  k = source.shape[1]
  selected = torch.gather(source, 1, states.clamp_max(k - 1)[:, :, None].expand(-1, -1, source.shape[-1]))
  explicit = torch.logsumexp(selected[:, :, None, :] + target[:, None, :, :], dim=-1)
  rows = torch.cat((explicit, explicit.new_zeros(*states.shape, 1)), dim=-1)
  return torch.where(states.eq(k)[:, :, None], torch.zeros_like(rows), rows)


@torch.no_grad()
def level_sampler(node_log_potentials, left_factors, right_factors, edge_index,
                  num_samples, *, edge_mask=None, state_mask=None,
                  clamped_states=None, max_components=None,
                  max_component_size=None, generator=None, parallel_children=True,
                  tensor_gathers=False, buffered_noise=False, aligned_noise=False):
  utils._require(isinstance(num_samples, int) and num_samples > 0,
                 'num_samples must be a positive integer')
  nodes, left, right, edges, _, topologies = utils._validate_low_rank_inputs(
    node_log_potentials, left_factors, right_factors, edge_index,
    edge_mask, state_mask, clamped_states, max_components, max_component_size)
  batch_samples = []
  for batch, topology in enumerate(topologies):
    _, marginals, messages = utils._single_low_rank_sum_product(
      nodes[batch], left[batch], right[batch], edges[batch], topology, sampling_only=True)
    children = {node: [neighbor for neighbor, _, _ in topology.adjacency[node]
                       if neighbor != topology.parent[node]] for node in topology.order}
    nonroots = [node for node in topology.order if topology.parent[node] >= 0]
    # Keep exactly one original-shaped exponential call per node, including
    # clamped nodes. Roots first, then original traversal, separately per batch.
    # These noise values are independent of conditional probabilities.
    noise_order = list(topology.roots) + nonroots
    noise = {}
    if buffered_noise:
      noise_bank = noise_storage(len(noise_order), num_samples, nodes.shape[-1],
        dtype=nodes.dtype, device=nodes.device, aligned=aligned_noise)
      noise_slots = {node: slot for slot, node in enumerate(noise_order)}
      for node, row in zip(noise_order, noise_bank.unbind()):
        noise[node] = row.exponential_(1, generator=generator)
      def gather_noise(group):
        return noise_bank[[noise_slots[node] for node in group]]
    else:
      for node in noise_order:
        noise[node] = torch.empty((num_samples, nodes.shape[-1]),
          dtype=nodes.dtype, device=nodes.device).exponential_(1, generator=generator)
      def gather_noise(group):
        return torch.stack([noise[node] for node in group])
    if tensor_gathers:
      factor_bank = torch.cat((left[batch], right[batch]), dim=0)
      edge_count = left.shape[1]
      message_slots = {node: slot for slot, node in enumerate(nonroots)}
      message_bank = (torch.stack([messages[(node, topology.parent[node])] for node in nonroots])
        if nonroots else nodes.new_empty((0, nodes.shape[-1])))
    samples = torch.empty(num_samples, nodes.shape[1], dtype=torch.long, device=nodes.device)
    roots = list(topology.roots)
    root_logits = torch.stack([marginals[root] for root in roots])[:, None, :].expand(-1, num_samples, -1)
    samples[:, roots] = draw_with_noise(root_logits, gather_noise(roots)).T
    if not parallel_children:
      for node in nonroots:
        parent = topology.parent[node]
        edge = topology.parent_edge[node]
        local = nodes[batch, node]
        for child in children[node]:
          local = local + messages[(child, node)]
        source, target = ((left[batch, edge], right[batch, edge])
          if topology.edge_left[edge] == parent else (right[batch, edge], left[batch, edge]))
        logits = local[None, :] + utils._low_rank_pair_rows(samples[:, parent], source, target)
        samples[:, node] = draw_with_noise(logits, noise[node])
    else:
      depth = {}
      groups = defaultdict(list)
      for node in topology.order:
        parent = topology.parent[node]
        depth[node] = 0 if parent < 0 else depth[parent] + 1
        if parent >= 0:
          groups[(depth[node], len(children[node]))].append(node)
      for level, degree in sorted(groups):
        group = groups[(level, degree)]
        local = nodes[batch, group]
        for slot in range(degree):
          incoming = (message_bank[[message_slots[children[node][slot]] for node in group]]
            if tensor_gathers else torch.stack([messages[(children[node][slot], node)] for node in group]))
          local = local + incoming
        sources, targets = [], []
        parents = [topology.parent[node] for node in group]
        for node, parent in zip(group, parents):
          edge = topology.parent_edge[node]
          if tensor_gathers:
            source, target = ((edge, edge_count + edge) if topology.edge_left[edge] == parent
              else (edge_count + edge, edge))
          else:
            source, target = ((left[batch, edge], right[batch, edge])
              if topology.edge_left[edge] == parent else (right[batch, edge], left[batch, edge]))
          sources.append(source)
          targets.append(target)
        source_factors = factor_bank[sources] if tensor_gathers else torch.stack(sources)
        target_factors = factor_bank[targets] if tensor_gathers else torch.stack(targets)
        pairs = grouped_pair_rows(samples[:, parents].T, source_factors, target_factors)
        logits = local[:, None, :] + pairs
        samples[:, group] = draw_with_noise(logits, gather_noise(group)).T
    batch_samples.append(samples)
  return torch.stack(batch_samples)


@contextmanager
def experiment(mode):
  if mode not in MODES:
    with v3_experiment(mode):
      yield
    return
  with v3_experiment(BEST_V3), mock.patch.object(utils, 'sample_forest_low_rank',
      partial(level_sampler, parallel_children=(mode != 'root_draws'),
              tensor_gathers=mode in ('tensor_traversal', 'buffered_noise', 'aligned_noise'),
              buffered_noise=mode in ('buffered_noise', 'aligned_noise'),
              aligned_noise=(mode == 'aligned_noise'))):
    yield
