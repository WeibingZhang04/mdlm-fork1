"""Historical level-draw sampler; numerical functions copied without changes.

Extracted from the original audit modules; omitted CLI/profiling code.
See provenance.json and README.md for source identity and attribution.
"""
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from unittest import mock
import torch
import inspect
import models.structured_decoder as decoder
import structured_utils as utils
ORIGINAL_INFER = utils._single_low_rank_sum_product

def unchecked_rows(logits, generator):
  # Same softmax, exponential draw shape/order, division, and argmax as native
  # multinomial with num_samples=1. Only valid-input checks are omitted.
  probabilities = torch.softmax(logits, dim=-1)
  noise = torch.empty_like(probabilities).exponential_(1, generator=generator)
  torch.div(probabilities, noise, out=noise)
  return noise.argmax(dim=-1)


def batched_upward(nodes, left, right, edges, topology, *, sampling_only=False,
                   batch_roots=False):
  if not sampling_only:
    return ORIGINAL_INFER(nodes, left, right, edges, topology)
  children = {node: [neighbor for neighbor, _, _ in topology.adjacency[node]
                     if neighbor != topology.parent[node]]
              for node in topology.order}
  depth = {}
  buckets = defaultdict(list)
  for node in topology.order:
    parent = topology.parent[node]
    depth[node] = 0 if parent < 0 else depth[parent] + 1
    if parent >= 0:
      buckets[(depth[node], len(children[node]))].append(node)
  messages = {}
  for level, degree in sorted(buckets, reverse=True):
    group = buckets[(level, degree)]
    local = nodes[group]
    # Preserve adjacency summation order; do not replace with a tree reduction.
    for slot in range(degree):
      local = local + torch.stack([messages[(children[node][slot], node)] for node in group])
    source = []
    target = []
    for node in group:
      edge = topology.parent_edge[node]
      a, b = (left[edge], right[edge]) if topology.edge_left[edge] == node else (right[edge], left[edge])
      source.append(a)
      target.append(b)
    outgoing = utils._batched_low_rank_message(
      local, torch.stack(source), torch.stack(target), safe_logsumexp=False)
    for slot, node in enumerate(group):
      messages[(node, topology.parent[node])] = outgoing[slot]
  beliefs = {}
  if batch_roots:
    root_groups = defaultdict(list)
    for root in topology.roots:
      root_groups[len(topology.adjacency[root])].append(root)
    for degree, roots in root_groups.items():
      value = nodes[roots]
      for slot in range(degree):
        value = value + torch.stack([
          messages[(topology.adjacency[root][slot][0], root)] for root in roots])
      for root, row in zip(roots, value.unbind()):
        beliefs[root] = row
    values = torch.stack([beliefs[root] for root in topology.roots])
    partitions = torch.logsumexp(values, -1)
    utils._require(bool(torch.isfinite(partitions).all().item()),
                   'constraints leave the forest with no finite-probability state')
    normalized = values - partitions[:, None]
    normalized = normalized - torch.logsumexp(normalized, -1, keepdim=True)
    return partitions.sum(), dict(zip(topology.roots, normalized.unbind())), messages
  for root in topology.roots:
    value = nodes[root]
    for neighbor, _, _ in topology.adjacency[root]:
      value = value + messages[(neighbor, root)]
    beliefs[root] = value
  partitions = torch.stack([torch.logsumexp(beliefs[root], -1) for root in topology.roots])
  utils._require(bool(torch.isfinite(partitions).all().item()),
                 'constraints leave the forest with no finite-probability state')
  marginals = {}
  for root in topology.roots:
    value = beliefs[root] - partitions[topology.component[root]]
    marginals[root] = value - torch.logsumexp(value, -1)
  return partitions.sum(), marginals, messages


def list_kruskal():
  """Keep stable argsort/selection logic, replace per-edge CPU tensor indexing."""
  source = inspect.getsource(decoder._bounded_kruskal_indices)
  source = source.replace('edges_cpu = proposal_edge_index.detach().cpu()',
                          'edges_cpu = proposal_edge_index.detach().cpu().tolist()')
  source = source.replace('active_cpu = active_mask.detach().cpu()',
                          'active_cpu = active_mask.detach().cpu().tolist()\n  score_values = scores_cpu.tolist()')
  source = source.replace('bool(active_cpu[batch_index, i])', 'active_cpu[batch_index][i]')
  source = source.replace('float(scores_cpu[batch_index, proposal_slot])', 'score_values[batch_index][proposal_slot]')
  source = source.replace('edges_cpu[batch_index, proposal_slot].tolist()', 'edges_cpu[batch_index][proposal_slot]')
  namespace = {}
  exec(compile(source, '<list-kruskal-prototype>', 'exec'), vars(decoder), namespace)
  return namespace['_bounded_kruskal_indices']


def noise_storage(count, samples, width, *, device, dtype, aligned=False):
  stride = samples * width
  if aligned:
    stride = ((stride + 63) // 64) * 64
  storage = torch.empty(count, stride, device=device, dtype=dtype)
  return storage[:, :samples * width].view(count, samples, width)


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
    if mode != 'level_draws':
        raise ValueError('Only the historically used level_draws mode is supported')
    with mock.patch.object(decoder, '_bounded_kruskal_indices', list_kruskal()), \
         mock.patch.object(utils, '_sample_rows', unchecked_rows), \
         mock.patch.object(utils, '_single_low_rank_sum_product', partial(batched_upward, batch_roots=True)), \
         mock.patch.object(utils, 'sample_forest_low_rank', partial(level_sampler,
             parallel_children=True, tensor_gathers=False, buffered_noise=False, aligned_noise=False)):
        yield
