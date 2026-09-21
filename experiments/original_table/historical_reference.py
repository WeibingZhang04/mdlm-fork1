"""Historical R8-DD gate reference: unchanged pre-optimization sampling functions.

Only these three functions differ from the current structured_utils sampler.
Shared validators and math helpers are imported from that module.
"""
from typing import Optional, Tuple
import torch
from structured_utils import (
    _require, _low_rank_message, _validate_low_rank_inputs, _low_rank_pair_rows,
    _ForestTopology, sample_forest, positive_pair_factors_to_log,
)

def _sample_rows(logits: torch.Tensor,
                 generator: Optional[torch.Generator]) -> torch.Tensor:
  probabilities = torch.softmax(logits, dim=-1)
  _require(bool(torch.isfinite(probabilities).all().item())
           and bool((probabilities.sum(dim=-1) > 0).all().item()),
           'cannot sample from an empty or non-finite categorical row')
  return torch.multinomial(
    probabilities, num_samples=1, replacement=True,
    generator=generator).squeeze(-1)

def _single_low_rank_sum_product(
    node_log_potentials: torch.Tensor,
    left_log_factors: torch.Tensor,
    right_log_factors: torch.Tensor,
    edge_index: torch.Tensor,
    topology: _ForestTopology,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
  num_nodes = node_log_potentials.shape[0]
  messages = {}

  # Leaves to roots.
  for node in reversed(topology.order):
    parent = topology.parent[node]
    if parent < 0:
      continue
    local = node_log_potentials[node]
    for neighbor, _, _ in topology.adjacency[node]:
      if neighbor != parent:
        local = local + messages[(neighbor, node)]
    edge_id = topology.parent_edge[node]
    left = topology.edge_left[edge_id]
    if left == node:
      source, target = (
        left_log_factors[edge_id], right_log_factors[edge_id])
    else:
      source, target = (
        right_log_factors[edge_id], left_log_factors[edge_id])
    messages[(node, parent)] = _low_rank_message(
      local, source, target)

  # Roots to leaves.
  for node in topology.order:
    neighbors = topology.adjacency[node]
    incoming = [messages[(neighbor, node)]
                for neighbor, _, _ in neighbors]
    # Prefix/suffix sums form every leave-one-neighbor-out cavity in O(d K),
    # rather than O(d^2 K) at a high-degree node.  They also avoid subtracting
    # log messages, which is undefined when a hard constraint yields -inf.
    prefix = [torch.zeros_like(node_log_potentials[node])]
    for message in incoming:
      prefix.append(prefix[-1] + message)
    suffix = [None] * (len(incoming) + 1)
    suffix[-1] = torch.zeros_like(node_log_potentials[node])
    for index in range(len(incoming) - 1, -1, -1):
      suffix[index] = suffix[index + 1] + incoming[index]

    for neighbor_index, (neighbor, edge_id, forward) in enumerate(neighbors):
      if topology.parent[neighbor] != node:
        continue
      local = (
        node_log_potentials[node]
        + prefix[neighbor_index] + suffix[neighbor_index + 1])
      if forward:
        source, target = (
          left_log_factors[edge_id], right_log_factors[edge_id])
      else:
        source, target = (
          right_log_factors[edge_id], left_log_factors[edge_id])
      messages[(node, neighbor)] = _low_rank_message(
        local, source, target)

  node_beliefs = []
  for node in range(num_nodes):
    belief = node_log_potentials[node]
    for neighbor, _, _ in topology.adjacency[node]:
      belief = belief + messages[(neighbor, node)]
    node_beliefs.append(belief)

  component_log_partitions = torch.stack([
    torch.logsumexp(node_beliefs[root], dim=-1)
    for root in topology.roots
  ])
  _require(bool(torch.isfinite(component_log_partitions).all().item()),
           'constraints leave the forest with no finite-probability state')
  log_partition = component_log_partitions.sum()
  node_log_marginals = []
  for node, belief in enumerate(node_beliefs):
    log_marginal = (
      belief - component_log_partitions[topology.component[node]])
    log_marginal = log_marginal - torch.logsumexp(log_marginal, dim=-1)
    node_log_marginals.append(log_marginal)
  return log_partition, torch.stack(node_log_marginals), messages

@torch.no_grad()
def sample_forest_low_rank(
    node_log_potentials: torch.Tensor,
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    edge_index: torch.Tensor,
    num_samples: int,
    *,
    edge_mask: Optional[torch.Tensor] = None,
    state_mask: Optional[torch.Tensor] = None,
    clamped_states: Optional[torch.Tensor] = None,
    max_components: Optional[int] = None,
    max_component_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
  """Draw exact joint forest samples without dense pair materialisation.

  The inference pass is ``O(E K R)``.  Producing ``S`` joint samples costs
  ``O(S E K R)`` because each sampled parent state induces one conditional
  row.  The returned shape is ``(B,S,N)``.
  """
  _require(isinstance(num_samples, int) and num_samples > 0,
           'num_samples must be a positive integer')
  (
    constrained_nodes,
    left_log_factors,
    right_log_factors,
    edge_index,
    _,
    topologies,
  ) = _validate_low_rank_inputs(
    node_log_potentials, left_factors, right_factors, edge_index,
    edge_mask, state_mask, clamped_states,
    max_components, max_component_size)

  batch_samples = []
  for batch_index, topology in enumerate(topologies):
    _, node_log_marginals, messages = _single_low_rank_sum_product(
      constrained_nodes[batch_index],
      left_log_factors[batch_index], right_log_factors[batch_index],
      edge_index[batch_index], topology)
    samples = torch.empty(
      num_samples, node_log_potentials.shape[1], dtype=torch.long,
      device=node_log_potentials.device)
    for root in topology.roots:
      samples[:, root] = _sample_rows(
        node_log_marginals[root].expand(num_samples, -1), generator)

    for node in topology.order:
      parent = topology.parent[node]
      if parent < 0:
        continue
      edge_id = topology.parent_edge[node]
      local = constrained_nodes[batch_index, node]
      for neighbor, _, _ in topology.adjacency[node]:
        if neighbor != parent:
          local = local + messages[(neighbor, node)]
      left = topology.edge_left[edge_id]
      if left == parent:
        source, target = (
          left_log_factors[batch_index, edge_id],
          right_log_factors[batch_index, edge_id])
      else:
        source, target = (
          right_log_factors[batch_index, edge_id],
          left_log_factors[batch_index, edge_id])
      log_pair_rows = _low_rank_pair_rows(
        samples[:, parent], source, target)
      samples[:, node] = _sample_rows(
        local.unsqueeze(0) + log_pair_rows, generator)
    batch_samples.append(samples)
  return torch.stack(batch_samples)
