"""Exact inference utilities for discrete distributions on forests.

The distribution represented here is

  p(x) = exp(sum_i theta_i(x_i)) prod_(u,v) psi_uv(x_u, x_v) / Z,

where every pair factor ``psi_uv`` is strictly positive and is *not* locally
normalised.  The public inference API accepts positive endpoint factors
and computes the single global normaliser ``Z`` with sum-product.  This is
different from a directed model made of row-normalised transition matrices.

Low-rank pair factors deserve particular care.  If ``A`` and ``B`` are
positive, ``psi = A @ B.T`` is low-rank in factor space and
``log(psi) = logsumexp(log(A) + log(B))``.  A low-rank matrix of *log
potentials* generally becomes full-rank after exponentiation and is not the
same parameterisation.  Messages contract the positive endpoint factors
without materialising a dense pair matrix.

Training batches messages by depth over a fixed input forest. Joint sampling
uses a sequential traversal with the original random-draw ordering. Tensor
computations remain differentiable after top-K plus residual compression.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

import runtime_validation


@dataclass(frozen=True)
class LowRankForestMarginals:
  """Exact forest statistics computed without dense pair materialisation.

  Only node marginals are returned because a dense ``(K+1) x (K+1)`` edge
  marginal necessarily costs ``O(E K^2)`` to write.  The partition and all
  messages underlying these node marginals cost ``O(E K R)``.
  """

  log_partition: torch.Tensor
  node_log_marginals: torch.Tensor

  @property
  def node_marginals(self) -> torch.Tensor:
    return self.node_log_marginals.exp()


@dataclass(frozen=True)
class _ForestTopology:
  adjacency: Tuple[Tuple[Tuple[int, int, bool], ...], ...]
  roots: Tuple[int, ...]
  parent: Tuple[int, ...]
  parent_edge: Tuple[int, ...]
  order: Tuple[int, ...]
  component: Tuple[int, ...]
  active_edges: Tuple[int, ...]
  # Already transferred by _build_topology; avoid per-edge GPU .item() calls.
  edge_left: Tuple[int, ...]


@dataclass(frozen=True)
class _ChildDegreeBucket:
  """Parents with equal child degree and their compact child indices."""

  parent_slots: torch.Tensor
  child_indices: torch.Tensor


@dataclass(frozen=True)
class _LowRankLevelSchedule:
  """Device-side indices for depth-batched forest message passing."""

  node_ids: Tuple[torch.Tensor, ...]
  parent_slots: Tuple[Optional[torch.Tensor], ...]
  edge_ids: Tuple[Optional[torch.Tensor], ...]
  child_is_left: Tuple[Optional[torch.Tensor], ...]
  child_degree_buckets: Tuple[
    Optional[Tuple[_ChildDegreeBucket, ...]], ...]
  parent_inverse_order: Tuple[Optional[torch.Tensor], ...]
  child_inverse_order: Tuple[Optional[torch.Tensor], ...]
  inverse_node_order: torch.Tensor
  roots_by_batch: torch.Tensor


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _canonical_topology(edge_index: torch.Tensor,
                        edge_mask: Optional[torch.Tensor],
                        batch_size: int,
                        edge_count: int,
                        device: torch.device
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
  if not torch.is_tensor(edge_index):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
  _require(edge_index.ndim in (2, 3),
           'edge_index must have shape (edges, 2) or (batch, edges, 2)')
  _require(edge_index.shape[-1] == 2,
           'the final edge_index dimension must have size 2')
  _require(edge_index.shape[-2] == edge_count,
           'edge_index and log_pair_factors disagree on edge count')
  if edge_index.ndim == 2:
    edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
  else:
    _require(edge_index.shape[0] == batch_size,
             'batched edge_index has the wrong batch size')
  edge_index = edge_index.to(device=device, dtype=torch.long)

  if edge_mask is None:
    edge_mask = torch.ones(
      batch_size, edge_count, dtype=torch.bool, device=device)
  else:
    edge_mask = torch.as_tensor(
      edge_mask, dtype=torch.bool, device=device)
    if edge_mask.ndim == 1:
      _require(edge_mask.shape[0] == edge_count,
               'edge_mask has the wrong edge count')
      edge_mask = edge_mask.unsqueeze(0).expand(batch_size, -1)
    _require(edge_mask.shape == (batch_size, edge_count),
             'edge_mask must have shape (edges,) or (batch, edges)')
  return edge_index, edge_mask


def _build_topology(edge_index: torch.Tensor,
                    edge_mask: torch.Tensor,
                    num_nodes: int,
                    max_components: Optional[int],
                    max_component_size: Optional[int]
                    ) -> List[_ForestTopology]:
  _require(num_nodes > 0, 'a forest must contain at least one node')
  if max_components is not None:
    _require(max_components > 0, 'max_components must be positive')
  if max_component_size is not None:
    _require(max_component_size > 0,
             'max_component_size must be positive')

  # One bulk device-to-host transfer is dramatically cheaper than calling
  # ``.item()`` for every edge on a GPU.  Topology is discrete/stop-gradient,
  # so keeping the traversal metadata on the host does not alter autograd.
  host_edges = edge_index.detach().cpu().tolist()
  host_mask = edge_mask.detach().cpu().tolist()
  topologies = []
  for batch_index in range(edge_index.shape[0]):
    adjacency: List[List[Tuple[int, int, bool]]] = [
      [] for _ in range(num_nodes)]
    parent_dsu = list(range(num_nodes))

    def find(node: int) -> int:
      while parent_dsu[node] != node:
        parent_dsu[node] = parent_dsu[parent_dsu[node]]
        node = parent_dsu[node]
      return node

    seen_edges = set()
    active_edges = []
    for edge_id in range(edge_index.shape[1]):
      if not host_mask[batch_index][edge_id]:
        continue
      left, right = host_edges[batch_index][edge_id]
      _require(0 <= left < num_nodes and 0 <= right < num_nodes,
               'active edge endpoint is outside the node range')
      _require(left != right, 'self loops are not valid forest edges')
      key = (min(left, right), max(left, right))
      _require(key not in seen_edges,
               'duplicate undirected edges are not valid forest edges')
      seen_edges.add(key)
      root_left, root_right = find(left), find(right)
      _require(root_left != root_right,
               'active topology contains a cycle')
      parent_dsu[root_right] = root_left
      adjacency[left].append((right, edge_id, True))
      adjacency[right].append((left, edge_id, False))
      active_edges.append(edge_id)

    roots = []
    parent = [-1] * num_nodes
    parent_edge = [-1] * num_nodes
    component = [-1] * num_nodes
    order = []
    visited = [False] * num_nodes
    for candidate_root in range(num_nodes):
      if visited[candidate_root]:
        continue
      component_id = len(roots)
      roots.append(candidate_root)
      visited[candidate_root] = True
      component[candidate_root] = component_id
      stack = [candidate_root]
      component_size = 0
      while stack:
        node = stack.pop()
        component_size += 1
        order.append(node)
        # Reversal keeps the traversal deterministic under the input order.
        for neighbor, edge_id, _ in reversed(adjacency[node]):
          if neighbor == parent[node]:
            continue
          _require(not visited[neighbor],
                   'active topology contains a cycle')
          visited[neighbor] = True
          parent[neighbor] = node
          parent_edge[neighbor] = edge_id
          component[neighbor] = component_id
          stack.append(neighbor)
      if max_component_size is not None:
        _require(component_size <= max_component_size,
                 f'forest component has {component_size} nodes, exceeding '
                 f'cap {max_component_size}')

    if max_components is not None:
      _require(len(roots) <= max_components,
               f'forest has {len(roots)} components, exceeding cap '
               f'{max_components}')
    topologies.append(_ForestTopology(
      adjacency=tuple(tuple(neighbors) for neighbors in adjacency),
      roots=tuple(roots),
      parent=tuple(parent),
      parent_edge=tuple(parent_edge),
      order=tuple(order),
      component=tuple(component),
      active_edges=tuple(active_edges),
      edge_left=tuple(edge[0] for edge in host_edges[batch_index])))
  return topologies


def _constrain_nodes(node_log_potentials: torch.Tensor,
                     state_mask: Optional[torch.Tensor],
                     clamped_states: Optional[torch.Tensor]) -> torch.Tensor:
  batch_size, num_nodes, num_states = node_log_potentials.shape
  allowed = torch.ones_like(node_log_potentials, dtype=torch.bool)
  if state_mask is not None:
    state_mask = torch.as_tensor(
      state_mask, dtype=torch.bool, device=node_log_potentials.device)
    _require(state_mask.shape == node_log_potentials.shape,
             'state_mask must match node_log_potentials')
    allowed = allowed & state_mask

  if clamped_states is not None:
    clamped_states = torch.as_tensor(
      clamped_states, dtype=torch.long, device=node_log_potentials.device)
    _require(clamped_states.shape == (batch_size, num_nodes),
             'clamped_states must have shape (batch, nodes)')
    if runtime_validation.enabled():
      _require(bool(((clamped_states >= -1)
                     & (clamped_states < num_states)).all().item()),
               'clamped state indices must be -1 or valid state indices')
    is_clamped = clamped_states >= 0
    state_ids = torch.arange(
      num_states, device=node_log_potentials.device)
    clamp_allowed = (
      ~is_clamped.unsqueeze(-1)
      | (state_ids == clamped_states.clamp_min(0).unsqueeze(-1)))
    allowed = allowed & clamp_allowed

  if runtime_validation.enabled():
    _require(bool(allowed.any(dim=-1).all().item()),
             'every node must retain at least one allowed state')
  return node_log_potentials.masked_fill(~allowed, -torch.inf)


def _validate_low_rank_inputs(
    node_log_potentials: torch.Tensor,
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: Optional[torch.Tensor],
    state_mask: Optional[torch.Tensor],
    clamped_states: Optional[torch.Tensor],
    max_components: Optional[int],
    max_component_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, List[_ForestTopology]]:
  _require(torch.is_tensor(node_log_potentials)
           and node_log_potentials.ndim == 3
           and node_log_potentials.is_floating_point(),
           'node_log_potentials must have shape (batch, nodes, states)')
  _require(torch.is_tensor(left_factors) and left_factors.ndim == 4
           and torch.is_tensor(right_factors) and right_factors.ndim == 4,
           'endpoint factors must have shape '
           '(batch, edges, explicit_states, rank)')
  _require(left_factors.shape == right_factors.shape,
           'left and right endpoint factors must have identical shapes')
  _require(left_factors.is_floating_point()
           and right_factors.is_floating_point(),
           'endpoint factors must be floating-point tensors')
  _require(node_log_potentials.device == left_factors.device
           and left_factors.device == right_factors.device,
           'unaries and endpoint factors must share a device')
  _require(node_log_potentials.dtype == left_factors.dtype
           and left_factors.dtype == right_factors.dtype,
           'unaries and endpoint factors must share a dtype')

  batch_size, num_nodes, num_states = node_log_potentials.shape
  factor_batch, edge_count, explicit_states, rank = left_factors.shape
  _require(factor_batch == batch_size,
           'unaries and endpoint factors have different batch sizes')
  _require(explicit_states > 0 and rank > 0,
           'endpoint factors need positive state and rank dimensions')
  _require(num_states == explicit_states + 1,
           'node states must be explicit endpoint states plus one residual')
  if runtime_validation.enabled():
    invalid_nodes = (
      torch.isnan(node_log_potentials).any()
      | torch.isposinf(node_log_potentials).any())
    _require(not bool(invalid_nodes.item()),
             'node_log_potentials may contain -inf, but not NaN or +inf')
    valid_factors = (
      torch.isfinite(left_factors).all()
      & torch.isfinite(right_factors).all()
      & (left_factors > 0).all()
      & (right_factors > 0).all())
    _require(bool(valid_factors.item()),
             'all endpoint factors must be finite and strictly positive')

  edge_index, edge_mask = _canonical_topology(
    edge_index, edge_mask, batch_size, edge_count,
    node_log_potentials.device)
  topologies = _build_topology(
    edge_index, edge_mask, num_nodes,
    max_components, max_component_size)
  constrained_nodes = _constrain_nodes(
    node_log_potentials, state_mask, clamped_states)
  return (
    constrained_nodes, left_factors.log(), right_factors.log(),
    edge_index, edge_mask, topologies)


def _sample_rows(logits: torch.Tensor,
                 generator: Optional[torch.Generator]) -> torch.Tensor:
  probabilities = torch.softmax(logits, dim=-1)
  # multinomial already validates its weights; avoid two extra host syncs.
  return torch.multinomial(
    probabilities, num_samples=1, replacement=True,
    generator=generator).squeeze(-1)


def _low_rank_message(local_log_potential: torch.Tensor,
                      source_log_factors: torch.Tensor,
                      target_log_factors: torch.Tensor) -> torch.Tensor:
  """Send one exact message through an implicit low-rank-plus-residual edge.

  For explicit states, ``psi(i,j) = sum_r A(i,r) B(j,r)``.  The final state
  is the residual, and ``psi(residual,j) = psi(i,residual) = 1``.  Summing
  first over source states and then rank gives ``O(K R)`` work.
  """
  explicit_local = local_log_potential[:-1]
  residual_local = local_log_potential[-1]
  rank_summary = torch.logsumexp(
    explicit_local.unsqueeze(-1) + source_log_factors, dim=0)
  explicit_message = torch.logsumexp(
    target_log_factors + rank_summary.unsqueeze(0), dim=-1)
  # A residual source interacts neutrally with every explicit target.
  explicit_message = torch.logaddexp(
    explicit_message, residual_local.expand_as(explicit_message))
  # Every source state interacts neutrally with a residual target.
  residual_message = torch.logsumexp(local_log_potential, dim=-1)
  return torch.cat((explicit_message, residual_message.unsqueeze(0)))


def _safe_logsumexp(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
  """``logsumexp`` with zero gradients for an identically ``-inf`` slice."""
  all_impossible = torch.isneginf(input_tensor).all(dim=dim, keepdim=True)
  safe_input = torch.where(
    all_impossible, torch.zeros_like(input_tensor), input_tensor)
  result = torch.logsumexp(safe_input, dim=dim)
  return result.masked_fill(all_impossible.squeeze(dim), -torch.inf)


def _batched_low_rank_message(
    local_log_potential: torch.Tensor,
    source_log_factors: torch.Tensor,
    target_log_factors: torch.Tensor,
    safe_logsumexp: bool,
    ) -> torch.Tensor:
  """Vectorised version of :func:`_low_rank_message` over many edges."""
  explicit_local = local_log_potential[:, :-1]
  residual_local = local_log_potential[:, -1]
  # Hard clamps may exclude every explicit source state while leaving the
  # residual state available.  Native logsumexp has undefined (NaN) gradients
  # on an all--inf slice, even though that slice contributes exactly zero.
  logsumexp = _safe_logsumexp if safe_logsumexp else torch.logsumexp
  rank_summary = logsumexp(
    explicit_local.unsqueeze(-1) + source_log_factors, dim=1)
  explicit_message = logsumexp(
    target_log_factors + rank_summary.unsqueeze(1), dim=-1)
  explicit_message = torch.logaddexp(
    explicit_message, residual_local.unsqueeze(-1))
  residual_message = torch.logsumexp(local_log_potential, dim=-1)
  return torch.cat((explicit_message, residual_message.unsqueeze(-1)), dim=-1)


def _build_low_rank_level_schedule(
    topologies: Sequence[_ForestTopology],
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_count: int,
    ) -> _LowRankLevelSchedule:
  """Group independent messages by tree depth across the whole batch."""
  device = edge_index.device
  levels: List[List[int]] = []
  parents: List[List[int]] = []
  level_edges: List[List[int]] = []

  for batch_index, topology in enumerate(topologies):
    depths = [0] * num_nodes
    for node in topology.order:
      parent = topology.parent[node]
      depth = 0 if parent < 0 else depths[parent] + 1
      depths[node] = depth
      while len(levels) <= depth:
        levels.append([])
        parents.append([])
        level_edges.append([])
      levels[depth].append(batch_index * num_nodes + node)
      if depth > 0:
        parents[depth].append(batch_index * num_nodes + parent)
        level_edges[depth].append(
          batch_index * edge_count + topology.parent_edge[node])

  node_ids = tuple(torch.tensor(
    level, dtype=torch.long, device=device) for level in levels)
  parent_slots: List[Optional[torch.Tensor]] = [None]
  edge_ids: List[Optional[torch.Tensor]] = [None]
  child_is_left: List[Optional[torch.Tensor]] = [None]
  child_degree_buckets: List[
    Optional[Tuple[_ChildDegreeBucket, ...]]] = [None]
  parent_inverse_order: List[Optional[torch.Tensor]] = [None]
  child_inverse_order: List[Optional[torch.Tensor]] = [None]
  flat_edges = edge_index.reshape(-1, 2)
  for depth in range(1, len(levels)):
    parent_position = {
      node_id: position for position, node_id in enumerate(levels[depth - 1])
    }
    depth_parent_slots = [
      parent_position[parent] for parent in parents[depth]
    ]
    parent_slots.append(torch.tensor(
      depth_parent_slots, dtype=torch.long, device=device))
    child_groups: List[List[int]] = [
      [] for _ in range(len(levels[depth - 1]))
    ]
    for child_index, parent_slot in enumerate(depth_parent_slots):
      child_groups[parent_slot].append(child_index)
    parents_by_degree: dict[int, List[int]] = {}
    for parent_slot, group in enumerate(child_groups):
      parents_by_degree.setdefault(len(group), []).append(parent_slot)
    depth_buckets = []
    bucket_parent_order = []
    bucket_child_order = []
    for degree in sorted(parents_by_degree):
      degree_parents = parents_by_degree[degree]
      bucket_parent_order.extend(degree_parents)
      degree_children = [child_groups[parent] for parent in degree_parents]
      for group in degree_children:
        bucket_child_order.extend(group)
      depth_buckets.append(_ChildDegreeBucket(
        parent_slots=torch.tensor(
          degree_parents, dtype=torch.long, device=device),
        child_indices=torch.tensor(
          degree_children, dtype=torch.long, device=device).reshape(
            len(degree_parents), degree)))
    inverse_parents = [0] * len(child_groups)
    for position, parent_slot in enumerate(bucket_parent_order):
      inverse_parents[parent_slot] = position
    inverse_children = [0] * len(levels[depth])
    for position, child_index in enumerate(bucket_child_order):
      inverse_children[child_index] = position
    child_degree_buckets.append(tuple(depth_buckets))
    parent_inverse_order.append(torch.tensor(
      inverse_parents, dtype=torch.long, device=device))
    child_inverse_order.append(torch.tensor(
      inverse_children, dtype=torch.long, device=device))
    depth_edge_ids = torch.tensor(
      level_edges[depth], dtype=torch.long, device=device)
    edge_ids.append(depth_edge_ids)
    local_node_ids = node_ids[depth].remainder(num_nodes)
    child_is_left.append(
      flat_edges.index_select(0, depth_edge_ids)[:, 0] == local_node_ids)

  traversal_order = [node for level in levels for node in level]
  inverse_order = [0] * (len(topologies) * num_nodes)
  for position, node_id in enumerate(traversal_order):
    inverse_order[node_id] = position
  inverse_node_order = torch.tensor(
    inverse_order, dtype=torch.long, device=device)
  root_groups: List[List[int]] = [[] for _ in topologies]
  for root_position, root_node_id in enumerate(levels[0]):
    root_groups[root_node_id // num_nodes].append(root_position)
  max_roots = max(map(len, root_groups))
  root_sentinel = len(levels[0])
  roots_by_batch = torch.tensor([
    roots + [root_sentinel] * (max_roots - len(roots))
    for roots in root_groups
  ], dtype=torch.long, device=device)
  return _LowRankLevelSchedule(
    node_ids=node_ids,
    parent_slots=tuple(parent_slots),
    edge_ids=tuple(edge_ids),
    child_is_left=tuple(child_is_left),
    child_degree_buckets=tuple(child_degree_buckets),
    parent_inverse_order=tuple(parent_inverse_order),
    child_inverse_order=tuple(child_inverse_order),
    inverse_node_order=inverse_node_order,
    roots_by_batch=roots_by_batch)


def _oriented_low_rank_factors(
    flat_left: torch.Tensor,
    flat_right: torch.Tensor,
    edge_ids: torch.Tensor,
    source_is_left: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  left = flat_left.index_select(0, edge_ids)
  right = flat_right.index_select(0, edge_ids)
  orientation = source_is_left[:, None, None]
  return (
    torch.where(orientation, left, right),
    torch.where(orientation, right, left))


def _bucketed_group_sum(
    values: torch.Tensor,
    buckets: Sequence[_ChildDegreeBucket],
    parent_inverse_order: torch.Tensor,
    ) -> torch.Tensor:
  """Sum ragged child groups with exactly ``O(E)`` gathered values."""
  bucket_sums = []
  value_shape = values.shape[1:]
  for bucket in buckets:
    parent_count, degree = bucket.child_indices.shape
    if degree == 0:
      bucket_sums.append(values.new_zeros(parent_count, *value_shape))
      continue
    grouped = values.index_select(
      0, bucket.child_indices.reshape(-1)).reshape(
        parent_count, degree, *value_shape)
    bucket_sums.append(grouped.sum(dim=1))
  return torch.cat(bucket_sums, dim=0).index_select(
    0, parent_inverse_order)


def _exclusive_sibling_sums(
    values: torch.Tensor,
    buckets: Sequence[_ChildDegreeBucket],
    child_inverse_order: torch.Tensor,
    ) -> torch.Tensor:
  """Stably exclude each child using compact degree-bucketed scans."""
  bucket_exclusive = []
  value_shape = values.shape[1:]
  for bucket in buckets:
    parent_count, degree = bucket.child_indices.shape
    if degree == 0:
      continue
    grouped = values.index_select(
      0, bucket.child_indices.reshape(-1)).reshape(
        parent_count, degree, *value_shape)
    if degree == 1:
      exclusive = torch.zeros_like(grouped)
    else:
      # ``torch.cumsum`` has no deterministic CUDA implementation in the
      # PyTorch versions used by our training images.  Explicit recurrences
      # retain the same O(E) work/storage while making strict deterministic
      # training possible.  Forest heads normally impose a small degree cap,
      # so the extra launch count is bounded in the production path.
      zero = torch.zeros_like(grouped[:, 0])
      running = zero
      prefix_exclusive = []
      for position in range(degree):
        prefix_exclusive.append(running)
        running = running + grouped[:, position]
      running = zero
      suffix_exclusive = [zero] * degree
      for position in range(degree - 1, -1, -1):
        suffix_exclusive[position] = running
        running = running + grouped[:, position]
      exclusive = torch.stack([
        prefix_exclusive[position] + suffix_exclusive[position]
        for position in range(degree)
      ], dim=1)
    bucket_exclusive.append(exclusive.reshape(-1, *value_shape))
  return torch.cat(bucket_exclusive, dim=0).index_select(
    0, child_inverse_order)


def _center_log_vectors(
    values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """Remove a state-independent offset from each finite log vector."""
  offsets = values.max(dim=-1).values
  return values - offsets.unsqueeze(-1), offsets


def _vectorized_low_rank_sum_product(
    node_log_potentials: torch.Tensor,
    left_log_factors: torch.Tensor,
    right_log_factors: torch.Tensor,
    edge_index: torch.Tensor,
    topologies: Sequence[_ForestTopology],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  """Depth-batched exact sum-product for a batch of arbitrary forests.

  Every message at a given depth is independent, so this reduces the CUDA
  launch count from ``O(B E)`` to ``O(max_depth)``.  Messages are centered and
  their scalar offsets tracked separately in the upward pass.  Downward
  cavities use prefix/suffix sibling sums, never ``total - child``; this is
  invariant to large state-independent message offsets and avoids catastrophic
  cancellation.  Degree-bucketed reductions store and visit each child once,
  retaining ``O(E K)`` work space even for unbalanced trees while avoiding
  nondeterministic atomic ``index_add`` accumulation on CUDA.
  """
  batch_size, num_nodes, num_states = node_log_potentials.shape
  edge_count = left_log_factors.shape[1]
  schedule = _build_low_rank_level_schedule(
    topologies, edge_index, num_nodes, edge_count)
  flat_nodes = node_log_potentials.reshape(-1, num_states)
  flat_left = left_log_factors.reshape(
    -1, left_log_factors.shape[-2], left_log_factors.shape[-1])
  flat_right = right_log_factors.reshape_as(flat_left)
  depth_count = len(schedule.node_ids)
  # The guarded reduction is needed only for hard constraints.  Keeping the
  # common all-finite path on native logsumexp avoids extra kernels per depth.
  safe_logsumexp = bool(torch.isneginf(node_log_potentials).any().item())

  subtree_locals: List[Optional[torch.Tensor]] = [None] * depth_count
  subtree_offsets: List[Optional[torch.Tensor]] = [None] * depth_count
  upward: List[Optional[torch.Tensor]] = [None] * depth_count
  upward_offsets: List[Optional[torch.Tensor]] = [None] * depth_count
  for depth in range(depth_count - 1, -1, -1):
    node_ids = schedule.node_ids[depth]
    local = flat_nodes.index_select(0, node_ids)
    local_offsets = torch.zeros(
      local.shape[0], dtype=local.dtype, device=local.device)
    if depth + 1 < depth_count:
      child_messages = upward[depth + 1]
      child_offsets = upward_offsets[depth + 1]
      child_buckets = schedule.child_degree_buckets[depth + 1]
      parent_inverse = schedule.parent_inverse_order[depth + 1]
      assert (child_messages is not None and child_offsets is not None
              and child_buckets is not None and parent_inverse is not None)
      local = local + _bucketed_group_sum(
        child_messages, child_buckets, parent_inverse)
      local_offsets = _bucketed_group_sum(
        child_offsets.unsqueeze(-1), child_buckets,
        parent_inverse).squeeze(-1)
    local, centering_offset = _center_log_vectors(local)
    local_offsets = local_offsets + centering_offset
    subtree_locals[depth] = local
    subtree_offsets[depth] = local_offsets
    if depth > 0:
      edge_ids = schedule.edge_ids[depth]
      child_is_left = schedule.child_is_left[depth]
      assert edge_ids is not None and child_is_left is not None
      source, target = _oriented_low_rank_factors(
        flat_left, flat_right, edge_ids, child_is_left)
      message = _batched_low_rank_message(
        local, source, target, safe_logsumexp)
      message, message_offset = _center_log_vectors(message)
      upward[depth] = message
      upward_offsets[depth] = local_offsets + message_offset

  downward: List[Optional[torch.Tensor]] = [None] * depth_count
  assert subtree_locals[0] is not None
  downward[0] = torch.zeros_like(subtree_locals[0])
  for depth in range(depth_count - 1):
    parent_slots = schedule.parent_slots[depth + 1]
    child_upward = upward[depth + 1]
    parent_downward = downward[depth]
    child_buckets = schedule.child_degree_buckets[depth + 1]
    child_inverse = schedule.child_inverse_order[depth + 1]
    assert (parent_slots is not None and child_upward is not None
            and parent_downward is not None and child_buckets is not None
            and child_inverse is not None)
    parent_base = (
      flat_nodes.index_select(0, schedule.node_ids[depth])
      + parent_downward)
    sibling_sum = _exclusive_sibling_sums(
      child_upward, child_buckets, child_inverse)
    cavity = parent_base.index_select(0, parent_slots) + sibling_sum
    cavity, _ = _center_log_vectors(cavity)
    edge_ids = schedule.edge_ids[depth + 1]
    child_is_left = schedule.child_is_left[depth + 1]
    assert edge_ids is not None and child_is_left is not None
    child_source, parent_target = _oriented_low_rank_factors(
      flat_left, flat_right, edge_ids, child_is_left)
    child_message = _batched_low_rank_message(
      cavity, parent_target, child_source, safe_logsumexp)
    downward[depth + 1], _ = _center_log_vectors(child_message)

  beliefs = []
  for subtree, parent_message in zip(subtree_locals, downward):
    assert subtree is not None and parent_message is not None
    beliefs.append(subtree + parent_message)
  assert subtree_offsets[0] is not None
  root_log_partitions = (
    torch.logsumexp(beliefs[0], dim=-1) + subtree_offsets[0])
  _require(bool(torch.isfinite(root_log_partitions).all().item()),
           'constraints leave the forest with no finite-probability state')
  padded_root_partitions = torch.cat((
    root_log_partitions,
    torch.zeros_like(root_log_partitions[:1])))
  log_partition = padded_root_partitions.index_select(
    0, schedule.roots_by_batch.reshape(-1)).reshape(
      batch_size, -1).sum(dim=1)

  level_marginals = [
    belief - torch.logsumexp(belief, dim=-1, keepdim=True)
    for belief in beliefs
  ]
  node_log_marginals = torch.cat(level_marginals, dim=0).index_select(
    0, schedule.inverse_node_order).reshape(batch_size, num_nodes, num_states)
  return log_partition, node_log_marginals


def _single_low_rank_sum_product(
    node_log_potentials: torch.Tensor,
    left_log_factors: torch.Tensor,
    right_log_factors: torch.Tensor,
    edge_index: torch.Tensor,
    topology: _ForestTopology,
    *, sampling_only: bool = False,
    ) -> tuple:
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

  # Joint sampling needs only upward messages; full marginals need both ways.
  for node in (() if sampling_only else topology.order):
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

  node_beliefs = {}
  for node in (topology.roots if sampling_only else range(num_nodes)):
    belief = node_log_potentials[node]
    for neighbor, _, _ in topology.adjacency[node]:
      belief = belief + messages[(neighbor, node)]
    node_beliefs[node] = belief

  component_log_partitions = torch.stack([
    torch.logsumexp(node_beliefs[root], dim=-1)
    for root in topology.roots
  ])
  _require(bool(torch.isfinite(component_log_partitions).all().item()),
           'constraints leave the forest with no finite-probability state')
  log_partition = component_log_partitions.sum()
  node_log_marginals = {}
  for node, belief in node_beliefs.items():
    log_marginal = (
      belief - component_log_partitions[topology.component[node]])
    log_marginal = log_marginal - torch.logsumexp(log_marginal, dim=-1)
    node_log_marginals[node] = log_marginal
  if not sampling_only:
    node_log_marginals = torch.stack(list(node_log_marginals.values()))
  return log_partition, node_log_marginals, messages


def forest_sum_product_low_rank(
    node_log_potentials: torch.Tensor,
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_mask: Optional[torch.Tensor] = None,
    state_mask: Optional[torch.Tensor] = None,
    clamped_states: Optional[torch.Tensor] = None,
    max_components: Optional[int] = None,
    max_component_size: Optional[int] = None,
    ) -> LowRankForestMarginals:
  """Exact forest inference directly from positive endpoint factors.

  Shapes and semantics:

  * ``node_log_potentials``: ``(B,N,K+1)``.  State ``K`` is residual.
  * ``left_factors``, ``right_factors``: ``(B,E,K,R)`` finite positive
    tensors aligned with the two axes of ``edge_index``.
  * For explicit states, ``psi_e(i,j) = sum_r L_e(i,r) R_e(j,r)``.
  * Every pair factor involving the residual state is exactly one.
  * Masked padded edges are ignored, which is equivalent to a neutral factor.

  The distribution is globally, not row-wise, normalised.  Sum-product costs
  ``O(B E K R + B N K)`` arithmetic and ``O(B E K + B N K)`` message memory;
  no dense ``K x K`` pair matrix is formed.  Topology traversal itself is over
  the fixed, stop-gradient input forest.
  """
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
  log_partition, node_log_marginals = _vectorized_low_rank_sum_product(
    constrained_nodes, left_log_factors, right_log_factors,
    edge_index, topologies)
  return LowRankForestMarginals(
    log_partition=log_partition,
    node_log_marginals=node_log_marginals)


def _low_rank_pair_rows(
    source_states: torch.Tensor,
    source_log_factors: torch.Tensor,
    target_log_factors: torch.Tensor,
    ) -> torch.Tensor:
  """Materialise only sampled conditional rows, including residual state."""
  explicit_states = source_log_factors.shape[0]
  source_is_residual = source_states == explicit_states
  safe_source_states = source_states.clamp_max(explicit_states - 1)
  selected_source = source_log_factors[safe_source_states]
  explicit_pair_rows = torch.logsumexp(
    selected_source.unsqueeze(-2)
    + target_log_factors.unsqueeze(0), dim=-1)
  pair_rows = torch.cat((
    explicit_pair_rows,
    torch.zeros(
      source_states.shape[0], 1,
      dtype=source_log_factors.dtype,
      device=source_log_factors.device)), dim=-1)
  return torch.where(
    source_is_residual.unsqueeze(-1),
    torch.zeros_like(pair_rows), pair_rows)


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
      edge_index[batch_index], topology, sampling_only=True)
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
