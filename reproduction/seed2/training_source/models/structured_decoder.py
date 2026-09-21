"""Lightweight contextual coupling-forest output head.

This module intentionally does not own a language-model backbone.  It consumes
the hidden states and unary logits already produced by a DIT (or another masked
diffusion model), then constructs the small structured object needed by exact
tree inference:

* target-independent top-K token states and one residual state per position;
* local and learned/contextual sparse edge proposals;
* a hard, stop-gradient, bounded-component maximum-spanning forest; and
* positive rank-R token factors on the selected edges.

The hot path never runs candidate-conditioned attention over the vocabulary.
It performs one ordinary top-K on the existing unary logits, token-embedding
lookups for retained candidates, and O(|E| K R) factor gathers.  Dense K-by-K
pair factors can be materialized for debugging, but exact inference should
consume the endpoint factors directly.
"""

import dataclasses
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_MODES = frozenset({'fixed', 'dynamic'})


def _check_mode(name: str, value: str) -> str:
  if value not in _VALID_MODES:
    raise ValueError(
      f'{name} must be one of {sorted(_VALID_MODES)}, got {value!r}')
  return value


@dataclasses.dataclass
class StructuredDecoderOutput:
  """Tensor bundle returned by :class:`ContextualCouplingForestHead`.

  Shapes use ``B`` for batch, ``L`` for sequence length, ``K`` for retained
  token candidates, ``P`` for proposals, ``E=L-1`` for padded selected edges,
  and ``R`` for pair-factor rank.

  ``unary_log_potentials`` has K explicit states followed by a residual state.
  ``pair_left_factors`` and ``pair_right_factors`` describe only the explicit
  K-by-K block.  A pair factor is their matrix product; every row or column
  involving the residual state is exactly one.
  """

  candidate_ids: torch.Tensor              # [B, L, K]
  unary_log_potentials: torch.Tensor        # [B, L, K + 1]
  candidate_state_mask: torch.Tensor        # [B, L, K + 1]
  retained_mass: torch.Tensor               # [B, L]
  residual_log_mass: torch.Tensor           # [B, L]

  proposal_edge_index: torch.Tensor         # [B, P, 2]
  proposal_edge_mask: torch.Tensor          # [B, P]
  proposal_scores: torch.Tensor             # [B, P]
  anchor_logits: torch.Tensor               # [B, L, A]
  anchor_indices: torch.Tensor              # [B, A]
  slot_logits: torch.Tensor                 # [B, L, A]

  edge_index: torch.Tensor                  # [B, E, 2]
  edge_mask: torch.Tensor                   # [B, E]
  edge_scores: torch.Tensor                 # [B, E]
  pair_left_factors: torch.Tensor           # [B, E, K, R]
  pair_right_factors: torch.Tensor          # [B, E, K, R]

  topology_mode: str
  factor_mode: str
  independent_mode: bool

  @property
  def num_candidate_states(self) -> int:
    """Number of states per node, including the residual state."""
    return self.unary_log_potentials.shape[-1]

  def materialize_pair_factors(self) -> torch.Tensor:
    """Return dense positive factors of shape ``[B,E,K+1,K+1]``.

    This is intended for exhaustive tests and small-K reference inference.
    Production sum-product should use the endpoint factors to retain
    O(|E| K R) message complexity.
    """
    explicit = torch.einsum(
      'beir,bejr->beij',
      self.pair_left_factors,
      self.pair_right_factors)
    dense = F.pad(explicit, (0, 1, 0, 1), value=1.0)
    return torch.where(
      self.edge_mask[:, :, None, None], dense, torch.ones_like(dense))

  def residual_log_probs(self, unary_logits: torch.Tensor) -> torch.Tensor:
    """Normalize logits over each residual state's omitted vocabulary.

    This deliberately materializes a vocabulary-sized tensor and should only
    be called if a sampled node actually lands in its residual state.  The
    main structured forward pass never pays this allocation.
    """
    if unary_logits.shape[:-1] != self.candidate_ids.shape[:-1]:
      raise ValueError(
        'unary_logits leading dimensions must match candidate_ids')
    tail_logits = unary_logits.float().clone()
    tail_logits.scatter_(-1, self.candidate_ids, -torch.inf)
    tail_normalizer = torch.logsumexp(tail_logits, dim=-1, keepdim=True)
    return tail_logits - tail_normalizer


class ScalarTimestepEmbedding(nn.Module):
  """Small sinusoidal MLP for one scalar diffusion time per example."""

  def __init__(self, dim: int, max_period: int = 10_000):
    super().__init__()
    if dim < 2:
      raise ValueError('timestep embedding dimension must be at least 2')
    self.dim = dim
    half = dim // 2
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
      -math.log(max_period)
      * torch.arange(half, dtype=torch.float32) / denominator)
    self.register_buffer('frequencies', frequencies, persistent=False)
    self.mlp = nn.Sequential(
      nn.Linear(dim, dim),
      nn.SiLU(),
      nn.Linear(dim, dim))

  def forward(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
    if timestep.ndim == 0:
      timestep = timestep.expand(batch_size)
    if timestep.ndim == 2 and timestep.shape[-1] == 1:
      timestep = timestep[:, 0]
    if timestep.ndim != 1 or timestep.shape[0] != batch_size:
      raise ValueError(
        'timestep must be scalar, [B], or [B,1]; '
        f'got {tuple(timestep.shape)} for B={batch_size}')

    compute_dtype = self.mlp[0].weight.dtype
    arguments = (
      timestep.to(dtype=compute_dtype)[:, None]
      * self.frequencies.to(dtype=compute_dtype)[None])
    embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
    if embedding.shape[-1] < self.dim:
      embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
    return self.mlp(embedding)


class SparseEdgeProposer(nn.Module):
  """Build local plus learned-anchor edge proposals in O(B L A).

  ``A`` learned anchor slots each select one active position from the context.
  Every position then chooses a small number of those slots.  This supplies
  context-dependent nonlocal proposals without computing an L-by-L attention
  matrix.  Duplicate undirected proposals are harmless: Kruskal selection
  below can accept at most one of them.
  """

  def __init__(self, topology_dim: int, local_window: int,
               num_anchor_slots: int, contextual_neighbors: int):
    super().__init__()
    if local_window < 0:
      raise ValueError('local_window must be nonnegative')
    if num_anchor_slots < 1:
      raise ValueError('num_anchor_slots must be positive')
    if not 0 <= contextual_neighbors <= num_anchor_slots:
      raise ValueError(
        'contextual_neighbors must lie in [0, num_anchor_slots]')
    self.local_window = local_window
    self.num_anchor_slots = num_anchor_slots
    self.contextual_neighbors = contextual_neighbors

    self.anchor_projection = nn.Linear(
      topology_dim, num_anchor_slots, bias=True)
    self.slot_projection = nn.Linear(
      topology_dim, num_anchor_slots, bias=True)
    self.edge_scorer = nn.Sequential(
      nn.Linear(2 * topology_dim, topology_dim),
      nn.SiLU(),
      nn.Linear(topology_dim, 1))

  @staticmethod
  def _gather_nodes(node_context: torch.Tensor,
                    node_indices: torch.Tensor) -> torch.Tensor:
    gather_index = node_indices[:, :, None].expand(
      -1, -1, node_context.shape[-1])
    return torch.gather(node_context, 1, gather_index)

  def score_edges(self, node_context: torch.Tensor,
                  edge_index: torch.Tensor,
                  edge_mask: torch.Tensor) -> torch.Tensor:
    """Score padded undirected edges with a symmetric contextual MLP."""
    if edge_index.shape[1] == 0:
      return node_context.new_zeros(edge_mask.shape)
    left = self._gather_nodes(node_context, edge_index[:, :, 0])
    right = self._gather_nodes(node_context, edge_index[:, :, 1])
    features = torch.cat(((left - right).abs(), left * right), dim=-1)
    scores = self.edge_scorer(features).squeeze(-1)
    return scores.masked_fill(~edge_mask, -torch.inf)

  def forward(self, node_context: torch.Tensor,
              active_mask: torch.Tensor
              ) -> Tuple[torch.Tensor, ...]:
    """Return proposals, scores, and learned-anchor diagnostics.

    Returns:
      edge_index: ``[B,P,2]`` canonical undirected node indices.
      edge_mask: ``[B,P]`` validity mask.
      scores: ``[B,P]`` differentiable contextual edge scores.
      anchor_logits: ``[B,L,A]`` logits for which node fills each slot.
      anchor_indices: ``[B,A]`` detached hard slot occupants.
      slot_logits: ``[B,L,A]`` per-node logits for choosing anchor slots.
    """
    batch_size, sequence_length, _ = node_context.shape
    device = node_context.device

    anchor_logits = self.anchor_projection(node_context)
    masked_anchor_logits = anchor_logits.masked_fill(
      ~active_mask[:, :, None], -torch.inf)
    anchor_indices = masked_anchor_logits.argmax(dim=1).detach()
    any_active = active_mask.any(dim=1)
    anchor_valid = any_active[:, None].expand(
      batch_size, self.num_anchor_slots)
    slot_logits = self.slot_projection(node_context)

    local_edges = []
    maximum_offset = min(self.local_window, sequence_length - 1)
    for offset in range(1, maximum_offset + 1):
      left = torch.arange(
        sequence_length - offset, device=device, dtype=torch.long)
      local_edges.append(torch.stack((left, left + offset), dim=-1))
    if local_edges:
      local_index = torch.cat(local_edges, dim=0)
      local_index = local_index[None].expand(batch_size, -1, -1)
      local_mask = (
        torch.gather(active_mask, 1, local_index[:, :, 0])
        & torch.gather(active_mask, 1, local_index[:, :, 1]))
    else:
      local_index = torch.zeros(
        batch_size, 0, 2, dtype=torch.long, device=device)
      local_mask = torch.zeros(
        batch_size, 0, dtype=torch.bool, device=device)

    if self.contextual_neighbors:
      chosen_slots = slot_logits.topk(
        self.contextual_neighbors, dim=-1).indices
      flat_slots = chosen_slots.reshape(batch_size, -1)
      contextual_right = torch.gather(
        anchor_indices, 1, flat_slots).reshape(
          batch_size, sequence_length, self.contextual_neighbors)
      contextual_left = torch.arange(
        sequence_length, device=device, dtype=torch.long)
      contextual_left = contextual_left[None, :, None].expand_as(
        contextual_right)
      contextual_slot_valid = torch.gather(
        anchor_valid, 1, flat_slots).reshape_as(contextual_right)
      contextual_mask = (
        active_mask[:, :, None]
        & torch.gather(active_mask, 1, contextual_right.reshape(
          batch_size, -1)).reshape_as(contextual_right)
        & contextual_slot_valid
        & (contextual_left != contextual_right))
      contextual_index = torch.stack((
        torch.minimum(contextual_left, contextual_right),
        torch.maximum(contextual_left, contextual_right)), dim=-1)
      contextual_index = contextual_index.reshape(batch_size, -1, 2)
      contextual_mask = contextual_mask.reshape(batch_size, -1)
    else:
      contextual_index = torch.zeros(
        batch_size, 0, 2, dtype=torch.long, device=device)
      contextual_mask = torch.zeros(
        batch_size, 0, dtype=torch.bool, device=device)

    edge_index = torch.cat((local_index, contextual_index), dim=1)
    edge_mask = torch.cat((local_mask, contextual_mask), dim=1)
    scores = self.score_edges(node_context, edge_index, edge_mask)
    return (
      edge_index, edge_mask, scores,
      anchor_logits, anchor_indices, slot_logits)


def _bounded_kruskal_indices(
    proposal_edge_index: torch.Tensor,
    proposal_scores: torch.Tensor,
    proposal_edge_mask: torch.Tensor,
    active_mask: torch.Tensor,
    component_size_cap: int,
    min_edge_score: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
  """Select a stop-gradient maximum-spanning forest with capped components.

  The hard selection is a deterministic Kruskal pass over detached scores.
  The component-size condition makes this a greedy bounded Kruskal forest;
  without the cap it is the exact maximum-spanning forest of the proposal
  graph.  Returned proposal slots are padded to ``L-1``.
  """
  batch_size, sequence_length = active_mask.shape
  max_edges = max(sequence_length - 1, 0)
  device = proposal_edge_index.device
  selected_slots = torch.zeros(
    batch_size, max_edges, dtype=torch.long, device=device)
  selected_mask = torch.zeros(
    batch_size, max_edges, dtype=torch.bool, device=device)
  if max_edges == 0 or proposal_edge_index.shape[1] == 0:
    return selected_slots, selected_mask

  edges_cpu = proposal_edge_index.detach().cpu()
  scores_cpu = proposal_scores.detach().float().cpu()
  masks_cpu = proposal_edge_mask.detach().cpu()
  active_cpu = active_mask.detach().cpu()
  chosen_per_batch = []

  for batch_index in range(batch_size):
    parent = list(range(sequence_length))
    component_size = [1 if bool(active_cpu[batch_index, i]) else 0
                      for i in range(sequence_length)]

    def find(node):
      while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
      return node

    valid_slots = masks_cpu[batch_index].nonzero(
      as_tuple=False).flatten()
    if valid_slots.numel():
      ordering = valid_slots[torch.argsort(
        scores_cpu[batch_index, valid_slots], descending=True,
        stable=True)]
    else:
      ordering = valid_slots

    chosen = []
    for proposal_slot in ordering.tolist():
      score = float(scores_cpu[batch_index, proposal_slot])
      if min_edge_score is not None and score < min_edge_score:
        break
      left, right = edges_cpu[batch_index, proposal_slot].tolist()
      left_root, right_root = find(left), find(right)
      if left_root == right_root:
        continue
      merged_size = (
        component_size[left_root] + component_size[right_root])
      if (component_size_cap > 0
          and merged_size > component_size_cap):
        continue
      if (component_size[left_root] < component_size[right_root]
          or (component_size[left_root] == component_size[right_root]
              and left_root > right_root)):
        left_root, right_root = right_root, left_root
      parent[right_root] = left_root
      component_size[left_root] = merged_size
      chosen.append(proposal_slot)
      if len(chosen) == max_edges:
        break
    chosen_per_batch.append(chosen)

  for batch_index, chosen in enumerate(chosen_per_batch):
    if not chosen:
      continue
    count = len(chosen)
    selected_slots[batch_index, :count] = torch.tensor(
      chosen, dtype=torch.long, device=device)
    selected_mask[batch_index, :count] = True
  return selected_slots.detach(), selected_mask.detach()


def _fixed_chain_edges(active_mask: torch.Tensor,
                       component_size_cap: int
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
  """Create a target-independent natural-order chain over active positions."""
  batch_size, sequence_length = active_mask.shape
  max_edges = max(sequence_length - 1, 0)
  device = active_mask.device
  edge_index = torch.zeros(
    batch_size, max_edges, 2, dtype=torch.long, device=device)
  edge_mask = torch.zeros(
    batch_size, max_edges, dtype=torch.bool, device=device)
  active_cpu = active_mask.detach().cpu()
  for batch_index in range(batch_size):
    nodes = active_cpu[batch_index].nonzero(
      as_tuple=False).flatten().tolist()
    if component_size_cap > 0:
      chunks = [nodes[start:start + component_size_cap]
                for start in range(0, len(nodes), component_size_cap)]
    else:
      chunks = [nodes]
    edges = []
    for chunk in chunks:
      edges.extend(zip(chunk[:-1], chunk[1:]))
    if edges:
      count = len(edges)
      edge_index[batch_index, :count] = torch.tensor(
        edges, dtype=torch.long, device=device)
      edge_mask[batch_index, :count] = True
  return edge_index.detach(), edge_mask.detach()


def _validated_fixed_edges(
    fixed_edge_index: torch.Tensor,
    fixed_edge_mask: Optional[torch.Tensor],
    active_mask: torch.Tensor,
    component_size_cap: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  """Canonicalize and validate a shared or batched static forest.

  Valid input edges must form a simple undirected forest independently in
  every batch item.  Masked input slots are ignored (and may contain padding
  sentinels).  The returned topology is detached and padded to ``L-1``.
  """
  batch_size, sequence_length = active_mask.shape
  device = active_mask.device
  if not torch.is_tensor(fixed_edge_index):
    fixed_edge_index = torch.as_tensor(fixed_edge_index, device=device)
  elif fixed_edge_index.device != device:
    fixed_edge_index = fixed_edge_index.to(device=device)
  if (fixed_edge_index.dtype == torch.bool
      or fixed_edge_index.is_floating_point()
      or fixed_edge_index.is_complex()):
    raise TypeError('fixed_edge_index must contain integer node indices')
  if fixed_edge_index.ndim == 2:
    if fixed_edge_index.shape[-1] != 2:
      raise ValueError('shared fixed_edge_index must have shape [E,2]')
    fixed_edge_index = fixed_edge_index[None].expand(
      batch_size, -1, -1)
  elif fixed_edge_index.ndim == 3:
    if (fixed_edge_index.shape[0] != batch_size
        or fixed_edge_index.shape[-1] != 2):
      raise ValueError('batched fixed_edge_index must have shape [B,E,2]')
  else:
    raise ValueError(
      'fixed_edge_index must have shared [E,2] or batched [B,E,2] shape')
  fixed_edge_index = fixed_edge_index.to(dtype=torch.long)
  input_edges = fixed_edge_index.shape[1]

  if fixed_edge_mask is None:
    fixed_edge_mask = torch.ones(
      batch_size, input_edges, dtype=torch.bool, device=device)
  else:
    if not torch.is_tensor(fixed_edge_mask):
      fixed_edge_mask = torch.as_tensor(fixed_edge_mask, device=device)
    elif fixed_edge_mask.device != device:
      fixed_edge_mask = fixed_edge_mask.to(device=device)
    if fixed_edge_mask.dtype != torch.bool:
      raise TypeError('fixed_edge_mask must be boolean')
    if fixed_edge_mask.ndim == 1:
      if fixed_edge_mask.shape[0] != input_edges:
        raise ValueError('shared fixed_edge_mask must have shape [E]')
      fixed_edge_mask = fixed_edge_mask[None].expand(batch_size, -1)
    elif fixed_edge_mask.ndim == 2:
      if fixed_edge_mask.shape != (batch_size, input_edges):
        raise ValueError('batched fixed_edge_mask must have shape [B,E]')
    else:
      raise ValueError(
        'fixed_edge_mask must have shared [E] or batched [B,E] shape')

  canonical_edges = torch.stack((
    fixed_edge_index.min(dim=-1).values,
    fixed_edge_index.max(dim=-1).values), dim=-1).detach()
  input_mask_cpu = fixed_edge_mask.detach().cpu()
  edges_cpu = canonical_edges.cpu()
  active_cpu = active_mask.detach().cpu()
  valid_edges_per_batch = []

  for batch_index in range(batch_size):
    parent = list(range(sequence_length))
    component_size = [1 if bool(active_cpu[batch_index, node]) else 0
                      for node in range(sequence_length)]

    def find(node):
      while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
      return node

    seen = set()
    valid_edges = []
    for edge_slot in input_mask_cpu[batch_index].nonzero(
        as_tuple=False).flatten().tolist():
      left, right = edges_cpu[batch_index, edge_slot].tolist()
      if (left < 0 or right < 0
          or left >= sequence_length or right >= sequence_length):
        raise ValueError(
          f'fixed edge ({left},{right}) is out of range for L='
          f'{sequence_length} in batch item {batch_index}')
      if left == right:
        raise ValueError(
          f'fixed topology contains self edge ({left},{right}) '
          f'in batch item {batch_index}')
      edge = (left, right)
      if edge in seen:
        raise ValueError(
          f'fixed topology contains duplicate edge {edge} '
          f'in batch item {batch_index}')
      seen.add(edge)
      if (not bool(active_cpu[batch_index, left])
          or not bool(active_cpu[batch_index, right])):
        raise ValueError(
          f'fixed edge {edge} has an endpoint outside active_mask '
          f'in batch item {batch_index}')

      left_root, right_root = find(left), find(right)
      if left_root == right_root:
        raise ValueError(
          f'fixed topology contains a cycle at edge {edge} '
          f'in batch item {batch_index}')
      merged_size = (
        component_size[left_root] + component_size[right_root])
      if (component_size_cap > 0
          and merged_size > component_size_cap):
        raise ValueError(
          f'fixed topology component would contain {merged_size} nodes, '
          f'exceeding component_size_cap={component_size_cap} '
          f'in batch item {batch_index}')
      if component_size[left_root] < component_size[right_root]:
        left_root, right_root = right_root, left_root
      parent[right_root] = left_root
      component_size[left_root] = merged_size
      valid_edges.append(edge)
    valid_edges_per_batch.append(valid_edges)

  max_edges = max(sequence_length - 1, 0)
  edge_index = torch.zeros(
    batch_size, max_edges, 2, dtype=torch.long, device=device)
  edge_mask = torch.zeros(
    batch_size, max_edges, dtype=torch.bool, device=device)
  for batch_index, valid_edges in enumerate(valid_edges_per_batch):
    if not valid_edges:
      continue
    if len(valid_edges) > max_edges:
      # A simple acyclic graph cannot reach this branch, but retain an explicit
      # guard in case validation changes.
      raise ValueError('fixed topology has more than L-1 valid edges')
    count = len(valid_edges)
    edge_index[batch_index, :count] = torch.tensor(
      valid_edges, dtype=torch.long, device=device)
    edge_mask[batch_index, :count] = True
  return edge_index.detach(), edge_mask.detach()


class ContextualCouplingForestHead(nn.Module):
  """Construct a compact globally-normalizable denoising forest.

  Args:
    hidden_size: Width of the backbone hidden states.
    vocab_size: Size of the unary vocabulary.
    top_k: Number of explicit token states retained at each position.
    rank: Nonnegative rank of each explicit token-pair factor.
    time_embed_dim: Width of the small scalar-time embedding.
    topology_dim: Width used for sparse proposals and edge scoring.
    local_window: Propose all edges within this positional radius.
    num_anchor_slots: Number of learned contextual anchor slots.
    contextual_neighbors: Anchor slots selected by each active node.
    component_size_cap: Maximum nodes in a selected tree; nonpositive disables.
    topology_mode: ``dynamic`` for learned forests or ``fixed`` for chains.
    factor_mode: ``dynamic`` for context/time FiLM or ``fixed`` token factors.
    independent_mode: Replace every pair factor by one while retaining all
      architecture parameters. Pair-factor parameters are inactive in this
      mode, so this is an architecture-count/no-edge control rather than an
      active-capacity-matched unary adapter.
    min_edge_score: Optional threshold below which dynamic edges are omitted.

  The structured part has O(VR + H(D+R) + DA) parameters.  Given existing
  unary logits, its principal work is O(BLV) for top-K, O(BLA) for contextual
  proposals, O(BLKR) for candidate features, O(P log P) for hard forest
  construction, and O(B|E|KR) per low-rank tree-message pass.
  """

  def __init__(
      self,
      hidden_size: int,
      vocab_size: int,
      top_k: int = 64,
      rank: int = 16,
      time_embed_dim: int = 64,
      topology_dim: int = 128,
      local_window: int = 2,
      num_anchor_slots: int = 16,
      contextual_neighbors: int = 4,
      component_size_cap: int = 32,
      topology_mode: str = 'dynamic',
      factor_mode: str = 'dynamic',
      independent_mode: bool = False,
      min_edge_score: Optional[float] = None):
    super().__init__()
    if hidden_size < 1 or vocab_size < 2:
      raise ValueError('hidden_size must be positive and vocab_size >= 2')
    if not 1 <= top_k <= vocab_size:
      raise ValueError('top_k must lie in [1, vocab_size]')
    if rank < 1 or topology_dim < 1:
      raise ValueError('rank and topology_dim must be positive')

    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.top_k = top_k
    self.rank = rank
    self.component_size_cap = component_size_cap
    self.topology_mode = _check_mode('topology_mode', topology_mode)
    self.factor_mode = _check_mode('factor_mode', factor_mode)
    self.independent_mode = bool(independent_mode)
    self.min_edge_score = min_edge_score

    self.hidden_norm = nn.LayerNorm(hidden_size)
    self.time_embedding = ScalarTimestepEmbedding(time_embed_dim)
    self.topology_hidden_projection = nn.Linear(
      hidden_size, topology_dim)
    self.topology_time_projection = nn.Linear(
      time_embed_dim, topology_dim, bias=False)
    self.edge_proposer = SparseEdgeProposer(
      topology_dim=topology_dim,
      local_window=local_window,
      num_anchor_slots=num_anchor_slots,
      contextual_neighbors=contextual_neighbors)

    self.token_factor_embedding = nn.Embedding(vocab_size, rank)
    self.factor_hidden_projection = nn.Linear(hidden_size, 2 * rank)
    self.factor_time_projection = nn.Linear(
      time_embed_dim, 2 * rank, bias=False)
    inverse_softplus_one = math.log(math.exp(1.0) - 1.0)
    nn.init.normal_(
      self.token_factor_embedding.weight,
      mean=inverse_softplus_one, std=0.01)
    # Start close to the neutral factor without making the dynamic mode
    # context-blind on its first forward pass.
    nn.init.normal_(self.factor_hidden_projection.weight, std=1e-3)
    nn.init.zeros_(self.factor_hidden_projection.bias)
    nn.init.normal_(self.factor_time_projection.weight, std=2e-3)

  @property
  def parameter_count(self) -> int:
    """Number of trainable parameters, unchanged in independent mode."""
    return sum(parameter.numel() for parameter in self.parameters()
               if parameter.requires_grad)

  def complexity_summary(self, batch_size: int, sequence_length: int,
                         selected_edges: int) -> Dict[str, int]:
    """Return leading tensor-operation sizes for experiment logging."""
    return {
      'unary_values_scanned': (
        batch_size * sequence_length * self.vocab_size),
      'candidate_factor_values': (
        batch_size * sequence_length * self.top_k * self.rank),
      'message_factor_products': (
        batch_size * selected_edges * self.top_k * self.rank),
    }

  def _candidate_lattice(
      self, unary_logits: torch.Tensor
      ) -> Tuple[torch.Tensor, ...]:
    # Accumulate half/bfloat16 unaries in FP32 but preserve FP64 in exactness
    # tests.  log(-expm1(delta)) is stable when the retained mass is near one.
    if unary_logits.dtype in (torch.float16, torch.bfloat16):
      work_logits = unary_logits.float()
    else:
      work_logits = unary_logits
    candidate_logits, candidate_ids = work_logits.topk(
      self.top_k, dim=-1)
    total_log_mass = torch.logsumexp(work_logits, dim=-1)
    kept_log_mass = torch.logsumexp(candidate_logits, dim=-1)

    if self.top_k < self.vocab_size:
      log_kept_fraction = kept_log_mass - total_log_mass
      kept_fraction = torch.exp(log_kept_fraction)
      retained_mass = kept_fraction.clamp(0.0, 1.0)
      residual_log_mass = (
        total_log_mass + torch.log(-torch.expm1(log_kept_fraction)))

      # If total and retained log-mass round to the same value, logdiffexp
      # becomes -inf despite a nonempty tail.  Recompute only those rare rows
      # by explicitly masking K candidates; this preserves support without a
      # vocabulary-sized allocation on the ordinary path.
      collapsed = ~torch.isfinite(residual_log_mass)
      if bool(collapsed.any().item()):
        collapsed_logits = work_logits[collapsed].clone()
        collapsed_ids = candidate_ids[collapsed]
        collapsed_logits.scatter_(-1, collapsed_ids, -torch.inf)
        exact_tail = torch.logsumexp(collapsed_logits, dim=-1)
        residual_log_mass = residual_log_mass.masked_scatter(
          collapsed, exact_tail)
      residual_valid = torch.ones_like(
        residual_log_mass, dtype=torch.bool)
    else:
      retained_mass = torch.ones_like(total_log_mass)
      residual_log_mass = torch.full_like(total_log_mass, -torch.inf)
      residual_valid = torch.zeros_like(
        residual_log_mass, dtype=torch.bool)

    unary_log_potentials = torch.cat(
      (candidate_logits, residual_log_mass[:, :, None]), dim=-1)
    explicit_valid = torch.ones_like(candidate_ids, dtype=torch.bool)
    candidate_state_mask = torch.cat(
      (explicit_valid, residual_valid[:, :, None]), dim=-1)
    return (
      candidate_ids, unary_log_potentials, candidate_state_mask,
      retained_mass, residual_log_mass)

  def _selected_edges(
      self,
      topology_context: torch.Tensor,
      active_mask: torch.Tensor,
      proposal_edge_index: torch.Tensor,
      proposal_edge_mask: torch.Tensor,
      proposal_scores: torch.Tensor,
      topology_mode: str,
      fixed_edge_index: Optional[torch.Tensor] = None,
      fixed_edge_mask: Optional[torch.Tensor] = None,
      ) -> Tuple[torch.Tensor, ...]:
    batch_size, sequence_length = active_mask.shape
    max_edges = max(sequence_length - 1, 0)
    if topology_mode == 'fixed':
      if fixed_edge_index is None:
        edge_index, edge_mask = _fixed_chain_edges(
          active_mask, self.component_size_cap)
      else:
        edge_index, edge_mask = _validated_fixed_edges(
          fixed_edge_index=fixed_edge_index,
          fixed_edge_mask=fixed_edge_mask,
          active_mask=active_mask,
          component_size_cap=self.component_size_cap)
      edge_scores = self.edge_proposer.score_edges(
        topology_context, edge_index, edge_mask)
      edge_scores = edge_scores.masked_fill(~edge_mask, 0.0)
      return edge_index, edge_mask, edge_scores

    selected_slots, edge_mask = _bounded_kruskal_indices(
      proposal_edge_index=proposal_edge_index,
      proposal_scores=proposal_scores,
      proposal_edge_mask=proposal_edge_mask,
      active_mask=active_mask,
      component_size_cap=self.component_size_cap,
      min_edge_score=self.min_edge_score)
    if max_edges == 0 or proposal_edge_index.shape[1] == 0:
      edge_index = torch.zeros(
        batch_size, max_edges, 2, dtype=torch.long,
        device=active_mask.device)
      edge_scores = topology_context.new_zeros(batch_size, max_edges)
      return edge_index, edge_mask, edge_scores

    gather_edges = selected_slots[:, :, None].expand(-1, -1, 2)
    edge_index = torch.gather(
      proposal_edge_index, 1, gather_edges).detach()
    edge_scores = torch.gather(proposal_scores, 1, selected_slots)
    edge_scores = edge_scores.masked_fill(~edge_mask, 0.0)
    return edge_index, edge_mask, edge_scores

  def _node_candidate_factors(
      self,
      normalized_hidden: torch.Tensor,
      time_features: torch.Tensor,
      candidate_ids: torch.Tensor,
      factor_mode: str) -> torch.Tensor:
    raw_token_factors = self.token_factor_embedding(candidate_ids)
    if factor_mode == 'dynamic':
      film = self.factor_hidden_projection(normalized_hidden)
      film = film + self.factor_time_projection(time_features)[:, None, :]
      shift, scale = film.chunk(2, dim=-1)
      raw_token_factors = (
        raw_token_factors * (1.0 + scale.tanh()[:, :, None, :])
        + shift[:, :, None, :])
    return F.softplus(raw_token_factors) + 1e-6

  @staticmethod
  def _gather_candidate_factors(
      node_factors: torch.Tensor,
      edge_nodes: torch.Tensor) -> torch.Tensor:
    gather_index = edge_nodes[:, :, None, None].expand(
      -1, -1, node_factors.shape[2], node_factors.shape[3])
    return torch.gather(node_factors, 1, gather_index)

  def forward(
      self,
      hidden_states: torch.Tensor,
      unary_logits: torch.Tensor,
      timestep: torch.Tensor,
      active_mask: Optional[torch.Tensor] = None,
      *,
      topology_mode: Optional[str] = None,
      factor_mode: Optional[str] = None,
      independent_mode: Optional[bool] = None,
      fixed_edge_index: Optional[torch.Tensor] = None,
      fixed_edge_mask: Optional[torch.Tensor] = None,
      ) -> StructuredDecoderOutput:
    """Build candidate states, a hard forest, and positive pair factors.

    Args:
      hidden_states: Backbone states ``[B,L,H]``.
      unary_logits: Backbone token logits ``[B,L,V]``.  No clean targets are
        accepted, so candidate construction is target-independent by design.
      timestep: Scalar diffusion time as a scalar, ``[B]``, or ``[B,1]``.
      active_mask: Boolean ``[B,L]`` mask for currently unknown positions.
        All positions are active when omitted.
      topology_mode: Optional per-call ``fixed``/``dynamic`` ablation.
      factor_mode: Optional per-call ``fixed``/``dynamic`` ablation.
      independent_mode: Optional per-call neutral-factor ablation.  It changes
        no architecture parameters, but makes pair-factor parameters inactive;
        it is an architecture-count/no-edge control.
      fixed_edge_index: Optional static forest, shared as ``[E,2]`` or batched
        as ``[B,E,2]``.  Supplying it is an explicit static-topology override;
        edges are validated, canonicalized, detached, and padded to ``L-1``.
      fixed_edge_mask: Optional shared ``[E]`` or batched ``[B,E]`` validity
        mask.  It is only valid together with ``fixed_edge_index``.
    """
    if hidden_states.ndim != 3 or unary_logits.ndim != 3:
      raise ValueError('hidden_states and unary_logits must both be rank 3')
    if hidden_states.shape[:2] != unary_logits.shape[:2]:
      raise ValueError('hidden_states and unary_logits must agree on [B,L]')
    if hidden_states.shape[-1] != self.hidden_size:
      raise ValueError(
        f'expected hidden width {self.hidden_size}, '
        f'got {hidden_states.shape[-1]}')
    if unary_logits.shape[-1] != self.vocab_size:
      raise ValueError(
        f'expected vocabulary {self.vocab_size}, '
        f'got {unary_logits.shape[-1]}')
    batch_size, sequence_length, _ = hidden_states.shape
    if active_mask is None:
      active_mask = torch.ones(
        batch_size, sequence_length, dtype=torch.bool,
        device=hidden_states.device)
    elif (active_mask.shape != (batch_size, sequence_length)
          or active_mask.dtype != torch.bool):
      raise ValueError('active_mask must be boolean with shape [B,L]')

    requested_topology_mode = topology_mode
    if fixed_edge_mask is not None and fixed_edge_index is None:
      raise ValueError('fixed_edge_mask requires fixed_edge_index')
    if fixed_edge_index is not None:
      if (requested_topology_mode is not None
          and requested_topology_mode != 'fixed'):
        raise ValueError(
          'fixed_edge_index is incompatible with explicit '
          'topology_mode="dynamic"')
      topology_mode = 'fixed'
    else:
      topology_mode = topology_mode or self.topology_mode
    topology_mode = _check_mode('topology_mode', topology_mode)
    factor_mode = _check_mode(
      'factor_mode', factor_mode or self.factor_mode)
    if independent_mode is None:
      independent_mode = self.independent_mode

    (
      candidate_ids,
      unary_log_potentials,
      candidate_state_mask,
      retained_mass,
      residual_log_mass,
    ) = self._candidate_lattice(unary_logits)

    normalized_hidden = self.hidden_norm(hidden_states)
    time_features = self.time_embedding(timestep, batch_size)
    topology_context = self.topology_hidden_projection(normalized_hidden)
    topology_context = F.silu(
      topology_context
      + self.topology_time_projection(time_features)[:, None, :])
    (
      proposal_edge_index,
      proposal_edge_mask,
      proposal_scores,
      anchor_logits,
      anchor_indices,
      slot_logits,
    ) = self.edge_proposer(topology_context, active_mask)
    edge_index, edge_mask, edge_scores = self._selected_edges(
      topology_context=topology_context,
      active_mask=active_mask,
      proposal_edge_index=proposal_edge_index,
      proposal_edge_mask=proposal_edge_mask,
      proposal_scores=proposal_scores,
      topology_mode=topology_mode,
      fixed_edge_index=fixed_edge_index,
      fixed_edge_mask=fixed_edge_mask)

    node_factors = self._node_candidate_factors(
      normalized_hidden=normalized_hidden,
      time_features=time_features,
      candidate_ids=candidate_ids,
      factor_mode=factor_mode)
    pair_left = self._gather_candidate_factors(
      node_factors, edge_index[:, :, 0])
    pair_right = self._gather_candidate_factors(
      node_factors, edge_index[:, :, 1])
    neutral = pair_left.new_full(
      (), 1.0 / math.sqrt(self.rank))
    if independent_mode:
      pair_left = torch.ones_like(pair_left) * neutral
      pair_right = torch.ones_like(pair_right) * neutral
    else:
      pair_left = pair_left / math.sqrt(self.rank)
      pair_right = pair_right / math.sqrt(self.rank)
      pair_left = torch.where(
        edge_mask[:, :, None, None], pair_left,
        torch.ones_like(pair_left) * neutral)
      pair_right = torch.where(
        edge_mask[:, :, None, None], pair_right,
        torch.ones_like(pair_right) * neutral)

    return StructuredDecoderOutput(
      candidate_ids=candidate_ids,
      unary_log_potentials=unary_log_potentials,
      candidate_state_mask=candidate_state_mask,
      retained_mass=retained_mass,
      residual_log_mass=residual_log_mass,
      proposal_edge_index=proposal_edge_index,
      proposal_edge_mask=proposal_edge_mask,
      proposal_scores=proposal_scores,
      anchor_logits=anchor_logits,
      anchor_indices=anchor_indices,
      slot_logits=slot_logits,
      edge_index=edge_index,
      edge_mask=edge_mask,
      edge_scores=edge_scores,
      pair_left_factors=pair_left,
      pair_right_factors=pair_right,
      topology_mode=topology_mode,
      factor_mode=factor_mode,
      independent_mode=bool(independent_mode))


def _shape_self_test() -> None:
  """Torch-only smoke test, runnable as ``python structured_decoder.py``."""
  torch.manual_seed(7)
  batch_size, sequence_length = 2, 7
  hidden_size, vocab_size, top_k, rank = 24, 17, 5, 4
  head = ContextualCouplingForestHead(
    hidden_size=hidden_size,
    vocab_size=vocab_size,
    top_k=top_k,
    rank=rank,
    time_embed_dim=12,
    topology_dim=16,
    local_window=2,
    num_anchor_slots=4,
    contextual_neighbors=2,
    component_size_cap=3)
  hidden = torch.randn(batch_size, sequence_length, hidden_size,
                       requires_grad=True)
  logits = torch.randn(batch_size, sequence_length, vocab_size,
                       requires_grad=True)
  active = torch.tensor([
    [True, True, True, True, True, False, False],
    [True, False, True, True, False, True, True],
  ])
  output = head(hidden, logits, torch.tensor([0.2, 0.8]), active)

  assert output.candidate_ids.shape == (
    batch_size, sequence_length, top_k)
  assert output.unary_log_potentials.shape == (
    batch_size, sequence_length, top_k + 1)
  assert output.edge_index.shape == (
    batch_size, sequence_length - 1, 2)
  assert output.pair_left_factors.shape == (
    batch_size, sequence_length - 1, top_k, rank)
  dense = output.materialize_pair_factors()
  assert dense.shape == (
    batch_size, sequence_length - 1, top_k + 1, top_k + 1)
  assert bool((dense > 0).all())
  assert torch.allclose(dense[..., -1, :], torch.ones_like(dense[..., -1, :]))
  assert torch.allclose(dense[..., :, -1], torch.ones_like(dense[..., :, -1]))

  # The retained candidates are exactly unary top-K, with no target input.
  assert torch.equal(output.candidate_ids, logits.topk(top_k, dim=-1).indices)
  tail = logits.detach().clone()
  tail.scatter_(-1, output.candidate_ids, -torch.inf)
  expected_tail = torch.logsumexp(tail.float(), dim=-1)
  assert torch.allclose(
    output.residual_log_mass, expected_tail, atol=2e-5, rtol=2e-5)

  independent = head(
    hidden, logits, torch.tensor([0.2, 0.8]), active,
    topology_mode='fixed', factor_mode='fixed', independent_mode=True)
  independent_dense = independent.materialize_pair_factors()
  assert torch.allclose(
    independent_dense, torch.ones_like(independent_dense), atol=1e-6)

  # A supplied corpus-level forest overrides the configured dynamic topology.
  # Reversed input edges are canonicalized, and the topology is shared here
  # only because all referenced endpoints are active in both examples.
  shared_static_edges = torch.tensor([[2, 0], [3, 2]])
  shared_static = head(
    hidden, logits, torch.tensor([0.2, 0.8]), active,
    fixed_edge_index=shared_static_edges)
  assert shared_static.topology_mode == 'fixed'
  assert shared_static.edge_mask.sum(dim=1).tolist() == [2, 2]
  expected_shared = torch.tensor([[0, 2], [2, 3]])
  assert torch.equal(shared_static.edge_index[0, :2], expected_shared)
  assert torch.equal(shared_static.edge_index[1, :2], expected_shared)
  assert not shared_static.edge_index.requires_grad

  batched_static_edges = torch.tensor([
    [[1, 0], [2, 1], [-1, -1]],
    [[2, 0], [3, 2], [6, 5]],
  ])
  batched_static_mask = torch.tensor([
    [True, True, False],
    [True, True, True],
  ])
  batched_static = head(
    hidden, logits, torch.tensor([0.2, 0.8]), active,
    topology_mode='fixed', fixed_edge_index=batched_static_edges,
    fixed_edge_mask=batched_static_mask)
  assert batched_static.edge_mask.sum(dim=1).tolist() == [2, 3]

  def assert_static_rejected(edges, message, mask=None, mode='fixed'):
    try:
      head(
        hidden, logits, torch.tensor([0.2, 0.8]), active,
        topology_mode=mode, fixed_edge_index=edges,
        fixed_edge_mask=mask)
    except (TypeError, ValueError) as error:
      assert message in str(error), str(error)
    else:
      raise AssertionError(f'expected invalid static topology: {message}')

  assert_static_rejected(torch.tensor([[0, 0]]), 'self edge')
  assert_static_rejected(
    torch.tensor([[0, 2], [2, 0]]), 'duplicate edge')
  assert_static_rejected(
    torch.tensor([[0, 1], [1, 2], [2, 0]]), 'cycle')
  assert_static_rejected(torch.tensor([[0, 7]]), 'out of range')
  assert_static_rejected(
    torch.tensor([[0, 5]]), 'outside active_mask')
  assert_static_rejected(
    torch.tensor([[0, 1], [1, 2], [2, 3]]),
    'exceeding component_size_cap')
  assert_static_rejected(
    shared_static_edges, 'incompatible', mode='dynamic')
  try:
    head(
      hidden, logits, torch.tensor([0.2, 0.8]), active,
      fixed_edge_mask=torch.tensor([True]))
  except ValueError as error:
    assert 'requires fixed_edge_index' in str(error)
  else:
    raise AssertionError('fixed_edge_mask without edges must fail')

  finite_scores = output.edge_scores.masked_select(output.edge_mask)
  loss = (output.unary_log_potentials.sum()
          + output.pair_left_factors.sum()
          + output.pair_right_factors.sum()
          + finite_scores.sum())
  loss.backward()
  assert hidden.grad is not None and logits.grad is not None
  assert float(hidden.grad.abs().sum()) > 0.0
  assert float(logits.grad.abs().sum()) > 0.0
  print('structured decoder shape self-test: OK')
  print(f'trainable parameters: {head.parameter_count:,}')
  print(head.complexity_summary(batch_size, sequence_length,
                                int(output.edge_mask.sum())))


if __name__ == '__main__':
  _shape_self_test()
