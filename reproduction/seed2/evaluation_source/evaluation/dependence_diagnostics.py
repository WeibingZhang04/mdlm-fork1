"""Separate singleton prediction gains from useful forest dependence.

For a forest with calibrated beliefs ``b_i`` and ``b_ij``, its probability is
``prod_i b_i(x_i) prod_ij b_ij(x_i,x_j)/(b_i(x_i)b_j(x_j))``.  Replacing each
edge belief by ``(1-lambda) b_i b_j + lambda b_ij`` leaves every singleton
marginal unchanged.  This gives an exact dependence intervention, rather
than a temperature on the original pair potentials (which changes marginals).

All scores are per example and in nats, before dividing by active token
count.  Positive edge contributions mean the dependence helps the observed
token pair.  Their held-out expectation need not be positive: it is not the
model's mutual information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from models.structured_decoder import StructuredDecoderOutput
from structured_objective import (
  StructuredInference,
  compressed_states_for_tokens,
  factorized_token_log_probability,
  infer_structured_distribution,
)
import structured_utils


Strength = Union[float, torch.Tensor]


def _edge_strength(strength: Strength, reference: torch.Tensor) -> torch.Tensor:
  value = torch.as_tensor(
    strength, dtype=reference.dtype, device=reference.device)
  if not bool((torch.isfinite(value) & (value >= 0) & (value <= 1)).all()):
    raise ValueError('dependence strength must lie in [0,1]')
  try:
    return torch.broadcast_to(value, reference.shape)
  except RuntimeError as error:
    raise ValueError('strength must broadcast to [batch, edges]') from error


@dataclass(frozen=True)
class DependenceDecomposition:
  """Observed singleton and edge contributions, with padded entries zero.

  ``node_log_probability`` includes the residual decoder correction for
  full-vocabulary tokens.  This correction cancels from the edge ratio, so
  ``edge_log_dependence`` can be computed on compressed candidate states.
  An inactive position contributes zero.  At a zero-probability singleton,
  the edge ratio is defined as one: the joint is already zero through the
  singleton term and the ratio itself is unidentifiable off support.
  """

  node_log_probability: torch.Tensor       # [B,L]
  edge_log_dependence: torch.Tensor        # [B,E]
  backbone_log_probability: torch.Tensor  # [B]
  active_token_count: torch.Tensor        # [B]

  @property
  def marginal_log_probability(self) -> torch.Tensor:
    return self.node_log_probability.sum(dim=-1)

  @property
  def dependence_log_probability(self) -> torch.Tensor:
    return self.edge_log_dependence.sum(dim=-1)

  @property
  def joint_log_probability(self) -> torch.Tensor:
    return self.marginal_log_probability + self.dependence_log_probability

  @property
  def marginal_gain(self) -> torch.Tensor:
    """Improvement due to singleton predictions, relative to the backbone."""
    return self.marginal_log_probability - self.backbone_log_probability

  @property
  def joint_gain(self) -> torch.Tensor:
    return self.joint_log_probability - self.backbone_log_probability

  def shrinkage_log_probability(self, strength: Strength) -> torch.Tensor:
    """Score exact marginal-preserving shrinkage, with lambda in [0,1].

    A scalar uses one lambda for all edges; tensors broadcast to ``[B,E]``.
    Lambda zero is the independent product of the model's own marginals;
    lambda one recovers the original joint distribution.  Different edges
    may have different strengths and the result is still normalized.
    """
    value = _edge_strength(strength, self.edge_log_dependence)
    edge_terms = torch.logaddexp(
      torch.log1p(-value), value.log() + self.edge_log_dependence)
    return self.marginal_log_probability + edge_terms.sum(dim=-1)


def _validate_context(
    output: StructuredDecoderOutput,
    active_mask: torch.Tensor,
    inference: Optional[StructuredInference] = None) -> torch.Tensor:
  shape = output.candidate_ids.shape[:2]
  if active_mask.shape != shape or active_mask.dtype != torch.bool:
    raise ValueError('active_mask must be boolean with shape [B,L]')
  active_mask = active_mask.to(output.candidate_ids.device)
  clamped = torch.where(
    active_mask, torch.full_like(active_mask, -1, dtype=torch.long),
    torch.zeros_like(active_mask, dtype=torch.long))
  if inference is not None and not torch.equal(inference.clamped_states, clamped):
    raise ValueError('inference was computed for a different active_mask')
  # The production contract isolates inactive nodes.  Reject a conflicting
  # graph rather than accidentally scoring a conditional on arbitrary zeros.
  if bool(output.edge_mask.any()):
    edges = output.edge_index.clamp_min(0)
    connected_active = torch.gather(
      active_mask, 1, edges.reshape(shape[0], -1)).reshape_as(edges).all(-1)
    if bool((output.edge_mask & ~connected_active).any()):
      raise ValueError('active forest edges must connect active nodes')
  return active_mask


def _low_rank_observed_edge_dependence(
    output: StructuredDecoderOutput,
    states: torch.Tensor,
    inference: StructuredInference) -> torch.Tensor:
  """Recover selected edge beliefs from messages, without E*K*K storage."""
  (nodes, left_logs, right_logs, edges, _, topologies
   ) = structured_utils._validate_low_rank_inputs(
    output.unary_log_potentials,
    output.pair_left_factors, output.pair_right_factors,
    output.edge_index, output.edge_mask, output.candidate_state_mask,
    inference.clamped_states, None, None)
  batch_contributions = []
  explicit_count = output.candidate_ids.shape[-1]
  for batch, topology in enumerate(topologies):
    _, node_logs, messages = structured_utils._single_low_rank_sum_product(
      nodes[batch], left_logs[batch], right_logs[batch], edges[batch], topology)
    contributions = [nodes.new_zeros(()) for _ in range(edges.shape[1])]
    for child in topology.order:
      parent = topology.parent[child]
      if parent < 0:
        continue
      edge = topology.parent_edge[child]
      child_state, parent_state = states[batch, child], states[batch, parent]
      child_log_p = node_logs[child, child_state]
      parent_log_p = node_logs[parent, parent_state]
      if not bool(torch.isfinite(child_log_p) & torch.isfinite(parent_log_p)):
        continue
      # The subtree cavity includes every incoming message except parent's.
      # Its contraction with the pair factor is m(child -> parent), so their
      # difference is log p(x_child | x_parent).  Subtract log p(x_child)
      # to get log b_ij/(b_i*b_j), independently of root orientation.
      cavity = nodes[batch, child, child_state]
      for neighbor, _, _ in topology.adjacency[child]:
        if neighbor != parent:
          cavity = cavity + messages[(neighbor, child)][child_state]
      left, right = (int(value) for value in edges[batch, edge])
      left_state, right_state = states[batch, left], states[batch, right]
      if bool((left_state == explicit_count) | (right_state == explicit_count)):
        pair_log = nodes.new_zeros(())
      else:
        pair_log = torch.logsumexp(
          left_logs[batch, edge, left_state]
          + right_logs[batch, edge, right_state], dim=-1)
      contributions[edge] = (
        cavity + pair_log
        - messages[(child, parent)][parent_state] - child_log_p)
    batch_contributions.append(
      torch.stack(contributions) if contributions else nodes.new_empty(0))
  return torch.stack(batch_contributions)


def decompose_structured_log_probability(
    output: StructuredDecoderOutput,
    unary_logits: torch.Tensor,
    token_ids: torch.Tensor,
    active_mask: torch.Tensor,
    inference: Optional[StructuredInference] = None) -> DependenceDecomposition:
  """Score singleton prediction gains and observed dependence separately.

  Reuse ``inference`` from the same output when available.  Dense inference
  already has edge beliefs.  Low-rank inference recomputes messages once in
  O(E K R), then reads only the observed entry of each edge belief.  It never
  constructs dense pair factors or dense edge marginals.  Reuse the returned
  decomposition to score any number of shrinkage strengths without inference.
  """
  active_mask = _validate_context(output, active_mask, inference)
  if token_ids.shape != active_mask.shape or unary_logits.shape[:2] != token_ids.shape:
    raise ValueError('tokens and logits must have leading shape [B,L]')
  # Context tokens do not enter the likelihood and need not be candidates.
  safe_tokens = torch.where(active_mask, token_ids, output.candidate_ids[..., 0])
  states = compressed_states_for_tokens(output, safe_tokens)
  inference = inference or infer_structured_distribution(output, active_mask)
  node_logs = inference.marginals.node_log_marginals
  observed_node_logs = torch.gather(node_logs, -1, states.unsqueeze(-1)).squeeze(-1)

  if isinstance(inference.marginals, structured_utils.ForestMarginals):
    # Dense beliefs provide an independent reference and avoid numeric
    # differences between dense and endpoint inference in small debug cases.
    edges = output.edge_index.clamp_min(0)
    left_states = torch.gather(states, 1, edges[..., 0])
    right_states = torch.gather(states, 1, edges[..., 1])
    flat_index = left_states * output.num_candidate_states + right_states
    observed_edge_logs = torch.gather(
      inference.marginals.edge_log_marginals.flatten(-2),
      -1, flat_index.unsqueeze(-1)).squeeze(-1)
    left_logs = torch.gather(observed_node_logs, 1, edges[..., 0])
    right_logs = torch.gather(observed_node_logs, 1, edges[..., 1])
    valid = output.edge_mask & torch.isfinite(left_logs) & torch.isfinite(right_logs)
    edge_dependence = torch.where(
      valid, observed_edge_logs - left_logs - right_logs,
      torch.zeros_like(observed_edge_logs))
  else:
    edge_dependence = _low_rank_observed_edge_dependence(output, states, inference)

  uses_residual = active_mask & states.eq(output.num_candidate_states - 1)
  if bool(uses_residual.any()):
    tail_logs = output.residual_log_probs(unary_logits)
    tail_correction = torch.gather(tail_logs, -1, safe_tokens.unsqueeze(-1)).squeeze(-1)
    observed_node_logs = observed_node_logs + torch.where(
      uses_residual, tail_correction, torch.zeros_like(tail_correction))
  per_node = torch.where(
    active_mask, observed_node_logs, torch.zeros_like(observed_node_logs))
  return DependenceDecomposition(
    node_log_probability=per_node,
    edge_log_dependence=edge_dependence,
    backbone_log_probability=factorized_token_log_probability(
      unary_logits, safe_tokens, active_mask),
    active_token_count=active_mask.sum(-1))


@torch.no_grad()
def sample_shrunk_structured_tokens(
    output: StructuredDecoderOutput,
    unary_logits: torch.Tensor,
    active_mask: torch.Tensor,
    strength: Strength,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None) -> torch.Tensor:
  """Draw exact full-vocabulary samples while varying only dependence.

  Roots follow their original marginals.  Each child uses the mixture
  ``(1-lambda) b_child + lambda p(child | parent)``.  The resulting forest
  has edge beliefs ``(1-lambda)b_parent*b_child + lambda*b_parent,child``
  and preserves all original node marginals.  Strength may be scalar or
  broadcast to ``[B,E]``.  The returned shape is ``[B,S,L]``; inactive
  positions contain their first candidate, as in the existing joint sampler.

  The sampler uses O(E K R) message work and O(S E K R) conditional-row
  work, without materializing full K-by-K pair factors.  Residual states
  expand through the original tail decoder, identically at every strength.
  """
  if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
    raise ValueError('num_samples must be a positive integer')
  active_mask = _validate_context(output, active_mask)
  if unary_logits.shape[:2] != active_mask.shape:
    raise ValueError('logits must have leading shape [B,L]')
  strengths = _edge_strength(
    strength, output.unary_log_potentials.new_zeros(output.edge_mask.shape))
  clamped = torch.where(
    active_mask, torch.full_like(active_mask, -1, dtype=torch.long),
    torch.zeros_like(active_mask, dtype=torch.long))
  (nodes, left_logs, right_logs, edges, _, topologies
   ) = structured_utils._validate_low_rank_inputs(
    output.unary_log_potentials,
    output.pair_left_factors, output.pair_right_factors,
    output.edge_index, output.edge_mask, output.candidate_state_mask,
    clamped, None, None)
  batches = []
  for batch, topology in enumerate(topologies):
    _, node_logs, messages = structured_utils._single_low_rank_sum_product(
      nodes[batch], left_logs[batch], right_logs[batch], edges[batch], topology)
    states = torch.empty(
      num_samples, nodes.shape[1], dtype=torch.long, device=nodes.device)
    for root in topology.roots:
      states[:, root] = structured_utils._sample_rows(
        node_logs[root].expand(num_samples, -1), generator)
    for child in topology.order:
      parent = topology.parent[child]
      if parent < 0:
        continue
      edge = topology.parent_edge[child]
      cavity = nodes[batch, child]
      for neighbor, _, _ in topology.adjacency[child]:
        if neighbor != parent:
          cavity = cavity + messages[(neighbor, child)]
      if int(edges[batch, edge, 0]) == parent:
        source, target = left_logs[batch, edge], right_logs[batch, edge]
      else:
        source, target = right_logs[batch, edge], left_logs[batch, edge]
      conditional = cavity.unsqueeze(0) + structured_utils._low_rank_pair_rows(
        states[:, parent], source, target)
      conditional = conditional - torch.logsumexp(conditional, dim=-1, keepdim=True)
      value = strengths[batch, edge]
      mixture = torch.logaddexp(
        torch.log1p(-value) + node_logs[child].unsqueeze(0),
        value.log() + conditional)
      states[:, child] = structured_utils._sample_rows(mixture, generator)
    batches.append(states)
  states = torch.stack(batches)
  explicit_count = output.candidate_ids.shape[-1]
  candidates = output.candidate_ids[:, None].expand(-1, num_samples, -1, -1)
  tokens = torch.gather(
    candidates, -1, states.clamp_max(explicit_count - 1).unsqueeze(-1)).squeeze(-1)
  uses_residual = states.eq(explicit_count)
  # Only expand tails for nodes where a residual was actually drawn.  In
  # particular, never normalize the empty tail of a full-vocabulary node.
  if bool(uses_residual.any()):
    batch_ids, node_ids = uses_residual.any(dim=1).nonzero(as_tuple=True)
    tail_logits = unary_logits[batch_ids, node_ids].float().clone()
    tail_logits.scatter_(-1, output.candidate_ids[batch_ids, node_ids], -torch.inf)
    tail_draws = torch.multinomial(
      tail_logits.softmax(dim=-1), num_samples=num_samples,
      replacement=True, generator=generator)
    tokens[batch_ids, :, node_ids] = torch.where(
      uses_residual[batch_ids, :, node_ids], tail_draws,
      tokens[batch_ids, :, node_ids])
  return tokens
