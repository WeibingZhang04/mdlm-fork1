"""Paired causal controls for real-text contextual-forest evaluation.

Every score in this module is computed from one backbone forward pass and the
same corruption draw.  The resulting comparisons therefore isolate:

* released factorized MDLM unaries versus the structured adapter;
* the product of exact forest singleton marginals versus the full joint; and
* the learned forest versus a component/degree-matched node permutation.

The no-edge score is intentionally an algebraic control.  Neutralizing every
pair factor in the fully instantiated head gives exactly the released
factorized distribution because the head does not alter unary logits.  We
record both names and test the identity rather than paying for redundant tree
inference on every paper-scale batch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch

from models.structured_decoder import (
  ContextualCouplingForestHead,
  StructuredDecoderOutput,
)
from structured_objective import (
  factorized_token_log_probability,
  infer_structured_distribution,
  structured_marginal_token_log_probability,
  structured_token_log_probability,
)


@dataclass(frozen=True)
class CandidateSupportSweep:
  """Per-example top-K support diagnostics for several K values."""

  candidate_ks: tuple[int, ...]
  candidate_hits: torch.Tensor          # [B,Q]
  retained_mass_sum: torch.Tensor       # [B,Q]


@dataclass(frozen=True)
class CausalDenoisingMetrics:
  """Per-example likelihood and topology-control sufficient statistics."""

  structured_joint_nll_sum: torch.Tensor
  structured_marginal_nll_sum: torch.Tensor
  factorized_backbone_nll_sum: torch.Tensor
  parameter_matched_no_edge_nll_sum: torch.Tensor
  matched_permuted_topology_nll_sum: torch.Tensor
  active_tokens: torch.Tensor
  candidate_support: CandidateSupportSweep
  selected_edges: torch.Tensor
  permuted_changed_edges: torch.Tensor
  selected_degree_sequences: tuple[tuple[int, ...], ...]
  permuted_degree_sequences: tuple[tuple[int, ...], ...]
  selected_component_sizes: tuple[tuple[int, ...], ...]
  permuted_component_sizes: tuple[tuple[int, ...], ...]

  def as_record_metrics(self) -> dict[str, object]:
    """Return the schema-v2 fields consumed by the streaming writer."""
    return {
      # Preserve the schema-v1 name as the authoritative joint score so old
      # paired tooling can still summarize a schema-v2 run after validation.
      'nll_sum': self.structured_joint_nll_sum,
      'structured_marginal_nll_sum': self.structured_marginal_nll_sum,
      'factorized_backbone_nll_sum': self.factorized_backbone_nll_sum,
      'parameter_matched_no_edge_nll_sum': (
        self.parameter_matched_no_edge_nll_sum),
      'matched_permuted_topology_nll_sum': (
        self.matched_permuted_topology_nll_sum),
      'active_tokens': self.active_tokens,
      'candidate_support_ks': list(self.candidate_support.candidate_ks),
      'candidate_support_hits': self.candidate_support.candidate_hits,
      'candidate_support_retained_mass_sum': (
        self.candidate_support.retained_mass_sum),
      'selected_edges': self.selected_edges,
      'permuted_changed_edges': self.permuted_changed_edges,
      'selected_degree_sequence': self.selected_degree_sequences,
      'permuted_degree_sequence': self.permuted_degree_sequences,
      'selected_component_sizes': self.selected_component_sizes,
      'permuted_component_sizes': self.permuted_component_sizes,
    }


def _validated_candidate_ks(
    candidate_ks: Sequence[int], vocab_size: int) -> tuple[int, ...]:
  if isinstance(candidate_ks, (str, bytes)):
    raise TypeError('candidate_ks must be a sequence of integers')
  result = tuple(candidate_ks)
  if not result:
    raise ValueError('candidate_ks must be non-empty')
  if any(not isinstance(value, int) or isinstance(value, bool)
         for value in result):
    raise TypeError('candidate_ks must contain only integers')
  if any(value < 1 or value > vocab_size for value in result):
    raise ValueError('candidate_ks must lie in [1, vocabulary size]')
  if tuple(sorted(set(result))) != result:
    raise ValueError('candidate_ks must be unique and increasing')
  return result


@torch.no_grad()
def candidate_support_sweep(
    unary_logits: torch.Tensor,
    clean_tokens: torch.Tensor,
    active_mask: torch.Tensor,
    candidate_ks: Sequence[int],
) -> CandidateSupportSweep:
  """Measure top-K target recall and retained unary mass in one top-k call.

  This diagnostic depends only on the released backbone unaries, so it does
  not require fitting a separate adapter for K=32/64/128/256.  Fresh adapters
  are still required for any *quality* comparison at a different training K.
  """
  if unary_logits.ndim != 3:
    raise ValueError('unary_logits must have shape [B,L,V]')
  if unary_logits.shape[:2] != clean_tokens.shape:
    raise ValueError('clean_tokens leading dimensions differ from logits')
  if active_mask.shape != clean_tokens.shape or active_mask.dtype != torch.bool:
    raise ValueError('active_mask must be boolean with shape [B,L]')
  candidate_ks = _validated_candidate_ks(
    candidate_ks, unary_logits.shape[-1])
  maximum_k = candidate_ks[-1]
  work_logits = (
    unary_logits.float()
    if unary_logits.dtype in (torch.float16, torch.bfloat16)
    else unary_logits)
  candidate_logits, candidate_ids = work_logits.topk(maximum_k, dim=-1)
  total_log_mass = torch.logsumexp(work_logits, dim=-1)

  hit_columns = []
  mass_columns = []
  for candidate_k in candidate_ks:
    hits = candidate_ids[..., :candidate_k].eq(
      clean_tokens[:, :, None]).any(dim=-1)
    hit_columns.append((hits & active_mask).sum(dim=-1))
    retained = torch.exp(
      torch.logsumexp(candidate_logits[..., :candidate_k], dim=-1)
      - total_log_mass).clamp(0.0, 1.0)
    mass_columns.append(
      retained.masked_fill(~active_mask, 0.0).sum(dim=-1))
  return CandidateSupportSweep(
    candidate_ks=candidate_ks,
    candidate_hits=torch.stack(hit_columns, dim=-1),
    retained_mass_sum=torch.stack(mass_columns, dim=-1))


def _stable_permutation(active_nodes: list[int], seed: int) -> list[int]:
  """Return a deterministic non-identity permutation of active node IDs."""
  if len(active_nodes) < 2:
    return list(active_nodes)
  active_fingerprint = ','.join(str(node) for node in active_nodes)
  keyed = []
  for node in active_nodes:
    digest = hashlib.sha256(
      f'contextual-forest-topology-control-v1|{seed}|'
      f'{active_fingerprint}|{node}'.encode('ascii')).digest()
    keyed.append((digest, node))
  permuted = [node for _, node in sorted(keyed)]
  if permuted == active_nodes:
    permuted = active_nodes[1:] + active_nodes[:1]
  return permuted


@torch.no_grad()
def matched_permuted_forest_edges(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Permute node labels while preserving forest degree/components exactly.

  The mapping is a deterministic bijection over each example's active nodes.
  Consequently it preserves edge count, the degree multiset, component sizes,
  and acyclicity while breaking alignment between context and selected edges.
  ``changed_edges`` counts edge slots whose unordered endpoints changed.
  """
  if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
    raise ValueError('seed must be a non-negative integer')
  if edge_index.ndim != 3 or edge_index.shape[-1] != 2:
    raise ValueError('edge_index must have shape [B,E,2]')
  if edge_mask.shape != edge_index.shape[:2] or edge_mask.dtype != torch.bool:
    raise ValueError('edge_mask must be boolean with shape [B,E]')
  if (active_mask.ndim != 2 or active_mask.dtype != torch.bool
      or active_mask.shape[0] != edge_index.shape[0]):
    raise ValueError('active_mask must be boolean with shape [B,L]')

  result = edge_index.detach().clone()
  changed = torch.zeros(
    edge_index.shape[0], dtype=torch.long, device=edge_index.device)
  for batch_index in range(edge_index.shape[0]):
    active_nodes = torch.nonzero(
      active_mask[batch_index], as_tuple=False).squeeze(-1).cpu().tolist()
    permuted = _stable_permutation(active_nodes, seed)
    mapping = dict(zip(active_nodes, permuted))
    for edge_slot in torch.nonzero(
        edge_mask[batch_index], as_tuple=False).squeeze(-1).cpu().tolist():
      left, right = edge_index[batch_index, edge_slot].cpu().tolist()
      if left not in mapping or right not in mapping:
        raise ValueError('selected edge references an inactive node')
      mapped = sorted((mapping[left], mapping[right]))
      result[batch_index, edge_slot] = torch.tensor(
        mapped, dtype=edge_index.dtype, device=edge_index.device)
      if mapped != sorted((left, right)):
        changed[batch_index] += 1
  return result.detach(), edge_mask.detach().clone(), changed


def _forest_structure_signatures(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
  """Return sorted degree sequences and component sizes for each forest.

  This deliberately recomputes the two invariants from edge endpoints rather
  than inferring them from the node-permutation construction.  The signatures
  are persisted in schema-v2 records so later promotion replay can verify the
  matched-topology claim independently of the control's NLL.
  """
  if edge_index.ndim != 3 or edge_index.shape[-1] != 2:
    raise ValueError('edge_index must have shape [B,E,2]')
  if edge_mask.shape != edge_index.shape[:2] or edge_mask.dtype != torch.bool:
    raise ValueError('edge_mask must be boolean with shape [B,E]')
  if (active_mask.ndim != 2 or active_mask.dtype != torch.bool
      or active_mask.shape[0] != edge_index.shape[0]):
    raise ValueError('active_mask must be boolean with shape [B,L]')

  degree_signatures = []
  component_signatures = []
  for batch_index in range(edge_index.shape[0]):
    active_nodes = torch.nonzero(
      active_mask[batch_index], as_tuple=False).squeeze(-1).cpu().tolist()
    if not active_nodes:
      raise ValueError('forest structure signature requires active nodes')
    active_set = set(active_nodes)
    parent = {node: node for node in active_nodes}
    size = {node: 1 for node in active_nodes}
    degree = {node: 0 for node in active_nodes}
    seen_edges: set[tuple[int, int]] = set()

    def find(node: int) -> int:
      while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
      return node

    for edge_slot in torch.nonzero(
        edge_mask[batch_index], as_tuple=False).squeeze(-1).cpu().tolist():
      left, right = (
        int(value) for value in edge_index[batch_index, edge_slot].cpu().tolist())
      if left not in active_set or right not in active_set:
        raise ValueError('selected edge references an inactive node')
      if left == right:
        raise ValueError('forest structure signature encountered a self-loop')
      edge = tuple(sorted((left, right)))
      if edge in seen_edges:
        raise ValueError('forest structure signature encountered a duplicate edge')
      seen_edges.add(edge)
      left_root = find(left)
      right_root = find(right)
      if left_root == right_root:
        raise ValueError('forest structure signature encountered a cycle')
      if size[left_root] < size[right_root]:
        left_root, right_root = right_root, left_root
      parent[right_root] = left_root
      size[left_root] += size[right_root]
      degree[left] += 1
      degree[right] += 1

    components: dict[int, int] = {}
    for node in active_nodes:
      root = find(node)
      components[root] = components.get(root, 0) + 1
    degree_signatures.append(tuple(sorted(degree.values())))
    component_signatures.append(tuple(sorted(components.values())))
  return tuple(degree_signatures), tuple(component_signatures)


@torch.no_grad()
def causal_denoising_metrics(
    *,
    head: ContextualCouplingForestHead,
    primary_output: StructuredDecoderOutput,
    hidden_states: torch.Tensor,
    unary_logits: torch.Tensor,
    timestep: torch.Tensor,
    clean_tokens: torch.Tensor,
    active_mask: torch.Tensor,
    candidate_ks: Sequence[int],
    topology_permutation_seed: int,
    structured_joint_nll_sum: torch.Tensor | None = None,
) -> CausalDenoisingMetrics:
  """Compute all paper-facing causal scores from one encoded batch."""
  inference = infer_structured_distribution(primary_output, active_mask)
  if structured_joint_nll_sum is None:
    structured_joint_nll_sum = -structured_token_log_probability(
      output=primary_output,
      unary_logits=unary_logits,
      token_ids=clean_tokens,
      active_mask=active_mask,
      inference=inference)
  elif structured_joint_nll_sum.shape != (clean_tokens.shape[0],):
    raise ValueError('structured_joint_nll_sum must have shape [B]')

  structured_marginal_nll_sum = (
    -structured_marginal_token_log_probability(
      output=primary_output,
      unary_logits=unary_logits,
      token_ids=clean_tokens,
      active_mask=active_mask,
      inference=inference))
  factorized_backbone_nll_sum = -factorized_token_log_probability(
    unary_logits=unary_logits,
    token_ids=clean_tokens,
    active_mask=active_mask)

  permuted_edges, permuted_edge_mask, changed_edges = (
    matched_permuted_forest_edges(
      primary_output.edge_index,
      primary_output.edge_mask,
      active_mask,
      seed=topology_permutation_seed))
  permuted_output = head(
    hidden_states=hidden_states,
    unary_logits=unary_logits,
    timestep=timestep,
    active_mask=active_mask,
    factor_mode=primary_output.factor_mode,
    fixed_edge_index=permuted_edges,
    fixed_edge_mask=permuted_edge_mask)
  matched_permuted_topology_nll_sum = -structured_token_log_probability(
    output=permuted_output,
    unary_logits=unary_logits,
    token_ids=clean_tokens,
    active_mask=active_mask)
  selected_degree_sequences, selected_component_sizes = (
    _forest_structure_signatures(
      primary_output.edge_index, primary_output.edge_mask, active_mask))
  permuted_degree_sequences, permuted_component_sizes = (
    _forest_structure_signatures(
      permuted_output.edge_index, permuted_output.edge_mask, active_mask))

  support = candidate_support_sweep(
    unary_logits=unary_logits,
    clean_tokens=clean_tokens,
    active_mask=active_mask,
    candidate_ks=candidate_ks)
  return CausalDenoisingMetrics(
    structured_joint_nll_sum=structured_joint_nll_sum.detach(),
    structured_marginal_nll_sum=structured_marginal_nll_sum.detach(),
    factorized_backbone_nll_sum=factorized_backbone_nll_sum.detach(),
    # Neutral pair factors cancel exactly from the normalized forest, so this
    # named parameter-count control is algebraically the factorized backbone.
    parameter_matched_no_edge_nll_sum=(
      factorized_backbone_nll_sum.detach().clone()),
    matched_permuted_topology_nll_sum=(
      matched_permuted_topology_nll_sum.detach()),
    active_tokens=active_mask.sum(dim=-1).detach(),
    candidate_support=support,
    selected_edges=primary_output.edge_mask.sum(dim=-1).detach(),
    permuted_changed_edges=changed_edges.detach(),
    selected_degree_sequences=selected_degree_sequences,
    permuted_degree_sequences=permuted_degree_sequences,
    selected_component_sizes=selected_component_sizes,
    permuted_component_sizes=permuted_component_sizes)
