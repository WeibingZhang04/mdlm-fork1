"""Training and sampling adapters for contextual coupling-forest outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

import structured_utils
from models.structured_decoder import StructuredDecoderOutput


@dataclass(frozen=True)
class StructuredInference:
  """Exact forest result plus the tensors used to obtain it."""

  marginals: Union[
    structured_utils.ForestMarginals,
    structured_utils.LowRankForestMarginals]
  clamped_states: torch.Tensor
  backend: str
  log_pair_factors: Optional[torch.Tensor] = None


def _validate_active_mask(
    output: StructuredDecoderOutput,
    active_mask: torch.Tensor) -> torch.Tensor:
  expected = output.candidate_ids.shape[:2]
  if active_mask.shape != expected or active_mask.dtype != torch.bool:
    raise ValueError(
      f'active_mask must be boolean with shape {tuple(expected)}')
  return active_mask.to(device=output.candidate_ids.device)


def compressed_states_for_tokens(
    output: StructuredDecoderOutput,
    token_ids: torch.Tensor) -> torch.Tensor:
  """Map full-vocabulary tokens to explicit candidates or the residual."""
  if token_ids.shape != output.candidate_ids.shape[:2]:
    raise ValueError('token_ids must have shape [B,L]')
  matches = output.candidate_ids.eq(token_ids[:, :, None])
  explicit = matches.any(dim=-1)
  explicit_state = matches.to(torch.long).argmax(dim=-1)
  residual = torch.full_like(explicit_state, output.num_candidate_states - 1)
  states = torch.where(explicit, explicit_state, residual)
  allowed = torch.gather(
    output.candidate_state_mask, -1, states[:, :, None]).squeeze(-1)
  if not bool(allowed.all().item()):
    raise ValueError('a token maps to a disabled residual state')
  return states


def infer_structured_distribution(
    output: StructuredDecoderOutput,
    active_mask: torch.Tensor,
    backend: str = 'auto') -> StructuredInference:
  """Run exact sum-product, cancelling nodes outside the masked set."""
  active_mask = _validate_active_mask(output, active_mask)
  # Forest edges already connect active nodes only.  Clamping every inactive
  # isolated node to an arbitrary valid candidate makes its unary cancel
  # exactly between the assignment score and partition function.
  clamped_states = torch.where(
    active_mask,
    torch.full_like(active_mask, -1, dtype=torch.long),
    torch.zeros_like(active_mask, dtype=torch.long))
  if backend == 'auto':
    # Tiny synthetic lattices are faster with dense kernels; realistic top-K
    # heads use the endpoint path to avoid K-squared time and memory.
    backend = (
      'dense' if output.candidate_ids.shape[-1] <= 16 else 'low_rank')
  if backend not in {'dense', 'low_rank'}:
    raise ValueError("backend must be 'auto', 'dense', or 'low_rank'")
  log_pair_factors = None
  if backend == 'dense':
    log_pair_factors = structured_utils.positive_pair_factors_to_log(
      output.materialize_pair_factors())
    marginals = structured_utils.forest_sum_product(
      output.unary_log_potentials,
      log_pair_factors,
      output.edge_index,
      edge_mask=output.edge_mask,
      state_mask=output.candidate_state_mask,
      clamped_states=clamped_states,
      max_component_size=None)
  else:
    marginals = structured_utils.forest_sum_product_low_rank(
      output.unary_log_potentials,
      output.pair_left_factors,
      output.pair_right_factors,
      output.edge_index,
      edge_mask=output.edge_mask,
      state_mask=output.candidate_state_mask,
      clamped_states=clamped_states,
      max_component_size=None)
  return StructuredInference(
    marginals=marginals,
    clamped_states=clamped_states,
    backend=backend,
    log_pair_factors=log_pair_factors)


def structured_token_log_probability(
    output: StructuredDecoderOutput,
    unary_logits: torch.Tensor,
    token_ids: torch.Tensor,
    active_mask: torch.Tensor,
    inference: Optional[StructuredInference] = None) -> torch.Tensor:
  """Exact log p(tokens at active nodes | context) under full support.

  When a token falls outside top-K, its compressed-state probability is
  multiplied by the normalized residual decoder probability.  Inactive nodes
  are clamped and cancel from the normalized likelihood.
  """
  active_mask = _validate_active_mask(output, active_mask)
  if unary_logits.shape[:2] != token_ids.shape:
    raise ValueError('unary_logits and token_ids leading shapes differ')
  if unary_logits.shape[-1] <= int(output.candidate_ids.max().item()):
    raise ValueError('unary_logits vocabulary is incompatible with candidates')
  states = compressed_states_for_tokens(output, token_ids)
  states = torch.where(
    active_mask, states, torch.zeros_like(states))
  inference = inference or infer_structured_distribution(output, active_mask)

  node_score = torch.gather(
    output.unary_log_potentials, -1, states[:, :, None]).squeeze(-1).sum(-1)
  if output.edge_index.shape[1]:
    left_nodes = output.edge_index[:, :, 0]
    right_nodes = output.edge_index[:, :, 1]
    left_states = torch.gather(states, 1, left_nodes)
    right_states = torch.gather(states, 1, right_nodes)
    explicit_count = output.candidate_ids.shape[-1]
    left_is_residual = left_states.eq(explicit_count)
    right_is_residual = right_states.eq(explicit_count)
    safe_left = left_states.clamp_max(explicit_count - 1)
    safe_right = right_states.clamp_max(explicit_count - 1)
    left_index = safe_left[:, :, None, None].expand(
      -1, -1, 1, output.pair_left_factors.shape[-1])
    right_index = safe_right[:, :, None, None].expand(
      -1, -1, 1, output.pair_right_factors.shape[-1])
    selected_left = torch.gather(
      output.pair_left_factors, 2, left_index).squeeze(2)
    selected_right = torch.gather(
      output.pair_right_factors, 2, right_index).squeeze(2)
    # Score the selected low-rank factor in log space.  Multiplying endpoint
    # factors first can overflow even when the normalized forest probability
    # is perfectly finite (for example, 1e30 * 1e30 in float32).
    edge_score = torch.logsumexp(
      selected_left.log() + selected_right.log(), dim=-1)
    edge_score = torch.where(
      left_is_residual | right_is_residual,
      torch.zeros_like(edge_score), edge_score)
    edge_score = edge_score.masked_fill(~output.edge_mask, 0.0).sum(-1)
  else:
    edge_score = node_score.new_zeros(node_score.shape)

  residual_state = output.num_candidate_states - 1
  uses_residual = active_mask & states.eq(residual_state)
  residual_correction = node_score.new_zeros(node_score.shape)
  if bool(uses_residual.any().item()):
    tail_log_probs = output.residual_log_probs(unary_logits)
    token_tail_log_prob = torch.gather(
      tail_log_probs, -1, token_ids[:, :, None]).squeeze(-1)
    residual_correction = torch.where(
      uses_residual, token_tail_log_prob,
      torch.zeros_like(token_tail_log_prob)).sum(-1)
  return (
    node_score + edge_score
    - inference.marginals.log_partition
    + residual_correction)


def structured_marginal_token_log_probability(
    output: StructuredDecoderOutput,
    unary_logits: torch.Tensor,
    token_ids: torch.Tensor,
    active_mask: torch.Tensor,
    inference: Optional[StructuredInference] = None) -> torch.Tensor:
  """Log probability under the product of exact singleton marginals.

  This is the likelihood counterpart of
  :func:`sample_structured_marginal_tokens`.  It uses the *same* exact forest
  inference result as joint likelihood/sampling, but deliberately discards
  all cross-node dependence after marginalization.  Comparing it with
  :func:`structured_token_log_probability` therefore isolates the value of
  representing the joint rather than changes to singleton predictions.

  The implementation stays on the compressed top-K-plus-residual lattice.
  It never materializes a ``[B,L,V]`` probability tensor, which would be
  prohibitively large for the released MDLM vocabulary.
  """
  active_mask = _validate_active_mask(output, active_mask)
  if unary_logits.shape[:2] != token_ids.shape:
    raise ValueError('unary_logits and token_ids leading shapes differ')
  if unary_logits.shape[-1] <= int(output.candidate_ids.max().item()):
    raise ValueError('unary_logits vocabulary is incompatible with candidates')

  inference = inference or infer_structured_distribution(output, active_mask)
  states = compressed_states_for_tokens(output, token_ids)
  states = torch.where(active_mask, states, torch.zeros_like(states))
  compressed_log_probability = torch.gather(
    inference.marginals.node_log_marginals,
    -1,
    states[:, :, None]).squeeze(-1)

  residual_state = output.num_candidate_states - 1
  uses_residual = active_mask & states.eq(residual_state)
  residual_correction = compressed_log_probability.new_zeros(
    compressed_log_probability.shape)
  if bool(uses_residual.any().item()):
    tail_log_probs = output.residual_log_probs(unary_logits)
    token_tail_log_prob = torch.gather(
      tail_log_probs, -1, token_ids[:, :, None]).squeeze(-1)
    residual_correction = torch.where(
      uses_residual,
      token_tail_log_prob,
      torch.zeros_like(token_tail_log_prob))

  per_node = compressed_log_probability + residual_correction
  return torch.where(
    active_mask, per_node, torch.zeros_like(per_node)).sum(dim=-1)


def factorized_token_log_probability(
    unary_logits: torch.Tensor,
    token_ids: torch.Tensor,
    active_mask: torch.Tensor) -> torch.Tensor:
  """Per-example log probability under the original factorized backbone.

  ``unary_logits`` are the released MDLM backbone logits before any forest
  factor is applied.  Returning one value per example makes this baseline
  pairable with joint and marginal-product likelihoods on the exact same
  corruption draw.
  """
  if unary_logits.ndim != 3:
    raise ValueError('unary_logits must have shape [B,L,V]')
  if unary_logits.shape[:2] != token_ids.shape:
    raise ValueError('unary_logits and token_ids leading shapes differ')
  if active_mask.shape != token_ids.shape or active_mask.dtype != torch.bool:
    raise ValueError('active_mask must be boolean with shape [B,L]')
  selected = torch.gather(
    unary_logits, -1, token_ids[:, :, None]).squeeze(-1)
  per_node = selected - torch.logsumexp(unary_logits, dim=-1)
  return torch.where(
    active_mask, per_node, torch.zeros_like(per_node)).sum(dim=-1)


def full_vocabulary_marginals(
    output: StructuredDecoderOutput,
    unary_logits: torch.Tensor,
    active_mask: torch.Tensor,
    inference: Optional[StructuredInference] = None) -> torch.Tensor:
  """Expand exact compressed node marginals back to the full vocabulary."""
  active_mask = _validate_active_mask(output, active_mask)
  inference = inference or infer_structured_distribution(output, active_mask)
  vocab_size = unary_logits.shape[-1]
  result = unary_logits.new_zeros(
    *unary_logits.shape, dtype=inference.marginals.node_marginals.dtype)
  explicit = inference.marginals.node_marginals[..., :-1]
  result.scatter_add_(-1, output.candidate_ids, explicit)
  residual_probability = inference.marginals.node_marginals[..., -1:]
  if bool(output.candidate_state_mask[..., -1].any().item()):
    tail_probability = output.residual_log_probs(unary_logits).exp()
    result = result + residual_probability * tail_probability
  return result


def sample_structured_tokens(
    output: StructuredDecoderOutput,
    unary_logits: torch.Tensor,
    active_mask: torch.Tensor,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
    inference: Optional[StructuredInference] = None) -> torch.Tensor:
  """Jointly sample full-vocabulary tokens; output shape is [B,S,L]."""
  active_mask = _validate_active_mask(output, active_mask)
  inference = inference or infer_structured_distribution(output, active_mask)
  if inference.backend == 'dense':
    states = structured_utils.sample_forest(
      output.unary_log_potentials,
      inference.log_pair_factors,
      output.edge_index,
      num_samples=num_samples,
      edge_mask=output.edge_mask,
      state_mask=output.candidate_state_mask,
      clamped_states=inference.clamped_states,
      generator=generator)
  else:
    states = structured_utils.sample_forest_low_rank(
      output.unary_log_potentials,
      output.pair_left_factors,
      output.pair_right_factors,
      output.edge_index,
      num_samples=num_samples,
      edge_mask=output.edge_mask,
      state_mask=output.candidate_state_mask,
      clamped_states=inference.clamped_states,
      generator=generator)
  batch_size, _, sequence_length = states.shape
  explicit_states = states.clamp_max(output.candidate_ids.shape[-1] - 1)
  candidates = output.candidate_ids[:, None].expand(
    batch_size, num_samples, sequence_length, -1)
  tokens = torch.gather(
    candidates, -1, explicit_states[:, :, :, None]).squeeze(-1)
  residual_state = output.num_candidate_states - 1
  uses_residual = states.eq(residual_state)
  if bool(uses_residual.any().item()):
    tail_probabilities = output.residual_log_probs(unary_logits).exp()
    tail_draws = torch.multinomial(
      tail_probabilities.reshape(-1, tail_probabilities.shape[-1]),
      num_samples=num_samples,
      replacement=True,
      generator=generator)
    tail_draws = tail_draws.reshape(
      batch_size, sequence_length, num_samples).transpose(1, 2)
    tokens = torch.where(uses_residual, tail_draws, tokens)
  return tokens


def sample_structured_marginal_tokens(
    output: StructuredDecoderOutput,
    unary_logits: torch.Tensor,
    active_mask: torch.Tensor,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
    inference: Optional[StructuredInference] = None) -> torch.Tensor:
  """Sample nodes independently from exact structured node marginals.

  This is the controlled marginal-sampling ablation: inference still uses the
  same forest and checkpoint as joint sampling, but the final categorical
  draws deliberately discard cross-node correlations.  Output is ``[B,S,L]``.
  """
  if num_samples < 1:
    raise ValueError('num_samples must be positive')
  active_mask = _validate_active_mask(output, active_mask)
  probabilities = full_vocabulary_marginals(
    output=output,
    unary_logits=unary_logits,
    active_mask=active_mask,
    inference=inference).clamp_min(0)
  normalizer = probabilities.sum(dim=-1, keepdim=True)
  if bool((normalizer <= 0).any().item()):
    raise RuntimeError('structured node marginal has zero total mass')
  probabilities = probabilities / normalizer
  batch_size, sequence_length, vocab_size = probabilities.shape
  draws = torch.multinomial(
    probabilities.reshape(-1, vocab_size),
    num_samples=num_samples,
    replacement=True,
    generator=generator)
  return draws.reshape(
    batch_size, sequence_length, num_samples).transpose(1, 2)
