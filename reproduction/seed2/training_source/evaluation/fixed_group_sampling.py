"""Compare token sampling methods with identical, fixed reveal positions.

Each unfinished row reveals exactly min(group_size, remaining_masks) tokens
per encoder call. Reveal order is supplied or generated once from independent
per-prompt seeds; sampled token identities cannot change it. No confidence
ranking or idle denoising steps enter this controlled experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import math
from typing import Callable, Optional, Sequence

import torch

from models.structured_decoder import StructuredDecoderOutput
from structured_objective import (
  infer_structured_distribution,
  sample_structured_tokens,
)


GROUP_SIZES = (1, 2, 4, 8, 16)
SAMPLING_MODES = ('joint', 'marginal', 'backbone')
ModelCallback = Callable[
  [torch.Tensor, torch.Tensor, torch.Tensor],
  tuple[StructuredDecoderOutput, torch.Tensor]]


@dataclass(frozen=True)
class FixedGroupStep:
  """One encoder batch; all tensors are detached CPU snapshots.

  ``row_indices`` maps this possibly shortened batch to original prompts.
  Edges follow the model's original padded edge layout.  An edge counts as
  jointly committed only when both endpoints enter ``revealed_mask`` here.
  ``proposal_tokens`` preserves context. Multi-token rows include draws for
  every active position; singleton rows draw only their selected position,
  leaving other masks untouched. Only ``revealed_mask`` is committed.
  """

  step: int
  row_indices: torch.Tensor
  tokens_before: torch.Tensor
  active_before: torch.Tensor
  masked_fraction: torch.Tensor
  diffusion_time: torch.Tensor
  sigma: torch.Tensor
  revealed_mask: torch.Tensor
  proposal_tokens: torch.Tensor
  tokens_after: torch.Tensor
  active_after: torch.Tensor
  edge_index: torch.Tensor
  edge_mask: torch.Tensor
  jointly_committed_edge_mask: torch.Tensor


@dataclass(frozen=True)
class FixedGroupGeneration:
  """Final tokens retain the input device; counts, order and traces use CPU."""

  final_tokens: torch.Tensor
  nfe: torch.Tensor
  batch_calls: int
  reveal_order: torch.Tensor
  group_size: int
  mode: str
  steps: tuple[FixedGroupStep, ...]


def make_reveal_order(sequence_length: int, reveal_seeds: Sequence[int]) -> torch.Tensor:
  """Draw one CPU permutation per prompt without touching any global RNG."""
  if type(sequence_length) is not int or sequence_length < 1:
    raise ValueError('sequence_length must be a positive integer')
  if not reveal_seeds or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in reveal_seeds):
    raise ValueError('reveal_seeds must contain one integer in [0,2**63) per prompt')
  return torch.stack([
    torch.randperm(sequence_length, generator=torch.Generator().manual_seed(seed))
    for seed in reveal_seeds])


def _validated_order(tokens: torch.Tensor, reveal_order: Optional[torch.Tensor],
                     reveal_seeds: Optional[Sequence[int]]) -> torch.Tensor:
  if (reveal_order is None) == (reveal_seeds is None):
    raise ValueError('provide exactly one of reveal_order or reveal_seeds')
  if reveal_order is None:
    if len(reveal_seeds) != tokens.shape[0]:
      raise ValueError('reveal_seeds must contain one seed per prompt')
    reveal_order = make_reveal_order(tokens.shape[1], reveal_seeds)
  if not torch.is_tensor(reveal_order) or reveal_order.dtype != torch.long or reveal_order.shape != tokens.shape:
    raise ValueError('reveal_order must be a long tensor with the same [B,L] shape as tokens')
  expected = torch.arange(tokens.shape[1], device=reveal_order.device).expand_as(reveal_order)
  if not torch.equal(reveal_order.sort(dim=-1).values, expected):
    raise ValueError('every reveal_order row must be a permutation of all positions')
  return reveal_order.to(tokens.device).clone()


def _validate_output(output: StructuredDecoderOutput, logits: torch.Tensor,
                     tokens: torch.Tensor, active: torch.Tensor, mask_index: int) -> None:
  if output.candidate_ids.shape[:2] != tokens.shape or logits.ndim != 3 or logits.shape[:2] != tokens.shape:
    raise ValueError('model output must match the unfinished [B,L] token batch')
  if logits.device != tokens.device or output.candidate_ids.device != tokens.device:
    raise ValueError('model output and tokens must use the same device')
  if mask_index < logits.shape[-1] and not bool(torch.isneginf(logits[..., mask_index]).all()):
    raise ValueError('callback must exclude the absorbing mask from clean-token logits')
  if bool((torch.isnan(logits) | torch.isposinf(logits)).any()):
    raise ValueError('clean-token logits cannot contain NaN or +inf')
  if not bool(torch.isfinite(logits).any(dim=-1).all()):
    raise ValueError('each position must have some clean-token support')
  for row in range(tokens.shape[0]):
    edges = output.edge_index[row, output.edge_mask[row]]
    if bool(((edges < 0) | (edges >= tokens.shape[1])).any()):
      raise ValueError('active forest edge endpoint lies outside the sequence')
    if edges.numel() and not bool(active[row, edges].all()):
      raise ValueError('forest edges must join currently active positions')


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
  return tensor.detach().cpu().clone()


def _sample_exact_marginals(output, logits, inference, generator):
  """Draw compressed marginals, expanding only residual states actually drawn.

  This avoids full-vocabulary marginal tensors and never normalizes an empty
  residual, whose marginal probability is zero. Observed rows are clamped to
  their first candidate by the existing inference implementation.
  """
  probabilities = inference.marginals.node_log_marginals.exp()
  states = torch.multinomial(
    probabilities.reshape(-1, probabilities.shape[-1]), num_samples=1,
    generator=generator).reshape(logits.shape[:2])
  explicit_count = output.candidate_ids.shape[-1]
  tokens = torch.gather(
    output.candidate_ids, -1,
    states.clamp_max(explicit_count - 1).unsqueeze(-1)).squeeze(-1)
  residual = states.eq(explicit_count)
  if bool(residual.any()):
    if type(output).residual_log_probs is StructuredDecoderOutput.residual_log_probs:
      # Keep the original ordinary-output arithmetic and RNG path unchanged.
      tail = logits[residual].clone()
      tail.scatter_(-1, output.candidate_ids[residual], -torch.inf)
      tail_probabilities = tail.softmax(dim=-1)
    else:
      # Specialized outputs may define a higher-precision conditional. Do not
      # silently replace it with a softmax in the raw backbone-logit dtype.
      tail_probabilities = output.residual_log_probs(logits)[residual].exp()
    tokens[residual] = torch.multinomial(
      tail_probabilities, num_samples=1, generator=generator).squeeze(-1)
  return tokens


def _sample_selected_marginals(output, logits, inference, selected, generator):
  """Sample only selected nodes, with the unchanged base conditional tail.

  For one committed node this is exactly the joint law marginalized over all
  uncommitted nodes. Both comparison modes call this same function, so their
  categorical draws, residual draws, and RNG consumption agree exactly.
  """
  probabilities = inference.marginals.node_log_marginals[selected].exp()
  states = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
  candidates = output.candidate_ids[selected]
  explicit_count = candidates.shape[-1]
  tokens = candidates.gather(
    -1, states.clamp_max(explicit_count - 1).unsqueeze(-1)).squeeze(-1)
  residual = states.eq(explicit_count)
  if bool(residual.any()):
    rows, positions = selected.nonzero(as_tuple=True)
    if type(output).residual_log_probs is StructuredDecoderOutput.residual_log_probs:
      tail = logits[rows[residual], positions[residual]].clone()
      tail.scatter_(-1, candidates[residual], -torch.inf)
      tail_probabilities = tail.softmax(dim=-1)
    else:
      tail_probabilities = output.residual_log_probs(logits)[
        rows[residual], positions[residual]].exp()
    tokens[residual] = torch.multinomial(
      tail_probabilities, 1, generator=generator).squeeze(-1)
  return tokens


def _batch_rows(bundle, rows):
  """Slice the batch axis of the output/inference dataclass tensor bundles."""
  changes = {}
  for field in fields(bundle):
    value = getattr(bundle, field.name)
    if torch.is_tensor(value):
      changes[field.name] = value.index_select(0, rows)
    elif is_dataclass(value):
      changes[field.name] = _batch_rows(value, rows)
  return replace(bundle, **changes)


@torch.no_grad()
def generate_fixed_groups(
    model: ModelCallback,
    initial_tokens: torch.Tensor,
    *,
    mask_index: int,
    group_size: int,
    mode: str,
    reveal_order: Optional[torch.Tensor] = None,
    reveal_seeds: Optional[Sequence[int]] = None,
    sampling_generator: Optional[torch.Generator] = None,
    noise_eps: float = 1e-3,
    inference_backend: str = 'low_rank') -> FixedGroupGeneration:
  """Generate fixed-size token groups using joint, marginal or base draws.

  ``model(tokens, sigma, active_mask)`` returns a structured output and raw,
  mask-excluded backbone logits. ``sigma`` has shape [unfinished_batch].
  Both backbone and head must receive this same noise conditioning. To wrap
  the production method, pass ``conditioning=sigma[:, None]`` to
  ``model._structured_head_output``.

  Noise reflects the current mask fraction over the full sequence length:
  ``t = min(mask_fraction/(1-eps), 1)`` and
  ``sigma = -log(1-min(mask_fraction,1-eps))``. Fully masked rows use t=1;
  observed tokens remain fixed. Completed rows never enter another callback.

  A supplied sampling generator is used only for token identities. If absent,
  a local generator on the input device uses seed zero. Reveal seeds always
  use separate CPU generators and never consume this identity RNG.

  Rows committing one token use the same selected-node marginal draw in joint
  and marginal modes. At group_size=1, identical callbacks, orders, and initial
  RNG states therefore give identical trajectories and final RNG states.
  In mixed-quota batches, singleton rows draw first in original row order;
  other rows retain the full-forest joint or product-marginal sampling law.
  """
  if initial_tokens.ndim != 2 or initial_tokens.dtype != torch.long or min(initial_tokens.shape) < 1:
    raise ValueError('initial_tokens must be a nonempty long [B,L] tensor')
  if type(mask_index) is not int or mask_index < 0 or bool((initial_tokens < 0).any()):
    raise ValueError('mask_index and token IDs must be nonnegative integers')
  if type(group_size) is not int or group_size not in GROUP_SIZES:
    raise ValueError(f'group_size must be one of {GROUP_SIZES}')
  if mode not in SAMPLING_MODES:
    raise ValueError(f'mode must be one of {SAMPLING_MODES}')
  if not math.isfinite(noise_eps) or not 0 < noise_eps < 1:
    raise ValueError('noise_eps must lie strictly between zero and one')
  order = _validated_order(initial_tokens, reveal_order, reveal_seeds)
  if sampling_generator is None:
    sampling_generator = torch.Generator(device=initial_tokens.device).manual_seed(0)
  tokens = initial_tokens.clone()
  active = tokens.eq(mask_index)
  initial_active_counts = active.sum(dim=-1)
  nfe = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
  traces = []
  while bool(active.any()):
    row_indices = active.any(dim=-1).nonzero().flatten()
    current = tokens.index_select(0, row_indices)
    current_active = active.index_select(0, row_indices)
    current_order = order.index_select(0, row_indices)
    masked_fraction = current_active.sum(dim=-1).float() / tokens.shape[1]
    effective_rate = masked_fraction.clamp_max(1.0 - noise_eps)
    diffusion_time = (masked_fraction / (1.0 - noise_eps)).clamp_max(1.0)
    sigma = -torch.log1p(-effective_rate)
    output, logits = model(current.clone(), sigma, current_active.clone())
    _validate_output(output, logits, current, current_active, mask_index)

    # Choose the quota before sampling so singleton rows can marginalize out
    # every uncommitted node instead of drawing and discarding a full forest.
    ordered_active = torch.gather(current_active, 1, current_order)
    selected_in_order = ordered_active & (ordered_active.long().cumsum(-1) <= group_size)
    revealed = torch.zeros_like(current_active).scatter(1, current_order, selected_in_order)
    expected_quota = current_active.sum(-1).clamp_max(group_size)
    if not torch.equal(revealed.sum(-1), expected_quota):
      raise AssertionError('reveal quota drifted')

    if mode == 'backbone':
      proposed = current.clone()
      proposed[current_active] = torch.multinomial(
        logits[current_active].softmax(dim=-1), num_samples=1,
        generator=sampling_generator).squeeze(-1)
    else:
      inference = infer_structured_distribution(output, current_active, inference_backend)
      proposed = current.clone()
      singleton_rows = expected_quota.eq(1)
      singleton_selected = revealed & singleton_rows[:, None]
      if bool(singleton_rows.any()):
        proposed[singleton_selected] = _sample_selected_marginals(
          output, logits, inference, singleton_selected, sampling_generator)
      multiple_rows = (~singleton_rows).nonzero().flatten()
      if multiple_rows.numel():
        # Leave the original all-multi-token path untouched. In mixed batches
        # remove singleton rows from both random draws and tail expansion.
        if bool(singleton_rows.any()):
          draw_output = _batch_rows(output, multiple_rows)
          draw_inference = _batch_rows(inference, multiple_rows)
          draw_logits = logits.index_select(0, multiple_rows)
          draw_active = current_active.index_select(0, multiple_rows)
        else:
          draw_output, draw_inference = output, inference
          draw_logits, draw_active = logits, current_active
        if mode == 'joint':
          sampled = sample_structured_tokens(
            draw_output, draw_logits, draw_active, num_samples=1,
            generator=sampling_generator, inference=draw_inference)[:, 0]
        else:
          sampled = _sample_exact_marginals(
            draw_output, draw_logits, draw_inference, sampling_generator)
        proposed[multiple_rows] = torch.where(
          draw_active, sampled, current.index_select(0, multiple_rows))
    if bool(proposed[revealed].eq(mask_index).any()):
      raise ValueError('sampler returned the absorbing mask as a clean token')
    updated = torch.where(revealed, proposed, current)
    next_active = current_active & ~revealed
    edges = output.edge_index.clamp(0, tokens.shape[1] - 1)
    joined = output.edge_mask & torch.gather(revealed, 1, edges[..., 0]) & torch.gather(revealed, 1, edges[..., 1])
    traces.append(FixedGroupStep(
      step=len(traces) + 1, row_indices=_snapshot(row_indices),
      tokens_before=_snapshot(current), active_before=_snapshot(current_active),
      masked_fraction=_snapshot(masked_fraction), diffusion_time=_snapshot(diffusion_time),
      sigma=_snapshot(sigma), revealed_mask=_snapshot(revealed),
      proposal_tokens=_snapshot(proposed), tokens_after=_snapshot(updated),
      active_after=_snapshot(next_active), edge_index=_snapshot(output.edge_index),
      edge_mask=_snapshot(output.edge_mask), jointly_committed_edge_mask=_snapshot(joined)))
    tokens.index_copy_(0, row_indices, updated)
    active.index_copy_(0, row_indices, next_active)
    nfe[row_indices] += 1
  expected_nfe = (initial_active_counts + group_size - 1) // group_size
  if not torch.equal(nfe, expected_nfe):
    raise AssertionError('encoder calls do not match fixed-group quotas')
  observed = ~initial_tokens.eq(mask_index)
  if not torch.equal(tokens[observed], initial_tokens[observed]):
    raise AssertionError('an observed token changed')
  return FixedGroupGeneration(
    final_tokens=tokens, nfe=_snapshot(nfe), batch_calls=len(traces),
    reveal_order=_snapshot(order), group_size=group_size, mode=mode,
    steps=tuple(traces))
