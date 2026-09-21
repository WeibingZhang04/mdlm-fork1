"""Torch-only training losses for the real-data coupling-forest adapter.

The structured objective here is deliberately named a conditional denoising
loss, not a diffusion ELBO.  It trains the forest distribution on clean tokens
at forward-masked positions while retaining full vocabulary support through
the residual state.

Dynamic topology is trained by gold-reveal influence *distillation*: during
training only, one masked clean token is revealed to a detached backbone
teacher and the change in held-out clean-token log probability supervises
anchors, slot routing, and proposal scores.  Clean tokens are never inputs to
the structured head and are not available to topology selection at inference.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from models.structured_decoder import StructuredDecoderOutput
from structured_objective import (
  factorized_token_log_probability,
  structured_token_log_probability,
)


STRUCTURED_OBJECTIVE_NAME = (
  'conditional_denoising_nll_not_diffusion_elbo')
STRUCTURED_SAMPLING_MODES = frozenset({
  'factorized', 'factorized_confidence_gated',
  'structured_marginal', 'structured_joint'})
STRUCTURED_TOKEN_SAMPLING_MODES = frozenset({
  'structured_marginal', 'structured_joint'})


def validate_structured_objective_name(name: str) -> str:
  """Reject objective labels that could misstate the optimized quantity."""
  if name != STRUCTURED_OBJECTIVE_NAME:
    raise ValueError(
      'structured objective_name must be '
      f'{STRUCTURED_OBJECTIVE_NAME!r}, got {name!r}')
  return name


def validate_structured_sampling_mode(mode: str) -> str:
  """Validate the parameter-invariant structured sampling policy."""
  if mode not in STRUCTURED_SAMPLING_MODES:
    raise ValueError(
      'structured sampling mode must be one of '
      f'{sorted(STRUCTURED_SAMPLING_MODES)}, got {mode!r}')
  return mode


def uses_structured_token_distribution(mode: str) -> bool:
  """Whether ``mode`` samples clean identities from the coupling forest.

  ``factorized_confidence_gated`` remains a factorized-backbone control: only
  its reveal-position policy changes.  Keeping that distinction explicit
  prevents generic ``mode != 'factorized'`` checks from accidentally routing
  the control through the structured head.
  """
  validate_structured_sampling_mode(mode)
  return mode in STRUCTURED_TOKEN_SAMPLING_MODES


def validated_ema_shadow_parameters(
    ema_state,
    expected_parameters,
    context: str,
    allow_extra: bool = False):
  """Return shape-checked EMA shadows or raise an actionable error."""
  if not isinstance(ema_state, dict):
    raise ValueError(
      f'{context} has no EMA state; disable EMA loading explicitly')
  shadow_parameters = ema_state.get('shadow_params', None)
  if not isinstance(shadow_parameters, (list, tuple)):
    raise ValueError(f'{context} EMA has no shadow_params sequence')
  expected_parameters = list(expected_parameters)
  expected_count = len(expected_parameters)
  shadow_count = len(shadow_parameters)
  count_matches = (
    shadow_count >= expected_count if allow_extra
    else shadow_count == expected_count)
  if not count_matches:
    qualifier = 'at least ' if allow_extra else ''
    raise ValueError(
      f'{context} EMA has {shadow_count} shadows; expected '
      f'{qualifier}{expected_count}')
  for index, (shadow, parameter) in enumerate(zip(
      shadow_parameters, expected_parameters)):
    if not torch.is_tensor(shadow):
      raise ValueError(
        f'{context} EMA shadow {index} is not a tensor')
    if shadow.shape != parameter.shape:
      raise ValueError(
        f'{context} EMA shadow {index} has shape '
        f'{tuple(shadow.shape)}; expected {tuple(parameter.shape)}')
  return list(shadow_parameters[:expected_count])


@dataclass(frozen=True)
class StructuredDenoisingLoss:
  """Conditional structured NLL and diagnostics for one corrupted batch."""

  loss: torch.Tensor
  per_example_nll: torch.Tensor
  distributed_nll: torch.Tensor
  candidate_recall: torch.Tensor
  retained_mass: torch.Tensor
  active_tokens: torch.Tensor
  nll_sum: torch.Tensor
  candidate_hits: torch.Tensor
  retained_mass_sum: torch.Tensor


@dataclass(frozen=True)
class TopologyDistillationLoss:
  """Gold-reveal influence losses; all teacher quantities are detached."""

  loss: torch.Tensor
  edge_loss: torch.Tensor
  anchor_loss: torch.Tensor
  slot_loss: torch.Tensor
  valid_examples: torch.Tensor
  mean_influence: torch.Tensor
  edge_coverage: torch.Tensor
  edge_coverage_numerator: torch.Tensor
  edge_coverage_denominator: torch.Tensor
  anchor_coverage: torch.Tensor
  anchor_coverage_numerator: torch.Tensor
  anchor_coverage_denominator: torch.Tensor
  slot_coverage: torch.Tensor
  slot_coverage_numerator: torch.Tensor
  slot_coverage_denominator: torch.Tensor


def _validate_token_batch(
    output: StructuredDecoderOutput,
    clean_tokens: torch.Tensor,
    active_mask: torch.Tensor) -> None:
  expected = output.candidate_ids.shape[:2]
  if clean_tokens.shape != expected:
    raise ValueError(f'clean_tokens must have shape {tuple(expected)}')
  if active_mask.shape != expected or active_mask.dtype != torch.bool:
    raise ValueError(
      f'active_mask must be boolean with shape {tuple(expected)}')


def structured_denoising_loss(
    output: StructuredDecoderOutput,
    unary_logits: torch.Tensor,
    clean_tokens: torch.Tensor,
    active_mask: torch.Tensor,
    ) -> StructuredDenoisingLoss:
  """Return exact joint conditional NLL normalized per masked token.

  ``output.candidate_ids`` came only from ``unary_logits``; targets are used
  after candidate selection for likelihood evaluation and recall logging.
  ``distributed_nll`` assigns each example's joint NLL uniformly to its active
  sites solely so the repository's token-weighted metric accumulator can log
  the same per-masked-token quantity.
  """
  _validate_token_batch(output, clean_tokens, active_mask)
  log_probability = structured_token_log_probability(
    output=output,
    unary_logits=unary_logits,
    token_ids=clean_tokens,
    active_mask=active_mask)
  per_example_nll = -log_probability
  active_per_example = active_mask.sum(dim=-1)
  active_tokens = active_per_example.sum()
  nll_sum = per_example_nll.sum()
  loss = nll_sum / active_tokens.clamp_min(1)
  distributed_nll = (
    per_example_nll[:, None]
    / active_per_example.clamp_min(1)[:, None]
    * active_mask)

  target_is_explicit = output.candidate_ids.eq(
    clean_tokens[:, :, None]).any(dim=-1)
  candidate_hits = (target_is_explicit & active_mask).sum().to(loss.dtype)
  retained_mass_sum = output.retained_mass.masked_fill(
    ~active_mask, 0.0).sum()
  candidate_recall = candidate_hits / active_tokens.clamp_min(1)
  retained_mass = retained_mass_sum / active_tokens.clamp_min(1)
  if not bool(active_tokens.item()):
    candidate_recall = candidate_recall.new_tensor(1.0)
    retained_mass = retained_mass.new_tensor(1.0)
  return StructuredDenoisingLoss(
    loss=loss,
    per_example_nll=per_example_nll,
    distributed_nll=distributed_nll,
    candidate_recall=candidate_recall,
    retained_mass=retained_mass,
    active_tokens=active_tokens,
    nll_sum=nll_sum,
    candidate_hits=candidate_hits,
    retained_mass_sum=retained_mass_sum)


def factorized_denoising_nll(
    unary_logits: torch.Tensor,
    clean_tokens: torch.Tensor,
  active_mask: torch.Tensor) -> torch.Tensor:
  """Ordinary independent denoising NLL per active token."""
  active_tokens = active_mask.sum()
  log_probability = factorized_token_log_probability(
    unary_logits=unary_logits,
    token_ids=clean_tokens,
    active_mask=active_mask)
  return -(log_probability.sum() / active_tokens.clamp_min(1))


def sample_active_sources(
    active_mask: torch.Tensor,
    generator: Optional[torch.Generator] = None) -> torch.Tensor:
  """Sample one topology-teacher reveal position per nonempty example."""
  if active_mask.ndim != 2 or active_mask.dtype != torch.bool:
    raise ValueError('active_mask must be boolean [B,L]')
  weights = active_mask.float()
  valid = active_mask.any(dim=-1)
  safe_weights = weights.clone()
  if safe_weights.shape[1] == 0:
    raise ValueError('sequence length must be positive')
  safe_weights[~valid, 0] = 1.0
  sources = torch.multinomial(
    safe_weights, num_samples=1, replacement=True,
    generator=generator).squeeze(-1)
  return torch.where(valid, sources, torch.full_like(sources, -1))


def _gold_token_log_probability(
    logits: torch.Tensor, clean_tokens: torch.Tensor) -> torch.Tensor:
  selected = torch.gather(
    logits, -1, clean_tokens[:, :, None]).squeeze(-1)
  return selected - torch.logsumexp(logits, dim=-1)


def _masked_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    minimum_choices: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
  valid_examples = valid_mask.sum(dim=-1) >= minimum_choices
  if not bool(valid_examples.any().item()):
    finite_student = torch.where(
      torch.isfinite(student_logits), student_logits,
      torch.zeros_like(student_logits))
    return finite_student.sum() * 0.0, valid_examples
  masked_student = student_logits.masked_fill(~valid_mask, -torch.inf)
  masked_teacher = teacher_logits.masked_fill(~valid_mask, -torch.inf)
  teacher_probability = F.softmax(
    masked_teacher[valid_examples], dim=-1)
  student_log_probability = F.log_softmax(
    masked_student[valid_examples], dim=-1)
  valid_terms = valid_mask[valid_examples]
  cross_entropy = torch.where(
    valid_terms,
    teacher_probability * student_log_probability,
    torch.zeros_like(student_log_probability))
  loss = -cross_entropy.sum(-1).mean()
  return loss, valid_examples


def gold_reveal_influence_topology_loss(
    output: StructuredDecoderOutput,
    base_unary_logits: torch.Tensor,
    revealed_unary_logits: torch.Tensor,
    clean_tokens: torch.Tensor,
    active_mask: torch.Tensor,
    source_positions: torch.Tensor,
    temperature: float = 0.25,
    minimum_choices: int = 2,
    edge_weight: float = 1.0,
    anchor_weight: float = 0.25,
    slot_weight: float = 0.25,
    ) -> TopologyDistillationLoss:
  """Distill detached conditional influence into dynamic topology scores.

  For each example, ``revealed_unary_logits`` comes from a teacher input where
  exactly ``source_positions[b]`` has been replaced by its clean token.  The
  absolute change in clean-token log probability at every other masked site is
  the influence target.  This target is used only in the training loss; the
  head still receives the original corrupted context at inference.
  """
  _validate_token_batch(output, clean_tokens, active_mask)
  if base_unary_logits.shape != revealed_unary_logits.shape:
    raise ValueError('base and revealed unary logits must have equal shape')
  if base_unary_logits.shape[:2] != clean_tokens.shape:
    raise ValueError('unary logits and clean_tokens leading shapes differ')
  if source_positions.shape != (clean_tokens.shape[0],):
    raise ValueError('source_positions must have shape [B]')
  if temperature <= 0:
    raise ValueError('temperature must be positive')

  with torch.no_grad():
    base_gold = _gold_token_log_probability(
      base_unary_logits.float(), clean_tokens)
    revealed_gold = _gold_token_log_probability(
      revealed_unary_logits.float(), clean_tokens)
    node_influence = (revealed_gold - base_gold).abs() / temperature
    source_valid = source_positions >= 0
    safe_sources = source_positions.clamp_min(0)
    source_one_hot = F.one_hot(
      safe_sources, num_classes=clean_tokens.shape[1]).bool()
    teacher_node_mask = (
      active_mask & ~source_one_hot & source_valid[:, None])
    node_influence = node_influence.masked_fill(~teacher_node_mask, -torch.inf)

  # Train global anchor occupancy toward influential target positions.  The
  # log-sum-exp removes arbitrary anchor-slot permutations.
  anchor_node_logits = torch.logsumexp(output.anchor_logits, dim=-1)
  anchor_loss, anchor_valid = _masked_distillation(
    anchor_node_logits, node_influence, teacher_node_mask,
    minimum_choices=minimum_choices)

  # Train the selected source's slot router toward anchors currently occupying
  # influential positions.  Direct anchor supervision above lets this improve
  # even though hard slot occupants are detached in the forward topology.
  batch_index = torch.arange(
    clean_tokens.shape[0], device=clean_tokens.device)
  source_slot_logits = output.slot_logits[batch_index, safe_sources]
  anchor_teacher = torch.gather(
    node_influence, 1, output.anchor_indices)
  anchor_choice_mask = torch.gather(
    teacher_node_mask, 1, output.anchor_indices)
  slot_loss, slot_valid = _masked_distillation(
    source_slot_logits, anchor_teacher, anchor_choice_mask,
    minimum_choices=minimum_choices)

  # Rank the actual sparse proposals incident to the revealed source.
  proposal_left = output.proposal_edge_index[:, :, 0]
  proposal_right = output.proposal_edge_index[:, :, 1]
  incident = (
    proposal_left.eq(safe_sources[:, None])
    | proposal_right.eq(safe_sources[:, None]))
  other_node = torch.where(
    proposal_left.eq(safe_sources[:, None]),
    proposal_right, proposal_left)
  edge_teacher = torch.gather(node_influence, 1, other_node)
  edge_valid_mask = (
    output.proposal_edge_mask & incident & source_valid[:, None]
    & torch.gather(teacher_node_mask, 1, other_node))
  edge_loss, edge_valid = _masked_distillation(
    output.proposal_scores, edge_teacher, edge_valid_mask,
    minimum_choices=minimum_choices)

  loss = (
    edge_weight * edge_loss
    + anchor_weight * anchor_loss
    + slot_weight * slot_loss)
  finite_influence = torch.where(
    teacher_node_mask, node_influence, torch.zeros_like(node_influence))
  mean_influence = (
    finite_influence.sum()
    / teacher_node_mask.sum().clamp_min(1))
  valid_examples = edge_valid | anchor_valid | slot_valid
  # Coverage asks whether each student view contains enough alternatives to
  # define its distillation loss.  All three rates use the same transparent
  # teacher-eligible population, but retain separate denominators so runs can
  # be aggregated and audited without assuming that equality.
  eligible = (
    source_valid
    & (teacher_node_mask.sum(dim=-1) >= minimum_choices))

  def coverage(
      valid: torch.Tensor,
      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denominator = eligible.sum()
    numerator = (valid & eligible).sum()
    rate = (
      numerator.to(loss.dtype)
      / denominator.clamp_min(1).to(loss.dtype))
    return rate, numerator, denominator

  edge_coverage, edge_numerator, edge_denominator = coverage(edge_valid)
  anchor_coverage, anchor_numerator, anchor_denominator = coverage(
    anchor_valid)
  slot_coverage, slot_numerator, slot_denominator = coverage(slot_valid)
  return TopologyDistillationLoss(
    loss=loss,
    edge_loss=edge_loss,
    anchor_loss=anchor_loss,
    slot_loss=slot_loss,
    valid_examples=valid_examples.sum(),
    mean_influence=mean_influence,
    edge_coverage=edge_coverage,
    edge_coverage_numerator=edge_numerator,
    edge_coverage_denominator=edge_denominator,
    anchor_coverage=anchor_coverage,
    anchor_coverage_numerator=anchor_numerator,
    anchor_coverage_denominator=anchor_denominator,
    slot_coverage=slot_coverage,
    slot_coverage_numerator=slot_numerator,
    slot_coverage_denominator=slot_denominator)
