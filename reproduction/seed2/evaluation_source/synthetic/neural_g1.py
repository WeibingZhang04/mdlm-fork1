"""Learned frozen-backbone Gate-1 benchmark for contextual forests."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import structured_utils
from evaluation.structured_metrics import (
  edge_scores,
  kl_divergence,
  total_variation,
)
from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import (
  infer_structured_distribution,
  sample_structured_tokens,
  structured_token_log_probability,
)
from synthetic.distributions import ContextSwitchingMatching, Edge
from synthetic.g1_benchmark import paired_bootstrap_ci


# This is the protocol frozen after development on seeds 1--8 and used for
# the reported held-out result.  Keep the CLI, named config, and dataclass
# defaults aligned with these constants.
REPORTED_HELDOUT_SEEDS = (9, 10, 11, 12, 13)
REPORTED_TRAINING_STEPS = 900
REPORTED_FACTOR_WARMUP_STEPS = 300
REPORTED_FACTOR_INIT_STD = 0.25
REPORTED_FACTOR_INIT_SEED = 1729


@dataclasses.dataclass(frozen=True)
class NeuralModelSpec:
  name: str
  topology_mode: str
  factor_mode: str
  independent_mode: bool = False
  fixed_edges: Optional[tuple[Edge, ...]] = None


@dataclasses.dataclass(frozen=True)
class NeuralTrainConfig:
  steps: int = REPORTED_TRAINING_STEPS
  batch_size: int = 64
  learning_rate: float = 1e-2
  dependency_weight: float = 1.0
  factor_init_std: float = REPORTED_FACTOR_INIT_STD
  factor_init_seed: int = REPORTED_FACTOR_INIT_SEED
  factor_warmup_steps: int = REPORTED_FACTOR_WARMUP_STEPS
  gradient_clip: float = 5.0
  eval_samples: int = 20000
  log_every: int = 100
  inference_backend: str = 'auto'


def model_specs(task: ContextSwitchingMatching) -> dict[str, NeuralModelSpec]:
  # With symmetric contexts the corpus-level optimum is tied; fixing the first
  # matching is deterministic and does not privilege the contextual model.
  static_edges = task.matchings[0]
  return {
    # Legacy identifier retained for artifact compatibility.  Pair parameters
    # are inactive, so this is only an architecture-count/no-edge control.
    'parameter_matched_independent': NeuralModelSpec(
      'parameter_matched_independent', 'dynamic', 'dynamic', True),
    'natural_chain': NeuralModelSpec(
      'natural_chain', 'fixed', 'dynamic'),
    'static_forest': NeuralModelSpec(
      'static_forest', 'dynamic', 'fixed', fixed_edges=static_edges),
    'fixed_topology_dynamic_factors': NeuralModelSpec(
      'fixed_topology_dynamic_factors', 'dynamic', 'dynamic',
      fixed_edges=static_edges),
    'dynamic_topology_fixed_factors': NeuralModelSpec(
      'dynamic_topology_fixed_factors', 'dynamic', 'fixed'),
    'contextual_forest': NeuralModelSpec(
      'contextual_forest', 'dynamic', 'dynamic'),
  }


class FrozenContextFeatures(nn.Module):
  """Deterministic context/position features standing in for a frozen LM."""

  def __init__(self, num_contexts: int, length: int, vocab_size: int):
    super().__init__()
    self.num_contexts = num_contexts
    self.length = length
    self.vocab_size = vocab_size
    self.hidden_size = length + num_contexts + num_contexts * length + 2
    features = torch.zeros(
      num_contexts, length, self.hidden_size, dtype=torch.float32)
    for context in range(num_contexts):
      for position in range(length):
        features[context, position, position] = 1.0
        features[context, position, length + context] = 1.0
        cross_offset = length + num_contexts + context * length
        features[context, position, cross_offset + position] = 1.0
        features[context, position, -2] = position / max(length - 1, 1)
        features[context, position, -1] = 1.0
    self.register_buffer('features', features, persistent=True)
    self.register_buffer(
      'unary_logits',
      torch.zeros(num_contexts, length, vocab_size),
      persistent=True)

  def forward(self, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.features[contexts], self.unary_logits[contexts]


class SyntheticForestAdapter(nn.Module):
  """A coupling head trained on top of frozen, target-independent features."""

  def __init__(self, task: ContextSwitchingMatching,
               spec: NeuralModelSpec,
               factor_init_std: float = 0.25,
               factor_init_seed: int = 1729):
    super().__init__()
    if factor_init_std <= 0.0:
      raise ValueError('factor_init_std must be positive')
    self.task = task
    self.spec = spec
    self.backbone = FrozenContextFeatures(
      task.num_contexts, task.length, task.vocab_size)
    self.head = ContextualCouplingForestHead(
      hidden_size=self.backbone.hidden_size,
      vocab_size=task.vocab_size,
      top_k=task.vocab_size,
      rank=max(task.vocab_size, 4),
      time_embed_dim=16,
      topology_dim=32,
      local_window=task.length - 1,
      num_anchor_slots=2,
      contextual_neighbors=0,
      component_size_cap=2,
      topology_mode=spec.topology_mode,
      factor_mode=spec.factor_mode,
      independent_mode=spec.independent_mode,
      min_edge_score=0.0)
    # The production head starts extremely close to the neutral factor.  That
    # is conservative for a pretrained LM but creates a near-symmetric saddle
    # in this tiny vocabulary.  A generic (label-agnostic) wider random start
    # lets the structural gate test learning rather than symmetry breaking.
    inverse_softplus_one = math.log(math.e - 1.0)
    factor_generator = torch.Generator(device='cpu').manual_seed(
      factor_init_seed)
    nn.init.normal_(
      self.head.token_factor_embedding.weight,
      mean=inverse_softplus_one, std=factor_init_std,
      generator=factor_generator)
    if spec.fixed_edges is not None:
      self.register_buffer(
        'fixed_edges', torch.tensor(spec.fixed_edges, dtype=torch.long),
        persistent=True)
    else:
      self.fixed_edges = None
    self.register_buffer(
      'dependency_adjacency', dependency_adjacency(task), persistent=True)

  def forward(self, contexts: torch.Tensor, timestep: torch.Tensor,
              factor_mode: Optional[str] = None):
    hidden, unary_logits = self.backbone(contexts)
    active = torch.ones(
      contexts.shape[0], self.task.length,
      dtype=torch.bool, device=contexts.device)
    kwargs = {}
    if self.fixed_edges is not None:
      kwargs['fixed_edge_index'] = self.fixed_edges
    else:
      kwargs['topology_mode'] = self.spec.topology_mode
    output = self.head(
      hidden, unary_logits, timestep, active,
      factor_mode=factor_mode or self.spec.factor_mode,
      independent_mode=self.spec.independent_mode,
      **kwargs)
    return output, unary_logits, active


def sample_training_batch(
    task: ContextSwitchingMatching,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  contexts = torch.randint(
    task.num_contexts, (batch_size,), generator=generator, device=device)
  tokens = torch.randint(
    task.vocab_size, (batch_size, task.length),
    generator=generator, device=device)
  for context, matching in enumerate(task.matchings):
    rows = contexts.eq(context)
    for first, second in matching:
      tokens[rows, second] = tokens[rows, first]
  timestep = torch.rand(batch_size, generator=generator, device=device)
  return contexts, tokens, timestep


def dependency_adjacency(
    task: ContextSwitchingMatching,
    device: Optional[torch.device] = None) -> torch.Tensor:
  """Compute the exact context-level conditional-influence table once."""
  adjacency = torch.zeros(
    task.num_contexts, task.length, task.length,
    dtype=torch.bool, device=device)
  for context in range(task.num_contexts):
    for first in range(task.length):
      for second in range(first + 1, task.length):
        pair = task.pair_marginal(first, second, context)
        first_marginal = pair.sum(axis=1, keepdims=True)
        second_marginal = pair.sum(axis=0, keepdims=True)
        independent = first_marginal @ second_marginal
        positive = pair > 0
        influence = float(np.sum(
          pair[positive]
          * (np.log(pair[positive]) - np.log(independent[positive]))))
        if influence > 1e-8:
          adjacency[context, first, second] = True
          adjacency[context, second, first] = True
  return adjacency


def dependency_targets(
    task: ContextSwitchingMatching,
    contexts: torch.Tensor,
    edge_index: torch.Tensor,
    adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
  # This is the exact synthetic counterpart of the paper's cached
  # conditional-influence target: an edge is positive when revealing one
  # endpoint changes the other endpoint's distribution.  Mutual information
  # computes that criterion without reading ``task.true_edges``.  It is still
  # privileged supervised topology and should be reported as a favorable-
  # supervision diagnostic, not as unsupervised graph discovery.
  if adjacency is None:
    adjacency = dependency_adjacency(task, contexts.device)
  expected = (task.num_contexts, task.length, task.length)
  if adjacency.shape != expected or adjacency.dtype != torch.bool:
    raise ValueError(
      f'adjacency must be boolean with shape {expected}')
  adjacency = adjacency.to(contexts.device)
  return adjacency[
    contexts[:, None], edge_index[:, :, 0], edge_index[:, :, 1]]


def dependency_loss(
    task: ContextSwitchingMatching,
    contexts: torch.Tensor,
    output,
    adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
  valid = output.proposal_edge_mask
  targets = dependency_targets(
    task, contexts, output.proposal_edge_index, adjacency).to(
      output.proposal_scores.dtype)
  logits = output.proposal_scores.masked_select(valid)
  targets = targets.masked_select(valid)
  positives = targets.sum().clamp_min(1.0)
  negatives = (1.0 - targets).sum().clamp_min(1.0)
  return F.binary_cross_entropy_with_logits(
    logits, targets, pos_weight=negatives / positives)


def train_adapter(
    task: ContextSwitchingMatching,
    spec: NeuralModelSpec,
    seed: int,
    config: NeuralTrainConfig,
    device: torch.device) -> tuple[SyntheticForestAdapter, list[dict[str, float]]]:
  torch.manual_seed(seed)
  if device.type == 'cuda':
    torch.cuda.manual_seed_all(seed)
  generator = torch.Generator(device=device).manual_seed(seed + 10_000)
  if config.inference_backend not in {'auto', 'dense', 'low_rank'}:
    raise ValueError(
      "inference_backend must be 'auto', 'dense', or 'low_rank'")
  if config.factor_warmup_steps < 0:
    raise ValueError('factor_warmup_steps must be nonnegative')
  model = SyntheticForestAdapter(
    task, spec,
    factor_init_std=config.factor_init_std,
    factor_init_seed=config.factor_init_seed).to(device)
  optimizer = torch.optim.AdamW(
    model.head.parameters(), lr=config.learning_rate, weight_decay=1e-4)
  history = []
  start = time.perf_counter()
  for step in range(1, config.steps + 1):
    contexts, tokens, timestep = sample_training_batch(
      task, config.batch_size, generator, device)
    warmup_factor_mode = (
      'fixed'
      if (spec.factor_mode == 'dynamic'
          and step <= config.factor_warmup_steps)
      else None)
    output, unary_logits, active = model(
      contexts, timestep, factor_mode=warmup_factor_mode)
    inference = infer_structured_distribution(
      output, active, backend=config.inference_backend)
    nll = -structured_token_log_probability(
      output, unary_logits, tokens, active, inference).mean()
    dep = dependency_loss(
      task, contexts, output, model.dependency_adjacency)
    loss = nll + config.dependency_weight * dep
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
      model.head.parameters(), config.gradient_clip)
    optimizer.step()
    if (step == 1 or step == config.steps
        or step % config.log_every == 0):
      history.append({
        'step': float(step),
        'loss': float(loss.detach()),
        'structured_nll': float(nll.detach()),
        'dependency_loss': float(dep.detach()),
        'gradient_norm': float(gradient_norm.detach()),
        'elapsed_seconds': time.perf_counter() - start,
      })
  return model, history


def _token_distribution_from_enumeration(
    output,
    inference,
    vocab_size: int) -> np.ndarray:
  reference = structured_utils.enumerate_forest_distribution(
    output.unary_log_potentials.double(),
    structured_utils.positive_pair_factors_to_log(
      structured_utils.materialize_low_rank_pair_factors(
        output.pair_left_factors.double(),
        output.pair_right_factors.double(),
        edge_mask=output.edge_mask)),
    output.edge_index,
    edge_mask=output.edge_mask,
    state_mask=output.candidate_state_mask,
    max_configurations=100_000)
  states = reference.configurations
  log_probability = reference.log_probabilities[0]
  finite = torch.isfinite(log_probability)
  states = states[finite]
  probabilities = log_probability[finite].exp()
  candidates = output.candidate_ids[0].cpu()
  positions = torch.arange(candidates.shape[0])[None, :]
  tokens = candidates[positions, states.cpu()]
  powers = torch.tensor([
    vocab_size ** exponent
    for exponent in reversed(range(tokens.shape[1]))], dtype=torch.long)
  lexicographic_index = (tokens * powers[None]).sum(dim=-1)
  distribution = torch.zeros(
    vocab_size ** tokens.shape[1], dtype=torch.float64)
  distribution.scatter_add_(0, lexicographic_index, probabilities.cpu())
  return distribution.numpy()


@torch.no_grad()
def evaluate_adapter(
    model: SyntheticForestAdapter,
    seed: int,
    eval_samples: int = 20000) -> list[dict[str, object]]:
  model.eval()
  task = model.task
  device = next(model.head.parameters()).device
  rows = []
  for context in range(task.num_contexts):
    contexts = torch.tensor([context], device=device)
    timestep = torch.tensor([0.5], device=device)
    output, unary_logits, active = model(contexts, timestep)
    inference = infer_structured_distribution(output, active)
    probability = _token_distribution_from_enumeration(
      output, inference, task.vocab_size)
    target = task.probabilities(context)
    valid = task.is_valid(task.support(), context)
    samples = sample_structured_tokens(
      output, unary_logits, active,
      num_samples=eval_samples,
      generator=torch.Generator(device=device).manual_seed(
        seed * 100 + context),
      inference=inference)[0].cpu().numpy()
    predicted = []
    for edge, enabled in zip(
        output.edge_index[0].cpu().tolist(),
        output.edge_mask[0].cpu().tolist()):
      if enabled:
        predicted.append(tuple(edge))
    scores = edge_scores(predicted, task.true_edges(context))
    rows.append({
      'seed': seed,
      'context': context,
      'model': model.spec.name,
      'kl': kl_divergence(target, probability),
      'tv': total_variation(target, probability),
      'invalid_rate': float(probability[~valid].sum()),
      'sampled_invalid_rate': float(
        1.0 - task.is_valid(samples, context).mean()),
      'edge_precision': scores['precision'],
      'edge_recall': scores['recall'],
      'edge_f1': scores['f1'],
      'predicted_edges': [list(edge) for edge in predicted],
      'true_edges': [list(edge) for edge in task.true_edges(context)],
      'retained_mass': float(output.retained_mass.mean().cpu()),
    })
  return rows


def _mean(rows: Sequence[dict[str, object]], seed: int,
          model: str, metric: str) -> float:
  values = [float(row[metric]) for row in rows
            if int(row['seed']) == seed and row['model'] == model]
  if not values:
    raise ValueError(f'missing {seed}/{model}/{metric}')
  return float(np.mean(values))


def evaluate_neural_gate(
    rows: Sequence[dict[str, object]],
    seeds: Sequence[int]) -> dict[str, object]:
  static_models = (
    'natural_chain', 'static_forest',
    'fixed_topology_dynamic_factors')
  improvements = []
  reductions = []
  invalids = []
  edge_f1s = []
  best_static_values = []
  contextual_values = []
  for seed in seeds:
    contextual = _mean(rows, seed, 'contextual_forest', 'tv')
    best_static = min(_mean(rows, seed, model, 'tv')
                      for model in static_models)
    independent = _mean(
      rows, seed, 'parameter_matched_independent', 'invalid_rate')
    joint = _mean(rows, seed, 'contextual_forest', 'invalid_rate')
    improvements.append(best_static - contextual)
    reductions.append((independent - joint) / max(independent, 1e-12))
    invalids.append(joint)
    edge_f1s.append(_mean(rows, seed, 'contextual_forest', 'edge_f1'))
    best_static_values.append(best_static)
    contextual_values.append(contextual)
  best_static_mean = float(np.mean(best_static_values))
  contextual_mean = float(np.mean(contextual_values))
  relative_tv = (
    best_static_mean - contextual_mean) / max(best_static_mean, 1e-12)
  screen_checks = {
    'joint_invalid_absolute': float(np.mean(invalids)) <= 0.05,
    'joint_invalid_relative_reduction': float(np.mean(reductions)) >= 0.80,
    'contextual_tv_relative_reduction': relative_tv >= 0.25,
    'contextual_edge_f1': float(np.mean(edge_f1s)) >= 0.80,
    'all_seeds_same_direction': all(value > 0 for value in improvements),
  }
  result = {
    'gate_name': 'g1_learned_frozen_adapter',
    'scientific_scope': (
      'frozen target-independent features with exact synthetic '
      'conditional-influence supervision for topology; this is a '
      'favorable-supervision diagnostic, not unsupervised graph discovery'),
    'screen_passed': bool(all(screen_checks.values())),
    'checks': screen_checks,
    'metrics': {
      'contextual_invalid_rate': float(np.mean(invalids)),
      'invalid_relative_reduction': float(np.mean(reductions)),
      'best_static_tv': best_static_mean,
      'contextual_tv': contextual_mean,
      'contextual_tv_relative_reduction': relative_tv,
      'contextual_edge_f1': float(np.mean(edge_f1s)),
      'paired_tv_improvements_by_seed': improvements,
    },
  }
  if len(seeds) >= 3:
    interval = paired_bootstrap_ci(improvements)
    ci_check = interval[0] > 0
    result['checks']['paired_95pct_ci_excludes_zero'] = ci_check
    result['metrics']['paired_tv_improvement_95pct_ci'] = list(interval)
    result['passed'] = bool(result['screen_passed'] and ci_check)
  else:
    result['passed'] = False
    result['pending_confirmation'] = 'requires at least three seeds'
  return result
