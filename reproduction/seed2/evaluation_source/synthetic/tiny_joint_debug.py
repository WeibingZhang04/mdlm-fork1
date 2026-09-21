"""Two-token learning diagnostic using the actual forest objective/samplers.

The visible context, position features, timestep, and uniform frozen unaries
are identical for every target.  Full vocabulary means K=V=2, with no tail.
Training enumerates the target distribution exactly; stochastic minibatches
cannot conceal mode collapse or add noise to the independent control.
"""

from __future__ import annotations

import dataclasses
import math
import time

import torch
import torch.nn as nn

from models.directional_forest import DirectionalCouplingForestHead
from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import (
  full_vocabulary_marginals,
  infer_structured_distribution,
  sample_structured_marginal_tokens,
  sample_structured_tokens,
  structured_token_log_probability,
)


OUTCOMES = ('AA', 'AB', 'BA', 'BB')
TOKENS = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
TARGETS = {
  'opposite': torch.tensor([0.0, 0.5, 0.5, 0.0]),
  'independent': torch.tensor([0.25, 0.25, 0.25, 0.25]),
  'equal': torch.tensor([0.5, 0.0, 0.0, 0.5]),
}


@dataclasses.dataclass(frozen=True)
class TinyVariant:
  name: str
  kind: str = 'shared'
  factor_mode: str = 'dynamic'
  init_std: float = 0.01
  warmup_steps: int = 0
  rank: int = 4


VARIANTS = (
  TinyVariant('unrestricted_table', kind='table'),
  TinyVariant('shared_static_wide', factor_mode='fixed', init_std=0.25),
  TinyVariant('shared_contextual_production'),
  TinyVariant('shared_contextual_wide', init_std=0.25),
  TinyVariant('shared_contextual_warmup', init_std=0.25, warmup_steps=100),
  TinyVariant('directional_contextual_production', kind='directional'),
  TinyVariant('directional_contextual_wide', kind='directional', init_std=0.25),
  TinyVariant('directional_static_wide', kind='directional',
              factor_mode='fixed', init_std=0.25),
  TinyVariant('directional_contextual_warmup', kind='directional',
              init_std=0.25, warmup_steps=100),
  TinyVariant('directional_contextual_rank2_wide', kind='directional',
              init_std=0.25, rank=2),
  TinyVariant('directional_contextual_rank2_warmup', kind='directional',
              init_std=0.25, warmup_steps=100, rank=2),
)


def fixed_inputs(batch_size=1, dtype=torch.float32):
  hidden = torch.tensor([[[1., 0., 1., 0., 1.],
                          [0., 1., 1., 1., 1.]]], dtype=dtype)
  return (hidden.expand(batch_size, -1, -1),
          torch.zeros(batch_size, 2, 2, dtype=dtype),
          torch.full((batch_size,), 0.5, dtype=dtype),
          torch.ones(batch_size, 2, dtype=torch.bool))


def _head(cls=ContextualCouplingForestHead, rank=4):
  return cls(hidden_size=5, vocab_size=2, top_k=2, rank=rank,
             time_embed_dim=8, topology_dim=8, local_window=1,
             num_anchor_slots=1, contextual_neighbors=0,
             component_size_cap=2, topology_mode='fixed')


def _batch_output(output, batch_size):
  fields = {field.name: getattr(output, field.name).expand(
    batch_size, *getattr(output, field.name).shape[1:])
    for field in dataclasses.fields(output)
    if torch.is_tensor(getattr(output, field.name))}
  return dataclasses.replace(output, **fields)


def table_output(template, table, epsilon=1e-12):
  """Embed a 2x2 positive table in production endpoint-factor tensors.

  The right endpoint is identity plus epsilon because production requires
  strictly positive endpoint factors.  Its perturbation is reported and is
  below numerical precision at ordinary training accuracy.
  """
  left_ids = template.candidate_ids[:, 0]
  right_ids = template.candidate_ids[:, 1]
  right_basis = torch.eye(2, dtype=table.dtype, device=table.device) + epsilon
  return dataclasses.replace(
    template, pair_left_factors=table[left_ids][:, None],
    pair_right_factors=right_basis[right_ids][:, None])


class TinyModel(nn.Module):
  def __init__(self, variant, seed):
    super().__init__()
    torch.manual_seed(seed)
    self.variant = variant
    if variant.kind == 'table':
      self.table_logits = nn.Parameter(torch.zeros(2, 2))
      with torch.no_grad():
        hidden, logits, timestep, active = fixed_inputs()
        self.template = _head()(hidden, logits, timestep, active)
    else:
      cls = (DirectionalCouplingForestHead if variant.kind == 'directional'
             else ContextualCouplingForestHead)
      self.head = _head(cls, rank=variant.rank)
      if variant.init_std != 0.01:
        generator = torch.Generator().manual_seed(seed + 1729)
        for name in ('token_factor_embedding', 'right_token_factor_embedding'):
          if hasattr(self.head, name):
            nn.init.normal_(getattr(self.head, name).weight,
                            mean=math.log(math.expm1(1.0)),
                            std=variant.init_std, generator=generator)

  def forward(self, step=None):
    hidden, logits, timestep, active = fixed_inputs()
    if self.variant.kind == 'table':
      # A common offset cancels under global normalization.
      table = (self.table_logits - self.table_logits.max()).exp()
      output = table_output(self.template, table)
    else:
      mode = ('fixed' if step is not None and step <= self.variant.warmup_steps
              else self.variant.factor_mode)
      output = self.head(hidden, logits, timestep, active, factor_mode=mode)
    return output, logits, active


def exact_log_probabilities(output, logits, active, backend='low_rank'):
  """Enumerate all four token outcomes via the production likelihood API."""
  expanded = _batch_output(output, 4)
  expanded_logits = logits.expand(4, -1, -1)
  expanded_active = active.expand(4, -1)
  inference = infer_structured_distribution(expanded, expanded_active,
                                           backend=backend)
  return structured_token_log_probability(
    expanded, expanded_logits, TOKENS.to(logits.device), expanded_active,
    inference)


def distribution_metrics(probabilities, target):
  probability = probabilities.detach().double()
  normalization_error = abs(float(probability.sum()) - 1.0)
  # Accumulating four FP32 likelihoods can leave ~1e-7 normalization error.
  # Report it separately; normalize in FP64 before KL and MI so that a good
  # independent fit does not acquire an artificial negative divergence.
  probability = probability / probability.sum()
  target = target.double()
  positive = target > 0
  marginals = torch.stack((probability.reshape(2, 2).sum(1),
                           probability.reshape(2, 2).sum(0)))
  target_marginals = torch.stack((target.reshape(2, 2).sum(1),
                                  target.reshape(2, 2).sum(0)))
  independent = marginals[0, :, None] * marginals[1, None, :]
  invalid = target == 0
  return {
    'probabilities': dict(zip(OUTCOMES, probability.tolist())),
    'normalization_error': normalization_error,
    'kl_target_to_model': float((target[positive] * (
      target[positive].log() - probability[positive].log())).sum()),
    'total_variation': float((target - probability).abs().sum() / 2),
    'invalid_mass': float(probability[invalid].sum()),
    'marginals': marginals.tolist(),
    'max_marginal_error': float((marginals - target_marginals).abs().max()),
    'independent_marginal_probabilities': dict(zip(
      OUTCOMES, independent.flatten().tolist())),
    'independent_marginal_invalid_mass': float(
      independent.flatten()[invalid].sum()),
    'mutual_information': float((probability * (
      probability.clamp_min(1e-300).log()
      - independent.flatten().clamp_min(1e-300).log())).sum()),
  }


@torch.no_grad()
def evaluate_output(output, logits, active, target, samples=20000, seed=1):
  probability = exact_log_probabilities(output, logits, active).exp()
  result = distribution_metrics(probability, target)
  dense = exact_log_probabilities(output, logits, active, backend='dense').exp()
  result['max_dense_low_rank_probability_error'] = float(
    (dense - probability).abs().max())
  inference = infer_structured_distribution(output, active, backend='low_rank')
  marginals = full_vocabulary_marginals(output, logits, active, inference)[0]
  result['max_enumerated_inference_marginal_error'] = float(
    (marginals.double() - torch.tensor(
      result['marginals'], dtype=torch.float64)).abs().max())
  if samples:
    for name, sampler in (('joint', sample_structured_tokens),
                          ('marginal', sample_structured_marginal_tokens)):
      draws = sampler(output, logits, active, num_samples=samples,
                      generator=torch.Generator().manual_seed(seed),
                      inference=inference)[0]
      counts = torch.bincount(draws[:, 0] * 2 + draws[:, 1], minlength=4)
      frequencies = counts.double() / samples
      result[f'{name}_sample_frequencies'] = dict(zip(
        OUTCOMES, frequencies.tolist()))
      result[f'{name}_sample_invalid_rate'] = float(
        frequencies[target == 0].sum())
    result['samples_per_sampler'] = samples
  return result


def oracle_result(task, samples=20000, seed=1):
  # Zero-probability factors are excluded by the production interface.  Use a
  # 1e-12 floor and report its exact nonzero error against the true target.
  target = TARGETS[task].double()
  with torch.no_grad():
    hidden, logits, timestep, active = fixed_inputs(dtype=torch.float64)
    template = _head().double()(hidden, logits, timestep, active)
    output = table_output(template, target.reshape(2, 2) + 1e-12)
    metrics = evaluate_output(output, logits, active, target, samples, seed)
  return {'task': task, 'variant': 'oracle_positive_floor',
          'target': dict(zip(OUTCOMES, target.tolist())),
          'factor_epsilon': 1e-12, **metrics}


def train_tiny(variant, task, seed, max_steps=600, learning_rate=0.03,
               samples=20000, check_every=25):
  if max_steps < 1 or learning_rate <= 0 or check_every < 1:
    raise ValueError('steps, learning rate, and check interval must be positive')
  model = TinyModel(variant, seed)
  target = TARGETS[task]
  positive = target > 0
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  history = []
  active_names = set()
  start = time.perf_counter()
  stopped = 'max_steps'
  for step in range(1, max_steps + 1):
    output, logits, active = model(step)
    log_probabilities = exact_log_probabilities(output, logits, active)
    loss = -(target[positive] * log_probabilities[positive]).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    active_names.update(name for name, value in model.named_parameters()
                        if value.grad is not None)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    if step == 1 or step % check_every == 0 or step == max_steps:
      with torch.no_grad():
        current = model()
        metrics = distribution_metrics(
          exact_log_probabilities(*current).exp(), target)
      history.append({'step': step, 'nll_before_update': float(loss.detach()),
                      'gradient_norm': float(gradient_norm),
                      'kl': metrics['kl_target_to_model'],
                      'invalid_mass': metrics['invalid_mass'],
                      'probabilities': metrics['probabilities']})
      if step >= max(50, variant.warmup_steps + 50):
        if (metrics['kl_target_to_model'] < 0.01
            and metrics['invalid_mass'] < 0.01
            and metrics['max_marginal_error'] < 0.02):
          stopped = 'distribution_target_reached'
          break
        if (task == 'opposite' and variant.kind == 'shared'
            and metrics['kl_target_to_model'] <= math.log(2) + 1e-5):
          stopped = 'proven_shared_head_capacity_bound_reached'
          break
  output, logits, active = model()
  evaluation = evaluate_output(output, logits, active, target,
                               samples=samples, seed=seed + 100000)
  parameters = dict(model.named_parameters())
  return {
    'task': task, 'variant': variant.name, 'seed': seed,
    'variant_config': dataclasses.asdict(variant), 'max_steps': max_steps,
    'steps_run': step, 'stop_reason': stopped,
    'learning_rate': learning_rate, 'seconds': time.perf_counter() - start,
    'parameter_count': sum(p.numel() for p in parameters.values()),
    'pair_factor_rank': output.pair_left_factors.shape[-1],
    'gradient_active_parameter_count': sum(parameters[n].numel()
                                           for n in active_names),
    'active_parameter_names': sorted(active_names),
    'history': history, **evaluation,
  }
