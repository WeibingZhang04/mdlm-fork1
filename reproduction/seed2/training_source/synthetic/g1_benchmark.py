"""Finite-data structural sanity gate for contextual coupling forests.

This module intentionally does not claim to train the neural decoder.  It fits
an additively smoothed empirical forest projection of small, exactly enumerable
target distributions from sampled data.  The result is an auditable
oracle/table-fit gate: it verifies the task, topology ablations, sampling code,
and frozen decision rule can distinguish adaptive topology before GPU
time is spent on the learned head.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping, Sequence

import numpy as np

from evaluation.structured_metrics import (
  edge_scores,
  kl_divergence,
  total_variation,
)
from synthetic.distributions import (
  AmbiguousPairs,
  ContextSwitchingMatching,
  CyclicXOR,
  Edge,
  ExactSyntheticDistribution,
  MarkovLanguage,
  canonical_edges,
  contexts_for,
)


MODELS = (
  'factorized',
  'parameter_matched_independent',
  'natural_chain',
  'static_forest',
  'fixed_topology_dynamic_factors',
  'dynamic_topology_fixed_factors',
  'contextual_forest',
)


@dataclasses.dataclass(frozen=True)
class EstimatedMoments:
  node: np.ndarray
  pair: Mapping[Edge, np.ndarray]


@dataclasses.dataclass(frozen=True)
class BenchmarkRecord:
  seed: int
  task: str
  context: int
  model: str
  sampler: str
  kl: float
  tv: float
  invalid_rate: float
  sampled_invalid_rate: float
  edge_precision: float
  edge_recall: float
  edge_f1: float
  predicted_edges: tuple[Edge, ...]
  true_edges: tuple[Edge, ...]

  def as_dict(self) -> dict[str, object]:
    result = dataclasses.asdict(self)
    result['predicted_edges'] = [list(edge) for edge in self.predicted_edges]
    result['true_edges'] = [list(edge) for edge in self.true_edges]
    return result


def all_edges(length: int) -> tuple[Edge, ...]:
  return tuple((first, second)
               for first in range(length)
               for second in range(first + 1, length))


def estimate_moments(
    samples: np.ndarray,
    vocab_size: int,
    alpha: float = 0.25) -> EstimatedMoments:
  """Estimate consistent-enough smoothed moments for a finite-data fit."""
  samples = np.asarray(samples, dtype=np.int64)
  if samples.ndim != 2 or samples.shape[0] == 0:
    raise ValueError('samples must be a non-empty matrix')
  if alpha <= 0:
    raise ValueError('alpha must be positive')
  length = samples.shape[1]
  node = np.full((length, vocab_size), alpha, dtype=np.float64)
  for position in range(length):
    np.add.at(node[position], samples[:, position], 1.0)
  node /= node.sum(axis=1, keepdims=True)

  pair = {}
  for first, second in all_edges(length):
    counts = np.full((vocab_size, vocab_size), alpha, dtype=np.float64)
    np.add.at(counts, (samples[:, first], samples[:, second]), 1.0)
    pair[(first, second)] = counts / counts.sum()
  return EstimatedMoments(node=node, pair=pair)


def mutual_information(joint: np.ndarray) -> float:
  joint = np.asarray(joint, dtype=np.float64)
  joint = joint / joint.sum()
  first = joint.sum(axis=1, keepdims=True)
  second = joint.sum(axis=0, keepdims=True)
  independent = first @ second
  positive = joint > 0
  return float(np.sum(
    joint[positive]
    * (np.log(joint[positive]) - np.log(independent[positive]))))


def maximum_weight_forest(
    length: int,
    weights: Mapping[Edge, float],
    edge_budget: int) -> tuple[Edge, ...]:
  """Deterministic Kruskal maximum-weight forest."""
  if edge_budget < 0 or edge_budget >= length:
    raise ValueError('edge_budget must be in [0, length - 1]')
  parent = list(range(length))

  def find(node: int) -> int:
    while parent[node] != node:
      parent[node] = parent[parent[node]]
      node = parent[node]
    return node

  chosen = []
  ranked = sorted(
    ((float(score), edge) for edge, score in weights.items()),
    key=lambda item: (-item[0], item[1]))
  for _, (first, second) in ranked:
    root_first, root_second = find(first), find(second)
    if root_first == root_second:
      continue
    parent[root_second] = root_first
    chosen.append((first, second))
    if len(chosen) == edge_budget:
      break
  if len(chosen) != edge_budget:
    raise ValueError('not enough acyclic candidate edges for edge budget')
  return canonical_edges(chosen)


def forest_projection(
    support: np.ndarray,
    moments: EstimatedMoments,
    edges: Iterable[Edge]) -> np.ndarray:
  """Construct the forest MLE/I-projection from node and pair moments."""
  support = np.asarray(support, dtype=np.int64)
  edges = canonical_edges(edges)
  log_prob = np.zeros(support.shape[0], dtype=np.float64)
  for position in range(support.shape[1]):
    log_prob += np.log(moments.node[position, support[:, position]])
  for first, second in edges:
    joint = moments.pair[(first, second)]
    first_values = support[:, first]
    second_values = support[:, second]
    log_prob += np.log(joint[first_values, second_values])
    log_prob -= np.log(moments.node[first, first_values])
    log_prob -= np.log(moments.node[second, second_values])
  log_prob -= log_prob.max()
  probability = np.exp(log_prob)
  return probability / probability.sum()


def shared_factor_projection(
    support: np.ndarray,
    node_marginals: np.ndarray,
    shared_pair: np.ndarray,
    edges: Iterable[Edge]) -> np.ndarray:
  """Fit with one context-independent token coupling shared by all edges."""
  support = np.asarray(support, dtype=np.int64)
  shared_pair = np.asarray(shared_pair, dtype=np.float64)
  shared_pair = shared_pair / shared_pair.sum()
  shared_first = shared_pair.sum(axis=1)
  shared_second = shared_pair.sum(axis=0)
  log_ratio = (
    np.log(shared_pair)
    - np.log(shared_first[:, None])
    - np.log(shared_second[None, :]))
  log_prob = np.zeros(support.shape[0], dtype=np.float64)
  for position in range(support.shape[1]):
    log_prob += np.log(node_marginals[position, support[:, position]])
  for first, second in canonical_edges(edges):
    log_prob += log_ratio[support[:, first], support[:, second]]
  log_prob -= log_prob.max()
  probability = np.exp(log_prob)
  return probability / probability.sum()


def _sample_from_probability(
    support: np.ndarray,
    probability: np.ndarray,
    count: int,
    rng: np.random.Generator) -> np.ndarray:
  return support[rng.choice(support.shape[0], size=count, p=probability)]


def _edge_budget(task: ExactSyntheticDistribution) -> int:
  contexts = contexts_for(task)
  true_count = max(len(task.true_edges(context)) for context in contexts)
  # XOR has no pairwise ground-truth edge; allow the strongest possible tree
  # to demonstrate that even an oracle pairwise forest cannot express parity.
  return min(task.length - 1, max(1, true_count))


def evaluate_task(
    task_name: str,
    task: ExactSyntheticDistribution,
    seed: int,
    train_samples_per_context: int = 4096,
    eval_samples_per_model: int = 20000,
    alpha: float = 0.25) -> list[BenchmarkRecord]:
  """Fit all structural ablations and evaluate exact and sampled metrics."""
  rng = np.random.default_rng(seed)
  contexts = tuple(contexts_for(task))
  support = task.support()
  training = {
    context: task.sample(train_samples_per_context, context=context, rng=rng)
    for context in contexts
  }
  moments = {
    context: estimate_moments(training[context], task.vocab_size, alpha)
    for context in contexts
  }
  pooled = estimate_moments(
    np.concatenate([training[context] for context in contexts], axis=0),
    task.vocab_size,
    alpha)
  edge_budget = _edge_budget(task)
  chain = tuple((index, index + 1) for index in range(task.length - 1))
  if len(chain) > edge_budget:
    chain = chain[:edge_budget]

  per_context_weights = {
    context: {
      edge: mutual_information(moments[context].pair[edge])
      for edge in all_edges(task.length)
    }
    for context in contexts
  }
  average_weights = {
    edge: float(np.mean([
      per_context_weights[context][edge] for context in contexts]))
    for edge in all_edges(task.length)
  }
  pooled_weights = {
    edge: mutual_information(pooled.pair[edge])
    for edge in all_edges(task.length)
  }
  fixed_topology = maximum_weight_forest(
    task.length, average_weights, edge_budget)
  static_topology = maximum_weight_forest(
    task.length, pooled_weights, edge_budget)
  dynamic_topology = {
    context: maximum_weight_forest(
      task.length, per_context_weights[context], edge_budget)
    for context in contexts
  }

  # A single positive token-pair table is shared across contexts/edges for the
  # topology-only ablation. Correct matching edges all implement the same
  # equality relation, so this ablation can succeed without dynamic factors.
  shared_counts = np.full(
    (task.vocab_size, task.vocab_size), alpha, dtype=np.float64)
  for context in contexts:
    samples = training[context]
    for first, second in dynamic_topology[context]:
      np.add.at(shared_counts, (samples[:, first], samples[:, second]), 1.0)
  shared_pair = shared_counts / shared_counts.sum()

  records = []
  for context in contexts:
    model_distributions = {
      'factorized': forest_projection(support, moments[context], ()),
      'parameter_matched_independent': forest_projection(
        support, moments[context], ()),
      'natural_chain': forest_projection(
        support, moments[context], chain),
      'static_forest': forest_projection(
        support, pooled, static_topology),
      'fixed_topology_dynamic_factors': forest_projection(
        support, moments[context], fixed_topology),
      'dynamic_topology_fixed_factors': shared_factor_projection(
        support, moments[context].node, shared_pair,
        dynamic_topology[context]),
      'contextual_forest': forest_projection(
        support, moments[context], dynamic_topology[context]),
    }
    model_edges = {
      'factorized': (),
      'parameter_matched_independent': (),
      'natural_chain': chain,
      'static_forest': static_topology,
      'fixed_topology_dynamic_factors': fixed_topology,
      'dynamic_topology_fixed_factors': dynamic_topology[context],
      'contextual_forest': dynamic_topology[context],
    }
    target = task.probabilities(context)
    valid = task.is_valid(support, context)
    for model in MODELS:
      probability = model_distributions[model]
      sampled = _sample_from_probability(
        support, probability, eval_samples_per_model, rng)
      scores = edge_scores(model_edges[model], task.true_edges(context))
      records.append(BenchmarkRecord(
        seed=seed,
        task=task_name,
        context=context,
        model=model,
        sampler=('independent_marginals'
                 if model in {'factorized', 'parameter_matched_independent'}
                 else 'enumerated_joint'),
        kl=kl_divergence(target, probability),
        tv=total_variation(target, probability),
        invalid_rate=float(probability[~valid].sum()),
        sampled_invalid_rate=float(
          1.0 - task.is_valid(sampled, context).mean()),
        edge_precision=scores['precision'],
        edge_recall=scores['recall'],
        edge_f1=scores['f1'],
        predicted_edges=model_edges[model],
        true_edges=task.true_edges(context)))
  return records


def default_tasks() -> dict[str, ExactSyntheticDistribution]:
  return {
    'ambiguous_pairs': AmbiguousPairs(),
    'markov_025': MarkovLanguage(length=6, vocab_size=4, coupling=0.25),
    'markov_050': MarkovLanguage(length=6, vocab_size=4, coupling=0.50),
    'markov_075': MarkovLanguage(length=6, vocab_size=4, coupling=0.75),
    'markov_090': MarkovLanguage(length=6, vocab_size=4, coupling=0.90),
    'context_switching_matching': ContextSwitchingMatching(vocab_size=4),
    'cyclic_xor': CyclicXOR(),
  }


def run_benchmark(
    seeds: Sequence[int] = (1, 2, 3),
    train_samples_per_context: int = 4096,
    eval_samples_per_model: int = 20000,
    alpha: float = 0.25) -> list[BenchmarkRecord]:
  records = []
  for seed in seeds:
    for name, task in default_tasks().items():
      records.extend(evaluate_task(
        task_name=name,
        task=task,
        seed=int(seed),
        train_samples_per_context=train_samples_per_context,
        eval_samples_per_model=eval_samples_per_model,
        alpha=alpha))
  return records


def _mean_for(
    records: Sequence[BenchmarkRecord],
    task: str,
    seed: int,
    model: str,
    field: str) -> float:
  values = [getattr(record, field) for record in records
            if record.task == task
            and record.seed == seed
            and record.model == model]
  if not values:
    raise ValueError(f'no records for {task}/{seed}/{model}/{field}')
  return float(np.mean(values))


def paired_bootstrap_ci(
    differences: Sequence[float],
    seed: int = 1701,
    draws: int = 20000) -> tuple[float, float]:
  differences = np.asarray(differences, dtype=np.float64)
  if differences.ndim != 1 or differences.size == 0:
    raise ValueError('differences must be a non-empty vector')
  rng = np.random.default_rng(seed)
  indices = rng.integers(
    0, differences.size, size=(draws, differences.size))
  means = differences[indices].mean(axis=1)
  lower, upper = np.quantile(means, [0.025, 0.975])
  return float(lower), float(upper)


def evaluate_frozen_gate(
    records: Sequence[BenchmarkRecord],
    seeds: Sequence[int] = (1, 2, 3),
    task: str = 'context_switching_matching') -> dict[str, object]:
  """Evaluate the paper's synthetic topology/validity kill criterion."""
  static_models = (
    'natural_chain', 'static_forest',
    'fixed_topology_dynamic_factors')
  topology_differences = []
  invalid_reductions = []
  contextual_invalid = []
  contextual_edge_f1 = []
  for seed in seeds:
    contextual_tv = _mean_for(
      records, task, seed, 'contextual_forest', 'tv')
    best_static_tv = min(
      _mean_for(records, task, seed, model, 'tv')
      for model in static_models)
    topology_differences.append(best_static_tv - contextual_tv)

    independent_invalid = _mean_for(
      records, task, seed, 'factorized', 'invalid_rate')
    joint_invalid = _mean_for(
      records, task, seed, 'contextual_forest', 'invalid_rate')
    contextual_invalid.append(joint_invalid)
    invalid_reductions.append(
      (independent_invalid - joint_invalid)
      / max(independent_invalid, 1e-12))
    contextual_edge_f1.append(_mean_for(
      records, task, seed, 'contextual_forest', 'edge_f1'))

  topology_ci = paired_bootstrap_ci(topology_differences)
  best_static_mean = float(np.mean([
    _mean_for(records, task, seed, model, 'tv')
    for seed in seeds for model in [min(
      static_models,
      key=lambda candidate: _mean_for(
        records, task, seed, candidate, 'tv'))]
  ]))
  contextual_mean = float(np.mean([
    _mean_for(records, task, seed, 'contextual_forest', 'tv')
    for seed in seeds]))
  relative_topology_reduction = (
    (best_static_mean - contextual_mean) / max(best_static_mean, 1e-12))
  checks = {
    'joint_invalid_absolute': float(np.mean(contextual_invalid)) <= 0.05,
    'joint_invalid_relative_reduction': (
      float(np.mean(invalid_reductions)) >= 0.80),
    'contextual_tv_relative_reduction': relative_topology_reduction >= 0.25,
    'contextual_edge_f1': float(np.mean(contextual_edge_f1)) >= 0.80,
    'all_seeds_same_direction': all(value > 0 for value in topology_differences),
    'paired_95pct_ci_excludes_zero': topology_ci[0] > 0,
  }
  return {
    'gate_name': 'g1_table_fit_structural_sanity',
    'scientific_scope': (
      'finite-data additively smoothed empirical forest projection; '
      'not neural-head '
      'training evidence'),
    'passed': bool(all(checks.values())),
    'checks': checks,
    'metrics': {
      'contextual_invalid_rate': float(np.mean(contextual_invalid)),
      'invalid_relative_reduction': float(np.mean(invalid_reductions)),
      'best_static_tv': best_static_mean,
      'contextual_tv': contextual_mean,
      'contextual_tv_relative_reduction': relative_topology_reduction,
      'contextual_edge_f1': float(np.mean(contextual_edge_f1)),
      'paired_tv_improvement_95pct_ci': list(topology_ci),
      'paired_tv_improvements_by_seed': topology_differences,
    },
  }
