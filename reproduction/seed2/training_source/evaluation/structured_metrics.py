"""Exact, dependency-focused metrics for contextual-forest experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from synthetic.distributions import (
  Edge,
  ExactSyntheticDistribution,
  canonical_edges,
)


def _probability_vector(values: np.ndarray, name: str) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 1:
    raise ValueError(f'{name} must be one-dimensional')
  if np.any(values < 0) or not np.all(np.isfinite(values)):
    raise ValueError(f'{name} must be finite and nonnegative')
  total = float(values.sum())
  if total <= 0:
    raise ValueError(f'{name} has no mass')
  return values / total


def total_variation(target: np.ndarray, model: np.ndarray) -> float:
  target = _probability_vector(target, 'target')
  model = _probability_vector(model, 'model')
  if target.shape != model.shape:
    raise ValueError('target and model shapes differ')
  return float(0.5 * np.abs(target - model).sum())


def kl_divergence(
    target: np.ndarray,
    model: np.ndarray,
    epsilon: float = 1e-12) -> float:
  target = _probability_vector(target, 'target')
  model = _probability_vector(model, 'model')
  if target.shape != model.shape:
    raise ValueError('target and model shapes differ')
  if epsilon <= 0:
    raise ValueError('epsilon must be positive')
  positive = target > 0
  return float(np.sum(
    target[positive]
    * (np.log(target[positive]) - np.log(np.maximum(model[positive], epsilon)))))


def invalid_rate(
    samples: np.ndarray,
    distribution: ExactSyntheticDistribution,
    context: int = 0) -> float:
  return float(1.0 - distribution.is_valid(samples, context).mean())


def edge_scores(
    predicted: Iterable[Edge], true: Iterable[Edge]) -> dict[str, float]:
  predicted_set = set(canonical_edges(predicted))
  true_set = set(canonical_edges(true))
  true_positive = len(predicted_set & true_set)
  precision = (
    true_positive / len(predicted_set) if predicted_set else float(not true_set))
  recall = true_positive / len(true_set) if true_set else float(not predicted_set)
  f1 = (
    2.0 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0.0)
  return {
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'symmetric_difference': float(len(predicted_set ^ true_set)),
  }


def pairwise_mutual_information(pair_marginal: np.ndarray) -> float:
  pair = np.asarray(pair_marginal, dtype=np.float64)
  if pair.ndim != 2:
    raise ValueError('pair_marginal must be a matrix')
  pair = pair / pair.sum()
  first = pair.sum(axis=1, keepdims=True)
  second = pair.sum(axis=0, keepdims=True)
  independent = first @ second
  positive = pair > 0
  return float(np.sum(
    pair[positive] * (np.log(pair[positive]) - np.log(independent[positive]))))


def empirical_distribution(
    samples: np.ndarray,
    support: np.ndarray) -> np.ndarray:
  samples = np.asarray(samples, dtype=np.int64)
  support = np.asarray(support, dtype=np.int64)
  if samples.ndim != 2 or support.ndim != 2:
    raise ValueError('samples and support must be matrices')
  if samples.shape[1] != support.shape[1]:
    raise ValueError('sample and support lengths differ')
  lookup = {tuple(row.tolist()): index for index, row in enumerate(support)}
  counts = np.zeros(support.shape[0], dtype=np.float64)
  for row in samples:
    key = tuple(row.tolist())
    if key not in lookup:
      raise ValueError(f'sample {key} is outside support')
    counts[lookup[key]] += 1.0
  return counts / counts.sum()


def categorical_ece(
    confidence: np.ndarray,
    correct: np.ndarray,
    num_bins: int = 10) -> float:
  confidence = np.asarray(confidence, dtype=np.float64)
  correct = np.asarray(correct, dtype=np.float64)
  if confidence.shape != correct.shape or confidence.ndim != 1:
    raise ValueError('confidence and correct must be matching vectors')
  if np.any(confidence < 0) or np.any(confidence > 1):
    raise ValueError('confidence must lie in [0, 1]')
  if num_bins <= 0:
    raise ValueError('num_bins must be positive')
  boundaries = np.linspace(0.0, 1.0, num_bins + 1)
  result = 0.0
  for index in range(num_bins):
    lower, upper = boundaries[index], boundaries[index + 1]
    in_bin = (
      (confidence >= lower)
      & (confidence <= upper if index == num_bins - 1 else confidence < upper))
    if np.any(in_bin):
      result += float(in_bin.mean()) * abs(
        float(confidence[in_bin].mean()) - float(correct[in_bin].mean()))
  return result
