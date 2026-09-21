"""Exact synthetic distributions for the coupling-forest kill gate.

Every task has a small enumerable support. This keeps the primary Gate 1
metrics (conditional NLL, KL, TV, edge recovery, and invalid-sample rate)
independent of learned evaluators or language-model proxies.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Iterable, Optional, Sequence

import numpy as np


Edge = tuple[int, int]


def canonical_edges(edges: Iterable[Edge]) -> tuple[Edge, ...]:
  """Return sorted undirected edges with duplicate/self edges removed."""
  normalized = {
    (min(int(i), int(j)), max(int(i), int(j)))
    for i, j in edges
    if int(i) != int(j)
  }
  return tuple(sorted(normalized))


def enumerate_sequences(length: int, vocab_size: int) -> np.ndarray:
  """Enumerate all sequences in lexicographic order."""
  if length <= 0:
    raise ValueError('length must be positive')
  if vocab_size <= 1:
    raise ValueError('vocab_size must be at least two')
  return np.asarray(
    list(itertools.product(range(vocab_size), repeat=length)),
    dtype=np.int64)


def _normalize(weights: np.ndarray) -> np.ndarray:
  weights = np.asarray(weights, dtype=np.float64)
  if weights.ndim != 1:
    raise ValueError('weights must be one-dimensional')
  if np.any(weights < 0) or not np.all(np.isfinite(weights)):
    raise ValueError('weights must be finite and nonnegative')
  total = float(weights.sum())
  if total <= 0:
    raise ValueError('weights must have positive mass')
  return weights / total


@dataclasses.dataclass(frozen=True)
class ExactSyntheticDistribution:
  """Base class for a context-indexed enumerable distribution."""

  length: int
  vocab_size: int

  def support(self) -> np.ndarray:
    return enumerate_sequences(self.length, self.vocab_size)

  def probabilities(self, context: int = 0) -> np.ndarray:
    raise NotImplementedError

  def true_edges(self, context: int = 0) -> tuple[Edge, ...]:
    del context
    return ()

  def is_valid(self, samples: np.ndarray, context: int = 0) -> np.ndarray:
    """Whether samples have positive probability under the task."""
    samples = self._validate_samples(samples)
    support = self.support()
    positive = self.probabilities(context) > 0
    valid_set = {tuple(row.tolist()) for row in support[positive]}
    return np.asarray(
      [tuple(row.tolist()) in valid_set for row in samples], dtype=bool)

  def sample(
      self,
      count: int,
      context: int = 0,
      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if count <= 0:
      raise ValueError('count must be positive')
    rng = np.random.default_rng() if rng is None else rng
    support = self.support()
    indices = rng.choice(
      support.shape[0], size=count, p=self.probabilities(context))
    return support[indices]

  def marginal(self, position: int, context: int = 0) -> np.ndarray:
    if position < 0 or position >= self.length:
      raise IndexError(position)
    support = self.support()
    probs = self.probabilities(context)
    result = np.zeros(self.vocab_size, dtype=np.float64)
    np.add.at(result, support[:, position], probs)
    return result

  def pair_marginal(
      self, first: int, second: int, context: int = 0) -> np.ndarray:
    if first == second:
      raise ValueError('positions must be distinct')
    support = self.support()
    probs = self.probabilities(context)
    result = np.zeros(
      (self.vocab_size, self.vocab_size), dtype=np.float64)
    np.add.at(result, (support[:, first], support[:, second]), probs)
    return result

  def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.int64)
    if samples.ndim != 2 or samples.shape[1] != self.length:
      raise ValueError(
        f'samples must have shape (count, {self.length})')
    if np.any(samples < 0) or np.any(samples >= self.vocab_size):
      raise ValueError('sample tokens are outside the vocabulary')
    return samples


@dataclasses.dataclass(frozen=True)
class AmbiguousPairs(ExactSyntheticDistribution):
  """Binary pair with modes 00 and 11 but uniform single-site marginals."""

  length: int = 2
  vocab_size: int = 2

  def probabilities(self, context: int = 0) -> np.ndarray:
    del context
    support = self.support()
    return _normalize((support[:, 0] == support[:, 1]).astype(np.float64))

  def true_edges(self, context: int = 0) -> tuple[Edge, ...]:
    del context
    return ((0, 1),)

  def is_valid(self, samples: np.ndarray, context: int = 0) -> np.ndarray:
    del context
    samples = self._validate_samples(samples)
    return samples[:, 0] == samples[:, 1]


@dataclasses.dataclass(frozen=True)
class MarkovLanguage(ExactSyntheticDistribution):
  """Stationary categorical Markov chain with adjustable copy strength."""

  length: int = 5
  vocab_size: int = 3
  coupling: float = 0.8

  def __post_init__(self) -> None:
    if not 0.0 <= self.coupling <= 1.0:
      raise ValueError('coupling must be in [0, 1]')

  def probabilities(self, context: int = 0) -> np.ndarray:
    del context
    support = self.support()
    base = (1.0 - self.coupling) / self.vocab_size
    same = self.coupling + base
    transition = np.where(
      support[:, 1:] == support[:, :-1], same, base)
    weights = np.full(support.shape[0], 1.0 / self.vocab_size)
    weights *= transition.prod(axis=1)
    return _normalize(weights)

  def true_edges(self, context: int = 0) -> tuple[Edge, ...]:
    del context
    return tuple((index, index + 1) for index in range(self.length - 1))


@dataclasses.dataclass(frozen=True)
class ContextSwitchingMatching(ExactSyntheticDistribution):
  """A context selects a nonlocal matching whose token pairs must agree.

  All single-site marginals stay uniform.  The default matchings have a cyclic
  union: no single forest can contain every ground-truth edge, while each
  context-specific matching is itself a forest.  This makes adaptive topology
  identifiable without imposing an artificially tiny one-edge budget.
  """

  length: int = 6
  vocab_size: int = 2
  matchings: tuple[tuple[Edge, ...], ...] = (
    ((0, 1), (2, 3), (4, 5)),
    ((0, 2), (1, 4), (3, 5)),
    ((0, 3), (1, 5), (2, 4)),
  )

  def __post_init__(self) -> None:
    normalized = []
    for matching in self.matchings:
      edges = canonical_edges(matching)
      if not edges:
        raise ValueError('every context matching must be non-empty')
      if any(i < 0 or j >= self.length for i, j in edges):
        raise ValueError('matching edge is outside the sequence')
      endpoints = [node for edge in edges for node in edge]
      if len(endpoints) != len(set(endpoints)):
        raise ValueError('edges within a context must form a matching')
      normalized.append(edges)
    if not normalized:
      raise ValueError('matchings must be non-empty')
    object.__setattr__(self, 'matchings', tuple(normalized))

  @property
  def num_contexts(self) -> int:
    return len(self.matchings)

  def selected_matching(self, context: int) -> tuple[Edge, ...]:
    if context < 0 or context >= self.num_contexts:
      raise ValueError(
        f'context must be in [0, {self.num_contexts - 1}]')
    return self.matchings[context]

  def probabilities(self, context: int = 0) -> np.ndarray:
    support = self.support()
    valid = np.ones(support.shape[0], dtype=bool)
    for first, second in self.selected_matching(context):
      valid &= support[:, first] == support[:, second]
    return _normalize(valid.astype(np.float64))

  def true_edges(self, context: int = 0) -> tuple[Edge, ...]:
    return self.selected_matching(context)

  def is_valid(self, samples: np.ndarray, context: int = 0) -> np.ndarray:
    samples = self._validate_samples(samples)
    valid = np.ones(samples.shape[0], dtype=bool)
    for first, second in self.selected_matching(context):
      valid &= samples[:, first] == samples[:, second]
    return valid


@dataclasses.dataclass(frozen=True)
class CyclicXOR(ExactSyntheticDistribution):
  """Even-parity binary triple, a deliberate pairwise-tree failure case."""

  length: int = 3
  vocab_size: int = 2

  def probabilities(self, context: int = 0) -> np.ndarray:
    del context
    support = self.support()
    return _normalize((support.sum(axis=1) % 2 == 0).astype(np.float64))

  def is_valid(self, samples: np.ndarray, context: int = 0) -> np.ndarray:
    del context
    samples = self._validate_samples(samples)
    return samples.sum(axis=1) % 2 == 0


def product_of_marginals(
    distribution: ExactSyntheticDistribution,
    context: int = 0) -> np.ndarray:
  """Factorized projection used by the equal-marginal control."""
  support = distribution.support()
  probs = np.ones(support.shape[0], dtype=np.float64)
  for position in range(distribution.length):
    probs *= distribution.marginal(position, context)[support[:, position]]
  return _normalize(probs)


def contexts_for(
    distribution: ExactSyntheticDistribution) -> Sequence[int]:
  if isinstance(distribution, ContextSwitchingMatching):
    return tuple(range(distribution.num_contexts))
  return (0,)
