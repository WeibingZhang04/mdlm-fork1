import numpy as np

from evaluation.structured_metrics import (
  edge_scores,
  invalid_rate,
  pairwise_mutual_information,
  total_variation,
)
from synthetic.distributions import (
  AmbiguousPairs,
  ContextSwitchingMatching,
  CyclicXOR,
  MarkovLanguage,
  product_of_marginals,
)


def test_ambiguous_pair_exposes_factorized_mode_mixing():
  task = AmbiguousPairs()
  target = task.probabilities()
  factorized = product_of_marginals(task)
  assert np.allclose(target, [0.5, 0.0, 0.0, 0.5])
  assert np.allclose(factorized, 0.25)
  assert total_variation(target, factorized) == 0.5
  samples = np.repeat(task.support(), 25, axis=0)
  assert invalid_rate(samples, task) == 0.5


def test_markov_language_normalizes_and_strengthens_neighbor_information():
  weak = MarkovLanguage(coupling=0.1)
  strong = MarkovLanguage(coupling=0.9)
  assert np.isclose(weak.probabilities().sum(), 1.0)
  assert np.isclose(strong.probabilities().sum(), 1.0)
  assert pairwise_mutual_information(strong.pair_marginal(0, 1)) > (
    pairwise_mutual_information(weak.pair_marginal(0, 1)))


def test_context_switches_nonlocal_edge_without_changing_unaries():
  task = ContextSwitchingMatching()
  union = set()
  for context, expected in enumerate(task.matchings):
    assert task.true_edges(context) == expected
    union.update(expected)
    for position in range(task.length):
      assert np.allclose(task.marginal(position, context), [0.5, 0.5])
  assert len(union) > task.length - 1
  assert edge_scores(task.matchings[0], task.true_edges(0))['f1'] == 1.0
  assert edge_scores(task.matchings[0], task.true_edges(1))['f1'] < 1.0


def test_xor_has_uniform_pair_marginals_but_nontrivial_global_constraint():
  task = CyclicXOR()
  assert np.count_nonzero(task.probabilities()) == 4
  for first, second in [(0, 1), (0, 2), (1, 2)]:
    assert np.allclose(task.pair_marginal(first, second), 0.25)
    assert np.isclose(
      pairwise_mutual_information(task.pair_marginal(first, second)), 0.0)
  assert total_variation(
    task.probabilities(), product_of_marginals(task)) == 0.5
