import numpy as np

from synthetic.distributions import ContextSwitchingMatching
from synthetic.g1_benchmark import (
  estimate_moments,
  evaluate_frozen_gate,
  evaluate_task,
  forest_projection,
  maximum_weight_forest,
)


def test_forest_projection_normalizes_and_retains_shape():
  task = ContextSwitchingMatching(vocab_size=2)
  samples = task.sample(1000, context=0, rng=np.random.default_rng(4))
  moments = estimate_moments(samples, task.vocab_size)
  probability = forest_projection(
    task.support(), moments, task.true_edges(0))
  assert probability.shape == task.probabilities(0).shape
  assert np.isclose(probability.sum(), 1.0)
  assert np.all(probability > 0)


def test_maximum_weight_forest_is_acyclic_and_deterministic():
  weights = {
    (0, 1): 4.0, (1, 2): 3.0, (0, 2): 2.0,
    (2, 3): 1.0, (1, 3): 0.5, (0, 3): 0.0,
  }
  assert maximum_weight_forest(4, weights, 3) == (
    (0, 1), (1, 2), (2, 3))


def test_contextual_topology_beats_every_fixed_topology_ablation():
  records = []
  task = ContextSwitchingMatching(vocab_size=2)
  for seed in (1, 2, 3):
    records.extend(evaluate_task(
      'context_switching_matching', task, seed,
      train_samples_per_context=2048,
      eval_samples_per_model=2000))
  gate = evaluate_frozen_gate(records)
  assert gate['passed'], gate
  assert gate['metrics']['contextual_tv'] < gate['metrics']['best_static_tv']
