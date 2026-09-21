import math
import unittest

import torch

from synthetic.tiny_joint_debug import (
  TinyModel, TinyVariant, exact_log_probabilities, oracle_result,
  train_tiny,
)
from models.directional_forest import DirectionalCouplingForestHead
from structured_objective import infer_structured_distribution, structured_token_log_probability


class TinyJointDebugTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(1)

  def test_oracle_uses_joint_sampler_and_preserves_both_modes(self):
    result = oracle_result('opposite', samples=12000, seed=19)
    self.assertLess(result['invalid_mass'], 1e-10)
    self.assertEqual(result['joint_sample_invalid_rate'], 0)
    self.assertAlmostEqual(result['marginal_sample_invalid_rate'], 0.5, delta=0.02)
    self.assertAlmostEqual(result['joint_sample_frequencies']['AB'], 0.5, delta=0.02)
    self.assertLess(result['max_dense_low_rank_probability_error'], 1e-12)

  def test_shared_head_bound_holds_far_from_initialization(self):
    for seed in range(10):
      model = TinyModel(TinyVariant('stress', init_std=3.0), seed)
      with torch.no_grad():
        model.head.factor_hidden_projection.weight.normal_(0, 4)
        model.head.factor_hidden_projection.bias.normal_(0, 4)
        probability = exact_log_probabilities(*model()).exp()
      self.assertGreaterEqual(float(probability[0] + probability[3]), 0.5 - 2e-6)

  def test_unrestricted_table_learns_opposite_and_independent(self):
    for task in ('opposite', 'independent'):
      result = train_tiny(TinyVariant('table', kind='table'), task, seed=1,
                          max_steps=600, samples=0)
      self.assertLess(result['kl_target_to_model'], 0.01)
      self.assertLess(result['invalid_mass'], 0.01)
      self.assertEqual(result['gradient_active_parameter_count'], 4)

  def test_directional_head_learns_opposite(self):
    result = train_tiny(TinyVariant('directional', kind='directional',
                                   init_std=0.25), 'opposite', seed=1,
                        max_steps=600, samples=5000)
    self.assertLess(result['kl_target_to_model'], 0.01)
    self.assertLess(result['invalid_mass'], 0.01)
    for mode in ('AB', 'BA'):
      self.assertAlmostEqual(result['probabilities'][mode], 0.5, delta=0.02)
    self.assertLess(result['max_dense_low_rank_probability_error'], 2e-6)
    self.assertAlmostEqual(result['marginal_sample_invalid_rate'], 0.5, delta=0.03)

  def test_directional_wide_control_fits_equal_and_independent(self):
    for task in ('equal', 'independent'):
      result = train_tiny(TinyVariant('directional', kind='directional',
                                     init_std=0.25), task, seed=1,
                          max_steps=600, samples=0)
      self.assertLess(result['kl_target_to_model'], 0.01)
      self.assertLess(result['invalid_mass'], 0.01)

  def test_directional_tail_normalization_and_backend_agreement(self):
    torch.manual_seed(37)
    head = DirectionalCouplingForestHead(
      hidden_size=5, vocab_size=4, top_k=2, rank=3, time_embed_dim=8,
      topology_dim=8, local_window=1, num_anchor_slots=1,
      contextual_neighbors=0, component_size_cap=2, topology_mode='fixed')
    hidden, logits = torch.randn(1, 2, 5), torch.randn(1, 2, 4)
    active = torch.ones(1, 2, dtype=torch.bool)
    output = head(hidden, logits, torch.tensor([0.5]), active)
    results = []
    for backend in ('dense', 'low_rank'):
      inference = infer_structured_distribution(output, active, backend=backend)
      probabilities = torch.stack([
        structured_token_log_probability(output, logits, torch.tensor([[a, b]]),
                                          active, inference).exp()[0]
        for a in range(4) for b in range(4)])
      self.assertAlmostEqual(float(probabilities.sum().detach()), 1.0, places=5)
      results.append(probabilities)
    torch.testing.assert_close(results[0], results[1], atol=2e-6, rtol=2e-6)
    neutral = head(hidden, logits, torch.tensor([0.5]), active,
                   independent_mode=True).materialize_pair_factors()
    torch.testing.assert_close(neutral, torch.ones_like(neutral))

  def test_shared_learning_cannot_exceed_proven_bound(self):
    result = train_tiny(TinyVariant('shared', init_std=0.25), 'opposite',
                        seed=1, max_steps=100, samples=0)
    self.assertGreaterEqual(result['kl_target_to_model'], math.log(2) - 2e-6)
    self.assertGreaterEqual(result['invalid_mass'], 0.5 - 2e-6)

  def test_half_rank_directional_matches_active_capacity(self):
    shared = train_tiny(TinyVariant('shared'), 'opposite', seed=3,
                        max_steps=1, samples=0)
    directional = train_tiny(TinyVariant('matched', kind='directional',
      rank=2, init_std=0.25, warmup_steps=100), 'equal', seed=3,
      max_steps=600, samples=0)
    self.assertEqual(shared['gradient_active_parameter_count'],
                     directional['gradient_active_parameter_count'])
    self.assertEqual(shared['parameter_count'], directional['parameter_count'])
    self.assertEqual(directional['pair_factor_rank'], 2)
    self.assertLess(directional['invalid_mass'], 0.01)


if __name__ == '__main__':
  unittest.main()
