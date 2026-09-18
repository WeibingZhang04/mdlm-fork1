"""Endpoint-factor configuration, learning, exact inference and sampling."""

import itertools
import math
from pathlib import Path
import unittest

import torch

from models.directional_forest import DirectionalCouplingForestHead
from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import (
  full_vocabulary_marginals,
  infer_structured_distribution,
  sample_structured_tokens,
  structured_token_log_probability,
)
from structured_training import structured_denoising_loss


def make_head(**overrides):
  options = dict(
    hidden_size=5, vocab_size=4, top_k=2, rank=3, time_embed_dim=8,
    topology_dim=8, local_window=1, num_anchor_slots=1,
    contextual_neighbors=0, component_size_cap=3, topology_mode='fixed')
  options.update(overrides)
  return ContextualCouplingForestHead(**options)


class FactorEmbeddingModesTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(1)

  def setUp(self):
    torch.manual_seed(37)
    self.hidden = torch.randn(1, 3, 5)
    self.logits = torch.randn(1, 3, 4)
    self.time = torch.tensor([0.5])
    self.active = torch.ones(1, 3, dtype=torch.bool)

  def test_default_shared_preserves_state_and_seeded_outputs(self):
    torch.manual_seed(7)
    default = make_head()
    torch.manual_seed(7)
    shared = make_head(factor_embedding_mode='shared')
    self.assertFalse(any(k.startswith('right_') for k in default.state_dict()))
    shared.load_state_dict(default.state_dict(), strict=True)
    for key, value in default.state_dict().items():
      torch.testing.assert_close(value, shared.state_dict()[key], rtol=0, atol=0)
    args = (self.hidden, self.logits, self.time, self.active)
    a, b = default(*args), shared(*args)
    torch.testing.assert_close(a.pair_left_factors, b.pair_left_factors,
                               rtol=0, atol=0)
    torch.testing.assert_close(a.pair_right_factors, b.pair_right_factors,
                               rtol=0, atol=0)

  def test_rejects_unknown_mode(self):
    with self.assertRaisesRegex(ValueError, 'factor_embedding_mode'):
      make_head(factor_embedding_mode='typo')

  def test_hydra_option_works_in_old_and_new_model_configs(self):
    import hydra

    config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
    for model_name in ('contextual-forest-small',
                       'contextual-forest-endpoints-small'):
      with self.subTest(model=model_name):
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
          config = hydra.compose(config_name='config', overrides=[
            f'model={model_name}',
            '++model.structured_decoder.factor_embedding_mode=separate',
            'model.structured_decoder.rank=8',
          ])
        self.assertEqual(config.model.structured_decoder.factor_embedding_mode,
                         'separate')
        self.assertEqual(config.model.structured_decoder.rank, 8)
        self.assertEqual(config.model.hidden_size, 768)

  def test_half_rank_separate_matches_parameter_count(self):
    shared = make_head(rank=16)
    separate = make_head(rank=8, factor_embedding_mode='separate')
    self.assertEqual(shared.parameter_count, separate.parameter_count)
    self.assertEqual(len(shared.state_dict()) + 4, len(separate.state_dict()))

  def test_legacy_directional_name_keeps_state_keys_and_outputs(self):
    separate = make_head(factor_embedding_mode='separate')
    legacy = DirectionalCouplingForestHead(
      hidden_size=5, vocab_size=4, top_k=2, rank=3, time_embed_dim=8,
      topology_dim=8, local_window=1, num_anchor_slots=1,
      contextual_neighbors=0, component_size_cap=3, topology_mode='fixed')
    legacy.load_state_dict(separate.state_dict(), strict=True)
    args = (self.hidden, self.logits, self.time, self.active)
    torch.testing.assert_close(separate(*args).materialize_pair_factors(),
                               legacy(*args).materialize_pair_factors())

  def test_endpoints_follow_position_not_supplied_edge_orientation(self):
    head = make_head(factor_embedding_mode='separate', factor_mode='fixed')
    with torch.no_grad():
      head.token_factor_embedding.weight.copy_(torch.arange(12).reshape(4, 3))
      head.right_token_factor_embedding.weight.copy_(
        torch.arange(12).reshape(4, 3).flip(0) * 0.3)
    output = head(self.hidden, self.logits, self.time, self.active,
                  fixed_edge_index=torch.tensor([[2, 0]]))
    edge = output.edge_index[0, output.edge_mask[0]][0]
    self.assertEqual(edge.tolist(), [0, 2])
    for side, node, table in (
        ('left', 0, head.token_factor_embedding),
        ('right', 2, head.right_token_factor_embedding)):
      expected = (torch.nn.functional.softplus(
        table(output.candidate_ids[:, node])) + 1e-6) / math.sqrt(head.rank)
      actual = getattr(output, f'pair_{side}_factors')[output.edge_mask]
      torch.testing.assert_close(actual, expected)

  def test_both_tables_and_dynamic_projections_learn_in_both_topologies(self):
    for topology, factor in itertools.product(('fixed', 'dynamic'), repeat=2):
      with self.subTest(topology=topology, factor=factor):
        head = make_head(factor_embedding_mode='separate',
                         topology_mode=topology, factor_mode=factor)
        before = {name: p.detach().clone() for name, p in head.named_parameters()}
        output = head(self.hidden, self.logits, self.time, self.active)
        self.assertTrue(output.edge_mask.any())
        # Explicit candidates ensure the target exercises both factor tables.
        target = output.candidate_ids[..., 0]
        loss = structured_denoising_loss(
          output, self.logits, target, self.active).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        torch.optim.SGD(head.parameters(), lr=0.1).step()
        for prefix in ('', 'right_'):
          for suffix in ('token_factor_embedding.weight',
                         'factor_hidden_projection.weight',
                         'factor_time_projection.weight'):
            name = prefix + suffix
            parameter = dict(head.named_parameters())[name]
            if factor == 'fixed' and 'projection' in suffix:
              self.assertIsNone(parameter.grad)
            else:
              self.assertIsNotNone(parameter.grad)
              self.assertTrue(torch.isfinite(parameter.grad).all())
              self.assertGreater(float(parameter.grad.abs().sum()), 0)
              self.assertFalse(torch.equal(before[name], parameter.detach()))

  def test_separate_normalizes_with_residual_and_backends_agree(self):
    head = make_head(factor_embedding_mode='separate')
    output = head(self.hidden, self.logits, self.time, self.active)
    results = []
    for backend in ('dense', 'low_rank'):
      inference = infer_structured_distribution(output, self.active, backend=backend)
      probabilities = torch.cat([
        structured_token_log_probability(
          output, self.logits, torch.tensor([tokens]), self.active, inference).exp()
        for tokens in itertools.product(range(4), repeat=3)])
      torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0),
                                 atol=2e-6, rtol=2e-6)
      results.append(probabilities)
    torch.testing.assert_close(*results, atol=2e-6, rtol=2e-6)

  def test_joint_samples_match_exact_separate_marginals(self):
    head = make_head(factor_embedding_mode='separate')
    with torch.no_grad():
      head.token_factor_embedding.weight.normal_(0, 2)
      head.right_token_factor_embedding.weight.normal_(0, 2)
      output = head(self.hidden, self.logits, self.time, self.active)
      inference = infer_structured_distribution(output, self.active)
      expected = full_vocabulary_marginals(
        output, self.logits, self.active, inference)[0]
      samples = sample_structured_tokens(
        output, self.logits, self.active, num_samples=15000,
        generator=torch.Generator().manual_seed(71), inference=inference)[0]
      observed = torch.nn.functional.one_hot(samples, num_classes=4).float().mean(0)
      torch.testing.assert_close(observed, expected, atol=0.02, rtol=0)

  def test_independent_and_empty_forests_remain_neutral(self):
    head = make_head(factor_embedding_mode='separate')
    for active, independent in ((self.active, True),
                                (torch.zeros_like(self.active), False)):
      output = head(self.hidden, self.logits, self.time, active,
                    independent_mode=independent)
      factors = output.materialize_pair_factors()
      torch.testing.assert_close(factors, torch.ones_like(factors))


if __name__ == '__main__':
  unittest.main()
