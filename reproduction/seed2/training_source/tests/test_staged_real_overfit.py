"""Small end-to-end checks for the real-checkpoint experiment runner."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from noise_schedule import LogLinearNoise

from scripts.run_staged_real_overfit import (
  FrozenCache, assert_disjoint, build_cache, check_serial_batch, evaluate,
  fixed_masks, make_head, read_examples, score_batch, train_head,
)


class TinyFrozenBackbone(torch.nn.Module):
  """Same runner interface, no production model downloads in unit tests."""

  mask_index = 10

  def __init__(self):
    super().__init__()
    self.embedding = torch.nn.Embedding(11, 8)
    self.output = torch.nn.Linear(8, 11)
    self.noise = LogLinearNoise()
    self.requires_grad_(False)

  def _structured_backbone_output(self, tokens, sigma, force_no_grad=True):
    hidden = self.embedding(tokens)
    hidden = hidden + hidden.mean(1, keepdim=True) + sigma[..., None]
    logits = self.output(hidden)
    logits[..., self.mask_index] = -torch.inf
    return hidden, logits


class StagedRealOverfitTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(477)
    torch.set_num_threads(1)
    self.tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7],
                                [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]])
    self.model = TinyFrozenBackbone().eval()
    self.active = fixed_masks(self.tokens, 0.5, 42)
    self.cache, self.cache_report = build_cache(
      self.model, self.tokens, self.active, 0.5, 'cpu', backbone_batch_size=2)

  def head(self, variant):
    return make_head(variant, 8, 11, seed=8, rank=4, unary_rank=4,
                     top_k=4, time_embed_dim=6, init_std=0.25)

  def test_fixed_masks_reproducible_and_exact_count(self):
    torch.testing.assert_close(self.active, fixed_masks(self.tokens, 0.5, 42))
    torch.testing.assert_close(self.active.sum(-1), torch.full((4,), 3))
    self.assertFalse(torch.equal(self.active, fixed_masks(self.tokens, 0.5, 43)))
    self.assertAlmostEqual(self.cache_report['diffusion_time'], 0.5 / 0.999)
    self.assertAlmostEqual(self.cache_report['head_and_backbone_sigma'], -torch.log(torch.tensor(0.5)).item())
    self.assertEqual(self.cache_report['mask_rate'], 0.5)

  def test_memory_disk_and_backbone_batching_preserve_inputs(self):
    with tempfile.TemporaryDirectory() as directory:
      cache, report = build_cache(
        self.model, self.tokens, self.active, 0.5, 'cpu', Path(directory) / 'cache',
        backbone_batch_size=1)
      self.assertEqual(len(cache), 4)
      expected, actual = self.cache.batch([0, 2]), cache.batch([0, 2])
      for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
      self.assertEqual(report['active_tokens'], 12)
      self.assertAlmostEqual(report['backbone_nll_per_masked_token'],
                             self.cache_report['backbone_nll_per_masked_token'], places=6)
      self.assertFalse(actual['logits'].requires_grad)
      self.assertFalse(actual['hidden'].requires_grad)
      self.assertTrue(bool(torch.isneginf(actual['logits'][..., 10]).all()))
      with self.assertRaises(FileExistsError):
        build_cache(self.model, self.tokens, self.active, 0.5, 'cpu', Path(directory) / 'cache')

  def test_shared_directional_capacity_and_graph_match(self):
    shared, directional, unary = (self.head(name) for name in ('shared', 'directional', 'unary'))
    active_count = lambda head: sum(p.numel() for p in head.parameters() if p.requires_grad)
    self.assertEqual(shared.rank, 4)
    self.assertEqual(directional.rank, 2)
    self.assertEqual(active_count(shared), active_count(directional))
    self.assertEqual(active_count(shared) - active_count(unary), 4 * (8 + 6 + 1))
    batch = self.cache.batch([0, 1])
    args = (batch['hidden'], batch['logits'], batch['sigma'], batch['active'])
    left, right = shared(*args), directional(*args)
    torch.testing.assert_close(left.candidate_ids, right.candidate_ids)
    torch.testing.assert_close(left.edge_index, right.edge_index)
    torch.testing.assert_close(left.edge_mask, right.edge_mask)
    self.assertFalse(torch.equal(directional.token_factor_embedding.weight,
                                 directional.right_token_factor_embedding.weight))
    for head in (shared, directional):
      self.assertFalse(any(parameter.requires_grad for parameter in head.edge_proposer.parameters()))

  def test_scores_decompose_and_serial_batch_agree(self):
    for variant in ('shared', 'directional', 'unary'):
      with self.subTest(variant=variant):
        head = self.head(variant)
        score = score_batch(head, self.cache.batch([0, 1]))
        torch.testing.assert_close(
          score['joint_log_probability'],
          score['marginal_log_probability'] + score['dependence_log_probability'],
          atol=1e-5, rtol=1e-5)
        self.assertTrue(check_serial_batch(head, self.cache, 'cpu')['passed'])
        metrics = evaluate(head, self.cache, 2, 'cpu')
        self.assertAlmostEqual(metrics['joint_gain_nats_per_masked_token'],
                               metrics['marginal_gain_nats_per_masked_token']
                               + metrics['dependence_gain_nats_per_masked_token'], places=5)
        if variant == 'unary':
          self.assertEqual(metrics['dependence_gain_nats_per_masked_token'], 0)

  def test_all_arms_fit_frozen_inputs_without_backbone_updates(self):
    before = {key: value.clone() for key, value in self.model.state_dict().items()}
    train = FrozenCache(records=[self.cache.record(0), self.cache.record(1)])
    dev = FrozenCache(records=[self.cache.record(2), self.cache.record(3)])
    for variant in ('shared', 'directional', 'unary'):
      with self.subTest(variant=variant):
        calls = []
        result = train_head(self.head(variant), train, dev, steps=20,
                            batch_size=2, learning_rate=0.05, eval_every=10,
                            seed=2, device='cpu', callback=calls.append)
        self.assertTrue(result['train_loss_fell'])
        self.assertGreater(result['train_nll_reduction'], 0.01)
        self.assertEqual([row['step'] for row in result['curve']], [0, 10, 20])
        self.assertEqual(len(calls), 3)
        self.assertIn('gradient_norm_before_clip', calls[-1])
        self.assertIn('dev', calls[-1])
    for key, value in self.model.state_dict().items():
      torch.testing.assert_close(value, before[key], atol=0, rtol=0)
      self.assertIsNone(value.grad)

  def test_input_validation_deduplication_and_disjointness(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'examples.jsonl'
      rows = [{'input_ids': [1, 2]}, {'input_ids': self.tokens[0].tolist()},
              {'input_ids': self.tokens[0].tolist()}, {'input_ids': self.tokens[1].tolist()}]
      path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
      tokens, report = read_examples(path, 2, 6, 11, 10)
      torch.testing.assert_close(tokens, self.tokens[:2])
      self.assertEqual(report['short_records_skipped'], 1)
      self.assertEqual(report['duplicate_records_skipped'], 1)
      with self.assertRaisesRegex(ValueError, 'requested 3'):
        read_examples(path, 3, 6, 11, 10)
      assert_disjoint(self.tokens[:2], self.tokens[2:])
      with self.assertRaisesRegex(ValueError, 'identical'):
        assert_disjoint(self.tokens[:2], self.tokens[1:])


if __name__ == '__main__':
  unittest.main()
