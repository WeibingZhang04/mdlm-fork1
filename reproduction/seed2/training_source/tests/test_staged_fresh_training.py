"""Fresh shared corruptions, frozen inputs, document splits and exact resume."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.run_staged_fresh_training import (
  ARMS, FreshTrainer, assert_document_disjoint, backbone_runtime_identity, build_dev_caches,
  evaluate_development, read_documents, save_checkpoint,
)
from scripts.run_staged_real_overfit import file_sha256, make_head, training_loss
from structured_objective import structured_token_log_probability
from tests.test_staged_real_overfit import TinyFrozenBackbone


class StagedFreshTrainingTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(486)
    torch.set_num_threads(1)
    self.backbone = TinyFrozenBackbone().eval()
    self.tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7],
                                [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]])

  def trainer(self, *, batch_size=2, warmup=0, identity=None, device='cpu'):
    backbone = copy.deepcopy(self.backbone).to(device)
    call_counter = {'count': 0}

    def encode(*args, **kwargs):
      call_counter['count'] += 1
      return backbone._structured_backbone_output(*args, **kwargs)

    model = SimpleNamespace(backbone=backbone, noise=backbone.noise,
                            mask_index=backbone.mask_index, eval=backbone.eval,
                            _structured_backbone_output=encode)
    heads = {arm: make_head(arm, 8, 11, seed=3, rank=4, unary_rank=4,
                            top_k=4, time_embed_dim=6, init_std=0.25) for arm in ARMS}
    trainer = FreshTrainer(model, heads, self.tokens, seed=3, device=device,
                            batch_size=batch_size, backbone_batch_size=1,
                            mask_rates=(0.5, 0.75), learning_rate=0.01,
                            pair_warmup_steps=warmup, identity=identity)
    return trainer, call_counter

  def test_masks_change_and_every_arm_reuses_identical_detached_inputs(self):
    trainer, counter = self.trainer(batch_size=4)
    captured = []

    def observe(head, batch, warmup=False):
      captured.append({key: (value.data_ptr(), value.detach().clone())
                       for key, value in batch.items()})
      self.assertFalse(batch['hidden'].requires_grad)
      self.assertFalse(batch['logits'].requires_grad)
      return training_loss(head, batch, warmup)

    original = {name: tensor.clone() for name, tensor in trainer.model.backbone.state_dict().items()}
    with patch('scripts.run_staged_fresh_training.training_loss', side_effect=observe):
      first, second = trainer.train_step(), trainer.train_step()
    self.assertEqual(counter['count'], 8)  # two batches, four serial calls each
    self.assertEqual(first['backbone_calls'], 4)
    self.assertNotEqual(first['mask_sha256'], second['mask_sha256'])
    for start in (0, 3):
      for arm_index in (1, 2):
        for key in captured[start]:
          self.assertEqual(captured[start][key][0], captured[start + arm_index][key][0])
          torch.testing.assert_close(captured[start][key][1], captured[start + arm_index][key][1])
    for name, tensor in trainer.model.backbone.state_dict().items():
      torch.testing.assert_close(tensor, original[name], atol=0, rtol=0)
    self.assertTrue(all(parameter.grad is None for parameter in trainer.model.backbone.parameters()))

  def test_every_loss_is_normalized_by_the_shared_active_token_count(self):
    trainer, _ = self.trainer()
    checks = []

    def validate(head, batch, warmup=False):
      actual = training_loss(head, batch, warmup)
      if hasattr(head, 'nll_loss'):
        expected_sum = -head.target_log_probs(
          batch['hidden'], batch['logits'], batch['sigma'], batch['targets'], batch['active']).sum()
      else:
        output = head(batch['hidden'], batch['logits'], batch['sigma'], batch['active'])
        expected_sum = -structured_token_log_probability(
          output, batch['logits'], batch['targets'], batch['active']).sum()
      torch.testing.assert_close(actual, expected_sum / batch['active'].sum())
      checks.append(float(expected_sum.detach()))
      return actual

    with patch('scripts.run_staged_fresh_training.training_loss', side_effect=validate):
      record = trainer.train_step()
    for arm, nll_sum in zip(ARMS, checks):
      self.assertAlmostEqual(record['arms'][arm]['nll_sum'], nll_sum, places=5)

  def test_pair_warmup_does_not_disable_unary_training(self):
    trainer, _ = self.trainer(warmup=1)
    first, second = trainer.train_step(), trainer.train_step()
    self.assertEqual(first['arms']['shared']['factor_mode'], 'fixed')
    self.assertEqual(first['arms']['directional']['factor_mode'], 'fixed')
    self.assertEqual(first['arms']['unary']['factor_mode'], 'unary')
    self.assertEqual(second['arms']['shared']['factor_mode'], 'dynamic')
    self.assertGreater(first['arms']['unary']['gradient_norm_before_clip'], 0)

  def test_checkpoint_resume_preserves_updates_rng_and_cursor_exactly(self):
    uninterrupted, _ = self.trainer(warmup=2)
    for _ in range(6):
      uninterrupted.train_step()
    interrupted, _ = self.trainer(warmup=2)
    for _ in range(3):
      interrupted.train_step()
    with tempfile.TemporaryDirectory() as directory:
      checkpoint = Path(directory) / 'old-run' / 'step-000003.pt'
      digest = save_checkpoint(interrupted, checkpoint)
      self.assertEqual(digest, file_sha256(checkpoint))
      with self.assertRaises(FileExistsError):
        save_checkpoint(interrupted, checkpoint)
      resumed, _ = self.trainer(warmup=2)
      resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
      for _ in range(3):
        resumed.train_step()
      save_checkpoint(resumed, Path(directory) / 'new-run' / 'step-000006.pt')
    self.assertEqual(resumed.completed_steps, uninterrupted.completed_steps)
    self.assertEqual(resumed.cursor, uninterrupted.cursor)
    self.assertEqual(resumed.epoch, uninterrupted.epoch)
    torch.testing.assert_close(resumed.permutation, uninterrupted.permutation, atol=0, rtol=0)
    for arm in ARMS:
      for key, value in resumed.heads[arm].state_dict().items():
        torch.testing.assert_close(value, uninterrupted.heads[arm].state_dict()[key], atol=0, rtol=0)
    for actual, expected in zip(resumed.step_history, uninterrupted.step_history):
      for key in ('document_indices', 'mask_rate', 'mask_sha256', 'corrupted_token_sha256'):
        self.assertEqual(actual[key], expected[key])
      for arm in ARMS:
        self.assertEqual(actual['arms'][arm]['nll_per_masked_token'], expected['arms'][arm]['nll_per_masked_token'])
    wrong, _ = self.trainer(warmup=0)
    with self.assertRaisesRegex(ValueError, 'configuration differs'):
      wrong.load_state_dict(resumed.state_dict())

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA resume regression requires a GPU')
  def test_cuda_map_location_resume_matches_uninterrupted_updates_exactly(self):
    """Exercise the production CUDA load, including relocated Adam step tensors.

    No optimizer-state normalization or runner patch is applied in this test:
    if CUDA relocation changes updates, the strict comparison must expose it.
    """
    for backend in (torch.backends.cuda.matmul, torch.backends.cudnn):
      previous = backend.allow_tf32
      self.addCleanup(setattr, backend, 'allow_tf32', previous)
      backend.allow_tf32 = False
    uninterrupted, _ = self.trainer(warmup=2, device='cuda')
    for _ in range(6):
      uninterrupted.train_step()
    expected = copy.deepcopy(uninterrupted.state_dict())
    interrupted, _ = self.trainer(warmup=2, device='cuda')
    for _ in range(3):
      interrupted.train_step()
    # Default non-capturable AdamW creates CPU scalar step counters even when
    # its parameters and moment estimates are on CUDA.
    for optimizer in interrupted.optimizers.values():
      self.assertTrue(optimizer.state)
      for parameter_state in optimizer.state.values():
        self.assertEqual(parameter_state['step'].device.type, 'cpu')
    with tempfile.TemporaryDirectory() as directory:
      checkpoint = Path(directory) / 'step-000003.pt'
      save_checkpoint(interrupted, checkpoint)
      resumed, _ = self.trainer(warmup=2, device='cuda')
      loaded = torch.load(checkpoint, map_location='cuda', weights_only=True)
      for optimizer_state in loaded['optimizers'].values():
        for parameter_state in optimizer_state['state'].values():
          self.assertEqual(parameter_state['step'].device.type, 'cuda')
      resumed.load_state_dict(loaded)
      for _ in range(3):
        resumed.train_step()
    actual = resumed.state_dict()
    for key in ('completed_steps', 'cursor', 'epoch', 'config', 'identity'):
      self.assertEqual(actual[key], expected[key], key)
    for key in ('permutation', 'order_rng_state', 'mask_rng_state',
                'rate_rng_state', 'torch_rng_state'):
      torch.testing.assert_close(actual[key].cpu(), expected[key].cpu(), atol=0, rtol=0)
    self.assertEqual(len(actual['cuda_rng_states']), len(expected['cuda_rng_states']))
    self.assertTrue(actual['cuda_rng_states'])
    for actual_rng, expected_rng in zip(actual['cuda_rng_states'], expected['cuda_rng_states']):
      torch.testing.assert_close(actual_rng.cpu(), expected_rng.cpu(), atol=0, rtol=0)
    for arm in ARMS:
      with self.subTest(arm=arm):
        for name, value in actual['heads'][arm].items():
          torch.testing.assert_close(value, expected['heads'][arm][name], atol=0, rtol=0)
        optimizer_actual, optimizer_expected = actual['optimizers'][arm], expected['optimizers'][arm]
        self.assertEqual(optimizer_actual['param_groups'], optimizer_expected['param_groups'])
        self.assertEqual(optimizer_actual['state'].keys(), optimizer_expected['state'].keys())
        for parameter, values in optimizer_actual['state'].items():
          self.assertEqual(values.keys(), optimizer_expected['state'][parameter].keys())
          for name, value in values.items():
            reference = optimizer_expected['state'][parameter][name]
            if torch.is_tensor(value):
              torch.testing.assert_close(value.cpu(), reference.cpu(), atol=0, rtol=0)
            else:
              self.assertEqual(value, reference)
    self.assertEqual(len(actual['step_history']), 6)
    for actual_step, expected_step in zip(actual['step_history'], expected['step_history']):
      for key in ('document_indices', 'mask_rate', 'active_token_count', 'mask_sha256',
                  'clean_token_sha256', 'corrupted_token_sha256'):
        self.assertEqual(actual_step[key], expected_step[key], key)
      for arm in ARMS:
        for key in ('nll_per_masked_token', 'nll_sum', 'gradient_norm_before_clip', 'factor_mode'):
          self.assertEqual(actual_step['arms'][arm][key], expected_step['arms'][arm][key],
                           f'step {actual_step["step"]}: {arm}/{key}')

  def test_fixed_dev_cache_records_pair_and_include_step_zero(self):
    trainer, _ = self.trainer()
    documents = ['dev-a', 'dev-b']
    with tempfile.TemporaryDirectory() as directory:
      groups, reports = build_dev_caches(
        trainer.model, self.tokens[:2], documents, rates=(0.5, 0.75), seed=3,
        device='cpu', output_dir=Path(directory) / 'cache', backbone_batch_size=1)
      first = evaluate_development(trainer, groups, checkpoint_sha256='first',
                                     dataset='toy', batch_size=2)
      trainer.train_step()
      evaluate_development(trainer, groups, checkpoint_sha256='second',
                            dataset='toy', batch_size=2)
    self.assertEqual(first['step'], 0)
    self.assertEqual(len(trainer.dev_records), 2 * 2 * 2 * 3)
    reference = {}
    for row in trainer.dev_records:
      key = (row['document_id'], row['mask_rate'], row['corruption_seed'])
      signature = tuple(row[field] for field in (
        'mask_sha256', 'clean_token_sha256', 'corrupted_token_sha256',
        'active_token_count', 'backbone_log_probability'))
      self.assertEqual(reference.setdefault(key, signature), signature)
      self.assertEqual(row['split'], 'dev')
      self.assertEqual(row['training_seed'], 3)
      self.assertEqual(row['dataset'], 'toy')
    self.assertEqual(len(reports), 2)

  def test_production_runtime_identity_roundtrips_with_weights_only_loading(self):
    trainer, _ = self.trainer(identity={
      'runtime': {'torch_version': torch.__version__, 'device_type': 'cpu'},
      'head_config': {'rank': 4}, 'source_sha256': {'example.py': 'a' * 64}})
    self.assertIs(type(trainer.identity['runtime']['torch_version']), str)
    with tempfile.TemporaryDirectory() as directory:
      checkpoint = Path(directory) / 'runtime.pt'
      save_checkpoint(trainer, checkpoint)
      state = torch.load(checkpoint, weights_only=True)
      self.assertEqual(state['identity'], trainer.identity)

  def test_backbone_identity_pins_noise_and_resolves_model_settings(self):
    from omegaconf import OmegaConf
    model = SimpleNamespace(config=OmegaConf.create({
      'length': 128, 'backbone': 'dit', 'noise': {'type': 'loglinear', 'eps': 0.001},
      'model': {'length': '${length}', 'hidden_size': 8, 'structured_decoder': {'rank': 4}}}),
      parameterization='subs', time_conditioning=True, subs_masking=False, T=0)
    identity = backbone_runtime_identity(model)
    self.assertEqual(identity['model'], {'length': 128, 'hidden_size': 8})
    self.assertEqual(identity['noise']['eps'], 0.001)
    model.config.noise.eps = 0.01
    self.assertNotEqual(backbone_runtime_identity(model), identity)

  def test_document_identity_checks_catch_overlap_even_with_different_tokens(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'train.jsonl'
      rows = [{'document_id': 'doc-a', 'input_ids': self.tokens[0].tolist()},
              {'document_id': 'doc-b', 'input_ids': self.tokens[1].tolist()}]
      path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
      tokens, documents, source = read_documents(path, file_sha256(path), 2, 6, 11, 10)
      self.assertEqual(documents, ['doc-a', 'doc-b'])
      self.assertEqual(source['examples'], 2)
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        read_documents(path, '0' * 64, 2, 6, 11, 10)
      with self.assertRaisesRegex(ValueError, 'document identities overlap'):
        assert_document_disjoint(tokens, documents, self.tokens[2:], ['doc-a', 'doc-c'])
      assert_document_disjoint(tokens, documents, self.tokens[2:], ['doc-c', 'doc-d'])


if __name__ == '__main__':
  unittest.main()
