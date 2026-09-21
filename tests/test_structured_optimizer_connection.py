"""Verify optimizer ownership, real CCF updates, and baseline AdamW behavior."""

import copy
import itertools

import pytest
import torch

import structured_training
from test_structured_initialization import (
  _checkpoint, _config, _head_config, _model, offline_eval_tokenizer,
)


pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason='real DiT requires CUDA')


def _optimizer_config(cfg, warmup=0):
  cfg.optim.update({'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8,
                    'weight_decay': 0.01})
  cfg.lr_scheduler = {
    '_target_': 'transformers.get_constant_schedule_with_warmup',
    'num_warmup_steps': warmup,
  }
  return cfg


def _ccf(tmp_path, topology='fixed', factor='fixed'):
  torch.manual_seed(73)
  path, _ = _checkpoint(tmp_path)
  cfg = _optimizer_config(_config(_head_config(path, topology, factor)))
  cfg.model.length = 6
  cfg.model.structured_decoder.training.topology_weight = 0.35
  model = _model(cfg).cuda()
  model.on_train_epoch_start()
  return model


def _optimizer(model):
  optimizers, schedulers = model.configure_optimizers()
  assert len(optimizers) == len(schedulers) == 1
  assert schedulers[0]['interval'] == 'step'
  assert schedulers[0]['scheduler'].optimizer is optimizers[0]
  return optimizers[0], schedulers[0]['scheduler']


def _ids(optimizer):
  return [id(p) for group in optimizer.param_groups for p in group['params']]


@pytest.mark.parametrize('topology,factor', [
  ('fixed', 'fixed'), ('fixed', 'dynamic'),
  ('dynamic', 'fixed'), ('dynamic', 'dynamic'),
])
def test_four_arms_update_head_but_never_backbone(tmp_path, monkeypatch,
                                                topology, factor):
  model = _ccf(tmp_path, topology, factor)
  optimizer, scheduler = _optimizer(model)
  expected = [p for module in (model.structured_head, model.noise)
              for p in module.parameters() if p.requires_grad]
  assert _ids(optimizer) == [id(p) for p in expected]
  assert len(_ids(optimizer)) == len(set(_ids(optimizer)))
  assert not set(_ids(optimizer)) & {id(p) for p in model.backbone.parameters()}
  backbone_before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
  head_before = {k: v.clone() for k, v in model.structured_head.named_parameters()}
  gold = torch.tensor([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], device='cuda')
  # Fixed corruption isolates optimizer behavior; the real loss/teacher still run.
  corrupted = gold.clone()
  corrupted[:, :4] = model.mask_index
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted.clone())
  monkeypatch.setattr(structured_training, 'sample_active_sources',
                      lambda *args, **kwargs: torch.tensor([0, 0], device='cuda'))
  for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    loss = model._loss(gold, torch.ones_like(gold)).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in expected)
    optimizer.step()
    scheduler.step()
  changed = {name for name, p in model.structured_head.named_parameters()
             if not torch.equal(p, head_before[name])}
  assert 'token_factor_embedding.weight' in changed
  if factor == 'dynamic':
    assert any(name.startswith('factor_hidden_projection.') for name in changed)
  if topology == 'dynamic':
    assert any(name.startswith('edge_proposer.edge_scorer.') for name in changed)
    assert any(name.startswith('edge_proposer.anchor_projection.') for name in changed)
  for name, value in model.backbone.state_dict().items():
    torch.testing.assert_close(value, backbone_before[name], rtol=0, atol=0)
  assert not model.backbone.training


@pytest.mark.parametrize('head_lr', ['missing', None, 0.0012, 0.0])
def test_head_lr_and_warmup(tmp_path, head_lr):
  model = _ccf(tmp_path)
  if head_lr != 'missing':
    model.structured_training_config.head_lr = head_lr
  model.config.lr_scheduler.num_warmup_steps = 2
  optimizer, scheduler = _optimizer(model)
  expected = model.config.optim.lr if head_lr in ('missing', None) else head_lr
  group = optimizer.param_groups[0]
  assert group['name'] == 'structured_head'
  assert group['initial_lr'] == expected
  assert group['lr'] == 0.0
  for fraction in (0.5, 1.0):
    optimizer.step()
    scheduler.step()
    assert group['lr'] == pytest.approx(expected * fraction)


@pytest.mark.parametrize('head_lr', [-0.01, float('nan'), float('inf')])
def test_invalid_head_lr_rejected(tmp_path, head_lr):
  model = _ccf(tmp_path)
  model.structured_training_config.head_lr = head_lr
  with pytest.raises(ValueError, match='head_lr must be finite and non-negative'):
    model.configure_optimizers()


def test_parameter_filter_includes_trainable_noise_and_excludes_frozen_head(tmp_path):
  model = _ccf(tmp_path)
  # Current loglinear noise has no parameters; this checks the historical group
  # policy without claiming support for a new learned-noise training objective.
  noise_parameter = torch.nn.Parameter(torch.tensor(0.4, device='cuda'))
  model.noise.register_parameter('test_parameter', noise_parameter)
  frozen = model.structured_head.token_factor_embedding.weight
  frozen.requires_grad_(False)
  optimizer, _ = _optimizer(model)
  assert id(noise_parameter) in _ids(optimizer)
  assert id(frozen) not in _ids(optimizer)


def test_baseline_optimizer_matches_original_adamw(tmp_path):
  cfg = _optimizer_config(_config())
  model = _model(cfg).cuda()
  reference = _model(copy.deepcopy(cfg)).cuda()
  reference.load_state_dict(model.state_dict())
  optimizer, scheduler = _optimizer(model)
  reference_optimizer = torch.optim.AdamW(
    itertools.chain(reference.backbone.parameters(), reference.noise.parameters()),
    lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2),
    eps=cfg.optim.eps, weight_decay=cfg.optim.weight_decay)
  assert _ids(optimizer) == [id(p) for module in (model.backbone, model.noise)
                             for p in module.parameters()]
  assert 'name' not in optimizer.param_groups[0]
  for _ in range(2):
    for actual, expected in zip(model.backbone.parameters(), reference.backbone.parameters()):
      actual.grad = torch.full_like(actual, 0.125)
      expected.grad = actual.grad.clone()
    optimizer.step()
    reference_optimizer.step()
    scheduler.step()
  for actual, expected in zip(model.backbone.parameters(), reference.backbone.parameters()):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_optimizer_and_scheduler_state_roundtrip(tmp_path):
  model = _ccf(tmp_path)
  model.config.lr_scheduler.num_warmup_steps = 3
  optimizer, scheduler = _optimizer(model)
  for p in model.structured_head.parameters():
    p.grad = torch.full_like(p, 0.125)
  optimizer.step()
  scheduler.step()
  path = tmp_path / 'optimizer-state.pt'
  torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict()}, path)
  restored = _model(copy.deepcopy(model.config)).cuda()
  restored_optimizer, restored_scheduler = _optimizer(restored)
  saved = torch.load(path, weights_only=False)
  restored.load_state_dict(saved['model'], strict=True)
  restored_optimizer.load_state_dict(saved['optimizer'])
  restored_scheduler.load_state_dict(saved['scheduler'])
  for left, right in zip(model.structured_head.parameters(), restored.structured_head.parameters()):
    left.grad = torch.full_like(left, -0.25)
    right.grad = left.grad.clone()
  optimizer.step()
  restored_optimizer.step()
  scheduler.step()
  restored_scheduler.step()
  for left, right in zip(model.parameters(), restored.parameters()):
    torch.testing.assert_close(left, right, rtol=0, atol=0)
  assert scheduler.state_dict() == restored_scheduler.state_dict()
  # This checks optimizer moments/LR restoration, not Trainer or corruption RNG resume.
