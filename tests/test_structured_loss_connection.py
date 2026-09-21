"""Real DiT -> CCF -> objective checks, before optimizer/teacher integration."""

import copy

import pytest
import torch

import diffusion
from structured_training import structured_denoising_loss
from test_structured_initialization import (
  _checkpoint, _config, _head_config, _model, offline_eval_tokenizer,
)


pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason='real DiT requires CUDA')


def _configured_model(tmp_path, topology='fixed', factor='fixed'):
  path, _ = _checkpoint(tmp_path)
  config = _config(_head_config(path, topology, factor))
  config.model.length = 4
  model = _model(config).cuda().train()
  model.on_train_epoch_start()
  return model


@pytest.mark.parametrize('topology,factor', [
  ('fixed', 'fixed'), ('fixed', 'dynamic'),
  ('dynamic', 'fixed'), ('dynamic', 'dynamic'),
])
def test_loss_route_and_gradient_boundary(tmp_path, monkeypatch, topology, factor):
  model = _configured_model(tmp_path, topology, factor)
  gold = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], device='cuda')
  attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], device='cuda')
  corrupted = torch.tensor([[11, 11, 3, 11], [2, 11, 11, 5]], device='cuda')
  time = torch.tensor([0.4, 0.8], device='cuda')
  monkeypatch.setattr(model, '_sample_t', lambda *args, **kwargs: time)
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted)
  active = (corrupted == 11) & attention.bool()
  seen_inputs = []
  seen_times = []
  hook = model.backbone.vocab_embed.register_forward_pre_hook(
    lambda module, args: seen_inputs.append(args[0].clone()))
  head_hook = model.structured_head.register_forward_pre_hook(
    lambda module, args, kwargs: seen_times.append(kwargs['timestep'].clone()),
    with_kwargs=True)
  before = {name: value.clone() for name, value in model.backbone.state_dict().items()}
  loss = model._loss(gold, attention)
  hook.remove()
  head_hook.remove()
  assert len(seen_inputs) == 1
  assert torch.equal(seen_inputs[0], corrupted)
  sigma, _ = model.noise(time)
  torch.testing.assert_close(seen_times[0], sigma)
  output, logits = model._structured_head_output(corrupted, sigma[:, None], active)
  expected = structured_denoising_loss(output, logits, gold, active)
  torch.testing.assert_close(loss.loss, expected.loss)
  assert torch.equal(loss.token_mask, active)
  torch.testing.assert_close(loss.nlls.sum(), loss.loss * active.sum())
  loss.loss.backward()
  assert all(parameter.grad is None for parameter in model.backbone.parameters())
  grads = [p.grad for p in model.structured_head.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads)
  assert model.structured_head.token_factor_embedding.weight.grad.abs().sum() > 0
  # Score-selection is discrete: the zero connection must not pretend to train it.
  for parameter in model.structured_head.edge_proposer.parameters():
    assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
  for name, value in before.items():
    torch.testing.assert_close(model.backbone.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('single_node,empty', [(False, True), (True, False)])
def test_empty_and_singleton_batches_can_backward(tmp_path, monkeypatch, single_node, empty):
  model = _configured_model(tmp_path)
  gold = torch.tensor([[1]] if single_node else [[1, 2, 3, 4]], device='cuda')
  corrupted = gold.clone() if empty else torch.full_like(gold, 11)
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted)
  loss = model._loss(gold, None)
  assert torch.isfinite(loss.loss)
  if empty:
    torch.testing.assert_close(loss.loss, torch.zeros_like(loss.loss), atol=1e-5, rtol=0)
  loss.loss.backward()
  assert all(p.grad is None for p in model.backbone.parameters())


def test_corruption_stream_is_separate_from_global_rng(tmp_path):
  first = _configured_model(tmp_path)
  second = _configured_model(tmp_path)
  gold = torch.ones(2, 4, dtype=torch.long, device='cuda')
  for _ in range(3):
    g1 = first._structured_training_corruption_generator
    g2 = second._structured_training_corruption_generator
    t1 = first._sample_t(2, gold.device, generator=g1)
    x1 = first.q_xt(gold, t1[:, None], generator=g1)
    torch.rand(37, device='cuda')  # e.g. extra draws in another arm
    t2 = second._sample_t(2, gold.device, generator=g2)
    x2 = second.q_xt(gold, t2[:, None], generator=g2)
    assert torch.equal(t1, t2) and torch.equal(x1, x2)


def test_training_step_logs_conditional_metric_not_baseline_ppl(tmp_path, monkeypatch):
  model = _configured_model(tmp_path)
  gold = torch.tensor([[1, 2, 3, 4]], device='cuda')
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: torch.full_like(gold, 11))
  logged = []
  monkeypatch.setattr(model, 'log_dict', lambda metrics, **kwargs: logged.append(metrics))
  monkeypatch.setattr(model, 'log', lambda *args, **kwargs: None)
  loss = model.training_step({'input_ids': gold}, 0)
  assert torch.isfinite(loss)
  assert set(logged[0].keys()) == {'train/conditional_nll_per_masked_token'}
  assert model.train_metrics.nll.weight == 0
  torch.testing.assert_close(
    model.structured_train_metrics.compute()['train/conditional_nll_per_masked_token'],
    loss.detach().double())


def test_gold_targets_do_not_enter_primary_head(tmp_path, monkeypatch):
  model = _configured_model(tmp_path, 'dynamic', 'dynamic')
  gold = torch.tensor([[1, 2, 3, 4]], device='cuda')
  corrupted = torch.full_like(gold, 11)
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted)
  monkeypatch.setattr(model, '_sample_t', lambda *args, **kwargs: torch.tensor([0.5], device='cuda'))
  results = []
  hook = model.structured_head.register_forward_hook(
    lambda module, args, output: results.append(output))
  model._loss(gold, None)
  model._loss(gold.flip(-1), None)
  hook.remove()
  for field in ('candidate_ids', 'edge_index', 'edge_mask', 'pair_left_factors',
                'pair_right_factors', 'unary_log_potentials'):
    torch.testing.assert_close(getattr(results[0], field), getattr(results[1], field),
                               rtol=0, atol=0)
