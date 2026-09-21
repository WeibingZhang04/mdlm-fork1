"""Check reveal routing, loss composition, gating and gradient isolation."""

from dataclasses import replace

import pytest
import torch

import structured_training
from test_structured_initialization import (
  _checkpoint, _config, _head_config, _model, offline_eval_tokenizer,
)


pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason='real DiT requires CUDA')


def _teacher_model(tmp_path, topology='dynamic', weight=0.35,
                   independent=False, training=True, on_validation=False):
  torch.manual_seed(73)
  path, _ = _checkpoint(tmp_path)
  cfg = _config(_head_config(path, topology=topology, factor='dynamic'))
  cfg.model.length = 6
  cfg.model.structured_decoder.independent_mode = independent
  cfg.model.structured_decoder.training.topology_weight = weight
  cfg.model.structured_decoder.training.topology_on_validation = on_validation
  model = _model(cfg).cuda()
  model.on_train_epoch_start()
  return model.train(training)


def _batch():
  gold = torch.tensor([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9],
                       [7, 8, 9, 1, 2, 3]], device='cuda')
  corrupted = gold.clone()
  corrupted[0, :3] = 11
  corrupted[0, 5] = 11  # padded position must never be revealed
  corrupted[1, 1] = 11
  attention = torch.ones_like(gold)
  attention[0, 5] = 0
  return gold, corrupted, attention


def test_reveal_inputs_weighting_and_gradient_routes(tmp_path, monkeypatch):
  model = _teacher_model(tmp_path)
  gold, corrupted, attention = _batch()
  original_corrupted = corrupted.clone()
  sources = torch.tensor([0, 1, -1], device='cuda')
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted)
  monkeypatch.setattr(structured_training, 'sample_active_sources',
                      lambda *args, **kwargs: sources)
  backbone_calls, head_calls, topology_calls = [], [], []
  original_backbone = model._structured_backbone_output
  original_head = model._structured_head_output
  original_teacher = structured_training.gold_reveal_influence_topology_loss

  def backbone(tokens, conditioning):
    result = original_backbone(tokens, conditioning)
    backbone_calls.append((tokens.clone(), conditioning.clone(), result))
    return result

  def head(*args):
    result = original_head(*args)
    head_calls.append(result)
    return result

  def teacher(**kwargs):
    result = original_teacher(**kwargs)
    topology_calls.append((kwargs, result))
    return result

  monkeypatch.setattr(model, '_structured_backbone_output', backbone)
  monkeypatch.setattr(model, '_structured_head_output', head)
  monkeypatch.setattr(structured_training, 'gold_reveal_influence_topology_loss', teacher)
  loss = model._loss(gold, attention)
  assert len(backbone_calls) == 2 and len(head_calls) == 1
  assert torch.equal(corrupted, original_corrupted)
  assert torch.equal(backbone_calls[0][0], corrupted)
  expected_reveal = corrupted.clone()
  expected_reveal[0, 0], expected_reveal[1, 1] = gold[0, 0], gold[1, 1]
  assert torch.equal(backbone_calls[1][0], expected_reveal)
  assert (backbone_calls[1][0] != corrupted).sum(-1).tolist() == [1, 1, 0]
  torch.testing.assert_close(backbone_calls[0][1], backbone_calls[1][1])
  for _, _, (hidden, logits) in backbone_calls:
    assert not hidden.requires_grad and not logits.requires_grad
  kwargs, topology = topology_calls[0]
  output, logits = head_calls[0]
  assert kwargs['output'] is output
  active = corrupted.eq(11) & attention.bool()
  assert torch.equal(kwargs['active_mask'], active)
  assert torch.equal(kwargs['source_positions'], sources)
  denoising = structured_training.structured_denoising_loss(output, logits, gold, active)
  torch.testing.assert_close(loss.loss, denoising.loss + 0.35 * topology.loss)
  torch.testing.assert_close(loss.nlls, denoising.distributed_nll)
  assert topology.edge_coverage_denominator == 1
  assert topology.anchor_coverage_denominator == 1
  assert topology.edge_coverage_numerator == 1
  assert topology.anchor_coverage_numerator == 1
  loss.loss.backward()
  assert all(p.grad is None for p in model.backbone.parameters())
  for module in (model.structured_head.edge_proposer.edge_scorer,
                 model.structured_head.edge_proposer.anchor_projection):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
  assert all(not value.requires_grad
             for value in model._last_structured_topology_metrics.values())


def test_slot_router_gradient_with_two_valid_anchor_choices(tmp_path, monkeypatch):
  model = _teacher_model(tmp_path)
  gold, corrupted, attention = _batch()
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted)
  monkeypatch.setattr(structured_training, 'sample_active_sources',
                      lambda *args, **kwargs: torch.tensor([0, 1, -1], device='cuda'))
  original = model._structured_head_output

  def controlled_occupancy(*args):
    output, logits = original(*args)
    # Only this test fixes hard slot occupants to guarantee router coverage.
    output = replace(output, anchor_indices=torch.tensor(
      [[1, 2], [0, 2], [0, 1]], device='cuda'))
    return output, logits

  monkeypatch.setattr(model, '_structured_head_output', controlled_occupancy)
  loss = model._loss(gold, attention)
  loss.loss.backward()
  assert model._last_structured_topology_metrics['slot_coverage_numerator'] == 1
  grads = [p.grad for p in model.structured_head.edge_proposer.slot_projection.parameters()]
  assert all(g is not None and torch.isfinite(g).all() for g in grads)
  assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize('topology,weight,independent,training,on_validation,passes', [
  ('fixed', 0.35, False, True, False, 1),
  ('dynamic', 0.0, False, True, False, 1),
  ('dynamic', 0.35, True, True, False, 1),
  ('dynamic', 0.35, False, False, False, 1),
  ('dynamic', 0.35, False, False, True, 2),
  ('dynamic', 0.35, False, True, False, 2),
])
def test_teacher_gating(tmp_path, monkeypatch, topology, weight, independent,
                       training, on_validation, passes):
  model = _teacher_model(tmp_path, topology, weight, independent, training, on_validation)
  gold, corrupted, attention = _batch()
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted)
  calls = []
  hook = model.backbone.vocab_embed.register_forward_pre_hook(
    lambda module, args: calls.append(args[0].clone()))
  before = model._structured_training_topology_generator.get_state()
  result = model._loss(gold, attention)
  hook.remove()
  assert torch.isfinite(result.loss)
  assert len(calls) == passes
  if passes == 1:
    assert not model._last_structured_topology_metrics
    assert torch.equal(before, model._structured_training_topology_generator.get_state())
  else:
    assert model._last_structured_topology_metrics


def test_teacher_sampling_does_not_change_next_corruption(tmp_path):
  enabled = _teacher_model(tmp_path, weight=0.35)
  disabled = _teacher_model(tmp_path, weight=0.0)
  gold, _, attention = _batch()
  for _ in range(3):
    on = enabled._loss(gold, attention)
    off = disabled._loss(gold, attention)
    assert torch.equal(on.token_mask, off.token_mask)
    torch.testing.assert_close(on.nlls, off.nlls, rtol=0, atol=0)
    assert torch.equal(enabled._structured_training_corruption_generator.get_state(),
                       disabled._structured_training_corruption_generator.get_state())


def test_all_inactive_teacher_loss_is_zero_and_backward_finite(tmp_path, monkeypatch):
  model = _teacher_model(tmp_path)
  gold, _, attention = _batch()
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: gold.clone())
  loss = model._loss(gold, attention)
  torch.testing.assert_close(loss.loss, loss.loss.new_zeros(()), rtol=0, atol=1e-5)
  loss.loss.backward()
  assert model._last_structured_topology_metrics['loss'] == 0
  assert model._last_structured_topology_metrics['valid_examples'] == 0
  for parameter in model.structured_head.parameters():
    assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_logged_conditional_nll_excludes_teacher_loss(tmp_path, monkeypatch):
  model = _teacher_model(tmp_path)
  gold, corrupted, attention = _batch()
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted)
  monkeypatch.setattr(model, 'log_dict', lambda *args, **kwargs: None)
  loss = model._compute_loss({'input_ids': gold, 'attention_mask': attention}, 'train')
  nll = model.structured_train_metrics.compute()['train/conditional_nll_per_masked_token']
  auxiliary = model._last_structured_topology_metrics['loss']
  assert auxiliary > 0
  torch.testing.assert_close(loss.detach().double(), nll + 0.35 * auxiliary.double())
