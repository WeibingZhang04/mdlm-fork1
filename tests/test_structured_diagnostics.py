"""Check diagnostic populations, aggregation, logging, and gradient isolation."""

from dataclasses import replace
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

import diffusion
import structured_training
from test_structured_initialization import offline_eval_tokenizer
from test_structured_optimizer_connection import _ccf


def test_ratio_uses_total_counts_and_handles_empty_reset():
  metric = diffusion.RatioMetric()
  metric.update(torch.tensor(1., requires_grad=True), 1)
  metric.update(0., 9)
  metric.update(0., 0)
  assert metric.compute() == 0.1  # not (1 + 0) / 2
  assert not metric.numerator.requires_grad
  metric.reset()
  metric.update(0., 0)
  assert metric.compute() == 0


def _distributed_worker(rank, rendezvous, result):
  torch.distributed.init_process_group(
    'gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
  try:
    ratio, count = diffusion.RatioMetric(), diffusion.DistributedSumMetric()
    ratio.update(1 if rank == 0 else 0, 1 if rank == 0 else 3)
    count.update(rank + 1)
    value, total = ratio.compute(), count.compute()
    if rank == 0:
      torch.save((value, total), result)
  finally:
    torch.distributed.destroy_process_group()


def test_distributed_metrics_reduce_counts_before_division(tmp_path):
  result = str(tmp_path / 'distributed.pt')
  torch.multiprocessing.spawn(
    _distributed_worker, args=(str(tmp_path / 'rendezvous'), result),
    nprocs=2, join=True)
  ratio, count = torch.load(result, weights_only=False)
  assert ratio == 0.25 and count == 3


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='real DiT requires CUDA')


def _capture(model, monkeypatch):
  records = []
  original = model._structured_head_output
  def head(*args):
    result = original(*args)
    records.append(result)
    return result
  monkeypatch.setattr(model, '_structured_head_output', head)
  monkeypatch.setattr(model, 'log_dict', lambda *args, **kwargs: None)
  monkeypatch.setattr(model, '_sample_t', lambda n, device, **kwargs:
                      torch.full((n,), 0.5, device=device))
  return records


@cuda
def test_token_metrics_match_independent_oracles_over_unequal_batches(tmp_path, monkeypatch):
  model = _ccf(tmp_path)
  records = _capture(model, monkeypatch)
  total_nll = baseline_nll = hits = retained = 0.
  active_count = attention_count = 0
  for gold_rows, mask_rows, attention_rows in [
    ([[1, 2, 3, 4]], [[True, False, False, True]], [[1, 1, 1, 0]]),
    ([[4, 5, 6, 7], [7, 8, 9, 1]], [[True]*4, [True, False, False, False]],
     [[1]*4, [1]*4]),
    ([[1, 2, 3, 4]], [[False]*4], [[1]*4]),
  ]:
    gold = torch.tensor(gold_rows, device='cuda')
    attention = torch.tensor(attention_rows, device='cuda')
    corrupted = gold.masked_fill(torch.tensor(mask_rows, device='cuda'), model.mask_index)
    monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted.clone())
    loss = model._compute_loss({'input_ids': gold, 'attention_mask': attention}, 'train')
    output, logits = records[-1]
    active = corrupted.eq(model.mask_index) & attention.bool()
    count = active.sum().item()
    total_nll += loss.detach().item() * count  # no teacher in fixed topology
    log_probs = torch.log_softmax(logits.double(), -1)
    gold_log_probs = log_probs.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    baseline_nll += -gold_log_probs[active].sum().item()
    hits += (output.candidate_ids.eq(gold.unsqueeze(-1)).any(-1) & active).sum().item()
    retained += log_probs.exp().gather(-1, output.candidate_ids).sum(-1)[active].sum().item()
    active_count += count
    attention_count += attention.sum().item()
    assert all(not value.requires_grad
               for args in model._last_structured_metric_updates.values() for value in args)
  nll = model.structured_train_metrics.compute()['train/conditional_nll_per_masked_token']
  metrics = model.structured_train_diagnostics.compute()
  prefix = 'train/structured/'
  assert nll.item() == pytest.approx(total_nll / active_count, abs=1e-6)
  assert metrics[prefix+'factorized_nll_per_masked_token'].item() == pytest.approx(
    baseline_nll / active_count, abs=1e-6)
  assert metrics[prefix+'candidate_recall'].item() == pytest.approx(hits / active_count)
  assert metrics[prefix+'retained_unary_mass'].item() == pytest.approx(retained / active_count, abs=1e-6)
  assert metrics[prefix+'active_fraction'].item() == pytest.approx(active_count / attention_count)
  assert metrics[prefix+'active_tokens'] == active_count
  assert metrics[prefix+'teacher_examples'] == 0
  assert model.train_metrics.nll.weight == 0  # never relabel as diffusion PPL


@cuda
def test_diagnostics_leave_loss_gradients_and_rng_unchanged(tmp_path, monkeypatch):
  model = _ccf(tmp_path, 'dynamic', 'dynamic')
  _capture(model, monkeypatch)
  gold = torch.tensor([[1, 2, 3, 4, 5, 6]], device='cuda')
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: torch.full_like(gold, 11))
  monkeypatch.setattr(structured_training, 'sample_active_sources',
                      lambda *args, **kwargs: torch.tensor([0], device='cuda'))
  def run():
    model.zero_grad(set_to_none=True)
    loss = model._loss(gold, torch.ones_like(gold)).loss
    loss.backward()
    gradients = {name: None if p.grad is None else p.grad.clone()
                 for name, p in model.named_parameters()}
    return loss.detach(), gradients, torch.cuda.get_rng_state().clone()
  # Exact comparison needs deterministic CUDA reductions; this is a test-only
  # setting, not a change to production training's determinism policy.
  monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
  previous = torch.are_deterministic_algorithms_enabled()
  previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
  torch.use_deterministic_algorithms(True)
  try:
    expected_loss, expected_grad, expected_rng = run()
    monkeypatch.setattr(model, '_record_structured_diagnostics', lambda *args: None)
    actual_loss, actual_grad, actual_rng = run()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert torch.equal(actual_rng, expected_rng)
    for name, expected in expected_grad.items():
      actual = actual_grad[name]
      if expected is None:
        assert actual is None
      else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  finally:
    torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@cuda
def test_duplicate_anchor_loss_count_differs_from_coverage_population(tmp_path, monkeypatch):
  model = _ccf(tmp_path, 'dynamic', 'dynamic')
  _capture(model, monkeypatch)
  # Only one other active position: teacher minimum_choices=2 is not met,
  # but both slots can point to it and still produce a two-slot loss.
  gold = torch.tensor([[1, 2, 3, 4, 5, 6]], device='cuda')
  corrupted = gold.clone()
  corrupted[:, :2] = 11
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted.clone())
  monkeypatch.setattr(structured_training, 'sample_active_sources',
                      lambda *args, **kwargs: torch.tensor([0], device='cuda'))
  original_head = model._structured_head_output
  def head(*args):
    output, logits = original_head(*args)
    return replace(output, anchor_indices=torch.tensor([[1, 1]], device='cuda')), logits
  monkeypatch.setattr(model, '_structured_head_output', head)
  model._compute_loss({'input_ids': gold}, 'train')
  metrics = model.structured_train_diagnostics.compute()
  prefix = 'train/structured/'
  assert metrics[prefix+'topology_slot_valid_examples'] == 1
  assert metrics[prefix+'topology_slot_coverage_denominator'] == 0
  assert metrics[prefix+'topology_slot_coverage_numerator'] == 0
  assert metrics[prefix+'topology_slot_loss'] > 0
  assert metrics[prefix+'teacher_examples'] == 1
  assert metrics[prefix+'teacher_eligible_fraction'] == 0


@cuda
def test_topology_loss_aggregation_uses_each_actual_population(tmp_path):
  model = _ccf(tmp_path)
  gold = torch.tensor([[1, 2]], device='cuda')
  active = torch.ones_like(gold, dtype=torch.bool)
  logits = torch.zeros(1, 2, 12, device='cuda')
  tensor = lambda value: torch.tensor(float(value), device='cuda')
  denoising = SimpleNamespace(active_tokens=tensor(2), loss=tensor(1),
    nll_sum=tensor(2), candidate_hits=tensor(1), retained_mass_sum=tensor(1))
  for edge_loss, edge_count, slot_loss, slot_count, eligible, batch_size in [
    (2, 1, 9, 0, 1, 1), (6, 3, 4, 2, 3, 4),
  ]:
    top = SimpleNamespace(loss=tensor(edge_loss + slot_loss))
    for view, loss, count in [('edge',edge_loss,edge_count), ('anchor',1,eligible),
                              ('slot',slot_loss,slot_count)]:
      setattr(top, view+'_loss', tensor(loss))
      setattr(top, view+'_valid_examples', tensor(count))
      setattr(top, view+'_coverage_numerator', tensor(count))
      setattr(top, view+'_coverage_denominator', tensor(eligible))
    model._record_structured_diagnostics(denoising, logits.expand(batch_size,-1,-1),
      gold.expand(batch_size,-1), active.expand(batch_size,-1),
      active.expand(batch_size,-1), top)
    for name, args in model._last_structured_metric_updates.items():
      if name != 'conditional_nll_per_masked_token':
        model.structured_train_diagnostics[name].update(*args)
  metrics = model.structured_train_diagnostics.compute()
  assert metrics['train/structured/topology_edge_loss'] == 5  # (2*1 + 6*3)/4
  assert metrics['train/structured/topology_slot_loss'] == 4
  assert metrics['train/structured/topology_slot_coverage'] == 0.5
  assert metrics['train/structured/teacher_eligible_fraction'] == 0.8


@cuda
def test_empty_validation_metrics_are_finite_and_teacher_is_not_run(tmp_path, monkeypatch):
  model = _ccf(tmp_path, 'dynamic', 'dynamic').eval()
  records = _capture(model, monkeypatch)
  gold = torch.tensor([[1, 2, 3, 4]], device='cuda')
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: gold.clone())
  model._compute_loss({'input_ids': gold}, 'val')
  assert len(records) == 1 and not model._last_structured_topology_metrics
  for collection in (model.structured_valid_metrics, model.structured_valid_diagnostics):
    assert all(torch.isfinite(v) and v == 0 for v in collection.compute().values())
  assert model.structured_train_metrics['conditional_nll_per_masked_token'].denominator == 0


@cuda
def test_lightning_validation_logs_and_resets_metrics(tmp_path, monkeypatch):
  model = _ccf(tmp_path)
  model.config.eval.generate_samples = False
  model.config.eval.compute_perplexity_on_sanity = False
  monkeypatch.setattr(model, '_sample_t', lambda n, device, **kwargs:
                      torch.full((n,), 0.5, device=device))
  monkeypatch.setattr(model, 'q_xt', lambda gold, *args, **kwargs: torch.full_like(gold, 11))
  dataset = [
    {'input_ids': torch.tensor([1,2,3,4]), 'attention_mask': torch.tensor([1,0,0,0])},
    {'input_ids': torch.tensor([4,5,6,7]), 'attention_mask': torch.tensor([1,1,1,1])},
  ]
  loader = DataLoader(dataset, batch_size=1)
  trainer = L.Trainer(accelerator='gpu', devices=1, logger=False,
    enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
  first = trainer.validate(model, dataloaders=loader, verbose=False)[0]
  second = trainer.validate(model, dataloaders=loader, verbose=False)[0]
  assert first == second
  assert first['val/structured/active_tokens'] == 5
  assert first['val/structured/active_fraction'] == 1
  assert first['val/structured/candidate_recall'] >= 0
  assert 'val/ppl' not in first
  assert model.structured_valid_metrics['conditional_nll_per_masked_token'].denominator == 0
