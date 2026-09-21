"""Verify packed data, real tokenizer padding, and active-node likelihoods."""

import itertools

import datasets
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import dataloader
from test_structured_initialization import offline_eval_tokenizer
from test_structured_loss_connection import _configured_model


@pytest.mark.parametrize('tokens,expected', [
  ([], []),
  ([4, 5, 6], []),
  ([4, 5, 6, 7], [[0, 4, 5, 6, 7, 1]]),
  (list(range(4, 14)), [[0, 4, 5, 6, 7, 1], [0, 8, 9, 10, 11, 1]]),
])
def test_group_texts_full_blocks_and_dropped_remainder(tokens, expected):
  result = dataloader._group_texts(
    {'input_ids': [tokens[:2], tokens[2:]]}, block_size=6, bos=0, eos=1)
  assert result['input_ids'] == expected
  assert len(result['attention_mask']) == len(expected)
  for mask in result['attention_mask']:
    assert torch.equal(mask, torch.ones(6))


@pytest.mark.parametrize('wrap', [False, True])
def test_real_tokenization_branch_padding(tmp_path, monkeypatch, wrap):
  raw = Tokenizer(WordLevel(
    {'[BOS]': 0, '[EOS]': 1, '[PAD]': 2, '[UNK]': 3, 'a': 4, 'b': 5, 'c': 6},
    unk_token='[UNK]'))
  raw.pre_tokenizer = Whitespace()
  tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw, bos_token='[BOS]', eos_token='[EOS]',
    pad_token='[PAD]', unk_token='[UNK]')
  source = datasets.Dataset.from_dict({'text': ['a b', 'c']})
  monkeypatch.setattr(dataloader.datasets, 'load_dataset',
                      lambda *args, **kwargs: {'train': source})
  result = dataloader.get_dataset(
    'mask_fixture', tokenizer, wrap=wrap, mode='train', cache_dir=str(tmp_path),
    block_size=6, num_proc=1, streaming=False)
  if wrap:
    # EOS is appended to each document, then concatenation drops the tail.
    assert result['input_ids'].tolist() == [[0, 4, 5, 1, 6, 1]]
    assert result['attention_mask'].bool().all()
    assert not result['input_ids'].eq(2).any()
  else:
    assert result['input_ids'].tolist() == [[4, 5, 2, 2, 2, 2], [6, 2, 2, 2, 2, 2]]
    assert result['attention_mask'].tolist() == [[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
    assert torch.equal(result['attention_mask'].bool(), result['input_ids'].ne(2))


def _enumerated_nll(output, logits, gold, active, mask_id):
  """Enumerate actual vocabulary tokens, never compressed/residual states."""
  vocabulary = [v for v in range(logits.shape[-1]) if v != mask_id]
  total_nll = logits.new_zeros(())
  for b in range(gold.shape[0]):
    nodes = active[b].nonzero().flatten().tolist()
    if not nodes:
      continue
    assignments = torch.tensor(list(itertools.product(vocabulary, repeat=len(nodes))),
                               device=logits.device)
    log_weight = torch.stack([
      logits[b, node, assignments[:, col]] for col, node in enumerate(nodes)
    ]).sum(0)
    column = {node: col for col, node in enumerate(nodes)}
    for edge in output.edge_mask[b].nonzero().flatten().tolist():
      left, right = output.edge_index[b, edge].tolist()
      assert left in column and right in column
      dense = logits.new_ones((logits.shape[-1], logits.shape[-1]))
      ids_left = output.candidate_ids[b, left]
      ids_right = output.candidate_ids[b, right]
      dense[ids_left[:, None], ids_right[None, :]] = (
        output.pair_left_factors[b, edge] @ output.pair_right_factors[b, edge].T)
      log_weight = log_weight + dense[
        assignments[:, column[left]], assignments[:, column[right]]].log()
    is_gold = assignments.eq(gold[b, nodes]).all(-1)
    assert is_gold.sum() == 1
    total_nll = total_nll + torch.logsumexp(log_weight, 0) - log_weight[is_gold].squeeze(0)
  return total_nll / active.sum().clamp_min(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real DiT needs CUDA')
@pytest.mark.parametrize('topology,factor', [
  ('fixed', 'fixed'), ('fixed', 'dynamic'), ('dynamic', 'fixed'), ('dynamic', 'dynamic'),
])
def test_padded_loss_and_gradients_match_full_vocabulary(tmp_path, monkeypatch, topology, factor):
  torch.manual_seed(49)
  model = _configured_model(tmp_path, topology, factor)
  gold = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]], device='cuda')
  attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], device='cuda')
  corrupted = torch.tensor([[11, 2, 11, 11], [4, 11, 6, 11]], device='cuda')
  expected_active = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], device='cuda').bool()
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: corrupted)
  monkeypatch.setattr(model, '_sample_t', lambda *args, **kwargs: torch.tensor([0.3, 0.7], device='cuda'))
  captured = []
  original = model._structured_head_output

  def capture(tokens, conditioning, active_mask):
    assert torch.equal(active_mask, expected_active)
    result = original(tokens, conditioning, active_mask)
    captured.append(result)
    return result

  monkeypatch.setattr(model, '_structured_head_output', capture)
  result = model._loss(gold, attention)
  output, logits = captured[0]
  assert torch.equal(result.token_mask, expected_active)
  assert not result.nlls[~expected_active].any()
  for b in range(2):
    endpoints = output.edge_index[b, output.edge_mask[b]]
    assert expected_active[b, endpoints].all()
  oracle = _enumerated_nll(output, logits, gold, expected_active, model.mask_index)
  torch.testing.assert_close(result.loss, oracle, rtol=1e-5, atol=3e-6)
  parameter = model.structured_head.token_factor_embedding.weight
  actual_grad = torch.autograd.grad(result.loss, parameter, retain_graph=True)[0]
  oracle_grad = torch.autograd.grad(oracle, parameter, retain_graph=True)[0]
  torch.testing.assert_close(actual_grad, oracle_grad, rtol=2e-4, atol=3e-6)
  # Keep the corrupted context fixed; excluded GOLD labels must not affect loss.
  changed_gold = gold.clone()
  changed_gold[~expected_active] = 9
  changed = model._loss(changed_gold, attention)
  torch.testing.assert_close(changed.loss, result.loss, rtol=0, atol=0)
  assert all(p.grad is None for p in model.backbone.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real DiT needs CUDA')
def test_missing_mask_matches_all_ones_for_wrapped_batch(tmp_path):
  model = _configured_model(tmp_path)
  gold = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device='cuda')
  generator = model._structured_training_corruption_generator
  state = generator.get_state()
  without = model._loss(gold, None)
  for dtype in (torch.bool, torch.int64, torch.float32):
    generator.set_state(state)
    with_ones = model._loss(gold, torch.ones_like(gold, dtype=dtype))
    torch.testing.assert_close(with_ones.loss, without.loss, rtol=0, atol=0)
    assert torch.equal(with_ones.token_mask, without.token_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real DiT needs CUDA')
def test_all_padding_contributes_zero_loss_and_zero_head_gradients(tmp_path, monkeypatch):
  model = _configured_model(tmp_path)
  gold = torch.zeros(2, 4, device='cuda', dtype=torch.long)
  monkeypatch.setattr(model, 'q_xt', lambda *args, **kwargs: torch.full_like(gold, 11))
  result = model._loss(gold, torch.zeros_like(gold))
  assert not result.token_mask.any()
  assert not result.nlls.any()
  torch.testing.assert_close(result.loss, result.loss.new_zeros(()), rtol=0, atol=1e-5)
  result.loss.backward()
  for parameter in model.structured_head.parameters():
    assert parameter.grad is None or (torch.isfinite(parameter.grad).all()
                                     and torch.count_nonzero(parameter.grad) == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real DiT needs CUDA')
def test_q_xt_does_not_itself_filter_padding(tmp_path):
  model = _configured_model(tmp_path)
  tokens = torch.tensor([[1, 2, 0, 0]], device='cuda')
  fully_masked = model.q_xt(tokens, torch.ones(1, 1, device='cuda'))
  assert fully_masked.eq(model.mask_index).all()
  attention = torch.tensor([[1, 1, 0, 0]], device='cuda').bool()
  assert torch.equal(fully_masked.eq(model.mask_index) & attention, attention)
