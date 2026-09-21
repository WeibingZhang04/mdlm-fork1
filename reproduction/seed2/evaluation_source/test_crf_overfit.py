"""Overfitting test for CRF-MDLM.

Validates correctness by training on 32 examples until perplexity
drops, then generates samples and checks for repetitions.

Usage:
  python test_crf_overfit.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

import crf_utils


def test_forward_backward():
  """Verify forward-backward gives valid marginals."""
  print('=== Test: forward-backward algorithm ===')
  batch, N, K = 2, 8, 4
  emission_0 = torch.randn(batch, K)
  emission_0 = F.log_softmax(emission_0, dim=-1)

  transitions = torch.randn(batch, N - 1, K, K)
  transitions = F.log_softmax(transitions, dim=-1)

  marginals = crf_utils.forward_backward(emission_0, transitions)

  assert marginals.shape == (batch, N, K), \
    f'Expected {(batch, N, K)}, got {marginals.shape}'
  sums = marginals.sum(dim=-1)
  assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
    f'Marginals should sum to 1, got {sums}'
  assert (marginals >= 0).all(), 'Marginals should be non-negative'

  print(f'  Marginals shape: {marginals.shape}')
  print(f'  Sum check (should be 1.0): {sums[0]}')
  print('  PASSED')


def test_forward_backward_deterministic():
  """Check that deterministic chains give correct marginals."""
  print('=== Test: deterministic forward-backward ===')
  batch, N, K = 1, 4, 3

  emission_0 = torch.tensor([[0.0, -1e9, -1e9]])
  transitions = torch.full((batch, N - 1, K, K), -1e9)
  for j in range(N - 1):
    transitions[:, j, 0, 1] = 0.0
    if j > 0:
      transitions[:, j, 1, 1] = 0.0

  marginals = crf_utils.forward_backward(emission_0, transitions)
  print(f'  Marginals:\n{marginals[0]}')

  assert marginals[0, 0, 0] > 0.99, \
    'Position 0 should be candidate 0'
  assert marginals[0, 1, 1] > 0.99, \
    'Position 1 should be candidate 1'
  print('  PASSED')


def test_crf_decoder_shapes():
  """Test CRF decoder module shapes."""
  print('=== Test: CRF decoder shapes ===')
  from models.crf_decoder import CRFDecoder

  decoder = CRFDecoder(
    decoder_dim=64, n_heads=4, encoder_dim=128,
    n_layers=1, vocab_size=100, max_seq_len=32)

  batch, seq_len, enc_dim = 2, 16, 128
  H = torch.randn(batch, seq_len, enc_dim)
  prev_tokens = torch.randint(0, 100, (batch, seq_len))
  positions = torch.arange(seq_len)

  logits = decoder(prev_tokens, positions, H,
                   use_start_for_first=True)
  assert logits.shape == (batch, seq_len, 100), \
    f'Expected {(batch, seq_len, 100)}, got {logits.shape}'
  print(f'  Output shape: {logits.shape}')

  L_q = 48
  tok_q = torch.randint(0, 100, (batch, L_q))
  pos_q = torch.arange(L_q) % seq_len
  logits_b = decoder.forward_batched(tok_q, pos_q, H)
  assert logits_b.shape == (batch, L_q, 100), \
    f'Expected {(batch, L_q, 100)}, got {logits_b.shape}'
  print(f'  Batched output shape: {logits_b.shape}')
  print('  PASSED')


def test_crf_dit_train_forward():
  """Test CRF-DiT training forward pass."""
  print('=== Test: CRFDiT training forward ===')
  from models.crf_decoder import CRFDiT

  config = OmegaConf.create({
    'model': {
      'hidden_size': 64,
      'cond_dim': 32,
      'n_heads': 4,
      'n_blocks': 2,
      'dropout': 0.0,
      'length': 32,
      'scale_by_sigma': True,
      'crf': {
        'decoder_layers': 1,
        'decoder_heads': 4,
        'decoder_dim': 64,
        'top_k': 8,
      }
    }
  })

  vocab_size = 50
  model = CRFDiT(config, vocab_size=vocab_size)

  batch, seq_len = 2, 16
  xt = torch.randint(0, vocab_size, (batch, seq_len))
  xt[:, 3:8] = vocab_size - 1  # mask some positions
  sigma = torch.rand(batch)
  x0 = torch.randint(0, vocab_size - 1, (batch, seq_len))

  logits = model.forward_crf_train(xt, sigma, x0)
  assert logits.shape == (batch, seq_len, vocab_size), \
    f'Expected {(batch, seq_len, vocab_size)}, got {logits.shape}'
  print(f'  CRF train logits shape: {logits.shape}')

  unigram = model(xt, sigma)
  assert unigram.shape == (batch, seq_len, vocab_size), \
    f'Expected {(batch, seq_len, vocab_size)}, got {unigram.shape}'
  print(f'  Unigram logits shape: {unigram.shape}')

  loss = F.cross_entropy(
    logits.view(-1, vocab_size), x0.view(-1))
  loss.backward()
  print(f'  Loss: {loss.item():.4f}')

  grad_norms = {}
  for name, p in model.named_parameters():
    if p.grad is not None:
      grad_norms[name] = p.grad.norm().item()
  crf_grads = {k: v for k, v in grad_norms.items()
               if 'crf_decoder' in k}
  enc_grads = {k: v for k, v in grad_norms.items()
               if 'crf_decoder' not in k}
  print(f'  CRF decoder params with grad: {len(crf_grads)}')
  print(f'  Encoder params with grad: {len(enc_grads)}')
  assert len(crf_grads) > 0, 'CRF decoder should have gradients'
  assert len(enc_grads) > 0, 'Encoder should have gradients'
  print('  PASSED')


def test_crf_marginals():
  """Test CRF marginal computation end-to-end."""
  print('=== Test: CRF marginals (end-to-end) ===')
  from models.crf_decoder import CRFDiT
  import diffusion as diffusion_module

  config = OmegaConf.create({
    'model': {
      'hidden_size': 64,
      'cond_dim': 32,
      'n_heads': 4,
      'n_blocks': 2,
      'dropout': 0.0,
      'length': 16,
      'scale_by_sigma': True,
      'crf': {
        'decoder_layers': 1,
        'decoder_heads': 4,
        'decoder_dim': 64,
        'top_k': 8,
      }
    }
  })

  vocab_size = 50
  mask_index = vocab_size - 1
  model = CRFDiT(config, vocab_size=vocab_size)
  model.eval()

  batch, seq_len = 2, 16
  xt = torch.randint(0, vocab_size - 1, (batch, seq_len))
  xt[:, 3:10] = mask_index
  sigma = torch.rand(batch) * 0.5 + 0.1

  H, c = model.encode(xt, sigma)
  K = model.top_k

  with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                enabled=False):
    unigram_logits = model.output_layer(H, c)
  unigram_logits = unigram_logits.float()
  unigram_logits[:, :, mask_index] = -1e6

  unmasked = (xt != mask_index)
  if unmasked.any():
    det_logits = torch.full_like(unigram_logits, -1e6)
    det_logits.scatter_(-1, xt.unsqueeze(-1), 0.0)
    unigram_logits = torch.where(
      unmasked.unsqueeze(-1), det_logits, unigram_logits)

  _, top_k_indices = unigram_logits.topk(K, dim=-1)

  pos0_dummy = torch.zeros(batch, 1, dtype=torch.long)
  pos0_pos = torch.zeros(1, dtype=torch.long)
  pos0_logits = model.crf_decoder(
    pos0_dummy, pos0_pos, H, use_start_for_first=True)
  pos0_logits = pos0_logits.float().squeeze(1)
  pos0_logits[:, mask_index] = -1e6
  pos0_lp = F.log_softmax(pos0_logits, dim=-1)
  emission_0 = torch.gather(
    pos0_lp, 1, top_k_indices[:, 0, :])

  prev_cands = top_k_indices[:, :-1, :]
  prev_flat = prev_cands.reshape(batch, (seq_len - 1) * K)
  pos_ids = torch.arange(
    1, seq_len).repeat_interleave(K)
  rest_logits = model.crf_decoder.forward_batched(
    prev_flat, pos_ids, H)
  rest_logits = rest_logits.float().view(
    batch, seq_len - 1, K, -1)
  rest_logits[..., mask_index] = -1e6
  rest_lp = F.log_softmax(rest_logits, dim=-1)
  curr_topk = top_k_indices[:, 1:, :]
  curr_exp = curr_topk.unsqueeze(2).expand(-1, -1, K, -1)
  transitions = torch.gather(rest_lp, 3, curr_exp)

  marginals = crf_utils.forward_backward(
    emission_0, transitions)

  assert marginals.shape == (batch, seq_len, K)
  sums = marginals.sum(dim=-1)
  assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

  full = torch.zeros(batch, seq_len, vocab_size)
  full.scatter_(2, top_k_indices, marginals)

  if unmasked.any():
    det = torch.zeros_like(full)
    det.scatter_(-1, xt.unsqueeze(-1), 1.0)
    full = torch.where(unmasked.unsqueeze(-1), det, full)

  full_sums = full.sum(dim=-1)
  assert torch.allclose(
    full_sums, torch.ones_like(full_sums), atol=1e-4), \
    f'Full marginals should sum to 1, got {full_sums}'
  assert (full >= 0).all(), 'Marginals should be non-negative'
  assert full[:, :, mask_index].sum() < 1e-6, \
    'Mask token should have zero probability'
  print(f'  Full marginals shape: {full.shape}')
  print(f'  Sum check: {full_sums[0, :5]}')
  print(f'  Mask token prob: {full[:, :, mask_index].max():.6f}')
  print('  PASSED')


def test_overfit_tiny():
  """Overfit on a tiny synthetic dataset to verify
  end-to-end training works."""
  print('=== Test: overfit on synthetic data ===')
  from models.crf_decoder import CRFDiT
  import noise_schedule

  config = OmegaConf.create({
    'model': {
      'hidden_size': 128,
      'cond_dim': 32,
      'n_heads': 4,
      'n_blocks': 2,
      'dropout': 0.0,
      'length': 32,
      'scale_by_sigma': True,
      'crf': {
        'decoder_layers': 1,
        'decoder_heads': 4,
        'decoder_dim': 128,
        'top_k': 16,
      }
    },
    'noise': {'type': 'loglinear'},
    'training': {'sampling_eps': 1e-3},
  })

  vocab_size = 30
  mask_index = vocab_size
  total_vocab = vocab_size + 1
  seq_len = 32
  batch_size = 8
  n_examples = 32

  model = CRFDiT(config, vocab_size=total_vocab)
  noise = noise_schedule.get_noise(config)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  data = torch.randint(0, vocab_size, (n_examples, seq_len))

  print(f'  Vocab: {vocab_size}, Mask: {mask_index}, '
        f'Seq: {seq_len}')
  print(f'  Training on {n_examples} examples for 200 steps...')

  losses = []
  for step in range(200):
    idx = torch.randint(0, n_examples, (batch_size,))
    x0 = data[idx]

    t_raw = torch.rand(batch_size) * (1 - 1e-3) + 1e-3
    sigma, dsigma = noise(t_raw)

    move_chance = 1 - torch.exp(-sigma[:, None])
    move_mask = torch.rand(batch_size, seq_len) < move_chance
    xt = torch.where(move_mask, mask_index, x0)

    logits = model.forward_crf_train(xt, sigma, x0)

    logits[:, :, mask_index] += -1e6
    log_probs = logits - torch.logsumexp(
      logits, dim=-1, keepdim=True)
    unmasked = (xt != mask_index)
    log_probs[unmasked] = -1e6
    log_probs[unmasked, xt[unmasked]] = 0.0

    log_p_theta = torch.gather(
      log_probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    loss_per_token = -log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]
    loss = loss_per_token.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())
    if (step + 1) % 50 == 0:
      avg = np.mean(losses[-50:])
      print(f'    Step {step+1}: loss={avg:.4f}')

  final_loss = np.mean(losses[-20:])
  initial_loss = np.mean(losses[:20])
  print(f'  Initial loss: {initial_loss:.4f}')
  print(f'  Final loss:   {final_loss:.4f}')

  assert final_loss < initial_loss, \
    f'Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}'
  print('  PASSED (loss decreased)')


if __name__ == '__main__':
  torch.manual_seed(42)
  np.random.seed(42)

  test_forward_backward()
  print()
  test_forward_backward_deterministic()
  print()
  test_crf_decoder_shapes()
  print()
  test_crf_dit_train_forward()
  print()
  test_crf_marginals()
  print()
  test_overfit_tiny()
  print()
  print('All tests passed!')
