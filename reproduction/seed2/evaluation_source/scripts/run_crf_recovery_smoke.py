"""GPU smoke/overfit gate for the recovered CRF-MDLM training path."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import crf_utils
import noise_schedule
from models.crf_decoder import CRFDiT


def build_model(vocab_size, sequence_length):
  config = OmegaConf.create({
    'model': {
      'hidden_size': 128,
      'cond_dim': 32,
      'n_heads': 4,
      'n_blocks': 2,
      'dropout': 0.0,
      'length': sequence_length,
      'scale_by_sigma': True,
      'crf': {
        'decoder_layers': 1,
        'decoder_heads': 4,
        'decoder_dim': 128,
        'top_k': 16,
        'inference_query_chunk_size': 64,
      },
    },
    'noise': {'type': 'loglinear'},
    'training': {'sampling_eps': 1e-3},
  })
  return CRFDiT(config, vocab_size=vocab_size), config


def losses_and_recall(model, x0, xt, sigma, diffusion_weight,
                      mask_index, aux_weight):
  crf_logits, unigram_logits = model.forward_crf_train(
    xt, sigma, x0, return_unigram_logits=True)

  valid_crf_logits = crf_logits.float().clone()
  valid_crf_logits[..., mask_index] = -1e6
  target_crf_nll = -F.log_softmax(
    valid_crf_logits, dim=-1).gather(
      -1, x0.unsqueeze(-1)).squeeze(-1)
  masked = (xt == mask_index).to(target_crf_nll.dtype)
  primary = target_crf_nll * masked * diffusion_weight[:, None]

  auxiliary = crf_utils.unigram_denoising_loss(
    logits=unigram_logits,
    xt=xt,
    x0=x0,
    mask_index=mask_index,
    token_weight=diffusion_weight)
  objective = (primary + aux_weight * auxiliary).mean()

  valid_unigram_logits = unigram_logits.float().clone()
  valid_unigram_logits[..., mask_index] = -1e6
  top_k = model.top_k
  candidates = valid_unigram_logits.topk(top_k, dim=-1).indices
  hits = (candidates == x0.unsqueeze(-1)).any(dim=-1) & (xt == mask_index)
  masked_count = (xt == mask_index).sum().clamp_min(1)
  recall = hits.sum().float() / masked_count
  return objective, primary.mean(), auxiliary.mean(), recall


@torch.no_grad()
def evaluate(model, data, fixed_mask, mask_index):
  model.eval()
  x0 = data
  xt = torch.where(fixed_mask, mask_index, x0)
  sigma = torch.full(
    (x0.shape[0],), 0.7, device=x0.device, dtype=torch.float32)
  diffusion_weight = torch.ones_like(sigma)
  objective, primary, auxiliary, recall = losses_and_recall(
    model, x0, xt, sigma, diffusion_weight, mask_index, 0.1)
  return {
    'objective': float(objective),
    'crf_nll': float(primary),
    'unigram_nll': float(auxiliary),
    'top_k_recall': float(recall),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=300)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError('This smoke gate requires CUDA')
  torch.manual_seed(args.seed)
  device = torch.device('cuda')

  token_vocab_size = 256
  mask_index = token_vocab_size
  total_vocab_size = token_vocab_size + 1
  sequence_length = 32
  example_count = 32
  batch_size = 8

  model, config = build_model(total_vocab_size, sequence_length)
  model = model.to(device)
  noise = noise_schedule.get_noise(config).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

  data = torch.randint(
    0, token_vocab_size,
    (example_count, sequence_length), device=device)
  fixed_generator = torch.Generator(device=device).manual_seed(
    args.seed + 1)
  fixed_mask = torch.rand(
    data.shape, generator=fixed_generator, device=device) < 0.5

  initial = evaluate(model, data, fixed_mask, mask_index)
  history = []
  last_head_grad = 0.0
  last_decoder_grad = 0.0

  model.train()
  for step in range(args.steps):
    indices = torch.randint(0, example_count, (batch_size,), device=device)
    x0 = data[indices]
    t = torch.rand(batch_size, device=device) * (1 - 1e-3) + 1e-3
    sigma, dsigma = noise(t)
    move_chance = 1 - torch.exp(-sigma[:, None])
    xt = torch.where(
      torch.rand_like(x0, dtype=torch.float32) < move_chance,
      mask_index, x0)
    diffusion_weight = dsigma / torch.expm1(sigma)
    aux_weight = crf_utils.linear_warmup_weight(
      step, 0.0, 0.1, max(args.steps // 2, 1))

    objective, primary, auxiliary, recall = losses_and_recall(
      model, x0, xt, sigma, diffusion_weight,
      mask_index, aux_weight)
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    last_head_grad = float(model.output_layer.linear.weight.grad.norm())
    last_decoder_grad = float(
      model.crf_decoder.output_proj.weight.grad.norm())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    history.append({
      'step': step + 1,
      'objective': float(objective.detach()),
      'crf_nll': float(primary.detach()),
      'unigram_nll': float(auxiliary.detach()),
      'top_k_recall': float(recall.detach()),
      'aux_weight': aux_weight,
    })

  final = evaluate(model, data, fixed_mask, mask_index)
  result = {
    'seed': args.seed,
    'steps': args.steps,
    'device': torch.cuda.get_device_name(0),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'initial': initial,
    'final': final,
    'last_head_grad_norm': last_head_grad,
    'last_decoder_grad_norm': last_decoder_grad,
    'last_20_mean_objective': sum(
      item['objective'] for item in history[-20:]) / min(20, len(history)),
    'history_every_25_steps': history[24::25],
  }

  if final['objective'] >= initial['objective']:
    raise AssertionError(
      f"objective did not decrease: {initial['objective']} -> "
      f"{final['objective']}")
  if last_head_grad <= 0 or last_decoder_grad <= 0:
    raise AssertionError('both CRF and unigram heads must receive gradients')

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  if args.checkpoint is not None:
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
      'model': model.state_dict(),
      'config': OmegaConf.to_container(config, resolve=True),
      'token_vocab_size': token_vocab_size,
      'mask_index': mask_index,
      'data': data,
      'fixed_mask': fixed_mask,
      'result': result,
    }, args.checkpoint)
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
