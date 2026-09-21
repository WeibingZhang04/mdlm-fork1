"""Exercise CRF marginals and reveal modes from a recovery checkpoint."""

import argparse
import json
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf
import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from diffusion import Diffusion


class DummyModelTokenizer:
  vocab_size = 256
  mask_token = None
  mask_token_id = None
  pad_token_id = 0


class DummyEvalTokenizer:
  pad_token = '<pad>'
  pad_token_id = 0
  eos_token = '<eos>'
  eos_token_id = 0


def make_config(model_config, reveal_mode):
  return OmegaConf.create({
    'diffusion': 'absorbing_state',
    'backbone': 'crf_dit',
    'parameterization': 'subs',
    'time_conditioning': False,
    'T': 0,
    'subs_masking': False,
    'model': model_config,
    'noise': {'type': 'loglinear'},
    'sampling': {
      'predictor': 'ddpm',
      'steps': model_config['length'],
      'noise_removal': True,
      'crf_reveal': reveal_mode,
    },
    'loader': {'eval_batch_size': 2},
    'training': {
      'ema': 0.0,
      'antithetic_sampling': True,
      'importance_sampling': False,
      'sampling_eps': 1e-3,
      'change_of_variables': False,
    },
    'eval': {'gen_ppl_eval_model_name_or_path': 'unused'},
    'optim': {'lr': 1e-3},
  })


def sample_metrics(model, training_data, mode):
  model.crf_reveal = mode
  start = time.perf_counter()
  samples = model.restore_model_and_sample(
    num_steps=model.config.model.length)
  runtime = time.perf_counter() - start
  distances = (samples[:, None, :] != training_data[None, :, :]).sum(dim=-1)
  return {
    'runtime_seconds': runtime,
    'mask_tokens_remaining': int((samples == model.mask_index).sum()),
    'unique_token_count': int(samples.unique().numel()),
    'nearest_training_hamming': distances.min(dim=1).values.tolist(),
    'samples': samples.tolist(),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError('This inference smoke requires CUDA')
  device = torch.device('cuda')
  # This checkpoint is produced by our paired smoke-training script and includes
  # trusted metadata (TorchVersion/config), so opt out of PyTorch's tensor-only
  # loader default introduced in recent releases.
  checkpoint = torch.load(
    args.checkpoint, map_location='cpu', weights_only=False)
  config = make_config(checkpoint['config']['model'], 'sequential')

  original_loader = transformers.AutoTokenizer.from_pretrained
  transformers.AutoTokenizer.from_pretrained = (
    lambda *unused_args, **unused_kwargs: DummyEvalTokenizer())
  try:
    model = Diffusion(config, tokenizer=DummyModelTokenizer())
  finally:
    transformers.AutoTokenizer.from_pretrained = original_loader
  model.backbone.load_state_dict(checkpoint['model'])
  model = model.to(device).eval()

  x0 = checkpoint['data'][:2].to(device)
  fixed_mask = checkpoint['fixed_mask'][:2].to(device)
  xt = torch.where(fixed_mask, model.mask_index, x0)
  sigma = torch.full((x0.shape[0],), 0.7, device=device)

  torch.cuda.synchronize()
  marginal_start = time.perf_counter()
  marginals = model._compute_crf_marginals(xt, sigma)
  torch.cuda.synchronize()
  marginal_runtime = time.perf_counter() - marginal_start
  probability_sum_error = float(
    (marginals.sum(dim=-1) - 1).abs().max())
  observed_probability_error = float(
    (marginals.gather(-1, x0.unsqueeze(-1)).squeeze(-1)[~fixed_mask]
     - 1).abs().max())
  masked_target_supported = (
    marginals.gather(-1, x0.unsqueeze(-1)).squeeze(-1)[fixed_mask] > 0)

  all_masked = torch.full_like(x0, model.mask_index)
  all_masked_marginals = model._compute_crf_marginals(all_masked, sigma)
  model.crf_reveal = 'sequential'
  sequential_once = model._crf_gated_update(
    all_masked, all_masked_marginals,
    move_chance_t=torch.ones(x0.shape[0], 1, 1, device=device),
    move_chance_s=torch.full(
      (x0.shape[0], 1, 1), 0.5, device=device))
  model.crf_reveal = 'gated'
  gated_once = model._crf_gated_update(
    all_masked, all_masked_marginals,
    move_chance_t=torch.ones(x0.shape[0], 1, 1, device=device),
    move_chance_s=torch.full(
      (x0.shape[0], 1, 1), 0.5, device=device))

  training_data = checkpoint['data'].to(device)
  result = {
    'device': torch.cuda.get_device_name(0),
    'marginals': {
      'shape': list(marginals.shape),
      'runtime_seconds': marginal_runtime,
      'max_probability_sum_error': probability_sum_error,
      'max_observed_probability_error': observed_probability_error,
      'masked_target_support_recall': float(
        masked_target_supported.float().mean()),
    },
    'single_update_revealed_per_row': {
      'sequential': (
        (sequential_once != model.mask_index).sum(dim=-1).tolist()),
      'gated_at_half_schedule': (
        (gated_once != model.mask_index).sum(dim=-1).tolist()),
    },
    'sequential_sample': sample_metrics(
      model, training_data, 'sequential'),
    'gated_sample': sample_metrics(model, training_data, 'gated'),
  }

  if probability_sum_error > 1e-5 or observed_probability_error > 1e-6:
    raise AssertionError('CRF marginal normalization or evidence clamp failed')
  if result['single_update_revealed_per_row']['sequential'] != [1, 1]:
    raise AssertionError('sequential mode must reveal one site per row')
  if (result['sequential_sample']['mask_tokens_remaining'] != 0
      or result['gated_sample']['mask_tokens_remaining'] != 0):
    raise AssertionError('sampling left mask tokens unresolved')

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
