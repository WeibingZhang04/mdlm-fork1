#!/usr/bin/env python3
"""Measure released-backbone serial/batched differences on fixed corruptions.

This report does not impose a tolerance after looking at the outcome. DIT's
production encode/decode path explicitly uses BF16 autocast on CUDA, even
when parameters and returned logits are FP32. We report that behavior and
the resulting numerical differences, including changes to top-K support.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.run_staged_real_overfit import (
  file_sha256, fixed_masks, read_examples, token_digest,
)


@torch.no_grad()
def compare_outputs(serial_hidden: torch.Tensor, serial_logits: torch.Tensor,
                    batched_hidden: torch.Tensor, batched_logits: torch.Tensor,
                    targets: torch.Tensor, active: torch.Tensor,
                    top_k: int) -> dict:
  """Pure numerical comparison, also usable on captured outputs in tests."""
  if not 1 <= top_k <= serial_logits.shape[-1]:
    raise ValueError('invalid top_k')
  if not bool(active.any()):
    raise ValueError('at least one active position is required')
  if serial_hidden.shape != batched_hidden.shape or serial_logits.shape != batched_logits.shape:
    raise ValueError('serial and batched output shapes differ')
  if serial_logits.shape[:2] != active.shape or targets.shape != active.shape:
    raise ValueError('targets and masks must match output positions')
  hidden_delta = (serial_hidden.float() - batched_hidden.float()).abs()
  serial_finite = torch.isfinite(serial_logits)
  batch_finite = torch.isfinite(batched_logits)
  finite = serial_finite & batch_finite
  raw_delta = (serial_logits[finite].double() - batched_logits[finite].double()).abs()

  def target_logp(logits):
    return logits.gather(-1, targets[..., None]).squeeze(-1) - logits.logsumexp(-1)

  target_delta = (target_logp(serial_logits) - target_logp(batched_logits)).abs()
  serial_ids = serial_logits.topk(top_k, dim=-1).indices
  batched_ids = batched_logits.topk(top_k, dim=-1).indices
  serial_sorted = serial_ids.sort(dim=-1).values.contiguous()
  locations = torch.searchsorted(serial_sorted, batched_ids.contiguous())
  matching = serial_sorted.gather(-1, locations.clamp_max(top_k - 1)).eq(batched_ids)
  matching = matching & (locations < top_k)
  overlap = matching.sum(-1)
  argmax_changed = serial_logits.argmax(-1) != batched_logits.argmax(-1)
  rows = []
  for index in range(len(active)):
    row_mask = active[index]
    rows.append({
      'example_index': index,
      'hidden_max_absolute_difference': float(hidden_delta[index].max()),
      'active_target_logprob_max_absolute_difference_nats': float(target_delta[index][row_mask].max()),
      'active_target_logprob_mean_absolute_difference_nats': float(target_delta[index][row_mask].mean()),
      'active_argmax_disagreement_fraction': float(argmax_changed[index][row_mask].float().mean()),
      'active_top_k_set_disagreement_fraction': float((overlap[index][row_mask] != top_k).float().mean()),
      'active_top_k_mean_overlap_fraction': float((overlap[index][row_mask].float() / top_k).mean()),
    })
  return {
    'examples': len(active), 'active_tokens': int(active.sum()), 'top_k': top_k,
    'hidden_max_absolute_difference': float(hidden_delta.max()),
    'hidden_mean_absolute_difference': float(hidden_delta.mean()),
    'finite_logits_max_absolute_difference': float(raw_delta.max()) if raw_delta.numel() else None,
    'finite_logits_mean_absolute_difference': float(raw_delta.mean()) if raw_delta.numel() else None,
    'finite_support_matches_exactly': bool(torch.equal(serial_finite, batch_finite)),
    'negative_infinity_masks_match_exactly': bool(torch.equal(torch.isneginf(serial_logits), torch.isneginf(batched_logits))),
    'serial_nan_count': int(torch.isnan(serial_logits).sum()),
    'batched_nan_count': int(torch.isnan(batched_logits).sum()),
    'active_target_logprob_max_absolute_difference_nats': float(target_delta[active].max()),
    'active_target_logprob_mean_absolute_difference_nats': float(target_delta[active].mean()),
    'active_argmax_disagreement_fraction': float(argmax_changed[active].float().mean()),
    'active_top_k_set_disagreement_fraction': float((overlap[active] != top_k).float().mean()),
    'active_top_k_mean_overlap_fraction': float((overlap[active].float() / top_k).mean()),
    'per_example': rows,
  }


@torch.no_grad()
def audit_model(model, tokens: torch.Tensor, active: torch.Tensor,
                mask_rate: float, device: str, top_k: int) -> dict:
  from noise_schedule import LogLinearNoise
  if not isinstance(model.noise, LogLinearNoise):
    raise ValueError('audit requires the production loglinear noise schedule')
  if not 0 < mask_rate < 1 or not 0 < model.noise.eps < 1:
    raise ValueError('invalid mask rate or schedule epsilon')
  diffusion_time = mask_rate / (1.0 - model.noise.eps)
  if diffusion_time > 1.0:
    raise ValueError('mask rate exceeds the loglinear schedule maximum')
  model.eval()
  clean, mask = tokens.to(device), active.to(device)
  timestep = torch.full((len(tokens),), diffusion_time, dtype=torch.float32, device=device)
  sigma = model.noise(timestep)[0]
  corrupted = torch.where(mask, torch.full_like(clean, model.mask_index), clean)
  batched_hidden, batched_logits = model._structured_backbone_output(
    corrupted, sigma[:, None], force_no_grad=True)
  hidden_dtype, logits_dtype = str(batched_hidden.dtype), str(batched_logits.dtype)
  batched_hidden, batched_logits = batched_hidden.float().cpu(), batched_logits.float().cpu()
  serial_hidden, serial_logits = [], []
  for index in range(len(tokens)):
    hidden, logits = model._structured_backbone_output(
      corrupted[index:index + 1], sigma[index:index + 1, None], force_no_grad=True)
    serial_hidden.append(hidden.float().cpu())
    serial_logits.append(logits.float().cpu())
  report = compare_outputs(
    torch.cat(serial_hidden), torch.cat(serial_logits),
    batched_hidden, batched_logits, tokens, active, top_k)
  report.update({
    'mask_rate': mask_rate, 'diffusion_time': diffusion_time,
    'sigma': float(sigma[0]), 'loglinear_eps': model.noise.eps,
    'returned_hidden_dtype': hidden_dtype, 'returned_logits_dtype': logits_dtype,
    'production_cuda_autocast_dtype': 'bfloat16' if device.startswith('cuda') else 'disabled on CPU',
    'parameter_dtype': str(next(model.backbone.parameters()).dtype),
    'comparison_only_no_posthoc_pass_threshold': True,
    'clean_token_sha256': token_digest(tokens), 'mask_sha256': token_digest(active),
    'corrupted_token_sha256': token_digest(corrupted.cpu()),
  })
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--jsonl', '--train-jsonl', dest='jsonl', type=Path, required=True)
  parser.add_argument('--backbone-checkpoint', type=Path, required=True)
  parser.add_argument('--expectations', type=Path, required=True)
  parser.add_argument('--expected-expectations-sha256', required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--model-config', default='contextual-forest-small')
  parser.add_argument('--data-config', default='train_openwebtext_pinned')
  parser.add_argument('--data-cache-dir', type=Path)
  parser.add_argument('--override', action='append', default=[])
  parser.add_argument('--examples', type=int, default=3)
  parser.add_argument('--length', type=int, default=128)
  parser.add_argument('--mask-rate', type=float, default=0.5)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--top-k', type=int, default=64)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--threads', type=int, default=4)
  args = parser.parse_args(argv)
  if args.examples < 2 or args.length < 2 or args.threads < 1:
    raise ValueError('at least two examples, length two and one CPU thread are required')
  if args.output.exists():
    raise FileExistsError(args.output)
  torch.set_num_threads(args.threads)
  torch.manual_seed(args.seed)
  if args.device.startswith('cuda'):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
  from scripts.export_contextual_forest_adapter import (
    build_production_model, load_production_expectations)
  expectations = load_production_expectations(
    args.expectations, expected_sha256=args.expected_expectations_sha256)
  model = build_production_model(
    model_config=args.model_config, data_config=args.data_config,
    backbone_checkpoint=args.backbone_checkpoint, expectations=expectations,
    overrides=args.override, runtime_mode='ppl_eval', data_cache_dir=args.data_cache_dir,
    checkpoint_save_dir=args.output.parent).to(args.device).eval()
  tokens, source = read_examples(args.jsonl, args.examples, args.length,
                                 model.structured_head.vocab_size, model.mask_index,
                                 model.tokenizer)
  active = fixed_masks(tokens, args.mask_rate, args.seed + 1000)
  report = audit_model(model, tokens, active, args.mask_rate, args.device, args.top_k)
  report.update({
    'protocol': 'released_backbone_serial_batch_numerical_audit_v1',
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'source': source, 'backbone_provenance': model.structured_backbone_provenance,
    'torch_version': torch.__version__, 'device': args.device,
    'gpu_name': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
    'source_sha256': {path: file_sha256(REPO_ROOT / path) for path in (
      'scripts/audit_staged_backbone_batch.py', 'scripts/run_staged_real_overfit.py',
      'models/dit.py', 'diffusion.py')},
  })
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('x') as output:
    output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
  print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
  main()
