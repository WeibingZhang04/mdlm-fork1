#!/usr/bin/env python3
"""Compute the unavoidable NLL from fixed top-K residual conditionals.

An omitted target has probability p(residual)*q_base(target | residual),
which cannot exceed q_base(target | residual).  Summing its negative log
conditional probability gives a lower bound for every candidate-only arm.
This calculation reads saved backbone caches and never fits or changes a head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


@torch.no_grad()
def support_floor(logits: torch.Tensor, targets: torch.Tensor,
                  active: torch.Tensor, top_k: int,
                  mask_index: int | None = None) -> dict:
  """Return exact residual bounds, retaining the original top-K precision.

  Top-K is chosen before conversion to FP64, on the caller's device and full
  batch shape, matching the candidate selection in the training heads.  The
  bound applies to fixed residual conditionals, not the full-vocabulary unary
  control that can change omitted token logits.
  """
  if logits.ndim < 2 or targets.shape != logits.shape[:-1] or active.shape != targets.shape:
    raise ValueError('logits, targets and active mask have incompatible shapes')
  if targets.dtype != torch.long or active.dtype != torch.bool:
    raise ValueError('targets must be long and active mask boolean')
  if not logits.is_floating_point() or not 1 <= top_k <= logits.shape[-1]:
    raise ValueError('floating-point logits and valid top_k are required')
  if bool((torch.isnan(logits) | torch.isposinf(logits)).any()):
    raise ValueError('logits may contain -inf exclusions, but not NaN or +inf')
  if bool(((targets < 0) | (targets >= logits.shape[-1])).any()):
    raise ValueError('target token ID outside vocabulary')
  if mask_index is not None:
    if not 0 <= mask_index < logits.shape[-1]:
      raise ValueError('invalid absorbing mask index')
    if not bool(torch.isneginf(logits[..., mask_index]).all()):
      raise ValueError('absorbing mask must have logit -inf')
    if bool((active & targets.eq(mask_index)).any()):
      raise ValueError('absorbing mask cannot be a clean target')
  values, ids = logits.topk(top_k, dim=-1)
  hit = ids.eq(targets[..., None]).any(-1)
  cutoff = values[..., -1]
  equal_count = logits.eq(cutoff[..., None]).sum(-1)
  selected_equal_count = values.eq(cutoff[..., None]).sum(-1)
  boundary_tie = (equal_count > selected_equal_count) & torch.isfinite(cutoff)
  # Count omitted targets tied at the cutoff as well as selected ones.
  ambiguous_target = boundary_tie & logits.gather(-1, targets[..., None]).squeeze(-1).eq(cutoff)
  base = logits[active].double()
  active_ids, active_targets = ids[active], targets[active]
  selected_target = base.gather(-1, active_targets[:, None]).squeeze(-1)
  if not bool(torch.isfinite(selected_target).all()):
    raise ValueError('an active clean target has zero backbone support')
  omitted = base.clone()
  omitted.scatter_(-1, active_ids, -torch.inf)
  residual_log_mass = torch.logsumexp(omitted, dim=-1)
  residual = ~hit[active]
  floor = torch.where(residual, residual_log_mass - selected_target,
                      torch.zeros_like(selected_target))
  backbone_nll = torch.logsumexp(base, dim=-1) - selected_target
  per_position = logits.new_zeros(targets.shape, dtype=torch.float64)
  per_position[active] = floor
  return {
    'active_tokens': int(active.sum()), 'candidate_hits': int((hit & active).sum()),
    'residual_targets': int(residual.sum()),
    'nll_floor_sum': float(floor.sum()), 'backbone_nll_sum': float(backbone_nll.sum()),
    'per_position_nll_floor': per_position.cpu().tolist(),
    'active_rows_with_boundary_ties': int((boundary_tie & active).sum()),
    'targets_tied_at_boundary': int((ambiguous_target & active).sum()),
    'candidate_ids_sha256': hashlib.sha256(ids.cpu().contiguous().numpy().tobytes()).hexdigest(),
  }


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cache-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--top-k', type=int, default=64)
  parser.add_argument('--device', default='cuda', help='match the training head top-K device')
  parser.add_argument('--batch-size', type=int, default=4, help='match the training/evaluation head batch shape')
  args = parser.parse_args(argv)
  if args.batch_size < 1:
    raise ValueError('batch-size must be positive')
  if args.output.exists():
    raise FileExistsError(args.output)
  paths = sorted(args.cache_dir.glob('example-*.pt'))
  if not paths:
    raise ValueError('cache directory has no example-*.pt records')
  batches, files = [], []
  totals = {key: 0 for key in ('active_tokens', 'candidate_hits', 'residual_targets',
                              'active_rows_with_boundary_ties', 'targets_tied_at_boundary')}
  totals.update(nll_floor_sum=0.0, backbone_nll_sum=0.0)
  for offset in range(0, len(paths), args.batch_size):
    selected = paths[offset:offset + args.batch_size]
    records = [torch.load(path, map_location='cpu', weights_only=True) for path in selected]
    fields = {key: torch.stack([record[key] for record in records]).to(args.device)
              for key in ('logits', 'targets', 'active', 'corrupted')}
    masked_values = fields['corrupted'][fields['active']].unique()
    if masked_values.numel() != 1:
      raise ValueError('cache must contain exactly one absorbing mask token at active positions')
    mask_index = int(masked_values.item())
    result = support_floor(fields['logits'], fields['targets'], fields['active'], args.top_k, mask_index)
    result['cache_files'] = [path.name for path in selected]
    batches.append(result)
    for key in totals:
      totals[key] += result[key]
    files.extend({'name': path.name, 'sha256': sha256_file(path)} for path in selected)
    del fields, records
  count = totals['active_tokens']
  report = {
    'artifact': 'fixed_candidate_residual_nll_floor', 'schema_version': 1,
    'applies_to': ['shared pair head', 'directional pair head', 'candidate-only unary adapter'],
    'definition': 'sum over omitted active targets of -log q_backbone(target | residual), divided by all active tokens',
    'interpretation': 'necessary lower bound; reaching it is not guaranteed by a given head parameterization',
    'top_k': args.top_k, 'topk_device': args.device, 'batch_size': args.batch_size,
    'tail_arithmetic_dtype': 'float64', 'examples': len(paths),
    'candidate_recall': totals['candidate_hits'] / count,
    'nll_floor_per_masked_token': totals['nll_floor_sum'] / count,
    'backbone_nll_per_masked_token': totals['backbone_nll_sum'] / count,
    'maximum_possible_gain_per_masked_token': (totals['backbone_nll_sum'] - totals['nll_floor_sum']) / count,
    **totals, 'batches': batches, 'cache_files': files,
    'script_sha256': sha256_file(Path(__file__)), 'torch_version': torch.__version__,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('x') as handle:
    json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')
  print(json.dumps({key: report[key] for key in (
    'candidate_recall', 'nll_floor_per_masked_token', 'backbone_nll_per_masked_token',
    'maximum_possible_gain_per_masked_token', 'targets_tied_at_boundary')}), flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
