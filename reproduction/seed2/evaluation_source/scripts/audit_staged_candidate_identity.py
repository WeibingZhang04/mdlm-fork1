#!/usr/bin/env python3
"""Compare the original pair and compact-unary top-K paths on saved caches.

The pair head calls topk on [B,L,V] after promoting half-precision logits.
The unary loss first selects active flattened rows in chunks of 32, converts
those rows to its working dtype, and calls topk on each [chunk,V] tensor.
This script reproduces both operations, including their shapes and ordering.
It does not impose a tie-breaking rule, fit a model, or change any logits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch


COUNT_FIELDS = (
  'active_tokens', 'ordered_id_differing_rows', 'ordered_id_differing_slots',
  'candidate_set_differing_rows', 'candidate_symmetric_difference_tokens',
  'gold_membership_differing_rows', 'pair_gold_hits', 'unary_gold_hits',
  'cutoff_differing_rows', 'active_rows_with_boundary_ties',
  'gold_targets_tied_at_boundary', 'set_differences_not_at_cutoff_rows',
)
SUM_FIELDS = ('pair_only_base_probability_mass_sum', 'unary_only_base_probability_mass_sum')
MAX_FIELDS = ('retained_base_probability_mass_max_absolute_difference',
              'cutoff_max_absolute_difference')


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
  return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _membership(haystack: torch.Tensor, needles: torch.Tensor) -> torch.Tensor:
  sorted_ids = haystack.sort(dim=-1).values.contiguous()
  locations = torch.searchsorted(sorted_ids, needles.contiguous())
  return (locations < sorted_ids.shape[-1]) & sorted_ids.gather(
    -1, locations.clamp_max(sorted_ids.shape[-1] - 1)).eq(needles)


def original_pair_candidates(logits: torch.Tensor, top_k: int):
  """Exactly the candidate-selection statements in the original forest head."""
  work_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
  values, ids = work_logits.topk(top_k, dim=-1)
  return values, ids


def original_unary_candidates(logits: torch.Tensor, active: torch.Tensor,
                              top_k: int, position_chunk_size: int = 32):
  """Exactly the row indexing, cast and top-K order in target_log_probs."""
  selected = active.flatten().nonzero().flatten()
  base_flat = logits.reshape(-1, logits.shape[-1])
  work_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
  values, ids, shapes = [], [], []
  for offset in range(0, selected.numel(), position_chunk_size):
    indices = selected[offset:offset + position_chunk_size]
    base = base_flat[indices].to(work_dtype)
    candidate_logits, candidate_ids = base.topk(top_k, dim=-1)
    values.append(candidate_logits)
    ids.append(candidate_ids)
    shapes.append(list(base.shape))
  return torch.cat(values), torch.cat(ids), selected, shapes


@torch.no_grad()
def compare_candidate_ids(logits: torch.Tensor, targets: torch.Tensor,
                          active: torch.Tensor, pair_values: torch.Tensor,
                          pair_ids: torch.Tensor, unary_values: torch.Tensor,
                          unary_ids: torch.Tensor, *, max_detail_rows: int = 20) -> dict:
  """Compare active rows and quantify the base probability mass of swaps."""
  top_k = pair_ids.shape[-1]
  pair = pair_ids[active]
  values = pair_values[active]
  clean = targets[active]
  if pair.shape != unary_ids.shape or values.shape != unary_values.shape:
    raise ValueError('candidate tensors disagree on active rows or top-K')
  pair_only = ~_membership(unary_ids, pair)
  unary_only = ~_membership(pair, unary_ids)
  ordered_difference = pair != unary_ids
  set_difference = pair_only.any(-1) | unary_only.any(-1)
  pair_gold = pair.eq(clean[:, None]).any(-1)
  unary_gold = unary_ids.eq(clean[:, None]).any(-1)
  cutoff = values[:, -1]
  unary_cutoff = unary_values[:, -1]
  work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
  full_cutoff = pair_values[..., -1]
  at_boundary = (work == full_cutoff[..., None]).sum(-1)
  selected_at_boundary = (pair_values == full_cutoff[..., None]).sum(-1)
  boundary_tie = ((at_boundary > selected_at_boundary) & torch.isfinite(full_cutoff))[active]
  gold_logits = work.gather(-1, targets[..., None]).squeeze(-1)[active]
  normalizer = torch.logsumexp(work.double(), dim=-1)[active]
  pair_probability = (values.double() - normalizer[:, None]).exp()
  unary_probability = (unary_values.double() - normalizer[:, None]).exp()
  pair_removed_mass = (pair_probability * pair_only).sum(-1)
  unary_added_mass = (unary_probability * unary_only).sum(-1)
  swaps_at_cutoff = (
    ((values == cutoff[:, None]) | ~pair_only).all(-1)
    & ((unary_values == unary_cutoff[:, None]) | ~unary_only).all(-1))
  positions = active.nonzero()
  details = []
  for row in (set_difference | (pair_gold != unary_gold)).nonzero().flatten()[:max_detail_rows]:
    index = int(row)
    details.append({
      'batch_row': int(positions[index, 0]), 'position': int(positions[index, 1]),
      'pair_only_ids': pair[index][pair_only[index]].cpu().tolist(),
      'unary_only_ids': unary_ids[index][unary_only[index]].cpu().tolist(),
      'pair_only_logits': values[index][pair_only[index]].cpu().tolist(),
      'unary_only_logits': unary_values[index][unary_only[index]].cpu().tolist(),
      'pair_cutoff': float(cutoff[index]), 'unary_cutoff': float(unary_cutoff[index]),
      'pair_only_base_probability_mass': float(pair_removed_mass[index]),
      'unary_only_base_probability_mass': float(unary_added_mass[index]),
      'gold_in_pair_candidates': bool(pair_gold[index]),
      'gold_in_unary_candidates': bool(unary_gold[index]),
    })
  return {
    'active_tokens': int(active.sum()),
    'ordered_id_differing_rows': int(ordered_difference.any(-1).sum()),
    'ordered_id_differing_slots': int(ordered_difference.sum()),
    'candidate_set_differing_rows': int(set_difference.sum()),
    'candidate_symmetric_difference_tokens': int(pair_only.sum() + unary_only.sum()),
    'gold_membership_differing_rows': int((pair_gold != unary_gold).sum()),
    'pair_gold_hits': int(pair_gold.sum()), 'unary_gold_hits': int(unary_gold.sum()),
    'cutoff_differing_rows': int((cutoff != unary_cutoff).sum()),
    'cutoff_max_absolute_difference': float((cutoff - unary_cutoff).abs().max()),
    'active_rows_with_boundary_ties': int(boundary_tie.sum()),
    'gold_targets_tied_at_boundary': int((boundary_tie & gold_logits.eq(cutoff)).sum()),
    'set_differences_not_at_cutoff_rows': int((set_difference & ~swaps_at_cutoff).sum()),
    'pair_only_base_probability_mass_sum': float(pair_removed_mass.sum()),
    'unary_only_base_probability_mass_sum': float(unary_added_mass.sum()),
    'retained_base_probability_mass_max_absolute_difference': float(
      (pair_probability.sum(-1) - unary_probability.sum(-1)).abs().max()),
    'pair_active_candidate_ids_sha256': tensor_sha256(pair),
    'unary_active_candidate_ids_sha256': tensor_sha256(unary_ids),
    'differing_rows_first': details,
    'detail_rows_truncated': int(set_difference.sum()) > len(details),
  }


@torch.no_grad()
def audit_batch(logits: torch.Tensor, targets: torch.Tensor, active: torch.Tensor,
                *, top_k: int = 64, position_chunk_size: int = 32,
                max_detail_rows: int = 20) -> dict:
  if logits.ndim != 3 or targets.shape != logits.shape[:2] or active.shape != targets.shape:
    raise ValueError('expected logits[B,L,V] and targets/active[B,L]')
  if active.dtype != torch.bool or targets.dtype != torch.long:
    raise ValueError('active must be bool and targets long')
  if not bool(active.any()) or position_chunk_size < 1 or not 1 <= top_k <= logits.shape[-1]:
    raise ValueError('invalid active mask, chunk size, or top-K')
  if bool(torch.isnan(logits).any() | torch.isposinf(logits).any()):
    raise ValueError('logits contain NaN or positive infinity')
  pair_values, pair_ids = original_pair_candidates(logits, top_k)
  unary_values, unary_ids, selected, shapes = original_unary_candidates(
    logits, active, top_k, position_chunk_size)
  report = compare_candidate_ids(
    logits, targets, active, pair_values, pair_ids, unary_values, unary_ids,
    max_detail_rows=max_detail_rows)
  report.update({'pair_input_shape': list(logits.shape), 'unary_chunk_input_shapes': shapes,
                 'active_flat_indices_sha256': tensor_sha256(selected),
                 'cached_logits_dtype': str(logits.dtype)})
  return report


def audit_split(cache_dir: Path, *, device: str, batch_size: int,
                top_k: int, position_chunk_size: int, max_detail_rows: int) -> dict:
  paths = sorted(cache_dir.glob('example-*.pt'))
  if not paths:
    raise ValueError(f'no cached examples in {cache_dir}')
  batches, files = [], []
  for offset in range(0, len(paths), batch_size):
    selected = paths[offset:offset + batch_size]
    records = [torch.load(path, map_location='cpu', weights_only=True) for path in selected]
    fields = {key: torch.stack([record[key] for record in records]).to(device)
              for key in ('logits', 'targets', 'active')}
    report = audit_batch(**fields, top_k=top_k, position_chunk_size=position_chunk_size,
                          max_detail_rows=max_detail_rows)
    report['cache_files'] = [path.name for path in selected]
    batches.append(report)
    files.extend({'name': path.name, 'sha256': sha256_file(path)} for path in selected)
  summary = {key: sum(batch[key] for batch in batches) for key in (*COUNT_FIELDS, *SUM_FIELDS)}
  summary.update({key: max(batch[key] for batch in batches) for key in MAX_FIELDS})
  summary.update({
    'all_active_ordered_ids_match': summary['ordered_id_differing_rows'] == 0,
    'all_active_candidate_sets_match': summary['candidate_set_differing_rows'] == 0,
    'all_active_gold_membership_matches': summary['gold_membership_differing_rows'] == 0,
    'pair_candidate_recall': summary['pair_gold_hits'] / summary['active_tokens'],
    'unary_candidate_recall': summary['unary_gold_hits'] / summary['active_tokens'],
    'examples': len(paths), 'batches': batches, 'cache_files': files,
  })
  return summary


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--top-k', type=int, default=64)
  parser.add_argument('--position-chunk-size', type=int, default=32)
  parser.add_argument('--max-detail-rows', type=int, default=20)
  args = parser.parse_args(argv)
  if min(args.batch_size, args.position_chunk_size, args.top_k) < 1 or args.max_detail_rows < 0:
    raise ValueError('invalid audit dimensions')
  if args.output.exists():
    raise FileExistsError(args.output)
  run = json.loads((args.run_dir / 'results.json').read_text())
  if args.batch_size != run['arguments']['batch_size'] or args.top_k != run['arguments']['top_k']:
    raise ValueError('audit batch size and top-K must match the original run')
  if args.position_chunk_size != 32:
    raise ValueError('the original unary runner used the default position_chunk_size=32')
  report = {
    'artifact': 'original_pair_unary_candidate_identity_audit', 'schema_version': 1,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'original_run_git_head': run['git_head'],
    'original_run_source_sha256': run['source_sha256'],
    'original_results_sha256': sha256_file(args.run_dir / 'results.json'),
    'script_sha256': sha256_file(Path(__file__)),
    'audit_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
    'torch_version': torch.__version__, 'device': args.device,
    'batch_size': args.batch_size, 'top_k': args.top_k,
    'position_chunk_size': args.position_chunk_size,
    'candidate_policy_changed': False, 'model_training_performed': False,
    'cache_order': 'sorted example files, matching the original evaluation order',
    'splits': {},
  }
  for split in ('train', 'dev'):
    summary = audit_split(
      args.run_dir / 'cache' / split, device=args.device, batch_size=args.batch_size,
      top_k=args.top_k, position_chunk_size=args.position_chunk_size,
      max_detail_rows=args.max_detail_rows)
    recorded_recall = run['arms']['shared']['curve'][-1][split]['candidate_recall']
    summary['pair_recall_matches_original_run'] = summary['pair_candidate_recall'] == recorded_recall
    report['splits'][split] = summary
    print(json.dumps({'split': split, **{key: value for key, value in summary.items()
                                       if key not in ('batches', 'cache_files')}}), flush=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('x') as output:
    output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
  main()
