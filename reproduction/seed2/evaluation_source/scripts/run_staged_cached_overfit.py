#!/usr/bin/env python3
"""Rerun a completed fixed-corruption fit for longer on its verified cache.

The original run did not save optimizer state, so this is a from-scratch
replay and extension, not a resumed optimizer. Shared evaluation points must
reproduce the original scores before their longer curves are interpreted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import torch

from scripts.run_staged_real_overfit import (
  CACHE_KEYS, FrozenCache, file_sha256, make_head, train_head,
)


def verified_cache(directory: Path, expected: dict) -> tuple[FrozenCache, dict]:
  paths = sorted(directory.glob('example-*.pt'))
  if len(paths) != expected['examples']:
    raise ValueError('cached example count differs from original run')
  digest = hashlib.sha256()
  active_tokens = 0
  for path in paths:
    record = torch.load(path, map_location='cpu', weights_only=True)
    if set(record) != set(CACHE_KEYS):
      raise ValueError('cached fields differ from the original schema')
    for key in CACHE_KEYS:
      value = record[key]
      if not isinstance(value, torch.Tensor):
        raise ValueError('cached fields must be tensors')
      digest.update(key.encode())
      digest.update(value.contiguous().numpy().tobytes())
    active_tokens += int(record['active'].sum())
  observed = digest.hexdigest()
  if observed != expected['cache_sha256'] or active_tokens != expected['active_tokens']:
    raise ValueError('cached tensor contents differ from the original run')
  return FrozenCache(paths=paths), {
    'examples': len(paths), 'active_tokens': active_tokens,
    'cache_sha256': observed, 'matches_original': True,
  }


def replay_error(actual: dict, expected: dict, tolerance: float) -> dict:
  errors = {}
  for split in ('train', 'dev'):
    for metric in ('backbone_nll_per_masked_token', 'joint_nll_per_masked_token',
                   'marginal_nll_per_masked_token',
                   'dependence_gain_nats_per_masked_token'):
      error = abs(actual[split][metric] - expected[split][metric])
      errors[f'{split}/{metric}'] = error
  maximum = max(errors.values())
  if not maximum <= tolerance:
    raise AssertionError(f'original prefix failed to replay: {maximum} > {tolerance}')
  return {'step': actual['step'], 'max_absolute_difference_nats_per_token': maximum,
          'tolerance': tolerance, 'passed': True, 'errors': errors}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source-run-dir', type=Path, required=True)
  parser.add_argument('--source-results-sha256', required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--steps', type=int, default=2000)
  parser.add_argument('--eval-every', type=int, default=100)
  parser.add_argument('--replay-tolerance', type=float, default=1e-4)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--threads', type=int, default=4)
  args = parser.parse_args(argv)
  source_path = args.source_run_dir / 'results.json'
  if file_sha256(source_path) != args.source_results_sha256:
    raise ValueError('original results hash mismatch')
  source = json.loads(source_path.read_text())
  if not source.get('completed'):
    raise ValueError('source run must be complete')
  settings = source['arguments']
  if (args.steps <= settings['steps'] or args.eval_every < 1
      or args.threads < 1 or args.replay_tolerance <= 0
      or settings['steps'] % args.eval_every):
    raise ValueError('extend the run and evaluate its original final update')
  args.output_dir.mkdir(parents=True, exist_ok=False)
  torch.set_num_threads(args.threads)
  if args.device.startswith('cuda'):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
  caches, attestations = {}, {}
  for split in ('train', 'dev'):
    caches[split], attestations[split] = verified_cache(
      args.source_run_dir / 'cache' / split, source['cache'][split])
    print(json.dumps({'event': 'cache_verified', 'split': split, **attestations[split]}), flush=True)
  source_arguments = dict(settings)
  settings = {**settings, 'steps': args.steps, 'eval_every': args.eval_every,
              'device': args.device, 'output_dir': str(args.output_dir)}
  protocol = {key: value for key, value in source.items()
              if key not in ('arms', 'completed', 'created_utc', 'arguments', 'git_head', 'source_sha256')}
  protocol.update({
    'protocol': 'verified_fixed_corruption_replay_and_extension_v1',
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'arguments': settings, 'source_arguments': source_arguments,
    'source_run_dir': str(args.source_run_dir.resolve()),
    'source_results_sha256': args.source_results_sha256,
    'source_run_git_head': source['git_head'],
    'optimizer_resumed': False,
    'restart_policy': 'same initialization, optimizer settings, cached tensors and minibatch RNG',
    'cache_attestations': attestations, 'replay_tolerance_nats_per_token': args.replay_tolerance,
    'torch_version': torch.__version__,
    'source_sha256': {path: file_sha256(ROOT / path) for path in (
      'scripts/run_staged_cached_overfit.py', 'scripts/run_staged_real_overfit.py',
      'models/contextual_unary.py', 'models/directional_forest.py',
      'models/structured_decoder.py', 'evaluation/dependence_diagnostics.py',
      'structured_objective.py', 'structured_utils.py', 'structured_training.py')},
    'arms': {}, 'completed': False,
  })
  (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
  for variant in settings['variants']:
    head = make_head(variant, source['hidden_size'], source['vocab_size'],
      **{key: settings[key] for key in ('seed', 'rank', 'directional_rank',
         'unary_rank', 'top_k', 'time_embed_dim', 'init_std', 'component_size_cap')})
    old_rows = {row['step']: row for row in source['arms'][variant]['curve']}
    checks = []
    curve_path = args.output_dir / f'{variant}-curve.jsonl'

    def record(row):
      if row['step'] in old_rows:
        checks.append(replay_error(row, old_rows[row['step']], args.replay_tolerance))
      with curve_path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
      if row['step'] in (source_arguments['steps'], args.steps):
        from safetensors.torch import save_file
        save_file({key: value.detach().cpu().contiguous()
                   for key, value in head.state_dict().items()},
                  str(args.output_dir / f'{variant}-step-{row["step"]:06d}.safetensors'))
      print(json.dumps({'variant': variant, **row}, allow_nan=False), flush=True)

    result = train_head(head, caches['train'], caches['dev'], steps=args.steps,
      batch_size=settings['batch_size'], learning_rate=settings['learning_rate'],
      eval_every=args.eval_every, seed=settings['seed'], device=args.device,
      warmup_steps=settings['warmup_steps'], callback=record)
    if not any(check['step'] == source_arguments['steps'] for check in checks):
      raise AssertionError('original final checkpoint was not replayed')
    checkpoint_path = args.output_dir / f'{variant}-step-{args.steps:06d}.safetensors'
    result.update({
      'prefix_replay_checks': checks, 'original_final_step_replayed': True,
      'total_parameters': sum(p.numel() for p in head.parameters()),
      'active_trainable_parameters': sum(p.numel() for p in head.parameters() if p.requires_grad),
      'inactive_frozen_parameters': sum(p.numel() for p in head.parameters() if not p.requires_grad),
      'rank': head.rank, 'checkpoint_sha256': file_sha256(checkpoint_path),
      'checkpoint_filename': checkpoint_path.name,
    })
    protocol['arms'][variant] = result
    (args.output_dir / 'results.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    del head
  protocol['completed'] = True
  (args.output_dir / 'results.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
  print(json.dumps({'event': 'finished', 'output_dir': str(args.output_dir)}), flush=True)


if __name__ == '__main__':
  main()
