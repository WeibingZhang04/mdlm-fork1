"""Read-only rank-checked checkpoint selection for separate-embedding screens."""

import argparse
import json
from pathlib import Path
import re
import sys

import torch


def validate_checkpoint(checkpoint, topology, rank=8):
  if rank not in (8, 16):
    raise ValueError('Expected separate embedding rank 8 or 16')
  step = checkpoint['global_step']
  if type(step) is not int or not 0 <= step <= 7000:
    raise ValueError('Expected checkpoint step in [0, 7000]')
  config = checkpoint['hyper_parameters']['config']
  head = config['model']['structured_decoder']
  expected = {'rank': rank, 'top_k': 128, 'factor_embedding_mode': 'separate',
              'topology_mode': topology, 'factor_mode': 'dynamic',
              'independent_mode': False}
  for key, value in expected.items():
    if head.get(key) != value:
      raise ValueError(f'Checkpoint has wrong {key}: {head.get(key)!r}')
  if head.get('factor_conditioner_hidden_dim', 0) != 0:
    raise ValueError('Checkpoint must use original affine FiLM')
  training = head['training']
  weight = 0.0 if topology == 'fixed' else 0.1
  if (training['backbone_mode'] != 'frozen'
      or training['topology_weight'] != weight
      or training['head_lr'] != 0.0003
      or config['model']['length'] != 1024 or config['seed'] != 1):
    raise ValueError('Checkpoint training settings do not match this experiment')
  schedule = config['lr_scheduler']
  if (schedule['_target_'] != 'transformers.get_constant_schedule_with_warmup'
      or schedule['num_warmup_steps'] != 50):
    raise ValueError('Checkpoint must use the original constant/warmup schedule')
  if not checkpoint.get('state_dict') or not checkpoint.get('loops'):
    raise ValueError('Missing model or training-loop state')
  if len(checkpoint.get('optimizer_states', [])) != 1:
    raise ValueError('Missing optimizer state')
  schedulers = checkpoint.get('lr_schedulers', [])
  if len(schedulers) != 1 or schedulers[0].get('last_epoch') != step:
    raise ValueError('Missing or inconsistent scheduler step')
  return step


def select_checkpoint(training_dir, topology, rank=8):
  if rank not in (8, 16):
    raise ValueError('Expected separate embedding rank 8 or 16')
  directory = Path(training_dir) / 'checkpoints'
  numbered = []
  for path in directory.glob('*.ckpt'):
    match = re.fullmatch(r'\d+-(\d+)\.ckpt', path.name)
    if match:
      numbered.append((int(match[1]), path))
  candidates = [(None, directory / 'last.ckpt')]
  candidates += sorted(numbered, key=lambda item: item[0], reverse=True)
  candidates.append((None, directory / 'best.ckpt'))
  result = {'checkpoint': None, 'global_step': 0}
  unreadable = []
  for expected_step, path in candidates:
    if not path.exists():
      continue
    if expected_step is not None and result['checkpoint'] and expected_step <= result['global_step']:
      continue
    # best is only a fallback; avoid re-reading a large older checkpoint.
    if path.name == 'best.ckpt' and result['checkpoint']:
      continue
    try:
      checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as error:
      unreadable.append(str(path))
      print(f'Ignoring unreadable checkpoint {path}: {error}', file=sys.stderr)
      continue
    # Configuration mismatches fail closed rather than silently mixing runs.
    step = validate_checkpoint(checkpoint, topology, rank)
    del checkpoint
    if expected_step is not None and step != expected_step:
      raise ValueError(f'Filename/global_step mismatch: {path}')
    if result['checkpoint'] is None or step > result['global_step']:
      result = {'checkpoint': str(path.resolve()), 'global_step': step}
  if unreadable and result['checkpoint'] is None:
    raise ValueError('Checkpoint files exist but none is readable; preserving them for recovery')
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--training-dir', type=Path, required=True)
  parser.add_argument('--topology', choices=('fixed', 'dynamic'), required=True)
  parser.add_argument('--rank', type=int, choices=(8, 16), default=8)
  args = parser.parse_args()
  print(json.dumps(select_checkpoint(args.training_dir, args.topology, args.rank)))
