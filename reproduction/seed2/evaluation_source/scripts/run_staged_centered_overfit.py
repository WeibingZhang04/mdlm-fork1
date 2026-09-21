#!/usr/bin/env python3
"""Fit only centered coupling on the OLD fixed-corruption debugging cache.

No backbone calls, fresh data, checkpoint selection, or reserved-test access.
The seed-1 step-1000 unary checkpoint and old cache contents are hard-pinned.
Optional development scores are repeated old-debug diagnostics, not a new
held-out evaluation. Verified cache tensors stay in CPU memory throughout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import torch

from evaluation.fresh_pair_statistics import canonical_sha256
from models.centered_forest import FrozenUnaryCenteredForestHead, centered_forest_log_probability
from models.contextual_unary import ContextualUnaryAdapter
from scripts.evaluate_staged_fresh_selected import score_head_float64
from scripts.run_staged_real_overfit import CACHE_KEYS, FrozenCache, file_sha256
from structured_objective import infer_structured_distribution, structured_token_log_probability

UNARY_CHECKPOINT_SHA256 = '2cdd03bcd6985a6c7c89b7bffcff391c6b43c3eff4284df67fefd5236bfe8701'
OLD_CACHE_SHA256 = {
  'train': 'f4d31bf31a5b34ffc29881abfe86e7937ce81a9f409f992df09b45ebd1b7780b',
  'dev': 'cb139453a5581161a0441793bbcf4f56029cc5f957db81055f40be35dcec87fc',
}
SOURCE_FILES = (
  'scripts/run_staged_centered_overfit.py', 'models/centered_forest.py',
  'evaluation/centered_coupling.py', 'models/contextual_unary.py',
  'models/structured_decoder.py', 'scripts/run_staged_real_overfit.py',
  'scripts/evaluate_staged_fresh_selected.py', 'structured_objective.py',
  'structured_utils.py',
)


def authenticated_bytes(path, expected):
  payload = Path(path).read_bytes()
  if hashlib.sha256(payload).hexdigest() != expected:
    raise ValueError(f'{Path(path).name}: input SHA256 mismatch')
  return payload


def tensor_state_sha256(state):
  digest = hashlib.sha256()
  for key, value in sorted(state.items()):
    value = value.detach().cpu().contiguous()
    digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
    digest.update(value.numpy().tobytes())
  return digest.hexdigest()


def load_unary(path):
  payload = authenticated_bytes(path, UNARY_CHECKPOINT_SHA256)
  with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
  if (state.get('schema_version') != 1 or state.get('completed_steps') != 1000
      or state.get('config', {}).get('seed') != 1
      or set(state.get('heads', {})) != {'unary', 'shared', 'directional'}):
    raise ValueError('expected completed seed-1 step-1000 fresh checkpoint')
  identity = state['identity']
  config = identity['head_config']
  if config['seed'] != 1 or state['config']['backbone_batch_size'] != 1:
    raise ValueError('checkpoint seed or serial-backbone identity differs')
  for name in ('models/contextual_unary.py', 'models/structured_decoder.py'):
    if identity['source_sha256'].get(name) != file_sha256(ROOT / name):
      raise ValueError(f'frozen unary implementation source changed: {name}')
  unary = ContextualUnaryAdapter(
    config['hidden_size'], config['vocab_size'], rank=config['unary_rank'],
    time_embed_dim=config['time_embed_dim'], top_k=config['top_k'], correction_domain='candidates')
  expected = unary.state_dict()
  loaded = state['heads']['unary']
  if set(loaded) != set(expected):
    raise ValueError('unary state keys differ')
  for name, value in loaded.items():
    if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
        or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())):
      raise ValueError(f'invalid unary tensor: {name}')
  unary.load_state_dict(loaded, strict=True)
  unary.requires_grad_(False).eval()
  return unary, {'checkpoint_sha256': UNARY_CHECKPOINT_SHA256, 'checkpoint_step': 1000,
                 'training_seed': 1, 'identity': identity,
                 'identity_sha256': canonical_sha256(identity),
                 'unary_state_sha256': tensor_state_sha256(unary.state_dict())}


def load_old_cache(directory, expected, split, *, hidden_size, vocab_size, length):
  """Authenticate exact old tensor bytes plus validate serialization metadata."""
  if expected['cache_sha256'] != OLD_CACHE_SHA256[split]:
    raise ValueError('only the pinned OLD debugging cache is allowed')
  paths = sorted(directory.glob('example-*.pt'))
  if len(paths) != expected['examples'] or not paths:
    raise ValueError('old cache example count differs')
  digest, records, files = hashlib.sha256(), [], []
  specifications = {
    'hidden': ((length, hidden_size), torch.float32),
    'logits': ((length, vocab_size), torch.float32),
    'targets': ((length,), torch.long), 'active': ((length,), torch.bool),
    'sigma': ((), torch.float32), 'corrupted': ((length,), torch.long),
  }
  for path in paths:
    payload = path.read_bytes()
    row = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if set(row) != set(CACHE_KEYS):
      raise ValueError('old cache tensor fields differ')
    for key in CACHE_KEYS:
      value = row[key]
      shape, dtype = specifications[key]
      if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
        raise ValueError(f'old cache shape/dtype differs: {key}')
      digest.update(key.encode())
      digest.update(value.contiguous().numpy().tobytes())
    if (not bool(torch.isfinite(row['hidden']).all()) or not bool(torch.isfinite(row['sigma']))
        or bool((torch.isnan(row['logits']) | torch.isposinf(row['logits'])).any())
        or not bool(torch.isfinite(row['logits']).any(-1).all())
        or not bool(row['active'].any())
        or bool(((row['targets'] < 0) | (row['targets'] >= vocab_size)).any())
        or not torch.equal(row['corrupted'][~row['active']], row['targets'][~row['active']])):
      raise ValueError('invalid old cached context, tokens, or support')
    records.append(row)
    files.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(payload).hexdigest()})
  active_tokens = sum(int(row['active'].sum()) for row in records)
  if digest.hexdigest() != OLD_CACHE_SHA256[split] or active_tokens != expected['active_tokens']:
    raise ValueError('old cached tensor contents differ from pinned input')
  return FrozenCache(records=records), {
    'cache_sha256': digest.hexdigest(), 'examples': len(records), 'active_tokens': active_tokens,
    'files': files, 'storage': 'authenticated immutable CPU tensor snapshot',
    'tensor_state_sha256': tensor_state_sha256(
      {f'{index}/{key}': value for index, row in enumerate(records) for key, value in row.items()}),
  }


@torch.no_grad()
def evaluate_debug(head, cache, batch_size, device):
  head.eval()
  totals = {'joint': 0., 'unary': 0., 'active': 0}
  marginal_error, score_error = 0., 0.
  for start in range(0, len(cache), batch_size):
    batch = cache.batch(range(start, min(start + batch_size, len(cache))), device)
    output = head(batch['hidden'], batch['logits'], batch['sigma'], batch['active'])
    joint = centered_forest_log_probability(output, batch['logits'], batch['targets'])
    with torch.autocast(device_type=batch['hidden'].device.type, enabled=False):
      reference = head._frozen_unary.candidate_lattice(
        batch['hidden'], batch['logits'], batch['sigma'], batch['active'])
      expected = reference.unary_log_potentials.double().log_softmax(-1).exp()[batch['active']]
      expected_tail = head._frozen_unary._tail_log_mass(batch['logits'].double(), reference.candidate_ids, 4096)
      unary = score_head_float64(head._frozen_unary, batch)['joint_log_probability']
    if not torch.equal(output.candidate_ids, reference.candidate_ids):
      raise AssertionError('centered head changed the frozen-unary candidate support')
    # Equality handles matching -inf empty tails without forming inf-inf.
    if not torch.equal(output.base_tail_log_mass, expected_tail):
      raise AssertionError('centered head changed the frozen-unary residual conditional')
    inference = infer_structured_distribution(output, batch['active'], backend='low_rank')
    actual = inference.marginals.node_marginals[batch['active']]
    marginal_errors = (actual - expected).abs()
    if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()
                & torch.isfinite(marginal_errors).all()):
      raise FloatingPointError('nonfinite frozen-unary marginal audit')
    marginal_error = max(marginal_error, float(marginal_errors.max()))
    inferred = structured_token_log_probability(
      output, batch['logits'], batch['targets'], batch['active'], inference)
    score_errors = (joint - inferred).abs()
    if not bool(torch.isfinite(joint).all() & torch.isfinite(unary).all() & torch.isfinite(inferred).all()):
      raise FloatingPointError('nonfinite debugging evaluation score')
    if not bool(torch.isfinite(score_errors).all()):
      raise FloatingPointError('nonfinite partition-free score audit')
    score_error = max(score_error, float(score_errors.max()))
    totals['joint'] += float(joint.sum())
    totals['unary'] += float(unary.sum())
    totals['active'] += int(batch['active'].sum())
  if marginal_error > 1e-10 or score_error > 1e-8:
    raise AssertionError('fixed-marginal or partition-free score audit failed')
  return {'active_tokens': totals['active'],
          'joint_nll_per_masked_token': -totals['joint'] / totals['active'],
          'frozen_unary_nll_per_masked_token': -totals['unary'] / totals['active'],
          'dependence_gain_nats_per_masked_token': (totals['joint'] - totals['unary']) / totals['active'],
          'max_marginal_error': marginal_error, 'max_direct_inferred_score_error_nats': score_error}


def write_json(path, value):
  with path.open('x') as handle:
    json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source-run-dir', type=Path, required=True)
  parser.add_argument('--source-results-sha256', required=True)
  parser.add_argument('--unary-checkpoint', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--include-dev', action='store_true')
  parser.add_argument('--steps', type=int, default=300)
  parser.add_argument('--eval-every', type=int, default=100)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--learning-rate', type=float, default=0.003)
  parser.add_argument('--feature-dim', type=int, default=8)
  parser.add_argument('--eta', type=float, default=0.95)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--threads', type=int, default=4)
  args = parser.parse_args(argv)
  if (min(args.steps, args.eval_every, args.batch_size, args.feature_dim, args.threads) < 1
      or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
      or not math.isfinite(args.eta) or not 0 < args.eta < 1):
    raise ValueError('invalid bounded training configuration')
  if args.output_dir.exists() or args.output_dir.is_symlink():
    raise FileExistsError(args.output_dir)
  torch.set_num_threads(args.threads)
  torch.manual_seed(args.seed)
  if torch.device(args.device).type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
  source = json.loads(authenticated_bytes(args.source_run_dir / 'results.json', args.source_results_sha256))
  if source.get('completed') is not True or source.get('backbone_frozen') is not True:
    raise ValueError('source must be a completed frozen-backbone debugging run')
  unary, unary_identity = load_unary(args.unary_checkpoint)
  if (source['hidden_size'] != unary.hidden_size or source['vocab_size'] != unary.vocab_size
      or source['backbone_provenance'] != unary_identity['identity']['backbone_provenance']):
    raise ValueError('old cache and unary checkpoint backbone identities differ')
  caches, attestations = {}, {}
  for split in ('train', 'dev') if args.include_dev else ('train',):
    caches[split], attestations[split] = load_old_cache(
      args.source_run_dir / 'cache' / split, source['cache'][split], split,
      hidden_size=unary.hidden_size, vocab_size=unary.vocab_size, length=source['arguments']['length'])
  torch.manual_seed(args.seed)
  head = FrozenUnaryCenteredForestHead(unary, feature_dim=args.feature_dim, eta=args.eta).to(args.device)
  frozen_sha = tensor_state_sha256(head._frozen_unary.state_dict())
  if frozen_sha != unary_identity['unary_state_sha256']:
    raise AssertionError('frozen unary changed during head construction')
  parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
  optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
  order_rng = torch.Generator().manual_seed(args.seed + 700001)
  args.output_dir.mkdir(parents=True, exist_ok=False)
  protocol = {
    'protocol': 'centered_old_debug_fixed_corruption_overfit_v1', 'schema_version': 1,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    'scope': 'old fixed-corruption fitting gate; optional old dev is diagnostic; no checkpoint selection',
    'reserved_test_access': False, 'backbone_calls': 0,
    'source_results_sha256': args.source_results_sha256, 'unary': unary_identity,
    'cache': attestations, 'parameters': head.parameter_summary(),
    'initial_head_sha256': tensor_state_sha256(head.state_dict()),
    'frozen_unary_sha256': frozen_sha, 'optimizer': {'name': 'AdamW', 'weight_decay': 0, 'gradient_clip': 1.},
    'runtime': {'torch_version': str(torch.__version__), 'device': args.device,
                'head_dtype': 'float32', 'probability_dtype': 'float64',
                'gpu_name': torch.cuda.get_device_name(args.device) if torch.device(args.device).type == 'cuda' else None},
    'source_sha256': {name: file_sha256(ROOT / name) for name in SOURCE_FILES},
  }
  write_json(args.output_dir / 'protocol.json', protocol)
  started, curve = time.monotonic(), []
  with (args.output_dir / 'curve.jsonl').open('x') as curve_file, \
       (args.output_dir / 'training-steps.jsonl').open('x') as steps_file:
    def record(step):
      if (tensor_state_sha256(head._frozen_unary.state_dict()) != frozen_sha
          or any(p.requires_grad or p.grad is not None for p in head._frozen_unary.parameters())):
        raise AssertionError('frozen unary weights or gradient contract changed')
      row = {'step': step, 'elapsed_seconds': time.monotonic() - started,
             'frozen_unary_sha256': frozen_sha,
             **{split: evaluate_debug(head, cache, args.batch_size, args.device) for split, cache in caches.items()}}
      if step == 0 and any(abs(row[split]['dependence_gain_nats_per_masked_token']) > 1e-10 for split in caches):
        raise AssertionError('initialization did not preserve exact independence')
      curve.append(row)
      curve_file.write(json.dumps(row, allow_nan=False) + '\n')
      curve_file.flush()
      print(json.dumps({'event': 'centered_debug_evaluation', **row}), flush=True)
    record(0)
    permutation, cursor = [], 0
    for step in range(1, args.steps + 1):
      if cursor >= len(permutation):
        permutation = torch.randperm(len(caches['train']), generator=order_rng).tolist()
        cursor = 0
      indices = permutation[cursor:cursor + args.batch_size]
      cursor += len(indices)
      batch = caches['train'].batch(indices, args.device)
      head.train()
      optimizer.zero_grad(set_to_none=True)
      output = head(batch['hidden'], batch['logits'], batch['sigma'], batch['active'])
      loss = -centered_forest_log_probability(output, batch['logits'], batch['targets']).sum() / batch['active'].sum()
      if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f'nonfinite coupling loss at update {step}')
      loss.backward()
      norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
      optimizer.step()
      if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise FloatingPointError('optimizer produced nonfinite coupling parameters')
      steps_file.write(json.dumps({'step': step, 'example_indices': indices,
        'active_tokens': int(batch['active'].sum()), 'nll_per_masked_token': float(loss.detach()),
        'gradient_norm_before_clip': float(norm)}, allow_nan=False) + '\n')
      steps_file.flush()
      if step % args.eval_every == 0 or step == args.steps:
        record(step)
  for split, cache in caches.items():
    current = tensor_state_sha256({f'{index}/{key}': value
      for index, row in enumerate(cache.records) for key, value in row.items()})
    if current != attestations[split]['tensor_state_sha256']:
      raise AssertionError('input cache snapshot was mutated')
  if (file_sha256(args.unary_checkpoint) != UNARY_CHECKPOINT_SHA256
      or tensor_state_sha256(unary.state_dict()) != unary_identity['unary_state_sha256']):
    raise AssertionError('source unary checkpoint or module was mutated')
  checkpoint = args.output_dir / 'final-checkpoint.pt'
  with checkpoint.open('xb') as handle:
    torch.save({'schema_version': 1, 'completed_steps': args.steps, 'head': head.state_dict(),
                'optimizer': optimizer.state_dict(), 'order_rng_state': order_rng.get_state(),
                'torch_rng_state': torch.get_rng_state(), 'permutation': permutation, 'cursor': cursor,
                'protocol_sha256': file_sha256(args.output_dir / 'protocol.json')}, handle)
  result = {'completed': True, 'completed_steps': args.steps, 'curve': curve,
            'reserved_test_access': False, 'backbone_calls': 0, 'frozen_unary_unchanged': True,
            'input_caches_unchanged': True, 'final_head_sha256': tensor_state_sha256(head.state_dict()),
            'checkpoint_sha256': file_sha256(checkpoint),
            'train_nll_reduction': curve[0]['train']['joint_nll_per_masked_token'] - curve[-1]['train']['joint_nll_per_masked_token']}
  write_json(args.output_dir / 'results.json', result)
  write_json(args.output_dir / 'hashes.json', {path.name: file_sha256(path) for path in sorted(args.output_dir.iterdir())})
  print(json.dumps({'event': 'centered_debug_complete', 'output_dir': str(args.output_dir),
                    'train_nll_reduction': result['train_nll_reduction']}), flush=True)


if __name__ == '__main__':
  main()
