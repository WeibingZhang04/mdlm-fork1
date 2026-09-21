#!/usr/bin/env python3
"""Fit output heads on 32 fixed corruptions from an authenticated MDLM release.

This is a memorization diagnostic, not a language-model benchmark. Frozen
backbone outputs, masks, token candidates, sigma values, and minibatch orders
are shared across arms. A distinct development corpus is evaluated without
gradient updates. Full-vocabulary logits are cached one example per CPU/disk
record, never as one large GPU-resident cache.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch

from evaluation.dependence_diagnostics import decompose_structured_log_probability
from models.contextual_unary import ContextualUnaryAdapter
from models.directional_forest import DirectionalCouplingForestHead
from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import (
  factorized_token_log_probability,
  infer_structured_distribution,
  structured_token_log_probability,
)
from structured_training import structured_denoising_loss


VARIANTS = ('shared', 'directional', 'unary')
CACHE_KEYS = ('hidden', 'logits', 'targets', 'active', 'sigma', 'corrupted')


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as source:
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def token_digest(tokens: torch.Tensor) -> str:
  return hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()


def read_examples(path: Path, count: int, length: int, vocab_size: int,
                  mask_index: int, tokenizer=None) -> tuple[torch.Tensor, dict]:
  """Take the first distinct full-length records, documenting every skip."""
  selected, seen = [], set()
  skipped_short = skipped_duplicate = 0
  scanned = 0
  with path.open() as handle:
    for line_number, line in enumerate(handle, 1):
      if not line.strip():
        continue
      scanned += 1
      record = json.loads(line)
      if 'input_ids' in record:
        ids = record['input_ids']
        if not isinstance(ids, list) or any(type(value) is not int for value in ids):
          raise ValueError(f'{path}:{line_number}: input_ids must be a list of integers')
      elif 'text' in record and tokenizer is not None:
        if not isinstance(record['text'], str):
          raise ValueError(f'{path}:{line_number}: text must be a string')
        ids = tokenizer.encode(record['text'], add_special_tokens=False)
      else:
        raise ValueError(f'{path}:{line_number}: expected input_ids or tokenizable text')
      if len(ids) < length:
        skipped_short += 1
        continue
      ids = ids[:length]
      if any(value < 0 or value >= vocab_size or value == mask_index for value in ids):
        raise ValueError(f'{path}:{line_number}: invalid clean-token ID')
      tensor = torch.tensor(ids, dtype=torch.long)
      digest = token_digest(tensor)
      if digest in seen:
        skipped_duplicate += 1
        continue
      seen.add(digest)
      selected.append(tensor)
      if len(selected) == count:
        break
  if len(selected) != count:
    raise ValueError(f'{path}: requested {count} distinct length-{length} records, found {len(selected)}')
  tokens = torch.stack(selected)
  return tokens, {
    'file_sha256': file_sha256(path), 'selected_token_sha256': token_digest(tokens),
    'examples': count, 'length': length, 'records_scanned': scanned,
    'short_records_skipped': skipped_short, 'duplicate_records_skipped': skipped_duplicate,
  }


def assert_disjoint(train: torch.Tensor, dev: torch.Tensor) -> None:
  overlap = {token_digest(row) for row in train} & {token_digest(row) for row in dev}
  if overlap:
    raise ValueError('train and development sets contain identical selected token sequences')


def fixed_masks(tokens: torch.Tensor, rate: float, seed: int) -> torch.Tensor:
  """Mask a fixed number of positions uniformly without replacement."""
  if not 0 < rate < 1:
    raise ValueError('mask rate must lie strictly between zero and one')
  generator = torch.Generator().manual_seed(seed)
  active = torch.zeros_like(tokens, dtype=torch.bool)
  count = min(tokens.shape[1], max(2, round(tokens.shape[1] * rate)))
  for row in active:
    row[torch.randperm(tokens.shape[1], generator=generator)[:count]] = True
  return active


class FrozenCache:
  """A CPU or disk cache of one-example backbone outputs."""

  def __init__(self, records=None, paths=None):
    self.records = records
    self.paths = paths
    if (records is None) == (paths is None):
      raise ValueError('supply exactly one of records or paths')

  def __len__(self):
    return len(self.records if self.records is not None else self.paths)

  def record(self, index: int) -> dict:
    if self.records is not None:
      return self.records[index]
    return torch.load(self.paths[index], map_location='cpu', weights_only=True)

  def batch(self, indices, device='cpu') -> dict:
    records = [self.record(int(index)) for index in indices]
    return {key: torch.stack([row[key] for row in records]).to(device)
            for key in CACHE_KEYS}


@torch.no_grad()
def build_cache(model, tokens: torch.Tensor, active: torch.Tensor,
                rate: float, device: str, output: Path | None = None,
                backbone_batch_size: int = 1) -> tuple[FrozenCache, dict]:
  """Use the production backbone path and retain FP32 head inputs unchanged."""
  from noise_schedule import LogLinearNoise
  if not isinstance(model.noise, LogLinearNoise):
    raise ValueError('staged fixed-rate caching requires the loglinear noise schedule')
  if not 0 < rate < 1 or not 0 < model.noise.eps < 1:
    raise ValueError('mask rate and loglinear eps must lie between zero and one')
  diffusion_time = rate / (1.0 - model.noise.eps)
  if diffusion_time > 1.0:
    raise ValueError('mask rate exceeds the loglinear schedule maximum')
  if output is not None:
    output.mkdir(parents=True, exist_ok=False)
  records, paths = [], []
  digest = hashlib.sha256()
  baseline_total = 0.0
  scalar_sigma = None
  for offset in range(0, len(tokens), backbone_batch_size):
    clean = tokens[offset:offset + backbone_batch_size].to(device)
    mask = active[offset:offset + backbone_batch_size].to(device)
    time_value = torch.full((len(clean),), diffusion_time, dtype=torch.float32, device=device)
    sigma = model.noise(time_value)[0]
    scalar_sigma = float(sigma[0])
    corrupted = torch.where(mask, torch.full_like(clean, model.mask_index), clean)
    hidden, logits = model._structured_backbone_output(
      corrupted, sigma[:, None], force_no_grad=True)
    hidden, logits = hidden.float(), logits.float()
    if not bool(torch.isneginf(logits[..., model.mask_index]).all()):
      raise AssertionError('released backbone path must exclude the absorbing mask token')
    baseline_total -= float(factorized_token_log_probability(logits, clean, mask).sum())
    fields = dict(hidden=hidden.cpu(), logits=logits.cpu(), targets=clean.cpu(),
                  active=mask.cpu(), sigma=sigma.cpu(), corrupted=corrupted.cpu())
    for row in range(len(clean)):
      # Clone views so a one-example .pt does not serialize a whole batched
      # backing storage when backbone_batch_size exceeds one.
      record = {key: value[row].clone().contiguous() for key, value in fields.items()}
      for key in CACHE_KEYS:
        digest.update(key.encode())
        digest.update(record[key].numpy().tobytes())
      if output is None:
        records.append(record)
      else:
        path = output / f'example-{offset + row:04d}.pt'
        torch.save(record, path)
        paths.append(path)
    del hidden, logits, fields
  cache = FrozenCache(records=records) if output is None else FrozenCache(paths=paths)
  return cache, {
    'cache_sha256': digest.hexdigest(), 'dtype': 'float32',
    'storage': 'memory' if output is None else 'disk',
    'active_tokens': int(active.sum()), 'examples': len(tokens),
    'backbone_nll_per_masked_token': baseline_total / int(active.sum()),
    'mask_rate': rate, 'diffusion_time': diffusion_time,
    'head_and_backbone_sigma': scalar_sigma, 'loglinear_eps': model.noise.eps,
    'mask_sha256': token_digest(active),
    'corruption': 'fixed-count uniformly selected positions, fixed for all updates',
  }


def make_head(variant: str, hidden_size: int, vocab_size: int, *, seed: int,
              rank: int = 16, directional_rank: int | None = None,
              unary_rank: int = 16, top_k: int = 64,
              time_embed_dim: int = 64, init_std: float = 0.25,
              component_size_cap: int = 0):
  torch.manual_seed(seed)
  if variant == 'unary':
    return ContextualUnaryAdapter(
      hidden_size, vocab_size, rank=unary_rank, time_embed_dim=time_embed_dim,
      top_k=top_k, correction_domain='candidates')
  if variant not in ('shared', 'directional'):
    raise ValueError(f'unknown variant {variant}')
  cls = DirectionalCouplingForestHead if variant == 'directional' else ContextualCouplingForestHead
  selected_rank = (directional_rank if directional_rank is not None else max(1, rank // 2))
  selected_rank = selected_rank if variant == 'directional' else rank
  head = cls(hidden_size=hidden_size, vocab_size=vocab_size, rank=selected_rank,
             time_embed_dim=time_embed_dim, top_k=top_k, topology_dim=8,
             local_window=1, num_anchor_slots=1, contextual_neighbors=0,
             component_size_cap=component_size_cap, topology_mode='fixed',
             factor_mode='dynamic')
  # These modules cannot affect the selected fixed chain or its pair factors.
  # Exclude their parameters from the optimizer and active-capacity counts.
  for module in (head.topology_hidden_projection, head.topology_time_projection,
                 head.edge_proposer):
    module.requires_grad_(False)
  generator = torch.Generator().manual_seed(seed + 1729)
  for name in ('token_factor_embedding', 'right_token_factor_embedding'):
    if hasattr(head, name):
      torch.nn.init.normal_(getattr(head, name).weight,
                            mean=math.log(math.expm1(1.0)), std=init_std,
                            generator=generator)
  return head


def head_output(head, batch: dict, warmup: bool = False):
  return head(batch['hidden'], batch['logits'], batch['sigma'], batch['active'],
              factor_mode='fixed' if warmup else 'dynamic')


def training_loss(head, batch: dict, warmup: bool = False) -> torch.Tensor:
  if isinstance(head, ContextualUnaryAdapter):
    return head.nll_loss(batch['hidden'], batch['logits'], batch['sigma'],
                         batch['targets'], batch['active'])
  output = head_output(head, batch, warmup)
  return structured_denoising_loss(
    output, batch['logits'], batch['targets'], batch['active']).loss


@torch.no_grad()
def score_batch(head, batch: dict) -> dict:
  base = factorized_token_log_probability(batch['logits'], batch['targets'], batch['active'])
  if isinstance(head, ContextualUnaryAdapter):
    marginal = head.target_log_probs(
      batch['hidden'], batch['logits'], batch['sigma'], batch['targets'], batch['active']).sum(-1)
    joint, dependence = marginal, torch.zeros_like(marginal)
    shrinkage = {str(value): joint for value in (0.0, 0.25, 0.5, 0.75, 1.0)}
    ids = batch['logits'].topk(head.top_k, dim=-1).indices
    edge_count = 0
    identity_error = 0.0
  else:
    output = head_output(head, batch)
    inference = infer_structured_distribution(output, batch['active'])
    decomposition = decompose_structured_log_probability(
      output, batch['logits'], batch['targets'], batch['active'], inference)
    joint = structured_token_log_probability(
      output, batch['logits'], batch['targets'], batch['active'], inference)
    marginal = decomposition.marginal_log_probability
    dependence = decomposition.dependence_log_probability
    identity_error = float((joint - marginal - dependence).abs().max())
    torch.testing.assert_close(joint, marginal + dependence, atol=5e-3, rtol=1e-5)
    shrinkage = {str(value): decomposition.shrinkage_log_probability(value)
                 for value in (0.0, 0.25, 0.5, 0.75, 1.0)}
    ids = output.candidate_ids
    expected = batch['logits'].topk(head.top_k, dim=-1).indices
    if not torch.equal(ids, expected):
      raise AssertionError('head changed target-independent base candidate IDs')
    edge_count = int(output.edge_mask.sum())
  return {
    'backbone_log_probability': base, 'joint_log_probability': joint,
    'marginal_log_probability': marginal, 'dependence_log_probability': dependence,
    'shrinkage_log_probability': shrinkage,
    'active_tokens': int(batch['active'].sum()), 'edge_count': edge_count,
    'candidate_hits': int((ids.eq(batch['targets'][..., None]).any(-1) & batch['active']).sum()),
    'max_decomposition_error_nats': identity_error,
  }


@torch.no_grad()
def evaluate(head, cache: FrozenCache, batch_size: int, device: str) -> dict:
  head.eval()
  totals = dict(backbone=0.0, joint=0.0, marginal=0.0, dependence=0.0)
  active_tokens = candidate_hits = edges = 0
  max_error = 0.0
  shrinkage = {str(value): 0.0 for value in (0.0, 0.25, 0.5, 0.75, 1.0)}
  for offset in range(0, len(cache), batch_size):
    batch = cache.batch(range(offset, min(offset + batch_size, len(cache))), device)
    scores = score_batch(head, batch)
    for key in totals:
      totals[key] += float(scores[f'{key}_log_probability'].sum())
    for strength in shrinkage:
      shrinkage[strength] += float(scores['shrinkage_log_probability'][strength].sum())
    active_tokens += scores['active_tokens']
    candidate_hits += scores['candidate_hits']
    edges += scores['edge_count']
    max_error = max(max_error, scores['max_decomposition_error_nats'])
  result = {f'{key}_nll_per_masked_token': -totals[key] / active_tokens
            for key in ('backbone', 'joint', 'marginal')}
  result.update({
    'dependence_gain_nats_per_masked_token': totals['dependence'] / active_tokens,
    'joint_gain_nats_per_masked_token': (totals['joint'] - totals['backbone']) / active_tokens,
    'marginal_gain_nats_per_masked_token': (totals['marginal'] - totals['backbone']) / active_tokens,
    'shrinkage_nll_per_masked_token': {key: -value / active_tokens for key, value in shrinkage.items()},
    'candidate_recall': candidate_hits / active_tokens,
    'active_tokens': active_tokens, 'edges': edges,
    'max_decomposition_error_nats': max_error,
  })
  return result


@torch.no_grad()
def check_serial_batch(head, cache: FrozenCache, device: str) -> dict:
  count = min(2, len(cache))
  together = score_batch(head, cache.batch(range(count), device))
  error = {}
  for field in ('joint_log_probability', 'marginal_log_probability', 'dependence_log_probability'):
    apart = torch.cat([score_batch(head, cache.batch([index], device))[field]
                       for index in range(count)])
    error[field] = float((apart - together[field]).abs().max())
    torch.testing.assert_close(apart, together[field], atol=5e-3, rtol=1e-5)
  return {'examples': count, 'max_absolute_errors_nats': error, 'passed': True}


def train_head(head, train: FrozenCache, dev: FrozenCache, *, steps: int,
               batch_size: int, learning_rate: float, eval_every: int,
               seed: int, device: str, warmup_steps: int = 0,
               callback=None) -> dict:
  if min(steps, batch_size, eval_every) < 1 or learning_rate <= 0:
    raise ValueError('steps, batch_size, eval_every and learning_rate must be positive')
  head.to(device)
  optimizer = torch.optim.AdamW(
    [p for p in head.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.0)
  generator = torch.Generator().manual_seed(seed + 50000)
  curve = []
  started = time.monotonic()

  def record(step, extra=None):
    row = {'step': step, 'elapsed_seconds': time.monotonic() - started,
           'train': evaluate(head, train, batch_size, device),
           'dev': evaluate(head, dev, batch_size, device)}
    row.update(extra or {})
    curve.append(row)
    if callback:
      callback(row)

  serial_before = check_serial_batch(head, train, device)
  record(0)
  permutation = torch.randperm(len(train), generator=generator).tolist()
  cursor = 0
  for step in range(1, steps + 1):
    if cursor >= len(permutation):
      permutation = torch.randperm(len(train), generator=generator).tolist()
      cursor = 0
    indices = permutation[cursor:cursor + batch_size]
    cursor += len(indices)
    batch = train.batch(indices, device)
    head.train()
    optimizer.zero_grad(set_to_none=True)
    loss = training_loss(head, batch, warmup=step <= warmup_steps)
    if not bool(torch.isfinite(loss)):
      raise FloatingPointError(f'nonfinite training loss at step {step}')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if step % eval_every == 0 or step == steps:
      record(step, {'last_minibatch_nll': float(loss.detach()),
                    'gradient_norm_before_clip': float(norm)})
  first = curve[0]['train']['joint_nll_per_masked_token']
  last = curve[-1]['train']['joint_nll_per_masked_token']
  return {'curve': curve, 'train_nll_reduction': first - last,
          'train_loss_fell': last < first,
          'serial_batch_before': serial_before,
          'serial_batch_after': check_serial_batch(head, train, device)}


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--train-jsonl', required=True, type=Path)
  parser.add_argument('--dev-jsonl', required=True, type=Path)
  parser.add_argument('--backbone-checkpoint', required=True, type=Path)
  parser.add_argument('--expectations', required=True, type=Path)
  parser.add_argument('--expected-expectations-sha256', required=True)
  parser.add_argument('--output-dir', required=True, type=Path)
  parser.add_argument('--model-config', default='contextual-forest-small')
  parser.add_argument('--data-config', default='train_openwebtext_pinned')
  parser.add_argument('--data-cache-dir', type=Path)
  parser.add_argument('--override', action='append', default=[])
  parser.add_argument('--examples', type=int, default=32)
  parser.add_argument('--dev-examples', type=int, default=32)
  parser.add_argument('--length', type=int, default=128)
  parser.add_argument('--mask-rate', type=float, default=0.5)
  parser.add_argument('--steps', type=int, default=500)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--backbone-batch-size', type=int, default=1)
  parser.add_argument('--learning-rate', type=float, default=0.003)
  parser.add_argument('--eval-every', type=int, default=50)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--threads', type=int, default=4)
  parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
  parser.add_argument('--rank', type=int, default=16)
  parser.add_argument('--directional-rank', type=int,
                      help='default: rank//2, giving equal active capacity to shared factors')
  parser.add_argument('--unary-rank', type=int, default=16)
  parser.add_argument('--top-k', type=int, default=64)
  parser.add_argument('--time-embed-dim', type=int, default=64)
  parser.add_argument('--factor-init-std', '--init-std', dest='init_std', type=float, default=0.25)
  parser.add_argument('--component-size-cap', type=int, default=0)
  parser.add_argument('--warmup-steps', type=int, default=0)
  parser.add_argument('--cache-mode', choices=('disk', 'memory'), default='disk')
  return parser.parse_args(argv)


def main(argv=None):
  args = _args(argv)
  if args.directional_rank is None:
    args.directional_rank = max(1, args.rank // 2)
  if (min(args.examples, args.dev_examples, args.steps, args.batch_size,
          args.backbone_batch_size, args.eval_every, args.threads) < 1
      or args.length < 2 or not 0 < args.mask_rate < 1
      or args.learning_rate <= 0 or args.init_std <= 0
      or not 0 <= args.warmup_steps < args.steps):
    raise ValueError('invalid data, optimization, or masking settings')
  if len(set(args.variants)) != len(args.variants):
    raise ValueError('variants must not contain duplicates')
  args.output_dir.mkdir(parents=True, exist_ok=False)
  torch.set_num_threads(args.threads)
  torch.manual_seed(args.seed)
  if args.device.startswith('cuda'):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
  # Heavy optional Lightning/Hydra imports remain outside the unit-test path.
  from scripts.export_contextual_forest_adapter import (
    build_production_model, load_production_expectations)
  expectations = load_production_expectations(
    args.expectations, expected_sha256=args.expected_expectations_sha256)
  model = build_production_model(
    model_config=args.model_config, data_config=args.data_config,
    backbone_checkpoint=args.backbone_checkpoint, expectations=expectations,
    overrides=args.override, runtime_mode='ppl_eval', data_cache_dir=args.data_cache_dir,
    checkpoint_save_dir=args.output_dir).to(args.device).eval()
  if any(parameter.requires_grad for parameter in model.backbone.parameters()):
    raise AssertionError('backbone must be frozen')
  hidden_size = model.structured_head.hidden_size
  vocab_size = model.structured_head.vocab_size
  train_tokens, train_source = read_examples(
    args.train_jsonl, args.examples, args.length, vocab_size, model.mask_index, model.tokenizer)
  dev_tokens, dev_source = read_examples(
    args.dev_jsonl, args.dev_examples, args.length, vocab_size, model.mask_index, model.tokenizer)
  assert_disjoint(train_tokens, dev_tokens)
  caches, cache_reports = {}, {}
  for split, tokens, mask_seed in (
      ('train', train_tokens, args.seed + 1000), ('dev', dev_tokens, args.seed + 2000)):
    active = fixed_masks(tokens, args.mask_rate, mask_seed)
    output = args.output_dir / 'cache' / split if args.cache_mode == 'disk' else None
    caches[split], cache_reports[split] = build_cache(
      model, tokens, active, args.mask_rate, args.device, output,
      backbone_batch_size=args.backbone_batch_size)
    print(json.dumps({'event': 'cache_ready', 'split': split, **cache_reports[split]}), flush=True)
  provenance = model.structured_backbone_provenance
  del model
  if args.device.startswith('cuda'):
    torch.cuda.empty_cache()
  sources = ('scripts/run_staged_real_overfit.py', 'models/contextual_unary.py',
             'models/directional_forest.py', 'models/structured_decoder.py',
             'evaluation/dependence_diagnostics.py', 'structured_objective.py',
             'structured_utils.py', 'structured_training.py', 'diffusion.py')
  protocol = {
    'protocol': 'fixed_corruption_32_example_overfit_v1',
    'purpose': 'staged memorization diagnostic; not benchmark or generation evidence',
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
    'source_sha256': {path: file_sha256(REPO_ROOT / path) for path in sources},
    'arguments': {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()},
    'backbone_provenance': provenance, 'torch_version': torch.__version__,
    'train_source': train_source, 'dev_source': dev_source,
    'cache': cache_reports, 'hidden_size': hidden_size, 'vocab_size': vocab_size,
    'graph': 'natural-order chain through active positions; identical in both pair arms',
    'candidate_domain': 'unchanged base top-K plus unchanged residual conditionals',
    'backbone_frozen': True, 'data_split_overlap': False,
    'arms': {},
  }
  (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
  for variant in args.variants:
    head = make_head(variant, hidden_size, vocab_size, seed=args.seed,
                     rank=args.rank, directional_rank=args.directional_rank,
                     unary_rank=args.unary_rank, top_k=args.top_k,
                     time_embed_dim=args.time_embed_dim, init_std=args.init_std,
                     component_size_cap=args.component_size_cap)
    parameter_report = {
      'total_parameters': sum(p.numel() for p in head.parameters()),
      'active_trainable_parameters': sum(p.numel() for p in head.parameters() if p.requires_grad),
      'inactive_frozen_parameters': sum(p.numel() for p in head.parameters() if not p.requires_grad),
      'rank': head.rank,
    }
    curve_path = args.output_dir / f'{variant}-curve.jsonl'

    def record(row):
      with curve_path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
      print(json.dumps({'variant': variant, **row}, allow_nan=False), flush=True)

    result = train_head(
      head, caches['train'], caches['dev'], steps=args.steps,
      batch_size=args.batch_size, learning_rate=args.learning_rate,
      eval_every=args.eval_every, seed=args.seed, device=args.device,
      warmup_steps=args.warmup_steps, callback=record)
    # Safetensors contains only the head, not the large immutable backbone.
    from safetensors.torch import save_file
    checkpoint_path = args.output_dir / f'{variant}-head.safetensors'
    save_file({key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
              str(checkpoint_path))
    result.update(parameter_report)
    result['checkpoint_sha256'] = file_sha256(checkpoint_path)
    protocol['arms'][variant] = result
    protocol['completed'] = False
    (args.output_dir / 'results.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    del head
  if 'unary' in protocol['arms']:
    unary_count = protocol['arms']['unary']['active_trainable_parameters']
    for variant, result in protocol['arms'].items():
      result['active_parameter_difference_from_unary'] = result['active_trainable_parameters'] - unary_count
  protocol['completed'] = True
  (args.output_dir / 'results.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
  print(json.dumps({'event': 'finished', 'output_dir': str(args.output_dir.resolve())}), flush=True)


if __name__ == '__main__':
  main()
