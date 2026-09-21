#!/usr/bin/env python3
"""Train three output heads with shared online corruptions and a frozen MDLM.

Only train and document-disjoint development files are accepted. Development
masks are fixed across evaluations; training masks and scheduled mask rates
are redrawn every update. Each encoded example is reused by all three heads.
The default backbone batch size is one in BOTH training and development,
because the released BF16 encoder is not invariant to changing batch shape.

Checkpoints contain all heads, separate AdamW states, RNG states, data order,
cursor, completed step, and preceding logs. Resume into a new output directory
with the same data/configuration/runtime; --steps is the new TOTAL step target.
The saved state precedes its development evaluation, so resuming recomputes
that evaluation and continues without repeating any optimizer update.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch

from models.contextual_unary import ContextualUnaryAdapter
from noise_schedule import LogLinearNoise
from scripts.run_staged_real_overfit import (
  FrozenCache, build_cache, file_sha256, fixed_masks, make_head, score_batch,
  token_digest, training_loss,
)


ARMS = ('shared', 'directional', 'unary')
SOURCE_FILES = (
  'scripts/run_staged_fresh_training.py', 'scripts/run_staged_real_overfit.py',
  'models/contextual_unary.py', 'models/directional_forest.py',
  'models/structured_decoder.py', 'structured_objective.py', 'structured_utils.py',
  'structured_training.py', 'evaluation/dependence_diagnostics.py',
  'models/dit.py', 'diffusion.py', 'noise_schedule.py',
)


def canonical_hash(value) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def backbone_runtime_identity(model):
  """Pin the resolved encoder/noise settings that can affect its outputs."""
  from omegaconf import OmegaConf
  encoder = OmegaConf.to_container(model.config.model, resolve=True)
  encoder.pop('structured_decoder', None)
  return {
    'model': encoder, 'noise': OmegaConf.to_container(model.config.noise, resolve=True),
    'parameterization': model.parameterization, 'time_conditioning': model.time_conditioning,
    'subs_masking': model.subs_masking, 'backbone': model.config.backbone, 'T': model.T,
  }


def read_documents(path: Path, expected_sha256: str, count: int, length: int,
                   vocab_size: int, mask_index: int, tokenizer=None):
  """Select distinct full-length documents with explicit document identities."""
  actual = file_sha256(path)
  if actual != expected_sha256.lower():
    raise ValueError(f'{path}: input SHA256 mismatch')
  tokens, documents, seen_ids, seen_tokens = [], [], set(), set()
  skipped_short = skipped_duplicate = scanned = 0
  with path.open() as handle:
    for line_number, line in enumerate(handle, 1):
      if not line.strip():
        continue
      scanned += 1
      row = json.loads(line)
      document = row.get('document_id', row.get('source_document_sha256'))
      if not isinstance(document, str) or not document:
        raise ValueError(f'{path}:{line_number}: explicit document_id is required')
      if 'input_ids' in row:
        ids = row['input_ids']
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
          raise ValueError('input_ids must be a list of integers')
      elif isinstance(row.get('text'), str) and tokenizer is not None:
        ids = tokenizer.encode(row['text'], add_special_tokens=False)
      else:
        raise ValueError('each document needs input_ids or tokenizable text')
      if len(ids) < length:
        skipped_short += 1
        continue
      ids = ids[:length]
      if any(token < 0 or token >= vocab_size or token == mask_index for token in ids):
        raise ValueError('invalid clean token in selected document')
      tensor = torch.tensor(ids, dtype=torch.long)
      token_sha = token_digest(tensor)
      if document in seen_ids or token_sha in seen_tokens:
        skipped_duplicate += 1
        continue
      seen_ids.add(document)
      seen_tokens.add(token_sha)
      documents.append(document)
      tokens.append(tensor)
      if len(tokens) == count:
        break
  if len(tokens) != count:
    raise ValueError(f'{path}: requested {count} distinct documents, found {len(tokens)}')
  tokens = torch.stack(tokens)
  return tokens, documents, {
    'file_sha256': actual, 'selected_token_sha256': token_digest(tokens),
    'document_ids_sha256': canonical_hash(documents), 'examples': count,
    'length': length, 'records_scanned': scanned,
    'short_records_skipped': skipped_short, 'duplicate_records_skipped': skipped_duplicate,
  }


def assert_document_disjoint(train_tokens, train_documents, dev_tokens, dev_documents):
  if set(train_documents) & set(dev_documents):
    raise ValueError('training and development document identities overlap')
  if {token_digest(row) for row in train_tokens} & {token_digest(row) for row in dev_tokens}:
    raise ValueError('training and development contain identical selected token sequences')


def fresh_mask(tokens: torch.Tensor, rate: float, generator: torch.Generator):
  """Draw uniform fixed-count masks; conditioning retains the scheduled rate."""
  count = min(tokens.shape[1], max(2, round(rate * tokens.shape[1])))
  active = torch.zeros_like(tokens, dtype=torch.bool, device='cpu')
  for row in active:
    row[torch.randperm(tokens.shape[1], generator=generator)[:count]] = True
  return active


def _synchronize(device):
  if torch.device(device).type == 'cuda':
    torch.cuda.synchronize(device)


@torch.no_grad()
def online_backbone_batch(model, targets, active, rate, device, backbone_batch_size):
  """One native encoder call per microbatch, detached and shared by all heads."""
  if not isinstance(model.noise, LogLinearNoise):
    raise ValueError('fresh-corruption training requires LogLinearNoise')
  diffusion_time = rate / (1.0 - model.noise.eps)
  if not 0 < rate < 1 or diffusion_time > 1:
    raise ValueError('mask rate exceeds the loglinear schedule')
  model.eval()
  targets, active = targets.to(device), active.to(device)
  times = torch.full((len(targets),), diffusion_time, device=device)
  sigma = model.noise(times)[0]
  corrupted = torch.where(active, torch.full_like(targets, model.mask_index), targets)
  hidden_rows, logits_rows = [], []
  for offset in range(0, len(targets), backbone_batch_size):
    hidden, logits = model._structured_backbone_output(
      corrupted[offset:offset + backbone_batch_size],
      sigma[offset:offset + backbone_batch_size, None], force_no_grad=True)
    hidden_rows.append(hidden.detach().float())
    logits_rows.append(logits.detach().float())
  logits = torch.cat(logits_rows)
  if not bool(torch.isneginf(logits[..., model.mask_index]).all()):
    raise AssertionError('absorbing mask token must be excluded from clean predictions')
  return {'hidden': torch.cat(hidden_rows), 'logits': logits, 'targets': targets,
          'active': active, 'sigma': sigma, 'corrupted': corrupted}


class FreshTrainer:
  """Stateful paired training with independently resumable stochastic streams."""

  def __init__(self, model, heads, train_tokens, *, device='cpu', seed=1,
                batch_size=4, backbone_batch_size=1,
                mask_rates=(0.25, 0.5, 0.75, 0.9), learning_rate=3e-4,
                pair_warmup_steps=100, gradient_clip=1.0, identity=None):
    if tuple(heads) != ARMS:
      raise ValueError(f'heads must have this fixed order: {ARMS}')
    if (train_tokens.ndim != 2 or train_tokens.shape[1] < 2
        or min(batch_size, backbone_batch_size) < 1
        or backbone_batch_size > batch_size or learning_rate <= 0
        or pair_warmup_steps < 0 or gradient_clip <= 0 or not mask_rates):
      raise ValueError('invalid training configuration')
    if not isinstance(model.noise, LogLinearNoise):
      raise ValueError('training requires LogLinearNoise')
    if any(not 0 < rate <= 1 - model.noise.eps for rate in mask_rates):
      raise ValueError('scheduled mask rate exceeds loglinear support')
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
      raise ValueError('backbone parameters must be frozen')
    self.model, self.device, self.heads = model, device, heads
    self.train_tokens = train_tokens.detach().cpu()
    self.config = dict(seed=seed, batch_size=batch_size,
                       backbone_batch_size=backbone_batch_size,
                       mask_rates=list(mask_rates), learning_rate=learning_rate,
                       pair_warmup_steps=pair_warmup_steps, gradient_clip=gradient_clip)
    # Store plain JSON primitives: torch.__version__ is a string subclass that
    # otherwise introduces a non-allowlisted global in weights-only checkpoints.
    self.identity = json.loads(json.dumps(
      identity or {'train_tokens_sha256': token_digest(self.train_tokens)}, allow_nan=False))
    self.optimizers = {}
    for arm, head in heads.items():
      head.to(device)
      self.optimizers[arm] = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=0.0)
    self.order_rng = torch.Generator().manual_seed(seed + 50000)
    self.mask_rng = torch.Generator().manual_seed(seed + 1000)
    self.rate_rng = torch.Generator().manual_seed(seed + 2000)
    self.permutation = torch.randperm(len(train_tokens), generator=self.order_rng)
    self.cursor = self.epoch = self.completed_steps = 0
    self.step_history, self.eval_history, self.dev_records = [], [], []

  def train_step(self):
    if self.cursor >= len(self.permutation):
      self.permutation = torch.randperm(len(self.train_tokens), generator=self.order_rng)
      self.cursor = 0
      self.epoch += 1
    indices = self.permutation[self.cursor:self.cursor + self.config['batch_size']]
    self.cursor += len(indices)
    targets = self.train_tokens[indices]
    rate_index = int(torch.randint(len(self.config['mask_rates']), (), generator=self.rate_rng))
    rate = self.config['mask_rates'][rate_index]
    active = fresh_mask(targets, rate, self.mask_rng)
    _synchronize(self.device)
    started = time.monotonic()
    batch = online_backbone_batch(self.model, targets, active, rate, self.device,
                                  self.config['backbone_batch_size'])
    _synchronize(self.device)
    backbone_seconds = time.monotonic() - started
    step = self.completed_steps + 1
    active_count = int(active.sum())
    record = {
      'step': step, 'epoch': self.epoch, 'document_indices': indices.tolist(),
      'mask_rate': rate, 'actual_mask_fraction': active_count / active.numel(),
      'active_token_count': active_count, 'diffusion_time': rate / (1 - self.model.noise.eps),
      'sigma': float(batch['sigma'][0]), 'mask_sha256': token_digest(active),
      'clean_token_sha256': token_digest(targets),
      'corrupted_token_sha256': token_digest(batch['corrupted'].cpu()),
      'backbone_calls': (len(indices) + self.config['backbone_batch_size'] - 1) // self.config['backbone_batch_size'],
      'backbone_seconds': backbone_seconds, 'arms': {},
    }
    for arm, head in self.heads.items():
      head.train()
      optimizer = self.optimizers[arm]
      optimizer.zero_grad(set_to_none=True)
      _synchronize(self.device)
      arm_started = time.monotonic()
      loss = training_loss(head, batch, warmup=step <= self.config['pair_warmup_steps'])
      if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f'{arm}: nonfinite loss at step {step}')
      loss.backward()
      norm = torch.nn.utils.clip_grad_norm_(
        head.parameters(), self.config['gradient_clip'], error_if_nonfinite=True)
      optimizer.step()
      _synchronize(self.device)
      record['arms'][arm] = {
        'nll_per_masked_token': float(loss.detach()),
        'nll_sum': float(loss.detach()) * active_count,
        'gradient_norm_before_clip': float(norm),
        'optimizer_seconds': time.monotonic() - arm_started,
        'factor_mode': ('unary' if isinstance(head, ContextualUnaryAdapter)
                        else 'fixed' if step <= self.config['pair_warmup_steps'] else 'dynamic'),
      }
    self.completed_steps = step
    self.step_history.append(record)
    return record

  def state_dict(self):
    return {
      'schema_version': 1, 'identity': self.identity, 'config': self.config,
      'completed_steps': self.completed_steps, 'permutation': self.permutation,
      'cursor': self.cursor, 'epoch': self.epoch,
      'heads': {arm: head.state_dict() for arm, head in self.heads.items()},
      'optimizers': {arm: optimizer.state_dict() for arm, optimizer in self.optimizers.items()},
      'order_rng_state': self.order_rng.get_state(), 'mask_rng_state': self.mask_rng.get_state(),
      'rate_rng_state': self.rate_rng.get_state(), 'torch_rng_state': torch.get_rng_state(),
      'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.device(self.device).type == 'cuda' else [],
      'step_history': self.step_history, 'eval_history': self.eval_history,
      'dev_records': self.dev_records,
    }

  def load_state_dict(self, state):
    if state['schema_version'] != 1 or state['identity'] != self.identity or state['config'] != self.config:
      raise ValueError('resume data, protocol identity or training configuration differs')
    for arm in ARMS:
      self.heads[arm].load_state_dict(state['heads'][arm], strict=True)
      self.optimizers[arm].load_state_dict(state['optimizers'][arm])
    self.permutation = state['permutation'].cpu()
    self.cursor, self.epoch = state['cursor'], state['epoch']
    self.completed_steps = state['completed_steps']
    self.order_rng.set_state(state['order_rng_state'].cpu())
    self.mask_rng.set_state(state['mask_rng_state'].cpu())
    self.rate_rng.set_state(state['rate_rng_state'].cpu())
    torch.set_rng_state(state['torch_rng_state'].cpu())
    if state['cuda_rng_states']:
      if torch.device(self.device).type != 'cuda':
        raise ValueError('cannot resume CUDA training on CPU with exact RNG semantics')
      torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng_states']])
    self.step_history, self.eval_history = state['step_history'], state['eval_history']
    self.dev_records = state['dev_records']


def save_checkpoint(trainer: FreshTrainer, path: Path):
  if path.exists():
    raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=path.parent)
  try:
    with os.fdopen(descriptor, 'wb') as handle:
      torch.save(trainer.state_dict(), handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)
  return file_sha256(path)


def build_dev_caches(model, tokens, documents, *, rates, seed, device,
                     output_dir, backbone_batch_size):
  groups, reports = [], []
  for rate_index, rate in enumerate(rates):
    corruption_seed = seed + 900000 + rate_index
    active = fixed_masks(tokens, rate, corruption_seed)
    cache, report = build_cache(model, tokens, active, rate, device,
                                output_dir / f'rate-{rate_index:02d}',
                                backbone_batch_size=backbone_batch_size)
    metadata = []
    for index, document in enumerate(documents):
      corrupted = torch.where(active[index], torch.full_like(tokens[index], model.mask_index), tokens[index])
      metadata.append({
        'document_id': document, 'example_index': index, 'mask_rate': rate,
        'actual_mask_fraction': float(active[index].float().mean()),
        'corruption_seed': corruption_seed, 'mask_sha256': token_digest(active[index]),
        'clean_token_sha256': token_digest(tokens[index]),
        'corrupted_token_sha256': token_digest(corrupted),
      })
    groups.append((cache, metadata))
    reports.append(report)
  return groups, reports


@torch.no_grad()
def evaluate_development(trainer, groups, *, checkpoint_sha256, dataset,
                          batch_size, record_sink=None):
  """Per-document records and token-weighted metrics on fixed dev corruptions."""
  totals = {arm: {'active': 0, 'base': 0., 'joint': 0., 'marginal': 0., 'dependence': 0.}
            for arm in ARMS}
  new_records = []
  for cache, metadata in groups:
    for offset in range(0, len(cache), batch_size):
      indices = list(range(offset, min(offset + batch_size, len(cache))))
      batch = cache.batch(indices, trainer.device)
      reference_base = None
      for arm, head in trainer.heads.items():
        head.eval()
        scores = score_batch(head, batch)
        if reference_base is None:
          reference_base = scores['backbone_log_probability']
        elif not torch.equal(reference_base, scores['backbone_log_probability']):
          raise AssertionError('arms did not see identical development backbone scores')
        for local_index, index in enumerate(indices):
          row = {
            **metadata[index], 'split': 'dev', 'dataset': dataset, 'arm': arm,
            'training_seed': trainer.config['seed'], 'checkpoint_step': trainer.completed_steps,
            'checkpoint_sha256': checkpoint_sha256,
            'active_token_count': int(batch['active'][local_index].sum()),
            'backbone_log_probability': float(scores['backbone_log_probability'][local_index]),
            'joint_log_probability': float(scores['joint_log_probability'][local_index]),
            'marginal_log_probability': float(scores['marginal_log_probability'][local_index]),
            'dependence_log_probability': float(scores['dependence_log_probability'][local_index]),
          }
          new_records.append(row)
          if record_sink:
            record_sink(row)
          totals[arm]['active'] += row['active_token_count']
          for key, field in (('base', 'backbone'), ('joint', 'joint'),
                              ('marginal', 'marginal'), ('dependence', 'dependence')):
            totals[arm][key] += row[f'{field}_log_probability']
  summary = {'step': trainer.completed_steps, 'checkpoint_sha256': checkpoint_sha256, 'arms': {}}
  for arm, values in totals.items():
    count = values['active']
    summary['arms'][arm] = {
      'active_token_count': count,
      'backbone_nll_per_masked_token': -values['base'] / count,
      'joint_nll_per_masked_token': -values['joint'] / count,
      'marginal_nll_per_masked_token': -values['marginal'] / count,
      'dependence_gain_nats_per_masked_token': values['dependence'] / count,
      'joint_gain_nats_per_masked_token': (values['joint'] - values['base']) / count,
      'marginal_gain_nats_per_masked_token': (values['marginal'] - values['base']) / count,
    }
  trainer.dev_records.extend(new_records)
  trainer.eval_history.append(summary)
  return summary


def _write_json(path, value):
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _append_jsonl(path, row):
  with path.open('a') as handle:
    handle.write(json.dumps(row, allow_nan=False) + '\n')


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--train-jsonl', type=Path, required=True)
  parser.add_argument('--dev-jsonl', type=Path, required=True)
  parser.add_argument('--expected-train-sha256', required=True)
  parser.add_argument('--expected-dev-sha256', required=True)
  parser.add_argument('--backbone-checkpoint', type=Path, required=True)
  parser.add_argument('--expectations', type=Path, required=True)
  parser.add_argument('--expected-expectations-sha256', required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--resume-checkpoint', type=Path)
  parser.add_argument('--expected-resume-sha256')
  parser.add_argument('--model-config', default='contextual-forest-small')
  parser.add_argument('--data-config', default='train_openwebtext_pinned')
  parser.add_argument('--data-cache-dir', type=Path)
  parser.add_argument('--override', action='append', default=[])
  parser.add_argument('--dataset-label', default='openwebtext')
  parser.add_argument('--train-examples', '--examples', dest='train_examples', type=int, default=2048)
  parser.add_argument('--dev-examples', type=int, default=64)
  parser.add_argument('--length', type=int, default=128)
  parser.add_argument('--steps', type=int, default=1000)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--backbone-batch-size', type=int, default=1)
  parser.add_argument('--mask-rates', type=float, nargs='+', default=[0.25, 0.5, 0.75, 0.9])
  parser.add_argument('--learning-rate', type=float, default=3e-4)
  parser.add_argument('--warmup-steps', type=int, default=100)
  parser.add_argument('--eval-every', type=int, default=100)
  parser.add_argument('--rank', type=int, default=16)
  parser.add_argument('--directional-rank', type=int, default=8)
  parser.add_argument('--unary-rank', type=int, default=16)
  parser.add_argument('--top-k', type=int, default=64)
  parser.add_argument('--time-embed-dim', type=int, default=64)
  parser.add_argument('--factor-init-std', type=float, default=0.25)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--threads', type=int, default=4)
  return parser.parse_args(argv)


def main(argv=None):
  args = _args(argv)
  if min(args.train_examples, args.dev_examples, args.steps, args.eval_every, args.threads) < 1 or args.length < 2:
    raise ValueError('invalid data counts, step target, or evaluation interval')
  if len(set(args.mask_rates)) != len(args.mask_rates):
    raise ValueError('mask rates must be distinct')
  if bool(args.resume_checkpoint) != bool(args.expected_resume_sha256):
    raise ValueError('resume checkpoint and its expected SHA256 must be supplied together')
  args.output_dir.mkdir(parents=True, exist_ok=False)
  torch.set_num_threads(args.threads)
  torch.manual_seed(args.seed)
  if torch.device(args.device).type == 'cuda':
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
    checkpoint_save_dir=args.output_dir).to(args.device).eval()
  hidden_size, vocab_size = model.structured_head.hidden_size, model.structured_head.vocab_size
  train_tokens, train_docs, train_source = read_documents(
    args.train_jsonl, args.expected_train_sha256, args.train_examples, args.length,
    vocab_size, model.mask_index, model.tokenizer)
  dev_tokens, dev_docs, dev_source = read_documents(
    args.dev_jsonl, args.expected_dev_sha256, args.dev_examples, args.length,
    vocab_size, model.mask_index, model.tokenizer)
  assert_document_disjoint(train_tokens, train_docs, dev_tokens, dev_docs)
  dev_groups, cache_reports = build_dev_caches(
    model, dev_tokens, dev_docs, rates=args.mask_rates, seed=args.seed,
    device=args.device, output_dir=args.output_dir / 'dev-cache',
    backbone_batch_size=args.backbone_batch_size)
  print(json.dumps({'event': 'dev_cache_ready', 'rates': args.mask_rates,
                    'examples_per_rate': len(dev_tokens)}), flush=True)
  head_config = dict(hidden_size=hidden_size, vocab_size=vocab_size, seed=args.seed,
                     rank=args.rank, directional_rank=args.directional_rank,
                     unary_rank=args.unary_rank, top_k=args.top_k,
                     time_embed_dim=args.time_embed_dim, init_std=args.factor_init_std,
                     component_size_cap=0)
  heads = {arm: make_head(arm, **head_config) for arm in ARMS}
  source_hashes = {path: file_sha256(REPO_ROOT / path) for path in SOURCE_FILES}
  runtime = {'torch_version': str(torch.__version__), 'device_type': torch.device(args.device).type,
             'gpu_name': torch.cuda.get_device_name(args.device) if torch.device(args.device).type == 'cuda' else None,
             'backbone_cuda_autocast': 'bfloat16', 'cached_output_dtype': 'float32'}
  identity = {
    'train_source': train_source, 'dev_source': dev_source,
    'dev_cache_sha256': [report['cache_sha256'] for report in cache_reports],
    'head_config': head_config, 'source_sha256': source_hashes,
    'backbone_provenance': model.structured_backbone_provenance,
    'backbone_runtime_config': backbone_runtime_identity(model),
    'runtime': runtime, 'eval_every': args.eval_every, 'dataset': args.dataset_label,
  }
  trainer = FreshTrainer(
    model, heads, train_tokens, device=args.device, seed=args.seed,
    batch_size=args.batch_size, backbone_batch_size=args.backbone_batch_size,
    mask_rates=args.mask_rates, learning_rate=args.learning_rate,
    pair_warmup_steps=args.warmup_steps, identity=identity)
  resumed_from = None
  if args.resume_checkpoint:
    resumed_from = file_sha256(args.resume_checkpoint)
    if args.expected_resume_sha256 and resumed_from != args.expected_resume_sha256:
      raise ValueError('resume checkpoint SHA256 mismatch')
    trainer.load_state_dict(torch.load(args.resume_checkpoint, map_location=args.device, weights_only=True))
    if args.steps < trainer.completed_steps:
      raise ValueError('--steps is a total target and precedes the resume checkpoint')
  parameter_counts = {arm: {
    'rank': head.rank,
    'active_trainable_parameters': sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad),
    'inactive_frozen_parameters': sum(parameter.numel() for parameter in head.parameters() if not parameter.requires_grad),
  } for arm, head in heads.items()}
  protocol = {
    'protocol': 'paired_fresh_corruption_frozen_backbone_v1', 'schema_version': 1,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
    'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    'identity': identity, 'identity_sha256': canonical_hash(identity),
    'training_config': trainer.config, 'parameter_counts': parameter_counts,
    'dev_cache': cache_reports, 'document_disjoint': True, 'test_data_access': False,
    'corruption': 'fresh uniform fixed-count masks: max(2,round(scheduled_rate*length)); sigma uses scheduled_rate',
    'backbone_policy': 'identical explicit microbatch size for training and dev; encoded outputs shared across all heads',
    'loss': 'joint conditional NLL summed over examples, divided by total active tokens in the shared minibatch',
    'resume_parent_sha256': resumed_from, 'resume_start_step': trainer.completed_steps,
    'resume_semantics': 'same-runtime exact state continuation; saved checkpoint precedes its recomputed dev evaluation',
    'step_zero_included': True,
  }
  _write_json(args.output_dir / 'protocol.json', protocol)
  # Rehydrate prefix logs carried by the checkpoint into the new directory.
  for record in trainer.step_history:
    _append_jsonl(args.output_dir / 'training-steps.jsonl', record)
  for record in trainer.dev_records:
    _append_jsonl(args.output_dir / 'dev-records.jsonl', record)
  for record in trainer.eval_history:
    for arm in ARMS:
      _append_jsonl(args.output_dir / f'{arm}-curve.jsonl',
                    {'step': record['step'], 'checkpoint_sha256': record['checkpoint_sha256'], **record['arms'][arm]})

  def checkpoint_and_evaluate():
    checkpoint = args.output_dir / 'checkpoints' / f'step-{trainer.completed_steps:06d}.pt'
    checkpoint_sha = save_checkpoint(trainer, checkpoint)
    summary = evaluate_development(
      trainer, dev_groups, checkpoint_sha256=checkpoint_sha, dataset=args.dataset_label,
      batch_size=args.batch_size,
      record_sink=lambda row: _append_jsonl(args.output_dir / 'dev-records.jsonl', row))
    for arm in ARMS:
      _append_jsonl(args.output_dir / f'{arm}-curve.jsonl',
                    {'step': summary['step'], 'checkpoint_sha256': checkpoint_sha, **summary['arms'][arm]})
    print(json.dumps({'event': 'dev_evaluation', **summary}), flush=True)
    best = {}
    for arm in ARMS:
      selected = min(trainer.eval_history,
                      key=lambda row: (row['arms'][arm]['joint_nll_per_masked_token'], row['step']))
      best[arm] = {'step': selected['step'], 'checkpoint_sha256': selected['checkpoint_sha256'],
                   'dev_joint_nll_per_masked_token': selected['arms'][arm]['joint_nll_per_masked_token']}
    _write_json(args.output_dir / 'results.json', {
      'protocol_sha256': file_sha256(args.output_dir / 'protocol.json'),
      'completed_steps': trainer.completed_steps, 'target_steps': args.steps,
      'completed': trainer.completed_steps == args.steps,
      'best_dev_checkpoints': best, 'evaluations': trainer.eval_history,
      'parameter_counts': parameter_counts,
      'training_mask_rate_histogram': {str(rate): sum(row['mask_rate'] == rate for row in trainer.step_history)
                                      for rate in args.mask_rates},
    })

  checkpoint_and_evaluate()
  while trainer.completed_steps < args.steps:
    record = trainer.train_step()
    _append_jsonl(args.output_dir / 'training-steps.jsonl', record)
    if trainer.completed_steps % args.eval_every == 0 or trainer.completed_steps == args.steps:
      checkpoint_and_evaluate()
  from safetensors.torch import save_file
  for arm, head in heads.items():
    save_file({key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
               str(args.output_dir / f'final-{arm}-head.safetensors'))
  hashed_files = [path for path in args.output_dir.iterdir() if path.is_file()]
  hashed_files.extend(sorted((args.output_dir / 'checkpoints').glob('*.pt')))
  _write_json(args.output_dir / 'hashes.json', {
    str(path.relative_to(args.output_dir)): file_sha256(path) for path in sorted(hashed_files)})
  print(json.dumps({'event': 'finished', 'output_dir': str(args.output_dir.resolve()),
                    'completed_steps': trainer.completed_steps}), flush=True)


if __name__ == '__main__':
  main()
