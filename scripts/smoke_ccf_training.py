#!/usr/bin/env python3
"""Exercise main._train for four CCF arms on an offline fixture, not a benchmark."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import datasets
import hydra
import lightning as L
from omegaconf import OmegaConf
import torch

import dataloader
import diffusion
import main as training_main
from scripts.prepare_released_mdlm_owt import convert_release

ARMS = ('static_static', 'fixed_dynamic', 'dynamic_fixed', 'dynamic_dynamic')


def tensor_digest(value):
  value = value.detach().cpu().contiguous()
  return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digest(state):
  result = hashlib.sha256()
  for name, value in sorted(state.items()):
    result.update(name.encode())
    result.update(str((value.dtype, tuple(value.shape))).encode())
    result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  return result.hexdigest()


class TrainingAudit(L.Callback):
  """Observe the real training route without replacing its data or loss methods."""
  def __init__(self, output_json):
    self.output_json = Path(output_json)
    self.training_batches = []
    self.head_calls = []
    self.backward_steps = 0

  def on_fit_start(self, trainer, pl_module):
    self.backbone_before = state_digest(pl_module.backbone.state_dict())
    self.head_before = state_digest(pl_module.structured_head.state_dict())
    self.token_factor_before = tensor_digest(pl_module.structured_head.token_factor_embedding.weight)
    def capture(module, args, kwargs):
      if pl_module.training:
        self.head_calls.append({
          'active_mask': tensor_digest(kwargs['active_mask']),
          'timestep': tensor_digest(kwargs['timestep']),
        })
    self.hook = pl_module.structured_head.register_forward_pre_hook(capture, with_kwargs=True)

  def on_before_optimizer_step(self, trainer, pl_module, optimizer):
    assert all(p.grad is None for p in pl_module.backbone.parameters())
    gradients = [p.grad for p in pl_module.structured_head.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) > 0 for g in gradients)
    assert not pl_module.backbone.training
    self.backward_steps += 1

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    assert torch.isfinite(outputs['loss'])
    self.training_batches.append({
      'gold_tokens': tensor_digest(batch['input_ids']),
      'loss': float(outputs['loss'].detach()),
      'teacher_ran': bool(pl_module._last_structured_topology_metrics),
      'topology_loss': float(pl_module._last_structured_topology_metrics.get('loss', 0.)),
    })

  def on_fit_end(self, trainer, pl_module):
    self.hook.remove()
    backbone_after = state_digest(pl_module.backbone.state_dict())
    head_after = state_digest(pl_module.structured_head.state_dict())
    assert self.backbone_before == backbone_after
    assert self.head_before != head_after
    assert self.token_factor_before != tensor_digest(pl_module.structured_head.token_factor_embedding.weight)
    assert trainer.global_step == 2 and self.backward_steps == 2
    assert len(self.training_batches) == len(self.head_calls) == 2
    topology = pl_module.structured_head.topology_mode
    assert all(row['teacher_ran'] == (topology == 'dynamic') for row in self.training_batches)
    metrics = {name: float(value.detach()) for name, value in trainer.callback_metrics.items()}
    assert 'val/conditional_nll_per_masked_token' in metrics
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    result = {
      'global_step': trainer.global_step,
      'precision': trainer.precision,
      'strategy': type(trainer.strategy).__name__,
      'backbone_before_sha256': self.backbone_before,
      'backbone_after_sha256': backbone_after,
      'head_before_sha256': self.head_before,
      'head_after_sha256': head_after,
      'training_batches': self.training_batches,
      'corruption_trace': self.head_calls,
      'metrics': metrics,
    }
    self.output_json.write_text(json.dumps(result, indent=2))


def fixture_rows(tokenizer, validation=False):
  sentences = (
    ['The quiet river flows beneath the old stone bridge.',
     'A scientist carefully records the results of an experiment.']
    if validation else
    ['The cat sat beside the window and watched the rain.',
     'A dog followed the trail through the green forest.',
     'The students discussed their homework after class.',
     'Fresh bread was cooling on the wooden kitchen table.'])
  rows = []
  for sentence in sentences:
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    middle = (tokens * (30 // len(tokens) + 1))[:30]
    rows.append({'input_ids': [tokenizer.bos_token_id] + middle + [tokenizer.eos_token_id],
                 'attention_mask': [1] * 32})
  return rows


def compose_arm(arm, overrides):
  with hydra.initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
    return hydra.compose(config_name='config', overrides=[f'+experiment=ccf/{arm}', *overrides])


def run(args):
  if not torch.cuda.is_available():
    raise RuntimeError('This smoke test requires one CUDA GPU')
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  wrapper = output / 'released-backbone.pt'
  release = convert_release(args.source.resolve(), wrapper)
  shared = [
    f'model.structured_decoder.training.backbone_checkpoint={wrapper}',
    'data.train=ccf-smoke-train', 'data.valid=ccf-smoke-valid',
    f'data.cache_dir={output / "fixture"}',
    f'data.tokenizer_name_or_path={args.tokenizer.resolve()}',
    f'eval.gen_ppl_eval_model_name_or_path={args.tokenizer.resolve()}',
    'model.length=32', 'loader.global_batch_size=2', 'loader.eval_global_batch_size=2',
    'loader.batch_size=2', 'loader.eval_batch_size=2', 'loader.num_workers=0',
    'trainer.max_steps=2', 'trainer.val_check_interval=2', 'trainer.limit_val_batches=1',
    'trainer.log_every_n_steps=1', 'lr_scheduler.num_warmup_steps=0',
    'callbacks.checkpoint_every_n_steps.every_n_train_steps=2',
    'callbacks.checkpoint_every_n_steps.save_top_k=0',
    '+trainer.enable_progress_bar=false', '+trainer.enable_model_summary=false',
  ]
  tokenizer = dataloader.get_tokenizer(compose_arm(ARMS[0], shared))
  assert tokenizer.vocab_size == 50257
  fixture = output / 'fixture'
  fixture.mkdir()
  for label, split, validation in [('train', 'train', False), ('valid', 'validation', True)]:
    dataset = datasets.Dataset.from_list(fixture_rows(tokenizer, validation))
    dataset.save_to_disk(str(fixture / f'ccf-smoke-{label}_{split}_bs32_wrapped.dat'))
  results = {}
  original_cwd = Path.cwd()
  try:
    for arm in ARMS:
      run_dir = output / arm
      run_dir.mkdir()
      cfg = compose_arm(arm, [*shared,
        f'checkpointing.save_dir={run_dir}',
        '+callbacks.smoke_audit._target_=scripts.smoke_ccf_training.TrainingAudit',
        f'+callbacks.smoke_audit.output_json={run_dir / "audit.json"}',
      ])
      OmegaConf.save(cfg, run_dir / 'resolved-config.yaml', resolve=True)
      L.seed_everything(cfg.seed)
      os.chdir(run_dir)
      training_main._train(cfg, training_main.utils.get_logger(__name__), tokenizer)
      os.chdir(original_cwd)
      audit = json.loads((run_dir / 'audit.json').read_text())
      checkpoint = run_dir / 'checkpoints' / 'last.ckpt'
      assert checkpoint.is_file() and (run_dir / 'checkpoints' / 'best.ckpt').is_file()
      loaded = diffusion.Diffusion.load_from_checkpoint(
        str(checkpoint), config=cfg, tokenizer=tokenizer, map_location='cpu')
      assert state_digest(loaded.backbone.state_dict()) == audit['backbone_after_sha256']
      assert state_digest(loaded.structured_head.state_dict()) == audit['head_after_sha256']
      audit['strict_checkpoint_reload'] = True
      audit['checkpoint_bytes'] = checkpoint.stat().st_size
      audit['checkpoint_retained'] = bool(args.keep_checkpoints)
      results[arm] = audit
      del loaded
      gc.collect()
      torch.cuda.empty_cache()
      if not args.keep_checkpoints:
        for path in (run_dir / 'checkpoints').glob('*.ckpt'):
          path.unlink()
  finally:
    os.chdir(original_cwd)
  reference = results[ARMS[0]]
  for result in results.values():
    assert result['backbone_before_sha256'] == reference['backbone_before_sha256']
    assert result['head_before_sha256'] == reference['head_before_sha256']
    assert result['corruption_trace'] == reference['corruption_trace']
    assert [r['gold_tokens'] for r in result['training_batches']] == [
      r['gold_tokens'] for r in reference['training_batches']]
  report = {
    'scope': 'released backbone, offline synthetic text fixture, real main._train; not paper reproduction',
    'release': release, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
    'four_arms_paired': True, 'arms': results,
  }
  (output / 'summary.json').write_text(json.dumps(report, indent=2))
  if not args.keep_checkpoints:
    wrapper.unlink()
  print(json.dumps({'summary': str(output / 'summary.json'), 'four_arms_passed': True}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True, help='cached released model.safetensors')
  parser.add_argument('--tokenizer', type=Path, required=True, help='local GPT-2 tokenizer snapshot')
  parser.add_argument('--output', type=Path, required=True, help='new directory, must not exist')
  parser.add_argument('--keep-checkpoints', action='store_true')
  run(parser.parse_args())
