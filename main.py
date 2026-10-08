import hashlib
import inspect
import json
import os
import platform
from pathlib import Path

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


@torch.no_grad()
def save_native_samples(model, config):
  """Warm up, record native draws, and export raw IDs for the common scorer."""
  if config.sampling.semi_ar or model.sampler != 'ddpm_cache' or model.parameterization != 'subs':
    raise ValueError('Native sample export currently requires SUBS ddpm_cache, not semi-AR')
  warmup = config.sampling.warmup_batches
  batches = config.sampling.num_sample_batches
  if type(warmup) is not int or warmup < 0 or batches < 1 or config.loader.eval_batch_size < 1:
    raise ValueError('Require nonnegative warmup and positive batch/sample counts')
  output = Path(config.eval.sample_output_dir)
  output.mkdir(parents=True, exist_ok=True)
  for name in ('samples.jsonl', 'metrics.json', 'manifest.json'):
    if (output/name).exists():
      raise FileExistsError(f'Choose a new native output directory: {output/name}')

  # Warmup must not advance the production RNG sequence from config.seed.
  devices = [model.device] if model.device.type == 'cuda' else []
  with torch.random.fork_rng(devices=devices):
    for _ in range(warmup):
      model.restore_model_and_sample(num_steps=config.sampling.steps)

  calls = 0

  def count_forward(module, inputs, output):
    nonlocal calls
    calls += 1

  source_root = Path(__file__).resolve().parent
  sources = [source_root/name for name in ('main.py', 'diffusion.py')]
  backbone_source = inspect.getsourcefile(type(model.backbone))
  if backbone_source:
    sources.append(Path(backbone_source))
  metadata = {
    'backend': 'native_mdlm', 'checkpoint': str(config.eval.checkpoint_path),
    'seed': config.seed, 'steps': config.sampling.steps, 'length': config.model.length,
    'batch_size': config.loader.eval_batch_size, 'warmup_batches': warmup,
    'noise_removal': config.sampling.noise_removal,
    'vocab_cap': config.sampling.vocab_cap,
    'timing_scope': 'native diffusion loop including optional final noise removal; excludes setup, EMA copying, warmup, decoding, I/O and scoring',
    'scoring': 'saved raw token IDs; use the existing GPT-2-large score-only path',
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'python': platform.python_version(), 'torch': str(torch.__version__),
    'cuda': torch.version.cuda, 'hostname': platform.node(),
    'cpu_affinity_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
    'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
    'gpu': torch.cuda.get_device_name(model.device) if model.device.type == 'cuda' else None,
  }
  (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
  handle = model.backbone.register_forward_hook(count_forward)
  elapsed = 0.
  total = 0
  text_samples = []
  try:
    with (output/'samples.jsonl').open('x') as stream:
      for batch in range(batches):
        before_calls = calls
        model._sample_elapsed_seconds = None
        samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
        seconds = model._sample_elapsed_seconds
        if seconds is None:
          raise RuntimeError('Native sampler did not record its generation timing')
        batch_calls = calls-before_calls
        rows = samples.cpu().tolist()
        text_samples = model.tokenizer.batch_decode(samples)
        for row, text in zip(rows, text_samples):
          record = {'sample_id': total, 'token_ids': row, 'text': text,
                    'prefix_length': 0, 'batch_id': batch, 'batch_size': len(rows),
                    'elapsed_seconds': seconds/len(rows),
                    'backbone_calls': batch_calls, 'vocab_cap': config.sampling.vocab_cap}
          stream.write(json.dumps(record, allow_nan=False)+'\n')
          total += 1
        stream.flush()
        elapsed += seconds
        print(json.dumps({'native_completed': total, 'batch_seconds': seconds,
                          'backbone_calls': batch_calls}), flush=True)
  finally:
    handle.remove()
  metrics = {**{k: metadata[k] for k in ('backend', 'vocab_cap', 'noise_removal', 'timing_scope')},
             'samples': total, 'elapsed_seconds': elapsed, 'seconds_per_sample': elapsed/total,
             'actual_backbone_calls': calls, 'backbone_calls_per_batch': calls/batches,
             'warmup_batches': warmup}
  (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
  print(json.dumps(metrics, indent=2), flush=True)
  return text_samples


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  if config.eval.get('sample_output_dir') is not None:
    return save_native_samples(model, config)
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()
