"""Check four-arm composition and zero-worker cached data loading."""
from pathlib import Path

import datasets
import hydra
from omegaconf import OmegaConf
import pytest
import torch

import main  # registers the repository's Hydra resolvers
import dataloader
from scripts.smoke_ccf_training import ARMS, compose_arm
from test_structured_initialization import offline_eval_tokenizer


@pytest.mark.parametrize('arm,topology,factor,weight', [
  ('static_static','fixed','fixed',0.0), ('fixed_dynamic','fixed','dynamic',0.0),
  ('dynamic_fixed','dynamic','fixed',0.1), ('dynamic_dynamic','dynamic','dynamic',0.1),
])
def test_arm_presets_require_data_and_checkpoint(arm, topology, factor, weight):
  cfg = compose_arm(arm, [])
  head = cfg.model.structured_decoder
  assert head.topology_mode == topology and head.factor_mode == factor
  assert head.training.topology_weight == weight
  assert head.top_k == 128 and head.rank == 16 and cfg.model.length == 1024
  assert head.factor_embedding_mode == 'shared' and head.factor_conditioner_hidden_dim == 0
  assert OmegaConf.is_missing(head.training, 'backbone_checkpoint')
  assert all(OmegaConf.is_missing(cfg.data, key) for key in ('train', 'valid', 'cache_dir'))
  assert cfg.training.ema == 0 and head.training.backbone_mode == 'frozen'
  assert cfg.trainer.precision == 'bf16-mixed' and cfg.trainer.max_steps == 1000
  assert cfg.strategy.find_unused_parameters and not cfg.checkpointing.resume_from_ckpt
  assert not cfg.eval.generate_samples and not cfg.eval.compute_generative_perplexity
  assert cfg.callbacks.checkpoint_monitor.monitor == 'val/conditional_nll_per_masked_token'


def test_arms_differ_only_in_the_intended_three_fields():
  reference = None
  for arm in ARMS:
    cfg = compose_arm(arm, [])
    head = cfg.model.structured_decoder
    head.topology_mode = 'fixed'
    head.factor_mode = 'fixed'
    head.training.topology_weight = 0.0
    actual = OmegaConf.to_container(cfg, resolve=False)
    if reference is None:
      reference = actual
    else:
      assert actual == reference


def test_baseline_config_is_unchanged():
  with hydra.initialize_config_dir(
      config_dir=str(Path(main.__file__).parent / 'configs'), version_base=None):
    cfg = hydra.compose(config_name='config')
  assert cfg.model.name == 'small' and 'structured_decoder' not in cfg.model
  assert cfg.training.ema == 0.9999 and not cfg.strategy.find_unused_parameters
  assert cfg.callbacks.checkpoint_monitor.monitor == 'val/nll'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='loader checks GPU batch accounting')
def test_cached_fixture_loaders_allow_zero_workers(tmp_path):
  cfg = compose_arm('static_static', [
    'data.train=smoke-train', 'data.valid=smoke-valid', f'data.cache_dir={tmp_path}',
    'model.length=4', 'loader.global_batch_size=2', 'loader.eval_global_batch_size=2',
    'loader.batch_size=2', 'loader.eval_batch_size=2', 'loader.num_workers=0'])
  rows = {'input_ids': [[1,2,3,4], [4,3,2,1]], 'attention_mask': [[1]*4, [1]*4]}
  for name, split in [('smoke-train','train'), ('smoke-valid','validation')]:
    datasets.Dataset.from_dict(rows).save_to_disk(
      str(tmp_path / f'{name}_{split}_bs4_wrapped.dat'))
  tokenizer = object()
  train, valid = dataloader.get_dataloaders(cfg, tokenizer)
  assert train.num_workers == valid.num_workers == 0
  assert not train.persistent_workers and not valid.persistent_workers
  assert next(iter(train))['input_ids'].shape == (2,4)
  assert next(iter(valid))['input_ids'].shape == (2,4)
  assert train.tokenizer is valid.tokenizer is tokenizer
