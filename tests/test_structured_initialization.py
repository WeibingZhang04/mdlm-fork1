"""CCF construction/loading/lifecycle checks; no training-loss integration yet."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import diffusion


@pytest.fixture(autouse=True)
def offline_eval_tokenizer(monkeypatch):
  # Constructor normally fetches a tokenizer for optional generative PPL.
  # That download is unrelated to these initialization tests.
  monkeypatch.setattr(
    diffusion.transformers.AutoTokenizer, 'from_pretrained',
    lambda *args, **kwargs: SimpleNamespace(pad_token='pad', pad_token_id=0))


def _config(structured=None):
  config = OmegaConf.create({
    'backbone': 'dit', 'parameterization': 'subs', 'T': 0,
    'subs_masking': False, 'time_conditioning': False,
    'sampling': {'predictor': 'ddpm_cache'},
    'eval': {'gen_ppl_eval_model_name_or_path': 'unused'},
    'training': {'antithetic_sampling': True, 'importance_sampling': False,
                 'change_of_variables': False, 'ema': 0, 'sampling_eps': 0.001},
    'optim': {'lr': 0.0003}, 'noise': {'type': 'loglinear'},
    'model': {'hidden_size': 64, 'cond_dim': 32, 'n_heads': 4,
              'n_blocks': 1, 'dropout': 0.2, 'scale_by_sigma': True},
  })
  if structured is not None:
    config.model.structured_decoder = structured
  return config


def _model(config):
  tokenizer = SimpleNamespace(vocab_size=11, mask_token=None)
  return diffusion.Diffusion(config, tokenizer)


def _head_config(path=None, topology='fixed', factor='fixed'):
  return {
    'enabled': True, 'top_k': 3, 'rank': 2, 'time_embed_dim': 8,
    'topology_dim': 8, 'num_anchor_slots': 2, 'contextual_neighbors': 1,
    'topology_mode': topology, 'factor_mode': factor,
    'training': {'backbone_mode': 'frozen', 'backbone_checkpoint': path},
  }


def _checkpoint(tmp_path, bare=False):
  backbone = _model(_config()).backbone
  with torch.no_grad():
    for parameter in backbone.parameters():
      parameter.uniform_(-0.1, 0.1)
  state = backbone.state_dict()
  path = tmp_path / 'backbone.ckpt'
  if bare:
    torch.save(state, path)
  else:
    torch.save({
      'state_dict': {**{'backbone.' + key: value for key, value in state.items()},
                     'unrelated_metric': torch.tensor(7.)},
      # Deliberately different: raw loading must not silently use EMA.
      'ema': {'shadow_params': [torch.zeros_like(p) for p in backbone.parameters()]},
    }, path)
  return str(path), state


def test_disabled_ccf_preserves_initialization_rng_and_baseline_modes():
  torch.manual_seed(41)
  baseline = _model(_config())
  rng = torch.get_rng_state()
  torch.manual_seed(41)
  disabled = _model(_config({'enabled': False}))
  assert torch.equal(torch.get_rng_state(), rng)
  assert baseline.state_dict().keys() == disabled.state_dict().keys()
  for key, value in baseline.state_dict().items():
    torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
  assert disabled.structured_head is None
  assert not disabled.structured_enabled
  disabled.eval().train()
  disabled.on_train_epoch_start()
  assert disabled.backbone.training
  assert all(p.requires_grad for p in disabled.backbone.parameters())


@pytest.mark.parametrize('topology,factor', [
  ('fixed', 'fixed'), ('fixed', 'dynamic'),
  ('dynamic', 'fixed'), ('dynamic', 'dynamic'),
])
def test_four_arms_load_register_and_freeze(tmp_path, topology, factor):
  path, state = _checkpoint(tmp_path)
  model = _model(_config(_head_config(path, topology, factor)))
  assert model.structured_enabled
  assert model.structured_head.topology_mode == topology
  assert model.structured_head.factor_mode == factor
  assert not model.backbone.training
  for key, value in state.items():
    torch.testing.assert_close(value, model.backbone.state_dict()[key], rtol=0, atol=0)
  assert all(not p.requires_grad for p in model.backbone.parameters())
  assert all(p.requires_grad for p in model.structured_head.parameters())
  assert any(key.startswith('structured_head.') for key in model.state_dict())
  model.eval()
  assert not model.structured_head.training
  assert model.train() is model
  model.on_train_epoch_start()
  assert model.structured_head.training
  assert not model.backbone.training
  assert all(not module.training for module in model.backbone.modules())
  # Registered head and frozen backbone survive an ordinary state-dict reload.
  restored = _model(_config(_head_config(path, topology, factor)))
  restored.load_state_dict(model.state_dict(), strict=True)


def test_bare_backbone_checkpoint_and_strict_mismatch(tmp_path):
  path, state = _checkpoint(tmp_path, bare=True)
  _model(_config(_head_config(path)))
  bad = dict(state)
  bad.pop(next(iter(bad)))
  torch.save(bad, path)
  with pytest.raises(RuntimeError, match='Missing key'):
    _model(_config(_head_config(path)))
  torch.save({**state, 'unexpected': torch.tensor(1.)}, path)
  with pytest.raises(RuntimeError, match='Unexpected key'):
    _model(_config(_head_config(path)))
  bad = dict(state)
  key = next(iter(bad))
  bad[key] = torch.zeros(1)
  torch.save(bad, path)
  with pytest.raises(RuntimeError, match='size mismatch'):
    _model(_config(_head_config(path)))


def test_pretrained_is_required_unless_explicit_smoke_test():
  config = _config(_head_config())
  with pytest.raises(ValueError, match='backbone_checkpoint'):
    _model(config)
  config.model.structured_decoder.training.require_pretrained_backbone = False
  with pytest.warns(UserWarning, match='random backbone'):
    model = _model(config)
  assert not model.backbone.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real DiT needs CUDA')
def test_frozen_backbone_features_repeat_after_train_mode(tmp_path):
  path, _ = _checkpoint(tmp_path)
  model = _model(_config(_head_config(path))).cuda().train()
  tokens = torch.tensor([[1, 2, 11, 3]], device='cuda')
  sigma = torch.tensor([0.5], device='cuda')
  hidden, logits = model._structured_backbone_output(tokens, sigma)
  hidden_again, logits_again = model._structured_backbone_output(tokens, sigma)
  torch.testing.assert_close(hidden, hidden_again, rtol=0, atol=0)
  torch.testing.assert_close(logits, logits_again, rtol=0, atol=0)
  assert not hidden.requires_grad
  assert not logits.requires_grad
  assert torch.isneginf(logits[:, :, model.mask_index]).all()
  with torch.no_grad():
    ordinary_logits = model.backbone(tokens, model._process_sigma(sigma)).float()
  torch.testing.assert_close(logits[:, :, :11], ordinary_logits[:, :, :11],
                             rtol=0, atol=0)
