"""Four-arm inference wiring and strict checkpoint tests; no quality benchmark.

The reference draw/reveal order below is from g-experiments diffusion.py at
2051502329429a252d3b806e0ed195ff379c42b6 (_structured_ddpm_update / _sample).
It intentionally uses the retained joint math and a separate generation loop.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import hydra
import lightning as L
from omegaconf import OmegaConf
import pytest
import torch

import diffusion
import main
from structured_objective import sample_structured_tokens


ARMS = [('fixed', 'fixed'), ('fixed', 'dynamic'),
        ('dynamic', 'fixed'), ('dynamic', 'dynamic')]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='DiT needs CUDA')


def tiny_config(topology='fixed', factor='fixed'):
  return OmegaConf.create({
    'seed': 1, 'backbone': 'dit', 'parameterization': 'subs', 'T': 0,
    'subs_masking': False, 'time_conditioning': False,
    'sampling': {'predictor': 'ddpm', 'steps': 4, 'noise_removal': True,
                 'semi_ar': False, 'num_sample_batches': 1,
                 'stride_length': 4, 'num_strides': 1},
    'loader': {'eval_batch_size': 2},
    'eval': {'gen_ppl_eval_model_name_or_path': 'must-not-be-downloaded',
             'compute_generative_perplexity': False, 'disable_ema': False},
    'training': {'antithetic_sampling': True, 'importance_sampling': False,
                 'change_of_variables': False, 'ema': 0, 'sampling_eps': 0.001},
    'optim': {'lr': 0.0003}, 'noise': {'type': 'loglinear'},
    'model': {'hidden_size': 64, 'cond_dim': 32, 'n_heads': 4,
              'n_blocks': 1, 'length': 8, 'dropout': 0.2, 'scale_by_sigma': True,
              'structured_decoder': {
                'enabled': True, 'top_k': 3, 'rank': 2, 'time_embed_dim': 8,
                'topology_dim': 8, 'num_anchor_slots': 2,
                'contextual_neighbors': 1, 'local_window': 2,
                'component_size_cap': 4, 'topology_mode': topology,
                'factor_mode': factor, 'sampling': {'mode': 'structured_joint'},
                'training': {'backbone_mode': 'frozen',
                             'backbone_checkpoint': 'unavailable-original.pt'}}}})


def tiny_model(cfg=None):
  model = diffusion.Diffusion(
    cfg if cfg is not None else tiny_config(),
    SimpleNamespace(vocab_size=11, mask_token=None),
    initialize_pretrained_backbone=False)
  # Nonzero output/factors prevent trivial uniform/independent-only fixtures.
  with torch.no_grad():
    for parameter in model.backbone.parameters():
      parameter.uniform_(-0.1, 0.1)
    for parameter in model.structured_head.parameters():
      parameter.uniform_(-0.3, 0.3)
  return model


@torch.no_grad()
def reference_clean(model, x, sigma):
  active = x.eq(model.mask_index)
  if not bool(active.any().item()):
    return x
  output, logits = model._structured_head_output(x, sigma, active)
  clean = sample_structured_tokens(output, logits, active, num_samples=1)[:, 0]
  return torch.where(active, clean, x)


@torch.no_grad()
def reference_trajectory(model, steps, eps):
  x = model._sample_prior(2, 8).to(model.device)
  times = torch.linspace(1, eps, steps + 1, device=model.device)
  dt = (1 - eps) / steps
  trajectory = []
  for time in times[:-1]:
    t = time * torch.ones(2, 1, device=model.device)
    sigma_t = model.noise(t)[0].squeeze(-1)
    sigma_s = model.noise(t - dt)[0].squeeze(-1)
    chance_t = 1 - torch.exp(-sigma_t)
    chance_s = 1 - torch.exp(-sigma_s)
    clean = reference_clean(model, x, sigma_t)
    probability = ((chance_t - chance_s) / chance_t.clamp_min(1e-12)).clamp(0, 1)
    reveal = torch.rand(x.shape, device=x.device) < probability[:, None]
    x = torch.where(reveal & x.eq(model.mask_index), clean, x)
    trajectory.append(x.clone())
  if model.config.sampling.noise_removal:
    sigma = model.noise(times[-1] * torch.ones(2, 1, device=model.device))[0]
    x = reference_clean(model, x, sigma)
  return trajectory, x


@CUDA
@pytest.mark.parametrize('topology,factor', ARMS)
@pytest.mark.parametrize('noise_removal', [False, True])
def test_seeded_trajectory_rng_and_input_boundary(monkeypatch, topology, factor, noise_removal):
  torch.manual_seed(7)
  model = tiny_model(tiny_config(topology, factor)).cuda().eval()
  model.config.sampling.noise_removal = noise_removal
  # No gold-token teacher or training loss is legal during generation.
  def forbidden(*args, **kwargs):
    raise AssertionError('training supervision called during inference')
  monkeypatch.setattr(model, '_forward_pass_structured', forbidden)
  monkeypatch.setattr(model, '_structured_topology_loss', forbidden)
  torch.manual_seed(123)
  expected_steps, expected = reference_trajectory(model, 4, 0.15)
  expected_rng = torch.cuda.get_rng_state()
  torch.manual_seed(123)
  steps, inputs, backbone_inputs = [], [], []
  update = model._structured_ddpm_update
  def capture_update(x, t, dt):
    result = update(x, t, dt)
    assert torch.equal(result[x.ne(model.mask_index)], x[x.ne(model.mask_index)])
    steps.append(result.clone())
    return result
  monkeypatch.setattr(model, '_structured_ddpm_update', capture_update)
  hook = model.structured_head.register_forward_pre_hook(
    lambda module, args, kwargs: inputs.append(kwargs), with_kwargs=True)
  backbone_hook = model.backbone.vocab_embed.register_forward_pre_hook(
    lambda module, args: backbone_inputs.append(args[0].clone()))
  actual = model.restore_model_and_sample(4, eps=0.15)
  hook.remove()
  backbone_hook.remove()
  assert inputs and len(steps) == 4
  assert len(inputs) == len(backbone_inputs)
  for kwargs, tokens in zip(inputs, backbone_inputs):
    assert 'gold_tokens' not in kwargs
    assert torch.equal(kwargs['active_mask'], tokens.eq(model.mask_index))
  for result, target in zip(steps, expected_steps):
    torch.testing.assert_close(result, target, rtol=0, atol=0)
  assert torch.equal(actual, expected)
  assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
  if noise_removal:
    assert not actual.eq(model.mask_index).any()
  assert not model.backbone.training and not model.structured_head.training
  assert all(p.grad is None for p in model.parameters())


@CUDA
def test_observed_tokens_short_circuit_and_final_joint_draw(monkeypatch):
  model = tiny_model().cuda().eval()
  x = torch.tensor([[1, 11, 2, 11]], device='cuda')
  torch.manual_seed(19)
  result = model._structured_clean_sample(x, torch.tensor([0.5], device='cuda'))
  assert torch.equal(result[x.ne(11)], x[x.ne(11)])
  assert not result.eq(11).any()
  before = torch.cuda.get_rng_state()
  assert model._structured_clean_sample(result, torch.tensor([0.5], device='cuda')) is result
  assert torch.equal(torch.cuda.get_rng_state(), before)
  # Leave all sites masked until the final step; it must call joint sampling.
  monkeypatch.setattr(model, '_ddpm_update', lambda x, t, dt: x)
  def forbidden(*args, **kwargs):
    raise AssertionError('ordinary factorized argmax used for final denoising')
  monkeypatch.setattr(model, 'forward', forbidden)
  result = model.restore_model_and_sample(1)
  assert not result.eq(11).any()


@CUDA
def test_factorized_control_bypasses_head_and_matches_backbone(monkeypatch):
  model = tiny_model().cuda().eval()
  model.structured_sampling_mode = 'factorized'
  def forbidden(*args, **kwargs):
    raise AssertionError('factorized control called CCF')
  monkeypatch.setattr(model.structured_head, 'forward', forbidden)
  baseline_cfg = deepcopy(model.config)
  baseline_cfg.model.structured_decoder.enabled = False
  baseline = diffusion.Diffusion(baseline_cfg, model.tokenizer).cuda().eval()
  baseline.backbone.load_state_dict(model.backbone.state_dict())
  baseline.noise.load_state_dict(model.noise.state_dict())
  torch.manual_seed(43)
  expected = baseline.restore_model_and_sample(4)
  expected_rng = torch.cuda.get_rng_state()
  torch.manual_seed(43)
  assert torch.equal(model.restore_model_and_sample(4), expected)
  assert torch.equal(torch.cuda.get_rng_state(), expected_rng)


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_sampling_restores_modes_and_weights(monkeypatch, training, fails):
  model = tiny_model().train(training)
  before = {k: v.clone() for k, v in model.state_dict().items()}
  def sample(**kwargs):
    assert not model.backbone.training and not model.structured_head.training
    if fails:
      raise RuntimeError('injected sampling failure')
    return torch.tensor([[1]])
  monkeypatch.setattr(model, '_sample', sample)
  if fails:
    with pytest.raises(RuntimeError, match='injected'):
      model.restore_model_and_sample(1)
  else:
    model.restore_model_and_sample(1)
  assert model.structured_head.training == training
  assert model.noise.training == training and not model.backbone.training
  for key, value in before.items():
    torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('predictor,semi_ar,mode', [
  ('ddpm_cache', False, 'structured_joint'), ('analytic', False, 'structured_joint'),
  ('ddpm', True, 'structured_joint'), ('ddpm', False, 'structured_marginal')])
def test_unsupported_generation_modes_fail(predictor, semi_ar, mode):
  cfg = tiny_config()
  cfg.sampling.predictor, cfg.sampling.semi_ar = predictor, semi_ar
  cfg.model.structured_decoder.sampling.mode = mode
  with pytest.raises(ValueError):
    tiny_model(cfg)


@CUDA
@pytest.mark.parametrize('topology,factor', ARMS)
def test_full_checkpoint_reload_without_original_backbone(tmp_path, topology, factor):
  model = tiny_model(tiny_config(topology, factor)).cuda().eval()
  cfg = model.config
  cfg.eval.checkpoint_path = str(tmp_path / 'full.ckpt')
  checkpoint = {
    'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
    'pytorch-lightning_version': L.__version__,
    'loops': {'fit_loop': {'epoch_progress': {'current': {'completed': 0}},
                          'epoch_loop.batch_progress': {'current': {'completed': 0}}}}}
  torch.save(checkpoint, cfg.eval.checkpoint_path)
  restored = main._load_from_checkpoint(cfg, model.tokenizer).cuda().eval()
  for key, value in model.state_dict().items():
    torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
  torch.manual_seed(71)
  expected = model.restore_model_and_sample(4)
  torch.manual_seed(71)
  assert torch.equal(expected, restored.restore_model_and_sample(4))
  for name in ('topology_mode', 'factor_mode'):
    wrong_arm = deepcopy(cfg)
    head = wrong_arm.model.structured_decoder
    head[name] = 'dynamic' if head[name] == 'fixed' else 'fixed'
    with pytest.raises(ValueError, match=name):
      main._load_from_checkpoint(wrong_arm, model.tokenizer)
  # A backbone-only file must never silently produce random-head CCF samples.
  checkpoint['state_dict'] = {
    k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('structured_head.')}
  torch.save(checkpoint, cfg.eval.checkpoint_path)
  with pytest.raises(RuntimeError, match='Missing key'):
    main._load_from_checkpoint(cfg, model.tokenizer)


@pytest.mark.parametrize('arm', ['static_static', 'fixed_dynamic', 'dynamic_fixed', 'dynamic_dynamic'])
def test_four_arm_presets_explicitly_generate_jointly(arm):
  with hydra.initialize_config_dir(
      config_dir=str(Path(main.__file__).parent / 'configs'), version_base=None):
    cfg = hydra.compose(config_name='config', overrides=[f'+experiment=ccf/{arm}'])
  assert cfg.model.structured_decoder.sampling.mode == 'structured_joint'
  assert cfg.sampling.predictor == 'ddpm' and not cfg.sampling.semi_ar
  assert not cfg.eval.compute_generative_perplexity


def test_disabled_perplexity_does_not_download(monkeypatch):
  def forbidden(*args, **kwargs):
    raise AssertionError('disabled evaluation attempted a download')
  monkeypatch.setattr(diffusion.transformers.AutoTokenizer, 'from_pretrained', forbidden)
  model = tiny_model()
  assert model.eval_model_tokenizer is None


@pytest.mark.parametrize('compute_ppl', [False, True])
def test_generation_entrypoint_honors_optional_evaluation(monkeypatch, compute_ppl):
  cfg = tiny_config()
  cfg.eval.compute_generative_perplexity = compute_ppl
  calls = []
  tokenizer = SimpleNamespace(batch_decode=lambda samples: ['fixture'])
  model = SimpleNamespace(
    to=lambda device: model, tokenizer=tokenizer,
    gen_ppl_metric=SimpleNamespace(reset=lambda: None, compute=lambda: 1.0),
    restore_model_and_sample=lambda **kwargs: torch.tensor([[1]]),
    compute_generative_perplexity=lambda text: calls.append(text))
  monkeypatch.setattr(main, '_load_from_checkpoint', lambda **kwargs: model)
  assert main.generate_samples(cfg, SimpleNamespace(info=lambda *args: None), tokenizer) == ['fixture']
  assert calls == ([['fixture']] if compute_ppl else [])
