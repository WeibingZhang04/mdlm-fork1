"""Opt-in CUDA comparison against the pinned, real Hugging Face MDLM release.

Set MDLM_RELEASE_SNAPSHOT and MDLM_TOKENIZER_SNAPSHOT to local snapshot dirs.
No downloads, model mocks, production edits, or training are performed.
"""
import gc
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='module')
def released_models(tmp_path_factory):
  if not torch.cuda.is_available():
    pytest.skip('released backbone comparison requires CUDA')
  snapshot = os.environ.get('MDLM_RELEASE_SNAPSHOT')
  tokenizer_path = os.environ.get('MDLM_TOKENIZER_SNAPSHOT')
  if not snapshot or not tokenizer_path:
    pytest.skip('set MDLM_RELEASE_SNAPSHOT and MDLM_TOKENIZER_SNAPSHOT')
  from transformers import AutoModelForMaskedLM, AutoTokenizer
  import hydra
  import main  # Registers the repository Hydra resolvers.
  from diffusion import Diffusion
  from scripts.prepare_released_mdlm_owt import convert_release

  directory = tmp_path_factory.mktemp('released-mdlm')
  wrapper = directory / 'backbone.pt'
  convert_release(Path(snapshot) / 'model.safetensors', wrapper)
  root = Path(__file__).resolve().parents[1]
  with hydra.initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
    config = hydra.compose(config_name='config', overrides=[
      '+experiment=ccf/static_static',
      f'model.structured_decoder.training.backbone_checkpoint={wrapper}',
      f'eval.gen_ppl_eval_model_name_or_path={tokenizer_path}',
    ])
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
  local = Diffusion(config, tokenizer).cuda().eval()
  reference, info = AutoModelForMaskedLM.from_pretrained(
    snapshot, trust_remote_code=True, local_files_only=True,
    output_loading_info=True)
  for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
    assert not info.get(key), (key, info[key])
  reference = reference.cuda().eval().requires_grad_(False)
  assert not reference.config.time_conditioning
  assert not local.time_conditioning
  print('ENVIRONMENT', torch.__version__, torch.cuda.get_device_name(0), flush=True)
  yield local, reference
  del local, reference
  wrapper.unlink()
  gc.collect()
  torch.cuda.empty_cache()


def inputs(local, batch, length, pattern):
  rng = torch.Generator().manual_seed(71 + length)
  tokens = torch.randint(0, local.mask_index, (batch, length), generator=rng)
  if pattern == 'all_masked':
    tokens.fill_(local.mask_index)
  elif pattern == 'mixed':
    tokens[:, ::3] = local.mask_index
  sigma = torch.linspace(0.2, 1.7, batch, device='cuda')[:, None]
  return tokens.cuda(), sigma


def compare(actual, expected, label):
  delta = (actual.float() - expected.float()).abs()
  print(f'{label}: max_abs={delta.max().item():.9g}, '
        f'mean_abs={delta.mean().item():.9g}, dtype={actual.dtype}', flush=True)
  assert torch.isfinite(actual).all()
  assert torch.isfinite(expected).all()
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_released_weights_are_exactly_equal(released_models):
  local, reference = released_models
  actual = local.backbone.state_dict()
  expected = reference.backbone.state_dict()
  assert actual.keys() == expected.keys()
  for name in actual:
    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
  assert all(not p.requires_grad for p in local.backbone.parameters())
  assert not local.backbone.training


@pytest.mark.parametrize('outer_bf16', [False, True], ids=['sampling_context', 'training_bf16'])
@pytest.mark.parametrize('batch,length,pattern', [
  (2, 32, 'mixed'), (1, 128, 'all_masked'),
  (1, 1024, 'mixed'), (1, 32, 'unmasked'),
])
@torch.no_grad()
def test_same_precision_hidden_states_and_logits(
    released_models, outer_bf16, batch, length, pattern):
  local, reference = released_models
  tokens, sigma = inputs(local, batch, length, pattern)
  with torch.cuda.amp.autocast(enabled=outer_bf16, dtype=torch.bfloat16):
    expected = reference(tokens, sigma.squeeze(-1), output_hidden_states=True,
                         return_dict=True)
    hidden, conditioning = local.backbone.encode(tokens, local._process_sigma(sigma))
    logits = local.backbone.decode(hidden, conditioning)
    direct = local.backbone(tokens, local._process_sigma(sigma))
    helper_hidden, helper_logits = local._structured_backbone_output(tokens, sigma)
  compare(hidden, expected.hidden_states[-1], 'hidden')
  compare(logits, expected.logits, 'raw logits')
  compare(direct, expected.logits, 'local forward')
  compare(helper_hidden, hidden, 'CCF hidden')
  compare(helper_logits[..., :local.mask_index], expected.logits[..., :local.mask_index].float(),
          'CCF clean-token logits')
  assert torch.isneginf(helper_logits[..., local.mask_index]).all()
  assert not helper_hidden.requires_grad and not helper_logits.requires_grad


@pytest.mark.parametrize('outer_bf16', [False, True])
@torch.no_grad()
def test_actual_paths_ignore_noise_conditioning(released_models, outer_bf16):
  local, reference = released_models
  tokens, sigma = inputs(local, 2, 32, 'mixed')
  with torch.cuda.amp.autocast(enabled=outer_bf16, dtype=torch.bfloat16):
    _, first = local._structured_backbone_output(tokens, sigma)
    _, second = local._structured_backbone_output(tokens, sigma + 3)
    hf_first = reference(tokens, sigma.squeeze(-1))
    hf_second = reference(tokens, sigma.squeeze(-1) + 3)
  compare(first[..., :local.mask_index], second[..., :local.mask_index], 'CCF time invariance')
  compare(hf_first, hf_second, 'HF time invariance')


@torch.no_grad()
def test_sampling_vs_training_precision_backbone_outputs(released_models):
  """Detect any difference caused by the actual outer precision contexts.

  This is intentionally separate from same-precision implementation equivalence.
  Exact equality is a hypothesis under test, not an assumed BF16 tolerance.
  """
  local, reference = released_models
  tokens, sigma = inputs(local, 2, 32, 'mixed')
  with torch.cuda.amp.autocast(enabled=False):
    expected = reference(tokens, sigma.squeeze(-1)).float()
  with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    _, actual = local._structured_backbone_output(tokens, sigma)
  expected = expected[..., :local.mask_index]
  actual = actual[..., :local.mask_index]
  logp_expected = expected.log_softmax(-1)
  logp_actual = actual.log_softmax(-1)
  print('CROSS_PRECISION', {
    'logit_max_abs': (actual - expected).abs().max().item(),
    'log_probability_max_abs': (logp_actual - logp_expected).abs().max().item(),
    'probability_max_abs': (logp_actual.exp() - logp_expected.exp()).abs().max().item(),
    'argmax_disagreements': (actual.argmax(-1) != expected.argmax(-1)).sum().item(),
  }, flush=True)
  compare(actual, expected, 'sampling versus training raw logits')
