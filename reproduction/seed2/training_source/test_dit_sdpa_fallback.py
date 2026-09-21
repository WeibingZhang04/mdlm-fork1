"""Focused CPU tests for the optional DiT PyTorch attention fallback."""

from contextlib import contextmanager
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

import torch


@contextmanager
def stub_optional_dependencies():
  """Temporarily provide only the non-attention import surface DiT needs."""
  stubs = {}

  huggingface_stub = types.ModuleType('huggingface_hub')
  huggingface_stub.PyTorchModelHubMixin = object
  stubs['huggingface_hub'] = huggingface_stub

  omegaconf_stub = types.ModuleType('omegaconf')
  omegaconf_stub.OmegaConf = type(
    'OmegaConf', (), {'create': staticmethod(lambda value: value)})
  stubs['omegaconf'] = omegaconf_stub

  einops_stub = types.ModuleType('einops')

  def rearrange_stub(tensor, pattern, **axes):
    if pattern == 'b s (three h d) -> b s three h d':
      batch, sequence, packed = tensor.shape
      three = axes['three']
      heads = axes['h']
      return tensor.reshape(
        batch, sequence, three, heads,
        packed // (three * heads))
    raise AssertionError(f'unsupported test rearrange pattern: {pattern}')

  einops_stub.rearrange = rearrange_stub
  stubs['einops'] = einops_stub
  stubs['flash_attn'] = None

  previous = {name: sys.modules.get(name) for name in stubs}
  sys.modules.update(stubs)
  try:
    yield
  finally:
    for name, old_module in previous.items():
      if old_module is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = old_module


def load_dit_without_optional_dependencies():
  """Load models/dit.py while explicitly simulating absent FlashAttention."""
  with stub_optional_dependencies():
    module_path = Path(__file__).parent / 'models' / 'dit.py'
    spec = importlib.util.spec_from_file_location(
      '_dit_sdpa_fallback_test_module', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DitSdpaFallbackTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.dit = load_dit_without_optional_dependencies()

  def test_import_and_rotary_shape_value_and_gradient(self):
    self.assertFalse(self.dit.FLASH_ATTN_AVAILABLE)
    torch.manual_seed(17)
    batch, sequence, heads, head_dim = 2, 5, 3, 4
    qkv = torch.randn(
      batch, sequence, 3, heads, head_dim,
      requires_grad=True)
    rotary = self.dit.Rotary(head_dim)
    cos, sin = rotary(torch.empty(batch, sequence, head_dim))

    output = self.dit.apply_rotary_pos_emb(qkv, cos, sin)
    reference = qkv * cos + self.dit.rotate_half(qkv) * sin
    self.assertEqual(output.shape, qkv.shape)
    self.assertTrue(torch.allclose(output, reference, atol=1e-7))
    # Rotary caches identity cos/sin for V.
    self.assertTrue(torch.equal(output[:, :, 2], qkv[:, :, 2]))

    output.square().sum().backward()
    self.assertIsNotNone(qkv.grad)
    self.assertGreater(qkv.grad.abs().sum().item(), 0.0)

  def test_crf_package_import_does_not_eagerly_import_ar_backend(self):
    saved_models = {
      name: module for name, module in sys.modules.items()
      if name == 'models' or name.startswith('models.')}
    for name in saved_models:
      sys.modules.pop(name, None)
    try:
      with stub_optional_dependencies():
        package = importlib.import_module('models')
        self.assertNotIn('crf_decoder', vars(package))
        self.assertNotIn('autoregressive', vars(package))
        self.assertIsNotNone(package.crf_decoder)
        self.assertIn('crf_decoder', vars(package))
        self.assertFalse(package.dit.FLASH_ATTN_AVAILABLE)
    finally:
      for name in list(sys.modules):
        if name == 'models' or name.startswith('models.'):
          sys.modules.pop(name, None)
      sys.modules.update(saved_models)

  def test_sdpa_matches_explicit_attention_and_has_gradients(self):
    torch.manual_seed(23)
    batch, sequence, heads, head_dim = 2, 6, 2, 4
    qkv = torch.randn(
      batch, sequence, 3, heads, head_dim,
      requires_grad=True)

    output = self.dit.scaled_dot_product_attention_qkv(qkv)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    weights = torch.softmax(
      torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim),
      dim=-1)
    reference = torch.matmul(weights, v).transpose(1, 2).contiguous()

    self.assertEqual(
      output.shape, (batch, sequence, heads, head_dim))
    self.assertTrue(torch.allclose(output, reference, atol=1e-6))
    output.square().mean().backward()
    self.assertIsNotNone(qkv.grad)
    self.assertGreater(qkv.grad.abs().sum().item(), 0.0)

  def test_ddit_block_runs_end_to_end_on_cpu_without_flash(self):
    torch.manual_seed(29)
    dim, heads, cond_dim = 8, 2, 4
    block = self.dit.DDiTBlock(
      dim=dim, n_heads=heads, cond_dim=cond_dim,
      dropout=0.0)
    # The production zero init closes both residual gates. Open them in this
    # test so gradient flow through SDPA and the MLP is observable.
    with torch.no_grad():
      block.adaLN_modulation.bias[2 * dim:3 * dim].fill_(1.0)
      block.adaLN_modulation.bias[5 * dim:6 * dim].fill_(1.0)

    x = torch.randn(2, 5, dim, requires_grad=True)
    c = torch.randn(2, cond_dim)
    rotary = self.dit.Rotary(dim // heads)
    output = block(x, rotary(x), c, seqlens=None)
    self.assertEqual(output.shape, x.shape)

    output.square().mean().backward()
    self.assertGreater(x.grad.abs().sum().item(), 0.0)
    self.assertGreater(block.attn_qkv.weight.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
  unittest.main()
