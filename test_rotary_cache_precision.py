"""Regression checks for explicit rotary precision; no training/data files used."""
import os
from types import SimpleNamespace
import unittest
import torch
from test_dit_sdpa_fallback import load_dit_without_optional_dependencies


class RotaryPrecisionTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dit = load_dit_without_optional_dependencies()
    if os.environ.get('ROTARY_REQUIRE_CUDA') == '1':
      assert torch.cuda.is_available(), 'CUDA test allocation required'
    cls.devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

  def test_context_and_first_call_cannot_change_precision(self):
    for device in self.devices:
      for precision, dtype in [('bf16', torch.bfloat16), ('fp32', torch.float32)]:
        x = torch.empty(1, 1024, 64, device=device)
        probe_first = self.dit.Rotary(64, cache_precision=precision).to(device)
        train_first = self.dit.Rotary(64, cache_precision=precision).to(device)
        expected = probe_first(x)
        with torch.autocast(device, dtype=torch.bfloat16):
          actual = train_first(x)
          reused = probe_first(x)
        for a, b, c in zip(expected, actual, reused):
          self.assertEqual(a.dtype, dtype)
          self.assertTrue(torch.equal(a, b))
          self.assertIs(a, c)
        self.assertTrue(torch.all(expected[0][:, :, 2] == 1))
        self.assertTrue(torch.all(expected[1][:, :, 2] == 0))

  def test_cuda_matches_historical_bf16_and_fp32_computation(self):
    if not torch.cuda.is_available():
      self.skipTest('CUDA numerical reference requires a GPU')
    x = torch.empty(1, 1024, 64, device='cuda')
    for precision in ['bf16', 'fp32']:
      rotary = self.dit.Rotary(64, cache_precision=precision).cuda()
      with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
        t = torch.arange(1024, device='cuda').type_as(rotary.inv_freq)
        freqs = torch.einsum('i,j->ij', t, rotary.inv_freq.clone())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        cos[:, :, 2].fill_(1); sin[:, :, 2].fill_(0)
      actual = rotary(x)
      self.assertTrue(torch.equal(actual[0], cos), precision)
      self.assertTrue(torch.equal(actual[1], sin), precision)

  def test_cache_rebuilds_for_policy_length_and_device(self):
    r = self.dit.Rotary(64)
    x = torch.empty(1, 1024, 64)
    old = r(x)[0]
    self.assertEqual(old.dtype, torch.bfloat16)
    r.cache_precision = 'fp32'
    new = r(x)[0]
    self.assertEqual(new.dtype, torch.float32)
    self.assertIsNot(old, new)
    self.assertFalse(torch.equal(old, new.to(torch.bfloat16)))
    self.assertEqual(r(x[:, :32])[0].shape[1], 32)
    if torch.cuda.is_available():
      self.assertEqual(r.cuda()(x[:, :32].cuda())[0].device.type, 'cuda')
    self.assertEqual(list(r.state_dict()), ['inv_freq'])

  def test_config_wiring_default_and_invalid_value(self):
    fields = dict(hidden_size=8, cond_dim=4, n_heads=2, n_blocks=0,
                  dropout=0., scale_by_sigma=True)
    for precision in [None, 'bf16', 'fp32']:
      model = SimpleNamespace(**fields)
      if precision is not None: model.rotary_cache_precision = precision
      dit = self.dit.DIT(SimpleNamespace(model=model), vocab_size=16)
      self.assertEqual(dit.rotary_emb.cache_precision, precision or 'bf16')
    with self.assertRaisesRegex(ValueError, 'bf16 or fp32'):
      self.dit.Rotary(64, cache_precision='fp16')


if __name__ == '__main__':
  unittest.main()
