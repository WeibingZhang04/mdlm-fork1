"""CUDA regression checks for exposing the existing DiT hidden states."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _original_forward(model, indices, sigma):
  """Original MDLM forward, before the encode/decode split."""
  x = model.vocab_embed(indices)
  c = F.silu(model.sigma_map(sigma))
  rotary_cos_sin = model.rotary_emb(x)
  with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    for block in model.blocks:
      x = block(x, rotary_cos_sin, c, seqlens=None)
    x = model.output_layer(x, c)
  return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real DiT needs CUDA')
@pytest.mark.parametrize('training', [False, True], ids=['eval', 'train'])
def test_encode_decode_preserves_original_forward_and_backward(training):
  from models.dit import DIT

  torch.manual_seed(37)
  config = SimpleNamespace(model=SimpleNamespace(
    hidden_size=64, cond_dim=32, n_heads=4, n_blocks=2,
    dropout=0.2, scale_by_sigma=True))
  model = DIT(config, vocab_size=23).cuda().train(training)
  # Fresh DiT initializes gates and the vocabulary projection to zero.
  # Make those nonzero so this actually tests attention, dropout and gradients.
  with torch.no_grad():
    for block in model.blocks:
      block.adaLN_modulation.weight.normal_(std=0.03)
      block.adaLN_modulation.bias.normal_(std=0.03)
    model.output_layer.linear.weight.normal_(std=0.03)
    model.output_layer.adaLN_modulation.weight.normal_(std=0.03)
  reference = copy.deepcopy(model)
  split_model = copy.deepcopy(model)
  checkpoint = copy.deepcopy(model.state_dict())
  assert not split_model.load_state_dict(checkpoint, strict=True).missing_keys
  indices = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device='cuda')
  sigma = torch.tensor([0.2, 0.8], device='cuda')
  weight = torch.randn(2, 4, 23, device='cuda')
  rng = torch.cuda.get_rng_state().clone()

  def evaluate(network, operation):
    torch.cuda.set_rng_state(rng)
    logits = operation(network)
    final_rng = torch.cuda.get_rng_state().clone()
    (logits.float() * weight).sum().backward()
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in network.named_parameters()}
    return logits.detach(), gradients, final_rng

  expected, expected_grads, expected_rng = evaluate(
    reference, lambda net: _original_forward(net, indices, sigma))
  actual, actual_grads, actual_rng = evaluate(
    model, lambda net: net(indices, sigma))

  def split_forward(net):
    captured = []
    handle = net.blocks[-1].register_forward_hook(
      lambda module, args, output: captured.append(output))
    hidden, conditioning = net.encode(indices, sigma)
    handle.remove()
    assert hidden is captured[0]
    assert hidden.shape == (2, 4, 64)
    assert conditioning.shape == (2, 32)
    assert hidden.requires_grad
    return net.decode(hidden, conditioning)

  split, split_grads, split_rng = evaluate(split_model, split_forward)
  assert expected.abs().max() > 0
  assert set(actual_grads) == set(expected_grads) == set(split_grads)
  assert model.state_dict().keys() == checkpoint.keys()
  for logits, grads, final_rng in (
      (actual, actual_grads, actual_rng), (split, split_grads, split_rng)):
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert torch.equal(final_rng, expected_rng)
    for name, grad in grads.items():
      assert torch.isfinite(grad).all(), name
      torch.testing.assert_close(
        grad, expected_grads[name], rtol=1e-5, atol=1e-6, msg=name)
