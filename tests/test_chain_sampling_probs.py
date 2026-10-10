"""Regression coverage for the CUDA FP64 categorical normalization failure."""
import pytest
import torch

from chain_crf.core import _sampling_probs, sample_chain


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("states", [129, 501, 1001])
def test_padded_point_masses(device, states):
    scores = torch.full((23, states), -torch.inf, dtype=torch.float64)
    scores[:, 0] = -1.35
    actual = _sampling_probs(scores.to(device)).cpu()
    expected = torch.zeros_like(scores)
    expected[:, 0] = 1.
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_normalization_matches_cpu_with_masking_and_strides(device):
    generator = torch.Generator().manual_seed(13)
    scores = torch.randn(23, 2002, generator=generator, dtype=torch.float64) * 70
    scores[:, ::7] = -torch.inf
    scores += torch.linspace(-1e6, 1e6, 23, dtype=torch.float64)[:, None]
    source = scores.to(device)[:, ::2]  # Exercise noncontiguous inputs.
    before = source.clone()
    actual = _sampling_probs(source).cpu()
    torch.testing.assert_close(actual, scores[:, ::2].softmax(-1), rtol=1e-12, atol=0)
    assert torch.equal(source, before)
    assert actual.dtype == torch.float64
    # Retain probabilities that would underflow if normalized in FP32.
    rare = _sampling_probs(torch.tensor([[0., -500.]], device=device)).cpu()
    assert 0 < rare[0, 1] < 1e-200


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_distributions_are_not_repaired(device):
    scores = torch.tensor([[-torch.inf, -torch.inf], [0., torch.nan],
                           [0., torch.inf]], device=device)
    assert not _sampling_probs(scores).isfinite().any()


@pytest.mark.parametrize("device", DEVICES)
def test_backward_sampling_clamped_chain(device):
    # FP64 also forces the reference FFBS path on deployments with fused CUDA.
    states = 1001
    unary = torch.full((2, 3, states), -torch.inf, dtype=torch.float64, device=device)
    expected = torch.tensor([[7, 0, 1000], [0, 9, 0]], device=device)
    unary.scatter_(-1, expected[..., None], -1.35)
    edge = torch.zeros(2, 2, states, states, dtype=torch.float64, device=device)
    assert torch.equal(sample_chain(unary, edge), expected)
