"""Correctness and dispatch checks for inference-only fused forward filtering."""
import pytest
import torch

from chain_crf import core


def reference_forward(unary, edge):
    values = [unary[:, 0]]
    for pos in range(1, unary.shape[1]):
        values.append(unary[:, pos] + torch.logsumexp(
            values[-1].unsqueeze(-1) + edge[:, pos - 1], dim=-2))
    return values


@pytest.fixture
def cuda_forward():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip('NVIDIA CUDA required')
    pytest.importorskip('triton')
    from chain_crf.cuda_forward import fused_forward
    return fused_forward


@pytest.mark.parametrize('batch,length,states', [(1, 2, 1), (3, 17, 3), (2, 33, 64),
                                                (1, 1024, 65), (2, 11, 128)])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'expanded'])
@torch.no_grad()
def test_fused_messages_and_conditionals(cuda_forward, batch, length, states, layout):
    torch.manual_seed(109)
    unary = torch.randn(batch, length, states, device='cuda') - 5
    edge = torch.randn(batch, length - 1, states, states, device='cuda') * .4
    if states > 1:
        # Clamps, forbidden transitions, a residual-like neutral state, and
        # completely unreachable destinations. Keep state zero reachable.
        unary[:, 2::7, 1:] = -torch.inf
        edge[..., -1, :] = 0
        edge[..., :, -1] = 0
        edge[:, ::5, :, -1] = -torch.inf
    if layout == 'strided':
        unary = unary.transpose(1, 2).contiguous().transpose(1, 2)
        edge = edge.transpose(2, 3).contiguous().transpose(2, 3)
    elif layout == 'expanded':
        unary = unary[:1].expand(batch, -1, -1)
        edge = edge[:1].expand(batch, -1, -1, -1)
    actual = cuda_forward(unary, edge)
    expected = torch.stack(reference_forward(unary, edge), dim=1)
    assert actual is not None
    assert torch.equal(actual.isneginf(), expected.isneginf())
    assert not actual.isnan().any()
    torch.testing.assert_close(actual, expected, atol=.002, rtol=2e-6)
    # Also compare probabilities, so large accumulated log offsets cannot
    # hide an error that would affect backward categorical sampling.
    torch.testing.assert_close(actual.double().softmax(-1), expected.double().softmax(-1),
                               atol=.0005, rtol=.002)


@torch.no_grad()
def test_all_impossible_chain_stays_negative_infinity(cuda_forward):
    unary = torch.zeros(2, 6, 3, device='cuda')
    edge = torch.zeros(2, 5, 3, 3, device='cuda')
    unary[0, 2] = -torch.inf
    edge[1, 3] = -torch.inf
    actual = cuda_forward(unary, edge)
    expected = torch.stack(reference_forward(unary, edge), dim=1)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 2:].isneginf().all()
    assert actual[1, 4:].isneginf().all()


def test_training_and_fp64_use_reference(cuda_forward, monkeypatch):
    import chain_crf.cuda_forward as module
    unary = torch.randn(2, 4, 3, device='cuda', requires_grad=True)
    edge = torch.randn(2, 3, 3, 3, device='cuda', requires_grad=True)
    assert cuda_forward(unary, edge) is None
    with torch.no_grad():
        assert cuda_forward(unary.double(), edge.double()) is None
        assert cuda_forward(unary[:, :1], edge[:, :0]) is None
        assert cuda_forward(torch.zeros(1, 2, 129, device='cuda'),
                            torch.zeros(1, 1, 129, 129, device='cuda')) is None
    def unexpected(*args):
        raise AssertionError('Fused inference must not run during training')
    monkeypatch.setattr(module, 'fused_forward', unexpected)
    actual = core.chain_log_partition(unary, edge)
    expected = reference_forward(unary, edge)[-1].logsumexp(-1)
    actual_grad = torch.autograd.grad(actual.sum(), (unary, edge))
    expected_grad = torch.autograd.grad(expected.sum(), (unary, edge))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


def test_cpu_and_missing_triton_fallback(monkeypatch):
    import chain_crf.cuda_forward as module
    unary = torch.randn(2, 4, 3)
    edge = torch.randn(2, 3, 3, 3)
    monkeypatch.setattr(module, 'triton', None)
    with torch.no_grad():
        assert module.fused_forward(unary, edge) is None
        torch.testing.assert_close(torch.stack(core._forward(unary, edge)),
                                   torch.stack(reference_forward(unary, edge)))


@torch.no_grad()
def test_fused_partition_marginals_and_joint_samples(cuda_forward, monkeypatch):
    unary = torch.zeros(1, 2, 2, device='cuda')
    edge = torch.tensor([[[[2., -1.], [-1., 2.]]]], device='cuda')
    actual_z = core.chain_log_partition(unary, edge)
    actual_marginals = core.chain_marginals(unary, edge)
    with monkeypatch.context() as patch:
        patch.setattr(core, '_forward', reference_forward)
        torch.testing.assert_close(actual_z, core.chain_log_partition(unary, edge))
        torch.testing.assert_close(actual_marginals, core.chain_marginals(unary, edge))
    # Check the joint distribution, not merely the uniform node marginals.
    draws = core.sample_chain(unary.expand(12000, -1, -1), edge.expand(12000, -1, -1, -1),
                              generator=torch.Generator(device='cuda').manual_seed(41))
    histogram = torch.bincount(draws[:, 0] * 2 + draws[:, 1], minlength=4) / 12000
    expected = edge.flatten().softmax(0)
    torch.testing.assert_close(histogram, expected, atol=.02, rtol=0)
