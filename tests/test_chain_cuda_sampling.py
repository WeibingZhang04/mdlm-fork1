"""Check fused backward sampling against fixed-noise and exact-law oracles."""
import itertools

import pytest
import torch

from chain_crf import core
from chain_crf import cuda_sampling
from chain_crf.segments import sample_segmented_chain


@pytest.fixture
def cuda():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip('NVIDIA CUDA required')
    pytest.importorskip('triton')
    return 'cuda'


def noise_oracle(alpha, edge, log_noise):
    """Independent sequential implementation with identical supplied noise."""
    alpha, edge, log_noise = alpha.cpu(), edge.cpu(), log_noise.cpu()
    result = torch.empty(alpha.shape[:2], dtype=torch.long)
    result[:, -1] = (alpha[:, -1].double() - log_noise[:, -1]).argmax(-1)
    for pos in range(alpha.shape[1] - 2, -1, -1):
        column = edge[:, pos].gather(-1, result[:, pos + 1, None, None].expand(-1, alpha.shape[-1], 1)).squeeze(-1)
        scores = (alpha[:, pos] + column).double() - log_noise[:, pos]
        result[:, pos] = scores.argmax(-1)
    return result


@pytest.mark.parametrize('length,states', [(1, 1), (1, 65), (5, 3), (9, 64), (1024, 65), (7, 128)])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'expanded'])
@torch.no_grad()
def test_fixed_noise_matches_sequential_oracle(cuda, length, states, layout):
    torch.manual_seed(17)
    unary = torch.randn(3, length, states, device=cuda)
    edge = torch.randn(3, length - 1, states, states, device=cuda) * .7
    if states > 1:
        unary[:, 1::4, 1:] = -torch.inf
        edge[:, ::3, :, -1] = -torch.inf
    if layout == 'strided':
        edge = edge.transpose(2, 3).contiguous().transpose(2, 3)
    elif layout == 'expanded':
        edge = edge[:1].expand(3, -1, -1, -1)
    alpha = torch.stack(core._forward(unary, edge), dim=1)
    noise = torch.empty(alpha.shape, dtype=torch.double, device=cuda).exponential_().log_()
    expected = noise_oracle(alpha, edge, noise)
    actual = cuda_sampling._sample_with_log_noise(alpha, edge, noise)
    assert torch.equal(actual.cpu(), expected)
    if states > 1:
        assert actual[:, 1::4].eq(0).all()


@torch.no_grad()
def test_fp64_noise_comparison_can_select_rare_state(cuda):
    # A 1e-13-probability state wins under this deliberately supplied rare
    # event. Converting the perturbation to FP32 erases the winning margin.
    alpha = torch.tensor([[[0., -30., -torch.inf]]], device=cuda)
    noise = torch.tensor([[[0., -30. - 2**-30, -1000.]]], dtype=torch.double, device=cuda)
    edge = torch.empty(1, 0, 3, 3, device=cuda)
    result = cuda_sampling._sample_with_log_noise(alpha, edge, noise)
    assert result.item() == 1


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
@torch.no_grad()
def test_invalid_distributions_raise(cuda, invalid):
    alpha = torch.full((2, 3, 3), invalid, device=cuda)
    edge = torch.zeros(2, 2, 3, 3, device=cuda)
    with pytest.raises(RuntimeError, match='Invalid CRF sampling distribution'):
        cuda_sampling._sample_with_log_noise(alpha, edge, torch.zeros_like(alpha, dtype=torch.double))


@torch.no_grad()
def test_explicit_generator_reproducible_and_advances(cuda):
    unary = torch.zeros(4, 12, 5, device=cuda)
    edge = torch.zeros(4, 11, 5, 5, device=cuda)
    rng = torch.Generator(device=cuda).manual_seed(92)
    state_before = rng.get_state().clone()
    a = core.sample_chain(unary, edge, rng)
    assert not torch.equal(state_before, rng.get_state())
    b = core.sample_chain(unary, edge, rng)
    assert not torch.equal(a, b)
    rng.manual_seed(92)
    assert torch.equal(a, core.sample_chain(unary, edge, rng))
    torch.manual_seed(77)
    c = core.sample_chain(unary, edge)
    torch.manual_seed(77)
    assert torch.equal(c, core.sample_chain(unary, edge))


@pytest.mark.parametrize('segmented', [False, True])
@torch.no_grad()
def test_joint_law_matches_enumeration_with_clamps_and_forbidden_edges(cuda, segmented):
    # Four positions, middle visible token, asymmetric pair scores, one
    # forbidden transition, and a neutral residual-like state at index 2.
    unary = torch.tensor([[[.2, -.5, -.1], [0., -torch.inf, -torch.inf],
                            [-.4, .1, -.2], [.3, -.2, -.5]]], device=cuda)
    edge = torch.tensor([[[[.8, -.4, 0.], [-.7, .9, 0.], [0., 0., 0.]],
                           [[-.2, .7, 0.], [.9, -.4, 0.], [0., 0., 0.]],
                           [[.4, -torch.inf, 0.], [-.6, .8, 0.], [0., 0., 0.]]]], device=cuda)
    paths = torch.tensor(list(itertools.product(range(3), repeat=3)), device=cuda)
    rows = torch.stack((paths[:, 0], torch.zeros_like(paths[:, 0]), paths[:, 1], paths[:, 2]), dim=1)
    expected = core.chain_log_prob(unary.expand(27, -1, -1), edge.expand(27, -1, -1, -1), rows).exp()
    n = 30000
    u, e = unary.expand(n, -1, -1), edge.expand(n, -1, -1, -1)
    rng = torch.Generator(device=cuda).manual_seed(811)
    if segmented:
        mask = torch.tensor([[True, False, True, True]], device=cuda).expand(n, -1)
        draws = sample_segmented_chain(u, e, mask, rng)
    else:
        draws = core.sample_chain(u, e, rng)
    assert draws[:, 1].eq(0).all()
    assert not ((draws[:, 2] == 0) & (draws[:, 3] == 1)).any()
    codes = draws[:, 0] * 9 + draws[:, 2] * 3 + draws[:, 3]
    frequencies = torch.bincount(codes, minlength=27) / n
    torch.testing.assert_close(frequencies, expected, atol=.012, rtol=0)


@torch.no_grad()
def test_fallback_does_not_consume_rng(cuda, monkeypatch):
    rng = torch.Generator(device=cuda).manual_seed(44)
    before = rng.get_state().clone()
    for states, dtype in [(3, torch.double), (129, torch.float32)]:
        messages = [torch.zeros(2, states, dtype=dtype, device=cuda)] * 3
        edge = torch.zeros(2, 2, states, states, dtype=dtype, device=cuda)
        assert cuda_sampling.fused_backward_sample(messages, edge, rng) is None
    monkeypatch.setattr(cuda_sampling, 'triton', None)
    assert cuda_sampling.fused_backward_sample([torch.zeros(2, 3, device=cuda)],
                                               torch.empty(2, 0, 3, 3, device=cuda), rng) is None
    assert torch.equal(before, rng.get_state())


def test_cpu_and_autograd_fallback_preserves_generator():
    rng = torch.Generator().manual_seed(44)
    before = rng.get_state().clone()
    messages = [torch.zeros(2, 3)] * 2
    edge = torch.zeros(2, 1, 3, 3)
    assert cuda_sampling.fused_backward_sample(messages, edge, rng) is None
    with torch.no_grad():
        assert cuda_sampling.fused_backward_sample(messages, edge, rng) is None
    assert torch.equal(before, rng.get_state())
