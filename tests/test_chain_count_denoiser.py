"""Exact toy-law checks for the frozen count adapter, no model download."""

import hashlib
import itertools

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf.count_denoiser import CountCRFDenoiser


def make(mode="conditional", strength=.1, backend="segments", chunk=3, mask_id=3):
    head = CountBigramHead(4, smoothing=.2).fit([[0, 1, 0, 1, 2], [2, 2, 1], [0, 1]])
    return CountCRFDenoiser(head, mask_id=mask_id, mode=mode, strength=strength,
                           backend=backend, sparse_chunk_size=chunk)


def inputs():
    logs = torch.randn(4, 3, 4, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(91)).log_softmax(-1)
    xt = torch.tensor([[3, 3, 3], [0, 3, 2], [2, 1, 0], [3, 1, 3]])
    return logs, xt


def enumeration(adapter, logs, xt):
    unary = adapter._unary(logs, xt)
    batch, length, vocab = unary.shape
    states = torch.tensor(list(itertools.product(range(vocab), repeat=length)), device=logs.device)
    p = adapter.potential
    pair = (p.left[:, None] * p.right[None, :] + p.sparse.to_dense()).log() + p.log_scale
    scores = unary[:, torch.arange(length, device=logs.device)[None, :], states].sum(-1)
    scores += pair[states[:, :-1], states[:, 1:]].sum(-1)[None, :]
    probs = (scores - scores.logsumexp(-1)[:, None]).exp()
    marginal = torch.zeros_like(unary)
    for pos in range(length):
        marginal[:, pos].scatter_add_(1, states[:, pos].expand(batch, -1), probs)
    return states, probs, marginal


@pytest.mark.parametrize("mode", ["conditional", "pmi"])
@pytest.mark.parametrize("strength", [.025, .1, .25, 1.])
@pytest.mark.parametrize("backend", ["reference", "gpu", "segments"])
def test_marginals_equal_dense_enumeration(mode, strength, backend):
    adapter = make(mode, strength, backend)
    logs, xt = inputs()
    original = logs.clone()
    _, _, expected = enumeration(adapter, logs, xt)
    actual = adapter(logs, xt, torch.ones(4, 1)).exp()
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    assert torch.equal(logs, original)
    assert torch.equal(actual[..., 3], torch.zeros_like(actual[..., 3]))
    assert torch.equal(actual[xt != 3].argmax(-1), xt[xt != 3])
    assert torch.equal(actual[xt != 3].max(-1).values, torch.ones_like(actual[xt != 3].max(-1).values))


def test_conditional_is_normalized_conditional_not_pmi():
    conditional, pmi = make(strength=1.), make(mode="pmi", strength=1.)
    p = conditional.potential
    matrix = (p.left[:, None] * p.right[None, :] + p.sparse.to_dense()) * torch.exp(torch.tensor(p.log_scale))
    torch.testing.assert_close(matrix.sum(-1), torch.ones(4, dtype=torch.float64), atol=1e-7, rtol=1e-7)
    logs, xt = inputs()
    assert not torch.allclose(conditional(logs, xt).exp(), pmi(logs, xt).exp())
    assert conditional.identity()["mode"] == "conditional"
    assert pmi.identity()["mode"] == "pmi"


@pytest.mark.parametrize("backend", ["reference", "gpu", "segments"])
def test_zero_strength_preserves_official_log_tensor_loss_and_rng(backend):
    adapter = make(strength=0., backend=backend)
    logs, xt = inputs()
    # Literal SUBS-like finite sentinel, with exact visible-token log mass zero.
    logs[..., 3] = -1e6
    logs -= logs.logsumexp(-1, keepdim=True)
    logs[xt != 3] = -1e6
    logs.scatter_(-1, xt[..., None], torch.where(xt.ne(3), 0., logs.gather(-1, xt[..., None]).squeeze(-1))[..., None])
    truth = torch.tensor([[1, 2, 0], [0, 1, 2], [2, 1, 0], [2, 1, 1]])
    sigma = torch.tensor([.05, .3, 1., 2.], dtype=torch.float64)
    dsigma = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    result = adapter(logs, xt, sigma)
    assert result is logs
    assert torch.equal(before, torch.random.get_rng_state())
    baseline = -logs.gather(-1, truth[..., None]).squeeze(-1) * (dsigma / sigma.expm1())[:, None]
    changed = -result.gather(-1, truth[..., None]).squeeze(-1) * (dsigma / sigma.expm1())[:, None]
    assert torch.equal(baseline, changed)
    assert torch.equal(changed[xt != 3], torch.zeros_like(changed[xt != 3]))


def test_zero_strength_sampling_law_is_original_clean_independence():
    adapter = make(strength=0.)
    logs, xt = inputs()
    _, _, expected = enumeration(adapter, logs, xt)
    independent = adapter._unary(logs, xt).softmax(-1)
    torch.testing.assert_close(expected, independent, atol=2e-15, rtol=2e-15)
    selected = torch.tensor([[True, False, True]]).expand(4, -1)
    values = torch.tensor([[0, 1, 2], [0, 0, 2], [2, 1, 0], [1, 1, 2]])
    expected_log = independent.log().gather(-1, values[..., None]).squeeze(-1).masked_fill(~selected, 0.).sum(-1)
    torch.testing.assert_close(adapter.selected_log_probability(logs, xt, selected, values), expected_log,
                               atol=2e-15, rtol=2e-15)


@pytest.mark.parametrize("backend", ["reference", "gpu", "segments"])
@pytest.mark.parametrize("mode", ["conditional", "pmi"])
def test_selected_block_probability_is_joint_marginal_not_product(backend, mode):
    adapter = make(mode, 1., backend)
    logs, xt = inputs()
    states, probs, marginal = enumeration(adapter, logs, xt)
    selected = torch.tensor([[True, False, True], [False, True, True], [True, True, True], [True, False, True]])
    values = torch.tensor([[0, 0, 2], [0, 1, 2], [2, 1, 0], [1, 1, 2]])
    matching = ((states[None, :, :] == values[:, None, :]) | ~selected[:, None, :]).all(-1)
    expected = probs.masked_fill(~matching, 0.).sum(-1).log()
    actual = adapter.selected_log_probability(logs, xt, selected, values)
    torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)
    product = marginal.gather(-1, values[..., None]).squeeze(-1).log().masked_fill(~selected, 0.).sum(-1)
    assert not torch.isclose(actual[0], product[0], atol=1e-5, rtol=1e-5)
    assert torch.equal(adapter.selected_log_probability(logs, xt, torch.zeros_like(selected), values), torch.zeros(4, dtype=torch.float64))
    values[0, 0] = 3  # Selecting the mask is impossible.
    values[2, 1] = 0  # Contradicts an already visible token.
    impossible = adapter.selected_log_probability(logs, xt, selected, values)
    assert torch.isneginf(impossible[[0, 2]]).all()
    torch.testing.assert_close(impossible[[1, 3]], actual[[1, 3]], atol=0., rtol=0.)


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_empirical_proposal_and_external_reveal_mask(sampling):
    adapter = make(strength=1., chunk=4096)
    # Strongly non-independent two-token model, and one visible context token.
    logs = torch.tensor([[[0., -.3, -1., -1e6], [-.2, .3, -.4, -1e6], [.1, .2, -.1, -1e6]]], dtype=torch.float64)
    xt = torch.tensor([[3, 3, 2]])
    states, probs, marginal = enumeration(adapter, logs, xt)
    n = 16000
    draws = adapter.propose(logs.expand(n, -1, -1), xt.expand(n, -1),
                            torch.Generator().manual_seed(315), sampling=sampling)
    frequency = torch.bincount(draws[:, 0] * 4 + draws[:, 1], minlength=16).double() / n
    if sampling == "joint":
        expected = torch.zeros(16, dtype=torch.float64).scatter_add_(0, states[:, 0] * 4 + states[:, 1], probs[0])
    else:
        expected = (marginal[0, 0, :, None] * marginal[0, 1, None, :]).reshape(-1)
    torch.testing.assert_close(frequency, expected, atol=.012, rtol=0.)
    assert torch.equal(draws[:, 2], torch.full((n,), 2))
    reveal = torch.tensor([[True, False, True]]).expand(n, -1)
    output = adapter.reveal(logs.expand(n, -1, -1), xt.expand(n, -1), reveal,
                            torch.Generator().manual_seed(315), sampling=sampling)
    assert torch.equal(output[:, 0], draws[:, 0])
    assert torch.equal(output[:, 1], torch.full((n,), 3))
    assert torch.equal(output[:, 2], torch.full((n,), 2))


def test_shift_time_dtype_and_nonlast_mask():
    adapter = make(mask_id=1)
    logs, _ = inputs()
    xt = torch.tensor([[1, 1, 1], [0, 1, 2], [2, 3, 0], [1, 0, 1]])
    expected = adapter(logs, xt, torch.ones(4))
    actual = adapter(logs + torch.tensor([100., -20., 5.])[None, :, None], xt, torch.zeros(4))
    torch.testing.assert_close(actual.exp(), expected.exp(), atol=1e-14, rtol=1e-14)
    assert torch.isneginf(actual[..., 1]).all()
    assert adapter(logs.float(), xt).dtype == torch.float32
    assert not expected.requires_grad
    length_one = adapter(logs[:, :1], xt[:, :1]).exp()
    torch.testing.assert_close(length_one, adapter._unary(logs[:, :1], xt[:, :1]).softmax(-1), atol=1e-15, rtol=1e-15)


def test_file_factory_identity_and_default_conditional(tmp_path):
    path = tmp_path / "counts.pt"
    head = CountBigramHead(4, mode="pmi", strength=7.).fit([[0, 1, 2], [0, 1]])
    head.save(path)
    adapter = CountCRFDenoiser.from_file(path, mask_id=3)
    identity = adapter.identity()
    assert identity["mode"] == "conditional" and identity["strength"] == 1.
    assert identity["counts_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["sparse_chunk_size"] == 512
    assert len(identity["source_sha256"]) == 5
    assert all(len(value) == 64 for value in identity["source_sha256"].values())
    assert all(not key.startswith("/") for key in identity["source_sha256"])
    other = CountCRFDenoiser.from_file(path, mask_id=3, mode="pmi", strength=.25,
                                      backend="reference", sparse_chunk_size=8)
    assert other.identity()["mode"] == "pmi" and other.identity()["strength"] == .25


@pytest.mark.parametrize("kwargs", [{"backend": "unknown"}, {"mode": "logits"}, {"strength": -1.},
                                    {"strength": float("nan")}, {"chunk": 0}, {"mask_id": 4}])
def test_invalid_model_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        make(**kwargs)


def test_invalid_inputs_and_probabilities_fail():
    adapter = make()
    logs, xt = inputs()
    with pytest.raises(ValueError):
        adapter(logs[..., :2], xt)
    with pytest.raises(ValueError):
        adapter(logs, xt.int())
    with pytest.raises(ValueError):
        adapter(logs, xt + 10)
    with pytest.raises(ValueError):
        adapter.reveal(logs, xt, torch.ones_like(xt))
    with pytest.raises(ValueError):
        adapter.propose(logs, xt, sampling="argmax")
    logs[0, 0] = -torch.inf
    with pytest.raises((ValueError, FloatingPointError)):
        adapter(logs, xt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_against_cpu_dense_marginals_and_clamps():
    cpu = make()
    head = CountBigramHead(4, smoothing=.2).fit([[0, 1, 0, 1, 2], [2, 2, 1], [0, 1]]).cuda()
    gpu = CountCRFDenoiser(head, mask_id=3, strength=.1, sparse_chunk_size=3)
    logs, xt = inputs()
    expected = cpu(logs, xt)
    torch.testing.assert_close(gpu(logs.cuda(), xt.cuda()).cpu(), expected, atol=3e-13, rtol=3e-13)
    draw = gpu.propose(logs.cuda(), xt.cuda()).cpu()
    assert torch.equal(draw[xt != 3], xt[xt != 3])
    assert draw.ne(3).all()
