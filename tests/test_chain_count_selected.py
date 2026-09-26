"""Selected-block laws checked by enumerating every internal categorical draw."""

import itertools

import pytest
import torch

from chain_crf.count_selected import SelectedCountCRFDenoiser, sample_selected_chain
from chain_crf.count_denoiser import CountCRFDenoiser
from chain_crf import count_selected, sparse_count_segments as segmented
from chain_crf.counts import CountBigramHead


def model(mode="conditional", strength=.1):
    head = CountBigramHead(3, mode=mode, strength=strength, smoothing=.2)
    head.fit([[0, 1, 0, 2, 0], [1, 2, 2], [2, 1]])
    return segmented.SegmentedCountPotential.from_head(head)


def exact_law(unary, potential, selected, sampling):
    """Dense enumeration is only for tiny CPU fixtures, never production."""
    length, vocab = unary.shape[1:]
    states = torch.tensor(list(itertools.product(range(vocab), repeat=length)))
    weights = (potential.left[:, None] * potential.right[None] + potential.sparse.to_dense())
    scores = unary[0, torch.arange(length)[None], states].sum(-1)
    scores += (weights.log()[states[:, :-1], states[:, 1:]] + potential.log_scale).sum(-1)
    probs = scores.softmax(0)
    sites = selected[0].nonzero().flatten()
    index_weights = vocab ** torch.arange(len(sites) - 1, -1, -1)
    result = torch.zeros(vocab ** len(sites), dtype=torch.float64)
    if sampling == "joint":
        result.scatter_add_(0, (states[:, sites] * index_weights).sum(-1), probs)
    else:
        marginal = torch.zeros(length, vocab, dtype=torch.float64)
        for pos in range(length):
            marginal[pos].scatter_add_(0, states[:, pos], probs)
        choices = torch.tensor(list(itertools.product(range(vocab), repeat=len(sites))))
        result = marginal[sites[None], choices].prod(-1)
    return result


def enumerate_sampler(unary, potential, selection, sampling, monkeypatch):
    """Enumerate algorithm branches, multiplying its actual normalized weights.

    Stopping at the next categorical call discovers the branching tree. This
    checks the exact distribution, not approximate sampling frequencies or
    seed-by-seed equality with a different sampling algorithm.
    """
    vocab = unary.shape[-1]
    sites = selection[0].nonzero().flatten()
    result = torch.zeros(vocab ** len(sites), dtype=torch.float64)
    stack = [()]

    class NextDraw(Exception):
        pass

    while stack:
        prefix = stack.pop()
        calls, mass = 0, 1.

        def draw(probability, num_samples, *, generator=None):
            nonlocal calls, mass
            assert probability.shape == (1, vocab) and num_samples == 1
            if calls == len(prefix):
                for token in range(vocab):
                    if probability[0, token] > 0:
                        stack.append(prefix + (token,))
                raise NextDraw
            token = prefix[calls]
            mass *= float(probability[0, token] / probability.sum())
            calls += 1
            return torch.tensor([[token]])

        with monkeypatch.context() as patch:
            patch.setattr(torch, "multinomial", draw)
            try:
                tokens = sample_selected_chain(unary, potential, selection, sampling=sampling,
                                                max_chunk_tokens=1)
            except NextDraw:
                continue
        assert calls == len(prefix)
        assert (tokens[~selection] == -1).all()
        index = int((tokens[0, sites] * (vocab ** torch.arange(len(sites) - 1, -1, -1))).sum())
        result[index] += mass
    return result


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
@pytest.mark.parametrize("mode,strength", [("conditional", .1), ("conditional", 1.),
                                           ("pmi", .25), ("pmi", 0.)])
@pytest.mark.parametrize("selection", [[1, 0, 1, 0], [0, 1, 0, 1]])
def test_exact_selected_law_across_original_clamped_neighbors(sampling, mode, strength,
                                                             selection, monkeypatch):
    potential = model(mode, strength)
    unary = torch.tensor([[[.3, -.7, .1], [-.5, .6, .2],
                           [-torch.inf, 7., -torch.inf], [.8, -.6, .2]]], dtype=torch.float64)
    selected = torch.tensor([selection], dtype=torch.bool)
    actual = enumerate_sampler(unary, potential, selected, sampling, monkeypatch)
    torch.testing.assert_close(actual, exact_law(unary, potential, selected, sampling),
                               atol=5e-14, rtol=5e-14)


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_unselected_latent_gap_is_integrated_not_deleted(sampling, monkeypatch):
    unary = torch.tensor([[[.3, -.7, .1], [-.5, .6, .2], [.8, -.6, .2]]], dtype=torch.float64)
    selected = torch.tensor([[True, False, True]])
    potential = model("pmi", 1.)
    actual = enumerate_sampler(unary, potential, selected, sampling, monkeypatch)
    torch.testing.assert_close(actual, exact_law(unary, potential, selected, sampling),
                               atol=5e-14, rtol=5e-14)
    # The fixture must distinguish q(X0,X2) from inventing an edge 0--2.
    wrong = exact_law(unary[:, [0, 2]], potential, torch.ones(1, 2, dtype=torch.bool), sampling)
    assert (actual - wrong).abs().max() > .005


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_only_selected_spans_run_messages_and_no_full_marginal_output(sampling, monkeypatch):
    unary = torch.zeros(2, 11, 3, dtype=torch.float64)
    unary[:, [2, 7]] = -torch.inf
    unary[:, [2, 7], 1] = 5.
    selected = torch.zeros(2, 11, dtype=torch.bool)
    selected[0, 0] = selected[1, 9] = True
    observed, original = [], segmented._forward

    def observe(values, *args, **kwargs):
        observed.append(tuple(values.shape))
        return original(values, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("full-chain sampler/marginal allocation is forbidden")

    monkeypatch.setattr(segmented, "_forward", observe)
    monkeypatch.setattr(segmented, "sparse_chain_marginals", forbidden)
    monkeypatch.setattr(segmented, "sample_sparse_chain", forbidden)
    result = sample_selected_chain(unary, model(), selected, sampling=sampling, max_chunk_tokens=2)
    assert observed == [(1, 2, 3), (1, 3, 3)]  # Not both rows' length-four middle spans.
    assert (result[~selected] == -1).all() and (result[selected] >= 0).all()


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
@pytest.mark.parametrize("all_visible", [False, True])
def test_no_selected_latents_uses_no_rng_or_dp(sampling, all_visible, monkeypatch):
    unary = torch.zeros(3, 5, 3, dtype=torch.float64)
    unary[:, 0, 1:] = -torch.inf
    selected = torch.zeros(3, 5, dtype=torch.bool)
    selected[:, 0] = True
    if all_visible:
        unary[..., 1:] = -torch.inf
        selected[:, 2] = True
    generator = torch.Generator().manual_seed(99)
    before = generator.get_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("no latent selected sites means no messages or draws")

    monkeypatch.setattr(segmented, "_forward", forbidden)
    monkeypatch.setattr(torch, "multinomial", forbidden)
    result = sample_selected_chain(unary, model(), selected, generator, sampling=sampling)
    assert torch.equal(before, generator.get_state())
    assert (result[selected] == 0).all() and (result[~selected] == -1).all()


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
@pytest.mark.parametrize("length", [1, 4])
def test_all_selected_law_and_partial_support(sampling, length, monkeypatch):
    unary = torch.randn(1, length, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(42))
    unary[:, :, 2] = -torch.inf
    selected = torch.ones(1, length, dtype=torch.bool)
    potential = model()
    actual = enumerate_sampler(unary, potential, selected, sampling, monkeypatch)
    torch.testing.assert_close(actual, exact_law(unary, potential, selected, sampling),
                               atol=5e-14, rtol=5e-14)


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_heterogeneous_batched_selection_matches_singleton_probabilities(sampling):
    n = 12000
    unary = torch.tensor([[[.2, -.3, .1], [-torch.inf, 3., -torch.inf], [.8, -.3, -.9]],
                          [[.1, -.4, .5], [.7, .1, -.5], [-torch.inf, -torch.inf, -2.]]], dtype=torch.float64)
    selected = torch.tensor([[True, True, False], [False, True, True]])
    potential = model("pmi", .5)
    result = sample_selected_chain(unary.repeat(n, 1, 1), potential, selected.repeat(n, 1),
                                   torch.Generator().manual_seed(721), sampling=sampling,
                                   max_chunk_tokens=512).reshape(n, 2, 3)
    expected = segmented.sparse_chain_marginals(unary, potential)
    for batch, pos in [(0, 0), (1, 1)]:
        frequency = torch.bincount(result[:, batch, pos], minlength=3).double() / n
        torch.testing.assert_close(frequency, expected[batch, pos], atol=.015, rtol=0.)
    assert (result[:, 0, 1] == 1).all() and (result[:, 1, 2] == 2).all()
    assert (result[:, 0, 2] == -1).all() and (result[:, 1, 0] == -1).all()


def test_validation_and_no_input_mutation():
    unary = torch.zeros(2, 3, 3, dtype=torch.float64)
    original = unary.clone()
    selected, potential = torch.ones(2, 3, dtype=torch.bool), model()
    sample_selected_chain(unary, potential, selected)
    assert torch.equal(unary, original)
    with pytest.raises(ValueError, match="sampling"):
        sample_selected_chain(unary, potential, selected, sampling="viterbi")
    for bad in (selected.long(), selected[:, :1]):
        with pytest.raises(ValueError, match="selected_mask"):
            sample_selected_chain(unary, potential, bad)
    for budget in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            sample_selected_chain(unary, potential, selected, max_chunk_tokens=budget)
    with pytest.raises(ValueError, match="singleton"):
        sample_selected_chain(unary, potential, selected, clamped_states=torch.zeros(2, 3, dtype=torch.long))
    for bad in (float("nan"), float("inf"), -torch.inf):
        invalid = unary.clone()
        invalid[:, 0] = bad
        with pytest.raises(ValueError):
            sample_selected_chain(invalid, potential, torch.zeros_like(selected))


def test_zero_strength_caller_delegation_remains_bitwise_unchanged(monkeypatch):
    # The new backend is never installed implicitly, and SAR's zero-strength
    # fast path must not call any clean proposal, selected or otherwise.
    from types import SimpleNamespace
    from chain_crf.count_sar import CountSARUpdate

    def forbidden(*args, **kwargs):
        raise AssertionError("zero strength must delegate the entire original update")

    def original_update(**kwargs):
        return kwargs["p_x0"], kwargs["x"] + torch.randint(3, kwargs["x"].shape)

    head = CountBigramHead(4, smoothing=.2).fit([[0, 1, 2], [0, 1, 0]])
    adapter = SelectedCountCRFDenoiser(head, mask_id=3, strength=0.)
    monkeypatch.setattr(adapter, "reveal", forbidden)
    model_stub = SimpleNamespace(mask_index=3, config=SimpleNamespace(noise=SimpleNamespace(type="loglinear")))
    update = CountSARUpdate(model_stub, adapter, original_update)
    monkeypatch.setattr(count_selected, "sample_selected_chain", forbidden)
    x, t, cache = torch.zeros(2, 5, dtype=torch.long), torch.ones(2), object()
    torch.manual_seed(93)
    expected = original_update(x=x, t=t, dt=.001, p_x0=cache)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(93)
    actual = update(x, t, .001, cache)
    assert actual[0] is cache and expected[0] is cache
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
@pytest.mark.parametrize("mode", ["conditional", "pmi"])
def test_selected_adapter_reveal_has_same_block_law_and_visible_clamps(sampling, mode):
    head = CountBigramHead(4, smoothing=.2).fit([[0, 1, 0, 2], [1, 2, 1], [0, 1, 2]])
    adapter = SelectedCountCRFDenoiser(head, mask_id=3, strength=.5, mode=mode,
                                       sparse_chunk_size=512)
    xt = torch.tensor([[3, 3, 3, 1]])
    logs = torch.tensor([[[.2, -.3, .1, -1e6], [-.5, .7, -.1, -1e6],
                          [-.3, .3, .2, -1e6], [-1e6, 0., -1e6, -1e6]]], dtype=torch.float64)
    # Selected endpoints share an unselected latent neighbor. A reveal flag at
    # an already visible site is allowed but cannot change that token.
    reveal = torch.tensor([[True, False, True, True]])
    states = torch.tensor(list(itertools.product(range(3), repeat=2)))
    n = len(states)
    values = xt.expand(n, -1).clone()
    values[:, [0, 2]] = states
    if sampling == "joint":
        expected = adapter.selected_log_probability(logs.expand(n, -1, -1), xt.expand(n, -1),
                                                     reveal.expand(n, -1), values).exp()
    else:
        marginal = adapter(logs, xt).exp()
        expected = marginal[0, 0, states[:, 0]] * marginal[0, 2, states[:, 1]]
    draws = 18000
    actual = adapter.reveal(logs.expand(draws, -1, -1), xt.expand(draws, -1),
                            reveal.expand(draws, -1), torch.Generator().manual_seed(77),
                            sampling=sampling)
    frequency = torch.bincount(actual[:, 0] * 3 + actual[:, 2], minlength=9).double() / draws
    torch.testing.assert_close(frequency, expected, atol=.012, rtol=0.)
    assert (actual[:, 1] == 3).all() and (actual[:, 3] == 1).all()


def test_selected_adapter_keeps_ct_propose_identity_and_file_loading(tmp_path):
    head = CountBigramHead(4, smoothing=.2).fit([[0, 1, 2], [2, 0, 1]])
    baseline = CountCRFDenoiser(head, mask_id=3, strength=.1)
    selected = SelectedCountCRFDenoiser(head, mask_id=3, strength=.1)
    assert SelectedCountCRFDenoiser.__call__ is CountCRFDenoiser.__call__
    assert SelectedCountCRFDenoiser.propose is CountCRFDenoiser.propose
    logs = torch.zeros(1, 3, 4, dtype=torch.float64)
    xt = torch.tensor([[3, 1, 3]])
    torch.testing.assert_close(selected(logs, xt), baseline(logs, xt), atol=0., rtol=0.)
    assert selected.identity()["selected_reveal"]["format"] == "selected_count_spans_v1"
    assert len(selected.identity()["selected_reveal"]["source_sha256"]) == 64
    assert "selected_reveal" not in baseline.identity()
    path = tmp_path / "counts.pt"
    head.save(path)
    loaded = SelectedCountCRFDenoiser.from_file(path, mask_id=3, strength=.1)
    assert isinstance(loaded, SelectedCountCRFDenoiser)
    assert loaded.counts_sha256 is not None
    for backend in ("gpu", "reference", "other"):
        with pytest.raises(ValueError, match="segments"):
            SelectedCountCRFDenoiser(head, mask_id=3, backend=backend)
    with pytest.raises(ValueError, match="reveal_mask"):
        selected.reveal(logs, xt, torch.ones_like(xt))
    with pytest.raises(ValueError, match="sampling"):
        selected.reveal(logs, xt, torch.zeros_like(xt, dtype=torch.bool), sampling="greedy")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA selected-law preflight")
@pytest.mark.parametrize("mode", ["conditional", "pmi"])
@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_cuda_selected_law_agrees_with_cpu_exact_enumeration(mode, sampling):
    unary = torch.tensor([[[.2, -.3, .1], [-.5, .7, -.1],
                           [-.3, .3, .2], [-torch.inf, 2., -torch.inf]]], dtype=torch.float64)
    selected = torch.tensor([[True, False, True, True]])
    head = CountBigramHead(3, mode=mode, strength=.5, smoothing=.2).fit(
        [[0, 1, 0, 2, 0], [1, 2, 2], [2, 1]])
    cpu = segmented.SegmentedCountPotential.from_head(head)
    expected = exact_law(unary, cpu, selected, sampling)
    gpu = segmented.SegmentedCountPotential.from_head(head.to("cuda"))
    n = 16000
    actual = sample_selected_chain(unary.cuda().expand(n, -1, -1), gpu,
                                   selected.cuda().expand(n, -1),
                                   torch.Generator(device="cuda").manual_seed(313),
                                   sampling=sampling, max_chunk_tokens=512)
    assert (actual[:, 1] == -1).all() and (actual[:, 3] == 1).all()
    index = actual[:, 0] * 9 + actual[:, 2] * 3 + actual[:, 3]
    frequency = torch.bincount(index, minlength=27).double().cpu() / n
    torch.testing.assert_close(frequency, expected, atol=.012, rtol=0.)
