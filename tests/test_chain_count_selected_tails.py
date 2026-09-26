"""Exhaustively enumerate the new algorithm's own random branches on CPU."""

import itertools

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf import count_selected_tails as tails
from chain_crf import sparse_count_segments as segmented


def model(mode="conditional", strength=.1):
    head = CountBigramHead(3, mode=mode, strength=strength, smoothing=.2)
    head.fit([[0, 1, 0, 2, 0], [1, 2, 2], [2, 1]])
    return segmented.SegmentedCountPotential.from_head(head)


def dense_law(unary, potential, selected, sampling):
    length, vocab = unary.shape[1:]
    states = torch.tensor(list(itertools.product(range(vocab), repeat=length)))
    weight = potential.left[:, None] * potential.right[None] + potential.sparse.to_dense()
    score = unary[0, torch.arange(length)[None], states].sum(-1)
    score += (weight.log()[states[:, :-1], states[:, 1:]] + potential.log_scale).sum(-1)
    probability = score.softmax(0)
    sites = selected[0].nonzero().flatten()
    result = torch.zeros(vocab ** len(sites), dtype=torch.float64)
    if sampling == "joint":
        index = (states[:, sites] * vocab ** torch.arange(len(sites) - 1, -1, -1)).sum(-1)
        result.scatter_add_(0, index, probability)
    else:
        marginal = torch.zeros(length, vocab, dtype=torch.float64)
        for pos in range(length):
            marginal[pos].scatter_add_(0, states[:, pos], probability)
        choices = torch.tensor(list(itertools.product(range(vocab), repeat=len(sites))), dtype=torch.long)
        result = marginal[sites[None], choices].prod(-1)
    return result


def sampler_law(unary, potential, selected, sampling, monkeypatch):
    """Discover and sum every categorical branch, not empirical frequencies."""
    vocab = unary.shape[-1]
    sites = selected[0].nonzero().flatten()
    result = torch.zeros(vocab ** len(sites), dtype=torch.float64)
    prefixes = [()]

    class NeedChoice(Exception):
        pass

    while prefixes:
        prefix, calls, mass = prefixes.pop(), 0, 1.

        def draw(probability, num_samples, *, generator=None):
            nonlocal calls, mass
            assert probability.shape == (1, vocab) and num_samples == 1
            if calls == len(prefix):
                prefixes.extend(prefix + (token,) for token in range(vocab)
                                if probability[0, token] > 0)
                raise NeedChoice
            token = prefix[calls]
            mass *= float(probability[0, token] / probability.sum())
            calls += 1
            return torch.tensor([[token]])

        with monkeypatch.context() as patch:
            patch.setattr(torch, "multinomial", draw)
            try:
                output = tails.sample_selected_chain_tails(unary, potential, selected,
                    sampling=sampling, max_chunk_tokens=1)
            except NeedChoice:
                continue
        assert calls == len(prefix)
        assert (output[~selected] == -1).all()
        index = int((output[0, sites] * vocab ** torch.arange(len(sites) - 1, -1, -1)).sum())
        result[index] += mass
    return result


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
@pytest.mark.parametrize("mode,strength", [("conditional", .1), ("conditional", 1.),
                                           ("pmi", .25), ("pmi", 0.)])
def test_every_selection_subset_exact_law_with_unselected_tails(sampling, mode, strength, monkeypatch):
    unary = torch.tensor([[[.3, -.7, .1], [-.5, .6, .2], [.8, -.6, .2],
                           [-.1, .4, -.3]]], dtype=torch.float64)
    potential = model(mode, strength)
    for subset in itertools.product((False, True), repeat=4):
        selected = torch.tensor([subset])
        actual = sampler_law(unary, potential, selected, sampling, monkeypatch)
        expected = dense_law(unary, potential, selected, sampling)
        torch.testing.assert_close(actual, expected, atol=8e-14, rtol=8e-14)


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
@pytest.mark.parametrize("mode", ["conditional", "pmi"])
@pytest.mark.parametrize("subset", [[False, True, False, False, False],
                                    [False, False, False, True, False],
                                    [True, True, False, True, True]])
def test_original_left_right_clamps_and_internal_latent_gap(sampling, mode, subset, monkeypatch):
    unary = torch.tensor([[[-torch.inf, 4., -torch.inf], [.1, .7, -.3],
                           [.6, -.4, .1], [-.1, -.5, .8],
                           [-torch.inf, -torch.inf, -3.]]], dtype=torch.float64)
    potential = model(mode, .5)
    selected = torch.tensor([subset])
    actual = sampler_law(unary, potential, selected, sampling, monkeypatch)
    expected = dense_law(unary, potential, selected, sampling)
    torch.testing.assert_close(actual, expected, atol=8e-14, rtol=8e-14)


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
@pytest.mark.parametrize("site", [0, 2, 5])
def test_one_selected_site_retains_one_alpha_and_draws_once(sampling, site, monkeypatch):
    unary = torch.zeros(1, 6, 3, dtype=torch.float64)
    selected = torch.zeros(1, 6, dtype=torch.bool)
    selected[0, site] = True
    potential = model()
    count = {"forward": 0, "backward": 0, "categorical": 0}
    forward, backward, multinomial = potential.forward_mul, potential.backward_mul, torch.multinomial

    def f(self, value):
        count["forward"] += 1
        return forward(value)

    def b(self, value):
        count["backward"] += 1
        return backward(value)

    def draw(*args, **kwargs):
        count["categorical"] += 1
        return multinomial(*args, **kwargs)

    original = tails._envelope_messages

    def check(*args, **kwargs):
        out = original(*args, **kwargs)
        assert len(out[1]) == 1
        return out

    monkeypatch.setattr(type(potential), "forward_mul", f)
    monkeypatch.setattr(type(potential), "backward_mul", b)
    monkeypatch.setattr(torch, "multinomial", draw)
    monkeypatch.setattr(tails, "_envelope_messages", check)
    output = tails.sample_selected_chain_tails(unary, potential, selected, sampling=sampling)
    assert count == {"forward": site, "backward": 5 - site, "categorical": 1}
    assert output[0, site] >= 0 and (output[~selected] == -1).all()


def test_joint_traceback_covers_envelope_not_tails(monkeypatch):
    unary = torch.zeros(1, 7, 3, dtype=torch.float64)
    selected = torch.tensor([[False, True, False, False, True, False, False]])
    calls, original = [], torch.multinomial

    def draw(probability, *args, **kwargs):
        calls.append(probability.shape)
        return original(probability, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", draw)
    output = tails.sample_selected_chain_tails(unary, model(), selected)
    assert len(calls) == 4  # Positions4,3,2,1; internal gaps remain sampled.
    assert (output[~selected] == -1).all()


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
@pytest.mark.parametrize("all_visible", [False, True])
def test_no_selected_latents_no_rng_or_messages(sampling, all_visible, monkeypatch):
    unary = torch.zeros(2, 5, 3, dtype=torch.float64)
    unary[:, 0, 1:] = -torch.inf
    selected = torch.zeros(2, 5, dtype=torch.bool)
    selected[:, 0] = True
    if all_visible:
        unary[..., 1:] = -torch.inf
        selected[:, 3] = True
    generator = torch.Generator().manual_seed(51)
    before = generator.get_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("must skip all unselected spans")

    monkeypatch.setattr(tails, "_envelope_messages", forbidden)
    monkeypatch.setattr(torch, "multinomial", forbidden)
    result = tails.sample_selected_chain_tails(unary, model(), selected, generator, sampling=sampling)
    assert torch.equal(before, generator.get_state())
    assert (result[selected] == 0).all() and (result[~selected] == -1).all()


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_multiple_spans_heterogeneous_envelopes_and_partial_support(sampling):
    unary = torch.tensor([[[.1, .3, -.2], [-torch.inf, 6., -torch.inf],
                           [.8, -torch.inf, .2], [-.3, .2, .1], [.4, .2, -.6]],
                          [[.5, -.3, -.4], [.1, .4, -.2], [.7, .1, -.3],
                           [-torch.inf, -torch.inf, -1.], [-.4, .3, .7]]], dtype=torch.float64)
    selected = torch.tensor([[True, True, False, True, False], [False, True, False, False, True]])
    potential = model("pmi", .6)
    n = 18000
    result = tails.sample_selected_chain_tails(unary.repeat(n, 1, 1), potential,
        selected.repeat(n, 1), torch.Generator().manual_seed(981), sampling=sampling,
        max_chunk_tokens=127).reshape(n, 2, 5)
    expected = segmented.sparse_chain_marginals(unary, potential)
    for batch, pos in [(0, 0), (0, 3), (1, 1), (1, 4)]:
        empirical = torch.bincount(result[:, batch, pos], minlength=3).double() / n
        torch.testing.assert_close(empirical, expected[batch, pos], atol=.015, rtol=0.)
    assert (result[:, 0, 1] == 1).all()
    assert (result.masked_select((~selected)[None].expand(n, -1, -1)) == -1).all()


def test_single_selected_site_joint_and_marginal_paths_coincide():
    unary = torch.randn(4, 9, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(31))
    selected = torch.zeros(4, 9, dtype=torch.bool)
    selected[torch.arange(4), torch.tensor([0, 3, 5, 8])] = True
    potential = model()
    joint = tails.sample_selected_chain_tails(unary, potential, selected,
        torch.Generator().manual_seed(123), sampling="joint", max_chunk_tokens=4)
    marginal = tails.sample_selected_chain_tails(unary, potential, selected,
        torch.Generator().manual_seed(123), sampling="marginal", max_chunk_tokens=4)
    assert torch.equal(joint, marginal)


def test_chunk_bound_and_message_storage_depend_on_selected_envelope():
    unary = torch.zeros(3, 9, 3, dtype=torch.float64)
    selected = torch.zeros(3, 9, dtype=torch.bool)
    selected[0, 1] = selected[0, 5] = selected[1, 4] = selected[2, 0] = True
    potential = model()
    _, buckets, _, _ = segmented._prepare(unary, potential, None, 4)
    shapes = []
    for _, _, values, first, last, _ in tails._tail_chunks(unary, potential, buckets,
                                                        selected.tolist(), 4):
        assert values.shape[0] * values.shape[1] <= 4 or values.shape[0] == 1
        validation = tails._Validation(len(values), values.device)
        _, alpha, _ = tails._envelope_messages(values, potential, validation, first, last)
        assert len(alpha) == last - first + 1
        shapes.append((len(values), first, last))
    assert shapes == [(1, 0, 0), (1, 1, 5), (1, 4, 4)]


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_length_one_offsets_zero_strength_and_no_input_mutation(sampling):
    unary = torch.tensor([[[.1, -.5, .3]], [[-torch.inf, 14., -torch.inf]]], dtype=torch.float64)
    saved = unary.clone()
    selected = torch.ones(2, 1, dtype=torch.bool)
    potential = model(strength=0.)
    result = tails.sample_selected_chain_tails(unary, potential, selected,
                                              torch.Generator().manual_seed(12), sampling=sampling)
    shifted = tails.sample_selected_chain_tails(unary + 10000., potential.shifted(-10000.), selected,
                                               torch.Generator().manual_seed(12), sampling=sampling)
    assert torch.equal(result, shifted) and result[1, 0] == 1
    assert torch.equal(saved, unary)


def test_invalid_input_validation_is_preserved():
    unary = torch.zeros(1, 3, 3, dtype=torch.float64)
    selected, potential = torch.ones(1, 3, dtype=torch.bool), model()
    with pytest.raises(ValueError, match="sampling"):
        tails.sample_selected_chain_tails(unary, potential, selected, sampling="greedy")
    for bad in (selected.long(), selected[:, :1]):
        with pytest.raises(ValueError, match="selected_mask"):
            tails.sample_selected_chain_tails(unary, potential, bad)
    for budget in (0, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            tails.sample_selected_chain_tails(unary, potential, selected, max_chunk_tokens=budget)
    with pytest.raises(ValueError, match="singleton"):
        tails.sample_selected_chain_tails(unary, potential, selected,
                                          clamped_states=torch.zeros(1, 3, dtype=torch.long))
    for bad in (float("nan"), float("inf"), -torch.inf):
        broken = unary.clone()
        broken[:, 1] = bad
        with pytest.raises(ValueError):
            tails.sample_selected_chain_tails(broken, potential, selected)
