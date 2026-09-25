"""CPU exactness/law/memory-structure checks for isolated span inference."""

from dataclasses import replace
import itertools

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf import sparse_count as reference
from chain_crf import sparse_count_segments as segmented


def model(mode='pmi', strength=.5, empty=False):
    head = CountBigramHead(4, mode, strength, .1)
    if not empty:
        head.fit([[0, 1, 0, 1, 2], [3, 2, 1], [0, 1], [2, 2]])
    old = reference.SparseCountPotential.from_head(head)
    return old, segmented.SegmentedCountPotential.from_reference(old)


def enumerate_chain(unary, potential):
    batch, length, vocab = unary.shape
    all_states = torch.tensor(list(itertools.product(range(vocab), repeat=length)))
    pair = (potential.left[:, None] * potential.right[None, :] + potential.sparse.to_dense()).log() + potential.log_scale
    scores = unary[:, torch.arange(length)[None, :], all_states].sum(-1)
    scores = scores + pair[all_states[:, :-1], all_states[:, 1:]].sum(-1)[None, :]
    logz = scores.logsumexp(-1)
    probabilities = (scores - logz[:, None]).exp()
    marginals = torch.zeros_like(unary)
    for pos in range(length):
        marginals[:, pos].scatter_add_(1, all_states[:, pos].expand(batch, -1), probabilities)
    return logz, marginals, probabilities


def mixed_unaries():
    unary = torch.randn(6, 4, 4, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(313)) * 2
    clamps = [(1, 1, 2), (1, 2, 0), (2, 0, 3), (2, 2, 1),
              (3, 0, 1), (3, 1, 2), (3, 2, 3), (3, 3, 0),
              (4, 3, 2), (5, 1, 0)]
    for batch, pos, token in clamps:
        unary[batch, pos] = -torch.inf
        unary[batch, pos, token] = (pos + 1) * 1.7 - batch * .9
    unary[5, 3, 2:] = -torch.inf  # Non-singleton partial support remains a span.
    return unary


@pytest.mark.parametrize('mode', ['pmi', 'conditional'])
@pytest.mark.parametrize('strength', [0., .025, .5, 1.])
@pytest.mark.parametrize('budget', [1, 5, 4096])
def test_partition_marginals_and_clamps_equal_enumeration(mode, strength, budget):
    old, new = model(mode, strength)
    unary = mixed_unaries()
    expected_z, expected_m, _ = enumerate_chain(unary, old)
    actual_z = segmented.sparse_chain_log_partition(unary, new, max_chunk_tokens=budget)
    actual_m = segmented.sparse_chain_marginals(unary, new, max_chunk_tokens=budget)
    torch.testing.assert_close(actual_z, expected_z, atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(actual_m, expected_m, atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(actual_z, reference.sparse_chain_log_partition(unary, old), atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(actual_m, reference.sparse_chain_marginals(unary, old), atol=5e-13, rtol=5e-13)
    states = torch.where(torch.isfinite(unary).sum(-1) == 1, unary.argmax(-1), -1)
    torch.testing.assert_close(segmented.sparse_chain_log_partition(unary, new,
        clamped_states=states, max_chunk_tokens=budget), expected_z, atol=5e-13, rtol=5e-13)
    draws = segmented.sample_sparse_chain(unary, new, torch.Generator().manual_seed(31),
                                          max_chunk_tokens=budget)
    assert torch.equal(draws[states >= 0], states[states >= 0])
    assert torch.isfinite(unary.gather(-1, draws.unsqueeze(-1))).all()


@pytest.mark.parametrize('mode', ['pmi', 'conditional'])
@pytest.mark.parametrize('strength', [0., .25, 1.])
@pytest.mark.parametrize('empty', [False, True])
def test_asymmetric_rows_columns_and_selected_pairs(mode, strength, empty):
    old, new = model(mode, strength, empty)
    dense = old.left[:, None] * old.right[None, :] + old.sparse.to_dense()
    rows, columns = torch.tensor([0, 3, 1, 1]), torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(new.selected_rows(rows), dense[rows], atol=0., rtol=0.)
    torch.testing.assert_close(new.selected_columns(columns), dense[:, columns].T, atol=0., rtol=0.)
    torch.testing.assert_close(new.selected_entries(rows, columns), dense[rows, columns], atol=0., rtol=0.)


@pytest.mark.parametrize('length', [1, 7])
@pytest.mark.parametrize('visible', [False, True])
def test_all_masked_all_visible_singleton_and_nonzero_visible_unaries(length, visible):
    old, new = model()
    unary = torch.randn(3, length, 4, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(44))
    if visible:
        chosen = torch.arange(3 * length).reshape(3, length) % 4
        unary.fill_(-torch.inf)
        unary.scatter_(-1, chosen.unsqueeze(-1), 23.)
    torch.testing.assert_close(segmented.sparse_chain_log_partition(unary, new),
                               reference.sparse_chain_log_partition(unary, old), atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(segmented.sparse_chain_marginals(unary, new),
                               reference.sparse_chain_marginals(unary, old), atol=3e-13, rtol=3e-13)
    if visible:
        assert torch.equal(segmented.sample_sparse_chain(unary, new), chosen)


@pytest.mark.parametrize('shift', [-10000., 10000.])
def test_global_edge_scale_and_large_signed_unary_offsets(shift):
    _, new = model()
    unary = mixed_unaries()
    shifted = new.shifted(shift)
    torch.testing.assert_close(segmented.sparse_chain_log_partition(unary, shifted),
        segmented.sparse_chain_log_partition(unary, new) + 3 * shift, atol=2e-11, rtol=2e-13)
    torch.testing.assert_close(segmented.sparse_chain_marginals(unary, shifted),
        segmented.sparse_chain_marginals(unary, new), atol=0., rtol=0.)
    offsets = torch.tensor([10000., -10000., -10000., 10000.]).reshape(1, 4, 1)
    torch.testing.assert_close(segmented.sparse_chain_log_partition(unary + offsets, new),
        segmented.sparse_chain_log_partition(unary, new), atol=5e-12, rtol=5e-13)
    torch.testing.assert_close(segmented.sparse_chain_marginals(unary + offsets, new),
        segmented.sparse_chain_marginals(unary, new), atol=5e-13, rtol=5e-13)


def test_zero_pairs_recover_independent_and_partition_gradient_matches_marginals():
    _, new = model(strength=0.)
    unary = mixed_unaries().requires_grad_(True)
    logz = segmented.sparse_chain_log_partition(unary, new, max_chunk_tokens=3)
    torch.testing.assert_close(logz, unary.logsumexp(-1).sum(-1), atol=3e-13, rtol=3e-13)
    gradient = torch.autograd.grad(logz.sum(), unary)[0]
    torch.testing.assert_close(gradient, unary.softmax(-1), atol=3e-13, rtol=3e-13)
    _, interacting = model()
    logz = segmented.sparse_chain_log_partition(unary, interacting, max_chunk_tokens=3)
    gradient = torch.autograd.grad(logz.sum(), unary)[0]
    expected = segmented.sparse_chain_marginals(unary.detach(), interacting, max_chunk_tokens=3)
    torch.testing.assert_close(gradient, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize('mode,strength', [('pmi', .5), ('conditional', .25), ('pmi', 0.)])
def test_ffbs_full_joint_law_across_two_spans(mode, strength):
    old, new = model(mode, strength)
    unary = torch.tensor([[[.2, -.1, .5, -.7], [.3, .7, -.4, -.2],
                           [-torch.inf, 2., -torch.inf, -torch.inf],
                           [-.6, .3, .2, -.1]]], dtype=torch.float64)
    _, _, exact = enumerate_chain(unary, old)
    n = 24000
    draws = segmented.sample_sparse_chain(unary.expand(n, -1, -1), new,
        torch.Generator().manual_seed(811), max_chunk_tokens=4096)
    indices = (draws * torch.tensor([64, 16, 4, 1])).sum(-1)
    empirical = torch.bincount(indices, minlength=256).double() / n
    torch.testing.assert_close(empirical, exact[0], atol=.005, rtol=0.)


def test_exact_length_chunks_have_no_dummy_nodes_or_global_max_padding():
    _, new = model()
    unary = torch.zeros(4, 15, 4, dtype=torch.float64)
    for batch in range(4):
        for pos in range(batch, 15, batch + 2):
            unary[batch, pos] = -torch.inf
            unary[batch, pos, pos % 4] = .7
    states, buckets, _, _ = segmented._prepare(unary, new, None, 5)
    covered = torch.zeros(unary.shape[:2], dtype=torch.long)
    for batches, positions, values in segmented._chunks(unary, new, buckets, 5):
        assert values.shape[0] * values.shape[1] <= 5 or values.shape[0] == 1
        assert torch.all(states[batches, positions] == -1)
        covered[batches, positions] += 1
    assert torch.equal(covered, (states == -1).long())


@pytest.mark.parametrize('mode', ['pmi', 'conditional'])
@pytest.mark.parametrize('strength', [0., .25, 1.])
def test_empty_count_model_preserves_independent_marginals_and_full_partition(mode, strength):
    old, new = model(mode, strength, empty=True)
    unary = mixed_unaries()
    torch.testing.assert_close(segmented.sparse_chain_marginals(unary, new),
                               unary.softmax(-1), atol=3e-13, rtol=3e-13)
    expected, _, _ = enumerate_chain(unary, old)
    torch.testing.assert_close(segmented.sparse_chain_log_partition(unary, new),
                               expected, atol=3e-13, rtol=3e-13)


def test_singleton_spans_need_no_sparse_dp_transition_and_long_spans_stay_intact(monkeypatch):
    _, new = model()
    unary = torch.randn(2, 9, 4, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(901))
    for pos in (1, 3, 5, 7):
        unary[:, pos] = -torch.inf
        unary[:, pos, pos % 4] = -.7
    def forbidden(*args, **kwargs):
        raise AssertionError('A visible clamp must not become a sparse DP transition')
    with monkeypatch.context() as patch:
        patch.setattr(segmented.SegmentedCountPotential, 'forward_mul', forbidden)
        patch.setattr(segmented.SegmentedCountPotential, 'backward_mul', forbidden)
        segmented.sparse_chain_log_partition(unary, new)
        segmented.sparse_chain_marginals(unary, new)
        segmented.sample_sparse_chain(unary, new)
    all_free = torch.zeros(2, 9, 4, dtype=torch.float64)
    _, buckets, _, _ = segmented._prepare(all_free, new, None, 2)
    shapes = [tuple(values.shape) for _, _, values in segmented._chunks(all_free, new, buckets, 2)]
    assert shapes == [(1, 9, 4), (1, 9, 4)]  # Never split a dependent chain to meet a budget.


@pytest.mark.parametrize('function', [segmented.sparse_chain_log_partition,
                                    segmented.sparse_chain_marginals, segmented.sample_sparse_chain])
@pytest.mark.parametrize('all_visible', [False, True])
@pytest.mark.parametrize('bad_value', [0., float('nan'), float('inf')])
def test_invalid_messages_or_visible_edges_raise(function, all_visible, bad_value):
    _, new = model(strength=0.)
    broken = replace(new, left=torch.full_like(new.left, bad_value))
    unary = torch.zeros(2, 3, 4, dtype=torch.float64)
    if all_visible:
        unary[:, :, 1:] = -torch.inf
    with pytest.raises(FloatingPointError, match='message total'):
        function(unary, broken)


def test_impossible_unaries_and_false_explicit_clamps_rejected():
    _, new = model()
    for bad in (float('nan'), float('inf')):
        unary = torch.zeros(2, 3, 4)
        unary[1, 1, 1] = bad
        with pytest.raises(ValueError, match='negative infinity'):
            segmented.sparse_chain_log_partition(unary, new)
    with pytest.raises(ValueError, match='supported token'):
        segmented.sample_sparse_chain(torch.full((1, 2, 4), -torch.inf), new)
    with pytest.raises(ValueError, match='singleton'):
        segmented.sample_sparse_chain(torch.zeros(1, 2, 4), new, clamped_states=torch.zeros(1, 2, dtype=torch.long))
    for budget in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='positive integer'):
            segmented.sparse_chain_marginals(torch.zeros(1, 2, 4), new, max_chunk_tokens=budget)
