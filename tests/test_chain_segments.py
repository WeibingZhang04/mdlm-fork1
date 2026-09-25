import itertools

import pytest
import torch

from chain_crf.core import (build_candidates, chain_log_marginals,
                            chain_log_partition, chain_log_prob,
                            sample_candidate_tokens)
from chain_crf.segments import (pack_segments, sample_segmented_chain,
                               segmented_log_marginals, segmented_log_partition,
                               segmented_marginals)


def make_problem(masked, states=3, dtype=torch.double):
    torch.manual_seed(42)
    batch, length = masked.shape
    unary = torch.randn(batch, length, states, dtype=dtype)
    clamped = torch.arange(batch * length).reshape(batch, length) % states
    visible_unary = torch.full_like(unary, -torch.inf)
    visible_unary.scatter_(-1, clamped[..., None], unary.gather(-1, clamped[..., None]))
    unary = torch.where(masked[..., None], unary, visible_unary).requires_grad_()
    edge = torch.randn(batch, length - 1, states, states, dtype=dtype, requires_grad=True)
    return unary, edge


MASKS = [
    [[True]], [[False]], [[True], [False]],
    [[True] * 7], [[False] * 7],
    [[True, False, True, False, True, False, True]],
    [[False, True, True, False, True, True, True]],
    [[True, True, False, False, True, False, False]],
    [[True] * 7, [False] * 7,
     [False, True, False, True, True, False, True],
     [True, False, False, False, True, True, False]],
]


@pytest.mark.parametrize("mask_rows", MASKS)
def test_segmented_matches_dense_partition_and_node_laws(mask_rows):
    mask = torch.tensor(mask_rows)
    unary, edge = make_problem(mask)
    packed = pack_segments(unary, edge, mask)
    torch.testing.assert_close(packed.log_partition(), chain_log_partition(unary, edge))
    torch.testing.assert_close(packed.log_marginals(), chain_log_marginals(unary, edge))
    torch.testing.assert_close(packed.marginals().sum(-1), torch.ones_like(mask, dtype=unary.dtype))
    samples = packed.sample(torch.Generator().manual_seed(71))
    assert samples.shape == mask.shape
    assert torch.isfinite(unary.gather(-1, samples[..., None])).all()
    if packed.run_count:
        assert int(packed.lengths.sum()) == int(mask.sum())
        assert packed.padded_length == int(packed.lengths.max())
        if (~packed.valid).any():
            padding = packed.unary[~packed.valid]
            assert padding[:, 0].eq(0).all() and padding[:, 1:].isneginf().all()


def test_partition_and_marginal_gradients_match_dense():
    mask = torch.tensor([[False, True, True, False, True, False, True],
                         [True, False, False, True, True, True, False]])
    unary, edge = make_problem(mask)
    segmented = segmented_log_partition(unary, edge, mask).sum()
    dense = chain_log_partition(unary, edge).sum()
    actual = torch.autograd.grad(segmented, (unary, edge), retain_graph=True)
    expected = torch.autograd.grad(dense, (unary, edge), retain_graph=True)
    for a, e in zip(actual, expected):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e)
    weights = torch.randn_like(unary)
    segmented_loss = (segmented_marginals(unary, edge, mask) * weights).sum()
    dense_loss = (chain_log_marginals(unary, edge).exp() * weights).sum()
    actual = torch.autograd.grad(segmented_loss, (unary, edge), retain_graph=True)
    expected = torch.autograd.grad(dense_loss, (unary, edge))
    for a, e in zip(actual, expected):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-10)


def test_known_edge_constants_can_be_removed_without_changing_law():
    mask = torch.tensor([[False, False, True, False, False]])
    unary, edge = make_problem(mask)
    known_edges = ~mask[:, :-1] & ~mask[:, 1:]
    reduced = torch.where(known_edges[..., None, None], 0., edge)
    packed = pack_segments(unary, edge, mask)
    packed_reduced = pack_segments(unary, reduced, mask)
    torch.testing.assert_close(packed.marginals(), packed_reduced.marginals())
    torch.testing.assert_close(packed.log_partition() - packed_reduced.log_partition(),
                               packed.constant - packed_reduced.constant)
    torch.testing.assert_close(packed_reduced.log_partition(), chain_log_partition(unary, reduced))


def test_bruteforce_including_nonzero_visible_state_and_boundary_scores():
    mask = torch.tensor([[True, False, True, True]])
    unary, edge = make_problem(mask, states=2)
    rows = torch.tensor(list(itertools.product(range(2), repeat=4)))
    scores = []
    for row in rows:
        score = unary[0, torch.arange(4), row].sum()
        score = score + edge[0, torch.arange(3), row[:-1], row[1:]].sum()
        scores.append(score)
    scores = torch.stack(scores)
    probabilities = scores.softmax(0)
    marginals = torch.zeros_like(unary)
    for row, probability in zip(rows, probabilities):
        marginals[0, torch.arange(4), row] += probability
    torch.testing.assert_close(segmented_log_partition(unary, edge, mask)[0], scores.logsumexp(0))
    torch.testing.assert_close(segmented_marginals(unary, edge, mask), marginals)


def test_impossible_transition_and_candidate_slices_have_finite_gradients():
    mask = torch.tensor([[False, True, True, False, True]])
    unary = torch.tensor([[[0., -torch.inf], [.1, -.2], [.3, -torch.inf],
                           [-torch.inf, .8], [-.5, .6]]], requires_grad=True)
    edge = torch.tensor([[[[0., -torch.inf], [0., 0.]], [[.1, -.5], [0., -torch.inf]],
                          [[-.2, .2], [0., -torch.inf]], [[0., -torch.inf], [.3, -.4]]]],
                        requires_grad=True)
    actual = segmented_log_partition(unary, edge, mask)
    torch.testing.assert_close(actual, chain_log_partition(unary, edge))
    actual.sum().backward()
    assert torch.isfinite(unary.grad).all() and torch.isfinite(edge.grad).all()
    torch.testing.assert_close(segmented_log_marginals(unary, edge, mask),
                               chain_log_marginals(unary, edge))


def test_joint_sampling_matches_enumeration_and_independent_runs():
    mask = torch.tensor([[True, True, False, True, True]])
    unary = torch.zeros(1, 5, 2)
    unary[:, 2, 0] = -torch.inf  # Visible state need not be zero.
    edge = torch.zeros(1, 4, 2, 2)
    edge[:, 0] = torch.tensor([[2., -2.], [-2., 2.]])
    edge[:, 3] = torch.tensor([[-.3, .4], [.8, -.1]])
    n = 16000
    draws = sample_segmented_chain(unary.expand(n, -1, -1), edge.expand(n, -1, -1, -1),
                                    mask.expand(n, -1), torch.Generator().manual_seed(6))
    assert draws[:, 2].eq(1).all()
    code = draws[:, 0] * 8 + draws[:, 1] * 4 + draws[:, 3] * 2 + draws[:, 4]
    frequency = torch.bincount(code, minlength=16) / n
    rows = torch.tensor([[a, b, 1, c, d] for a, b, c, d in itertools.product(range(2), repeat=4)])
    expected = chain_log_prob(unary.expand(16, -1, -1), edge.expand(16, -1, -1, -1), rows).exp()
    torch.testing.assert_close(frequency, expected, atol=.012, rtol=0)
    assert draws[:, 0].eq(draws[:, 1]).float().mean() > .96


@pytest.mark.parametrize("k", [0, 1, 3, 5])
def test_zero_pairs_keep_full_support_tail_likelihood(k):
    torch.manual_seed(7)
    log_probs = torch.randn(2, 5, 6, dtype=torch.double).log_softmax(-1)
    corrupted = torch.tensor([[5, 2, 5, 5, 1], [2, 5, 0, 5, 5]])
    gold = torch.tensor([[0, 2, 4, 0, 1], [2, 1, 0, 3, 4]])
    packet = build_candidates(log_probs, corrupted, 5, k, gold=gold)
    states = packet.unary.shape[-1]
    edge = torch.zeros(2, 4, states, states, dtype=torch.double)
    log_z = segmented_log_partition(packet.unary, edge, packet.masked)
    score = packet.unary.gather(-1, packet.gold_states[..., None]).squeeze(-1).sum(-1)
    full_log_prob = score - log_z + packet.gold_tail_logprob.sum(-1)
    expected = packet.normalized_log_probs.gather(-1, gold[..., None]).squeeze(-1)
    torch.testing.assert_close(full_log_prob, torch.where(packet.masked, expected, 0.).sum(-1))
    torch.testing.assert_close(log_z, torch.zeros(2, dtype=torch.double))


def test_zero_pairs_and_tail_expansion_sample_backbone():
    n = 14000
    probabilities = torch.tensor([.1, .2, .3, .4, 0.])
    packet = build_candidates(probabilities.log()[None, None].expand(n, 3, -1),
                               torch.tensor([[4, 2, 4]]).expand(n, -1), 4, 1)
    edge = torch.zeros(n, 2, 2, 2)
    generator = torch.Generator().manual_seed(81)
    states = sample_segmented_chain(packet.unary, edge, packet.masked, generator)
    tokens = sample_candidate_tokens(packet, states, generator)
    assert tokens[:, 1].eq(2).all()
    for position in (0, 2):
        frequency = torch.bincount(tokens[:, position], minlength=5) / n
        torch.testing.assert_close(frequency, probabilities, atol=.012, rtol=0)


def test_visible_clamp_validation():
    with pytest.raises(ValueError, match="clamped"):
        pack_segments(torch.zeros(1, 1, 2), torch.zeros(1, 0, 2, 2), torch.tensor([[False]]))
    with pytest.raises(ValueError, match="boolean"):
        pack_segments(torch.zeros(1, 1, 2), torch.zeros(1, 0, 2, 2), torch.zeros(1, 1))
