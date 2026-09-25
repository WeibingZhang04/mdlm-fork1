import itertools

import pytest
import torch

from chain_crf import (
    build_candidates, chain_log_partition, chain_log_prob, chain_marginals,
    gold_log_prob, sample_candidate_tokens, sample_chain,
)


def enumerate_chain(unary, edge):
    states = torch.tensor(list(itertools.product(range(unary.shape[-1]), repeat=unary.shape[1])))
    values = []
    for row in states:
        value = unary[0, torch.arange(row.numel()), row].sum()
        if row.numel() > 1:
            value = value + edge[0, torch.arange(row.numel() - 1), row[:-1], row[1:]].sum()
        values.append(value)
    values = torch.stack(values)
    logz = values.logsumexp(0)
    probabilities = (values - logz).exp()
    marginals = torch.zeros_like(unary)
    for row, probability in zip(states, probabilities):
        marginals[0, torch.arange(row.numel()), row] += probability
    return logz, marginals, states, probabilities


@pytest.mark.parametrize("length,states", [(1, 3), (2, 2), (4, 3)])
def test_enumeration_partition_marginals_likelihood_and_gradients(length, states):
    torch.manual_seed(14)
    unary = torch.randn(1, length, states, dtype=torch.double, requires_grad=True)
    edge = torch.randn(1, length - 1, states, states, dtype=torch.double, requires_grad=True)
    brute_z, brute_marg, rows, probabilities = enumerate_chain(unary, edge)
    actual_z = chain_log_partition(unary, edge)
    torch.testing.assert_close(actual_z[0], brute_z)
    torch.testing.assert_close(chain_marginals(unary, edge), brute_marg)
    for row, probability in zip(rows, probabilities):
        torch.testing.assert_close(chain_log_prob(unary, edge, row.unsqueeze(0))[0], probability.log())
    actual_grad = torch.autograd.grad(actual_z.sum(), (unary, edge), allow_unused=True)
    brute_grad = torch.autograd.grad(brute_z, (unary, edge), allow_unused=True)
    for actual, expected in zip(actual_grad, brute_grad):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)


def test_negative_infinity_padding_has_finite_gradients():
    unary = torch.tensor([[[0., -torch.inf, -torch.inf], [0.3, -0.4, -torch.inf],
                           [-torch.inf, 0., -torch.inf]]], requires_grad=True)
    edge = torch.randn(1, 2, 3, 3, requires_grad=True)
    loss = chain_log_partition(unary, edge).sum()
    loss.backward()
    assert torch.isfinite(unary.grad).all() and torch.isfinite(edge.grad).all()
    assert unary.grad[0, 0, 1:].eq(0).all()
    torch.testing.assert_close(chain_marginals(unary, edge).sum(-1), torch.ones(1, 3))
    sampled = sample_chain(unary.expand(100, -1, -1), edge.expand(100, -1, -1, -1))
    assert sampled[:, 0].eq(0).all() and sampled[:, -1].eq(1).all()


def test_impossible_transition_slices_are_autograd_safe():
    unary = torch.tensor([[[0., 0.], [0., 0.], [0., 0.]]], requires_grad=True)
    edge = torch.tensor([[[[0., -torch.inf], [0., -torch.inf]],
                           [[0., -torch.inf], [-torch.inf, -torch.inf]]]], requires_grad=True)
    chain_log_partition(unary, edge).sum().backward()
    assert torch.isfinite(unary.grad).all() and torch.isfinite(edge.grad).all()
    actual = chain_marginals(unary, edge)
    torch.testing.assert_close(actual[0, 1:], torch.tensor([[1., 0.], [1., 0.]]))


def test_joint_sampler_matches_joint_not_product_marginals():
    n = 24000
    unary = torch.zeros(n, 2, 2)
    edge = torch.tensor([[[[2., -2.], [-2., 2.]]]]).expand(n, -1, -1, -1)
    generator = torch.Generator().manual_seed(9)
    draws = sample_chain(unary, edge, generator=generator)
    frequency = torch.bincount(draws[:, 0] * 2 + draws[:, 1], minlength=4) / n
    expected = torch.tensor([2., -2., -2., 2.]).softmax(0)
    torch.testing.assert_close(frequency, expected, atol=0.012, rtol=0)
    assert draws[:, 0].eq(draws[:, 1]).float().mean() > 0.96


@pytest.mark.parametrize("k", [0, 1, 3, 8])
def test_zero_pairs_recover_backbone_full_token_likelihood(k):
    torch.manual_seed(8)
    lp = torch.randn(2, 4, 7, dtype=torch.double).log_softmax(-1)
    masked_input = torch.tensor([[6, 2, 6, 1], [1, 6, 6, 6]])
    gold = torch.tensor([[0, 2, 5, 1], [1, 4, 0, 3]])
    packet = build_candidates(lp, masked_input, mask_id=6, k=k, gold=gold)
    size = packet.unary.shape[-1]
    edge = torch.zeros(2, 3, size, size, dtype=torch.double)
    expected = packet.normalized_log_probs.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    expected = (expected * packet.masked).sum(-1)
    torch.testing.assert_close(gold_log_prob(packet, edge), expected)
    torch.testing.assert_close(chain_log_partition(packet.unary, edge), torch.zeros(2, dtype=torch.double))
    assert packet.candidate_ids[~packet.masked][:, 0].tolist() == [2, 1, 1]
    assert packet.unary[~packet.masked][:, 0].eq(0).all()


def test_tail_expansion_recovers_backbone_sampling_and_clamping():
    n = 18000
    probabilities = torch.tensor([0.05, 0.1, 0.2, 0.25, 0.4, 0.])
    lp = probabilities.log()[None, None].expand(n, 2, -1)
    observed = torch.tensor([[5, 2]]).expand(n, -1)
    packet = build_candidates(lp, observed, mask_id=5, k=2)
    edge = torch.zeros(n, 1, 3, 3)
    generator = torch.Generator().manual_seed(7)
    states = sample_chain(packet.unary, edge, generator)
    tokens = sample_candidate_tokens(packet, states, generator)
    frequency = torch.bincount(tokens[:, 0], minlength=6) / n
    torch.testing.assert_close(frequency, probabilities, atol=.012, rtol=0)
    assert tokens[:, 1].eq(2).all()


def test_tiny_tail_is_not_lost_to_subtraction():
    lp = torch.tensor([[[0., -60., -80., -torch.inf]]])
    packet = build_candidates(lp, torch.tensor([[3]]), mask_id=3, k=1, gold=torch.tensor([[2]]))
    assert torch.isfinite(packet.unary[..., -1]).all()
    edge = torch.zeros(1, 0, 2, 2)
    torch.testing.assert_close(gold_log_prob(packet, edge), torch.tensor([-80.]))


def test_no_gold_insertion_and_gold_validation():
    lp = torch.tensor([[[0., -1., -5., -10.]]])
    packet = build_candidates(lp, torch.tensor([[3]]), 3, 1, gold=torch.tensor([[2]]))
    assert packet.candidate_ids.tolist() == [[[0, -1]]]
    assert packet.gold_states.item() == 1
    with pytest.raises(ValueError, match="clamped"):
        build_candidates(lp, torch.tensor([[0]]), 3, 1, gold=torch.tensor([[1]]))


def test_singleton_mask_joint_and_own_marginal_laws_agree():
    packet = build_candidates(torch.zeros(1, 3, 4), torch.tensor([[0, 3, 2]]), 3, 3)
    edge = torch.randn(1, 2, 4, 4)
    marginal = chain_marginals(packet.unary, edge)[0, 1]
    log_laws = []
    for state in range(3):
        log_laws.append(chain_log_prob(packet.unary, edge, torch.tensor([[0, state, 0]]))[0])
    torch.testing.assert_close(torch.stack(log_laws).exp(), marginal[:3])


def test_degenerate_support_excludes_mask_and_impossible_gold_has_minus_inf():
    lp = torch.tensor([[[0., -torch.inf, -torch.inf, -torch.inf]]])
    for k in (0, 1, 2, 3):
        packet = build_candidates(lp, torch.tensor([[1]]), 1, k, gold=torch.tensor([[3]]))
        assert not packet.candidate_ids.eq(1).any()
        edge = torch.zeros(1, 0, k + 1, k + 1)
        assert gold_log_prob(packet, edge).isneginf().all()
