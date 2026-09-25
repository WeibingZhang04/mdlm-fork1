"""Exactness and fail-safe checks for the separate experimental GPU backend."""

from dataclasses import replace
import itertools

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf import sparse_count as reference
from chain_crf import sparse_count_gpu as optimized


def potential(mode, strength, device):
    counts = CountBigramHead(5, mode, strength).fit(
        [[0, 1, 0, 2, 3], [3, 2, 1], [0, 1], [2, 2]])
    old = reference.SparseCountPotential.from_head(counts.to(device))
    return old, optimized.GPUCountPotential.from_reference(old)


def enumeration(unary, pair):
    batch, length, vocab = unary.shape
    states = torch.tensor(list(itertools.product(range(vocab), repeat=length)),
                          device=unary.device)
    scores = unary[:, torch.arange(length, device=unary.device)[None, :], states].sum(-1)
    scores = scores + pair[states[:, :-1], states[:, 1:]].sum(-1)[None, :]
    logz = scores.logsumexp(-1)
    prob = (scores - logz[:, None]).exp()
    marginals = torch.zeros_like(unary)
    for pos in range(length):
        marginals[:, pos].scatter_add_(1, states[:, pos].expand(batch, -1), prob)
    return logz, marginals, prob


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mode', ['conditional', 'pmi'])
@pytest.mark.parametrize('strength', [0., .025, .5, 1.])
@pytest.mark.parametrize('batch', [4, 8, 12])
def test_batched_dense_enum_reference_and_same_rng(device, mode, strength, batch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    old, new = potential(mode, strength, device)
    pair = (old.left[:, None] * old.right[None, :] + old.sparse.to_dense()).log() + old.log_scale
    unary = torch.randn(batch, 4, 5, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(829)).to(device)
    unary[1, 1] = -torch.inf
    unary[1, 1, 2] = 0.
    unary[2, 0] = -torch.inf
    unary[2, 0, 1] = 0.
    unary[3, 2, 0] = -torch.inf
    expected_z, expected_m, _ = enumeration(unary, pair)
    torch.testing.assert_close(optimized.sparse_chain_log_partition(unary, new), expected_z,
                               atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(optimized.sparse_chain_marginals(unary, new), expected_m,
                               atol=3e-13, rtol=3e-13)
    for selected in ([4], [3, 1, 3, 4], [4, 4, 4, 4]):
        indices = torch.tensor(selected, device=device)
        torch.testing.assert_close(new.selected_columns(indices), old.selected_columns(indices),
                                   atol=0., rtol=0.)
    torch.testing.assert_close(optimized.sparse_chain_log_partition(unary, new),
                               reference.sparse_chain_log_partition(unary, old), atol=0., rtol=0.)
    torch.testing.assert_close(optimized.sparse_chain_marginals(unary, new),
                               reference.sparse_chain_marginals(unary, old), atol=0., rtol=0.)
    old_draw = reference.sample_sparse_chain(unary, old,
        torch.Generator(device=device).manual_seed(172))
    new_draw = optimized.sample_sparse_chain(unary, new,
        torch.Generator(device=device).manual_seed(172))
    assert torch.equal(old_draw, new_draw)
    assert new_draw[1, 1].item() == 2 and new_draw[2, 0].item() == 1
    for shift in (-10000., 10000.):
        torch.testing.assert_close(optimized.sparse_chain_log_partition(unary, new.shifted(shift)),
            expected_z + (unary.shape[1]-1)*shift, atol=2e-11, rtol=2e-13)
        torch.testing.assert_close(optimized.sparse_chain_marginals(unary, new.shifted(shift)),
            expected_m, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_ffbs_empirical_joint(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    old, new = potential('pmi', .5, device)
    pair = (old.left[:, None]*old.right[None, :] + old.sparse.to_dense()).log()+old.log_scale
    unary = torch.tensor([[[.2, -.4, .1, -.1, -.3], [-.7, .3, .1, .2, -.5]]],
                         dtype=torch.float64, device=device)
    _, _, exact = enumeration(unary, pair)
    draws = optimized.sample_sparse_chain(unary.expand(16000, -1, -1), new,
        torch.Generator(device=device).manual_seed(84))
    frequencies = torch.bincount(draws[:, 0]*5+draws[:, 1], minlength=25).double()/len(draws)
    torch.testing.assert_close(frequencies, exact[0], atol=.009, rtol=0.)


@pytest.mark.parametrize('function', [optimized.sparse_chain_log_partition,
                                    optimized.sparse_chain_marginals,
                                    optimized.sample_sparse_chain])
@pytest.mark.parametrize('bad_value', [0., float('nan'), float('inf')])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_invalid_intermediate_messages_raise_instead_of_returning(function, bad_value, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    _, model = potential('conditional', 0., device)  # No sparse corrections.
    broken = replace(model, left=torch.full_like(model.left, bad_value))
    unary = torch.zeros(4, 3, 5, dtype=torch.float64, device=device)
    with pytest.raises(FloatingPointError, match='message total'):
        function(unary, broken)


def test_impossible_input_still_rejected():
    _, model = potential('pmi', .5, 'cpu')
    with pytest.raises(ValueError, match='supported token'):
        optimized.sample_sparse_chain(torch.full((4, 3, 5), -torch.inf), model)
