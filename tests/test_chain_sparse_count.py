"""CPU-only exact tests for the standalone full-vocabulary count prototype."""

import itertools

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf.core import chain_marginals
from chain_crf.sparse_count import (
    SparseCountPotential, sample_sparse_chain, sparse_chain_log_partition,
    sparse_chain_marginals,
)


def head(mode="conditional", strength=.5, smoothing=.1):
    return CountBigramHead(4, mode, strength, smoothing).fit(
        [[0, 1, 0, 1, 2], [3, 2, 1], [0, 1], [2, 2]])


def dense_log_potential(model):
    n = model.left_counts.sum()
    v = model.vocab_size
    bl = (model.left_counts+1)/(n+v)
    br = (model.right_counts+1)/(n+v)
    empirical = torch.zeros(v*v, dtype=torch.float64)
    empirical[model.pair_keys] = model.pair_counts/n.clamp_min(1)
    eps = model.smoothing if n > 0 else 1.
    joint = (1-eps)*empirical.reshape(v, v)+eps*bl[:, None]*br[None, :]
    score = joint.log()-joint.sum(1).log()[:, None]
    if model.mode == "pmi":
        score -= joint.sum(0).log()[None, :]
    return model.strength*score


def enumerate_model(unary, logw):
    b, length, vocab = unary.shape
    states = torch.tensor(list(itertools.product(range(vocab), repeat=length)))
    score = unary[:, torch.arange(length)[None, :], states].sum(-1)
    if length > 1:
        score += logw[states[:, :-1], states[:, 1:]].sum(-1)[None, :]
    logz = score.logsumexp(-1)
    probabilities = (score-logz[:, None]).exp()
    marginals = torch.zeros_like(unary)
    for pos in range(length):
        marginals[:, pos].scatter_add_(1, states[:, pos].expand(b, -1), probabilities)
    return logz, marginals, probabilities


@pytest.mark.parametrize("mode", ["conditional", "pmi"])
@pytest.mark.parametrize("strength", [.025, .5, 1.])
def test_fractional_power_sparse_factorization_and_exact_inference(mode, strength):
    model = head(mode, strength)
    potential = SparseCountPotential.from_head(model)
    logw = dense_log_potential(model)
    actual = (potential.left[:, None]*potential.right[None, :]+potential.sparse.to_dense())
    torch.testing.assert_close(actual.log()+potential.log_scale, logw, atol=2e-13, rtol=2e-13)
    unary = torch.randn(2, 4, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    unary[1, 1] = -torch.inf
    unary[1, 1, 2] = 0.  # Observed/clamped position.
    expected_z, expected_marginals, _ = enumerate_model(unary, logw)
    torch.testing.assert_close(sparse_chain_log_partition(unary, potential), expected_z,
                               atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(sparse_chain_marginals(unary, potential), expected_marginals,
                               atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("mode", ["conditional", "pmi"])
@pytest.mark.parametrize("strength", [.025, .5, 1.])
def test_ffbs_joint_distribution_matches_enumeration(mode, strength):
    model = head(mode, strength)
    potential = SparseCountPotential.from_head(model)
    unary = torch.tensor([[[.2, -.4, .1, -.1], [-.7, .3, .1, .2]]], dtype=torch.float64)
    _, _, probabilities = enumerate_model(unary, dense_log_potential(model))
    n = 16000
    draws = sample_sparse_chain(unary.expand(n, -1, -1), potential,
                                generator=torch.Generator().manual_seed(81))
    frequencies = torch.bincount(draws[:, 0]*4+draws[:, 1], minlength=16).double()/n
    torch.testing.assert_close(frequencies, probabilities[0], atol=.008, rtol=0.)


@pytest.mark.parametrize("shift", [-10000., -3., 7., 10000.])
def test_global_edge_scaling_changes_only_partition(shift):
    potential = SparseCountPotential.from_head(head())
    unary = torch.randn(2, 3, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(42))
    changed = potential.shifted(shift)
    torch.testing.assert_close(sparse_chain_marginals(unary, changed),
                               sparse_chain_marginals(unary, potential), atol=0., rtol=0.)
    torch.testing.assert_close(sparse_chain_log_partition(unary, changed),
                               sparse_chain_log_partition(unary, potential)+2*shift)
    actual = sample_sparse_chain(unary, changed, torch.Generator().manual_seed(7))
    expected = sample_sparse_chain(unary, potential, torch.Generator().manual_seed(7))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", ["conditional", "pmi"])
@pytest.mark.parametrize("strength", [0., .025, .5, 1.])
def test_empty_counts_and_one_position(mode, strength):
    model = CountBigramHead(4, mode, strength)
    potential = SparseCountPotential.from_head(model)
    unary = torch.tensor([[[.1, -1., 2., -.4]]], dtype=torch.float64)
    assert potential.sparse._nnz() == 0
    torch.testing.assert_close(sparse_chain_marginals(unary, potential), unary.softmax(-1))
    torch.testing.assert_close(sparse_chain_log_partition(unary, potential), unary.logsumexp(-1).sum(-1))
    repeated = unary.expand(1, 4, 4)
    torch.testing.assert_close(sparse_chain_marginals(repeated, potential), repeated.softmax(-1))


def test_neutral_tail_gate_breaks_constant_invariance():
    # An empty head is exactly uniform over V: a full-vocab conditional bigram
    # is constant and MUST leave all unaries unchanged. The gated model does not.
    ids = torch.tensor([0, 1, -1])[None, None, :].expand(1, 4, 3)
    unary = torch.tensor([.6, .3, .1], dtype=torch.float64).log()[None, None, :].expand(1, 4, 3)
    model = CountBigramHead(50000, "conditional", .25)
    gated = model(ids).double()
    gate_marginals = chain_marginals(unary, gated)
    assert gate_marginals[0, 1, -1] > .60  # Baseline tail was only .10.
    constant = torch.full_like(gated, -.25*torch.log(torch.tensor(50000., dtype=torch.float64)))
    torch.testing.assert_close(chain_marginals(unary, constant), unary.exp())


def test_large_unary_offsets_and_zero_support():
    potential = SparseCountPotential.from_head(head("pmi"))
    unary = torch.tensor([[[0., -torch.inf, -5., -3.], [1., 3., -2., -torch.inf]]], dtype=torch.float64)
    offsets = torch.tensor([[[10000.], [-10000.]]], dtype=torch.float64)
    torch.testing.assert_close(sparse_chain_marginals(unary+offsets, potential),
                               sparse_chain_marginals(unary, potential), atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(sparse_chain_log_partition(unary+offsets, potential),
                               sparse_chain_log_partition(unary, potential), atol=2e-12, rtol=2e-12)
    draws = sample_sparse_chain(unary.expand(100, -1, -1), potential)
    assert not (draws[:, 0] == 1).any()
    assert not (draws[:, 1] == 3).any()


def test_rejects_invalid_statistics_and_impossible_unaries():
    with pytest.raises(ValueError, match="nonnegative"):
        SparseCountPotential.from_head(head(), strength=-.1)
    model = head()
    model.left_counts[0] += 1
    with pytest.raises(ValueError, match="marginals"):
        SparseCountPotential.from_head(model)
    with pytest.raises(ValueError, match="supported token"):
        sparse_chain_log_partition(torch.full((1, 2, 4), -torch.inf),
                                   SparseCountPotential.from_head(head()))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["conditional", "pmi"])
@pytest.mark.parametrize("strength", [.025, .5, 1.])
def test_cached_csr_orientations_and_columns_match_dense_on_device(device, mode, strength):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; GPU parity is run explicitly on the experiment worker")
    model = head(mode, strength)
    logw_cpu = dense_log_potential(model)
    potential = SparseCountPotential.from_head(model.to(device))
    assert potential.sparse.layout == torch.sparse_csr
    assert potential.sparse_transpose.layout == torch.sparse_csr
    assert potential.column_offsets.device.type == "cpu"
    dense = (logw_cpu-potential.log_scale).exp().to(device)
    message = torch.rand(3, 4, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(106)).to(device)
    torch.testing.assert_close(potential.forward_mul(message), message@dense, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(potential.backward_mul(message), message@dense.T, atol=3e-13, rtol=3e-13)
    for selected in ([2], [3, 1, 3]):
        indices = torch.tensor(selected, device=device)
        torch.testing.assert_close(potential.selected_columns(indices), dense[:, indices].T,
                                   atol=3e-13, rtol=3e-13)
    unary = torch.randn(2, 3, 4, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(19))
    unary[1, 1] = -torch.inf
    unary[1, 1, 2] = 0.
    expected_z, expected_marginals, _ = enumerate_model(unary, logw_cpu)
    torch.testing.assert_close(sparse_chain_log_partition(unary.to(device), potential).cpu(),
                               expected_z, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(sparse_chain_marginals(unary.to(device), potential).cpu(),
                               expected_marginals, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_csr_ffbs_matches_dense_joint_on_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; GPU parity is run explicitly on the experiment worker")
    model = head("conditional", .5)
    logw = dense_log_potential(model)
    potential = SparseCountPotential.from_head(model.to(device))
    unary = torch.tensor([[[.2, -.4, .1, -.1], [-.7, .3, .1, .2]]], dtype=torch.float64)
    _, _, probabilities = enumerate_model(unary, logw)
    n = 16000
    draws = sample_sparse_chain(unary.to(device).expand(n, -1, -1), potential,
                                generator=torch.Generator(device=device).manual_seed(83))
    frequencies = torch.bincount(draws[:, 0]*4+draws[:, 1], minlength=16).double().cpu()/n
    torch.testing.assert_close(frequencies, probabilities[0], atol=.009, rtol=0.)
