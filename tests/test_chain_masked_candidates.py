"""Compare masked-only candidate construction against the prior dense algorithm."""
import pytest
import torch
from torch import Tensor
from typing import Optional
from chain_crf import core
from chain_crf.core import CandidateBatch, _logsumexp
from chain_crf.segments import segmented_log_partition, segmented_log_marginals

# Frozen pre-optimization reference: computes top-K/tail for all rows before
# clamping. Keep this independent of build_candidates to detect regressions.
def dense_reference(log_probs: Tensor, masked_input: Tensor, mask_id: int, k: int,
                     gold: Optional[Tensor] = None) -> CandidateBatch:
    """Build candidates without changing held-out support to include gold.

    Input probabilities are normalized after excluding the mask token. Tail
    mass is computed with logsumexp of excluded tokens (not 1-sum(topK)), so
    small tails remain numerically stable. k may be zero or >= vocabulary.
    """
    if log_probs.ndim != 3 or masked_input.shape != log_probs.shape[:2]:
        raise ValueError("log_probs [B,L,V] and masked_input [B,L] are required")
    if k < 0 or not 0 <= mask_id < log_probs.shape[-1]:
        raise ValueError("k must be nonnegative and mask_id inside the vocabulary")
    vocab = log_probs.shape[-1]
    k = min(k, vocab - 1)
    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    lp = log_probs.to(dtype).clone()
    lp[..., mask_id] = -torch.inf
    norm = _logsumexp(lp, -1).unsqueeze(-1)
    # Observed MDLM rows may be point masses. A visible mask-token point mass
    # is invalid; well-formed visible rows and all masked rows normalize here.
    lp = lp - norm
    # Ask for one extra item and remove MASK explicitly. Merely assigning it
    # -inf is insufficient when other real tokens also have zero probability:
    # topk can break -inf ties by returning MASK instead of a real token.
    with torch.profiler.record_function('crf.topk'):
        extra_values, extra_ids = torch.topk(lp, k + 1, dim=-1)
    order = torch.arange(k + 1, device=lp.device)
    mask_position = torch.where(extra_ids.eq(mask_id), order, k + 1).amin(-1, keepdim=True)
    keep = torch.arange(k, device=lp.device).expand(*lp.shape[:2], k)
    keep = keep + keep.ge(mask_position).long()
    top_values = extra_values.gather(-1, keep)
    top_ids = extra_ids.gather(-1, keep)
    with torch.profiler.record_function('crf.tail_mass'):
        tail = lp.clone()
        tail.scatter_(-1, top_ids, -torch.inf)
        tail_mass = _logsumexp(tail, -1)
    ids = torch.cat((top_ids, torch.full_like(masked_input.unsqueeze(-1), -1)), -1)
    unary = torch.cat((top_values, tail_mass.unsqueeze(-1)), -1)
    masked = masked_input.eq(mask_id)
    visible_ids = torch.full_like(ids, -1)
    visible_ids[..., 0] = masked_input
    visible_unary = torch.full_like(unary, -torch.inf)
    visible_unary[..., 0] = 0.
    ids = torch.where(masked.unsqueeze(-1), ids, visible_ids)
    unary = torch.where(masked.unsqueeze(-1), unary, visible_unary)
    gold_states = gold_tail = None
    if gold is not None:
        if gold.shape != masked_input.shape:
            raise ValueError("gold must have shape [B,L]")
        if torch.any((gold != masked_input) & ~masked):
            raise ValueError("gold disagrees with an observed/clamped token")
        if torch.any(gold.eq(mask_id)):
            raise ValueError("clean gold sequences cannot contain the mask token")
        matches = ids.eq(gold.unsqueeze(-1)) & ids.ge(0)
        explicit = matches.any(-1)
        gold_states = torch.where(explicit, matches.long().argmax(-1), torch.full_like(gold, k))
        gold_lp = lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
        # Avoid undefined -inf - -inf on unused states of point-mass rows.
        is_tail = masked & ~explicit & torch.isfinite(tail_mass)
        numerator = torch.where(is_tail, gold_lp, torch.zeros_like(gold_lp))
        denominator = torch.where(is_tail, tail_mass, torch.zeros_like(tail_mass))
        gold_tail = numerator - denominator
    return CandidateBatch(ids, unary, masked, gold_states, gold_tail, lp)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def inputs(device, pattern, dtype=torch.float64):
    torch.manual_seed(84)
    # Noncontiguous inputs and a MASK id inside the vocabulary.
    logits = torch.randn(2, 7, 5, device=device, dtype=dtype).transpose(1, 2)
    gold = torch.tensor([[0, 1, 2, 4, 6], [5, 4, 2, 1, 0]], device=device)
    mask = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 1, 1, 0]], device=device, dtype=torch.bool)
    if pattern == 'all':
        mask.fill_(True)
    elif pattern == 'none':
        mask.fill_(False)
    observed = gold.masked_fill(mask, 3)
    return logits, observed, gold


@pytest.mark.parametrize('pattern', ['all', 'mixed', 'none'])
@pytest.mark.parametrize('k', [0, 2, 6, 20])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_candidate_fields_match_dense_reference(device, pattern, k, dtype):
    logits, observed, gold = inputs(device, pattern, dtype)
    before = logits.clone()
    actual = core.build_candidates(logits, observed, 3, k, gold)
    expected = dense_reference(logits, observed, 3, k, gold)
    for name in CandidateBatch.__dataclass_fields__:
        a, b = getattr(actual, name), getattr(expected, name)
        # Compacting noncontiguous rows can change reduction order by a few
        # ulps. IDs, state mappings and the full normalized field stay exact.
        tolerance = 4 * torch.finfo(dtype).eps if name in ('unary', 'gold_tail_logprob') else 0
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
    assert actual.unary.shape == (2, 5, min(k, 6) + 1)


@pytest.mark.parametrize('pattern', ['all', 'mixed', 'none'])
def test_visible_rows_never_enter_topk_or_tail_reduction(device, pattern, monkeypatch):
    logits, observed, gold = inputs(device, pattern)
    calls, reductions = [], []
    topk, reduce = torch.topk, core._logsumexp
    def tracked_topk(values, *args, **kwargs):
        calls.append(tuple(values.shape))
        return topk(values, *args, **kwargs)
    def tracked_reduce(values, *args, **kwargs):
        reductions.append(tuple(values.shape))
        return reduce(values, *args, **kwargs)
    monkeypatch.setattr(torch, 'topk', tracked_topk)
    monkeypatch.setattr(core, '_logsumexp', tracked_reduce)
    core.build_candidates(logits, observed, 3, 2, gold)
    masked_count = int(observed.eq(3).sum())
    active_shapes = [(masked_count, 7)] if masked_count else []
    assert calls == active_shapes
    # The full normalized backbone field is retained for diagnostics.
    assert reductions == [(2, 5, 7)] + active_shapes


@pytest.mark.parametrize('pattern', ['all', 'mixed', 'none'])
def test_likelihood_and_backbone_gradients_unchanged(device, pattern):
    logits, observed, gold = inputs(device, pattern)
    logits.requires_grad_()
    actual = core.build_candidates(logits, observed, 3, 2, gold)
    expected = dense_reference(logits, observed, 3, 2, gold)
    edge = torch.randn(2, 4, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
    actual_score = core.gold_log_prob(actual, edge)
    expected_score = core.gold_log_prob(expected, edge)
    torch.testing.assert_close(actual_score, expected_score)
    actual_grads = torch.autograd.grad(actual_score.sum(), (logits, edge))
    expected_grads = torch.autograd.grad(expected_score.sum(), (logits, edge))
    for a, b in zip(actual_grads, expected_grads):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b)
    if pattern == 'none':
        assert actual_grads[0].eq(0).all()
        # Partition-only callers also retain a zero gradient to the backbone.
        packet = core.build_candidates(logits, observed, 3, 2)
        grad, = torch.autograd.grad(core.chain_log_partition(packet.unary, edge).sum(), (logits,))
        assert grad.eq(0).all()


@torch.no_grad()
def test_dense_and_segmented_inference_remain_equivalent(device):
    logits, observed, gold = inputs(device, 'mixed')
    packet = core.build_candidates(logits, observed, 3, 2, gold)
    reference = dense_reference(logits, observed, 3, 2, gold)
    edge = torch.randn(2, 4, 3, 3, device=device, dtype=torch.float64)
    torch.testing.assert_close(core.chain_log_partition(packet.unary, edge),
                               segmented_log_partition(packet.unary, edge, packet.masked))
    torch.testing.assert_close(core.chain_log_marginals(packet.unary, edge),
                               segmented_log_marginals(packet.unary, edge, packet.masked))
    draws = []
    for candidates in (packet, reference):
        rng = torch.Generator(device=device).manual_seed(9)
        states = core.sample_chain(candidates.unary, edge, rng)
        draws.append(core.sample_candidate_tokens(candidates, states, rng))
    assert torch.equal(*draws)
    assert torch.equal(draws[0][~packet.masked], observed[~packet.masked])


@pytest.mark.parametrize('k', [0, 1, 2, 3])
def test_tiny_tail_and_point_mass_rows(device, k):
    logits = torch.tensor([[[0., -60., -80., -torch.inf],
                            [0., -torch.inf, -torch.inf, -torch.inf]]], device=device)
    observed = torch.tensor([[3, 0]], device=device)
    gold = torch.tensor([[2, 0]], device=device)
    packet = core.build_candidates(logits, observed, 3, k, gold)
    edge = torch.zeros(1, 1, k + 1, k + 1, device=device)
    torch.testing.assert_close(core.gold_log_prob(packet, edge), torch.tensor([-80.], device=device))
    assert packet.candidate_ids[0, 1, 0] == 0
    assert packet.unary[0, 1, 0] == 0
