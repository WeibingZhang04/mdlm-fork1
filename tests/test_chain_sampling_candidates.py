"""Compact inference support agrees with the differentiable reference law."""
import pytest
import torch

from chain_crf.core import (
    build_candidates, build_sampling_candidates, sample_candidate_tokens,
)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cap", [None, 1, 5, 30])
@pytest.mark.parametrize("k", [0, 3, 30])
@pytest.mark.parametrize("pattern", ["all", "mixed", "none"])
@pytest.mark.parametrize("temperature", [.7, 1.])
def test_support_and_conditional_tail_match_reference(device, cap, k, pattern, temperature):
    rng = torch.Generator().manual_seed(123)
    # Noncontiguous scores and a nonterminal MASK exercise indexing assumptions.
    scores = torch.randn(2, 13, 4, generator=rng, dtype=torch.float64).transpose(1, 2).to(device)
    tokens = torch.full((2, 4), 6, device=device)
    if pattern == "mixed":
        tokens[:, ::2] = torch.tensor([12, 0], device=device)
    elif pattern == "none":
        tokens.fill_(12)
    original = scores.clone()
    with torch.no_grad():
        reference = build_candidates(scores / temperature, tokens, 6, k, vocab_cap=cap)
        actual = build_sampling_candidates(scores, tokens, 6, k,
                                            vocab_cap=cap, temperature=temperature)
    assert actual.candidate_ids.equal(reference.candidate_ids)
    assert actual.masked.equal(reference.masked)
    torch.testing.assert_close(actual.unary, reference.unary, rtol=1e-12, atol=1e-12)
    assert scores.equal(original)
    residual = actual.masked & actual.unary[..., -1].isfinite()
    if residual.any():
        ref_scores, _ = reference.residual_distribution(residual)
        new_scores, tail_ids = actual.residual_distribution(residual)
        probs = new_scores.double().softmax(-1)
        if tail_ids is not None:
            probs = torch.zeros_like(ref_scores).scatter(-1, tail_ids, probs)
        torch.testing.assert_close(probs, ref_scores.double().softmax(-1), rtol=1e-12, atol=1e-12)
        assert scores.equal(original)
    if cap is None:
        assert actual.source_log_probs is scores
    elif actual.masked.any():
        assert actual.source_log_probs is None
        assert actual.cap_log_probs.shape == (int(actual.masked.sum()), min(cap, 12))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cap", [None, 5])
def test_sampling_precision_and_input_ownership(device, dtype, cap):
    # Distinct exactly representable scores keep quantization out of top-K ties.
    scores = torch.arange(13, device=device, dtype=dtype).expand(2, 3, 13).clone()
    for mixed in (False, True):
        tokens = torch.full((2, 3), 6, device=device)
        if mixed:
            tokens[:, 0] = 12
        before = scores.clone()
        with torch.no_grad():
            ref = build_candidates(scores, tokens, 6, 2, vocab_cap=cap)
        got = build_sampling_candidates(scores, tokens, 6, 2, vocab_cap=cap)
        assert got.candidate_ids.equal(ref.candidate_ids)
        torch.testing.assert_close(got.unary, ref.unary, rtol=2e-6, atol=2e-6)
        states = torch.where(got.masked, 2, 0)
        resolve = got.masked.clone()
        resolve[:, -1] = False
        output = sample_candidate_tokens(got, states, resolve_mask=resolve)
        assert output[got.masked & ~resolve].eq(-1).all()
        assert output[~got.masked].eq(12).all()
        assert output[resolve].ge(0).all() and output[resolve].ne(6).all()
        assert scores.equal(before)


@pytest.mark.parametrize("device", DEVICES)
def test_tiny_residual_mass_and_impossible_token_ties(device):
    scores = torch.tensor([0., -100., -101., -torch.inf, -torch.inf], device=device).reshape(1, 1, 5)
    for cap in (None, 4):
        ref = build_candidates(scores, torch.tensor([[3]], device=device), 3, 1, vocab_cap=cap)
        got = build_sampling_candidates(scores, torch.tensor([[3]], device=device), 3, 1, vocab_cap=cap)
        assert got.unary[..., -1].isfinite().all()
        torch.testing.assert_close(got.unary, ref.unary)
        full = build_sampling_candidates(scores, torch.tensor([[3]], device=device), 3, 9, vocab_cap=cap)
        assert full.candidate_ids[..., :-1].ne(3).all()
        assert full.unary[..., -1].isneginf().all()


@pytest.mark.parametrize("cap", [None, 5])
def test_reductions_skip_visible_rows(monkeypatch, cap):
    scores = torch.randn(2, 4, 13)
    tokens = torch.tensor([[6, 0, 1, 2], [0, 1, 6, 2]])
    reductions = []
    original = torch.logsumexp

    def record(values, *args, **kwargs):
        reductions.append(tuple(values.shape))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch, "logsumexp", record)
    build_sampling_candidates(scores, tokens, 6, 2, vocab_cap=cap)
    assert reductions == ([(2, 13), (2, 13)] if cap is None else [(2, 5), (2, 3)])
