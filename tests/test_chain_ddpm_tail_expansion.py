"""Selective tail expansion preserves the revealed CRF law and avoids unused draws."""
import math
import pytest
import torch

from chain_crf.core import build_candidates, sample_candidate_tokens
from test_chain_ddpm_cache import Backbone, make_head, run


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("cap", [None, 2])
@pytest.mark.parametrize("selection", ["none", "mixed", "all"])
def test_only_selected_residual_rows_reach_multinomial(monkeypatch, device, cap, selection):
    scores = torch.tensor([.5, .3, .2, 0.], device=device).log().expand(1, 4, 4)
    tokens = torch.tensor([[3, 2, 3, 3]], device=device)
    packet = build_candidates(scores, tokens, 3, 1, vocab_cap=cap)
    states = torch.tensor([[1, 0, 1, 0]], device=device)
    resolve = torch.tensor([[True, False, False, True]], device=device)
    if selection == "none":
        resolve.fill_(False)
    elif selection == "all":
        resolve.fill_(True)
    initial_ids = packet.candidate_ids.gather(-1, states[..., None]).squeeze(-1)
    selected_tail = initial_ids.lt(0) & resolve
    before = packet.normalized_log_probs.clone()
    rng = torch.Generator(device=device).manual_seed(9)
    before_rng = rng.get_state()
    shapes = []
    original = torch.multinomial

    def record(probabilities, *args, **kwargs):
        shapes.append(probabilities.shape)
        return original(probabilities, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", record)
    actual = sample_candidate_tokens(packet, states, rng, resolve_mask=resolve)
    n = int(selected_tail.sum())
    assert shapes == ([torch.Size([n, 4])] if n else [])
    assert torch.equal(rng.get_state(), before_rng) == (n == 0)
    assert actual[initial_ids.lt(0) & ~resolve].eq(-1).all()
    assert actual[initial_ids.ge(0)].equal(initial_ids[initial_ids.ge(0)])
    assert ((actual[selected_tail] >= 1) & (actual[selected_tail] <= (1 if cap else 2))).all()
    torch.testing.assert_close(packet.normalized_log_probs, before, rtol=0, atol=0)
    if selection == "all":
        reference_rng = torch.Generator(device=device).manual_seed(9)
        expected = sample_candidate_tokens(packet, states, reference_rng)
        assert torch.equal(actual, expected)
        assert torch.equal(rng.get_state(), reference_rng.get_state())


@pytest.mark.parametrize("resolve", [torch.ones(1, 2), torch.ones(2, dtype=torch.bool)])
def test_invalid_resolution_mask_is_rejected(resolve):
    packet = build_candidates(torch.zeros(1, 2, 4), torch.full((1, 2), 3), 3, 1)
    with pytest.raises(ValueError, match="resolve_mask"):
        sample_candidate_tokens(packet, torch.zeros(1, 2, dtype=torch.long), resolve_mask=resolve)


@pytest.mark.parametrize("mode", ["count", "global", "contextual", "independent"])
@pytest.mark.parametrize("cap", [None, 2])
def test_ddpm_expands_each_committed_tail_once(monkeypatch, mode, cap):
    import chain_crf.generation as generation
    original = generation.sample_candidate_tokens
    counts = []

    def record(packet, states, generator=None, *, resolve_mask=None):
        assert resolve_mask is not None
        residual = packet.candidate_ids.gather(-1, states[..., None]).squeeze(-1).lt(0)
        counts.append((int(residual.sum()), int((residual & resolve_mask).sum())))
        return original(packet, states, generator, resolve_mask=resolve_mask)

    monkeypatch.setattr(generation, "sample_candidate_tokens", record)
    initial = torch.full((2, 24), 6)
    initial[:, :2] = torch.tensor([4, 5])
    # K=0 makes every masked state a tail; unrevealed rows must not be resolved.
    result, _ = run(Backbone(), initial.clone(), steps=5, mode=mode, head=make_head(mode),
                    k=0, vocab_cap=cap, inference="segments", eps=.2, noise_removal=False)
    assert len(counts) == 5  # Full proposals still occur even if reveals are empty.
    committed = int((initial.eq(6) & result.ne(6)).sum())
    assert sum(selected for _, selected in counts) == committed
    assert sum(full for full, _ in counts) > committed
    assert result.ge(0).all()  # No unresolved sentinel reaches the backbone.
    assert result[:, :2].equal(initial[:, :2])


def test_no_reveals_do_not_expand_any_tail(monkeypatch):
    import chain_crf.generation as generation
    original = generation.sample_candidate_tokens
    calls = []

    def record(packet, states, generator=None, *, resolve_mask=None):
        assert resolve_mask is not None and not resolve_mask.any()
        before = generator.get_state()
        result = original(packet, states, generator, resolve_mask=resolve_mask)
        assert torch.equal(generator.get_state(), before)
        calls.append(result)
        return result

    monkeypatch.setattr(generation, "sample_candidate_tokens", record)
    initial = torch.full((1, 4), 6)
    result, _ = run(Backbone(), initial.clone(), steps=4, mode="global",
                    head=make_head("global"), k=0, eps=.999999, noise_removal=False)
    assert len(calls) == 4 and result.equal(initial)


@pytest.mark.parametrize("inference", ["dense", "segments"])
@pytest.mark.parametrize("cap", [None, 2])
def test_partial_reveal_distribution_matches_enumerated_full_token_crf(inference, cap):
    class ThreeTokenBackbone:
        mask_id = 3
        time_conditioning = False

        def __call__(self, tokens, t):
            return {"log_probs": torch.tensor([.5, .3, .2, 0.]).log().expand(*tokens.shape, 4),
                    "hidden": torch.zeros(*tokens.shape, 1)}

    class FavorExplicitPair(torch.nn.Module):
        def forward(self, ids, hidden, t):
            return 1.3 * (ids[:, :-1, :, None].eq(0) & ids[:, 1:, None, :].eq(0)).float()

    result, _ = run(ThreeTokenBackbone(), torch.full((20000, 2), 3), steps=1,
                    mode="global", head=FavorExplicitPair(), k=1, vocab_cap=cap,
                    inference=inference, eps=.4, noise_removal=False)
    probabilities = torch.tensor([.5, .3, .2], dtype=torch.float64)
    if cap:
        probabilities[cap:] = 0
        probabilities /= probabilities.sum()
    # Enumerate the full clean-token law: only explicit (0,0) receives a bonus.
    joint = probabilities[:, None] * probabilities[None, :]
    joint[0, 0] *= math.exp(1.3)
    joint /= joint.sum()
    expected = torch.zeros(4, 4, dtype=torch.float64)
    expected[:3, :3] = joint * .6**2
    expected[:3, 3] = joint.sum(1) * .6 * .4
    expected[3, :3] = joint.sum(0) * .6 * .4
    expected[3, 3] = .4**2
    empirical = torch.bincount(result[:, 0] * 4 + result[:, 1], minlength=16).double() / len(result)
    torch.testing.assert_close(empirical, expected.flatten(), atol=.008, rtol=0)
