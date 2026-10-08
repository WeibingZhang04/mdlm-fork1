"""Fixed reveals use compact candidates and resolve only committed tails."""
import math
import pytest
import torch

from chain_crf.core import SamplingCandidateBatch
from chain_crf.generation import generate
from chain_crf import GlobalPairHead
from test_chain_ddpm_cache import Backbone, make_head
from test_chain_generation import ToyBackbone


@pytest.mark.parametrize("mode", ["count", "global", "contextual", "independent"])
@pytest.mark.parametrize("cap", [None, 2])
@pytest.mark.parametrize("inference,sampling", [("dense", "joint"), ("segments", "joint"),
                                             ("segments", "marginal")])
@pytest.mark.parametrize("steps", [3, 16])
def test_fixed_resolves_each_tail_once_and_keeps_schedule(monkeypatch, mode, cap, inference, sampling, steps):
    import chain_crf.generation as generation
    original = generation.sample_candidate_tokens
    seen = torch.zeros((2, 9), dtype=torch.long)
    counts = []

    def record(packet, states, generator=None, *, resolve_mask=None):
        assert isinstance(packet, SamplingCandidateBatch)
        assert states.shape == (2, 9)  # Joint states include unrevealed positions.
        assert resolve_mask is not None
        assert not (resolve_mask & ~packet.masked).any()
        seen.add_(resolve_mask.long())
        residual = packet.candidate_ids.gather(-1, states[..., None]).squeeze(-1).lt(0)
        counts.append((int(residual.sum()), int((residual & resolve_mask).sum())))
        result = original(packet, states, generator, resolve_mask=resolve_mask)
        assert result[residual & ~resolve_mask].eq(-1).all()
        assert result[resolve_mask].ge(0).all()
        return result

    monkeypatch.setattr(generation, "sample_candidate_tokens", record)
    backbone, baseline = Backbone(), Backbone()
    options = dict(length=7, steps=steps, batch_size=2, k=0, temperature=.7,
                   device="cpu", prefix=[4, 5], sample_offset=17,
                   inference=inference, sampling=sampling, vocab_cap=cap, sampler="fixed")
    output, stats = generate(backbone, make_head(mode), mode, **options)
    generate(baseline, mode="backbone", **options)
    assert seen[:, :2].eq(0).all() and seen[:, 2:].eq(1).all()
    assert sum(selected for _, selected in counts) == 14
    assert sum(full for full, _ in counts) > 14
    assert len(counts) == stats["backbone_calls"] == min(steps, 7)
    assert output[:, :2].equal(torch.tensor([[4, 5], [4, 5]]))
    assert output.ge(0).all() and output.ne(6).all()
    if cap is not None:
        assert output[:, 2:].lt(cap).all()
    for actual, expected in zip(backbone.history, baseline.history):
        assert actual.eq(6).equal(expected.eq(6))


@pytest.mark.parametrize("cap", [None, 2])
@pytest.mark.parametrize("temperature", [1., .7])
def test_first_fixed_reveal_matches_full_joint_crf_marginal(cap, temperature):
    class ThreeTokenBackbone:
        vocab_size = 4
        mask_id = 3
        def __init__(self):
            self.history = []
        def __call__(self, tokens, t):
            self.history.append(tokens.clone())
            return {"log_probs": torch.tensor([.5, .3, .2, 0.]).log().expand(*tokens.shape, 4),
                    "hidden": torch.zeros(*tokens.shape, 1)}

    class FavorExplicitPair(torch.nn.Module):
        def forward(self, ids, hidden, t):
            return 1.3 * (ids[:, :-1, :, None].eq(0) & ids[:, 1:, None, :].eq(0)).float()

    backbone = ThreeTokenBackbone()
    generate(backbone, FavorExplicitPair(), "global", length=2, steps=2, batch_size=12000,
             k=1, temperature=temperature, vocab_cap=cap, device="cpu",
             sampler="fixed", inference="segments")
    after_first = backbone.history[1]
    assert after_first.ne(3).sum(-1).eq(1).all()
    revealed = after_first[after_first.ne(3)]
    weights = torch.tensor([.5, .3, .2], dtype=torch.float64).pow(1 / temperature)
    if cap is not None:
        weights[cap:] = 0.
    joint = weights[:, None] * weights[None, :]
    joint[0, 0] *= math.exp(1.3)
    joint /= joint.sum()
    empirical = torch.bincount(revealed, minlength=3).double() / len(revealed)
    torch.testing.assert_close(empirical, joint.sum(1), atol=.012, rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("cap", [None, 3])
def test_fixed_compact_sampling_with_explicit_states_is_reproducible(device, cap):
    head = GlobalPairHead(5, 3).to(device)
    options = dict(length=9, steps=4, batch_size=2, k=1, temperature=.7,
                   device=device, prefix=[2, 1], vocab_cap=cap, sampler="fixed", inference="segments")
    first, _ = generate(ToyBackbone(), head, "global", **options)
    second, _ = generate(ToyBackbone(), head, "global", **options)
    assert first.equal(second)
    assert first[:, :2].eq(torch.tensor([2, 1], device=device)).all()
    assert first.ge(0).all() and first.ne(4).all()
