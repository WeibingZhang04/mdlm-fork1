"""SAR update-law/zero-strength checks, including optional actual upstream AST."""

import ast
import copy
import itertools
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chain_crf.count_denoiser import CountCRFDenoiser
from chain_crf.count_sar import CountSARCache, install_count_sar
from chain_crf.counts import CountBigramHead


class ToyModel:
    def __init__(self, time_conditioning=False):
        self.mask_index, self.vocab_size = 3, 4
        self.time_conditioning = time_conditioning
        self.backbone = torch.nn.Identity().eval()
        self.config = SimpleNamespace(noise=SimpleNamespace(type="loglinear"),
                                      model=SimpleNamespace(length=4))
        self.dtype, self.device = torch.float64, torch.device("cpu")
        self.forward_calls = 0
        self.tokenizer = SimpleNamespace(eos_token_id=2, batch_decode=lambda x: x.copy().tolist())

    def noise(self, t):
        return t + .3, torch.ones_like(t)

    def forward(self, x, sigma):
        self.forward_calls += 1
        raw = torch.tensor([-.2, .3, -.5, -1e6], dtype=torch.float64).expand(*x.shape, 4).clone()
        if self.time_conditioning:
            raw[..., 0] += sigma[:, None]
        raw -= raw.logsumexp(-1, keepdim=True)
        visible = x.ne(self.mask_index)
        raw.masked_fill_(visible[..., None], -1e6)
        raw.scatter_(-1, x[..., None], torch.where(visible, 0., raw.gather(-1, x[..., None]).squeeze(-1))[..., None])
        return raw

    def _sample_prior(self, batch, length):
        return torch.full((batch, length), self.mask_index)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        # A small literal test double; optional tests below execute the exact
        # upstream source supplied by the caller, not this reimplementation.
        sigma, _ = self.noise(t)
        t = t.reshape(-1)
        if p_x0 is None:
            p_x0 = self.forward(x, sigma).exp()
        q = p_x0 * dt
        q[..., self.mask_index] = (t - dt)[:, None]
        race = 1e-10 - (torch.rand_like(q) + 1e-10).log()
        proposal = (q / race).argmax(-1)
        return p_x0, torch.where(x != self.mask_index, x, proposal)


def adapter(strength=.1, mode="conditional"):
    head = CountBigramHead(4, smoothing=.2).fit([[0, 1, 0, 1], [2, 2, 1], [0, 1, 2]])
    return CountCRFDenoiser(head, mask_id=3, mode=mode, strength=strength,
                           sparse_chunk_size=4096)


def run_updates(model, steps=4, windows=3):
    target, intermediate, changed = None, [], 0
    for _ in range(windows):
        x = model._sample_prior(1, 4)
        if target is not None:
            x[:, :2] = target
        cache = None
        for i in range(steps + 1):
            cache, next_x = model._ddpm_caching_update(x=x, t=torch.tensor([1 - i / steps], dtype=torch.float64),
                                                       dt=1 / steps, p_x0=cache)
            if not torch.equal(x, next_x) or model.time_conditioning:
                cache = None
                changed += 1
            x = next_x
        assert x.ne(3).all()
        clean = model.forward(x, torch.zeros(1)).argmax(-1)
        assert torch.equal(x, clean)
        intermediate.append(x[:, :2].clone())
        target = x[:, 2:]
    intermediate.append(target)
    return torch.cat(intermediate, dim=1), changed


@pytest.mark.parametrize("time_conditioning", [False, True])
def test_zero_strength_complete_loop_exact_tokens_steps_and_rng(time_conditioning):
    plain, counted = ToyModel(time_conditioning), ToyModel(time_conditioning)
    initial = torch.Generator().manual_seed(711).get_state()
    torch.random.set_rng_state(initial)
    original = run_updates(plain)
    original_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial)
    with install_count_sar(counted, adapter(0.)) as update:
        result = run_updates(counted)
    assert torch.equal(original[0], result[0]) and original[1] == result[1]
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert plain.forward_calls == counted.forward_calls
    assert update.stats["baseline_delegations"] == 15
    assert update.stats["proposal_calls"] == 0
    assert "_ddpm_caching_update" not in counted.__dict__


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_full_loop_visible_prefix_and_cleanup(sampling):
    model = ToyModel()
    with install_count_sar(model, adapter(), sampling=sampling,
                           generator=torch.Generator().manual_seed(55), profile=True) as update:
        result, changed = run_updates(model)
        assert result.shape == (1, 8) and result.ne(3).all()
    assert update.stats["update_calls"] == 15
    assert update.stats["revealed_tokens"] == 8  # 4 first window, 2 each later window.
    assert 3 <= changed <= 12
    assert update.stats["proposal_calls"] == changed
    assert update.seconds["proposal"] > 0 and update.seconds["total_update"] > 0
    assert update.identity()["denoiser"]["mode"] == "conditional"


@pytest.mark.parametrize("sampling", ["joint", "marginal"])
def test_positive_transition_matches_joint_selected_block_and_bernoulli(sampling):
    model, denoiser, n = ToyModel(), adapter(1.), 24000
    one = torch.tensor([[3, 3]])
    logs = model.forward(one, torch.tensor([.8], dtype=torch.float64))
    states = torch.tensor(list(itertools.product(range(4), repeat=2)))
    selected = states.ne(3)
    repeated_logs, repeated_x = logs.expand(16, -1, -1), one.expand(16, -1)
    if sampling == "joint":
        clean = denoiser.selected_log_probability(repeated_logs, repeated_x, selected, states).exp()
    else:
        marginals = denoiser(logs, one).exp()
        probability = marginals.expand(16, -1, -1).gather(-1, states[..., None]).squeeze(-1)
        clean = torch.where(selected, probability, 1.).prod(-1)
    r = torch.tensor(.25, dtype=torch.float64)
    expected = clean * (r ** selected.sum(-1)) * ((1 - r) ** (~selected).sum(-1))
    torch.testing.assert_close(expected.sum(), torch.tensor(1., dtype=torch.float64), atol=1e-14, rtol=0.)
    with install_count_sar(model, denoiser, sampling=sampling,
                           generator=torch.Generator().manual_seed(835)):
        _, actual = model._ddpm_caching_update(one.expand(n, -1), torch.ones(n, dtype=torch.float64), .25)
    frequency = torch.bincount(actual[:, 0] * 4 + actual[:, 1], minlength=16).double() / n
    torch.testing.assert_close(frequency, expected, atol=.009, rtol=0.)


def test_zero_strength_transition_law_matches_original_factorized_q():
    model, denoiser = ToyModel(), adapter(0.)
    states = torch.tensor(list(itertools.product(range(4), repeat=2)))
    xt = torch.full((16, 2), 3)
    logs = model.forward(xt, torch.ones(16, dtype=torch.float64))
    selected = states.ne(3)
    r = torch.tensor(.2, dtype=torch.float64)
    clean = denoiser.selected_log_probability(logs, xt, selected, states).exp()
    count_transition = clean * r ** selected.sum(-1) * (1 - r) ** (~selected).sum(-1)
    original_weights = logs.exp() * .1
    original_weights[..., 3] = .5 - .1
    original_transition = (original_weights / .5).gather(-1, states[..., None]).squeeze(-1).prod(-1)
    torch.testing.assert_close(count_transition, original_transition, atol=2e-15, rtol=2e-15)


def test_empty_reveal_skips_forward_and_proposal_without_changing_law():
    model = ToyModel()
    with install_count_sar(model, adapter(), generator=torch.Generator().manual_seed(9)) as update:
        x = torch.full((1, 2), 3)
        cache, result = model._ddpm_caching_update(x, torch.ones(1), 1e-100)
    assert result is x and cache is None
    assert model.forward_calls == 0 and update.stats["proposal_calls"] == 0
    assert update.stats["empty_reveal_updates"] == 1


def test_cache_is_raw_logs_and_reuse_requires_exact_context():
    model = ToyModel()
    x, t = torch.tensor([[0, 3]]), torch.tensor([.2], dtype=torch.float64)
    with install_count_sar(model, adapter(), generator=torch.Generator().manual_seed(90)) as update:
        cache, first = model._ddpm_caching_update(x, t, .2)
        assert isinstance(cache, CountSARCache)
        assert cache.log_probs[0, 0, 3] == -1e6  # No exp/log conversion.
        calls = model.forward_calls
        _, second = model._ddpm_caching_update(x, t / 2, .2, cache)
        assert model.forward_calls == calls and update.stats["cache_hits"] == 1
        assert first[0, 0] == second[0, 0] == 0
        with pytest.raises(ValueError, match="context"):
            model._ddpm_caching_update(first, t, .2, cache)
        with pytest.raises(TypeError, match="cache"):
            model._ddpm_caching_update(x, t, .2, cache.log_probs.exp())
    model = ToyModel(time_conditioning=True)
    with install_count_sar(model, adapter()):
        cache, _ = model._ddpm_caching_update(x, t, .2)
        with pytest.raises(ValueError, match="time"):
            model._ddpm_caching_update(x, t / 2, .2, cache)


def test_endpoint_reveals_all_and_visible_values_never_change():
    model = ToyModel()
    with install_count_sar(model, adapter()):
        x = torch.tensor([[0, 3, 3, 1]])
        _, final = model._ddpm_caching_update(x, torch.zeros(1), .001)
        assert final.ne(3).all() and torch.equal(final[:, [0, 3]], x[:, [0, 3]])
        _, at_dt = model._ddpm_caching_update(x, torch.tensor([.001]), .001)
        assert at_dt.ne(3).all()


def test_context_restores_existing_override_on_exception():
    model = ToyModel()
    previous = model._ddpm_caching_update
    model._ddpm_caching_update = previous
    with pytest.raises(RuntimeError):
        with install_count_sar(model, adapter()):
            raise RuntimeError("intentional")
    assert model._ddpm_caching_update is previous


def test_invalid_schedule_mode_dropout_and_time_fail():
    model = ToyModel()
    with pytest.raises(ValueError):
        with install_count_sar(model, adapter(), sampling="greedy"):
            pass
    model.config.noise.type = "other"
    with pytest.raises(ValueError):
        with install_count_sar(model, adapter()):
            pass
    model.config.noise.type = "loglinear"
    with install_count_sar(model, adapter()):
        with pytest.raises(ValueError):
            model._ddpm_caching_update(torch.full((1, 2), 3), torch.tensor([float("nan")]), .1)
        with pytest.raises(ValueError):
            model._ddpm_caching_update(torch.full((1, 2), 3), torch.ones(1), 0.)
        model.backbone.train()
        with pytest.raises(ValueError, match="dropout"):
            model._ddpm_caching_update(torch.full((1, 2), 3), torch.ones(1), .1)


def load_upstream_methods():
    """Optional exact-source check, without an embedded machine/private path.

    MDLM_UPSTREAM_DIFFUSION names diffusion.py, or MDLM_UPSTREAM_ARCHIVE names
    an audited tar archive containing upstream/diffusion.py. No imports from
    that source execute; only three specified function ASTs are compiled.
    """
    source_path = os.environ.get("MDLM_UPSTREAM_DIFFUSION")
    archive_path = os.environ.get("MDLM_UPSTREAM_ARCHIVE")
    if source_path:
        source = Path(source_path).read_text()
    elif archive_path:
        with tarfile.open(archive_path) as archive:
            source = archive.extractfile("upstream/diffusion.py").read().decode()
    else:
        pytest.skip("Set MDLM_UPSTREAM_DIFFUSION or MDLM_UPSTREAM_ARCHIVE for exact upstream AST test")
    tree, selected = ast.parse(source), []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_sample_categorical":
            selected.append(copy.deepcopy(node))
        if isinstance(node, ast.ClassDef) and node.name == "Diffusion":
            selected.extend(copy.deepcopy(item) for item in node.body if isinstance(item, ast.FunctionDef)
                            and item.name in ("_ddpm_caching_update", "sample_subs_guidance"))
    assert len(selected) == 3
    for node in selected:
        node.decorator_list = []
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 "<provided-upstream-mdlm>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("time_conditioning", [False, True])
@pytest.mark.parametrize("length,stride,dt", [(4, 2, .25), (1024, 512, .001)])
def test_actual_upstream_sar_zero_strength_exact_rng_and_intermediates(time_conditioning, length, stride, dt):
    functions = load_upstream_methods()
    class UpstreamToy(ToyModel):
        _ddpm_caching_update = functions["_ddpm_caching_update"]
        sample_subs_guidance = functions["sample_subs_guidance"]
    original, counted = UpstreamToy(time_conditioning), UpstreamToy(time_conditioning)
    original.config.model.length = counted.config.model.length = length
    rng = torch.Generator().manual_seed(457).get_state()
    torch.random.set_rng_state(rng)
    expected = original.sample_subs_guidance(1, stride, 2, dt)
    final_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng)
    with install_count_sar(counted, adapter(0.)):
        actual = counted.sample_subs_guidance(1, stride, 2, dt)
    assert actual[0] == expected[0] and actual[1] == expected[1]
    np.testing.assert_array_equal(actual[2], expected[2])
    assert torch.equal(torch.random.get_rng_state(), final_rng)
    assert original.forward_calls == counted.forward_calls
    assert np.asarray(actual[1][-1]).shape == (1, length + 2 * stride)


def test_actual_upstream_accepts_positive_opaque_cache_and_original_cleanup():
    functions = load_upstream_methods()
    class UpstreamToy(ToyModel):
        _ddpm_caching_update = functions["_ddpm_caching_update"]
        sample_subs_guidance = functions["sample_subs_guidance"]
    model = UpstreamToy()
    with install_count_sar(model, adapter(), generator=torch.Generator().manual_seed(817)) as update:
        _, intermediate, _ = model.sample_subs_guidance(1, 2, 2, .25)
    assert len(intermediate) == 3
    assert np.asarray(intermediate[-1]).shape == (1, 8)
    assert 3 not in np.asarray(intermediate[-1])
    assert update.stats["revealed_tokens"] == 8
