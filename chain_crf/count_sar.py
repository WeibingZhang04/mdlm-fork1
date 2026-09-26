"""Count-CRF proposals inside the original MDLM semi-autoregressive loop.

This module replaces ONLY _ddpm_caching_update for the duration of a context.
The original loop retains its window layout, stride, cache invalidation,
iteration count, final forward/argmax cleanup and output decoding. No model
weights, model modes, noise schedule, or scoring code are changed.

At positive strength, draw independent reveal indicators with probability
min(dt/t,1) at t>0 (one at t<=0), then reveal a jointly sampled clean proposal
only at those sites. Its selected-block law is the CRF's joint marginal.
The own-marginal control differs only in the clean proposal distribution.
If nothing is revealed, no clean proposal or backbone evaluation is needed.

Unlike the original Gumbel-race update, positive-strength updates separate the
reveal draw from the clean draw. This preserves the specified probability law,
not seed-by-seed outputs. Strength zero delegates the original bound update
without drawing anything, preserving its exact values, cache and RNG state.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import torch


@dataclass(frozen=True)
class CountSARCache:
    """Opaque cache accepted by the unmodified upstream SAR loop.

    Retain original log probabilities rather than an exp/log roundtrip, which
    would drop extremely small probabilities through floating-point underflow.
    The loop only passes this object back or resets it to None.
    """

    log_probs: torch.Tensor
    tokens: torch.Tensor
    time: torch.Tensor


class CountSARUpdate:
    def __init__(self, model, adapter, original_update, *, sampling="joint",
                 generator=None, profile=False):
        if sampling not in ("joint", "marginal"):
            raise ValueError("sampling must be joint or marginal")
        if model.mask_index != adapter.mask_id:
            raise ValueError("model and count adapter mask IDs differ")
        if model.config.noise.type != "loglinear":
            raise ValueError("count SAR requires the original loglinear schedule")
        self.model, self.adapter, self.original_update = model, adapter, original_update
        self.sampling, self.generator, self.profile = sampling, generator, bool(profile)
        self.stats = {"update_calls": 0, "baseline_delegations": 0,
                      "backbone_calls": 0, "cache_hits": 0,
                      "proposal_calls": 0, "empty_reveal_updates": 0,
                      "revealed_tokens": 0}
        self.seconds = {"backbone": 0., "proposal": 0., "total_update": 0.}

    def identity(self):
        return {"format": "count_sar_update_v1", "sampling": self.sampling,
                "denoiser": self.adapter.identity(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reveal_law": "independent_min_dt_over_t_or_one_at_nonpositive_t",
                "cache": "original_log_unaries_unchanged_context",
                "zero_strength": "delegate_original_update_unchanged",
                "final_cleanup": "original_forward_argmax_unchanged",
                "profile_synchronized": self.profile}

    def _stamp(self, device):
        if not self.profile:
            return None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _record(self, name, start, device):
        if start is not None:
            self.seconds[name] += self._stamp(device) - start

    @torch.no_grad()
    def __call__(self, x, t, dt, p_x0=None):
        self.stats["update_calls"] += 1
        started = self._stamp(x.device)
        if self.adapter.strength == 0:
            # No schedule conversion, validation, or random draw before this
            # branch: the original cache/update remains the baseline authority.
            result = self.original_update(x=x, t=t, dt=dt, p_x0=p_x0)
            self.stats["baseline_delegations"] += 1
            self._record("total_update", started, x.device)
            return result
        if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")
        if x.ndim != 2 or x.dtype != torch.long:
            raise ValueError("x must be int64 [batch,length]")
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.shape != (x.shape[0],) or not t.is_floating_point() or t.device != x.device:
            raise ValueError("t must be floating [batch] or [batch,1] on x's device")
        if not bool(torch.isfinite(t).all()):
            raise ValueError("non-finite sampling time")
        if self.model.backbone.training:
            raise ValueError("the original SAR restore/eval wrapper must disable dropout")
        if p_x0 is not None:
            if not isinstance(p_x0, CountSARCache):
                raise TypeError("positive count SAR requires its own original-log-unary cache")
            if not torch.equal(x, p_x0.tokens):
                raise ValueError("stale count cache: visible context changed")
            if self.model.time_conditioning and not torch.equal(t, p_x0.time):
                raise ValueError("stale count cache: conditioned time changed")
        probability = torch.where(t > 0, (dt / t).clamp(max=1.), torch.ones_like(t))
        reveal = torch.rand(x.shape, device=x.device, dtype=t.dtype,
                            generator=self.generator) < probability[:, None]
        reveal &= x.eq(self.adapter.mask_id)
        number_revealed = int(reveal.sum())
        if number_revealed == 0:
            self.stats["empty_reveal_updates"] += 1
            self._record("total_update", started, x.device)
            return p_x0, x
        if p_x0 is None:
            stamp = self._stamp(x.device)
            sigma, _ = self.model.noise(t)
            log_probs = self.model.forward(x, sigma)
            self._record("backbone", stamp, x.device)
            p_x0 = CountSARCache(log_probs, x.clone(), t.clone())
            self.stats["backbone_calls"] += 1
        else:
            self.stats["cache_hits"] += 1
        stamp = self._stamp(x.device)
        result = self.adapter.reveal(p_x0.log_probs, x, reveal,
                                     self.generator, sampling=self.sampling)
        self._record("proposal", stamp, x.device)
        self.stats["proposal_calls"] += 1
        self.stats["revealed_tokens"] += number_revealed
        self._record("total_update", started, x.device)
        return p_x0, result


@contextmanager
def install_count_sar(model, adapter, *, sampling="joint", generator=None,
                      profile=False):
    """Yield counters while the original SAR method runs unchanged.

    Example::

        with install_count_sar(model, adapter) as update:
            result = model.restore_model_and_semi_ar_sample(512, 2, .001)
        counters = update.stats

    Keep the existing runner's token/intermediate capture and scorer. Optional
    profile=True synchronizes CUDA around backbone/proposal/total timing and
    includes that instrumentation overhead in the surrounding wall runtime.
    """
    name = "_ddpm_caching_update"
    previous_instance_value = model.__dict__.get(name)
    had_instance_value = name in model.__dict__
    update = CountSARUpdate(model, adapter, getattr(model, name),
                            sampling=sampling, generator=generator, profile=profile)
    setattr(model, name, update)
    try:
        yield update
    finally:
        if had_instance_value:
            setattr(model, name, previous_instance_value)
        else:
            delattr(model, name)
