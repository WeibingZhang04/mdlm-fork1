"""Frozen full-vocabulary count CRFs for an existing masked denoiser.

Given the *original* denoiser log probabilities u_i(a), this adapter defines

    q(x | x_t, t) = exp(sum_i u_i(x_i)) prod_i W(x_i, x_{i+1}) / Z.

Visible tokens are hard constraints; a clean token can never be the mask.
For the smoothed joint J from CountBigramHead, the two distinct methods are
W(a,b)=[J(a,b)/J_L(a)]**strength (conditional, the default) and
W(a,b)=[J(a,b)/(J_L(a) J_R(b))]**strength (PMI). All original adjacencies
remain, including edges through visible tokens. There is no top-k tail.

Calling the adapter returns log SINGLETON marginals, suitable for replacing
the original denoiser in an unchanged continuous-time masked-diffusion loss.
That objective measures an infinitesimal single-token generator; it is NOT a
joint masked-token log probability divided by time. propose/reveal instead
implement a joint clean proposal followed by an externally supplied reveal
mask. selected_log_probability gives its exact selected-block marginal.

Time dependence, model mode/dropout, corruption, loss weights, reveal schedule,
and final cleanup belong to the caller. Supply original backbone probabilities
to propose, never the output of this adapter (which would apply the CRF twice).
Inference is FP64 and frozen. Chunking preserves the probability law, not the
RNG sequence; strength-zero __call__ is the sole bitwise baseline bypass.
"""

import hashlib
from pathlib import Path

import torch

from chain_crf.counts import CountBigramHead
from chain_crf import sparse_count, sparse_count_gpu, sparse_count_segments


_BACKENDS = {
    "reference": (sparse_count, sparse_count.SparseCountPotential),
    "gpu": (sparse_count_gpu, sparse_count_gpu.GPUCountPotential),
    "segments": (sparse_count_segments, sparse_count_segments.SegmentedCountPotential),
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CountCRFDenoiser:
    """Build the static count potential once, then reuse for frozen inference.

    Full marginal output takes B*L*V*8 bytes before casting back to the input
    dtype; sparse_chunk_size bounds span work, not this unavoidable output.
    Use evaluation microbatches with saved original corruptions when needed.
    """

    def __init__(self, head, *, mask_id, mode="conditional", strength=1.,
                 backend="segments", sparse_chunk_size=512):
        if backend not in _BACKENDS:
            raise ValueError("backend must be reference, gpu, or segments")
        if isinstance(mask_id, bool) or not isinstance(mask_id, int) or not 0 <= mask_id < head.vocab_size:
            raise ValueError("mask_id must be an integer in the count vocabulary")
        if isinstance(sparse_chunk_size, bool) or not isinstance(sparse_chunk_size, int) or sparse_chunk_size < 1:
            raise ValueError("sparse_chunk_size must be a positive integer")
        self.module, potential_class = _BACKENDS[backend]
        # The exact potential constructor validates mode, strength and counts.
        self.potential = potential_class.from_head(head, mode=mode, strength=strength)
        self.mask_id, self.mode, self.strength = mask_id, mode, float(strength)
        self.backend, self.sparse_chunk_size = backend, sparse_chunk_size
        self.smoothing = float(head.smoothing)
        self.counts_sha256 = None

    @classmethod
    def from_file(cls, counts_path, *, mask_id, mode="conditional", strength=1.,
                  backend="segments", sparse_chunk_size=512, device="cpu"):
        """Read authenticated count bytes; no model or dataset is downloaded."""
        before = _sha256(counts_path)
        head = CountBigramHead.load(counts_path, mode=mode, strength=strength).to(device)
        result = cls(head, mask_id=mask_id, mode=mode, strength=strength,
                     backend=backend, sparse_chunk_size=sparse_chunk_size)
        if _sha256(counts_path) != before:
            raise RuntimeError("count file changed while loading")
        result.counts_sha256 = before
        return result

    def identity(self):
        """Portable source/model identity for experiment manifests and resume."""
        directory = Path(__file__).resolve().parent
        names = ("count_denoiser.py", "counts.py", "sparse_count.py",
                 "sparse_count_gpu.py", "sparse_count_segments.py")
        return {"format": "count_crf_denoiser_v1", "mode": self.mode,
                "strength": self.strength, "smoothing": self.smoothing,
                "mask_id": self.mask_id, "vocab_size": self.potential.vocab_size,
                "backend": self.backend, "sparse_chunk_size": self.sparse_chunk_size,
                "counts_sha256": self.counts_sha256,
                "arithmetic": "float64", "time_dependence": "original_unaries_only",
                "visible_tokens": "exact_clamp", "clean_mask": "forbidden",
                "zero_strength_call": "unchanged_input_no_rng",
                "source_sha256": {"chain_crf/" + name: _sha256(directory / name) for name in names}}

    def _validate(self, log_probs, xt):
        if log_probs.ndim != 3 or not log_probs.is_floating_point():
            raise ValueError("log_probs must be floating [batch,length,vocabulary]")
        if min(log_probs.shape) < 1 or log_probs.shape[-1] != self.potential.vocab_size:
            raise ValueError("log_probs has empty dimensions or wrong vocabulary")
        if xt.shape != log_probs.shape[:2] or xt.dtype != torch.long:
            raise ValueError("xt must be int64 [batch,length]")
        if xt.device != log_probs.device or log_probs.device != self.potential.left.device:
            raise ValueError("tokens, unaries and count potential must share a device")
        if bool(((xt < 0) | (xt >= self.potential.vocab_size)).any()):
            raise ValueError("xt token outside vocabulary")

    def _unary(self, log_probs, xt):
        self._validate(log_probs, xt)
        # Copy even FP64 inputs: the caller may cache its original unaries.
        unary = log_probs.to(torch.float64).clone()
        unary[..., self.mask_id] = -torch.inf
        visible = xt.ne(self.mask_id)
        unary.masked_fill_(visible[..., None], -torch.inf)
        current = unary.gather(-1, xt[..., None])
        unary.scatter_(-1, xt[..., None], torch.where(visible[..., None], 0., current))
        return unary

    def _run(self, name, unary, **kwargs):
        if self.backend == "segments":
            kwargs["max_chunk_tokens"] = self.sparse_chunk_size
        return getattr(self.module, name)(unary, self.potential, **kwargs)

    @torch.no_grad()
    def __call__(self, log_probs, xt, sigma=None):
        """Log singleton marginals, with the same shape/dtype as log_probs.

        sigma is accepted for forward-hook compatibility only. At strength 0,
        return the original tensor object unchanged, including its finite mask
        sentinel, for exact official continuous-time loss/RNG parity.
        """
        self._validate(log_probs, xt)
        if self.strength == 0:
            return log_probs
        unary = self._unary(log_probs, xt)
        marginals = self._run("sparse_chain_marginals", unary)
        return marginals.log_().to(log_probs.dtype)

    @torch.no_grad()
    def propose(self, log_probs, xt, generator=None, *, sampling="joint"):
        """Sample clean tokens jointly, or independently from own marginals.

        Even at zero strength, this is law-equivalent, not RNG-identical, to
        the original MDLM categorical sampler. Delegate the complete original
        update at strength zero if bitwise baseline sampling parity is needed.
        """
        if sampling not in ("joint", "marginal"):
            raise ValueError("sampling must be joint or marginal")
        unary = self._unary(log_probs, xt)
        if sampling == "joint":
            return self._run("sample_sparse_chain", unary, generator=generator)
        marginal = self._run("sparse_chain_marginals", unary)
        return torch.multinomial(marginal.reshape(-1, marginal.shape[-1]), 1,
                                 generator=generator).reshape(xt.shape)

    @torch.no_grad()
    def reveal(self, log_probs, xt, reveal_mask, generator=None, *, sampling="joint"):
        """Apply a supplied independent reveal mask; never resample the mask.

        True entries at already visible sites are harmless: those sites remain
        clamped. The caller owns Bernoulli probabilities and draws, SAR windows,
        caching, and any deterministic final cleanup.
        """
        if reveal_mask.shape != xt.shape or reveal_mask.dtype != torch.bool or reveal_mask.device != xt.device:
            raise ValueError("reveal_mask must be bool [batch,length] on xt's device")
        proposal = self.propose(log_probs, xt, generator, sampling=sampling)
        return torch.where(reveal_mask & xt.eq(self.mask_id), proposal, xt)

    @torch.no_grad()
    def selected_log_probability(self, log_probs, xt, selected_mask, values):
        """Exact log q(X_selected=values), by a constrained partition ratio.

        This is the clean selected-block probability, excluding the caller's
        reveal-mask probability. Impossible events return -inf. Constraints on
        visible sites are allowed and are either redundant or impossible.
        """
        unary = self._unary(log_probs, xt)
        if selected_mask.shape != xt.shape or selected_mask.dtype != torch.bool or selected_mask.device != xt.device:
            raise ValueError("selected_mask must be bool [batch,length] on xt's device")
        if values.shape != xt.shape or values.dtype != torch.long or values.device != xt.device:
            raise ValueError("values must be int64 [batch,length] on xt's device")
        if bool(((values < 0) | (values >= self.potential.vocab_size)).any()):
            raise ValueError("selected value outside vocabulary")
        at_values = unary.gather(-1, values[..., None]).squeeze(-1)
        impossible = (selected_mask & ~torch.isfinite(at_values)).any(-1)
        # Keep impossible rows unchanged so a backend never receives an empty
        # support. Their mathematical probability is restored to zero below.
        active = selected_mask & ~impossible[:, None]
        constrained = unary.clone().masked_fill_(active[..., None], -torch.inf)
        previous = constrained.gather(-1, values[..., None]).squeeze(-1)
        constrained.scatter_(-1, values[..., None], torch.where(active, at_values, previous)[..., None])
        answer = self._run("sparse_chain_log_partition", constrained) - self._run("sparse_chain_log_partition", unary)
        return torch.where(impossible, -torch.inf, answer)
