"""Exact selected-site draws from a full-vocabulary count chain.

Singleton-supported nodes separate the original chain into conditionally
independent spans. A span with no selected site integrates to one and needs no
messages or random draws. Retained spans still include ALL their unselected
latent sites and both original visible-boundary factors: deleting latent gaps
or joining selected sites as neighbors would change the selected-block law.

Joint mode uses the existing FP64 FFBS recursion within each retained span.
Marginal mode uses its forward/backward messages but independently draws only
selected sites, without allocating a full [B,L,V] marginal tensor. Both return
-1 outside the selection and deterministic states at selected visible sites.
Exact-length buckets preserve the existing chunk bound (one longer dependent
span must fit intact). Input support validation still scans the full input;
message work and message memory only cover retained spans.

This is a separate sampling API, not a replacement for the full singleton
marginals used by continuous-time evaluation. Omitting unused draws changes
RNG consumption, not the probability law. A strength-zero original-sampler
bypass belongs to the caller; this module does not provide bitwise RNG parity.
"""

import hashlib
from pathlib import Path

import torch

from chain_crf.count_denoiser import CountCRFDenoiser
from chain_crf import sparse_count_segments as segmented
from chain_crf.sparse_count_gpu import _Validation


def _selected_chunks(log_unary, potential, buckets, selection, budget):
    """Reuse the exact boundary absorption; selection metadata stays on CPU."""
    for length in sorted(buckets):
        spans = [span for span in buckets[length]
                 if any(selection[span[0]][span[1]:span[1] + length])]
        chunk_size = max(1, budget // length)
        for start in range(0, len(spans), chunk_size):
            chunk = spans[start:start + chunk_size]
            selected_rows = [[row for row, span in enumerate(chunk)
                              if selection[span[0]][span[1] + pos]]
                             for pos in range(length)]
            # Exactly one chunk; no artificial nodes or neutralized real edges.
            batches, positions, values = next(segmented._chunks(
                log_unary, potential, {length: chunk}, budget))
            yield batches, positions, values, selected_rows


@torch.no_grad()
def sample_selected_chain(log_unary, potential, selected_mask, generator=None,
                          *, sampling="joint", clamped_states=None,
                          max_chunk_tokens=512):
    """Return [B,L] int64 draws, with -1 at every unselected position.

    ``log_unary`` is the prepared [B,L,V] clean-token log potential: forbidden
    states are -inf and visible sites have singleton support. The potential
    must be a prebuilt SegmentedCountPotential. Finite visible unary constants
    may be nonzero; they and global edge scaling cancel from conditional draws.
    ``selected_mask`` is boolean on the same device. Selected visible sites are
    allowed and returned unchanged. Joint draws have q(X_selected | visible);
    marginal draws have the product of that same q's singleton marginals.

    No reveals, denoiser passes, times, cache invalidation, or dropout settings
    are changed here. The caller supplies the independently chosen selection.
    """
    if sampling not in ("joint", "marginal"):
        raise ValueError("sampling must be joint or marginal")
    if (selected_mask.shape != log_unary.shape[:2]
            or selected_mask.dtype != torch.bool
            or selected_mask.device != log_unary.device):
        raise ValueError("selected_mask must be bool [batch,length] on the unary device")
    states, buckets, _, valid = segmented._prepare(
        log_unary, potential, clamped_states, max_chunk_tokens)
    selection = selected_mask.detach().cpu().tolist()
    result = torch.where(selected_mask, states, -1)
    for batches, positions, values, selected_rows in _selected_chunks(
            log_unary, potential, buckets, selection, max_chunk_tokens):
        validation = _Validation(len(values), values.device)
        unary, alpha, _ = segmented._forward(
            values, potential, validation, need_logz=False, store=True)
        if sampling == "joint":
            draws = torch.empty(values.shape[:2], dtype=torch.long, device=values.device)
            draws[:, -1] = torch.multinomial(alpha[-1], 1, generator=generator).squeeze(-1)
            for pos in range(values.shape[1] - 2, -1, -1):
                probability = validation.normalize(
                    alpha[pos] * potential.selected_columns(draws[:, pos + 1]))[0]
                draws[:, pos] = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
            result[batches, positions] = torch.where(
                selected_mask[batches, positions], draws, -1)
        else:
            beta = torch.ones_like(alpha[-1])
            for pos in range(values.shape[1] - 1, -1, -1):
                if pos < values.shape[1] - 1:
                    beta, _ = validation.normalize(
                        potential.backward_mul(unary[:, pos + 1] * beta))
                if selected_rows[pos]:
                    probability = validation.normalize(alpha[pos] * beta)[0]
                    rows = torch.tensor(selected_rows[pos], dtype=torch.long, device=values.device)
                    draws = torch.multinomial(probability[rows], 1,
                                             generator=generator).squeeze(-1)
                    result[batches[rows, 0], positions[rows, pos]] = draws
        valid.logical_and_(validation.valid.all())
    segmented._finish(valid)
    return result


class SelectedCountCRFDenoiser(CountCRFDenoiser):
    """Count denoiser with selected-span sampling only in ``reveal``.

    Continuous-time singleton evaluation and full clean ``propose`` retain
    their inherited implementations. Use this adapter with the unchanged SAR
    integration: it already delegates the entire original update at strength
    zero, before calling reveal. Explicit reveal/propose calls preserve laws,
    not the original sampler's exact RNG consumption.
    """

    def __init__(self, head, *, backend="segments", **kwargs):
        if backend != "segments":
            raise ValueError("selected-span reveal requires the segments backend")
        super().__init__(head, backend=backend, **kwargs)

    def identity(self):
        identity = super().identity()
        identity["selected_reveal"] = {
            "format": "selected_count_spans_v1",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "joint": "FFBS_only_in_spans_containing_selected_latent_sites",
            "marginal": "independent_own_marginal_draws_only_at_selected_sites",
            "continuous_time_call": "unchanged_inherited_full_singleton_marginals",
            "full_propose": "unchanged_inherited_method",
            "rng": "law_equivalent_not_seed_identical",
        }
        return identity

    @torch.no_grad()
    def reveal(self, log_probs, xt, reveal_mask, generator=None, *, sampling="joint"):
        """Reveal only selected mask sites; never draw or alter the selection."""
        if (reveal_mask.shape != xt.shape or reveal_mask.dtype != torch.bool
                or reveal_mask.device != xt.device):
            raise ValueError("reveal_mask must be bool [batch,length] on xt's device")
        unary = self._unary(log_probs, xt)
        selected = reveal_mask & xt.eq(self.mask_id)
        proposal = sample_selected_chain(
            unary, self.potential, selected, generator, sampling=sampling,
            max_chunk_tokens=self.sparse_chunk_size)
        return torch.where(selected, proposal, xt)
