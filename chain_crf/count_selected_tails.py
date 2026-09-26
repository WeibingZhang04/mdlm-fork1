"""Exact selected-chain draws with unselected endpoint tails integrated out.

This isolated sampling backend does not modify the existing count denoiser,
SAR update, or selected-span implementation. After visible clamps separate
spans, only spans containing selected latent sites are processed. Within each
retained span, let l/r be the first/last selected sites. Forward messages are
computed through r, retaining only l..r; a backward message integrates the
unselected suffix after r. Joint mode draws the marginal at r then performs
FFBS down to l. Internal unselected gaps remain real latent chain nodes.

For one selected site this is exactly one marginal draw, with L-1 sparse
message products and no categorical draws at unwanted sites. For multiple
selected sites, joint mode still uses L-1 sparse products, but traces only the
selected envelope. Own-marginal mode takes L-1+(r-l) sparse products rather
than the existing full forward/backward 2(L-1). Message storage is O((r-l+1)V),
though the current chunk helper still materializes all L unaries in FP64.

Chunks are grouped by (length,l,r), bounded by total original span positions,
with one longer dependent span allowed intact. No dummy transitions, top-k,
precision reduction, or altered boundary factors are used. All probability
arithmetic is FP64. Bucketing and omitted random draws change RNG consumption;
the preserved guarantee is the distribution, not identical seeded outputs.
"""

import torch

from chain_crf import sparse_count_segments as segmented
from chain_crf.sparse_count_gpu import _Validation


def _tail_chunks(log_unary, potential, buckets, selection, budget):
    envelopes = {}
    for length, spans in buckets.items():
        for span in spans:
            sites = [pos for pos, chosen in enumerate(
                selection[span[0]][span[1]:span[1] + length]) if chosen]
            if sites:
                envelopes.setdefault((length, sites[0], sites[-1]), []).append(span)
    for (length, first, last), spans in sorted(envelopes.items()):
        chunk_size = max(1, budget // length)
        for start in range(0, len(spans), chunk_size):
            chunk = spans[start:start + chunk_size]
            selected_rows = [[row for row, span in enumerate(chunk)
                              if selection[span[0]][span[1] + pos]]
                             for pos in range(first, last + 1)]
            batches, positions, values = next(segmented._chunks(
                log_unary, potential, {length: chunk}, budget))
            yield batches, positions, values, first, last, selected_rows


def _envelope_messages(values, potential, validation, first, last):
    scales = values.amax(-1, keepdim=True)
    unary = (values - scales).exp()
    current, _ = validation.normalize(unary[:, 0])
    alpha = [current] if first == 0 else []
    for pos in range(1, last + 1):
        current, _ = validation.normalize(unary[:, pos] * potential.forward_mul(current))
        if pos >= first:
            alpha.append(current)
    beta = torch.ones_like(current)
    for pos in range(values.shape[1] - 2, last - 1, -1):
        beta, _ = validation.normalize(potential.backward_mul(unary[:, pos + 1] * beta))
    return unary, alpha, beta


@torch.no_grad()
def sample_selected_chain_tails(log_unary, potential, selected_mask, generator=None,
                                *, sampling="joint", clamped_states=None,
                                max_chunk_tokens=512):
    """Sample selected sites only; return int64 [B,L], -1 elsewhere.

    Inputs follow ``sample_selected_chain``: full prepared clean-token log
    unaries with -inf forbidden states, a SegmentedCountPotential, and a bool
    selection on the same device. Singleton support defines visible clamps;
    selected visible states are deterministic. Joint mode returns the exact
    selected-block marginal, marginal mode its product of singleton marginals.
    No selected latent site means no messages or random draws.

    Original denoiser calls, reveal probabilities, cleanup, and strength-zero
    original-sampler delegation remain the caller's responsibility. This API
    neither exposes a partition function nor replaces continuous-time scoring.
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
    for batches, positions, values, first, last, selected_rows in _tail_chunks(
            log_unary, potential, buckets, selection, max_chunk_tokens):
        validation = _Validation(len(values), values.device)
        unary, alpha, beta = _envelope_messages(
            values, potential, validation, first, last)
        if sampling == "joint":
            draws = torch.empty((len(values), last - first + 1),
                                dtype=torch.long, device=values.device)
            probability = validation.normalize(alpha[-1] * beta)[0]
            draws[:, -1] = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
            for pos in range(last - 1, first - 1, -1):
                probability = validation.normalize(alpha[pos - first] *
                    potential.selected_columns(draws[:, pos - first + 1]))[0]
                draws[:, pos - first] = torch.multinomial(
                    probability, 1, generator=generator).squeeze(-1)
            envelope = positions[:, first:last + 1]
            result[batches, envelope] = torch.where(selected_mask[batches, envelope], draws, -1)
        else:
            for pos in range(last, first - 1, -1):
                if pos < last:
                    beta, _ = validation.normalize(
                        potential.backward_mul(unary[:, pos + 1] * beta))
                if selected_rows[pos - first]:
                    probability = validation.normalize(alpha[pos - first] * beta)[0]
                    rows = torch.tensor(selected_rows[pos - first], dtype=torch.long,
                                        device=values.device)
                    draws = torch.multinomial(probability[rows], 1,
                                             generator=generator).squeeze(-1)
                    result[batches[rows, 0], positions[rows, pos]] = draws
        valid.logical_and_(validation.valid.all())
    segmented._finish(valid)
    return result
