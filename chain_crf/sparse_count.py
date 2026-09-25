"""Full-vocabulary count CRFs: an exact sparse-plus-rank-one prototype.

This module is intentionally separate from candidate/tail CRFs and generation.
For nonnegative strength, exponentiating a sparse-count + independence-backoff
joint leaves a rank-one background and NONNEGATIVE corrections on observed
pairs. No top-K restriction or neutral residual state is used.

Inference uses float64 probability-space messages, normalized at every step;
unaries and the pair table have separate log scales. This avoids accumulation
overflow, but cannot represent ratios below float64's dynamic range. Extremely
sharp unary/pair potentials can still underflow (a zero total raises an error).
The reference has O(L*(E+V)) work per sequence and O(E+L*V) storage. Both sparse
orientations are converted to CSR once during construction, not within DP.
"""

from dataclasses import dataclass, replace
import math
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class SparseCountPotential:
    """W = exp(log_scale) * (left[:, None]*right[None, :] + sparse).

CSR matrices cache corrections in both orientations. CSC-like arrays permit
O(V+degree) access to a selected column for backward sampling. Column offsets
are held on CPU to avoid device synchronizations for each slice boundary.
Shared across all chain edges.
"""

    left: Tensor
    right: Tensor
    sparse: Tensor
    sparse_transpose: Tensor
    column_offsets: Tensor
    column_rows: Tensor
    column_values: Tensor
    log_scale: float

    @property
    def vocab_size(self):
        return self.left.numel()

    @classmethod
    def from_head(cls, head, *, mode=None, strength=None):
        """Copy fitted CountBigramHead statistics without changing that head."""
        return cls.from_counts(
            head.left_counts, head.right_counts, head.pair_keys, head.pair_counts,
            mode=head.mode if mode is None else mode,
            strength=head.strength if strength is None else strength,
            smoothing=head.smoothing,
        )

    @classmethod
    def from_counts(cls, left_counts, right_counts, pair_keys, pair_counts,
                    *, mode="conditional", strength=1., smoothing=.1):
        if mode not in ("conditional", "pmi"):
            raise ValueError("mode must be conditional or pmi")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must lie in (0, 1]")
        lc = left_counts.detach().to(dtype=torch.float64).clone()
        rc = right_counts.detach().to(device=lc.device, dtype=torch.float64).clone()
        keys = pair_keys.detach().to(device=lc.device, dtype=torch.long).clone()
        cnt = pair_counts.detach().to(device=lc.device, dtype=torch.float64).clone()
        if lc.ndim != 1 or lc.numel() == 0 or rc.shape != lc.shape:
            raise ValueError("endpoint counts must have the same nonempty vector shape")
        if keys.ndim != 1 or cnt.shape != keys.shape:
            raise ValueError("pair keys/counts must be equal-length vectors")
        if any(not torch.isfinite(v).all() or (v < 0).any() for v in (lc, rc, cnt)):
            raise ValueError("counts must be finite and nonnegative")
        vocab = lc.numel()
        if keys.numel() and ((keys < 0).any() or (keys >= vocab*vocab).any()
                              or (keys[1:] <= keys[:-1]).any() or (cnt <= 0).any()):
            raise ValueError("pair keys must be unique/sorted/in range with positive counts")
        rows, cols = keys // vocab, keys % vocab
        observed_l, observed_r = torch.zeros_like(lc), torch.zeros_like(rc)
        observed_l.index_add_(0, rows, cnt)
        observed_r.index_add_(0, cols, cnt)
        if not torch.allclose(lc, observed_l) or not torch.allclose(rc, observed_r):
            raise ValueError("endpoint counts must be the marginals of pair counts")
        n = float(lc.sum())
        eps = float(smoothing) if n else 1.
        b_l, b_r = (lc + 1.)/(n + vocab), (rc + 1.)/(n + vocab)
        empirical_scale = (1.-eps)/max(n, 1.)
        m_l = empirical_scale*lc + eps*b_l
        m_r = empirical_scale*rc + eps*b_r

        # Unobserved probabilities factorize, even after an arbitrary power.
        log_left = strength*(math.log(eps) + b_l.log() - m_l.log())
        log_right = strength*b_r.log()
        if mode == "pmi":
            log_right = log_right - strength*m_r.log()
        log_background = log_left[rows] + log_right[cols]
        if keys.numel() and empirical_scale > 0 and strength > 0:
            log_ratio = (math.log(empirical_scale) + cnt.log() - math.log(eps)
                         - b_l[rows].log() - b_r[cols].log())
            # delta = log(W_observed / W_background), stable for large counts.
            delta = strength*torch.nn.functional.softplus(log_ratio)
            # log(exp(delta)-1), without overflow or cancellation.
            log_correction = log_background + delta + torch.log(-torch.expm1(-delta))
            observed_max = (log_background + delta).max()
        else:
            log_correction = torch.full_like(cnt, -torch.inf)
            observed_max = lc.new_tensor(-torch.inf)
        scale = float(torch.maximum(log_left.max() + log_right.max(), observed_max))
        pivot = float(log_left.max())
        left = (log_left-pivot).exp()
        right = (log_right+pivot-scale).exp()
        corrections = (log_correction-scale).exp()
        keep = corrections > 0
        rows, cols, corrections = rows[keep], cols[keep], corrections[keep]
        sparse = torch.sparse_coo_tensor(torch.stack((rows, cols)), corrections,
                                         (vocab, vocab), device=lc.device,
                                         check_invariants=True).coalesce()
        # Transpose/coalescing/CSR conversion are setup work. A fresh COO
        # transpose at every forward step can trigger repeated sparse work.
        sparse_transpose = sparse.transpose(0, 1).coalesce().to_sparse_csr()
        sparse = sparse.to_sparse_csr()
        offsets = sparse_transpose.crow_indices().cpu()
        return cls(left, right, sparse, sparse_transpose, offsets,
                   sparse_transpose.col_indices(), sparse_transpose.values(), scale)

    def shifted(self, log_constant):
        """Multiply EVERY full-vocabulary edge potential by exp(log_constant)."""
        if not math.isfinite(log_constant):
            raise ValueError("log_constant must be finite")
        return replace(self, log_scale=self.log_scale + float(log_constant))

    def forward_mul(self, message):
        """Row message times the scale-free W, shape [B,V]."""
        background = (message*self.left).sum(-1, keepdim=True)*self.right
        return background + torch.sparse.mm(self.sparse_transpose, message.T).T

    def backward_mul(self, message):
        """Scale-free W times column messages, represented as [B,V]."""
        background = (message*self.right).sum(-1, keepdim=True)*self.left
        return background + torch.sparse.mm(self.sparse, message.T).T

    def selected_columns(self, indices):
        """Return [B,V] scale-free W[:, indices[b]] without a dense V-by-V table."""
        out = self.left[None, :]*self.right[indices, None]
        if indices.numel() == 1:
            # The benchmark uses B=1: avoid GPU unique/nonzero kernels and two
            # separate device-to-host synchronizations for slice boundaries.
            column = int(indices.item())
            lo, hi = self.column_offsets[column:column+2].tolist()
            out[0, self.column_rows[lo:hi]] += self.column_values[lo:hi]
            return out
        for column in indices.unique().tolist():
            lo, hi = int(self.column_offsets[column]), int(self.column_offsets[column+1])
            batch_rows = torch.where(indices == column)[0]
            pair_rows = self.column_rows[lo:hi]
            out[batch_rows[:, None], pair_rows[None, :]] += self.column_values[None, lo:hi]
        return out


def _unaries(log_unary, potential):
    if log_unary.ndim != 3 or log_unary.shape[1] < 1 or log_unary.shape[2] != potential.vocab_size:
        raise ValueError("log_unary must have shape [batch,length>=1,vocab]")
    if log_unary.device != potential.left.device:
        raise ValueError("unaries and potential must use the same device")
    log_unary = log_unary.to(dtype=torch.float64)
    if torch.isnan(log_unary).any() or torch.isposinf(log_unary).any():
        raise ValueError("unaries may contain finite values or negative infinity only")
    maximum = log_unary.amax(-1, keepdim=True)
    if not torch.isfinite(maximum).all():
        raise ValueError("every position must have at least one supported token")
    return (log_unary-maximum).exp(), maximum.squeeze(-1)


def _normalize(value):
    total = value.sum(-1, keepdim=True)
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise FloatingPointError("nonpositive/nonfinite message total; check FP64 dynamic range")
    return value/total, total.squeeze(-1).log()


def _forward(log_unary, potential, store=True):
    unary, unary_scales = _unaries(log_unary, potential)
    current, normalizer = _normalize(unary[:, 0])
    messages = [current] if store else []
    log_z = normalizer + unary_scales.sum(-1)
    for pos in range(1, unary.shape[1]):
        current, normalizer = _normalize(unary[:, pos]*potential.forward_mul(current))
        log_z = log_z + normalizer + potential.log_scale
        if store:
            messages.append(current)
    return unary, messages, log_z


def sparse_chain_log_partition(log_unary: Tensor, potential: SparseCountPotential):
    """Exact full-vocabulary log partition up to floating-point arithmetic."""
    return _forward(log_unary, potential, store=False)[2]


def sparse_chain_marginals(log_unary: Tensor, potential: SparseCountPotential):
    """Forward/backward node marginals [B,L,V], no candidate aggregation."""
    unary, alpha, _ = _forward(log_unary, potential)
    beta = torch.ones_like(alpha[-1])
    marginals = [alpha[-1]]
    for pos in range(unary.shape[1]-2, -1, -1):
        beta, _ = _normalize(potential.backward_mul(unary[:, pos+1]*beta))
        marginals.append(_normalize(alpha[pos]*beta)[0])
    return torch.stack(marginals[::-1], dim=1)


@torch.no_grad()
def sample_sparse_chain(log_unary: Tensor, potential: SparseCountPotential,
                        generator: Optional[torch.Generator] = None):
    """Exact full-vocabulary joint draw via forward filtering/backward sampling."""
    _, alpha, _ = _forward(log_unary, potential)
    states = torch.empty(log_unary.shape[:2], dtype=torch.long, device=log_unary.device)
    states[:, -1] = torch.multinomial(alpha[-1], 1, generator=generator).squeeze(-1)
    for pos in range(log_unary.shape[1]-2, -1, -1):
        probabilities = _normalize(alpha[pos]*potential.selected_columns(states[:, pos+1]))[0]
        states[:, pos] = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    return states
