"""Exact FP64 sparse-count inference with deferred device validation.

This is an isolated experimental backend. The reference sparse_count module
is unchanged. Invalid messages use a harmless internal distribution only so
GPU work can finish safely; every public call checks accumulated validity and
raises before returning any invalid result.
"""

from dataclasses import dataclass

import torch

from chain_crf.sparse_count import SparseCountPotential, _unaries


@dataclass(frozen=True)
class GPUCountPotential(SparseCountPotential):
    device_column_offsets: torch.Tensor
    degree_offsets: torch.Tensor
    padded_column_rows: torch.Tensor
    padded_column_values: torch.Tensor

    @classmethod
    def from_reference(cls, reference):
        # This setup-only CPU maximum never occurs inside DP or FFBS.
        degrees = reference.column_offsets[1:] - reference.column_offsets[:-1]
        maximum_degree = int(degrees.max()) if degrees.numel() else 0
        device = reference.left.device
        return cls(
            **vars(reference),
            device_column_offsets=reference.column_offsets.to(device),
            degree_offsets=torch.arange(maximum_degree, device=device),
            padded_column_rows=torch.cat((reference.column_rows,
                                         torch.zeros(1, dtype=torch.long, device=device))),
            padded_column_values=torch.cat((reference.column_values,
                                           reference.left.new_zeros(1))),
        )

    @classmethod
    def from_head(cls, head, *, mode=None, strength=None):
        return cls.from_reference(SparseCountPotential.from_head(
            head, mode=mode, strength=strength))

    @classmethod
    def from_counts(cls, *args, **kwargs):
        return cls.from_reference(SparseCountPotential.from_counts(*args, **kwargs))

    def selected_columns(self, indices):
        """Fixed-shape masked CSC gather; no selected-index host transfer."""
        out = self.left[None, :] * self.right[indices, None]
        if self.degree_offsets.numel() == 0:
            return out
        lo = self.device_column_offsets[indices]
        hi = self.device_column_offsets[indices + 1]
        offsets = self.degree_offsets[None, :]
        valid = offsets < (hi - lo)[:, None]
        # The sentinel has row=0 and value=0, including zero-degree columns.
        selected = torch.where(valid, lo[:, None] + offsets,
                               self.padded_column_values.numel() - 1)
        rows = self.padded_column_rows[selected]
        values = self.padded_column_values[selected]
        # Padding has zero weight. Spread these zeros over distinct output
        # rows instead of making up to max_degree atomics contend on row 0.
        # A sparse column has at most V distinct rows, so offsets are in range.
        rows = torch.where(valid, rows, offsets)
        return out.scatter_add(1, rows, values)


class _Validation:
    def __init__(self, batch, device):
        self.valid = torch.ones(batch, dtype=torch.bool, device=device)

    def normalize(self, value):
        total = value.sum(-1, keepdim=True)
        valid = torch.isfinite(total) & (total > 0)
        self.valid.logical_and_(valid.squeeze(-1))
        safe_total = torch.where(valid, total, torch.ones_like(total))
        normalized = value / safe_total
        # Only invalid rows take this branch. Their results are never returned.
        normalized = torch.where(valid, normalized, 1. / value.shape[-1])
        return normalized, total.squeeze(-1)

    def finish(self):
        if not bool(self.valid.all()):
            raise FloatingPointError(
                'nonpositive/nonfinite message total; check FP64 dynamic range')


def _forward(log_unary, potential, validation, *, store=True, need_logz=True):
    # Input checks stay in the reference helper and run only once per chain.
    unary, unary_scales = _unaries(log_unary, potential)
    current, total = validation.normalize(unary[:, 0])
    messages = [current] if store else []
    logz = total.log() + unary_scales.sum(-1) if need_logz else None
    for pos in range(1, unary.shape[1]):
        current, total = validation.normalize(
            unary[:, pos] * potential.forward_mul(current))
        if need_logz:
            logz = logz + total.log() + potential.log_scale
        if store:
            messages.append(current)
    return unary, messages, logz


def sparse_chain_log_partition(log_unary, potential):
    validation = _Validation(log_unary.shape[0], log_unary.device)
    logz = _forward(log_unary, potential, validation, store=False)[2]
    validation.finish()
    return logz


def sparse_chain_marginals(log_unary, potential):
    validation = _Validation(log_unary.shape[0], log_unary.device)
    unary, alpha, _ = _forward(log_unary, potential, validation, need_logz=False)
    beta = torch.ones_like(alpha[-1])
    marginals = [alpha[-1]]
    for pos in range(unary.shape[1] - 2, -1, -1):
        beta, _ = validation.normalize(potential.backward_mul(unary[:, pos + 1] * beta))
        marginals.append(validation.normalize(alpha[pos] * beta)[0])
    result = torch.stack(marginals[::-1], dim=1)
    validation.finish()
    return result


@torch.no_grad()
def sample_sparse_chain(log_unary, potential, generator=None):
    validation = _Validation(log_unary.shape[0], log_unary.device)
    _, alpha, _ = _forward(log_unary, potential, validation, need_logz=False)
    states = torch.empty(log_unary.shape[:2], dtype=torch.long, device=log_unary.device)
    states[:, -1] = torch.multinomial(alpha[-1], 1, generator=generator).squeeze(-1)
    for pos in range(log_unary.shape[1] - 2, -1, -1):
        probability = validation.normalize(
            alpha[pos] * potential.selected_columns(states[:, pos + 1]))[0]
        states[:, pos] = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
    validation.finish()
    return states
