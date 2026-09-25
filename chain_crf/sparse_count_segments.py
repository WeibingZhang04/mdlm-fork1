"""Exact full-vocabulary count inference after eliminating visible chain nodes.

This isolated backend does not change sparse_count or sparse_count_gpu. It
conditions on singleton-supported nodes, absorbs their ORIGINAL neighboring
edge factors, and runs independent FFBS chains on the remaining contiguous
spans. Exact-length buckets avoid padding altogether: an artificial padded
transition through the shared non-neutral W would change the distribution.

All probability arithmetic is FP64. A chunk contains at most max_chunk_tokens
real positions, except that one longer span must fit intact. There is no
global-longest-span padding and no V-by-V materialization. With V=50,258 and
the default 4,096-position budget, each chunk-sized FP64 tensor is at most
1.65 GB (or one longer span); messages, unaries, and temporary arrays require
several such tensors. A full marginal result itself necessarily costs BLV*8
bytes. Lower the budget on smaller devices. Sampling needs only [B,L] output.

Topology is transferred to CPU once per call; all per-position DP/FFBS work
and sparse row/column access stay on-device. Bucketing changes RNG consumption
and sample order relative to unsegmented FFBS: the joint law is preserved, not
seed-by-seed outputs. As in the reference backend, FP64 underflow/overflow is
detected, not repaired by changing the model. These are inference APIs, not a
training implementation with a bounded backward-pass memory guarantee.
"""

from dataclasses import dataclass, fields
import math

import torch

from chain_crf.sparse_count import SparseCountPotential
from chain_crf.sparse_count_gpu import GPUCountPotential, _Validation


@dataclass(frozen=True)
class SegmentedCountPotential(GPUCountPotential):
    """Cached sparse orientations plus bounded device-only boundary lookup."""

    device_row_offsets: torch.Tensor
    row_degree_offsets: torch.Tensor
    padded_row_columns: torch.Tensor
    padded_row_values: torch.Tensor
    pair_keys: torch.Tensor

    @classmethod
    def from_reference(cls, reference):
        # Explicit base fields also accept an already prepared GPU potential.
        base = SparseCountPotential(**{f.name: getattr(reference, f.name)
                                      for f in fields(SparseCountPotential)})
        gpu = GPUCountPotential.from_reference(base)
        offsets = base.sparse.crow_indices()
        degrees = offsets[1:] - offsets[:-1]
        maximum_degree = int(degrees.max()) if degrees.numel() else 0
        rows = torch.repeat_interleave(torch.arange(base.vocab_size,
                                      device=base.left.device), degrees)
        return cls(
            **vars(gpu), device_row_offsets=offsets,
            row_degree_offsets=torch.arange(maximum_degree, device=base.left.device),
            padded_row_columns=torch.cat((base.sparse.col_indices(),
                torch.zeros(1, dtype=torch.long, device=base.left.device))),
            padded_row_values=torch.cat((base.sparse.values(), base.left.new_zeros(1))),
            pair_keys=rows * base.vocab_size + base.sparse.col_indices(),
        )

    @classmethod
    def from_head(cls, head, *, mode=None, strength=None):
        return cls.from_reference(SparseCountPotential.from_head(
            head, mode=mode, strength=strength))

    @classmethod
    def from_counts(cls, *args, **kwargs):
        return cls.from_reference(SparseCountPotential.from_counts(*args, **kwargs))

    def selected_rows(self, indices):
        """Return scale-free W[indices[b], :] without host-selected slices."""
        out = self.left[indices, None] * self.right[None, :]
        if self.row_degree_offsets.numel() == 0:
            return out
        lo, hi = self.device_row_offsets[indices], self.device_row_offsets[indices + 1]
        offsets = self.row_degree_offsets[None, :]
        valid = offsets < (hi - lo)[:, None]
        selected = torch.where(valid, lo[:, None] + offsets,
                               self.padded_row_values.numel() - 1)
        columns = torch.where(valid, self.padded_row_columns[selected], offsets)
        return out.scatter_add(1, columns, self.padded_row_values[selected])

    def selected_entries(self, rows, columns):
        """One scale-free W[row,col] per pair; O(log E) sparse lookup."""
        background = self.left[rows] * self.right[columns]
        if self.pair_keys.numel() == 0:
            return background
        keys = rows * self.vocab_size + columns
        index = torch.searchsorted(self.pair_keys, keys)
        safe = index.clamp_max(self.pair_keys.numel() - 1)
        found = (index < self.pair_keys.numel()) & (self.pair_keys[safe] == keys)
        return background + torch.where(found, self.sparse.values()[safe], 0.)


def _prepare(log_unary, potential, clamped_states, max_chunk_tokens):
    if not isinstance(potential, SegmentedCountPotential):
        raise TypeError('Construct SegmentedCountPotential once before inference')
    if log_unary.ndim != 3 or min(log_unary.shape[:2]) < 1 or log_unary.shape[-1] != potential.vocab_size:
        raise ValueError('log_unary must have shape [batch>=1,length>=1,vocab]')
    if not log_unary.is_floating_point() or log_unary.device != potential.left.device:
        raise ValueError('Floating-point unaries and potential must use the same device')
    if type(max_chunk_tokens) is not int or max_chunk_tokens < 1:
        raise ValueError('max_chunk_tokens must be a positive integer')
    if not math.isfinite(potential.log_scale):
        raise ValueError('Potential log_scale must be finite')
    finite = torch.isfinite(log_unary)
    counts = finite.sum(-1)
    invalid = torch.isnan(log_unary).any() | torch.isposinf(log_unary).any()
    if bool(invalid):
        raise ValueError('Unaries may contain finite values or negative infinity only')
    if bool((counts == 0).any()):
        raise ValueError('Every position must have at least one supported token')
    singleton = counts == 1
    states = torch.where(singleton, log_unary.argmax(-1), -1)
    del finite, counts
    if clamped_states is not None:
        if (clamped_states.shape != states.shape or clamped_states.device != states.device
                or clamped_states.dtype != torch.long or not torch.equal(clamped_states, states)):
            raise ValueError('clamped_states must exactly describe all singleton supports; use -1 elsewhere')
    # One topology transfer; no device-selected Python index inside DP/FFBS.
    topology = states.detach().cpu().tolist()
    buckets, visible_pairs = {}, []
    for batch, row in enumerate(topology):
        start = 0
        while start < len(row):
            if row[start] >= 0:
                if start and row[start - 1] >= 0:
                    visible_pairs.append((batch, row[start - 1], row[start]))
                start += 1
                continue
            end = start + 1
            while end < len(row) and row[end] < 0:
                end += 1
            buckets.setdefault(end - start, []).append((batch, start,
                row[start - 1] if start else -1, row[end] if end < len(row) else -1))
            start = end
    unary_visible = log_unary.gather(-1, states.clamp_min(0).unsqueeze(-1)).squeeze(-1).double()
    constant = torch.where(singleton, unary_visible, 0.).sum(-1)
    valid = torch.ones((), dtype=torch.bool, device=log_unary.device)
    if visible_pairs:
        indices = torch.tensor(visible_pairs, dtype=torch.long, device=log_unary.device)
        values = potential.selected_entries(indices[:, 1], indices[:, 2])
        valid.logical_and_((torch.isfinite(values) & (values > 0)).all())
        constant = constant.index_add(0, indices[:, 0], values.log())
    constant = constant + (log_unary.shape[1] - 1) * potential.log_scale
    return states, buckets, constant, valid


def _chunks(log_unary, potential, buckets, max_chunk_tokens):
    for length in sorted(buckets):
        spans = buckets[length]
        chunk_size = max(1, max_chunk_tokens // length)
        for begin in range(0, len(spans), chunk_size):
            ids = torch.tensor(spans[begin:begin + chunk_size], dtype=torch.long,
                               device=log_unary.device)
            batches = ids[:, 0, None]
            positions = ids[:, 1, None] + torch.arange(length, device=log_unary.device)
            values = log_unary[batches, positions].double()
            # -1 boundaries use any valid index, then a neutral factor is
            # substituted. No branch on a device value is needed here.
            left = potential.selected_rows(ids[:, 2].clamp_min(0)).log()
            right = potential.selected_columns(ids[:, 3].clamp_min(0)).log()
            values[:, 0] = values[:, 0] + torch.where(ids[:, 2, None] >= 0, left, 0.)
            values[:, -1] = values[:, -1] + torch.where(ids[:, 3, None] >= 0, right, 0.)
            yield batches, positions, values


def _forward(values, potential, validation, *, need_logz, store):
    scales = values.amax(-1, keepdim=True)
    unary = (values - scales).exp()
    current, total = validation.normalize(unary[:, 0])
    alpha = [current] if store else []
    logz = total.log() + scales.squeeze(-1).sum(-1) if need_logz else None
    for pos in range(1, values.shape[1]):
        current, total = validation.normalize(unary[:, pos] * potential.forward_mul(current))
        if need_logz:
            logz = logz + total.log()
        if store:
            alpha.append(current)
    return unary, alpha, logz


def _finish(valid):
    if not bool(valid):
        raise FloatingPointError('nonpositive/nonfinite message total; check FP64 dynamic range')


def sparse_chain_log_partition(log_unary, potential, *, clamped_states=None, max_chunk_tokens=4096):
    """Full original-chain partition, including visible unaries and edges."""
    _, buckets, logz, valid = _prepare(log_unary, potential, clamped_states, max_chunk_tokens)
    for batches, _, values in _chunks(log_unary, potential, buckets, max_chunk_tokens):
        validation = _Validation(len(values), values.device)
        chunk_z = _forward(values, potential, validation, need_logz=True, store=False)[2]
        logz = logz.index_add(0, batches[:, 0], chunk_z)
        valid.logical_and_(validation.valid.all())
    valid.logical_and_(torch.isfinite(logz).all())
    _finish(valid)
    return logz


def sparse_chain_marginals(log_unary, potential, *, clamped_states=None, max_chunk_tokens=4096):
    """Exact [B,L,V] marginals under the full-vocabulary conditioned chain."""
    states, buckets, _, valid = _prepare(log_unary, potential, clamped_states, max_chunk_tokens)
    result = torch.zeros_like(log_unary, dtype=torch.float64)
    result.scatter_(-1, states.clamp_min(0).unsqueeze(-1), (states >= 0).unsqueeze(-1).double())
    for batches, positions, values in _chunks(log_unary, potential, buckets, max_chunk_tokens):
        validation = _Validation(len(values), values.device)
        unary, alpha, _ = _forward(values, potential, validation, need_logz=False, store=True)
        beta = torch.ones_like(alpha[-1])
        marginals = [alpha[-1]]
        for pos in range(values.shape[1] - 2, -1, -1):
            beta, _ = validation.normalize(potential.backward_mul(unary[:, pos + 1] * beta))
            marginals.append(validation.normalize(alpha[pos] * beta)[0])
        result[batches, positions] = torch.stack(marginals[::-1], dim=1)
        valid.logical_and_(validation.valid.all())
    _finish(valid)
    return result


@torch.no_grad()
def sample_sparse_chain(log_unary, potential, generator=None, *, clamped_states=None, max_chunk_tokens=4096):
    """Joint FFBS draw; law-equivalent but not seed-identical to unsplit FFBS."""
    states, buckets, _, valid = _prepare(log_unary, potential, clamped_states, max_chunk_tokens)
    result = states.clone()
    for batches, positions, values in _chunks(log_unary, potential, buckets, max_chunk_tokens):
        validation = _Validation(len(values), values.device)
        _, alpha, _ = _forward(values, potential, validation, need_logz=False, store=True)
        draws = torch.empty(values.shape[:2], dtype=torch.long, device=values.device)
        draws[:, -1] = torch.multinomial(alpha[-1], 1, generator=generator).squeeze(-1)
        for pos in range(values.shape[1] - 2, -1, -1):
            probability = validation.normalize(alpha[pos] * potential.selected_columns(draws[:, pos + 1]))[0]
            draws[:, pos] = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
        result[batches, positions] = draws
        valid.logical_and_(validation.valid.all())
    _finish(valid)
    return result
