"""Exact chain inference after eliminating deterministic visible positions.

Visible positions split a chain into independent contiguous masked runs. Each
boundary edge is absorbed into the adjacent run's endpoint unary. Runs are
batched with deterministic state-zero padding, so dynamic-programming depth is
the longest masked run, rather than the original sequence length. This changes
neither the candidate support nor the within-tail token distribution.

The dense edge input still uses *original-position adjacency*. In particular,
two masks separated by a visible token must not receive a new pair factor.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .core import (_inputs, chain_log_marginals, chain_log_partition,
                   sample_chain)


@dataclass
class SegmentBatch:
    """Reusable packed runs and the map back to the original batch.

    ``unary`` is [runs, longest_run, states], ``edge`` is the corresponding
    padded chain, and ``constant`` contains visible unary and visible-visible
    edge scores [batch]. Consequently log_partition includes those constants
    exactly. If the caller has already removed visible-visible factors, they
    contribute zero here too. Padding is deterministic and has partition one.
    """

    unary: Tensor
    edge: Tensor
    constant: Tensor
    run_batch: Tensor
    run_start: Tensor
    lengths: Tensor
    positions: Tensor
    valid: Tensor
    clamped_states: Tensor

    @property
    def run_count(self) -> int:
        return self.unary.shape[0]

    @property
    def padded_length(self) -> int:
        return self.unary.shape[1]

    def log_partition(self) -> Tensor:
        if self.run_count == 0:
            return self.constant
        return self.constant.index_add(
            0, self.run_batch, chain_log_partition(self.unary, self.edge))

    def log_marginals(self) -> Tensor:
        result = self.unary.new_full(
            (*self.clamped_states.shape, self.unary.shape[-1]), -torch.inf)
        result.scatter_(-1, self.clamped_states.unsqueeze(-1), 0.)
        if self.run_count:
            values = chain_log_marginals(self.unary, self.edge)
            batches = self.run_batch[:, None].expand_as(self.positions)
            result[batches[self.valid], self.positions[self.valid]] = values[self.valid]
        return result

    def marginals(self) -> Tensor:
        return self.log_marginals().exp()

    @torch.no_grad()
    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        """Joint FFBS draw [B,L], independent across clamp-separated runs.

        The core sampler uses FP64 categorical probabilities. RNG consumption
        differs from dense FFBS; equivalence is in distribution, not draw IDs.
        """
        result = self.clamped_states.clone()
        if self.run_count:
            values = sample_chain(self.unary, self.edge, generator)
            batches = self.run_batch[:, None].expand_as(self.positions)
            result[batches[self.valid], self.positions[self.valid]] = values[self.valid]
        return result


def pack_segments(unary: Tensor, edge: Tensor, masked: Tensor) -> SegmentBatch:
    """Pack all contiguous masked runs without a Python loop over runs.

    Every unmasked row must have exactly one finite unary (any state index is
    accepted). Masked rows retain all original states, including impossible
    states and any residual state. Log scores may have either sign. Input and
    output remain differentiable with respect to finite unary/edge scores;
    the mask and clamp-state identities are discrete structure.
    """
    unary, edge = _inputs(unary, edge)
    batch, length, states = unary.shape
    if masked.shape != (batch, length) or masked.dtype != torch.bool:
        raise ValueError("masked must be a boolean [batch, length] tensor")
    if masked.device != unary.device or edge.device != unary.device:
        raise ValueError("unary, edge, and masked must share a device")
    finite = torch.isfinite(unary)
    if torch.any((~masked) & finite.sum(-1).ne(1)):
        raise ValueError("Every visible position must be clamped to one finite state")
    clamped = finite.long().argmax(-1)
    selected_unary = unary.gather(-1, clamped.unsqueeze(-1)).squeeze(-1)
    constant = torch.where(masked, torch.zeros_like(selected_unary), selected_unary).sum(-1)
    if length > 1:
        selected_edges = edge.gather(
            -2, clamped[:, :-1, None, None].expand(-1, -1, 1, states)).squeeze(-2)
        selected_edges = selected_edges.gather(-1, clamped[:, 1:, None]).squeeze(-1)
        known_edges = ~masked[:, :-1] & ~masked[:, 1:]
        constant = constant + torch.where(
            known_edges, selected_edges, torch.zeros_like(selected_edges)).sum(-1)

    starts = masked.clone()
    ends = masked.clone()
    starts[:, 1:] &= ~masked[:, :-1]
    ends[:, :-1] &= ~masked[:, 1:]
    run_indices = starts.nonzero(as_tuple=False)
    end_indices = ends.nonzero(as_tuple=False)
    run_batch, run_start = run_indices.unbind(-1)
    lengths = end_indices[:, 1] - run_start + 1
    width = int(lengths.max()) if lengths.numel() else 1
    offsets = torch.arange(width, device=unary.device)
    positions = run_start[:, None] + offsets
    valid = offsets[None] < lengths[:, None]
    # Dummy entries may extend past sequence end; their gather values are
    # discarded below before any inference. Padding contributes zero score.
    safe_positions = positions.clamp_max(length - 1)
    packed_unary = unary[run_batch[:, None], safe_positions]

    if length > 1 and lengths.numel():
        left_positions = (run_start - 1).clamp_min(0)
        left_states = clamped[run_batch, left_positions]
        left_edges = edge[run_batch, left_positions]
        left_scores = left_edges.gather(
            -2, left_states[:, None, None].expand(-1, 1, states)).squeeze(-2)
        left_scores = torch.where(
            (run_start > 0)[:, None], left_scores, torch.zeros_like(left_scores))
        packed_unary = packed_unary + torch.where(
            offsets[None, :, None].eq(0), left_scores[:, None], 0.)

        run_end = run_start + lengths - 1
        right_positions = run_end.clamp_max(length - 2)
        right_states = clamped[run_batch, (run_end + 1).clamp_max(length - 1)]
        right_edges = edge[run_batch, right_positions]
        right_scores = right_edges.gather(
            -1, right_states[:, None, None].expand(-1, states, 1)).squeeze(-1)
        right_scores = torch.where(
            (run_end < length - 1)[:, None], right_scores, torch.zeros_like(right_scores))
        packed_unary = packed_unary + torch.where(
            offsets[None, :, None].eq((lengths - 1)[:, None, None]),
            right_scores[:, None], 0.)

    dummy = unary.new_full((states,), -torch.inf)
    dummy[0] = 0.
    packed_unary = torch.where(valid[..., None], packed_unary, dummy)
    if width > 1:
        packed_edge = edge[run_batch[:, None], positions[:, :-1].clamp_max(length - 2)]
        packed_edge = torch.where(valid[:, 1:, None, None], packed_edge, 0.)
    else:
        # Preserve an empty differentiable view where no internal edge exists.
        packed_edge = edge[run_batch, :0]
    return SegmentBatch(packed_unary, packed_edge, constant, run_batch,
                        run_start, lengths, positions, valid, clamped)


def segmented_log_partition(unary: Tensor, edge: Tensor, masked: Tensor) -> Tensor:
    return pack_segments(unary, edge, masked).log_partition()


def segmented_log_marginals(unary: Tensor, edge: Tensor, masked: Tensor) -> Tensor:
    return pack_segments(unary, edge, masked).log_marginals()


def segmented_marginals(unary: Tensor, edge: Tensor, masked: Tensor) -> Tensor:
    return pack_segments(unary, edge, masked).marginals()


@torch.no_grad()
def sample_segmented_chain(unary: Tensor, edge: Tensor, masked: Tensor,
                           generator: Optional[torch.Generator] = None) -> Tensor:
    return pack_segments(unary, edge, masked).sample(generator)
