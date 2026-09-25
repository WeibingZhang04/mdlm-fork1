"""Dense, exact inference in a chain over top-K-plus-tail candidate states.

An edge table stores *log* potentials, indexed [left state, right state].
Residual candidates have neutral pair potentials. The resulting distribution
has full token support, but is not an exact full-vocabulary pairwise CRF.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


def _logsumexp(x: Tensor, dim: int) -> Tensor:
    """A logsumexp whose gradient is zero on an all-negative-infinity slice."""
    valid = torch.isfinite(x).any(dim=dim, keepdim=True)
    safe = torch.where(valid, x, torch.zeros_like(x))
    out = torch.logsumexp(safe, dim=dim)
    return torch.where(valid.squeeze(dim), out, torch.full_like(out, -torch.inf))


def _inputs(unary: Tensor, edge: Tensor):
    if unary.ndim != 3 or unary.shape[1] < 1:
        raise ValueError("unary must have shape [batch, length>=1, states]")
    b, length, states = unary.shape
    if edge.shape != (b, length - 1, states, states):
        raise ValueError("edge must have shape [batch, length-1, states, states]")
    dtype = torch.float64 if unary.dtype == torch.float64 or edge.dtype == torch.float64 else torch.float32
    return unary.to(dtype), edge.to(dtype)


def _forward(unary: Tensor, edge: Tensor):
    messages = [unary[:, 0]]
    for pos in range(1, unary.shape[1]):
        messages.append(unary[:, pos] + _logsumexp(messages[-1].unsqueeze(-1) + edge[:, pos - 1], -2))
    return messages


def chain_log_partition(unary: Tensor, edge: Tensor) -> Tensor:
    """Return log Z [B]; all inference is at least float32, never bf16."""
    unary, edge = _inputs(unary, edge)
    return _logsumexp(_forward(unary, edge)[-1], -1)


def chain_log_prob(unary: Tensor, edge: Tensor, states: Tensor) -> Tensor:
    """Normalized log probability of state sequences [B,L]."""
    unary, edge = _inputs(unary, edge)
    if states.shape != unary.shape[:2]:
        raise ValueError("states must have shape [batch, length]")
    score = unary.gather(-1, states.unsqueeze(-1)).squeeze(-1).sum(-1)
    if unary.shape[1] > 1:
        b = torch.arange(unary.shape[0], device=unary.device).unsqueeze(1)
        t = torch.arange(unary.shape[1] - 1, device=unary.device).unsqueeze(0)
        score = score + edge[b, t, states[:, :-1], states[:, 1:]].sum(-1)
    return score - chain_log_partition(unary, edge)


def chain_log_marginals(unary: Tensor, edge: Tensor) -> Tensor:
    """Exact node log probabilities [B,L,S], retaining very unlikely states."""
    unary, edge = _inputs(unary, edge)
    alpha = _forward(unary, edge)
    log_z = _logsumexp(alpha[-1], -1)
    beta = torch.zeros_like(unary[:, -1])
    marginals = [alpha[-1] - log_z[:, None]]
    for pos in range(unary.shape[1] - 2, -1, -1):
        beta = _logsumexp(edge[:, pos] + (unary[:, pos + 1] + beta).unsqueeze(-2), -1)
        marginals.append(alpha[pos] + beta - log_z[:, None])
    return torch.stack(marginals[::-1], dim=1)


def chain_marginals(unary: Tensor, edge: Tensor) -> Tensor:
    """Exact node probabilities [B,L,S], not independently drawn joint samples."""
    return chain_log_marginals(unary, edge).exp()


@torch.no_grad()
def sample_chain(unary: Tensor, edge: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """Forward filtering/backward sampling; returns a joint draw [B,L]."""
    unary, edge = _inputs(unary, edge)
    alpha = _forward(unary, edge)
    states = torch.empty(unary.shape[:2], dtype=torch.long, device=unary.device)
    states[:, -1] = torch.multinomial(alpha[-1].double().softmax(-1), 1, generator=generator).squeeze(-1)
    for pos in range(unary.shape[1] - 2, -1, -1):
        selected_edge = edge[:, pos].gather(-1, states[:, pos + 1, None, None].expand(-1, unary.shape[-1], 1)).squeeze(-1)
        probs = (alpha[pos] + selected_edge).double().softmax(-1)
        states[:, pos] = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return states


@dataclass
class CandidateBatch:
    """Candidate support and the exact within-tail distribution.

    Masked positions: K explicit tokens followed by residual id -1. Visible
    positions: observed token at slot zero, all other slots invalid (-inf
    unary, id -1). Gold states are never inserted into the support. The tail
    log probability is the gold token's conditional probability *inside* its
    residual state, and zero for explicit/visible targets.
    """

    candidate_ids: Tensor
    unary: Tensor
    masked: Tensor
    gold_states: Optional[Tensor]
    gold_tail_logprob: Optional[Tensor]
    normalized_log_probs: Tensor


def build_candidates(log_probs: Tensor, masked_input: Tensor, mask_id: int, k: int,
                     gold: Optional[Tensor] = None) -> CandidateBatch:
    """Build candidates without changing held-out support to include gold.

    Input probabilities are normalized after excluding the mask token. Tail
    mass is computed with logsumexp of excluded tokens (not 1-sum(topK)), so
    small tails remain numerically stable. k may be zero or >= vocabulary.
    """
    if log_probs.ndim != 3 or masked_input.shape != log_probs.shape[:2]:
        raise ValueError("log_probs [B,L,V] and masked_input [B,L] are required")
    if k < 0 or not 0 <= mask_id < log_probs.shape[-1]:
        raise ValueError("k must be nonnegative and mask_id inside the vocabulary")
    vocab = log_probs.shape[-1]
    k = min(k, vocab - 1)
    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    lp = log_probs.to(dtype).clone()
    lp[..., mask_id] = -torch.inf
    norm = _logsumexp(lp, -1).unsqueeze(-1)
    # Observed MDLM rows may be point masses. A visible mask-token point mass
    # is invalid; well-formed visible rows and all masked rows normalize here.
    lp = lp - norm
    # Ask for one extra item and remove MASK explicitly. Merely assigning it
    # -inf is insufficient when other real tokens also have zero probability:
    # topk can break -inf ties by returning MASK instead of a real token.
    extra_values, extra_ids = torch.topk(lp, k + 1, dim=-1)
    order = torch.arange(k + 1, device=lp.device)
    mask_position = torch.where(extra_ids.eq(mask_id), order, k + 1).amin(-1, keepdim=True)
    keep = torch.arange(k, device=lp.device).expand(*lp.shape[:2], k)
    keep = keep + keep.ge(mask_position).long()
    top_values = extra_values.gather(-1, keep)
    top_ids = extra_ids.gather(-1, keep)
    tail = lp.clone()
    tail.scatter_(-1, top_ids, -torch.inf)
    tail_mass = _logsumexp(tail, -1)
    ids = torch.cat((top_ids, torch.full_like(masked_input.unsqueeze(-1), -1)), -1)
    unary = torch.cat((top_values, tail_mass.unsqueeze(-1)), -1)
    masked = masked_input.eq(mask_id)
    visible_ids = torch.full_like(ids, -1)
    visible_ids[..., 0] = masked_input
    visible_unary = torch.full_like(unary, -torch.inf)
    visible_unary[..., 0] = 0.
    ids = torch.where(masked.unsqueeze(-1), ids, visible_ids)
    unary = torch.where(masked.unsqueeze(-1), unary, visible_unary)
    gold_states = gold_tail = None
    if gold is not None:
        if gold.shape != masked_input.shape:
            raise ValueError("gold must have shape [B,L]")
        if torch.any((gold != masked_input) & ~masked):
            raise ValueError("gold disagrees with an observed/clamped token")
        if torch.any(gold.eq(mask_id)):
            raise ValueError("clean gold sequences cannot contain the mask token")
        matches = ids.eq(gold.unsqueeze(-1)) & ids.ge(0)
        explicit = matches.any(-1)
        gold_states = torch.where(explicit, matches.long().argmax(-1), torch.full_like(gold, k))
        gold_lp = lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
        # Avoid undefined -inf - -inf on unused states of point-mass rows.
        is_tail = masked & ~explicit & torch.isfinite(tail_mass)
        numerator = torch.where(is_tail, gold_lp, torch.zeros_like(gold_lp))
        denominator = torch.where(is_tail, tail_mass, torch.zeros_like(tail_mass))
        gold_tail = numerator - denominator
    return CandidateBatch(ids, unary, masked, gold_states, gold_tail, lp)


def gold_log_prob(candidates: CandidateBatch, edge: Tensor, unary_delta: Optional[Tensor] = None) -> Tensor:
    """Exact clean-token conditional log probability [B], including tail tokens."""
    if candidates.gold_states is None or candidates.gold_tail_logprob is None:
        raise ValueError("build_candidates must receive gold for gold_log_prob")
    unary = candidates.unary if unary_delta is None else candidates.unary + unary_delta
    return chain_log_prob(unary, edge, candidates.gold_states) + candidates.gold_tail_logprob.sum(-1)


@torch.no_grad()
def sample_candidate_tokens(candidates: CandidateBatch, states: Tensor,
                            generator: Optional[torch.Generator] = None) -> Tensor:
    """Expand sampled residual states using the backbone conditional tail law."""
    tokens = candidates.candidate_ids.gather(-1, states.unsqueeze(-1)).squeeze(-1)
    residual = tokens.lt(0)
    if residual.any():
        tail_lp = candidates.normalized_log_probs[residual].clone()
        explicit = candidates.candidate_ids[residual]
        # All residual-selected rows are masked, so the first S-1 slots are
        # explicit candidates (possibly zero slots when K=0).
        tail_lp.scatter_(-1, explicit[:, :-1], -torch.inf)
        tokens[residual] = torch.multinomial(tail_lp.double().softmax(-1), 1, generator=generator).squeeze(-1)
    return tokens
