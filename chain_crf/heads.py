"""Small ordered pair scorers; all residual-state scores are exactly neutral."""

from typing import Optional

import torch
from torch import Tensor, nn


def _embeddings(table: nn.Embedding, ids: Tensor) -> Tensor:
    return table(ids.clamp_min(0)) * ids.ge(0).unsqueeze(-1)


class GlobalPairHead(nn.Module):
    """An unrestricted signed log pair score L(a)^T R(b).

    Low rank saves parameters, not DP work: exp(LR^T) is generally full rank.
    Separate left/right tables preserve direction and allow disagreement.
    One nonzero and one zero table initialize at the backbone without dead
    bilinear gradients. Hidden states and time are accepted for API parity.
    """

    def __init__(self, vocab_size: int, rank: int = 32):
        super().__init__()
        self.vocab_size, self.rank = vocab_size, rank
        self.left = nn.Embedding(vocab_size, rank)
        self.right = nn.Embedding(vocab_size, rank)
        nn.init.normal_(self.left.weight, std=0.02)
        nn.init.zeros_(self.right.weight)

    def forward(self, candidate_ids: Tensor, hidden: Optional[Tensor] = None,
                time: Optional[Tensor] = None) -> Tensor:
        left = _embeddings(self.left, candidate_ids[:, :-1]).float()
        right = _embeddings(self.right, candidate_ids[:, 1:]).float()
        return torch.einsum("blir,bljr->blij", left, right)


class ContextualPairHead(GlobalPairHead):
    """Global score plus a context/time-conditioned diagonal gate of that score.

    gate(h_i,h_{i+1},t) is a rank-dimensional signed vector. A zero-initialized
    final gate layer starts exactly at the global scorer; load_global can
    initialize from a previously trained global head without changing output.
    """

    def __init__(self, vocab_size: int, hidden_size: int, rank: int = 32, mlp_size: int = 128):
        super().__init__(vocab_size, rank)
        # Both factors must be nonzero here: with contradictory balanced
        # contexts, a zero global factor and zero gate can make *all* initial
        # conditional gradients cancel. Small random scores avoid that trap;
        # load_global overwrites these when warming from a trained head.
        nn.init.normal_(self.right.weight, std=0.02)
        self.hidden_size, self.mlp_size = hidden_size, mlp_size
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Sequential(nn.Linear(hidden_size * 2 + 1, mlp_size), nn.SiLU(), nn.Linear(mlp_size, rank))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def load_global(self, head: GlobalPairHead):
        self.left.load_state_dict(head.left.state_dict())
        self.right.load_state_dict(head.right.state_dict())
        return self

    def forward(self, candidate_ids: Tensor, hidden: Optional[Tensor] = None,
                time: Optional[Tensor] = None) -> Tensor:
        if hidden is None:
            raise ValueError("ContextualPairHead requires frozen backbone hidden states")
        if hidden.shape[:2] != candidate_ids.shape[:2] or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must have shape [B,L,hidden_size]")
        b, length = candidate_ids.shape[:2]
        if time is None:
            time = hidden.new_zeros(b)
        time = torch.as_tensor(time, device=hidden.device, dtype=hidden.dtype)
        if time.numel() == 1:
            time = time.expand(b)
        time = time.reshape(b, 1, 1).expand(b, max(length - 1, 0), 1)
        normalized = self.norm(hidden.to(self.norm.weight.dtype))
        gate_input = torch.cat((normalized[:, :-1], normalized[:, 1:], time.to(normalized.dtype)), -1)
        gate = self.gate(gate_input).float()
        left = _embeddings(self.left, candidate_ids[:, :-1]).float()
        right = _embeddings(self.right, candidate_ids[:, 1:]).float()
        return torch.einsum("blir,bljr->blij", left * (1. + gate.unsqueeze(-2)), right)


class IndependentHead(nn.Module):
    """Low-rank contextual unary adapter with approximately pair-head capacity.

    rank=64 roughly matches the 2*V*32 global pair embeddings. Residual token
    scores are zero, so its within-tail distribution stays the backbone law.
    All visible unary shifts cancel in normalization because they are clamped.
    """

    def __init__(self, vocab_size: int, hidden_size: int, rank: int = 64):
        super().__init__()
        self.vocab_size, self.hidden_size, self.rank = vocab_size, hidden_size, rank
        self.embedding = nn.Embedding(vocab_size, rank)
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size + 1, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, candidate_ids: Tensor, hidden: Optional[Tensor] = None,
                time: Optional[Tensor] = None) -> Tensor:
        if hidden is None:
            raise ValueError("IndependentHead requires frozen backbone hidden states")
        b, length = candidate_ids.shape[:2]
        if hidden.shape != (b, length, self.hidden_size):
            raise ValueError("hidden must have shape [B,L,hidden_size]")
        if time is None:
            time = hidden.new_zeros(b)
        time = torch.as_tensor(time, device=hidden.device, dtype=hidden.dtype)
        if time.numel() == 1:
            time = time.expand(b)
        time = time.reshape(b, 1, 1).expand(b, length, 1)
        normalized = self.norm(hidden.to(self.norm.weight.dtype))
        context = self.projection(torch.cat((normalized, time.to(normalized.dtype)), -1)).float()
        embeddings = _embeddings(self.embedding, candidate_ids).float()
        scores = torch.einsum("blsr,blr->bls", embeddings, context)
        # Visible rows have just one explicit state. A masked row may also
        # have one explicit state when K=1, so don't infer masking from ids.
        return scores
