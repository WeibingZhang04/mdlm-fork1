"""Training-document bigram baselines with sparse, vectorized pair lookup."""

from collections import Counter
from typing import Iterable, Optional

import torch
from torch import Tensor, nn


class CountBigramHead(nn.Module):
    """Smoothed PMI or log-conditional count score.

    For N observed within-document edges, empirical joint p_emp and add-one
    smoothed edge-endpoint backoff distributions bL,bR:

        p_pair = (1-epsilon) p_emp + epsilon bL bR.

    PMI and conditional scores use the actual marginals of this p_pair, not
    just the backoff distributions, so the conditional really normalizes.

    epsilon=smoothing is the independence-mixture weight in (0,1]. There
    are no across-document edges. The residual candidate always scores zero.
    Counts and marginals use float64; returned pair scores use float32.
    """

    def __init__(self, vocab_size: int, mode: str = "pmi", strength: float = 1., smoothing: float = 0.1):
        super().__init__()
        if mode not in ("pmi", "conditional"):
            raise ValueError("mode must be 'pmi' or 'conditional'")
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0,1]")
        self.vocab_size, self.mode = vocab_size, mode
        self.strength, self.smoothing = float(strength), float(smoothing)
        self.register_buffer("left_counts", torch.zeros(vocab_size, dtype=torch.float64))
        self.register_buffer("right_counts", torch.zeros(vocab_size, dtype=torch.float64))
        self.register_buffer("pair_keys", torch.empty(0, dtype=torch.long))
        self.register_buffer("pair_counts", torch.empty(0, dtype=torch.float64))

    def fit(self, token_sequences: Iterable):
        """Reset counts from an iterable of documents (lists or 1-D tensors)."""
        pairs = Counter()
        left = torch.zeros(self.vocab_size, dtype=torch.float64)
        right = torch.zeros(self.vocab_size, dtype=torch.float64)
        for sequence in token_sequences:
            ids = torch.as_tensor(sequence, dtype=torch.long, device="cpu").reshape(-1)
            if ids.numel() and (ids.min() < 0 or ids.max() >= self.vocab_size):
                raise ValueError("training token id outside vocabulary")
            if ids.numel() < 2:
                continue
            keys, counts = torch.unique(ids[:-1] * self.vocab_size + ids[1:], return_counts=True)
            pairs.update(dict(zip(keys.tolist(), counts.tolist())))
            left.index_add_(0, keys // self.vocab_size, counts.double())
            right.index_add_(0, keys % self.vocab_size, counts.double())
        keys = sorted(pairs)
        target = self.left_counts.device
        self.left_counts = left.to(target)
        self.right_counts = right.to(target)
        self.pair_keys = torch.tensor(keys, dtype=torch.long, device=target)
        self.pair_counts = torch.tensor([pairs[key] for key in keys], dtype=torch.float64, device=target)
        return self

    def forward(self, candidate_ids: Tensor, hidden: Optional[Tensor] = None,
                time: Optional[Tensor] = None) -> Tensor:
        a, b = candidate_ids[:, :-1, :, None], candidate_ids[:, 1:, None, :]
        valid = a.ge(0) & b.ge(0)
        safe_a, safe_b = a.clamp_min(0), b.clamp_min(0)
        keys = safe_a * self.vocab_size + safe_b
        n = self.left_counts.sum()
        pl = (self.left_counts + 1.) / (n + self.vocab_size)
        pr = (self.right_counts + 1.) / (n + self.vocab_size)
        base = pl[safe_a] * pr[safe_b]
        counts = torch.zeros_like(base)
        if self.pair_keys.numel():
            idx = torch.searchsorted(self.pair_keys, keys).clamp_max(self.pair_keys.numel() - 1)
            counts = torch.where(self.pair_keys[idx].eq(keys), self.pair_counts[idx], counts)
        empirical = counts / n.clamp_min(1.)
        # Empty training data means the independence model, not missing mass.
        eps = torch.where(n > 0, n.new_tensor(self.smoothing), n.new_tensor(1.))
        joint = (1. - eps) * empirical + eps * base
        marginal_l = (1. - eps) * self.left_counts / n.clamp_min(1.) + eps * pl
        marginal_r = (1. - eps) * self.right_counts / n.clamp_min(1.) + eps * pr
        score = joint.log() - marginal_l[safe_a].log()
        if self.mode == "pmi":
            score = score - marginal_r[safe_b].log()
        return torch.where(valid, score * self.strength, torch.zeros_like(score)).float()

    def save(self, path):
        torch.save({"format": "chain_crf_counts_v1", "vocab_size": self.vocab_size,
                    "mode": self.mode, "strength": self.strength, "smoothing": self.smoothing,
                    **{name: tensor.detach().cpu() for name, tensor in self.state_dict().items()}}, path)

    @classmethod
    def load(cls, path, map_location="cpu", mode=None, strength=None):
        data = torch.load(path, map_location=map_location, weights_only=True)
        if data.get("format") != "chain_crf_counts_v1":
            raise ValueError("unsupported count-file format")
        model = cls(data["vocab_size"], mode or data["mode"],
                    data["strength"] if strength is None else strength, data["smoothing"])
        for name in ("left_counts", "right_counts", "pair_keys", "pair_counts"):
            setattr(model, name, data[name])
        return model
