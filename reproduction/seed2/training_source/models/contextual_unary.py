"""An active unary control for a contextual token-pair output layer.

The adapter adds ``<E[token], W_h LN(hidden) + W_t time> / sqrt(rank)``
to each base logit. Every distribution still factorizes across positions.
There are no graph parameters, pair factors, or deliberately unused parameters.

``correction_domain='candidates'`` changes only the base model's top-K logits.
All omitted tokens keep their base logits, exactly matching the forest head's
candidate support and unchanged within-residual distribution. ``'full'`` allows
corrections to every vocabulary item and should be reported as a separate
baseline. Candidates are selected from base logits without seeing targets.

The time embedding, hidden normalization and V-by-R token embedding match the
forest pair branch. The context projections emit R values instead of its 2R
FiLM values, so this control has R*(H+T+1) fewer active parameters. Report that
difference; topology parameters in the forest do not belong to its pair branch.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.structured_decoder import ScalarTimestepEmbedding


@dataclass
class CandidateUnaryLattice:
  """Exact compact distribution: K explicit tokens and one residual state."""

  candidate_ids: torch.Tensor
  unary_log_potentials: torch.Tensor
  candidate_state_mask: torch.Tensor
  residual_log_mass: torch.Tensor

  @property
  def log_probs(self) -> torch.Tensor:
    return self.unary_log_potentials.log_softmax(dim=-1)


class ContextualUnaryAdapter(nn.Module):
  """Low-rank contextual corrections to a frozen backbone's unary logits.

  ``forward`` returns exact full-vocabulary logits for ordinary sampling.
  ``target_log_probs`` avoids materializing corrected B-by-L-by-V logits and
  uses checkpointed vocabulary chunks during training. ``candidate_lattice``
  exposes the equivalent K+1-state distribution in candidate mode.

  Projection weights start at zero and token embeddings start nonzero. Thus
  the first forward preserves the backbone exactly, the first backward trains
  the projections, and subsequent backwards also train the embedding, hidden
  normalization, and time MLP. No parameter is permanently inactive.

  Backbone freezing belongs to the caller: use ``torch.no_grad()`` around the
  backbone pass, or pass detached logits and hidden states.
  """

  def __init__(self, hidden_size: int, vocab_size: int, rank: int = 16,
               time_embed_dim: int = 64, top_k: int = 64,
               correction_domain: str = 'candidates'):
    super().__init__()
    if hidden_size < 1 or vocab_size < 2 or rank < 1:
      raise ValueError('hidden_size/rank must be positive and vocab_size >= 2')
    if not 1 <= top_k <= vocab_size:
      raise ValueError('top_k must lie in [1, vocab_size]')
    if correction_domain not in ('full', 'candidates'):
      raise ValueError("correction_domain must be 'full' or 'candidates'")
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.rank = rank
    self.time_embed_dim = time_embed_dim
    self.top_k = top_k
    self.correction_domain = correction_domain
    self.hidden_norm = nn.LayerNorm(hidden_size)
    self.time_embedding = ScalarTimestepEmbedding(time_embed_dim)
    self.token_embedding = nn.Embedding(vocab_size, rank)
    self.hidden_projection = nn.Linear(hidden_size, rank)
    self.time_projection = nn.Linear(time_embed_dim, rank, bias=False)
    nn.init.normal_(self.token_embedding.weight, std=0.02)
    nn.init.zeros_(self.hidden_projection.weight)
    nn.init.zeros_(self.hidden_projection.bias)
    nn.init.zeros_(self.time_projection.weight)

  @property
  def parameter_count(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def parameter_summary(self) -> dict:
    """Actual active capacity and the corresponding forest pair-branch count."""
    projection_difference = self.rank * (
      self.hidden_size + self.time_embed_dim + 1)
    return {
      'trainable_parameters': self.parameter_count,
      'forest_pair_branch_parameters': (
        self.parameter_count + projection_difference),
      'difference_from_forest_pair_branch': -projection_difference,
      'correction_domain': self.correction_domain,
      'top_k': self.top_k,
      'rank': self.rank,
    }

  def _validate(self, hidden: torch.Tensor, base_logits: torch.Tensor,
                active_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
      raise ValueError('hidden must have shape [B,L,hidden_size]')
    if (base_logits.ndim != 3
        or base_logits.shape[:2] != hidden.shape[:2]
        or base_logits.shape[-1] != self.vocab_size):
      raise ValueError('base_logits must have shape [B,L,vocab_size]')
    if hidden.device != base_logits.device:
      raise ValueError('hidden and base_logits must be on the same device')
    if active_mask is None:
      return torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    if (active_mask.shape != hidden.shape[:2]
        or active_mask.dtype != torch.bool
        or active_mask.device != hidden.device):
      raise ValueError('active_mask must be a boolean [B,L] tensor on device')
    return active_mask

  @staticmethod
  def _work_dtype(logits: torch.Tensor) -> torch.dtype:
    return torch.float64 if logits.dtype == torch.float64 else torch.float32

  def context_features(self, hidden: torch.Tensor,
                       timestep: torch.Tensor) -> torch.Tensor:
    """Compute one R-vector per position; no target tokens enter this path."""
    hidden = hidden.to(dtype=self.hidden_norm.weight.dtype)
    time = self.time_embedding(
      torch.as_tensor(timestep, device=hidden.device), hidden.shape[0])
    return (self.hidden_projection(self.hidden_norm(hidden))
            + self.time_projection(time)[:, None, :]) / math.sqrt(self.rank)

  def _corrections(self, features: torch.Tensor,
                   candidate_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    if candidate_ids is None:
      return features @ self.token_embedding.weight.T
    return (features.unsqueeze(-2)
            * self.token_embedding(candidate_ids)).sum(dim=-1)

  def forward(self, hidden: torch.Tensor, base_logits: torch.Tensor,
              timestep: torch.Tensor,
              active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return corrected full-vocabulary logits; inactive rows stay unchanged."""
    active = self._validate(hidden, base_logits, active_mask)
    features = self.context_features(hidden, timestep)
    work = base_logits.to(self._work_dtype(base_logits))
    if self.correction_domain == 'full':
      correction = self._corrections(features).to(work.dtype)
      return work + correction * active.unsqueeze(-1)
    ids = base_logits.topk(self.top_k, dim=-1).indices
    correction = self._corrections(features, ids).to(work.dtype)
    return work.scatter_add(-1, ids, correction * active.unsqueeze(-1))

  @staticmethod
  def _tail_log_mass(base: torch.Tensor, ids: torch.Tensor,
                     vocab_chunk_size: int) -> torch.Tensor:
    """Sum omitted base logits exactly, including nearly zero-mass tails."""
    tail = base.new_full(base.shape[:-1], -torch.inf)
    for start in range(0, base.shape[-1], vocab_chunk_size):
      stop = min(start + vocab_chunk_size, base.shape[-1])
      chunk = base[..., start:stop].clone()
      inside = (ids >= start) & (ids < stop)
      # Duplicate clamped indices are safe here: scatter_add counts membership
      # instead of overwriting a valid exclusion with an out-of-chunk entry.
      excluded = torch.zeros_like(chunk, dtype=torch.int32)
      excluded.scatter_add_(
        -1, (ids - start).clamp(0, stop - start - 1), inside.to(torch.int32))
      chunk = chunk.masked_fill(excluded.bool(), -torch.inf)
      finite_chunk = torch.isfinite(chunk).any(dim=-1)
      safe_chunk = torch.where(finite_chunk[..., None], chunk, torch.zeros_like(chunk))
      chunk_mass = torch.logsumexp(safe_chunk, dim=-1).masked_fill(
        ~finite_chunk, -torch.inf)
      has_mass = torch.isfinite(tail) | finite_chunk
      # Avoid differentiating logsumexp/logaddexp at an all--infinity input
      # when a chunk contains only explicit states (or K covers the vocabulary).
      tail = torch.logaddexp(
        torch.where(has_mass, tail, torch.zeros_like(tail)),
        torch.where(has_mass, chunk_mass, torch.zeros_like(chunk_mass)),
      ).masked_fill(~has_mass, -torch.inf)
    return tail

  def candidate_lattice(self, hidden: torch.Tensor, base_logits: torch.Tensor,
                        timestep: torch.Tensor,
                        active_mask: Optional[torch.Tensor] = None,
                        vocab_chunk_size: int = 4096) -> CandidateUnaryLattice:
    """Return the exact K+1 lattice with unmodified omitted-token conditionals.

    Only available in candidate mode. The returned residual probability is
    allocated among omitted tokens in proportion to their original base logits.
    """
    if self.correction_domain != 'candidates':
      raise ValueError('candidate_lattice requires candidate correction mode')
    if vocab_chunk_size < 1:
      raise ValueError('vocab_chunk_size must be positive')
    active = self._validate(hidden, base_logits, active_mask)
    features = self.context_features(hidden, timestep)
    base = base_logits.to(self._work_dtype(base_logits))
    values, ids = base.topk(self.top_k, dim=-1)
    values = values + self._corrections(features, ids).to(base.dtype) * active[..., None]
    residual = self._tail_log_mass(base, ids, vocab_chunk_size)
    potentials = torch.cat((values, residual[..., None]), dim=-1)
    return CandidateUnaryLattice(ids, potentials, torch.isfinite(potentials), residual)

  def target_log_probs(self, hidden: torch.Tensor, base_logits: torch.Tensor,
                       timestep: torch.Tensor, targets: torch.Tensor,
                       active_mask: Optional[torch.Tensor] = None,
                       position_chunk_size: int = 32,
                       vocab_chunk_size: int = 4096,
                       checkpoint_chunks: bool = True) -> torch.Tensor:
    """Exact log p(target) at active positions, with zeros at inactive ones.

    Additional full-vocabulary working tensors are at most
    ``position_chunk_size * vocab_size`` for the selected base rows, and
    ``position_chunk_size * vocab_chunk_size`` for corrected vocabulary chunks.
    Checkpointing recomputes chunk logits in backward instead of retaining all
    corrected vocabulary activations. There are never V-by-V allocations.
    Targets must be valid vocabulary IDs even when their positions are inactive.
    """
    if position_chunk_size < 1 or vocab_chunk_size < 1:
      raise ValueError('chunk sizes must be positive')
    active = self._validate(hidden, base_logits, active_mask)
    if targets.shape != active.shape or targets.dtype != torch.long:
      raise ValueError('targets must be a long [B,L] tensor')
    if (targets.device != hidden.device
        or bool(((targets < 0) | (targets >= self.vocab_size)).any())):
      raise ValueError('targets must be valid vocabulary IDs on device')
    features = self.context_features(hidden, timestep).reshape(-1, self.rank)
    selected = active.flatten().nonzero().flatten()
    result = base_logits.new_zeros(active.numel(), dtype=self._work_dtype(base_logits))
    # Preserve a differentiable zero for all-observed batches.
    result = result + features.sum() * 0.0
    base_flat = base_logits.reshape(-1, self.vocab_size)
    targets_flat = targets.flatten()
    values = []
    for offset in range(0, selected.numel(), position_chunk_size):
      indices = selected[offset:offset + position_chunk_size]
      base = base_flat[indices].to(result.dtype)
      feat = features[indices]
      target = targets_flat[indices]
      target_logits = base.gather(-1, target[:, None]).squeeze(-1)
      if self.correction_domain == 'candidates':
        candidate_logits, ids = base.topk(self.top_k, dim=-1)
        correction = self._corrections(feat, ids).to(base.dtype)
        residual = self._tail_log_mass(base, ids, vocab_chunk_size)
        normalizer = torch.logsumexp(torch.cat((
          candidate_logits + correction, residual[:, None]), dim=-1), dim=-1)
        matches = ids == target[:, None]
        target_logits = target_logits + (correction * matches).sum(dim=-1)
      else:
        target_logits = target_logits + (
          feat * self.token_embedding(target)).sum(dim=-1).to(base.dtype)
        normalizer = base.new_full((base.shape[0],), -torch.inf)
        for start in range(0, self.vocab_size, vocab_chunk_size):
          stop = min(start + vocab_chunk_size, self.vocab_size)

          def chunk_log_mass(base_chunk, context, embedding):
            corrected = base_chunk + (context @ embedding.T).to(base_chunk.dtype)
            nonempty = ~torch.isneginf(corrected).all(dim=-1)
            safe = torch.where(nonempty[:, None], corrected, torch.zeros_like(corrected))
            # A chunk containing only excluded tokens has zero probability.
            # Evaluate its reduction at finite values before masking it out,
            # so backward never encounters the undefined all--inf derivative.
            return torch.logsumexp(safe, dim=-1).masked_fill(~nonempty, -torch.inf)

          args = (base[:, start:stop], feat, self.token_embedding.weight[start:stop])
          if checkpoint_chunks and torch.is_grad_enabled():
            mass = checkpoint(chunk_log_mass, *args, use_reentrant=False)
          else:
            mass = chunk_log_mass(*args)
          nonempty = ~torch.isneginf(normalizer) | ~torch.isneginf(mass)
          normalizer = torch.logaddexp(
            torch.where(nonempty, normalizer, torch.zeros_like(normalizer)),
            torch.where(nonempty, mass, torch.zeros_like(mass)),
          ).masked_fill(~nonempty, -torch.inf)
      values.append(target_logits - normalizer)
    if values:
      result = result.index_copy(0, selected, torch.cat(values))
    return result.reshape(active.shape)

  def nll_loss(self, hidden: torch.Tensor, base_logits: torch.Tensor,
               timestep: torch.Tensor, targets: torch.Tensor,
               active_mask: torch.Tensor, **chunk_options) -> torch.Tensor:
    """Mean conditional denoising NLL over masked tokens (zero if empty)."""
    log_probs = self.target_log_probs(
      hidden, base_logits, timestep, targets, active_mask, **chunk_options)
    return -log_probs.sum() / active_mask.sum().clamp_min(1)
