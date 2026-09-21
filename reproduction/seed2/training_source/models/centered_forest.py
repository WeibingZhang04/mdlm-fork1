"""Experimental forest coupling that cannot change a frozen unary adapter.

This is a standalone learning test, not a production head or a real-text
result. A private, frozen candidate-only unary adapter supplies normalized
FP64 singleton beliefs b_i. On the natural-order chain of active positions,
each edge has ratio 1 + eta * mean_r f_i,r g_j,r, with features centered under
their endpoint beliefs. Consequently the forest has normalizer one and
exactly the supplied singleton marginals (up to FP64 arithmetic).

Only explicit top-K candidates interact. Residual interactions are neutral,
and the omitted-token conditional remains the backbone conditional. Positive
PMI per edge is bounded by log(1+eta); this is not a universal coupling family.
Endpoint rank is 2*feature_dim+1, not feature_dim. Existing forest inference
accepts the explicit factors and inserts neutral residual rows/columns itself.

The reference unary distribution matches the selected-test evaluator: evaluate
the frozen adapter's compact lattice in its ordinary precision, then promote
and normalize that lattice in FP64. Separately normalize the within-residual
backbone conditional in FP64. This deliberately retains the original compact
residual mass, not a newly recomputed FP64 aggregate. Original-dtype top-K and
explicit logit addition are preserved. Autocast is disabled. All backbone
inputs are detached; no backbone or copied unary parameter receives gradients.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn

from evaluation.centered_coupling import centered_coupling
from models.contextual_unary import ContextualUnaryAdapter
from models.structured_decoder import (
  ScalarTimestepEmbedding, StructuredDecoderOutput, _fixed_chain_edges,
)


@dataclass
class CenteredForestOutput(StructuredDecoderOutput):
  """Structured output plus the fixed context needed for direct scoring.

  ``residual_log_mass`` is the normalized residual-state log probability;
  ``base_tail_log_mass`` is the unnormalized omitted backbone logit mass.
  Do not modify this bundle or score it with a different backbone-logit input.
  """

  base_tail_log_mass: torch.Tensor          # [B,L], detached FP64
  scoring_active_mask: torch.Tensor         # [B,L], detached boolean copy
  vocab_size: int

  def _validate_logits(self, unary_logits: torch.Tensor) -> None:
    if (unary_logits.shape != (*self.candidate_ids.shape[:2], self.vocab_size)
        or unary_logits.device != self.candidate_ids.device
        or not unary_logits.is_floating_point()):
      raise ValueError('unary_logits must be the original floating [B,L,V] tensor on device')

  def residual_log_probs(self, unary_logits: torch.Tensor) -> torch.Tensor:
    """FP64 omitted-token conditional with a safe zero-mass extension.

    An empty residual has probability zero, so its conditional is undefined.
    Define that unused conditional as a point mass on candidate zero. Existing
    joint sampling draws residual tokens eagerly for every node; this extension
    avoids zero-sum multinomial rows in mixed-support batches. It cannot affect
    scores or marginals, and a disabled residual state is never sampled.
    """
    self._validate_logits(unary_logits)
    with torch.autocast(device_type=unary_logits.device.type, enabled=False):
      tail = unary_logits.detach().double().clone()
      tail.scatter_(-1, self.candidate_ids, -torch.inf)
      has_tail = torch.isfinite(self.base_tail_log_mass)
      normalizer = torch.where(has_tail, self.base_tail_log_mass,
                               torch.zeros_like(self.base_tail_log_mass))
      conditional = tail - normalizer[..., None]
      fallback = torch.full_like(tail, -torch.inf)
      fallback.scatter_(-1, self.candidate_ids[..., :1], 0.0)
      return torch.where(has_tail[..., None], conditional, fallback)

  def materialize_pair_factors(self) -> torch.Tensor:
    with torch.autocast(device_type=self.pair_left_factors.device.type, enabled=False):
      return super().materialize_pair_factors()


class FrozenUnaryCenteredForestHead(nn.Module):
  """Learn bounded dependence while keeping a private unary model frozen.

  ``forward(hidden, raw_logits, sigma, active_mask)`` accepts the backbone's
  frozen context; sigma is its scalar noise conditioning, not a mask rate.
  No targets enter candidate selection, topology, or coupling construction.

  Endpoint features are separate token embeddings, each multiplied by
  ``1 + W_h LN(hidden) + W_t time(sigma)``. Projections start at zero. The left
  embedding starts random and the right starts zero: initialization is exact
  independence with a nonzero right-embedding gradient on dependent examples.
  Other branches receive gradients after the right embedding leaves zero.

  The supplied adapter is deep-copied, not modified. Standard ``train`` and
  ``requires_grad_`` calls keep the private copy frozen. Converting this entire
  module's dtype also converts its private unary reference, as with any PyTorch
  module; construct it from the intended reference dtype before comparison.
  """

  def __init__(self, unary_adapter: ContextualUnaryAdapter, feature_dim: int = 8,
               time_embed_dim: int | None = None, eta: float = 0.95,
               embedding_init_std: float = 0.5):
    super().__init__()
    if not isinstance(unary_adapter, ContextualUnaryAdapter):
      raise ValueError('unary_adapter must be a preloaded ContextualUnaryAdapter')
    if unary_adapter.correction_domain != 'candidates':
      raise ValueError('the frozen unary adapter must use candidate-only corrections')
    if type(feature_dim) is not int or feature_dim < 1:
      raise ValueError('feature_dim must be a positive integer')
    if not math.isfinite(eta) or not 0 < eta < 1:
      raise ValueError('eta must lie strictly between zero and one')
    if not math.isfinite(embedding_init_std) or embedding_init_std <= 0:
      raise ValueError('embedding_init_std must be finite and positive')
    time_embed_dim = unary_adapter.time_embed_dim if time_embed_dim is None else time_embed_dim
    if type(time_embed_dim) is not int or time_embed_dim < 2:
      raise ValueError('time_embed_dim must be an integer >= 2')
    parameters = tuple(unary_adapter.parameters())
    device, dtype = parameters[0].device, parameters[0].dtype
    if (dtype not in (torch.float32, torch.float64)
        or any(p.device != device or p.dtype != dtype for p in parameters)):
      raise ValueError('the frozen adapter must have uniform FP32 or FP64 parameters')
    self.hidden_size = unary_adapter.hidden_size
    self.vocab_size = unary_adapter.vocab_size
    self.top_k = unary_adapter.top_k
    self.feature_dim = feature_dim
    self.rank = 2 * feature_dim + 1
    self.time_embed_dim = time_embed_dim
    self.eta = float(eta)
    self._frozen_unary = copy.deepcopy(unary_adapter).requires_grad_(False).eval()
    self.hidden_norm = nn.LayerNorm(self.hidden_size)
    self.time_embedding = ScalarTimestepEmbedding(time_embed_dim)
    self.left_token_embedding = nn.Embedding(self.vocab_size, feature_dim)
    self.right_token_embedding = nn.Embedding(self.vocab_size, feature_dim)
    self.left_hidden_projection = nn.Linear(self.hidden_size, feature_dim)
    self.right_hidden_projection = nn.Linear(self.hidden_size, feature_dim)
    self.left_time_projection = nn.Linear(time_embed_dim, feature_dim, bias=False)
    self.right_time_projection = nn.Linear(time_embed_dim, feature_dim, bias=False)
    nn.init.normal_(self.left_token_embedding.weight, std=embedding_init_std)
    nn.init.zeros_(self.right_token_embedding.weight)
    for projection in (self.left_hidden_projection, self.right_hidden_projection,
                       self.left_time_projection, self.right_time_projection):
      nn.init.zeros_(projection.weight)
      if projection.bias is not None:
        nn.init.zeros_(projection.bias)
    self.to(device=device, dtype=dtype)

  def train(self, mode: bool = True):
    super().train(mode)
    self._frozen_unary.eval()
    return self

  def requires_grad_(self, requires_grad: bool = True):
    super().requires_grad_(requires_grad)
    self._frozen_unary.requires_grad_(False)
    return self

  @property
  def parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

  def parameter_summary(self) -> dict:
    return {
      'trainable_parameters': self.parameter_count,
      'frozen_unary_parameters': sum(p.numel() for p in self._frozen_unary.parameters()),
      'feature_dim': self.feature_dim,
      'positive_factor_rank': self.rank,
      'eta': self.eta,
      'top_k': self.top_k,
      'topology': 'natural_order_active_chain',
      'status': 'standalone_experimental',
    }

  def forward(self, hidden: torch.Tensor, raw_logits: torch.Tensor,
              sigma: torch.Tensor, active_mask: torch.Tensor) -> CenteredForestOutput:
    active = self._frozen_unary._validate(hidden, raw_logits, active_mask)
    if hidden.shape[0] < 1 or hidden.shape[1] < 1:
      raise ValueError('batch and sequence dimensions must be nonempty')
    if raw_logits.dtype not in (torch.float32, torch.float64):
      raise ValueError('raw_logits must be FP32 or FP64 to specify the candidate policy')
    sigma = torch.as_tensor(sigma, device=hidden.device).detach()
    if (hidden.device != self.hidden_norm.weight.device
        or not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(sigma).all())
        or bool((torch.isnan(raw_logits) | torch.isposinf(raw_logits)).any())
        or not bool(torch.isfinite(raw_logits).any(-1).all())):
      raise ValueError('inputs need finite context and nonempty logit support on the head device')
    hidden, raw_logits = hidden.detach(), raw_logits.detach()
    with torch.autocast(device_type=hidden.device.type, enabled=False):
      with torch.no_grad():
        lattice = self._frozen_unary.candidate_lattice(hidden, raw_logits, sigma, active)
        # Match score_head_fp64's reference exactly. Recomputing the aggregate
        # residual before this normalization would change the frozen baseline.
        log_beliefs = lattice.unary_log_potentials.double().log_softmax(-1)
        # The within-residual conditional, in contrast, is normalized FP64.
        tail = self._frozen_unary._tail_log_mass(raw_logits.double(), lattice.candidate_ids, 4096)
        if bool(torch.isnan(log_beliefs).any()):
          raise ValueError('frozen unary produced nonfinite corrected support')
        beliefs = log_beliefs.exp()
        ids = lattice.candidate_ids
      context = self.hidden_norm(hidden.to(self.hidden_norm.weight.dtype))
      time = self.time_embedding(sigma, hidden.shape[0])
      left_scale = 1 + self.left_hidden_projection(context) + self.left_time_projection(time)[:, None]
      right_scale = 1 + self.right_hidden_projection(context) + self.right_time_projection(time)[:, None]
      raw_left = self.left_token_embedding(ids) * left_scale[..., None, :]
      raw_right = self.right_token_embedding(ids) * right_scale[..., None, :]
      edges, edge_mask = _fixed_chain_edges(active, component_size_cap=0)
      batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
      left_nodes, right_nodes = edges[..., 0], edges[..., 1]
      coupling = centered_coupling(
        beliefs[batch, left_nodes], beliefs[batch, right_nodes],
        raw_left[batch, left_nodes], raw_right[batch, right_nodes], eta=self.eta)
      left, right = coupling.left_factors[..., :-1, :], coupling.right_factors[..., :-1, :]
      batch_size, length = active.shape
      return CenteredForestOutput(
        candidate_ids=ids, unary_log_potentials=log_beliefs,
        candidate_state_mask=torch.isfinite(log_beliefs),
        retained_mass=beliefs[..., :-1].sum(-1), residual_log_mass=log_beliefs[..., -1],
        proposal_edge_index=edges[:, :0], proposal_edge_mask=edge_mask[:, :0],
        proposal_scores=log_beliefs.new_empty(batch_size, 0),
        anchor_logits=log_beliefs.new_empty(batch_size, length, 0),
        anchor_indices=edges.new_empty(batch_size, 0),
        slot_logits=log_beliefs.new_empty(batch_size, length, 0),
        edge_index=edges, edge_mask=edge_mask, edge_scores=log_beliefs.new_zeros(edge_mask.shape),
        pair_left_factors=left, pair_right_factors=right,
        topology_mode='fixed', factor_mode='dynamic', independent_mode=False,
        base_tail_log_mass=tail, scoring_active_mask=active.detach().clone(), vocab_size=self.vocab_size,
      )


def centered_forest_log_probability(output: CenteredForestOutput, raw_logits: torch.Tensor,
                                    targets: torch.Tensor,
                                    active_mask: torch.Tensor | None = None) -> torch.Tensor:
  """Exact active-token log score [B], with O(E*d) observed edge scoring.

  Top-K membership lookup additionally costs O(B*L*K). No partition function,
  messages, or E*K*K tensor is computed. Use the unchanged raw_logits supplied
  to forward. Inactive target values are ignored. Impossible active targets
  have score -inf and must not be used as finite training losses.
  """
  if not isinstance(output, CenteredForestOutput):
    raise ValueError('direct scoring requires an unmodified CenteredForestOutput')
  output._validate_logits(raw_logits)
  active = output.scoring_active_mask
  if active_mask is not None and (active_mask.device != active.device
      or active_mask.dtype != torch.bool or not torch.equal(active_mask, active)):
    raise ValueError('active_mask must equal the forward mask')
  if (targets.shape != active.shape or targets.dtype != torch.long
      or targets.device != active.device
      or bool(((targets < 0) | (targets >= output.vocab_size)).any())):
    raise ValueError('targets must be valid long [B,L] vocabulary IDs on device')
  with torch.autocast(device_type=active.device.type, enabled=False):
    safe_targets = torch.where(active, targets, output.candidate_ids[..., 0])
    matches = output.candidate_ids.eq(safe_targets[..., None])
    explicit = matches.any(-1)
    count = output.candidate_ids.shape[-1]
    states = torch.where(explicit, matches.long().argmax(-1),
                         torch.full_like(safe_targets, count))
    unary = output.unary_log_potentials.gather(-1, states[..., None]).squeeze(-1)
    unary = torch.where(active, unary, torch.zeros_like(unary)).sum(-1)
    target_logits = raw_logits.detach().double().gather(-1, safe_targets[..., None]).squeeze(-1)
    safe_tail = torch.where(torch.isfinite(output.base_tail_log_mass), output.base_tail_log_mass,
                            torch.zeros_like(output.base_tail_log_mass))
    tail_correction = torch.where(active & ~explicit, target_logits - safe_tail,
                                  torch.zeros_like(target_logits)).sum(-1)
    left_states = states.gather(1, output.edge_index[..., 0])
    right_states = states.gather(1, output.edge_index[..., 1])
    rank = output.pair_left_factors.shape[-1]
    left = output.pair_left_factors.gather(
      2, left_states.clamp_max(count - 1)[..., None, None].expand(-1, -1, 1, rank)).squeeze(2)
    right = output.pair_right_factors.gather(
      2, right_states.clamp_max(count - 1)[..., None, None].expand(-1, -1, 1, rank)).squeeze(2)
    edge_scores = torch.logsumexp(left.log() + right.log(), -1)
    interacting = output.edge_mask & (left_states < count) & (right_states < count)
    edge_scores = torch.where(interacting, edge_scores, torch.zeros_like(edge_scores)).sum(-1)
    return unary + tail_correction + edge_scores
