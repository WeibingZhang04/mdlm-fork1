"""A forest head with separate token factors for the two ends of an edge.

The original head shares its token embedding across both endpoints.  Its
positive contextual scaling cannot reverse a token ordering in any factor
dimension.  With uniform unaries and vocabulary {A, B}, this implies
F(A,A) + F(B,B) >= F(A,B) + F(B,A).  In particular it cannot fit a balanced
AB/BA distribution.  Separate endpoint embeddings remove that restriction.

This class deliberately leaves the existing head and its checkpoints intact.
It returns the same output type, uses the same exact inference and samplers,
and adds one embedding and two contextual projections.  The parameters added
are V*R + (H+1)*2R + T*2R; this is not a capacity-matched comparison.
"""

import copy
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.structured_decoder import ContextualCouplingForestHead


class DirectionalCouplingForestHead(ContextualCouplingForestHead):
  """Use position-ordered left/right factors on canonical forest edges."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.right_token_factor_embedding = nn.Embedding(self.vocab_size, self.rank)
    nn.init.normal_(self.right_token_factor_embedding.weight,
                    mean=math.log(math.expm1(1.0)), std=0.01)
    self.right_factor_hidden_projection = copy.deepcopy(
      self.factor_hidden_projection)
    self.right_factor_time_projection = copy.deepcopy(
      self.factor_time_projection)

  def forward(self, hidden_states, unary_logits, timestep,
              active_mask=None, **kwargs):
    output = super().forward(
      hidden_states, unary_logits, timestep, active_mask, **kwargs)
    if output.independent_mode:
      return output
    raw = self.right_token_factor_embedding(output.candidate_ids)
    if output.factor_mode == 'dynamic':
      hidden = self.hidden_norm(hidden_states)
      time = self.time_embedding(timestep, hidden_states.shape[0])
      film = (self.right_factor_hidden_projection(hidden)
              + self.right_factor_time_projection(time)[:, None, :])
      shift, scale = film.chunk(2, dim=-1)
      raw = raw * (1 + scale.tanh()[:, :, None, :]) + shift[:, :, None, :]
    right = self._gather_candidate_factors(
      F.softplus(raw) + 1e-6, output.edge_index[:, :, 1]) / math.sqrt(self.rank)
    right = torch.where(output.edge_mask[:, :, None, None], right,
                        torch.full_like(right, 1 / math.sqrt(self.rank)))
    return dataclasses.replace(output, pair_right_factors=right)
