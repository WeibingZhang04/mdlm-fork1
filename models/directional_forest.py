"""A forest head with separate token factors for the two ends of an edge.

The original head shares its token embedding across both endpoints.  Its
positive contextual scaling cannot reverse a token ordering in any factor
dimension.  With uniform unaries and vocabulary {A, B}, this implies
F(A,A) + F(B,B) >= F(A,B) + F(B,A).  In particular it cannot fit a balanced
AB/BA distribution.  Separate endpoint embeddings remove that restriction.

This backward-compatible name now selects the normal head's ``separate``
option. Existing directional checkpoints retain their parameter names.
At equal rank, the parameters added are V*R + (H+1)*2R + T*2R; halve the rank
to match the shared head's parameter count.
"""

from models.structured_decoder import ContextualCouplingForestHead


class DirectionalCouplingForestHead(ContextualCouplingForestHead):
  """Use position-ordered left/right factors on canonical forest edges."""

  def __init__(self, *args, **kwargs):
    mode = kwargs.pop('factor_embedding_mode', 'separate')
    if mode != 'separate':
      raise ValueError('DirectionalCouplingForestHead requires separate factors')
    super().__init__(*args, factor_embedding_mode=mode, **kwargs)
