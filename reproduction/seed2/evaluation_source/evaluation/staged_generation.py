"""Use an authenticated staged output head for fixed-group generation.

Checkpoint authentication and device placement belong to the caller. This
module neither loads a head nor uses the production model's attached head.
It consumes only the current corrupted tokens, sigma, and mask, never clean
targets. Every backbone example is encoded serially, as in staged training.

All arms select candidates from the same raw-logit top-K rule. That does not
imply identical candidate IDs after their generated contexts diverge. The
returned logits are the original backbone logits, including its absorbing
mask exclusion: residual draws must not use unary-adapter-corrected logits.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from models.centered_forest import FrozenUnaryCenteredForestHead
from models.contextual_unary import ContextualUnaryAdapter
from models.structured_decoder import (
  ContextualCouplingForestHead, StructuredDecoderOutput,
)


@contextmanager
def _temporary_eval(*roots):
  """Disable training behavior and restore even mixed child-module modes.

  No train hooks, parameter freezing, dtype casts, or RNG resets are needed.
  As with ordinary module.eval(), these modules must not concurrently train
  in another thread while a callback is using them.
  """
  modes = {}
  for root in roots:
    for module in root.modules():
      modes.setdefault(module, module.training)
  try:
    for module in modes:
      module.training = False
    yield
  finally:
    for module, training in modes.items():
      module.training = training


def _module_device(module, name, fallback=None):
  devices = {value.device for value in (*module.parameters(), *module.buffers())}
  if len(devices) > 1:
    raise ValueError(f'{name} must already be on one device')
  if not devices:
    if fallback is None:
      raise ValueError(f'{name} must have a parameter or buffer identifying its device')
    return fallback
  return next(iter(devices))


def _unary_output(head, hidden, logits, sigma, active):
  """Expose a candidate-only unary adapter as an edgeless forest."""
  lattice = head.candidate_lattice(hidden, logits, sigma, active)
  batch, length, candidates = lattice.candidate_ids.shape
  edge_index = torch.empty(batch, 0, 2, dtype=torch.long, device=logits.device)
  edge_mask = torch.empty(batch, 0, dtype=torch.bool, device=logits.device)
  edge_scores = logits.new_empty(batch, 0)
  empty_factors = logits.new_empty(batch, 0, candidates, 1)
  # Match the pair head's coverage diagnostic: mass retained under the
  # BACKBONE, not mass after the contextual unary correction.
  kept = logits.gather(-1, lattice.candidate_ids).logsumexp(-1)
  retained = (kept - logits.logsumexp(-1)).exp().clamp(0, 1)
  return StructuredDecoderOutput(
    candidate_ids=lattice.candidate_ids,
    unary_log_potentials=lattice.unary_log_potentials,
    candidate_state_mask=lattice.candidate_state_mask,
    retained_mass=retained,
    residual_log_mass=lattice.residual_log_mass,
    proposal_edge_index=edge_index, proposal_edge_mask=edge_mask,
    proposal_scores=edge_scores,
    anchor_logits=logits.new_empty(batch, length, 0),
    anchor_indices=torch.empty(batch, 0, dtype=torch.long, device=logits.device),
    slot_logits=logits.new_empty(batch, length, 0),
    edge_index=edge_index, edge_mask=edge_mask, edge_scores=edge_scores,
    pair_left_factors=empty_factors, pair_right_factors=empty_factors,
    topology_mode='fixed', factor_mode='dynamic', independent_mode=True)


class StagedGenerationAdapter:
  """Callable bridge for ``evaluation.fixed_group_sampling.generate_fixed_groups``.

  ``adapter(tokens[B,L], sigma[B], active_mask[B,L])`` returns a structured
  output and raw FP32 backbone logits. Shared and separate-endpoint forest
  heads retain their configured topology/factor modes. A candidate-only unary
  head returns an edgeless structured output with the same residual decoder.
  The experimental centered head retains its FP64 normalized beliefs, factors,
  and specialized residual decoder; its raw backbone logits still stay FP32.
  Inference handles inactive-node cancellation; the generation driver keeps
  observed tokens fixed. This adapter never replaces or reveals input tokens.

  ``logical_batch_calls`` counts callbacks passing input validation.
  ``physical_encoder_calls`` counts actual serial backbone invocations,
  including an invocation that raises. Neither counter includes a sampling
  operation. Completed rows omitted by the driver incur no encoder call.

  The preloaded model and selected head remain caller-owned. Parameters,
  gradients, dtype/device placement, and RNG states are not modified. Training
  flags are temporarily disabled during evaluation and restored on exit.
  """

  def __init__(self, model: nn.Module, head: nn.Module, *, mask_index=None):
    if not isinstance(model, nn.Module) or not callable(
        getattr(model, '_structured_backbone_output', None)):
      raise TypeError('model must expose the production structured backbone path')
    if not isinstance(head, (ContextualCouplingForestHead, ContextualUnaryAdapter,
                             FrozenUnaryCenteredForestHead)):
      raise TypeError('head must be a selected staged forest or unary adapter')
    if isinstance(head, ContextualUnaryAdapter) and head.correction_domain != 'candidates':
      raise ValueError('staged generation requires a candidate-only unary adapter')
    model_mask = getattr(model, 'mask_index', None)
    if type(model_mask) is not int or model_mask < 0:
      raise ValueError('model must define a nonnegative integer mask_index')
    if mask_index is None:
      mask_index = model_mask
    if type(mask_index) is not int or mask_index != model_mask:
      raise ValueError('mask_index must agree with the preloaded backbone')
    self.model = model
    self.head = head
    self.mask_index = mask_index
    self.logical_batch_calls = 0
    self.physical_encoder_calls = 0
    self._validate_modules()

  def _validate_modules(self):
    if getattr(self.model, 'mask_index', None) != self.mask_index:
      raise ValueError('backbone mask_index changed after adapter construction')
    head_device = _module_device(self.head, 'head')
    model_device = _module_device(self.model, 'model', fallback=head_device)
    if model_device != head_device:
      raise ValueError('backbone and head must already be on the same device')
    for value in (*self.head.parameters(), *self.head.buffers()):
      if (value.is_floating_point() or value.is_complex()) and value.dtype != torch.float32:
        raise ValueError('selected head must already be FP32; adapter never casts parameters')
    return head_device

  def _validate_inputs(self, tokens, sigma, active_mask):
    device = self._validate_modules()
    if (not torch.is_tensor(tokens) or tokens.ndim != 2
        or tokens.dtype != torch.long or min(tokens.shape) < 1):
      raise ValueError('tokens must be a nonempty long [B,L] tensor')
    if (not torch.is_tensor(sigma) or sigma.shape != (tokens.shape[0],)
        or not sigma.is_floating_point()):
      raise ValueError('sigma must be a floating [B] tensor')
    if (not torch.is_tensor(active_mask) or active_mask.shape != tokens.shape
        or active_mask.dtype != torch.bool):
      raise ValueError('active_mask must be a boolean [B,L] tensor')
    if any(value.device != device for value in (tokens, sigma, active_mask)):
      raise ValueError('tokens, sigma, and active_mask must match the module device')
    if not bool(torch.isfinite(sigma.float()).all()) or bool((sigma < 0).any()):
      raise ValueError('sigma must be finite in FP32 and nonnegative')
    if not torch.equal(active_mask, tokens.eq(self.mask_index)):
      raise ValueError('active_mask must identify exactly the absorbing mask tokens')
    clean_or_mask = ((tokens >= 0) & (tokens < self.head.vocab_size)) | tokens.eq(self.mask_index)
    if not bool(clean_or_mask.all()):
      raise ValueError('tokens contain an invalid vocabulary ID')

  @torch.no_grad()
  def __call__(self, tokens, sigma, active_mask):
    self._validate_inputs(tokens, sigma, active_mask)
    self.logical_batch_calls += 1
    hidden_rows, logit_rows = [], []
    with _temporary_eval(self.model, self.head):
      # Disable caller autocast; the production backbone owns its internal
      # BF16 path, and the selected head must run in its trained FP32 dtype.
      with torch.autocast(device_type=tokens.device.type, enabled=False):
        conditioning = sigma.detach().float()
        for index in range(tokens.shape[0]):
          self.physical_encoder_calls += 1
          hidden, logits = self.model._structured_backbone_output(
            tokens[index:index + 1], conditioning[index:index + 1, None],
            force_no_grad=True)
          if (not torch.is_tensor(hidden) or not torch.is_tensor(logits)
              or hidden.shape != (1, tokens.shape[1], self.head.hidden_size)
              or logits.shape != (1, tokens.shape[1], self.head.vocab_size)
              or hidden.device != tokens.device or logits.device != tokens.device
              or not hidden.is_floating_point() or not logits.is_floating_point()):
            raise ValueError('backbone must return hidden [1,L,H] and logits [1,L,V] on device')
          hidden, logits = hidden.detach().float(), logits.detach().float()
          if not bool(torch.isfinite(hidden).all()):
            raise ValueError('backbone hidden states must be finite')
          if bool((torch.isnan(logits) | torch.isposinf(logits)).any()):
            raise ValueError('backbone logits cannot contain NaN or +inf')
          if not bool(torch.isfinite(logits).any(-1).all()):
            raise ValueError('every backbone position must have clean-token support')
          if self.mask_index < self.head.vocab_size and not bool(
              torch.isneginf(logits[..., self.mask_index]).all()):
            raise ValueError('backbone logits must already exclude the absorbing mask')
          hidden_rows.append(hidden)
          logit_rows.append(logits)
        hidden = torch.cat(hidden_rows)
        raw_logits = torch.cat(logit_rows)
        if isinstance(self.head, ContextualUnaryAdapter):
          output = _unary_output(self.head, hidden, raw_logits, conditioning, active_mask)
        else:
          output = self.head(hidden, raw_logits, conditioning, active_mask)
    return output, raw_logits
