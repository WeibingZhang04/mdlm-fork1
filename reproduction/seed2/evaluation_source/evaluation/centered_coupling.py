"""A bounded positive coupling with analytically fixed singleton marginals.

The final state is residual. Explicit features are centered under the
conditional explicit-state marginal; residual features are zero. For bounded
centered features f,g and 0 < eta < 1, the edge density ratio is

    Q(a,b) / (b_left(a) b_right(b)) = 1 + eta * mean_r f_r(a) g_r(b).

Thus Q has the supplied marginals without iterative balancing. The ratio is
between 1-eta and 1+eta: positive PMI is bounded by log(1+eta). This is not a
universal family of couplings. It cannot change dependence involving residual
states or the within-residual token distribution.

This standalone mathematical primitive does not change any production head.
Its full endpoint factors include the residual state, whose *dot products*
with every opposing state are one. Slice ``[..., :-1, :]`` for the existing
forest routines: those accept only explicit factors and insert neutral
residual interactions themselves. All explicit factors are strictly positive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _center_features(probabilities: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
  with torch.autocast(device_type=probabilities.device.type, enabled=False):
    explicit = probabilities[..., :-1]
    mass = explicit.sum(-1, keepdim=True)
    # Dividing by one on empty support keeps both forward and backward finite.
    conditional = explicit / torch.where(mass > 0, mass, torch.ones_like(mass))
    # tanh rounds to +/-1 at large finite inputs. This machine-precision margin
    # retains strictly positive factors without clipping centered features and
    # thereby disturbing their weighted mean. It is not a statistical shrinkage.
    margin = 8 * torch.finfo(raw.dtype).eps
    z = torch.tanh(raw) * (1.0 - margin)
    mean = (conditional[..., None] * z).sum(-2, keepdim=True)
    centered = (z - mean) / (1.0 + mean.abs())
    centered = torch.where(mass[..., None] > 0, centered, torch.zeros_like(centered))
    return torch.cat((centered, torch.zeros_like(centered[..., :1, :])), dim=-2)


@dataclass(frozen=True)
class CenteredCoupling:
  """One edge or a batch of edges, with S=K+1 states and rank 2*d+1."""

  left_marginal: torch.Tensor             # [..., S]
  right_marginal: torch.Tensor            # [..., S]
  left_features: torch.Tensor             # [..., S, d], residual row zero
  right_features: torch.Tensor            # [..., S, d], residual row zero
  left_factors: torch.Tensor              # [..., S, 2*d+1]
  right_factors: torch.Tensor             # [..., S, 2*d+1]
  eta: torch.Tensor                       # [...] (one scalar per edge)

  def density_ratio(self) -> torch.Tensor:
    """Materialize the S-by-S ratio, for bounded diagnostics and tests."""
    with torch.autocast(device_type=self.left_factors.device.type, enabled=False):
      return self.left_factors @ self.right_factors.transpose(-1, -2)

  def joint(self) -> torch.Tensor:
    """Materialize the joint; training should use observed log probabilities."""
    with torch.autocast(device_type=self.left_factors.device.type, enabled=False):
      return self.left_marginal[..., :, None] * self.right_marginal[..., None, :] * self.density_ratio()

  def log_probability(self, left_states: torch.Tensor, right_states: torch.Tensor) -> torch.Tensor:
    """Score one pair per edge in O(d), without materializing S-by-S arrays.

    States have the batch shape of the marginal tensors (without S). A pair
    outside marginal support has log probability -infinity. Such impossible
    targets must not be used as finite training losses.
    """
    shape = self.left_marginal.shape[:-1]
    for states in (left_states, right_states):
      if (not torch.is_tensor(states) or states.dtype != torch.long or states.shape != shape
          or states.device != self.left_marginal.device
          or bool(((states < 0) | (states >= self.left_marginal.shape[-1])).any())):
        raise ValueError('state indices must be long tensors with the edge batch shape on device')
    with torch.autocast(device_type=self.left_factors.device.type, enabled=False):
      rank = self.left_factors.shape[-1]
      left = self.left_factors.gather(-2, left_states[..., None, None].expand(*shape, 1, rank)).squeeze(-2)
      right = self.right_factors.gather(-2, right_states[..., None, None].expand(*shape, 1, rank)).squeeze(-2)
      left_mass = self.left_marginal.gather(-1, left_states[..., None]).squeeze(-1)
      right_mass = self.right_marginal.gather(-1, right_states[..., None]).squeeze(-1)
      return left_mass.log() + right_mass.log() + torch.logsumexp(left.log() + right.log(), -1)

  @torch.no_grad()
  def sample(self, num_samples: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw left marginal then right conditional, returning [..., n, 2]."""
    if type(num_samples) is not int or num_samples < 1:
      raise ValueError('num_samples must be a positive integer')
    with torch.autocast(device_type=self.left_factors.device.type, enabled=False):
      shape = self.left_marginal.shape[:-1]
      count = math.prod(shape)
      states, rank = self.left_marginal.shape[-1], self.left_factors.shape[-1]
      left_marginal = self.left_marginal.reshape(count, states)
      right_marginal = self.right_marginal.reshape(count, states)
      left = torch.multinomial(left_marginal, num_samples, replacement=True, generator=generator)
      left_factors = self.left_factors.reshape(count, states, rank)
      selected = left_factors.gather(1, left[..., None].expand(count, num_samples, rank))
      right_factors = self.right_factors.reshape(count, states, rank)
      conditional = torch.einsum('bnr,bsr->bns', selected, right_factors) * right_marginal[:, None, :]
      right = torch.multinomial(conditional.reshape(-1, states), 1,
                                replacement=True, generator=generator).reshape(count, num_samples)
      return torch.stack((left, right), -1).reshape(*shape, num_samples, 2)


def centered_coupling(left_marginal: torch.Tensor, right_marginal: torch.Tensor,
                      left_raw: torch.Tensor, right_raw: torch.Tensor,
                      eta: float | torch.Tensor = 0.95) -> CenteredCoupling:
  """Construct normalized edge beliefs with fixed supplied marginals.

  Marginals have shape [..., K+1] and sum to one. Raw endpoint features have
  shape [..., K, d]. The final marginal state is residual, including when its
  probability is zero. Zero explicit mass is allowed and gives neutral
  features. Inputs must share a device. Marginals must be FP32 or FP64;
  lower-precision raw features are promoted to FP32. FP64 marginals or raw
  features produce FP64 factors. Marginals must sum to one in that working
  precision: FP32 probability vectors can fail this requirement when promoted
  alongside FP64 features. Recompute normalized probabilities in the intended
  precision before calling. Marginals are never detached or silently
  renormalized, and the centered mean remains in the gradient graph. Ambient
  autocast is disabled for the probability calculations.

  With z=tanh(raw), m=E[z | explicit], use f=(z-m)/(1+abs(m)). Unlike dividing
  by two, this lets balanced binary features approach +/-1 and strong AB/BA
  dependence. A tiny dtype-relative scaling of z prevents tanh saturation
  from generating zero endpoint factors in floating-point arithmetic.
  """
  inputs = (left_marginal, right_marginal, left_raw, right_raw)
  if any(not torch.is_tensor(value) or not value.is_floating_point() for value in inputs):
    raise ValueError('marginals and raw features must be floating-point tensors')
  if any(value.dtype not in (torch.float32, torch.float64) for value in (left_marginal, right_marginal)):
    raise ValueError('use FP32 or FP64 normalized marginals, not low-precision probabilities')
  if len({value.device for value in inputs}) != 1:
    raise ValueError('marginals and features must share a device')
  if (left_marginal.ndim < 1 or left_marginal.shape != right_marginal.shape
      or left_marginal.shape[-1] < 2 or left_raw.shape != right_raw.shape
      or left_raw.ndim != left_marginal.ndim + 1
      or left_raw.shape[:-2] != left_marginal.shape[:-1]
      or left_raw.shape[-2] != left_marginal.shape[-1] - 1 or left_raw.shape[-1] < 1):
    raise ValueError('expected matching [..., K+1] marginals and [..., K, d] features')
  if any(not bool(torch.isfinite(value).all()) for value in inputs):
    raise ValueError('marginals and features must be finite')
  dtype = torch.float64 if any(value.dtype == torch.float64 for value in inputs) else torch.float32
  left_marginal, right_marginal, left_raw, right_raw = (value.to(dtype) for value in inputs)
  # Promoting rounded FP32 masses to FP64 does not recover FP64 normalization.
  # Validate the actual arithmetic inputs rather than their original dtype.
  with torch.autocast(device_type=left_marginal.device.type, enabled=False):
    for marginal in (left_marginal, right_marginal):
      tolerance = 16 * torch.finfo(dtype).eps
      if (bool((marginal < 0).any()) or not torch.allclose(
          marginal.sum(-1), torch.ones_like(marginal[..., 0]), atol=tolerance, rtol=tolerance)):
        raise ValueError('marginals must be nonnegative and normalized in working precision')
  strength = torch.as_tensor(eta, device=left_marginal.device, dtype=dtype)
  try:
    strength = torch.broadcast_to(strength, left_marginal.shape[:-1])
  except RuntimeError as error:
    raise ValueError('eta must broadcast to the edge batch shape') from error
  if not bool((torch.isfinite(strength) & (strength > 0) & (strength < 1)).all()):
    raise ValueError('eta must lie strictly between zero and one')
  f, g = _center_features(left_marginal, left_raw), _center_features(right_marginal, right_raw)
  with torch.autocast(device_type=left_marginal.device.type, enabled=False):
    ones = torch.ones_like(f[..., :1])
    scale = strength[..., None, None] / (2 * f.shape[-1])
    left = torch.cat((ones, 1 + f, 1 - f), -1)
    right = torch.cat(((1 - strength[..., None, None]) * ones,
                        scale * (1 + g), scale * (1 - g)), -1)
  if not bool(((left > 0) & torch.isfinite(left)).all() & ((right > 0) & torch.isfinite(right)).all()):
    raise ValueError('eta/rank combination underflows strictly positive factors in this dtype')
  return CenteredCoupling(left_marginal, right_marginal, f, g, left, right, strength)
