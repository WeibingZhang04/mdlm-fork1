"""Utilities for first-order CRF inference.

Provides the forward-backward algorithm for computing per-position
marginals of a first-order chain, used during CRF-MDLM sampling.
"""

import torch
import torch.nn.functional as F


def linear_warmup_weight(step, start_weight, end_weight,
                         warmup_steps):
  """Linearly interpolate an auxiliary-loss weight by optimizer step.

  The endpoint is reached at ``warmup_steps`` and held thereafter.  Keeping
  this helper independent of Lightning makes the scientific schedule easy to
  test and to reproduce outside the training loop.
  """
  if warmup_steps < 0:
    raise ValueError('warmup_steps must be non-negative')
  if step < 0:
    raise ValueError('step must be non-negative')
  if warmup_steps == 0:
    return float(end_weight)
  progress = min(float(step) / float(warmup_steps), 1.0)
  return float(start_weight + progress * (end_weight - start_weight))


def unigram_denoising_loss(logits, xt, x0, mask_index,
                           token_weight):
  """Continuous-time SUBS denoising loss for an auxiliary unigram head.

  Args:
    logits: (batch, length, vocab) unnormalised unigram predictions.
    xt: (batch, length) noisy tokens.
    x0: (batch, length) clean targets.
    mask_index: vocabulary index of the absorbing mask token.
    token_weight: scalar, (batch,), or (batch, 1) diffusion weighting.

  Returns:
    (batch, length) loss. Uncorrupted positions contribute exactly zero.
  """
  if logits.ndim != 3:
    raise ValueError('logits must have shape (batch, length, vocab)')
  if xt.shape != x0.shape or logits.shape[:2] != xt.shape:
    raise ValueError('logits, xt, and x0 shapes are inconsistent')

  # Do not mutate the head output: callers may also use it for diagnostics.
  valid_logits = logits.float().clone()
  valid_logits[..., mask_index] = -1e6
  log_probs = F.log_softmax(valid_logits, dim=-1)
  target_log_probs = torch.gather(
    log_probs, -1, x0.unsqueeze(-1)).squeeze(-1)

  if not torch.is_tensor(token_weight):
    token_weight = torch.as_tensor(
      token_weight, device=logits.device, dtype=log_probs.dtype)
  token_weight = token_weight.to(
    device=logits.device, dtype=log_probs.dtype)
  if token_weight.ndim == 1:
    token_weight = token_weight[:, None]

  masked = (xt == mask_index).to(log_probs.dtype)
  return -target_log_probs * masked * token_weight


def confident_reveal_mask(x, probabilities, mask_index,
                          move_chance_t, move_chance_s):
  """Choose the most-confident masked positions for one DDPM step.

  The number retained as masks is ``ceil(M_t * chance_s / chance_t)``
  for each example, where ``M_t`` is its current number of masks.  Thus the
  deterministic gate preserves the diffusion step's reveal count (up to the
  unavoidable integer rounding) while replacing random position selection.

  Returns:
    Boolean tensor shaped like ``x``; True positions should be revealed.
  """
  if probabilities.shape[:2] != x.shape:
    raise ValueError('probabilities and x shapes are inconsistent')
  if probabilities.ndim != 3:
    raise ValueError(
      'probabilities must have shape (batch, length, vocab)')

  batch = x.shape[0]
  chance_t = torch.as_tensor(
    move_chance_t, device=x.device, dtype=probabilities.dtype)
  chance_s = torch.as_tensor(
    move_chance_s, device=x.device, dtype=probabilities.dtype)
  if chance_t.numel() == 1:
    chance_t = chance_t.reshape(1).expand(batch)
  else:
    chance_t = chance_t.reshape(batch, -1)[:, 0]
  if chance_s.numel() == 1:
    chance_s = chance_s.reshape(1).expand(batch)
  else:
    chance_s = chance_s.reshape(batch, -1)[:, 0]

  ratio = torch.where(
    chance_t > 0,
    (chance_s / chance_t).clamp(min=0.0, max=1.0),
    torch.zeros_like(chance_t))
  is_masked = (x == mask_index)
  masked_counts = is_masked.sum(dim=-1)
  remaining_counts = torch.ceil(
    masked_counts.to(probabilities.dtype) * ratio - 1e-7).long()
  reveal_counts = masked_counts - remaining_counts

  token_probs = probabilities.clone()
  token_probs[..., mask_index] = 0
  confidence = token_probs.max(dim=-1).values
  confidence = confidence.masked_fill(~is_masked, -torch.inf)

  reveal = torch.zeros_like(is_masked)
  for batch_index in range(batch):
    count = int(reveal_counts[batch_index].item())
    if count > 0:
      positions = torch.topk(
        confidence[batch_index], k=count).indices
      reveal[batch_index, positions] = True
  return reveal


def factorized_confidence_gated_update(
    x, probabilities, mask_index, move_chance_t, move_chance_s):
  """Sample independent token identities at confidence-ranked positions.

  The clean-token identities remain stochastic independent draws from the
  supplied factorized distribution.  Only the reveal-position policy changes:
  :func:`confident_reveal_mask` chooses the highest-confidence masked sites at
  the absorbing schedule's rounded per-row reveal count.
  """
  if probabilities.shape[:2] != x.shape or probabilities.ndim != 3:
    raise ValueError('probabilities and x shapes are inconsistent')
  if not 0 <= int(mask_index) < probabilities.shape[-1]:
    raise ValueError('mask_index lies outside the probability vocabulary')
  reveal = confident_reveal_mask(
    x=x,
    probabilities=probabilities,
    mask_index=mask_index,
    move_chance_t=move_chance_t,
    move_chance_s=move_chance_s)
  token_probs = probabilities.clone()
  token_probs[..., mask_index] = 0
  if bool((token_probs.sum(dim=-1) <= 0).any().item()):
    raise ValueError('factorized distribution has no non-mask token mass')
  gumbel_norm = (
    1e-10 - (torch.rand_like(token_probs) + 1e-10).log())
  proposed_tokens = (token_probs / gumbel_norm).argmax(dim=-1)
  return torch.where(reveal, proposed_tokens, x)


def sequential_reveal_mask(x, probabilities, mask_index):
  """Select exactly one most-confident masked position per unfinished row."""
  if probabilities.shape[:2] != x.shape or probabilities.ndim != 3:
    raise ValueError('probabilities and x shapes are inconsistent')
  token_probs = probabilities.clone()
  token_probs[..., mask_index] = 0
  confidence = token_probs.max(dim=-1).values
  is_masked = (x == mask_index)
  confidence = confidence.masked_fill(~is_masked, -torch.inf)

  reveal = torch.zeros_like(is_masked)
  for batch_index in range(x.shape[0]):
    if is_masked[batch_index].any():
      position = confidence[batch_index].argmax()
      reveal[batch_index, position] = True
  return reveal


def constrain_chain_potentials(emission_0, transitions,
                               candidate_valid,
                               neg_infinity=-1e6):
  """Hard-constrain invalid candidate states before chain inference.

  ``candidate_valid[b, i, k]`` indicates whether candidate ``k`` may occupy
  position ``i``.  Both source and destination axes of each transition are
  constrained so an observed token cannot be bypassed by an alternate state.
  """
  if candidate_valid.ndim != 3:
    raise ValueError(
      'candidate_valid must have shape (batch, length, candidates)')
  batch, length, candidates = candidate_valid.shape
  if emission_0.shape != (batch, candidates):
    raise ValueError('emission_0 shape is inconsistent')
  if transitions.shape != (
      batch, length - 1, candidates, candidates):
    raise ValueError('transitions shape is inconsistent')
  if not candidate_valid.any(dim=-1).all():
    raise ValueError('every chain position needs at least one valid candidate')

  constrained_emission = emission_0.masked_fill(
    ~candidate_valid[:, 0, :], neg_infinity)
  valid_edges = (
    candidate_valid[:, :-1, :, None]
    & candidate_valid[:, 1:, None, :])
  constrained_transitions = transitions.masked_fill(
    ~valid_edges, neg_infinity)
  return constrained_emission, constrained_transitions


def chunked_normalized_gather(logits_fn, gather_indices,
                              query_chunk_size,
                              excluded_index=None,
                              neg_infinity=-1e6):
  """Gather exact full-softmax log probabilities in query chunks.

  ``logits_fn(start, end)`` must return ``(batch, end-start, vocab)``.
  Chunking only the query axis preserves the exact full-vocabulary
  normaliser while bounding the largest decoder output allocation.
  """
  if gather_indices.ndim != 3:
    raise ValueError(
      'gather_indices must have shape (batch, queries, candidates)')
  query_count = gather_indices.shape[1]
  if query_count == 0:
    return torch.empty_like(gather_indices, dtype=torch.float32)
  if query_chunk_size <= 0:
    query_chunk_size = query_count

  gathered_chunks = []
  for start in range(0, query_count, query_chunk_size):
    end = min(start + query_chunk_size, query_count)
    logits = logits_fn(start, end).float()
    if logits.shape[:2] != (gather_indices.shape[0], end - start):
      raise ValueError('logits_fn returned an inconsistent shape')
    if excluded_index is not None:
      logits[..., excluded_index] = neg_infinity
    log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
    selected_logits = torch.gather(
      logits, -1, gather_indices[:, start:end, :])
    gathered_chunks.append(selected_logits - log_normalizer)
  return torch.cat(gathered_chunks, dim=1)


def forward_backward(emission_0, transitions):
  """Forward-backward algorithm for first-order CRF marginals.

  Computes exact per-position marginals in log-space for
  numerical stability.

  Args:
    emission_0: (batch, K) log probs at position 0.
        emission_0[:, k] = log P(x_{0,0} = candidate k | START, x_t)
    transitions: (batch, N-1, K, K) log transition probs.
        transitions[:, j, u, v] =
          log P(pos j+1 = v-th candidate | pos j = u-th candidate)

  Returns:
    marginals: (batch, N, K) per-position marginal probabilities
        normalised over the K candidates at each position.
  """
  batch, N_minus_1, K, _ = transitions.shape
  N = N_minus_1 + 1
  device = emission_0.device
  dtype = emission_0.dtype

  # --- Forward pass ---
  # alpha[i, k] = log sum_{x_{0:i-1}} P(x_{0:i}, x_{0,i}=k | x_t)
  log_alphas = []
  log_alpha = emission_0                           # (batch, K)
  log_alphas.append(log_alpha)

  for j in range(N_minus_1):
    # log_alpha:              (batch, K_prev)
    # transitions[:, j]:      (batch, K_prev, K_curr)
    # result:                 (batch, K_curr)
    T = transitions[:, j]
    log_alpha = torch.logsumexp(
      log_alpha.unsqueeze(-1) + T, dim=1)
    log_alphas.append(log_alpha)

  log_alphas = torch.stack(log_alphas, dim=1)       # (batch, N, K)

  # --- Backward pass ---
  # beta[i, k] = log sum_{x_{i+1:N-1}} P(x_{i+1:N-1} | x_{0,i}=k)
  log_betas_rev = []
  log_beta = torch.zeros(
    batch, K, device=device, dtype=dtype)
  log_betas_rev.append(log_beta)

  for j in range(N_minus_1 - 1, -1, -1):
    T = transitions[:, j]                           # (batch, K_prev, K_curr)
    # For each u (K_prev), sum over v (K_curr):
    #   T[u, v] + beta[v]
    log_beta = torch.logsumexp(
      T + log_beta.unsqueeze(1), dim=-1)            # (batch, K_prev)
    log_betas_rev.append(log_beta)

  log_betas_rev.reverse()
  log_betas = torch.stack(log_betas_rev, dim=1)     # (batch, N, K)

  # --- Marginals ---
  # P(x_{0,i} = k) proportional to alpha[i,k] * beta[i,k]
  log_marginals = log_alphas + log_betas
  marginals = F.softmax(log_marginals, dim=-1)

  return marginals
