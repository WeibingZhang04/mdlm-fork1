"""Optional fused backward sampling for the existing candidate-state CRF.

The recurrence is unchanged: sample the final forward message, then each
previous message plus the edge column selected by the next state. Independent
Exp(1) variates implement categorical draws via argmax(logits - log(E)).
PyTorch generates and logs the noise in FP64 using the caller's generator;
Triton never substitutes its default FP32 RNG or approximate logarithm.

Noise is generated for the entire chain at once, so the random stream differs
from per-position torch.multinomial. The distribution is the same, but old
seeds do not promise identical token sequences. CPU/FP64/large-state cases
retain the reference sampler and its random stream.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _backward_sample_kernel(
        alpha, edge, log_noise, output, invalid,
        length, states: tl.constexpr,
        eb, el: tl.constexpr, ei: tl.constexpr, ej: tl.constexpr,
        block: tl.constexpr,
    ):
        batch = tl.program_id(0)
        indices = tl.arange(0, block)
        valid = indices < states
        selected = tl.full((), 0, tl.int32)
        bad = tl.full((), False, tl.int1)
        for offset in range(length):
            pos = length - 1 - offset
            address = (batch * length + pos) * states + indices
            logits = tl.load(alpha + address, valid, other=-float('inf'))
            if offset > 0:
                column = tl.load(edge + batch * eb + pos * el + indices * ei + selected * ej,
                                 valid, other=0.)
                # Match the reference FP32 addition before casting to FP64.
                logits = logits + column
            maximum = tl.max(logits, axis=0)
            bad = bad | (maximum == -float('inf')) | (tl.sum(
                (valid & ((logits != logits) | (logits == float('inf')))).to(tl.int32), axis=0) > 0)
            perturbation = tl.load(log_noise + address, valid, other=0.)
            scores = logits.to(tl.float64) - perturbation
            # Tie-breaking mirrors argmax's first index; padding cannot win
            # for valid distributions, even if only one state is reachable.
            selected = tl.argmax(scores, axis=0, tie_break_left=True).to(tl.int32)
            # Continue safely after invalid input, then report once on host.
            selected = tl.where(bad, 0, selected)
            tl.store(output + batch * length + pos, selected)
        tl.store(invalid + batch, bad.to(tl.int32))


def _sample_with_log_noise(alpha, edge, log_noise):
    """Internal launch separated from RNG for deterministic oracle tests.

    alpha and log_noise must be contiguous [B,L,S] FP32/FP64 tensors. Edge
    strides may vary. Callers have already checked the supported input path.
    """
    batch, length, states = alpha.shape
    output = torch.empty((batch, length), dtype=torch.long, device=alpha.device)
    invalid = torch.empty(batch, dtype=torch.int32, device=alpha.device)
    with torch.cuda.device(alpha.device):
        _backward_sample_kernel[(batch,)](
            alpha, edge, log_noise, output, invalid, length, states,
            *edge.stride(), triton.next_power_of_2(states), num_warps=4,
        )
    # One validation/synchronization per chain batch, not per token. Retain
    # the reference sampler's rejection of NaN/+inf/all-impossible draws.
    if invalid.any().item():
        raise RuntimeError('Invalid CRF sampling distribution: NaN, +inf, or no reachable state')
    return output


def fused_backward_sample(messages, edge, generator=None):
    """Return joint states [B,L], or None without consuming RNG on fallback."""
    first = messages[0]
    if (triton is None or torch.is_grad_enabled() or not first.is_cuda
            or torch.version.hip is not None or first.dtype != torch.float32
            or edge.dtype != torch.float32 or edge.device != first.device
            or not 1 <= first.shape[-1] <= 128 or first.shape[0] == 0
            or first.numel() * len(messages) > 8 * 1024 * 1024):
        return None
    # Bound extra FP64 noise storage to 64 MiB; larger workloads use the
    # reference path. Stack once rather than gather/index tensors per token.
    alpha = torch.stack(messages, dim=1)
    with torch.profiler.record_function('crf.backward_rng'):
        log_noise = torch.empty(alpha.shape, dtype=torch.float64, device=alpha.device)
        log_noise.exponential_(generator=generator).log_()
    with torch.profiler.record_function('crf.backward_fused'):
        return _sample_with_log_noise(alpha, edge, log_noise)
