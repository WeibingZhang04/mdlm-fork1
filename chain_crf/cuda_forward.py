"""Optional inference-only CUDA forward filtering for small dense chains.

One GPU program owns one sequence and advances through its edges on device.
This removes Python/kernel-launch overhead without changing the CRF factors.
Inputs use finite or -inf log scores, as in the reference implementation.
Training and unsupported devices/dtypes/state sizes use core._forward instead.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _forward_kernel(
        unary, edge, output,
        length, states: tl.constexpr,
        ub, ul: tl.constexpr, us: tl.constexpr,
        eb, el: tl.constexpr, ei: tl.constexpr, ej: tl.constexpr,
        block: tl.constexpr,
    ):
        batch = tl.program_id(0)
        indices = tl.arange(0, block)
        valid = indices < states
        alpha = tl.load(unary + batch * ub + indices * us, valid, other=-float('inf'))
        tl.store(output + batch * length * states + indices, alpha, valid)
        pair_mask = valid[:, None] & valid[None, :]
        pair_offsets = indices[:, None] * ei + indices[None, :] * ej
        for pos in range(1, length):
            scores = tl.load(edge + batch * eb + (pos - 1) * el + pair_offsets,
                             pair_mask, other=-float('inf')) + alpha[:, None]
            maximum = tl.max(scores, axis=0)
            # For an unreachable destination all scores are -inf. Using zero
            # as the shift avoids -inf - -inf; log(0) then returns -inf.
            shift = tl.where(maximum == -float('inf'), 0., maximum)
            message = tl.log(tl.sum(tl.exp(scores - shift[None, :]), axis=0)) + shift
            current = tl.load(unary + batch * ub + pos * ul + indices * us,
                              valid, other=-float('inf'))
            alpha = current + message
            tl.store(output + (batch * length + pos) * states + indices, alpha, valid)


def fused_forward(unary, edge):
    """Return [B,L,S] messages, or None to request reference inference.

    FP32 only: do not silently downcast FP64 inference. This path is deliberately
    disabled with autograd enabled, even when the inputs do not require grads.
    Triton compiles on first use; exclude that call from steady-state timings.
    """
    if (triton is None or torch.is_grad_enabled() or not unary.is_cuda
            or torch.version.hip is not None or unary.dtype != torch.float32
            or edge.dtype != torch.float32 or edge.device != unary.device
            or not 1 <= unary.shape[-1] <= 128 or unary.shape[0] == 0
            or unary.shape[1] <= 1):
        return None
    batch, length, states = unary.shape
    output = torch.empty((batch, length, states), dtype=unary.dtype, device=unary.device)
    with torch.cuda.device(unary.device):
        _forward_kernel[(batch,)](
            unary, edge, output, length, states, *unary.stride(), *edge.stride(),
            triton.next_power_of_2(states), num_warps=4,
        )
    return output
