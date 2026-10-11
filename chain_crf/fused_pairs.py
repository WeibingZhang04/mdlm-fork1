"""Inference-only candidate CRFs without an [B,L,K,K] edge tensor.

Pair scores are computed inside tiled forward/backward message kernels and
inside backward sampling. This keeps the original signed log potentials (not
a new low-rank probability model). Work is still quadratic in candidate count.
Visible boundaries are absorbed into endpoint unaries; original adjacency and
the neutral residual state are preserved. Training uses the existing backend.

The FP32 recurrence uses IEEE dot products. Categorical perturbations remain
FP64. Reduction order can change seeded draws; the model law is unchanged up
to floating-point rounding. First-use compilation is not steady-state runtime.
"""
from dataclasses import dataclass
import weakref

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


_COUNT_CACHE = weakref.WeakKeyDictionary()


if triton is not None:
    @triton.jit
    def _count_scores(a, b, keys, values, left_bg, right_bg,
                      vocab: tl.constexpr, entries: tl.constexpr,
                      search_steps: tl.constexpr):
        valid = (a >= 0) & (b >= 0)
        wanted = a.to(tl.int64) * vocab + b
        lo = tl.full(wanted.shape, 0, tl.int64)
        hi = tl.full(wanted.shape, entries, tl.int64)
        for _ in range(search_steps):
            mid = (lo + hi) // 2
            key = tl.load(keys + mid, mid < entries, other=9223372036854775807)
            go_right = key < wanted
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        found = (lo < entries) & (tl.load(
            keys + lo, lo < entries, other=-2) == wanted)
        bg = tl.load(left_bg + a, a >= 0, other=0.) + tl.load(right_bg + b, b >= 0, other=0.)
        score = tl.load(values + lo, found, other=0.)
        return tl.where(valid, tl.where(found, score, bg).to(tl.float32), 0.)


    @triton.jit
    def _pair_tile(left, right, ids, keys, values, left_bg, right_bg,
                   original_row, edge_pos, a, b,
                   original_length: tl.constexpr, states: tl.constexpr,
                   rank: tl.constexpr, rank_block: tl.constexpr,
                   kind: tl.constexpr, vocab: tl.constexpr,
                   entries: tl.constexpr, search_steps: tl.constexpr):
        if kind == 0:
            r = tl.arange(0, rank_block)
            x = tl.load(left + ((original_row * original_length + edge_pos) * states
                                + a[:, None]) * rank + r[None, :],
                        (a[:, None] < states) & (r[None, :] < rank), other=0.)
            y = tl.load(right + ((original_row * original_length + edge_pos + 1) * states
                                 + b[None, :]) * rank + r[:, None],
                        (b[None, :] < states) & (r[:, None] < rank), other=0.)
            return tl.dot(x, y, input_precision='ieee')
        else:
            ai = tl.load(ids + (original_row * original_length + edge_pos) * states + a,
                         a < states, other=-1)
            bi = tl.load(ids + (original_row * original_length + edge_pos + 1) * states + b,
                         b < states, other=-1)
            return _count_scores(ai[:, None], bi[None, :], keys, values, left_bg, right_bg,
                                 vocab, entries, search_steps)


    @triton.jit
    def _message_kernel(unary, messages, left, right, ids, keys, values, left_bg, right_bg,
                        rows, starts, lengths, pos, width,
                        original_length: tl.constexpr, states: tl.constexpr,
                        rank: tl.constexpr, rank_block: tl.constexpr, kind: tl.constexpr,
                        vocab: tl.constexpr, entries: tl.constexpr, search_steps: tl.constexpr,
                        state_block: tl.constexpr, tile: tl.constexpr, backward: tl.constexpr):
        run = tl.program_id(0).to(tl.int64)
        selected = tl.program_id(1) * tile + tl.arange(0, tile)
        all_states = tl.arange(0, state_block)
        run_length = tl.load(lengths + run)
        row = tl.load(rows + run).to(tl.int64)
        start = tl.load(starts + run).to(tl.int64)
        if backward:
            if pos < run_length - 1:
                running_max = tl.full((tile,), -float('inf'), tl.float32)
                running_sum = tl.full((tile,), 0., tl.float32)
                for chunk in range((states + 127) // 128):
                    reduction_states = chunk * 128 + tl.arange(0, 128)
                    scores = _pair_tile(left, right, ids, keys, values, left_bg, right_bg,
                                        row, start + pos, selected, reduction_states,
                                        original_length, states, rank, rank_block, kind,
                                        vocab, entries, search_steps)
                    nxt = (run * width + pos + 1) * states + reduction_states
                    incoming = tl.load(messages + nxt, reduction_states < states, other=-float('inf'))
                    incoming += tl.load(unary + nxt, reduction_states < states, other=-float('inf'))
                    scores += incoming[None, :]
                    scores = tl.where(reduction_states[None, :] < states, scores, -float('inf'))
                    block_max = tl.max(scores, axis=1)
                    next_max = tl.maximum(running_max, block_max)
                    shift = tl.where(next_max == -float('inf'), 0., next_max)
                    running_sum = running_sum * tl.exp(running_max - shift) + tl.sum(
                        tl.exp(scores - shift[:, None]), axis=1)
                    running_max = next_max
                result = tl.log(running_sum) + running_max
            else:
                result = tl.full((tile,), 0., tl.float32)
        else:
            if pos < run_length:
                running_max = tl.full((tile,), -float('inf'), tl.float32)
                running_sum = tl.full((tile,), 0., tl.float32)
                for chunk in range((states + 127) // 128):
                    reduction_states = chunk * 128 + tl.arange(0, 128)
                    scores = _pair_tile(left, right, ids, keys, values, left_bg, right_bg,
                                        row, start + pos - 1, reduction_states, selected,
                                        original_length, states, rank, rank_block, kind,
                                        vocab, entries, search_steps)
                    incoming = tl.load(messages + (run * width + pos - 1) * states + reduction_states,
                                       reduction_states < states, other=-float('inf'))
                    scores += incoming[:, None]
                    scores = tl.where(reduction_states[:, None] < states, scores, -float('inf'))
                    block_max = tl.max(scores, axis=0)
                    next_max = tl.maximum(running_max, block_max)
                    shift = tl.where(next_max == -float('inf'), 0., next_max)
                    running_sum = running_sum * tl.exp(running_max - shift) + tl.sum(
                        tl.exp(scores - shift[None, :]), axis=0)
                    running_max = next_max
                result = tl.log(running_sum) + running_max
                result += tl.load(unary + (run * width + pos) * states + selected,
                                  selected < states, other=-float('inf'))
            else:
                # Neutral edges and point-mass padding reproduce packed chains.
                previous = tl.load(messages + (run * width + pos - 1) * states + all_states,
                                   all_states < states, other=-float('inf'))
                padding_max = tl.max(previous, axis=0)
                padding_shift = tl.where(padding_max == -float('inf'), 0., padding_max)
                total = tl.log(tl.sum(tl.exp(previous - padding_shift), axis=0)) + padding_shift
                result = tl.where(selected == 0, total, -float('inf'))
        tl.store(messages + (run * width + pos) * states + selected, result, selected < states)


    @triton.jit
    def _sample_kernel(alpha, noise, output, invalid, left, right, ids, keys, values,
                       left_bg, right_bg, rows, starts, lengths, width,
                       original_length: tl.constexpr, states: tl.constexpr,
                       rank: tl.constexpr, rank_block: tl.constexpr, kind: tl.constexpr,
                       vocab: tl.constexpr, entries: tl.constexpr, search_steps: tl.constexpr,
                       state_block: tl.constexpr):
        run = tl.program_id(0).to(tl.int64)
        a = tl.arange(0, state_block)
        row = tl.load(rows + run).to(tl.int64)
        start = tl.load(starts + run).to(tl.int64)
        run_length = tl.load(lengths + run)
        selected = tl.full((), 0, tl.int32)
        bad = tl.full((), False, tl.int1)
        for offset in range(width):
            pos = (width - 1 - offset).to(tl.int64)
            address = (run * width + pos) * states + a
            logits = tl.load(alpha + address, a < states, other=-float('inf'))
            if pos < run_length - 1:
                if kind == 0:
                    r = tl.arange(0, rank_block)
                    x = tl.load(left + ((row * original_length + start + pos) * states
                                        + a[:, None]) * rank + r[None, :],
                                (a[:, None] < states) & (r[None, :] < rank), other=0.)
                    y = tl.load(right + ((row * original_length + start + pos + 1) * states
                                         + selected) * rank + r,
                                r < rank, other=0.)
                    column = tl.sum(x * y[None, :], axis=1)
                else:
                    ai = tl.load(ids + (row * original_length + start + pos) * states + a,
                                 a < states, other=-1)
                    bi = tl.load(ids + (row * original_length + start + pos + 1) * states + selected)
                    column = _count_scores(ai, bi, keys, values, left_bg, right_bg,
                                           vocab, entries, search_steps)
                logits += column
            maximum = tl.max(logits, axis=0)
            bad = bad | (maximum == -float('inf')) | (tl.sum(
                ((a < states) & ((logits != logits) | (logits == float('inf')))).to(tl.int32), 0) > 0)
            perturbation = tl.load(noise + address, a < states, other=0.)
            scores = logits.to(tl.float64) - perturbation
            selected = tl.argmax(scores, axis=0, tie_break_left=True).to(tl.int32)
            selected = tl.where(bad, 0, selected)
            tl.store(output + run * width + pos, selected)
        tl.store(invalid + run, bad.to(tl.int32))


@dataclass
class PairFactors:
    left: torch.Tensor
    right: torch.Tensor
    ids: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    left_bg: torch.Tensor
    right_bg: torch.Tensor
    kind: int
    vocab: int
    rank: int

    def kernel_args(self):
        return (self.left, self.right, self.ids, self.keys, self.values,
                self.left_bg, self.right_bg)

    def constants(self):
        return (self.ids.shape[1], self.ids.shape[2], self.rank,
                max(32, triton.next_power_of_2(self.rank)), self.kind, self.vocab,
                self.keys.numel(), max(1, self.keys.numel().bit_length()))

    def boundary(self, rows, positions, states, *, left_boundary):
        """Only K scores per visible boundary, never a K-by-K table."""
        if self.kind == 0:
            if left_boundary:
                return (self.left[rows, positions, states, None, :] *
                        self.right[rows, positions + 1]).sum(-1)
            return (self.left[rows, positions] *
                    self.right[rows, positions + 1, states, None, :]).sum(-1)
        if left_boundary:
            a = self.ids[rows, positions, states, None]
            b = self.ids[rows, positions + 1]
        else:
            a = self.ids[rows, positions]
            b = self.ids[rows, positions + 1, states, None]
        query = a * self.vocab + b
        score = self.left_bg[a.clamp_min(0)] + self.right_bg[b.clamp_min(0)]
        if self.keys.numel():
            index = torch.searchsorted(self.keys, query).clamp_max(self.keys.numel() - 1)
            score = torch.where(self.keys[index] == query, self.values[index], score)
        return torch.where((a >= 0) & (b >= 0), score, 0.).float()


def _count_table(head):
    signature = (head.mode, head.strength, head.smoothing,
                 tuple((id(v), v._version, v.device) for v in
                       (head.left_counts, head.right_counts, head.pair_keys, head.pair_counts)))
    cached = _COUNT_CACHE.get(head)
    if cached is not None and cached[0] == signature:
        return cached[1]
    lc, rc = head.left_counts, head.right_counts
    n = lc.sum()
    pl, pr = (lc + 1.) / (n + head.vocab_size), (rc + 1.) / (n + head.vocab_size)
    eps = torch.where(n > 0, n.new_tensor(head.smoothing), n.new_tensor(1.))
    ml = (1. - eps) * lc / n.clamp_min(1.) + eps * pl
    mr = (1. - eps) * rc / n.clamp_min(1.) + eps * pr
    left_bg = head.strength * (eps.log() + pl.log() - ml.log())
    right_bg = head.strength * pr.log()
    if head.mode == 'pmi':
        right_bg -= head.strength * mr.log()
    keys = head.pair_keys
    a, b = keys // head.vocab_size, keys % head.vocab_size
    joint = (1. - eps) * (head.pair_counts / n.clamp_min(1.)) + eps * (pl[a] * pr[b])
    score = joint.log() - ml[a].log()
    if head.mode == 'pmi':
        score -= mr[b].log()
    result = (keys, (score * head.strength).float(), left_bg, right_bg)
    _COUNT_CACHE[head] = signature, result
    return result


def prepare_factors(packet, head, hidden=None, time=None):
    from .heads import GlobalPairHead, ContextualPairHead, _embeddings
    from .counts import CountBigramHead
    if (triton is None or not packet.unary.is_cuda or torch.version.hip is not None
            or packet.unary.dtype != torch.float32 or torch.is_grad_enabled()):
        raise ValueError('fused-pairs requires NVIDIA CUDA, Triton, FP32 unaries and inference mode')
    if not 1 <= packet.unary.shape[-1] <= 1024:
        raise ValueError('fused-pairs supports at most 1024 candidate states including the residual')
    ids = packet.candidate_ids.contiguous()
    empty = packet.unary.new_empty(0)
    if isinstance(head, CountBigramHead):
        keys, values, bg_l, bg_r = _count_table(head)
        return PairFactors(empty, empty, ids, keys, values, bg_l, bg_r, 1, head.vocab_size, 1)
    if not isinstance(head, GlobalPairHead) or not 1 <= head.rank <= 32:
        raise ValueError('fused-pairs supports count/global/contextual heads with learned rank <= 32')
    left = _embeddings(head.left, ids).float()
    right = _embeddings(head.right, ids).float()
    if isinstance(head, ContextualPairHead):
        if hidden is None or hidden.shape[:2] != ids.shape[:2] or hidden.shape[-1] != head.hidden_size:
            raise ValueError('Contextual head requires matching backbone hidden states')
        batch, length = ids.shape[:2]
        time = hidden.new_zeros(batch) if time is None else torch.as_tensor(time, device=hidden.device)
        if time.numel() == 1:
            time = time.expand(batch)
        time = time.reshape(batch, 1, 1).expand(batch, max(length - 1, 0), 1)
        normalized = head.norm(hidden.to(head.norm.weight.dtype))
        gate_input = torch.cat((normalized[:, :-1], normalized[:, 1:], time.to(normalized.dtype)), -1)
        left[:, :-1] *= 1. + head.gate(gate_input).float().unsqueeze(-2)
    return PairFactors(left.contiguous(), right.contiguous(), ids,
                       ids.new_empty(0), empty, empty, empty, 0, head.vocab_size, head.rank)


def _chunks(packet, factors, segment_batch_size):
    from .segments import validate_segment_batch_size
    validate_segment_batch_size(segment_batch_size)
    unary, masked = packet.unary, packet.masked
    batch, length, states = unary.shape
    finite = torch.isfinite(unary)
    if torch.any((~masked) & finite.sum(-1).ne(1)):
        raise ValueError('Every visible position must be clamped to one finite state')
    clamps = finite.long().argmax(-1)
    starts, ends = masked.clone(), masked.clone()
    starts[:, 1:] &= ~masked[:, :-1]
    ends[:, :-1] &= ~masked[:, 1:]
    rows, start = starts.nonzero(as_tuple=True)
    end = ends.nonzero(as_tuple=True)[1]
    lengths = end - start + 1
    # Same run order/chunking and padding as the dense segmented backend.
    limit = segment_batch_size or max(1, rows.numel())
    for offset in range(0, rows.numel(), limit):
        rb, rs, rl = rows[offset:offset+limit], start[offset:offset+limit], lengths[offset:offset+limit]
        width = int(rl.max())
        positions = rs[:, None] + torch.arange(width, device=unary.device)[None, :]
        valid = positions < (rs + rl)[:, None]
        packed = unary[rb[:, None], positions.clamp_max(length - 1)].clone()
        has_left = rs > 0
        if has_left.any():
            br, bp = rb[has_left], rs[has_left] - 1
            packed[has_left, 0] += factors.boundary(br, bp, clamps[br, bp], left_boundary=True)
        has_right = rs + rl < length
        if has_right.any():
            br, bp = rb[has_right], (rs + rl - 1)[has_right]
            packed[has_right, rl[has_right] - 1] += factors.boundary(
                br, bp, clamps[br, bp + 1], left_boundary=False)
        dummy = unary.new_full((states,), -torch.inf)
        dummy[0] = 0.
        packed = torch.where(valid[..., None], packed, dummy).contiguous()
        yield packed, rb.contiguous(), rs.contiguous(), rl.contiguous(), positions, valid


def _messages(unary, factors, rows, starts, lengths, *, backward=False):
    batch, width, states = unary.shape
    result = torch.empty_like(unary)
    if backward:
        result[:, -1] = 0.
        positions = range(width - 2, -1, -1)
    else:
        result[:, 0] = unary[:, 0]
        positions = range(1, width)
    tile = 32
    with torch.cuda.device(unary.device):
        for pos in positions:
            _message_kernel[(batch, triton.cdiv(states, tile))](
                unary, result, *factors.kernel_args(), rows, starts, lengths,
                pos, width, *factors.constants(), max(32, triton.next_power_of_2(states)),
                tile, backward, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return result


def _sample_with_noise(alpha, factors, rows, starts, lengths, log_noise):
    batch, width, states = alpha.shape
    result = torch.empty((batch, width), dtype=torch.long, device=alpha.device)
    invalid = torch.empty(batch, dtype=torch.int32, device=alpha.device)
    with torch.cuda.device(alpha.device):
        _sample_kernel[(batch,)](
            alpha, log_noise, result, invalid, *factors.kernel_args(), rows, starts,
            lengths, width, *factors.constants(), max(32, triton.next_power_of_2(states)),
            num_warps=4, enable_fp_fusion=False)
    if invalid.any().item():
        raise RuntimeError('Invalid CRF sampling distribution: NaN, +inf, or no reachable state')
    return result


def infer_fused_pairs(packet, head, hidden=None, time=None, *, generator=None,
                      segment_batch_size=None, sampling='joint'):
    """Joint state draws or exact node log marginals for a candidate packet.

    Visible-only constants are irrelevant to these conditional operations.
    No partition/training API is exposed. This is an explicit experimental
    backend: unsupported inputs raise instead of silently changing precision.
    """
    if sampling not in ('joint', 'log_marginals'):
        raise ValueError('sampling must be joint or log_marginals')
    with torch.profiler.record_function('crf.pair_factors'):
        factors = prepare_factors(packet, head, hidden, time)
    if sampling == 'joint':
        result = packet.unary.argmax(-1)
    else:
        result = packet.unary.new_full(packet.unary.shape, -torch.inf)
        result.scatter_(-1, packet.unary.argmax(-1, keepdim=True), 0.)
    for unary, rows, starts, lengths, positions, valid in _chunks(packet, factors, segment_batch_size):
        with torch.profiler.record_function('crf.fused_pair_forward'):
            alpha = _messages(unary, factors, rows, starts, lengths)
        if sampling == 'joint':
            with torch.profiler.record_function('crf.backward_rng'):
                noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(generator=generator).log_()
            with torch.profiler.record_function('crf.fused_pair_sample'):
                values = _sample_with_noise(alpha, factors, rows, starts, lengths, noise)
        else:
            with torch.profiler.record_function('crf.fused_pair_backward'):
                beta = _messages(unary, factors, rows, starts, lengths, backward=True)
                values = alpha + beta - torch.logsumexp(alpha[:, -1], -1)[:, None, None]
        batches = rows[:, None].expand_as(positions)
        result[batches[valid], positions[valid]] = values[valid]
    return result
