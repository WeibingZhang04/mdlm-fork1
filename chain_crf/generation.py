"""Fixed or native-style DDPM-cache generation and token-chain diagnostics."""
from __future__ import annotations
import math
import time
from collections import Counter
import torch
from chain_crf.core import (build_candidates, build_sampling_candidates, sample_chain, sample_candidate_tokens,
                            chain_marginals, chain_log_marginals, gold_log_prob, topk_clean)

def _ddpm_clean_probs(log_probs, tokens, mask_id, cap):
    """Native capped clean law in FP32, with visible tokens clamped (SUBS).

    FrozenMDLM already returns normalized clean log probabilities in FP32.
    Keep the full vocabulary tensor for native categorical sampling. This
    counterpart avoids importing diffusion.py's Lightning/HF dependencies.

    This is a remix of functions in diffusion.py
    """
    scores = log_probs.float().clone()
    scores[..., mask_id] = -torch.inf
    probabilities = scores.exp()
    masked = tokens.eq(mask_id)
    if cap is not None and masked.any():
        values, ids = topk_clean(scores[masked], mask_id, cap)
        probabilities[masked] = torch.zeros_like(scores[masked]).scatter(-1, ids, values.softmax(-1))
    # Unlike native Diffusion.forward, FrozenMDLM does not clamp visible rows.
    if (~masked).any():
        visible = torch.zeros_like(probabilities[~masked])
        visible.scatter_(-1, tokens[~masked].unsqueeze(-1), 1.)
        probabilities[~masked] = visible
    return probabilities


def _native_categorical(probabilities, generator):
    # _sample_categorical from diffusion.py from MDLM but with an explicit generator
    uniform = torch.rand(probabilities.shape, dtype=probabilities.dtype,
                         device=probabilities.device, generator=generator)
    denominator = 1e-10 - (uniform + 1e-10).log()
    return (probabilities / denominator).argmax(dim=-1)


def _ddpm_transition(p_x0, tokens, mask_id, t, dt, generator):
    """Adapted from MDLM's diffusion.Diffusion._ddpm_caching_update's sampling step.

    p_x0 contains clean-token probabilities; t is a vector of batch times.
    """
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, mask_id] = move_chance_s[:, :, 0]
    drawn = _native_categorical(q_xs, generator)
    return torch.where(tokens != mask_id, tokens, drawn)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def clean_token_ids(values, *, vocab_size, mask_id, label='Prefix'):
    """Validate exact IDs without coercion, tokenization, or special tokens."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f'{label} must be a list of integer token IDs')
    if any(type(token) is not int for token in values):
        raise ValueError(f'{label} must contain integer token IDs')
    if any(token < 0 or token >= vocab_size for token in values):
        raise ValueError(f'{label} contains out-of-vocabulary token IDs')
    if mask_id in values:
        raise ValueError(f'{label} must contain observed clean tokens, not absorbing masks')
    return list(values)


def potentials(packet, head, mode, hidden, time_value):
    unary = packet.unary
    b, length, states = unary.shape
    if mode == 'independent':
        delta = head(packet.candidate_ids, hidden, time_value)
        unary = unary + delta * packet.masked.unsqueeze(-1)
        edge = unary.new_zeros((b, length-1, states, states))
    elif head is None:
        edge = unary.new_zeros((b, length-1, states, states))
    else:
        edge = head(packet.candidate_ids, hidden, time_value)
        # Known-known factors are constants under the conditional model.
        # Remove them before DP to avoid cancelling large, irrelevant scores.
        active = packet.masked[:, :-1] | packet.masked[:, 1:]
        edge = torch.where(active[..., None, None], edge, torch.zeros_like(edge))
    return unary, edge


def _draw_structured(prediction, tokens, mask_id, head, mode, t, k,
                     sampling, temperature, inference, vocab_cap, generator, *,
                     resolve_mask=None, segment_batch_size=None):
    """Draw all chain states, resolving tail tokens only where requested."""
    with torch.profiler.record_function('crf.candidates'):
        if resolve_mask is None:
            packet = build_candidates(prediction['log_probs']/temperature,tokens,mask_id,k,
                                      vocab_cap=vocab_cap)
        else:
            # Generation needs sampled token IDs, not the training likelihood packet.
            packet = build_sampling_candidates(prediction['log_probs'], tokens, mask_id, k,
                                               vocab_cap=vocab_cap, temperature=temperature)
    with torch.profiler.record_function('crf.potentials'):
        unary, edge = potentials(packet,head,mode,prediction['hidden'],t)
    with torch.profiler.record_function('crf.retained_mass'):
        mass = (1-packet.unary[:,:,-1].exp())[packet.masked]
        retained_mass = float(mass.mean()) if mass.numel() else 1.
    with torch.profiler.record_function('crf.inference'):
        if mode == 'independent':
            probabilities = unary.double().softmax(-1)
            states = torch.multinomial(probabilities.reshape(-1,probabilities.shape[-1]),
                                       1,generator=generator).reshape(tokens.shape)
        elif sampling == 'marginal':
            if inference == 'segments':
                from chain_crf.segments import segmented_marginals
                probabilities = segmented_marginals(
                    unary, edge, packet.masked, segment_batch_size=segment_batch_size).double()
            else:
                probabilities = chain_marginals(unary,edge).double()
            states = torch.multinomial(probabilities.reshape(-1,probabilities.shape[-1]),
                                       1,generator=generator).reshape(tokens.shape)
        else:
            if inference == 'segments':
                from chain_crf.segments import sample_segmented_chain
                states = sample_segmented_chain(
                    unary, edge, packet.masked, generator=generator,
                    segment_batch_size=segment_batch_size)
            else:
                states = sample_chain(unary,edge,generator=generator)
    with torch.profiler.record_function('crf.residual_expand'):
        if resolve_mask is None:
            drawn = sample_candidate_tokens(packet,states,generator=generator)
        else:
            drawn = sample_candidate_tokens(packet,states,generator=generator,
                                           resolve_mask=resolve_mask)
    return drawn, retained_mass


def _denoise_structured(prediction, tokens, mask_id, head, mode, t, k,
                        inference, vocab_cap, *, segment_batch_size=None):
    """Greedy token marginals for the current top-K-plus-tail CRF model."""
    with torch.profiler.record_function('crf.candidates'):
        packet = build_candidates(prediction['log_probs'], tokens, mask_id, k,
                                  vocab_cap=vocab_cap)
    with torch.profiler.record_function('crf.potentials'):
        unary, edge = potentials(packet, head, mode, prediction['hidden'], t)
    with torch.profiler.record_function('crf.inference'):
        if mode == 'independent':
            log_marginals = unary.log_softmax(-1)
        elif inference == 'segments':
            from chain_crf.segments import segmented_log_marginals
            log_marginals = segmented_log_marginals(
                unary, edge, packet.masked, segment_batch_size=segment_batch_size)
        else:
            log_marginals = chain_log_marginals(unary, edge)
    with torch.profiler.record_function('crf.residual_argmax'):
        state_lp = log_marginals[packet.masked]
        ids = packet.candidate_ids[packet.masked]
        tail_mass = packet.unary[packet.masked][:, -1]
        # An impossible tail has no tokens; avoid -inf - -inf in that case.
        safe_tail_mass = torch.where(torch.isfinite(tail_mass), tail_mass,
                                     torch.zeros_like(tail_mass))
        token_lp = packet.normalized_log_probs[packet.masked].clone()
        # P(v) = P(tail state) * P_backbone(v | tail) for residual tokens.
        token_lp += (state_lp[:, -1] - safe_tail_mass)[:, None]
        # Explicit tokens instead have their own CRF state probabilities.
        token_lp.scatter_(-1, ids[:, :-1], state_lp[:, :-1])
        result = tokens.clone()
        result[packet.masked] = token_lp.argmax(-1)
    return result


@torch.no_grad()
def generate(backbone, head=None, mode='backbone', *, length=256, steps=16,
             batch_size=1, k=64, sampling='joint', temperature=1., device='cuda',
             sample_offset=0, prefix=None, inference='dense', prefixes=None,
             vocab_cap=None, sampler='fixed', noise_removal=None,
             sampling_eps=1e-5, stage_timing=None, segment_batch_size=None):
    """One batch using the fixed schedule or native-style DDPM-cache transitions.

    Fixed sampling shares a reveal permutation and has no final denoise.
    DDPM-cache uses native clean/MASK draws and defaults to final denoising.
    Runtime excludes model loading and includes backbone, support, pair scores,
    DP, stochastic sampling, token commitment, and any final denoise.
    ``prefixes`` accepts one equal-length exact-token prefix per batch row;
    different shapes must be placed in separate batches (never padded).
    A positive vocab_cap restricts baseline and structured draws to that many
    top clean tokens at each masked position, separately from CRF k. None
    preserves full token support; residuals contain only allowed tokens.
    """
    from chain_crf.segments import validate_segment_batch_size
    validate_segment_batch_size(segment_batch_size)
    if segment_batch_size is not None and inference != 'segments':
        raise ValueError('segment_batch_size requires inference=segments')
    if sampler not in ('fixed', 'ddpm_cache'):
        raise ValueError('sampler must be fixed or ddpm_cache')
    noise_removal = sampler == 'ddpm_cache' if noise_removal is None else noise_removal
    stage_timing = sampler == 'fixed' if stage_timing is None else stage_timing
    if sampler == 'ddpm_cache' and temperature != 1.:
        raise ValueError('Native-style ddpm_cache requires temperature=1')
    if sampler == 'fixed' and noise_removal:
        raise ValueError('noise_removal requires sampler=ddpm_cache')
    if sampler == 'fixed' and not stage_timing:
        raise ValueError('The fixed sampler retains its existing stage timing')
    if not 0 < sampling_eps < 1:
        raise ValueError('sampling_eps must be in (0,1)')
    if steps < 1 or length < 1 or temperature <= 0 or batch_size < 1:
        raise ValueError('Positive steps, generated length and temperature required')
    if sampling not in ('joint', 'marginal'):
        raise ValueError('sampling must be joint or marginal')
    if inference not in ('dense', 'segments'):
        raise ValueError('inference must be dense or segments')
    if vocab_cap is not None and (type(vocab_cap) is not int or vocab_cap < 1):
        raise ValueError('vocab_cap must be None or a positive integer')
    if prefixes is not None and prefix is not None:
        raise ValueError('Choose shared prefix or per-example prefixes, not both')
    if prefixes is None:
        prefix = clean_token_ids([] if prefix is None else prefix,
            vocab_size=backbone.vocab_size, mask_id=backbone.mask_id)
        prefix_rows = [prefix] * batch_size
    else:
        if not isinstance(prefixes, (list, tuple)) or len(prefixes) != batch_size:
            raise ValueError('Per-example prefixes must match batch_size')
        prefix_rows = [clean_token_ids(row, vocab_size=backbone.vocab_size,
                       mask_id=backbone.mask_id) for row in prefixes]
        if len({len(row) for row in prefix_rows}) != 1:
            raise ValueError('Per-example prefixes in one batch must have equal lengths')
    prefix_length = len(prefix_rows[0])
    prefix_tensor = torch.tensor(prefix_rows, dtype=torch.long, device=device)
    tokens = torch.full((batch_size, prefix_length+length), backbone.mask_id,
                        dtype=torch.long, device=device)
    if prefix_length:
        tokens[:, :prefix_length] = prefix_tensor
    if sampler == 'ddpm_cache':
        generator = torch.Generator(device=device).manual_seed(2718+sample_offset)
        reveal_generator = torch.Generator(device=device).manual_seed(1729+sample_offset)
        tokens, timing = _generate_ddpm_cache(
            backbone, tokens, steps, generator=generator, head=head, mode=mode,
            k=k, sampling=sampling, inference=inference, reveal_generator=reveal_generator,
            vocab_cap=vocab_cap, noise_removal=noise_removal, eps=sampling_eps,
            stage_timing=stage_timing, segment_batch_size=segment_batch_size)
        remaining_masks = int(tokens.eq(backbone.mask_id).sum())
        if noise_removal and remaining_masks:
            raise RuntimeError('Final denoising left absorbing masks in the output')
        if prefix_length and not torch.equal(tokens[:, :prefix_length], prefix_tensor):
            raise RuntimeError('Generation changed an observed prefix token')
        return tokens, {**timing, 'samples':batch_size, 'generated_tokens':batch_size*length,
                        'prefix_length':prefix_length, 'generated_length':length,
                        'vocab_cap':vocab_cap, 'sampler':sampler, 'remaining_masks':remaining_masks,
                        'token_seed':2718+sample_offset,
                        'reveal_seed':1729+sample_offset if mode != 'backbone' else None}
    # Reproducible inputs are not replicated experiments: each configuration
    # is run once and generates different examples across sample IDs.
    orders = []
    for sample_id in range(sample_offset, sample_offset+batch_size):
        rng = torch.Generator().manual_seed(1729+sample_id)
        orders.append(torch.randperm(length, generator=rng)+prefix_length)
    order = torch.stack(orders).to(device)
    generator = torch.Generator(device=device).manual_seed(2718+sample_offset)
    calls = 0
    backbone_seconds = 0.
    sampler_seconds = 0.
    retained_mass = []
    synchronize(device)
    start = time.perf_counter()
    committed = 0
    for step in range(steps):
        next_count = math.ceil((step+1)*length/steps)
        if next_count == committed:
            continue
        t = torch.full((batch_size,), 1.-committed/length, device=device)
        synchronize(device)
        before = time.perf_counter()
        with torch.profiler.record_function('mdlm.forward'):
            prediction = backbone(tokens, t)
            synchronize(device)
        backbone_seconds += time.perf_counter()-before
        calls += 1
        before = time.perf_counter()
        if mode == 'backbone':
            with torch.profiler.record_function('generation.backbone_sample'):
                # Only sample scheduled tokens; no chain DP is needed.
                # Sampling uses FP64 probabilities in both vocabulary modes.
                positions = order[:, committed:next_count]
                logits = prediction['log_probs'].gather(
                    1, positions.unsqueeze(-1).expand(-1,-1,backbone.vocab_size)) / temperature
                logits[..., backbone.mask_id] = -torch.inf
                if vocab_cap is not None:
                    values, ids = topk_clean(logits, backbone.mask_id, vocab_cap)
                    states = torch.multinomial(values.double().softmax(-1).reshape(-1,values.shape[-1]),
                                              1,generator=generator).reshape(*ids.shape[:-1],1)
                    drawn = ids.gather(-1,states).squeeze(-1)
                else:
                    drawn = torch.multinomial(logits.double().softmax(-1).reshape(-1,backbone.vocab_size),
                                              1,generator=generator).reshape(batch_size,-1)
                tokens.scatter_(1,positions,drawn)
        else:
            # The fixed permutation already tells us which token IDs are
            # needed. Keep the full joint state draw, but resolve only these
            # tails and use the compact inference candidate representation.
            positions = order[:, committed:next_count]
            resolve_mask = torch.zeros_like(tokens, dtype=torch.bool)
            resolve_mask.scatter_(1, positions, True)
            drawn, mass = _draw_structured(
                prediction, tokens, backbone.mask_id, head, mode, t, k,
                sampling, temperature, inference, vocab_cap, generator,
                resolve_mask=resolve_mask, segment_batch_size=segment_batch_size)
            retained_mass.append(mass)
            with torch.profiler.record_function('generation.commit'):
                tokens.scatter_(1,positions,drawn.gather(1,positions))
        committed = next_count
        synchronize(device)
        sampler_seconds += time.perf_counter()-before
    synchronize(device)
    elapsed = time.perf_counter()-start
    if tokens.eq(backbone.mask_id).any():
        raise RuntimeError('Generation left absorbing masks in the output')
    if prefix_length and not torch.equal(tokens[:, :prefix_length], prefix_tensor):
        raise RuntimeError('Generation changed an observed prefix token')
    return tokens, {'elapsed_seconds':elapsed,'backbone_seconds':backbone_seconds,
                    'sampling_seconds':sampler_seconds,'backbone_calls':calls,
                    'samples':batch_size,'generated_tokens':batch_size*length,
                    'mean_retained_mass':sum(retained_mass)/len(retained_mass) if retained_mass else 1.,
                    'prefix_length':prefix_length,'generated_length':length,
                    'vocab_cap':vocab_cap, 'sampler':'fixed', 'timing_mode':'stage_sync',
                    'noise_removal':False, 'noise_removal_method':'none', 'sampling_eps':None,
                    'reveal_sampler':'fixed_permutation', 'cache_hits':0, 'remaining_masks':0}


@torch.no_grad()
def _generate_ddpm_cache(backbone, tokens, steps, *, generator, head=None,
                         mode='backbone', k=64, sampling='joint', inference='dense',
                         reveal_generator=None, vocab_cap=None,
                         noise_removal=True, eps=1e-5, stage_timing=False,
                         segment_batch_size=None):
    """Native DDPM-cache baseline, or CRF clean draws with its reveal law.

    CRF token draws and native clean/MASK draws use separate RNG streams.
    For custom heads, only the native draw's reveal positions are retained;
    token identities come from the CRF. The head receives the current t on
    every step, even when the backbone is cached. Residual token identities
    are drawn only at revealed positions. This preserves the conditional law
    but changes token RNG consumption relative to eager tail expansion.
    Default timing synchronizes only at the outer measurement boundaries.
    stage_timing adds per-stage synchronization for diagnosis, with overhead.
    """
    if steps < 1 or not 0 < eps < 1:
        raise ValueError('Positive steps and sampling eps in (0,1) required')
    if mode not in ('backbone', 'count', 'global', 'contextual', 'independent'):
        raise ValueError('Unknown DDPM generation mode')
    if mode != 'backbone' and reveal_generator is None:
        raise ValueError('Custom heads require a separate reveal_generator')
    device = tokens.device
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps
    prediction_cache = p_x0_cache = None
    time_conditioning = getattr(backbone, 'time_conditioning', True)
    calls = cache_hits = 0
    backbone_seconds = sampler_seconds = 0.
    retained_mass = []

    def predict(t):
        nonlocal calls, backbone_seconds
        if stage_timing:
            synchronize(device)
            before = time.perf_counter()
        with torch.profiler.record_function('mdlm.forward'):
            prediction = backbone(tokens, t)
        if stage_timing:
            synchronize(device)
            backbone_seconds += time.perf_counter() - before
        calls += 1
        return prediction

    synchronize(device)
    start = time.perf_counter()
    for step in range(steps):
        t = timesteps[step] * torch.ones(tokens.shape[0], device=device)
        if prediction_cache is None:
            prediction_cache = predict(t)
        else:
            cache_hits += 1
        if stage_timing:
            before = time.perf_counter()
        if mode == 'backbone':
            with torch.profiler.record_function('generation.backbone_sample'):
                if p_x0_cache is None:
                    p_x0_cache = _ddpm_clean_probs(
                        prediction_cache['log_probs'], tokens, backbone.mask_id, vocab_cap)
                next_tokens = _ddpm_transition(
                    p_x0_cache, tokens, backbone.mask_id, t, dt, generator)
        else:
            with torch.profiler.record_function('generation.reveal'):
                # This native draw uses its own RNG, independent of CRF states.
                # Determine which token IDs will be needed before expanding tails.
                if p_x0_cache is None:
                    p_x0_cache = _ddpm_clean_probs(
                        prediction_cache['log_probs'], tokens, backbone.mask_id, vocab_cap)
                native_next = _ddpm_transition(
                    p_x0_cache, tokens, backbone.mask_id, t, dt, reveal_generator)
                reveal = tokens.eq(backbone.mask_id) & native_next.ne(backbone.mask_id)
            # Still sample the full joint chain: skipping unrevealed states
            # would lose their effect on the revealed states' distribution.
            drawn, mass = _draw_structured(
                prediction_cache, tokens, backbone.mask_id, head, mode, t, k,
                sampling, 1., inference, vocab_cap, generator, resolve_mask=reveal,
                segment_batch_size=segment_batch_size)
            retained_mass.append(mass)
            with torch.profiler.record_function('generation.commit'):
                # Unresolved tail IDs are -1 only where reveal is false.
                next_tokens = torch.where(reveal, drawn, tokens)
        if not torch.equal(next_tokens, tokens) or time_conditioning:
            prediction_cache = p_x0_cache = None
        tokens = next_tokens
        if stage_timing:
            synchronize(device)
            sampler_seconds += time.perf_counter() - before

    if noise_removal:
        # Fresh backbone prediction, plus the selected head for custom methods.
        t = timesteps[-1] * torch.ones(tokens.shape[0], device=device)
        prediction = predict(t)
        if stage_timing:
            before = time.perf_counter()
        if mode == 'backbone':
            scores = prediction['log_probs']
            if vocab_cap is not None:
                scores = _ddpm_clean_probs(scores, tokens, backbone.mask_id, vocab_cap)
            tokens = torch.where(tokens != backbone.mask_id, tokens, scores.argmax(-1))
        else:
            tokens = _denoise_structured(prediction, tokens, backbone.mask_id,
                                        head, mode, t, k, inference, vocab_cap,
                                        segment_batch_size=segment_batch_size)
        if stage_timing:
            synchronize(device)
            sampler_seconds += time.perf_counter() - before
    synchronize(device)
    elapsed = time.perf_counter() - start
    return tokens, {'elapsed_seconds': elapsed,
                    'backbone_seconds': backbone_seconds if stage_timing else None,
                    'sampling_seconds': sampler_seconds if stage_timing else None,
                    'timing_mode': 'stage_sync' if stage_timing else 'outer_sync',
                    'backbone_calls': calls,
                    'cache_hits': cache_hits, 'noise_removal': noise_removal,
                    'noise_removal_method': ('backbone_argmax' if mode == 'backbone'
                                             else 'token_marginal_argmax') if noise_removal else 'none',
                    'sampling_eps': eps, 'reveal_sampler': 'native_categorical',
                    'mean_retained_mass': sum(retained_mass)/len(retained_mass) if retained_mass else 1.}


@torch.no_grad()
def denoising(backbone, tokens, head=None, mode='backbone', *, k=64,
              mask_rates=(.25,.5,.75,.9), device='cuda', batch_size=1):
    if len(tokens) == 0 or batch_size < 1:
        raise ValueError('Denoising needs nonempty tokens and positive batch_size')
    if any(not 0 < rate <= 1 for rate in mask_rates):
        raise ValueError('Mask rates must be in (0,1]')
    results = []
    rng = torch.Generator(device=device).manual_seed(314159)
    for rate in mask_rates:
        totals = dict(base_nll=0.,joint_nll=0.,own_marginal_nll=0.,masked_tokens=0,
                      explicit_gold=0,retained_mass_sum=0.,examples=0)
        for offset in range(0,len(tokens),batch_size):
            clean = tokens[offset:offset+batch_size].to(device)
            mask = torch.rand(clean.shape,device=device,generator=rng)<rate
            if not mask.any():
                mask[0,0]=True
            corrupted = clean.masked_fill(mask,backbone.mask_id)
            t = torch.full((len(clean),),float(rate),device=device)
            pred = backbone(corrupted,t)
            packet = build_candidates(pred['log_probs'],corrupted,backbone.mask_id,k,gold=clean)
            unary,edge=potentials(packet,head,mode,pred['hidden'],t)
            # Avoid inf-inf on clamped invalid slots when computing delta.
            if mode=='independent':
                delta=head(packet.candidate_ids,pred['hidden'],t)*packet.masked.unsqueeze(-1)
                joint=gold_log_prob(packet,edge,delta)
            else:
                joint=gold_log_prob(packet,edge)
            marginal=chain_log_marginals(unary,edge).gather(-1,packet.gold_states.unsqueeze(-1)).squeeze(-1)
            own=marginal+packet.gold_tail_logprob
            base=packet.normalized_log_probs.gather(-1,clean.unsqueeze(-1)).squeeze(-1)
            totals['base_nll']-=float(base[mask].sum())
            totals['joint_nll']-=float(joint.sum())
            totals['own_marginal_nll']-=float(own[mask].sum())
            totals['masked_tokens']+=int(mask.sum())
            totals['explicit_gold']+=int(((packet.gold_states<packet.unary.shape[-1]-1)&mask).sum())
            totals['retained_mass_sum']+=float((1-packet.unary[:,:,-1].exp())[mask].sum())
            totals['examples']+=len(clean)
        n=totals['masked_tokens']
        results.append({'mask_rate':rate,**totals,
                        'base_nll_per_masked_token':totals['base_nll']/n,
                        'joint_nll_per_masked_token':totals['joint_nll']/n,
                        'own_marginal_nll_per_masked_token':totals['own_marginal_nll']/n,
                        'gold_coverage':totals['explicit_gold']/n,
                        'retained_mass':totals['retained_mass_sum']/n})
    return results


def token_statistics(sequences):
    counts=Counter(token for row in sequences for token in row)
    total=sum(counts.values())
    entropy=-sum((c/total)*math.log(c/total) for c in counts.values()) if total else 0.
    result={'token_entropy_nats':entropy,'unique_tokens':len(counts),'tokens':total}
    for n in (2,3,4):
        grams=[tuple(row[i:i+n]) for row in sequences for i in range(max(0,len(row)-n+1))]
        result[f'distinct_{n}']=len(set(grams))/len(grams) if grams else 0.
        fractions=[]
        for row in sequences:
            local=[tuple(row[i:i+n]) for i in range(max(0,len(row)-n+1))]
            if local:
                fractions.append(1-len(set(local))/len(local))
        result[f'within_sample_repeat_{n}']=sum(fractions)/len(fractions) if fractions else 0.
    return result
