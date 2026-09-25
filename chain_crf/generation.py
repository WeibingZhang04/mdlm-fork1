"""Fixed random-order generation and held-out diagnostics for a token chain."""
from __future__ import annotations
import math
import time
from collections import Counter
import torch
from chain_crf.core import (build_candidates, sample_chain, sample_candidate_tokens,
                            chain_marginals, chain_log_marginals, gold_log_prob)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


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


@torch.no_grad()
def generate(backbone, head=None, mode='backbone', *, length=256, steps=16,
             batch_size=1, k=64, sampling='joint', temperature=1., device='cuda',
             sample_offset=0, prefix=None):
    """One batch. Schedule randomness is separate from token-draw randomness.

    All systems receive the same sample-index-dependent reveal permutation.
    Runtime excludes model loading and includes backbone, support, pair scores,
    DP, stochastic sampling, and token commitment. No final greedy denoise.
    """
    if steps < 1 or length < 1 or temperature <= 0 or batch_size < 1:
        raise ValueError('Positive steps, generated length and temperature required')
    if sampling not in ('joint', 'marginal'):
        raise ValueError('sampling must be joint or marginal')
    prefix = [] if prefix is None else list(prefix)
    if backbone.mask_id in prefix:
        raise ValueError('Prefix must contain observed clean tokens')
    tokens = torch.full((batch_size, len(prefix)+length), backbone.mask_id,
                        dtype=torch.long, device=device)
    if prefix:
        tokens[:, :len(prefix)] = torch.tensor(prefix, device=device)
    # Reproducible inputs are not replicated experiments: each configuration
    # is run once and generates different examples across sample IDs.
    orders = []
    for sample_id in range(sample_offset, sample_offset+batch_size):
        rng = torch.Generator().manual_seed(1729+sample_id)
        orders.append(torch.randperm(length, generator=rng)+len(prefix))
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
        prediction = backbone(tokens, t)
        synchronize(device)
        backbone_seconds += time.perf_counter()-before
        calls += 1
        before = time.perf_counter()
        if mode == 'backbone':
            # Only sample scheduled tokens: native factorization needs no DP
            # or candidate construction. Sampling uses FP64 probabilities.
            positions = order[:, committed:next_count]
            logits = prediction['log_probs'].gather(
                1, positions.unsqueeze(-1).expand(-1,-1,backbone.vocab_size)) / temperature
            logits[..., backbone.mask_id] = -torch.inf
            drawn = torch.multinomial(logits.double().softmax(-1).reshape(-1,backbone.vocab_size),
                                      1,generator=generator).reshape(batch_size,-1)
            tokens.scatter_(1,positions,drawn)
        else:
            packet = build_candidates(prediction['log_probs']/temperature,tokens,backbone.mask_id,k)
            unary, edge = potentials(packet,head,mode,prediction['hidden'],t)
            retained_mass.append(float((1-packet.unary[:,:,-1].exp())[packet.masked].mean()))
            if mode == 'independent':
                probabilities = unary.double().softmax(-1)
                states = torch.multinomial(probabilities.reshape(-1,probabilities.shape[-1]),
                                           1,generator=generator).reshape(tokens.shape)
            elif sampling == 'marginal':
                probabilities = chain_marginals(unary,edge).double()
                states = torch.multinomial(probabilities.reshape(-1,probabilities.shape[-1]),
                                           1,generator=generator).reshape(tokens.shape)
            else:
                states = sample_chain(unary,edge,generator=generator)
            drawn = sample_candidate_tokens(packet,states,generator=generator)
            positions = order[:,committed:next_count]
            tokens.scatter_(1,positions,drawn.gather(1,positions))
        committed = next_count
        synchronize(device)
        sampler_seconds += time.perf_counter()-before
    synchronize(device)
    elapsed = time.perf_counter()-start
    if tokens.eq(backbone.mask_id).any():
        raise RuntimeError('Generation left absorbing masks in the output')
    return tokens, {'elapsed_seconds':elapsed,'backbone_seconds':backbone_seconds,
                    'sampling_seconds':sampler_seconds,'backbone_calls':calls,
                    'samples':batch_size,'generated_tokens':batch_size*length,
                    'mean_retained_mass':sum(retained_mass)/len(retained_mass) if retained_mass else 1.,
                    'prefix_length':len(prefix),'generated_length':length}


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
