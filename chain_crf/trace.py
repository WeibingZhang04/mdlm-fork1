"""Capture actual generation states without changing the sampler or its RNG.

These traces are for inspecting generation, not measuring latency: copying
states to the CPU synchronizes GPU work and adds overhead to the timed call.
"""
from __future__ import annotations

import inspect
import types

import torch

from chain_crf.generation import generate


class _RecordingBackbone:
    def __init__(self, backbone):
        self.backbone = backbone
        self.states = []
        self.times = []

    def __getattr__(self, name):
        return getattr(self.backbone, name)

    def __call__(self, tokens, time):
        # Record before the real call. No probabilities are sampled or
        # recomputed here, and no random numbers are consumed by the recorder.
        self.states.append(tokens.detach().cpu().clone())
        self.times.append(time.detach().cpu().clone())
        return self.backbone(tokens, time)


def generate_with_trace(backbone, head=None, mode='backbone', **kwargs):
    """Return ``(tokens, nonbenchmark_timing, trace)`` from one real batch.

    Every before-state was passed to the backbone; every after-state is the
    next actual before-state or the returned final output. Commit masks and
    token IDs are exact differences of those states, not inferred examples.
    Structured methods also record the actual full-chain proposals, including
    tokens not committed in that call. The factorized baseline samples only
    scheduled positions; unsampled proposal entries are explicitly ``None``.
    Keyword arguments are exactly those accepted by ``generation.generate``.
    """
    recorder = _RecordingBackbone(backbone)
    proposals = []
    production = inspect.unwrap(generate)
    original_sampler = production.__globals__['sample_candidate_tokens']

    def observe_sample(packet, states, **sample_kwargs):
        drawn = original_sampler(packet, states, **sample_kwargs)
        proposals.append({
            'tokens': drawn.detach().cpu().clone(),
            'states': states.detach().cpu().clone(),
            'selected_candidates': packet.candidate_ids.gather(
                -1, states.unsqueeze(-1)).squeeze(-1).detach().cpu().clone(),
        })
        return drawn

    # Execute the production function's exact bytecode with one observed
    # return value. Its private globals dictionary avoids monkeypatching the
    # live module, so concurrent untraced callers are unaffected. No sampler,
    # schedule, potential, RNG, or on-disk source is replaced.
    observed_globals = {**production.__globals__, 'sample_candidate_tokens': observe_sample}
    observed = types.FunctionType(production.__code__, observed_globals,
                                  production.__name__, production.__defaults__, production.__closure__)
    observed.__kwdefaults__ = production.__kwdefaults__
    tokens, timing = torch.no_grad()(observed)(recorder, head, mode, **kwargs)
    final = tokens.detach().cpu().clone()
    if not recorder.states or len(recorder.states) != timing['backbone_calls']:
        raise RuntimeError('Trace must contain every actual backbone call')
    if len(proposals) != (0 if mode == 'backbone' else timing['backbone_calls']):
        raise RuntimeError('Trace must observe every structured proposal exactly once')
    states = recorder.states + [final]
    mask_id = backbone.mask_id
    generated_length = int(timing['generated_length'])
    offset = kwargs.get('sample_offset', 0)
    examples = []
    for row in range(len(final)):
        events = []
        initial_mask = states[0][row].eq(mask_id)
        if int(initial_mask.sum()) != generated_length:
            raise RuntimeError('Initial trace mask does not match generated length')
        for index, (before, after) in enumerate(zip(states[:-1], states[1:])):
            before, after = before[row], after[row]
            masked = before.eq(mask_id)
            if not torch.equal(before[~masked], after[~masked]):
                raise RuntimeError('An observed token changed in the actual trace')
            committed = masked & after.ne(mask_id)
            if not bool(committed.any()):
                raise RuntimeError('An actual backbone call committed no tokens')
            positions = torch.where(committed)[0]
            event = {
                'call_index': index,
                'time': float(recorder.times[index][row]),
                'before_token_ids': before.tolist(),
                'after_token_ids': after.tolist(),
                'commit_mask': committed.tolist(),
                'committed_positions': positions.tolist(),
                'committed_token_ids': after[positions].tolist(),
            }
            if mode == 'backbone':
                proposed = [None] * len(before)
                for position in positions.tolist():
                    proposed[position] = int(after[position])
                event.update(proposal_scope='scheduled_positions_only',
                             proposed_token_ids=proposed,
                             proposed_candidate_state_indices=None,
                             proposed_candidate_ids=None)
            else:
                proposal = proposals[index]
                if not torch.equal(proposal['tokens'][row, positions], after[positions]):
                    raise RuntimeError('Committed tokens differ from the actual sampled proposal')
                event.update(proposal_scope='full_sequence_including_clamped_positions',
                             proposed_token_ids=proposal['tokens'][row].tolist(),
                             proposed_candidate_state_indices=proposal['states'][row].tolist(),
                             proposed_candidate_ids=proposal['selected_candidates'][row].tolist())
            events.append(event)
        if final[row].eq(mask_id).any():
            raise RuntimeError('Trace ends with an uncommitted mask')
        examples.append({
            'batch_row': row, 'draw_id': offset + row,
            'initial_token_ids': states[0][row].tolist(),
            'final_token_ids': final[row].tolist(), 'events': events,
        })
    trace = {
        'schema': 'chain_generation_trace_v1',
        'capture': 'actual_pre_forward_states_and_returned_final_output',
        'captures_discarded_proposals': mode != 'backbone',
        'candidate_id_note': '-1 denotes the residual state; proposed_token_ids contains its actual sampled token.',
        'benchmark_eligible': False,
        'mode': mode, 'sampling': kwargs.get('sampling', 'joint'),
        'inference': kwargs.get('inference', 'dense'),
        'requested_steps': kwargs.get('steps', 16),
        'actual_backbone_calls': timing['backbone_calls'],
        'generated_length': generated_length,
        'prefix_length': timing['prefix_length'],
        'batch_size': len(final), 'sample_offset': offset,
        'mask_id': mask_id, 'examples': examples,
    }
    return tokens, {**timing, 'trace_capture': True, 'benchmark_eligible': False}, trace


def masked_state_text(token_ids, mask_id, tokenizer):
    """Display observed spans and mask runs; exact token IDs remain authoritative.

    Absorbing-mask IDs are never passed to the clean-token tokenizer. Decoding
    separate visible spans can contain replacement characters when a mask
    interrupts a multi-token UTF-8 character; this is a display, not new text.
    """
    parts = []
    start = 0
    while start < len(token_ids):
        is_mask = token_ids[start] == mask_id
        stop = start + 1
        while stop < len(token_ids) and (token_ids[stop] == mask_id) == is_mask:
            stop += 1
        if is_mask:
            parts.append(f'[MASK × {stop-start}]')
        elif tokenizer is None:
            parts.append(' '.join(map(str, token_ids[start:stop])))
        else:
            parts.append(tokenizer.decode(token_ids[start:stop], clean_up_tokenization_spaces=False))
        start = stop
    return ''.join(parts)
