import json

import pytest
import torch

from chain_crf.backbone import SyntheticBackbone
from chain_crf.counts import CountBigramHead
from chain_crf.generation import generate
from chain_crf.heads import ContextualPairHead, GlobalPairHead, IndependentHead
from chain_crf.trace import generate_with_trace, masked_state_text
from scripts.trace_chain_generation import main


def make_head(mode):
    if mode == 'backbone':
        return None
    if mode == 'count':
        return CountBigramHead(17, strength=.5).fit([[0,1,2,0,1,3]]*5)
    if mode == 'contextual':
        return ContextualPairHead(17,12,rank=3,mlp_size=4)
    if mode == 'independent':
        return IndependentHead(17,12,rank=3)
    head = GlobalPairHead(17,rank=3)
    with torch.no_grad():
        head.right.weight.normal_(std=.4)
    return head


@pytest.mark.parametrize('inference', ['dense','segments'])
@pytest.mark.parametrize('mode,sampling', [
    ('backbone','joint'),('count','joint'),('global','joint'),('global','marginal'),
    ('contextual','joint'),('contextual','marginal'),('independent','joint')])
def test_trace_records_real_states_without_changing_draws(mode, sampling, inference):
    torch.manual_seed(123)
    head = make_head(mode)
    model = SyntheticBackbone()
    kwargs = dict(length=7, steps=3, batch_size=2, k=2, sampling=sampling,
                  device='cpu', sample_offset=73, prefixes=[[2,3],[0,1]], inference=inference)
    untraced, original_timing = generate(model, head, mode, **kwargs)
    rng_before = torch.get_rng_state().clone()
    traced, timing, trace = generate_with_trace(model, head, mode, **kwargs)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert torch.equal(traced, untraced)
    assert timing['benchmark_eligible'] is False and trace['benchmark_eligible'] is False
    assert trace['actual_backbone_calls'] == original_timing['backbone_calls'] == 3
    assert [r['draw_id'] for r in trace['examples']] == [73,74]
    for row, example in enumerate(trace['examples']):
        assert example['initial_token_ids'] == kwargs['prefixes'][row]+[16]*7
        assert example['final_token_ids'] == traced[row].tolist()
        state = example['initial_token_ids'].copy()
        seen = []
        for event in example['events']:
            assert event['before_token_ids'] == state
            positions = event['committed_positions']
            assert positions == [i for i, yes in enumerate(event['commit_mask']) if yes]
            assert event['time'] == pytest.approx((7-len(seen))/7)
            for position, token in zip(positions, event['committed_token_ids']):
                assert state[position] == 16 and token != 16
                assert event['proposed_token_ids'][position] == token
                state[position] = token
            if mode == 'backbone':
                assert event['proposal_scope'] == 'scheduled_positions_only'
                assert [i for i,t in enumerate(event['proposed_token_ids']) if t is not None] == positions
                assert event['proposed_candidate_state_indices'] is None
            else:
                assert event['proposal_scope'] == 'full_sequence_including_clamped_positions'
                assert len(event['proposed_token_ids']) == 9
                assert all(0 <= t < 16 for t in event['proposed_token_ids'])
                assert len(event['proposed_candidate_state_indices']) == 9
                assert len(event['proposed_candidate_ids']) == 9
                for token, candidate in zip(event['proposed_token_ids'], event['proposed_candidate_ids']):
                    assert candidate == -1 or candidate == token
            assert state == event['after_token_ids']
            seen.extend(positions)
        assert sorted(seen) == list(range(2,9))
        assert state == example['final_token_ids']


def test_recorded_snapshots_equal_backbone_inputs_and_skip_empty_steps():
    class RecordingSynthetic(SyntheticBackbone):
        def __init__(self):
            super().__init__()
            self.inputs = []
        def forward(self, tokens, time):
            self.inputs.append(tokens.clone())
            return super().forward(tokens,time)
    model = RecordingSynthetic()
    _, _, trace = generate_with_trace(model, length=3, steps=128, device='cpu')
    assert trace['requested_steps'] == 128 and trace['actual_backbone_calls'] == 3
    for actual, event in zip(model.inputs, trace['examples'][0]['events']):
        assert actual[0].tolist() == event['before_token_ids']


def test_proposal_observation_does_not_replace_live_module_functions():
    import chain_crf.generation as generation
    original_generate = generation.generate
    original_sample = generation.sample_candidate_tokens
    generate_with_trace(SyntheticBackbone(), make_head('global'), 'global',
                        length=5, steps=3, k=2, device='cpu')
    assert generation.generate is original_generate
    assert generation.sample_candidate_tokens is original_sample


def test_display_never_sends_absorbing_mask_to_tokenizer():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            assert 16 not in ids
            return ''.join(chr(97+i) for i in ids)
    assert masked_state_text([16,16,0,1,16,2],16,Tokenizer()) == '[MASK × 2]ab[MASK × 1]c'


def test_cli_preserves_all_requested_draws_and_refuses_overwrite(tmp_path):
    output = tmp_path/'traces'
    args = ['--output',str(output),'--synthetic','--device','cpu','--length','5',
            '--steps','3','--samples','3','--batch-size','2','--sample-offset','20000']
    main(args)
    manifest = json.loads((output/'manifest.json').read_text())
    artifact = json.loads((output/'traces.json').read_text())
    assert artifact['complete'] and artifact['samples'] == 3
    assert manifest['benchmark_eligible'] is False and manifest['backbone']['synthetic_only']
    assert 'chain_crf/trace.py' in manifest['source_sha256']
    examples = [e for batch in artifact['batches'] for e in batch['examples']]
    assert [e['draw_id'] for e in examples] == [20000,20001,20002]
    assert [e['sample_id'] for e in examples] == [0,1,2]
    before = (output/'traces.json').read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert (output/'traces.json').read_bytes() == before


def test_negative_draw_id_rejected_before_output(tmp_path):
    output = tmp_path/'invalid'
    with pytest.raises(ValueError,match='nonnegative'):
        main(['--output',str(output),'--synthetic','--sample-offset','-1'])
    assert not output.exists()
