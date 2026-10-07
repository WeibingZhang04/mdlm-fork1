"""Hard unary vocabulary caps are independent of explicit CRF state count."""
import json

import pytest
import torch

from chain_crf import (ContextualPairHead, CountBigramHead, GlobalPairHead,
                       IndependentHead, build_candidates, gold_log_prob,
                       sample_candidate_tokens)
from chain_crf.generation import generate
from scripts.evaluate_chain_crf import main, parse_vocab_cap


class RankedBackbone:
    vocab_size = 7
    mask_id = 6

    def __call__(self, tokens, time):
        # MASK deliberately has the largest logit; clean rank is 0..5.
        logits = torch.tensor([0., -.1, -.2, -.3, -.4, -.5, 3.])
        return {'log_probs': logits.log_softmax(-1).expand(*tokens.shape, 7),
                'hidden': torch.zeros(*tokens.shape, 3)}


@pytest.mark.parametrize('cap,k', [(4,2), (4,4), (2,4), (4,0), (100,2)])
def test_cap_renormalizes_support_and_preserves_observed_tokens(cap, k):
    tokens = torch.tensor([[6,5,6]])
    prediction = RankedBackbone()(tokens, None)
    packet = build_candidates(prediction['log_probs'], tokens, 6, k, vocab_cap=cap)
    allowed, explicit = min(cap,6), min(k,cap,6)
    expected = torch.full((7,), -torch.inf)
    expected[:allowed] = torch.arange(allowed)*-.1
    expected -= expected.logsumexp(-1)
    torch.testing.assert_close(packet.normalized_log_probs[0,0], expected)
    torch.testing.assert_close(packet.unary[0,0,:explicit], expected[:explicit])
    torch.testing.assert_close(packet.unary[0,0,-1], expected[explicit:].logsumexp(-1))
    assert packet.candidate_ids.shape[-1] == explicit+1
    assert packet.candidate_ids[0,1,0] == 5
    assert packet.unary[0,1,0] == 0
    assert packet.unary[0,1,1:].isneginf().all()
    if allowed > explicit:
        states = torch.tensor([[explicit,0,explicit]])
        for seed in range(8):
            drawn = sample_candidate_tokens(packet, states, torch.Generator().manual_seed(seed))
            assert drawn[0,1] == 5
            assert ((drawn[0,[0,2]] >= explicit) & (drawn[0,[0,2]] < allowed)).all()


@pytest.mark.parametrize('k', [0,2,4,8])
def test_zero_pair_token_probability_equals_capped_backbone_including_tail(k):
    tokens = torch.tensor([[6]])
    pred = RankedBackbone()(tokens, None)
    expected = torch.arange(4,dtype=torch.float32).mul(-.1).log_softmax(-1)
    for token in range(6):
        packet = build_candidates(pred['log_probs'], tokens, 6, k,
                                  gold=torch.tensor([[token]]), vocab_cap=4)
        states = packet.unary.shape[-1]
        actual = gold_log_prob(packet, torch.zeros(1,0,states,states))[0]
        if token < 4:
            torch.testing.assert_close(actual, expected[token])
        else:
            assert actual.isneginf()


def test_tied_cap_support_matches_explicit_and_residual_support():
    packet = build_candidates(torch.zeros(1,2,7), torch.full((1,2),6), 6, 2, vocab_cap=4)
    allowed = packet.normalized_log_probs.isfinite()
    assert allowed.sum(-1).eq(4).all()
    assert not allowed[...,6].any()
    assert allowed.gather(-1,packet.candidate_ids[...,:2]).all()
    assert packet.unary[...,-1].exp().eq(.5).all()


@pytest.mark.parametrize('mode', ['backbone','count','global','contextual','independent'])
@pytest.mark.parametrize('inference,sampling', [('dense','joint'), ('segments','joint'),
                                              ('dense','marginal'), ('segments','marginal')])
@pytest.mark.parametrize('cap,k', [(4,2), (2,2), (1,3)])
def test_all_generation_modes_obey_cap_with_and_without_residual(mode,inference,sampling,cap,k):
    torch.manual_seed(12)
    head = {'backbone': lambda: None,
            'count': lambda: CountBigramHead(7).fit([[5,4,3,2,1,0]]*3),
            'global': lambda: GlobalPairHead(7,2),
            'contextual': lambda: ContextualPairHead(7,3,2),
            'independent': lambda: IndependentHead(7,3,2)}[mode]()
    tokens, stats = generate(RankedBackbone(),head,mode,length=12,steps=3,batch_size=3,
        k=k,vocab_cap=cap,device='cpu',prefix=[5],inference=inference,sampling=sampling)
    assert tokens[:,0].eq(5).all()
    assert ((tokens[:,1:] >= 0) & (tokens[:,1:] < cap)).all()
    assert stats['vocab_cap'] == cap and stats['backbone_calls'] == 3


def test_baseline_cap_is_independent_of_k_and_uncapped_reaches_tail():
    kwargs = dict(length=200,steps=2,device='cpu')
    a,_ = generate(RankedBackbone(),k=1,vocab_cap=2,**kwargs)
    b,_ = generate(RankedBackbone(),k=5,vocab_cap=2,**kwargs)
    torch.testing.assert_close(a,b)
    assert a.lt(2).all()
    uncapped,_ = generate(RankedBackbone(),k=1,**kwargs)
    explicit_none,_ = generate(RankedBackbone(),k=5,vocab_cap=None,**kwargs)
    torch.testing.assert_close(uncapped,explicit_none)
    assert uncapped.ge(2).any()


@pytest.mark.parametrize('value,expected', [('none',None), ('500',500), ('1000',1000), ('2000',2000)])
def test_numeric_cli_cap(value,expected):
    assert parse_vocab_cap(value) == expected


@pytest.mark.parametrize('value', ['0','-1','topk','1.5'])
def test_invalid_cli_cap_creates_no_output(tmp_path,value):
    output = tmp_path/'bad'
    with pytest.raises(SystemExit):
        main(['--output',str(output),'--vocab-cap',value])
    assert not output.exists()


@pytest.mark.parametrize('cap', [None,4])
def test_evaluator_records_cap_and_rejects_changed_resume(tmp_path,cap):
    output = tmp_path/'run'
    args = ['--output',str(output),'--synthetic','--device','cpu','--warmup','0',
            '--length','8','--steps','2','--samples','2','--k','2',
            '--vocab-cap','none' if cap is None else str(cap)]
    main(args)
    assert json.loads((output/'manifest.json').read_text())['config']['vocab_cap'] == cap
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['vocab_cap'] == cap and metrics['k'] == 2
    rows = [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    assert all(row['vocab_cap'] == cap for row in rows)
    main(args+['--resume'])
    with pytest.raises(ValueError,match='Resume manifest'):
        main(args[:-1]+['3','--resume'])


def test_cap_cannot_silently_leave_denoising_uncapped(tmp_path):
    output = tmp_path/'bad'
    with pytest.raises(ValueError,match='generation-only'):
        main(['--output',str(output),'--vocab-cap','500','--denoise-only'])
    assert not output.exists()
