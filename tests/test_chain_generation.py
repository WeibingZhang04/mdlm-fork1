import pytest
import torch

from chain_crf import CountBigramHead, GlobalPairHead, IndependentHead, build_candidates, chain_marginals, gold_log_prob
from chain_crf.generation import denoising, generate, potentials, token_statistics


class ToyBackbone:
    vocab_size = 5
    mask_id = 4

    def __init__(self, mask_has_mass=False):
        self.history = []
        self.mask_has_mass = mask_has_mass

    def __call__(self, tokens, time):
        self.history.append(tokens.clone())
        logits = torch.tensor([0., -.4, -.8, -1.1, 2. if self.mask_has_mass else -torch.inf], device=tokens.device)
        lp = logits.log_softmax(-1).expand(*tokens.shape, self.vocab_size)
        hidden = torch.nn.functional.one_hot(tokens.remainder(3), num_classes=3).float()
        return {'log_probs':lp, 'hidden':hidden}


@pytest.mark.parametrize('mode,sampling', [('backbone','joint'),('global','joint'),('global','marginal'),('independent','joint')])
def test_generation_completes_preserves_prefix_and_is_reproducible(mode, sampling):
    head = None if mode == 'backbone' else IndependentHead(5, 3, 4) if mode == 'independent' else GlobalPairHead(5, 3)
    backbone = ToyBackbone(mask_has_mass=True)
    kwargs = dict(length=7, steps=3, batch_size=3, k=2, sampling=sampling, device='cpu', prefix=[2,1], sample_offset=17)
    tokens, stats = generate(backbone,head,mode,**kwargs)
    assert tokens.shape == (3,9) and not tokens.eq(4).any()
    assert tokens[:,:2].eq(torch.tensor([2,1])).all()
    assert stats['backbone_calls'] == 3 and stats['generated_tokens'] == 21
    assert stats['elapsed_seconds'] >= stats['backbone_seconds'] + stats['sampling_seconds']
    again, _ = generate(ToyBackbone(mask_has_mass=True),head,mode,**kwargs)
    torch.testing.assert_close(tokens,again)


def test_schedule_is_shared_across_methods_and_resume_offsets():
    reference, structured = ToyBackbone(), ToyBackbone()
    generate(reference,length=8,steps=4,batch_size=2,device='cpu',sample_offset=31)
    generate(structured,GlobalPairHead(5,2),'global',length=8,steps=4,batch_size=2,k=2,device='cpu',sample_offset=31)
    for base, pair in zip(reference.history,structured.history):
        torch.testing.assert_close(base.eq(4),pair.eq(4))
    resumed=ToyBackbone()
    generate(resumed,length=8,steps=4,batch_size=1,device='cpu',sample_offset=32)
    for full, resumed_batch in zip(reference.history,resumed.history):
        torch.testing.assert_close(full[1:2].eq(4),resumed_batch.eq(4))


def test_extra_steps_do_not_produce_empty_backbone_calls():
    output, stats=generate(ToyBackbone(),length=3,steps=128,batch_size=1,device='cpu')
    assert stats['backbone_calls'] == 3
    assert output.shape == (1,3)


@pytest.mark.parametrize('mode', ['backbone','global','independent'])
def test_zero_pair_and_identity_adapter_denoising_equal_base(mode):
    head=None if mode=='backbone' else GlobalPairHead(5,2) if mode=='global' else IndependentHead(5,3,4)
    tokens=torch.tensor([[0,1,2,3],[1,2,3,0],[2,0,1,3]])
    records=denoising(ToyBackbone(),tokens,head,mode,k=1,device='cpu',batch_size=2)
    for record in records:
        assert record['base_nll_per_masked_token'] == pytest.approx(record['joint_nll_per_masked_token'],abs=2e-6)
        assert record['base_nll_per_masked_token'] == pytest.approx(record['own_marginal_nll_per_masked_token'],abs=2e-6)


def test_count_chain_uses_real_adjacency_and_visible_boundaries():
    backbone=ToyBackbone()
    clean=torch.tensor([[0,1,2,3,0,1]])
    corrupted=torch.tensor([[4,1,4,4,0,1]])
    pred=backbone(corrupted,torch.tensor([.5]))
    packet=build_candidates(pred['log_probs'],corrupted,4,4,gold=clean)
    head=CountBigramHead(5,strength=.7).fit([[0,1,2,3,0,1]]*50)
    raw=head(packet.candidate_ids)
    unary,edge=potentials(packet,head,'count',pred['hidden'],torch.tensor([.5]))
    # The final edge is known-known and removable; all four others retain
    # original-position adjacency, including both masked-span boundaries.
    torch.testing.assert_close(edge[:,:4],raw[:,:4])
    assert edge[:,4].eq(0).all()
    torch.testing.assert_close(gold_log_prob(packet,edge),gold_log_prob(packet,raw),atol=2e-5,rtol=0)
    marginal=chain_marginals(unary,edge)
    assert marginal[0,1,0] == pytest.approx(1.,abs=2e-6)
    assert marginal[0,4,0] == pytest.approx(1.,abs=2e-6)


def test_singleton_joint_equals_own_marginal_with_boundary_counts():
    n=12000
    head=CountBigramHead(5,strength=.8).fit([[2,0],[2,0],[2,1],[3,3]])
    joint,_=generate(ToyBackbone(),head,'count',length=1,steps=1,batch_size=n,k=4,device='cpu',prefix=[2],sampling='joint')
    marginal,_=generate(ToyBackbone(),head,'count',length=1,steps=1,batch_size=n,k=4,device='cpu',prefix=[2],sampling='marginal')
    p=torch.bincount(joint[:,-1],minlength=5).float()/n
    q=torch.bincount(marginal[:,-1],minlength=5).float()/n
    torch.testing.assert_close(p,q,atol=.015,rtol=0)


def test_token_statistics_do_not_form_cross_document_ngrams():
    report=token_statistics([[0,1],[2,3]])
    assert report['distinct_2']==1.
    assert report['distinct_3']==0.
    assert report['tokens']==4
    assert report['within_sample_repeat_2']==0.
