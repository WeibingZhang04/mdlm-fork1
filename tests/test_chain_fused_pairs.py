"""GPU oracles for on-demand pair scores, clamps, residuals and joint draws."""
import itertools
import pytest
import torch

from chain_crf.core import build_candidates, _sampling_probs
from chain_crf.heads import GlobalPairHead, ContextualPairHead
from chain_crf.counts import CountBigramHead
from chain_crf import fused_pairs as fp
from chain_crf.segments import pack_segments, segmented_log_marginals


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='NVIDIA CUDA required')


def fixture(kind, k, length=9, batch=2):
    torch.manual_seed(517)
    vocab = max(k + 7, 23)
    lp = torch.randn(batch, length, vocab, device='cuda').log_softmax(-1)
    xt = torch.full((batch, length), vocab - 1, device='cuda', dtype=torch.long)
    if length > 3:
        xt[:, 0] = 2
        xt[0, 3] = 4
        xt[-1, -1] = 5
    packet = build_candidates(lp, xt, vocab - 1, k)
    hidden = torch.randn(batch, length, 8, device='cuda')
    time = torch.linspace(.2, .8, batch, device='cuda')
    if kind.startswith('count'):
        head = CountBigramHead(vocab, mode='conditional' if 'conditional' in kind else 'pmi',
                              strength=-.3 if 'negative' in kind else .4)
        if 'empty' not in kind:
            head.fit([[2, 4, 6, 4, 6, 8, 2], list(range(vocab-1))])
        head = head.cuda()
    else:
        head = (GlobalPairHead(vocab, 17) if kind == 'global' else
                ContextualPairHead(vocab, 8, rank=17)).cuda()
        for parameter in head.parameters():
            parameter.uniform_(-.35, .35)
    edge = head(packet.candidate_ids, hidden, time)
    return packet, head, hidden, time, edge


def oracle_forward(unary, edge):
    messages = [unary[:, 0]]
    for pos in range(1, unary.shape[1]):
        messages.append(unary[:, pos] + torch.logsumexp(messages[-1][:, :, None] + edge[:, pos-1], -2))
    return torch.stack(messages, 1)


@pytest.mark.parametrize('kind', ['global', 'contextual', 'count-pmi', 'count-conditional',
                                 'count-empty', 'count-negative'])
@pytest.mark.parametrize('k', [3, 64, 500])
@torch.no_grad()
def test_messages_marginals_and_fixed_noise(kind, k):
    packet, head, hidden, time, edge = fixture(kind, k)
    factors = fp.prepare_factors(packet, head, hidden, time)
    packed = pack_segments(packet.unary, edge, packet.masked)
    unary, rows, starts, lengths, positions, valid = next(fp._chunks(packet, factors, None))
    torch.testing.assert_close(unary, packed.unary, atol=2e-6, rtol=1e-6)
    actual = fp._messages(unary, factors, rows, starts, lengths)
    expected = oracle_forward(packed.unary, packed.edge)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-6)
    torch.testing.assert_close(_sampling_probs(actual), _sampling_probs(expected), atol=2e-6, rtol=1e-4)
    noise = torch.empty_like(actual, dtype=torch.float64).exponential_().log_()
    observed = fp._sample_with_noise(actual, factors, rows, starts, lengths, noise)
    reference = torch.empty(observed.shape, device='cuda', dtype=torch.long)
    for pos in range(actual.shape[1]-1, -1, -1):
        logits = expected[:, pos]
        if pos < actual.shape[1]-1:
            logits = logits + packed.edge[:, pos].gather(-1, reference[:, pos+1, None, None].expand(-1, logits.shape[-1], 1)).squeeze(-1)
        reference[:, pos] = (logits.double() - noise[:, pos]).argmax(-1)
    assert torch.equal(observed, reference)
    expected_marginals = segmented_log_marginals(packet.unary, edge, packet.masked)
    actual_marginals = fp.infer_fused_pairs(packet, head, hidden, time,
                                          sampling='log_marginals', segment_batch_size=2)
    torch.testing.assert_close(actual_marginals, expected_marginals, atol=3e-5, rtol=1e-5)


@pytest.mark.parametrize('length,k', [(1, 3), (2, 1000), (1024, 3)])
@torch.no_grad()
def test_length_and_state_boundaries(length, k):
    packet, head, hidden, time, edge = fixture('contextual', k, length=length, batch=1)
    expected = segmented_log_marginals(packet.unary, edge, packet.masked)
    actual = fp.infer_fused_pairs(packet, head, hidden, time, sampling='log_marginals')
    torch.testing.assert_close(actual, expected, atol=.002, rtol=2e-5)
    draws = fp.infer_fused_pairs(packet, head, hidden, time)
    assert draws.shape == packet.masked.shape
    assert torch.equal(draws[~packet.masked], packet.unary.argmax(-1)[~packet.masked])


@torch.no_grad()
def test_visible_only_and_impossible():
    packet, head, hidden, time, edge = fixture('global', 3)
    packet.masked[:] = False
    packet.unary[:] = -torch.inf
    packet.unary[..., 2] = 0.
    draws = fp.infer_fused_pairs(packet, head, hidden, time)
    assert (draws == 2).all()
    packet.masked[:] = True
    packet.unary[:, 2] = -torch.inf
    with pytest.raises(RuntimeError, match='Invalid CRF'):
        fp.infer_fused_pairs(packet, head, hidden, time)


@torch.no_grad()
def test_count_cache_invalidation():
    packet, head, hidden, time, edge = fixture('count-pmi', 3)
    first = fp.prepare_factors(packet, head)
    head.strength = .7
    second = fp.prepare_factors(packet, head)
    assert second.values.data_ptr() != first.values.data_ptr()
    head.fit([[2, 2, 3, 2, 2]])
    third = fp.prepare_factors(packet, head)
    assert third.keys.numel() == 3
    actual = fp.infer_fused_pairs(packet, head, sampling='log_marginals')
    expected = segmented_log_marginals(packet.unary, head(packet.candidate_ids), packet.masked)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('kind', ['global', 'contextual', 'count-pmi'])
@torch.no_grad()
def test_no_dense_head_call_and_generation(kind, monkeypatch):
    from chain_crf.backbone import SyntheticBackbone
    from chain_crf.generation import generate
    backbone = SyntheticBackbone(vocab_size=23, hidden_size=8, device='cuda')
    if kind == 'count-pmi':
        head = CountBigramHead(23, strength=.1).fit([[1, 2, 3, 2, 1]]).cuda()
        mode = 'count'
    else:
        mode = kind
        head = (GlobalPairHead(23, rank=17) if kind == 'global' else
                ContextualPairHead(23, 8, rank=17)).cuda()

    def forbidden(*args, **kwargs):
        raise AssertionError('Fused inference must not materialize dense head scores')

    monkeypatch.setattr(head, 'forward', forbidden)
    for sampler, sampling in [('fixed', 'joint'), ('ddpm_cache', 'marginal')]:
        result, timing = generate(backbone, head, mode, length=9, steps=3,
                                 k=3, prefix=[1, 2], device='cuda',
                                 inference='fused-pairs', segment_batch_size=2,
                                 sampler=sampler, sampling=sampling)
        assert result.shape == (1, 11)
        assert result[0, :2].tolist() == [1, 2]
        assert not result.eq(backbone.mask_id).any()


@torch.no_grad()
def test_joint_law_against_enumeration():
    # Strong positive/negative interactions exercise dependence, not just node marginals.
    packet, head, hidden, time, edge = fixture('global', 2, length=3, batch=1)
    head.left.weight.normal_(std=.8)
    head.right.weight.normal_(std=.8)
    edge = head(packet.candidate_ids)
    factors = fp.prepare_factors(packet, head)
    paths = torch.tensor(list(itertools.product(range(3), repeat=3)), device='cuda')
    scores = packet.unary[0, torch.arange(3, device='cuda'), paths].sum(-1)
    scores += edge[0, 0, paths[:, 0], paths[:, 1]] + edge[0, 1, paths[:, 1], paths[:, 2]]
    expected = scores.double().softmax(0)
    count = 20000
    rows = torch.zeros(count, dtype=torch.long, device='cuda')
    starts = rows.clone()
    lengths = torch.full_like(rows, 3)
    alpha = oracle_forward(packet.unary, edge).expand(count, -1, -1).contiguous()
    noise = torch.empty_like(alpha, dtype=torch.float64).exponential_().log_()
    draws = fp._sample_with_noise(alpha, factors, rows, starts, lengths, noise)
    bins = (draws * draws.new_tensor([9, 3, 1])).sum(-1)
    actual = torch.bincount(bins, minlength=27).double() / count
    assert (actual - expected).abs().max() < .015
