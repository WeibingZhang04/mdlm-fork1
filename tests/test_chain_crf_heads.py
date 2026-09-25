import torch

from chain_crf import (
    CountBigramHead, ContextualPairHead, GlobalPairHead, IndependentHead,
    build_candidates, chain_log_prob, gold_log_prob,
)


def test_global_is_ordered_signed_tail_neutral_and_non_dead():
    torch.manual_seed(1)
    head = GlobalPairHead(5, rank=2)
    ids = torch.tensor([[[0, 1, -1], [2, 3, -1]]])
    initial = head(ids)
    assert initial.eq(0).all()
    initial[0, 0, 0, 1].backward()
    assert head.right.weight.grad.abs().sum() > 0
    with torch.no_grad():
        head.left.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [2., -1.], [-2., 0.], [0., 0.]]))
        head.right.weight.copy_(torch.tensor([[0., 2.], [1., 0.], [-1., 1.], [3., -2.], [0., 0.]]))
    score = head(ids)
    assert score[0, 0, 0, 0] == -1
    assert score[0, 0, 0, 1] == 3
    assert score[..., -1, :].eq(0).all() and score[..., :, -1].eq(0).all()
    reversed_ids = ids.flip(1)
    assert not torch.equal(score, head(reversed_ids).transpose(-1, -2))


def test_context_initializes_from_global_and_gate_has_gradients():
    torch.manual_seed(2)
    global_head = GlobalPairHead(6, rank=3)
    torch.nn.init.normal_(global_head.right.weight)
    context = ContextualPairHead(6, hidden_size=4, rank=3, mlp_size=8).load_global(global_head)
    ids = torch.tensor([[[0, 1, -1], [2, 3, -1], [4, 5, -1]]])
    hidden = torch.randn(1, 3, 4)
    output = context(ids, hidden, torch.tensor([.6]))
    torch.testing.assert_close(output, global_head(ids))
    output.sum().backward()
    assert context.gate[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        context.gate[-1].weight.normal_()
    assert not torch.equal(context(ids, hidden, torch.tensor([.1])), context(ids, hidden, torch.tensor([.9])))
    assert context(ids, hidden)[..., -1, :].eq(0).all()


def test_independent_initially_identity_and_has_nonzero_gradient():
    torch.manual_seed(3)
    head = IndependentHead(8, hidden_size=4, rank=4)
    ids = torch.tensor([[[0, 1, -1], [2, 3, -1]]])
    output = head(ids, torch.randn(1, 2, 4), torch.tensor([.3]))
    assert output.eq(0).all()
    output.sum().backward()
    assert head.projection.weight.grad.abs().sum() > 0


def test_counts_never_cross_documents_and_lookup_is_smoothed(tmp_path):
    head = CountBigramHead(4, smoothing=.2).fit([[0, 1, 0], [2, 3], [3]])
    assert head.pair_keys.tolist() == [1, 4, 11]
    assert head.pair_counts.tolist() == [1., 1., 1.]
    assert head.left_counts.tolist() == [1., 1., 1., 0.]
    assert head.right_counts.tolist() == [1., 1., 0., 1.]
    ids = torch.tensor([[[0, 1, 2, 3, -1], [0, 1, 2, 3, -1]]])
    score = head(ids)
    pl = (head.left_counts + 1) / 7
    pr = (head.right_counts + 1) / 7
    marginal_l = .8 * head.left_counts / 3 + .2 * pl
    marginal_r = .8 * head.right_counts / 3 + .2 * pr
    expected = ((.8 / 3 + .2 * pl[0] * pr[1]) / (marginal_l[0] * marginal_r[1])).log()
    torch.testing.assert_close(score[0, 0, 0, 1], expected.float())
    expected_unseen = (.2 * pl[0] * pr[2] / (marginal_l[0] * marginal_r[2])).log()
    torch.testing.assert_close(score[0, 0, 0, 2], expected_unseen.float())
    assert score[..., -1, :].eq(0).all() and score[..., :, -1].eq(0).all()
    path = tmp_path / "counts.pt"
    head.save(path)
    loaded = CountBigramHead.load(path)
    torch.testing.assert_close(loaded(ids), score)
    conditional = CountBigramHead.load(path, mode="conditional", strength=.5)
    torch.testing.assert_close(conditional(ids)[0, 0, :4, :4], .5 * (score[0, 0, :4, :4] + marginal_r.log().float()[None]))
    conditional.strength = 1.
    torch.testing.assert_close(conditional(ids)[0, 0, :4, :4].exp().sum(-1), torch.ones(4))


def test_empty_counts_and_zero_strength_are_neutral():
    ids = torch.tensor([[[0, 1, -1], [2, 3, -1]]])
    torch.testing.assert_close(CountBigramHead(4)(ids), torch.zeros(1, 1, 3, 3), atol=1e-6, rtol=0)
    head = CountBigramHead(4, strength=0.).fit([[0, 1], [2, 3]])
    assert head(ids).eq(0).all()


def test_global_head_learns_agreement_and_disagreement():
    torch.manual_seed(13)
    # A realizable categorical chain: no backbone contextual ambiguity and no
    # candidate truncation. Two balanced correlated modes, not unary gains.
    for disagreement in (False, True):
        head = GlobalPairHead(3, rank=2)
        optimizer = torch.optim.Adam(head.parameters(), lr=.1)
        targets = torch.tensor([[0, 1], [1, 0]]) if disagreement else torch.tensor([[0, 0], [1, 1]])
        packet = build_candidates(torch.zeros(2, 2, 3), torch.full((2, 2), 2), 2, 2, gold=targets)
        for _ in range(70):
            loss = -gold_log_prob(packet, head(packet.candidate_ids)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final = -gold_log_prob(packet, head(packet.candidate_ids)).mean()
        assert final < .73, f"failed to learn pair law: NLL={final.item()}"
        assert final < 2 * torch.log(torch.tensor(2.)) - .6


def test_contextual_head_learns_opposite_joints_from_identical_unaries():
    torch.manual_seed(21)
    head = ContextualPairHead(3, hidden_size=4, rank=2, mlp_size=8)
    targets = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
    packet = build_candidates(torch.zeros(4, 2, 3), torch.full((4, 2), 2), 2, 2, gold=targets)
    context_a = torch.tensor([1., -1., 0., 0.])
    context_b = torch.tensor([-1., 1., 0., 0.])
    hidden = torch.stack((context_a, context_a, context_b, context_b))[:, None].expand(-1, 2, -1)
    optimizer = torch.optim.Adam(head.parameters(), lr=.05)
    for _ in range(160):
        loss = -gold_log_prob(packet, head(packet.candidate_ids, hidden)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = -gold_log_prob(packet, head(packet.candidate_ids, hidden)).mean()
    assert final < .75, f"contextual opposite joint was not learned: {final.item()}"
