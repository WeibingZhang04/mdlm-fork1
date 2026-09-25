"""Offline correctness checks; no released-model download or GPU required."""
import copy
import itertools
import json
import random

import pytest
import torch

from chain_crf.backbone import SyntheticBackbone
from chain_crf.core import build_candidates, gold_log_prob
from chain_crf.data import BatchStream, atomic_torch_save, canonical_hash
from chain_crf.generation import generate
from chain_crf.joint import (SyntheticTrainableMDLM, TrainableMDLM, checkpoint_payload,
    corrupt_joint, evaluate_joint, evaluation_components, make_joint_head, make_optimizer,
    models_from_checkpoint, restore_training, sequence_nll, train_update, validate_checkpoint,
    weighted_denoising_loss)
from scripts.train_chain_joint import main as train_main
from scripts.evaluate_chain_joint import main as evaluate_main


@pytest.fixture(autouse=True)
def reproducibility():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    random.seed(19)


def fixture(arm="contextual", dropout=.1):
    backbone = SyntheticTrainableMDLM(vocab_size=7, hidden_size=5, dropout=dropout)
    head = make_joint_head(arm, backbone, rank=3, mlp_size=8)
    config = {"arm": arm, "rank": 3, "mlp_size": 8, "gate_init_std": .01,
              "batch_size": 4, "length": 4, "k": 2, "synthetic_only": True}
    identity = {"config": config, "model_spec": backbone.model_spec,
                "initialization_backbone": backbone.initialization, "fixture": True}
    optimizer, scheduler = make_optimizer(backbone, head, backbone_lr=.001, head_lr=.01, warmup_steps=3)
    tokens = torch.arange(28).reshape(7, 4) % 6
    stream, rng = BatchStream(tokens, 4, 1), torch.Generator().manual_seed(2)
    return backbone, head, optimizer, scheduler, stream, rng, identity


@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_zero_pairs_recover_independent_loss_and_backbone_gradient(k):
    backbone, head, *_ = fixture(dropout=0.)
    clean = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
    x = torch.tensor([[6, 1, 6, 6], [4, 6, 2, 1]])
    t = torch.tensor([.25, .9])
    prediction = backbone(x, t)
    packet = build_candidates(prediction["log_probs"], x, 6, k, clean)
    assert head(packet.candidate_ids, prediction["hidden"], t).count_nonzero() == 0
    independent, _ = sequence_nll(prediction, x, clean, 6)
    joint, _ = sequence_nll(prediction, x, clean, 6, head=head, time=t, k=k)
    a = weighted_denoising_loss(independent, t, 4)
    b = weighted_denoising_loss(joint, t, 4)
    grad_a = torch.autograd.grad(a, tuple(backbone.parameters()), retain_graph=True)
    grad_b = torch.autograd.grad(b, tuple(backbone.parameters()))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    for first, second in zip(grad_a, grad_b):
        torch.testing.assert_close(first, second, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("k", [0, 1, 4])
def test_full_vocabulary_tail_likelihood_and_gradient_match_enumeration(k):
    logits = torch.tensor([[[1.3, .6, -.5, -1.2, .7], [.4, 1.2, -.1, -1.5, -.3]]],
                          dtype=torch.double, requires_grad=True)
    clean, x = torch.tensor([[3, 2]]), torch.tensor([[4, 4]])
    packet = build_candidates(logits.log_softmax(-1), x, 4, k, clean)
    edge = torch.randn(1, 1, k+1, k+1, dtype=torch.double) * .5
    edge[:, :, -1] = 0
    edge[:, :, :, -1] = 0
    actual = gold_log_prob(packet, edge)[0]
    normalized = logits[..., :4].log_softmax(-1)
    scores = []
    for left, right in itertools.product(range(4), repeat=2):
        ids = packet.candidate_ids[0]
        a = (ids[0] == left).nonzero()
        b = (ids[1] == right).nonzero()
        i = int(a[0, 0]) if a.numel() else k
        j = int(b[0, 0]) if b.numel() else k
        scores.append(normalized[0, 0, left] + normalized[0, 1, right] + edge[0, 0, i, j])
    scores = torch.stack(scores)
    expected = scores[3*4+2] - scores.logsumexp(0)
    a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    b = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    assert torch.isfinite(a).all() and a[..., :4].abs().min() > 0
    torch.testing.assert_close(a[..., 4], torch.zeros_like(a[..., 4]), atol=1e-15, rtol=0)


def test_all_visible_has_zero_loss_and_finite_zero_gradient():
    backbone, head, *_ = fixture(dropout=0.)
    clean, t = torch.tensor([[0, 1, 2, 3]]), torch.tensor([.001])
    for selected_head in (None, head):
        backbone.zero_grad(set_to_none=True)
        nll, _ = sequence_nll(backbone(clean, t), clean, clean, 6, head=selected_head, time=t, k=2)
        loss = weighted_denoising_loss(nll, t, 4)
        assert float(loss.detach()) == 0
        loss.backward()
        for parameter in backbone.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all() and parameter.grad.count_nonzero() == 0


def test_exact_time_weight_and_effective_batch_denominator():
    nll = torch.tensor([3., 6.], requires_grad=True)
    loss = weighted_denoising_loss(nll, torch.tensor([.25, .5]), length=3, effective_batch_size=4)
    assert float(loss.detach()) == 2.
    loss.backward()
    torch.testing.assert_close(nll.grad, torch.tensor([1/3, 1/6]))


def test_corruption_allows_all_visible_and_uses_effective_batch_strata():
    clean = torch.tensor([[1]])
    # Find an ordinary zero-mask draw, then check exactly one t/one mask RNG
    # draw were consumed: no conditional redraw is permitted.
    for seed in range(100):
        rng = torch.Generator().manual_seed(seed)
        x, active, t = corrupt_joint(clean, 6, rng)
        if not active.any():
            expected = torch.Generator().manual_seed(seed)
            torch.rand(1, generator=expected)
            torch.rand((1, 1), generator=expected)
            assert torch.equal(rng.get_state(), expected.get_state())
            assert torch.equal(clean, x)
            break
    else:
        pytest.fail("No all-visible draw found")
    _, _, t = corrupt_joint(torch.ones(4, 4, dtype=torch.long), 6, torch.Generator().manual_seed(1))
    strata = ((t-.001)/.999*4).floor().long()
    assert strata.tolist() == [0, 1, 2, 3]


def test_microbatch_accumulation_matches_whole_batch_update():
    first = fixture(dropout=0.)
    second = fixture(dropout=0.)
    first[1].right.weight.data.normal_(std=.02)
    second[0].load_state_dict(first[0].state_dict())
    second[1].load_state_dict(first[1].state_dict())
    clean = first[4].next()
    corruption = corrupt_joint(clean, 6, torch.Generator().manual_seed(77))
    metrics = []
    for setup, size in ((first, 1), (second, 4)):
        model, head, opt, sched, _, rng, _ = setup
        metrics.append(train_update(model, head, opt, sched, clean, rng, k=2,
                                    microbatch_size=size, corruption=corruption))
    assert metrics[0]["weighted_loss"] == pytest.approx(metrics[1]["weighted_loss"], rel=1e-6)
    for a, b in zip(list(first[0].parameters())+list(first[1].parameters()),
                    list(second[0].parameters())+list(second[1].parameters())):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=2e-5)


def test_fresh_zero_head_receives_contextual_gradient_and_both_modules_update():
    backbone, head, opt, sched, stream, rng, _ = fixture(dropout=0.)
    before_backbone = backbone.projection.detach().clone()
    before_right = head.right.weight.detach().clone()
    for _ in range(2):
        train_update(backbone, head, opt, sched, stream.next(), rng, k=6)
    assert not torch.equal(backbone.projection, before_backbone)
    assert not torch.equal(head.right.weight, before_right)
    assert head.gate[-1].weight.grad.abs().sum() > 0
    assert backbone.embeddings.grad.abs().sum() > 0
    frozen = SyntheticBackbone()
    output = frozen(torch.tensor([[16, 1]]), torch.tensor([.5]))
    assert not output["log_probs"].requires_grad and not output["hidden"].requires_grad


def test_zero_pairs_nonconstant_gate_avoids_balanced_context_dead_gradient():
    model = SyntheticTrainableMDLM(vocab_size=3, hidden_size=5, dropout=0.)
    head = make_joint_head("contextual", model, rank=3, mlp_size=8)
    clean = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
    x, t = torch.full_like(clean, 2), torch.full((4,), .5)
    context = torch.tensor([[1., 2., 3., -1., 0.], [-2., 1., 0., 3., -1.]])
    hidden = context[torch.tensor([0, 0, 1, 1])][:, None].expand(-1, 2, -1)
    lp = torch.tensor([-.69314718, -.69314718, -torch.inf]).expand(4, 2, 3)
    prediction = {"log_probs": lp, "hidden": hidden}
    nll, _ = sequence_nll(prediction, x, clean, 2, head=head, time=t, k=2)
    nll.mean().backward()
    assert head.right.weight.grad.abs().sum() > 1e-7
    head.zero_grad(set_to_none=True)
    with torch.no_grad():
        head.gate[-1].weight.zero_()
    nll, _ = sequence_nll(prediction, x, clean, 2, head=head, time=t, k=2)
    nll.mean().backward()
    torch.testing.assert_close(head.right.weight.grad, torch.zeros_like(head.right.weight.grad), atol=1e-9, rtol=0)


def test_native_wrapper_enables_encoder_gradients_and_passes_zero_time():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 5)
            self.output = torch.nn.Linear(5, 7)
            self.dropout = torch.nn.Dropout(.1)

        def encode(self, tokens, time):
            assert torch.equal(time, torch.zeros_like(time))
            return self.dropout(self.embedding(tokens)), time

        def decode(self, hidden, conditioning):
            return self.output(hidden)

    encoder = Encoder().requires_grad_(False).eval()
    spec = {"kind": "fixture_encoder", "vocab_size": 7, "hidden_size": 5, "mask_id": 6}
    model = TrainableMDLM(encoder, None, spec, {"synthetic_only": True}).train()
    assert model.encoder.training and all(p.requires_grad for p in model.parameters())
    prediction = model(torch.tensor([[6, 1]]), torch.tensor([.7]))
    assert prediction["hidden"].requires_grad and prediction["log_probs"].requires_grad
    assert prediction["log_probs"][..., 6].isneginf().all()
    loss = -prediction["log_probs"][0, 0, 2]
    loss.backward()
    assert model.encoder.embedding.weight.grad.abs().sum() > 0
    assert model.encoder.output.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", ["independent", "contextual"])
def test_coupled_checkpoint_roundtrip_and_exact_resume(tmp_path, arm):
    backbone, head, opt, sched, stream, rng, identity = fixture(arm)
    train_update(backbone, head, opt, sched, stream.next(), rng, k=2)
    path = tmp_path/"joint.pt"
    atomic_torch_save(checkpoint_payload(backbone, head, opt, sched, stream, rng,
                                         step=1, identity=identity), path)
    payload = torch.load(path, weights_only=True)
    expected_model, expected_head = models_from_checkpoint(payload)
    x, t = torch.tensor([[6, 1, 6, 2]]), torch.tensor([.5])
    backbone.eval()
    torch.testing.assert_close(expected_model(x, t)["log_probs"], backbone(x, t)["log_probs"], atol=0, rtol=0)
    if head is not None:
        for key in head.state_dict():
            torch.testing.assert_close(head.state_dict()[key], expected_head.state_dict()[key], atol=0, rtol=0)
    # Model construction consumes RNG; explicitly restore the checkpoint
    # before defining the uninterrupted reference continuation.
    restore_training(payload, backbone, head, opt, sched, stream, rng, identity)
    expected_metrics = train_update(backbone, head, opt, sched, stream.next(), rng, k=2)
    expected = copy.deepcopy((backbone.state_dict(), head.state_dict() if head is not None else None,
                              opt.state_dict(), sched.state_dict(), rng.get_state(), torch.get_rng_state()))
    fresh = fixture(arm)
    model2, head2, opt2, sched2, stream2, rng2, identity2 = fresh
    # Optimizer.load_state_dict may adopt CPU momentum tensors by reference.
    # Restore the immutable on-disk snapshot, not the payload used by the
    # already-updated reference optimizer.
    payload = torch.load(path, weights_only=True)
    assert restore_training(payload, model2, head2, opt2, sched2, stream2, rng2, identity2) == (1, None)
    actual_metrics = train_update(model2, head2, opt2, sched2, stream2.next(), rng2, k=2)
    assert expected_metrics == actual_metrics
    for key, value in expected[0].items():
        torch.testing.assert_close(model2.state_dict()[key], value, atol=0, rtol=0)
    if head2 is not None:
        for key, value in expected[1].items():
            torch.testing.assert_close(head2.state_dict()[key], value, atol=0, rtol=0)
    assert expected[3] == sched2.state_dict()
    assert torch.equal(expected[4], rng2.get_state())
    assert torch.equal(expected[5], torch.get_rng_state())
    for key, state in expected[2]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, opt2.state_dict()["state"][key][name], atol=0, rtol=0)


def test_coupled_checkpoint_rejects_shape_identity_and_schema_changes():
    model, head, opt, sched, stream, rng, identity = fixture()
    payload = checkpoint_payload(model, head, opt, sched, stream, rng, step=0, identity=identity)
    with pytest.raises(ValueError, match="coupled"):
        validate_checkpoint({"head": {}})
    invalid = copy.deepcopy(payload)
    invalid["config"]["k"] = 4
    with pytest.raises(ValueError, match="checksum"):
        validate_checkpoint(invalid)
    invalid = copy.deepcopy(payload)
    invalid["backbone_state"]["projection"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="shape/dtype"):
        models_from_checkpoint(invalid)
    invalid = copy.deepcopy(identity)
    invalid["fixture"] = False
    with pytest.raises(ValueError, match="Resume identity"):
        restore_training(payload, model, head, opt, sched, stream, rng, invalid)
    invalid = copy.deepcopy(payload)
    invalid["pair_state"] = None
    with pytest.raises(ValueError, match="arm/pair"):
        models_from_checkpoint(invalid)


def test_evaluation_controls_keep_same_tuned_backbone_and_correct_sampler():
    model, head, *_ = fixture(dropout=0.)
    model.eval()
    assert evaluation_components(model, head, "joint") == (head, "contextual", "joint")
    assert evaluation_components(model, head, "own-marginal") == (head, "contextual", "marginal")
    assert evaluation_components(model, head, "pair-disabled") == (None, "backbone", "joint")
    with pytest.raises(ValueError, match="no CRF"):
        evaluation_components(model, None, "own-marginal")
    for control in ("joint", "own-marginal", "pair-disabled"):
        selected, mode, sampling = evaluation_components(model, head, control)
        tokens, _ = generate(model, selected, mode, sampling=sampling, length=4,
                              steps=2, batch_size=2, device="cpu", k=2, prefix=[1])
        assert torch.all(tokens[:, 0] == 1) and not torch.any(tokens == 6)


def test_development_is_rng_neutral_and_restores_training_mode():
    model, head, _, _, stream, _, _ = fixture()
    model.train()
    head.train()
    before = torch.get_rng_state().clone()
    result = evaluate_joint(model, head, stream.tokens, k=2, batch_size=4)
    assert result["weighted_loss"] > 0
    assert model.training and head.training
    assert torch.equal(before, torch.get_rng_state())


def cli_args(data, output, arm="contextual"):
    return ["--data", str(data), "--output", str(output), "--arm", arm, "--synthetic-backbone",
            "--synthetic-vocab-size", "7", "--device", "cpu", "--length", "4", "--k", "2",
            "--rank", "3", "--mlp-size", "8", "--batch-size", "4", "--microbatch-size", "1",
            "--steps", "2", "--eval-every", "2", "--save-every", "1", "--threads", "1"]


def test_cli_matched_arms_resume_and_evaluation_partial_batch(tmp_path):
    data = tmp_path/"data"
    rng = torch.Generator().manual_seed(712)
    for split, size in (("train", 8), ("dev", 4)):
        atomic_torch_save({"tokens": torch.randint(0, 6, (size, 4), generator=rng),
                          "document_ids": [f"{split}-{i}" for i in range(size)]}, data/f"{split}.pt")
    train_main(cli_args(data, tmp_path/"contextual"))
    train_main(cli_args(data, tmp_path/"independent", "independent"))
    context = torch.load(tmp_path/"contextual/last.pt", weights_only=True)
    baseline = torch.load(tmp_path/"independent/last.pt", weights_only=True)
    assert torch.equal(context["training"]["mask_rng"], baseline["training"]["mask_rng"])
    assert torch.equal(context["training"]["stream"]["order"], baseline["training"]["stream"]["order"])
    assert torch.equal(context["training"]["torch_rng"], baseline["training"]["torch_rng"])
    args = cli_args(data, tmp_path/"contextual")
    args[args.index("--steps")+1] = "3"
    checkpoint = tmp_path/"contextual/last.pt"
    train_main(args + ["--resume", str(checkpoint)])
    assert torch.load(checkpoint, weights_only=True)["trained_tokens"] == 48
    for control in ("joint", "own-marginal", "pair-disabled"):
        output = tmp_path/control
        evargs = ["--checkpoint", str(checkpoint), "--output", str(output), "--control", control,
                  "--device", "cpu", "--length", "4", "--steps", "2", "--samples", "5",
                  "--batch-size", "3", "--sample-offset", "10000", "--warmup", "0", "--threads", "1"]
        evaluate_main(evargs)
        path = output/"samples.jsonl"
        expected = [json.loads(row) for row in path.read_text().splitlines()]
        for retained in (1, 4):
            path.write_text("".join(json.dumps(row)+"\n" for row in expected[:retained]))
            evaluate_main(evargs+["--resume"])
            actual = [json.loads(row) for row in path.read_text().splitlines()]
            assert [(r["draw_id"], r["token_ids"]) for r in actual] == [(r["draw_id"], r["token_ids"]) for r in expected]
        assert [r["draw_id"] for r in expected] == list(range(10000, 10005))
        manifest = json.loads((output/"manifest.json").read_text())
        assert manifest["synthetic_only"] and manifest["training_step"] == 3
