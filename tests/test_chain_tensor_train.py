from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_chain_tensor_train import (MatchedReveal, sampling_precision,
                                                 validate_dimensions, validate_sample_records)


@pytest.mark.parametrize("length", [256, 1024])
@pytest.mark.parametrize("steps", [8, 16, 32, 64, 128])
def test_official_k_means_tokens_per_step_not_number_of_steps(length, steps):
    assert validate_dimensions(length, steps, 64, 1) * steps == length


def test_nondivisible_steps_are_rejected_before_upstream_minus_one_indices():
    with pytest.raises(ValueError):
        validate_dimensions(17, 8, 1, 1)


def test_matched_reveal_uses_same_chain_sets_and_original_position_order():
    order = torch.randperm(16, generator=torch.Generator().manual_seed(1729 + 5))
    reveal = MatchedReveal(16, 1, 5, 4)
    x = torch.full((1, 16), 99)
    for step in range(4):
        selected = reveal("random", x, 4, 99)
        torch.testing.assert_close(selected[0], order[4*step:4*step+4].sort().values)
        x.scatter_(1, selected, 0)
    assert x.eq(0).all() and reveal.cursor == 16
    with pytest.raises(ValueError):
        reveal("random", x, 4, 99)


def test_float64_intervention_only_changes_sampler_input_dtype_and_restores():
    seen = []
    def sample(logits, temperature=1.):
        seen.append((logits.dtype, temperature))
        return logits.argmax(-1)
    module = SimpleNamespace(sample=sample)
    x = torch.tensor([[1., 2.]])
    with sampling_precision([module], "float64"):
        assert module.sample(x, temperature=.7).item() == 1
    assert module.sample is sample
    assert seen == [(torch.float64, .7)]
    with sampling_precision([module], "native"):
        module.sample(x)
    assert seen[-1][0] == torch.float32


def test_sampler_patch_restores_on_error():
    module = SimpleNamespace(sample=lambda logits, temperature=1.: logits)
    original = module.sample
    with pytest.raises(RuntimeError):
        with sampling_precision([module], "float64"):
            raise RuntimeError("fixture")
    assert module.sample is original


def test_draw_range_resume_validation():
    records = [{"sample_id": i, "draw_id": 1000 + i} for i in range(3)]
    validate_sample_records(records, sample_offset=1000, samples=4)
    with pytest.raises(ValueError, match="draw IDs"):
        validate_sample_records(records, sample_offset=0, samples=4)
    with pytest.raises(ValueError, match="sample IDs"):
        validate_sample_records(records[::-1], sample_offset=1000, samples=4)
    with pytest.raises(ValueError, match="sample IDs"):
        validate_sample_records(records, sample_offset=1000, samples=2)
    with pytest.raises(ValueError, match="draw IDs"):
        validate_sample_records([{"sample_id": 0}], sample_offset=0, samples=1)
    with pytest.raises(ValueError, match="nonnegative"):
        validate_sample_records([], sample_offset=-1, samples=1)


def test_cli_offset_reaches_sampler_and_disjoint_warmup(tmp_path, monkeypatch):
    import json
    from scripts import evaluate_chain_tensor_train as wrapper

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.tokenizer = SimpleNamespace(decode=lambda ids: " ".join(map(str, ids)))

    def load(args):
        generation = SimpleNamespace(length=args.length, batch_size=args.batch_size)
        return Fixture(), SimpleNamespace(generation=generation), None, {"fixture": True}

    calls = []
    def generate(model, config, upstream, *, sample_offset, sampling, schedule):
        calls.append((sample_offset, config.generation.batch_size, sampling, schedule))
        tokens = torch.full((config.generation.batch_size, config.generation.length), sample_offset)
        return tokens, {"elapsed_seconds": 1., "backbone_calls": 4,
                        "reported_steps": 4., "schedule_sha256": None}

    monkeypatch.setattr(wrapper, "load_official", load)
    monkeypatch.setattr(wrapper, "generate_batch", generate)
    output = tmp_path / "draws"
    argv = ["--source-root", str(tmp_path), "--checkpoint", str(tmp_path / "unused.pt"),
            "--cache-root", str(tmp_path), "--output", str(output), "--length", "8",
            "--steps", "4", "--samples", "5", "--sample-offset", "17",
            "--batch-size", "3", "--device", "cpu"]
    wrapper.main(argv)
    assert calls == [(22, 3, "native", "native"), (17, 3, "native", "native"),
                     (20, 2, "native", "native")]
    records = [json.loads(x) for x in (output / "samples.jsonl").read_text().splitlines()]
    assert [r["sample_id"] for r in records] == [0, 1, 2, 3, 4]
    assert [r["draw_id"] for r in records] == [17, 18, 19, 20, 21]
    calls.clear()
    wrapper.main(argv + ["--resume"])
    assert calls == [(22, 3, "native", "native")]
    calls.clear()
    (output / "samples.jsonl").write_text(json.dumps(records[0]) + "\n")
    wrapper.main(argv + ["--resume"])
    assert calls == [(22, 3, "native", "native"), (17, 3, "native", "native"),
                     (20, 2, "native", "native")]
    replayed = [json.loads(x) for x in (output / "samples.jsonl").read_text().splitlines()]
    assert replayed == records
    # Corrupted or numerically divergent replay must not append or rewrite any output.
    broken = dict(records[0], token_ids=[-1] * 8)
    (output / "samples.jsonl").write_text(json.dumps(broken) + "\n")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="saved prefix"):
        wrapper.main(argv + ["--resume"])
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
