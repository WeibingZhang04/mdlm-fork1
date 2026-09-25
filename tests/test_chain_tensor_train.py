from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_chain_tensor_train import MatchedReveal, sampling_precision, validate_dimensions


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
