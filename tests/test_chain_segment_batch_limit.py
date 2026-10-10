import itertools

import pytest
import torch

from chain_crf.core import (build_candidates, chain_log_marginals,
                            chain_log_partition, chain_log_prob,
                            sample_candidate_tokens)
from chain_crf.segments import (pack_segments, sample_segmented_chain,
                               segmented_log_marginals, segmented_log_partition,
                               segmented_marginals)


from test_chain_segments import make_problem, MASKS

@pytest.mark.parametrize("cap", [1, 2, 100])
@pytest.mark.parametrize("mask_rows", MASKS)
def test_capped_exact_partition_marginals_and_clamps(cap, mask_rows):
    mask = torch.tensor(mask_rows)
    unary, edge = make_problem(mask)
    torch.testing.assert_close(
        segmented_log_partition(unary, edge, mask, segment_batch_size=cap),
        chain_log_partition(unary, edge))
    torch.testing.assert_close(
        segmented_log_marginals(unary, edge, mask, segment_batch_size=cap),
        chain_log_marginals(unary, edge))
    kwargs = dict(segment_batch_size=cap)
    draws = sample_segmented_chain(unary, edge, mask, torch.Generator().manual_seed(19), **kwargs)
    again = sample_segmented_chain(unary, edge, mask, torch.Generator().manual_seed(19), **kwargs)
    assert torch.equal(draws, again)
    assert torch.isfinite(unary.gather(-1, draws[..., None])).all()


def test_capped_gradients_match_dense():
    mask = torch.tensor([[False, True, True, False, True, False, True],
                         [True, False, False, True, True, True, False]])
    unary, edge = make_problem(mask)
    for actual, expected in [
        (segmented_log_partition(unary, edge, mask, segment_batch_size=2),
         chain_log_partition(unary, edge)),
        (segmented_marginals(unary, edge, mask, segment_batch_size=2),
         chain_log_marginals(unary, edge).exp()),
    ]:
        weights = torch.randn_like(actual)
        a = torch.autograd.grad((actual * weights).sum(), (unary, edge), retain_graph=True)
        e = torch.autograd.grad((expected * weights).sum(), (unary, edge), retain_graph=True)
        for observed, reference in zip(a, e):
            torch.testing.assert_close(observed, reference)


@pytest.mark.parametrize("operation", [segmented_log_partition, segmented_log_marginals,
                                       sample_segmented_chain])
def test_cap_applies_before_packing_and_releases_previous_chunk(monkeypatch, operation):
    import weakref
    import chain_crf.segments as segments
    mask = torch.tensor([[True, True, True, True, False, True, False, True, False, True]])
    unary, edge = make_problem(mask)
    original = segments._pack_runs
    previous = []
    shapes = []

    def checked(*args):
        assert all(ref() is None for ref in previous)
        packed = original(*args)
        assert packed.run_count <= 2
        shapes.append(tuple(packed.edge.shape))
        previous[:] = [weakref.ref(packed.unary), weakref.ref(packed.edge)]
        return packed

    monkeypatch.setattr(segments, '_pack_runs', checked)
    with torch.no_grad():
        operation(unary, edge, mask, segment_batch_size=2)
    assert shapes == [(2, 3, 3, 3), (2, 0, 3, 3)]


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_segment_batch_size(cap):
    with pytest.raises(ValueError, match="segment_batch_size"):
        sample_segmented_chain(torch.zeros(1, 1, 2), torch.zeros(1, 0, 2, 2),
                               torch.ones(1, 1, dtype=torch.bool), segment_batch_size=cap)


@pytest.mark.parametrize('sampler', ['fixed', 'ddpm_cache'])
@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
def test_generation_passes_limit_through_sampling_and_cleanup(monkeypatch, sampler, sampling):
    import chain_crf.generation as generation
    from chain_crf.heads import GlobalPairHead
    from test_chain_generation import ToyBackbone
    observed = []
    for name in ['_draw_structured', '_denoise_structured']:
        original = getattr(generation, name)
        def checked(*args, _name=name, _original=original, **kwargs):
            assert kwargs['segment_batch_size'] == 2
            observed.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(generation, name, checked)
    head = GlobalPairHead(5, 3)
    kwargs = dict(length=12, steps=4, batch_size=3, k=2, device='cpu',
                  prefix=[2,1], inference='segments', segment_batch_size=2,
                  sampler=sampler, sampling=sampling, sample_offset=200)
    if sampler == 'ddpm_cache':
        kwargs.update(sampling_eps=.2, noise_removal=True)
    first, _ = generation.generate(ToyBackbone(), head, 'global', **kwargs)
    again, _ = generation.generate(ToyBackbone(), head, 'global', **kwargs)
    assert torch.equal(first, again)
    assert not first.eq(4).any()
    assert first[:, :2].eq(torch.tensor([2,1])).all()
    assert '_draw_structured' in observed
    assert ('_denoise_structured' in observed) == (sampler == 'ddpm_cache')


def test_cli_records_limit_and_rejects_changed_resume(tmp_path, monkeypatch):
    import json
    from scripts.evaluate_chain_crf import main
    from chain_crf.heads import GlobalPairHead
    head = GlobalPairHead(17, 3)
    monkeypatch.setattr('scripts.evaluate_chain_crf.load_head', lambda args, model: (head, {}))
    output = tmp_path/'run'
    args = ['--output',str(output),'--synthetic','--device','cpu','--mode','global',
            '--length','8','--steps','3','--samples','2','--warmup','0','--k','2',
            '--sampler','ddpm_cache','--noise-removal','--inference','segments',
            '--segment-batch-size','2']
    main(args)
    assert json.loads((output/'manifest.json').read_text())['config']['segment_batch_size'] == 2
    changed=args.copy()
    changed[-1]='1'
    with pytest.raises(ValueError, match='Resume manifest'):
        main(changed+['--resume'])
