"""Opt-in segmented generation, exact-ID replay, and bounded CUDA smoke tests.

Segmented joint draws need not equal unsplit draws for the same seed. Tests
compare probabilities and within-configuration replay, not cross-backend RNG.
CUDA memory checks use small synthetic tensors, not production memory claims.
"""
import json

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf import sparse_count as reference
from chain_crf import sparse_count_segments as segments
import scripts.evaluate_chain_sparse_count as evaluator


class RecordingBackbone:
    vocab_size = 5
    mask_id = 4

    def __init__(self):
        self.history = []

    def __call__(self, tokens, time):
        self.history.append(tokens.clone())
        logits = torch.tensor([0., -.4, -.8, -1.1, 2.], device=tokens.device)
        return {'log_probs': logits.log_softmax(-1).expand(*tokens.shape, 5)}


def potential(strength=.25, mode='pmi', device='cpu'):
    head = CountBigramHead(5, mode, strength).fit([[0, 1, 2], [2, 0, 1], [3, 2, 1]]*3)
    return segments.SegmentedCountPotential.from_head(head.to(device))


def source_rows():
    rows = []
    for index, (prefix, suffix) in enumerate([(2, 5), (2, 5), (2, 5), (3, 3), (3, 3), (1, 4)]):
        rows.append({'prefix_input_ids': [(index+j)%16 for j in range(prefix)],
                     'reference_continuation_ids': [(index+j+5)%16 for j in range(suffix)],
                     'prefix_length': prefix, 'document_id': f'article-{index//2}',
                     'chunk_id': f'chunk-{index}', 'source_split': 'validation',
                     'source_start_row': index*3, 'source_stop_row': index*3+3,
                     'token_start': 7, 'token_stop': 7+prefix+suffix})
    return rows


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def records(output):
    return [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]


def arguments(tmp_path, *, sampling='joint', continuation=False, output='run'):
    counts = tmp_path/'counts.pt'
    if not counts.exists():
        CountBigramHead(17).fit([[1, 2, 3, 4], [4, 3, 2, 1], [9, 2, 4]]).save(counts)
    args = ['--output', str(tmp_path/output), '--counts', str(counts), '--synthetic',
            '--device', 'cpu', '--length', '7', '--steps', '4', '--samples', '6',
            '--batch-size', '3', '--warmup', '1', '--sample-offset', '31000',
            '--backend', 'segments', '--sampling', sampling]
    if continuation:
        source = tmp_path/'source.jsonl'
        if not source.exists():
            write_rows(source, source_rows())
        args += ['--continuation-file', str(source)]
    else:
        args += ['--prefix-token-ids', '1', '2']
    return args


@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
@pytest.mark.parametrize('strength', [0., .25])
@pytest.mark.parametrize('budget', [None, 1, 7])
def test_generation_clamps_law_and_budget_forwarding(monkeypatch, sampling, strength, budget):
    model = RecordingBackbone()
    prepared = potential(strength)
    prefixes = [[0, 1], [2, 3], [1, 0]]
    original = (segments.sample_sparse_chain if sampling == 'joint'
                else segments.sparse_chain_marginals)
    marginal = segments.sparse_chain_marginals
    calls = []

    def checked(unary, model, *args, **kwargs):
        calls.append(kwargs['max_chunk_tokens'])
        actual = marginal(unary, model, max_chunk_tokens=kwargs['max_chunk_tokens'])
        expected = reference.sparse_chain_marginals(unary, model)
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)
        if strength == 0:
            torch.testing.assert_close(actual, unary.softmax(-1), atol=3e-13, rtol=3e-13)
        return original(unary, model, *args, **kwargs)

    monkeypatch.setattr(segments, 'sample_sparse_chain' if sampling == 'joint' else 'sparse_chain_marginals', checked)
    kwargs = dict(length=7, steps=4, batch_size=3, prefixes=prefixes, device='cpu',
                  sample_offset=29, sampling=sampling, inference_backend='segments', max_chunk_tokens=budget)
    tokens, timing = evaluator.generate_sparse_count(model, prepared, **kwargs)
    again, _ = evaluator.generate_sparse_count(RecordingBackbone(), prepared, **kwargs)
    assert torch.equal(tokens, again)
    assert calls == [512 if budget is None else budget]*8
    assert tokens[:, :2].tolist() == prefixes and not tokens.eq(4).any()
    assert timing['backbone_calls'] == 4 and timing['generated_tokens'] == 21
    for before, after in zip(model.history, model.history[1:]):
        assert torch.equal(before[before.ne(4)], after[before.ne(4)])


@pytest.mark.parametrize('continuation', [False, True])
@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
@pytest.mark.parametrize('saved_count', [1, 4, 6])
def test_manifest_and_partial_batch_resume(tmp_path, continuation, sampling, saved_count):
    args = arguments(tmp_path, sampling=sampling, continuation=continuation)+['--max-chunk-tokens', '2']
    evaluator.main(args)
    output = tmp_path/'run'
    expected = records(output)
    manifest_bytes = (output/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['config']['max_chunk_tokens'] == manifest['method']['max_chunk_tokens'] == 2
    assert manifest['method']['inference_backend'] == 'segments'
    assert 'not seed-by-seed' in manifest['method']['sampling_equivalence']
    for filename in ('sparse_count.py', 'sparse_count_gpu.py', 'sparse_count_segments.py'):
        assert manifest['source_sha256']['chain_crf/'+filename]
    write_rows(output/'samples.jsonl', expected[:saved_count])
    evaluator.main(args+['--resume'])
    actual = records(output)
    assert actual[:saved_count] == expected[:saved_count]
    keys = ['sample_id', 'draw_id', 'token_ids', 'prefix_length', 'generated_length',
            'batch_id', 'batch_size', 'manifest_sha256']
    if continuation:
        keys += ['document_id', 'chunk_id', 'input_index', 'prefix_input_ids',
                 'reference_continuation_ids', 'continuation_file_sha256']
        assert [r['batch_size'] for r in actual] == [3, 3, 3, 2, 2, 1]
        for row, source in zip(actual, source_rows()):
            assert row['token_ids'][:row['prefix_length']] == source['prefix_input_ids']
            assert row['reference_continuation_ids'] == source['reference_continuation_ids']
    assert [[r[k] for k in keys] for r in actual] == [[r[k] for k in keys] for r in expected]
    assert (output/'manifest.json').read_bytes() == manifest_bytes
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['tokens'] == (25 if continuation else 42)


@pytest.mark.parametrize('change', ['budget', 'backend', 'source', 'input', 'identity'])
def test_resume_rejects_changed_identity_before_appending(tmp_path, monkeypatch, change):
    args = arguments(tmp_path, continuation=True)
    evaluator.main(args)
    output = tmp_path/'run'
    assert json.loads((output/'manifest.json').read_text())['config']['max_chunk_tokens'] == 512
    saved = records(output)[:1]
    write_rows(output/'samples.jsonl', saved)
    if change == 'budget':
        args += ['--max-chunk-tokens', '511']
    elif change == 'backend':
        args += ['--backend', 'gpu']
    elif change == 'source':
        original = evaluator.file_sha256
        monkeypatch.setattr(evaluator, 'file_sha256', lambda path:
            'changed' if str(path).endswith('chain_crf/sparse_count_segments.py') else original(path))
    elif change == 'input':
        rows = source_rows()
        rows[0]['reference_continuation_ids'][0] = 14
        write_rows(tmp_path/'source.jsonl', rows)
    else:
        saved[0]['document_id'] = 'wrong-article'
        write_rows(output/'samples.jsonl', saved)
    before = (output/'samples.jsonl').read_bytes()
    with pytest.raises(ValueError, match='Resume manifest|Stored continuation'):
        evaluator.main(args+['--resume'])
    assert (output/'samples.jsonl').read_bytes() == before


@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
def test_reference_suffix_values_do_not_enter_segmented_generation(tmp_path, sampling):
    args = arguments(tmp_path, continuation=True, sampling=sampling)
    evaluator.main(args)
    expected = records(tmp_path/'run')
    rows = source_rows()
    for row in rows:
        row['reference_continuation_ids'] = [(v+3)%16 for v in row['reference_continuation_ids']]
    write_rows(tmp_path/'source.jsonl', rows)
    evaluator.main(arguments(tmp_path, continuation=True, sampling=sampling, output='other'))
    assert [r['token_ids'] for r in expected] == [r['token_ids'] for r in records(tmp_path/'other')]


def test_defaults_and_invalid_budget_do_not_change_other_backends(tmp_path):
    args = arguments(tmp_path)
    for backend in ('reference', 'gpu'):
        parsed = evaluator._arguments(args+['--backend', backend])
        assert parsed.max_chunk_tokens is None
        with pytest.raises(ValueError, match='requires --backend segments'):
            evaluator.main(args+['--backend', backend, '--max-chunk-tokens', '512'])
    implicit = args.copy()
    i = implicit.index('--backend')
    del implicit[i:i+2]
    assert evaluator._arguments(implicit).backend == 'reference'
    for budget in ('0', '-1'):
        with pytest.raises(ValueError, match='positive integer'):
            evaluator.main(args+['--max-chunk-tokens', budget])
    for budget in (False, 0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            evaluator.generate_sparse_count(RecordingBackbone(), potential(), device='cpu',
                inference_backend='segments', max_chunk_tokens=budget)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mode', ['pmi', 'conditional'])
@pytest.mark.parametrize('strength', [0., .25])
@pytest.mark.parametrize('budget', [1, 7, 512])
def test_partition_marginal_device_parity_and_support(device, mode, strength, budget):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; no GPU execution in CPU validation')
    model = potential(strength, mode, device)
    unary = torch.randn(4, 7, 5, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(118)).to(device)
    for row in range(4):
        for position in range(row, 7, row+1):
            unary[row, position] = -torch.inf
            unary[row, position, (row+position)%4] = .7
    unary[..., 4] = -torch.inf
    torch.testing.assert_close(segments.sparse_chain_log_partition(unary, model, max_chunk_tokens=budget),
                               reference.sparse_chain_log_partition(unary, model), atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(segments.sparse_chain_marginals(unary, model, max_chunk_tokens=budget),
                               reference.sparse_chain_marginals(unary, model), atol=5e-13, rtol=5e-13)
    sample = segments.sample_sparse_chain(unary, model,
        generator=torch.Generator(device=device).manual_seed(71), max_chunk_tokens=budget)
    assert torch.isfinite(unary.gather(-1, sample[..., None])).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA memory smoke requires an idle GPU')
@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
def test_cuda_chunk_budget_and_small_tensor_peak_memory(monkeypatch, record_property, sampling):
    # A bounded engineering smoke, not a benchmark at the publication workload.
    vocab, batch, length = 257, 4, 128
    head = CountBigramHead(vocab, 'pmi', .25).fit([list(range(vocab)), list(range(vocab-1, -1, -1))])
    model = segments.SegmentedCountPotential.from_head(head.to('cuda'))
    unary = torch.zeros(batch, length, vocab, device='cuda', dtype=torch.float64)
    unary[:, 1::2] = -torch.inf
    unary[:, 1::2, 3] = 0.
    original = segments._chunks
    seen = []

    def bounded(*args, **kwargs):
        for batches, positions, values in original(*args, **kwargs):
            seen.append(values.shape[0]*values.shape[1])
            assert seen[-1] <= 16 or values.shape[0] == 1
            yield batches, positions, values

    monkeypatch.setattr(segments, '_chunks', bounded)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    function = segments.sample_sparse_chain if sampling == 'joint' else segments.sparse_chain_marginals
    result = function(unary, model, max_chunk_tokens=16)
    torch.cuda.synchronize()
    additional_peak = torch.cuda.max_memory_allocated()-baseline
    record_property('additional_peak_allocated_bytes', additional_peak)
    assert seen and 0 < additional_peak < 128*1024**2
    assert torch.isfinite(result).all()
