"""Offline exact-prefix full-vocabulary transfer and immutable recovery checks."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf.sparse_count import SparseCountPotential
from chain_crf.sparse_count_gpu import GPUCountPotential
import scripts.evaluate_chain_crf as shared
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


def potential(backend, strength=.25):
    counts = CountBigramHead(5, 'pmi', strength).fit([[0, 1, 2], [2, 0, 1], [3, 2, 1]]*3)
    cls = SparseCountPotential if backend == 'reference' else GPUCountPotential
    return cls.from_head(counts)


def source_rows():
    rows = []
    for i, (prefix, suffix) in enumerate([(2, 4), (2, 4), (2, 4), (3, 2), (3, 2), (1, 5)]):
        rows.append({'prefix_input_ids': [(i+j)%16 for j in range(prefix)],
                     'reference_continuation_ids': [(i+j+5)%16 for j in range(suffix)],
                     'prefix_length': prefix, 'document_id': f'article-{i//2}',
                     'chunk_id': f'chunk-{i}', 'source_split': 'validation',
                     'source_start_row': i*3, 'source_stop_row': i*3+3,
                     'token_start': 7, 'token_stop': 7+prefix+suffix})
    return rows


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def records(output):
    return [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]


def arguments(tmp_path, *, backend='reference', sampling='joint', count=6, output='run'):
    counts = tmp_path/'counts.pt'
    if not counts.exists():
        CountBigramHead(17).fit([[1, 2, 3, 4], [4, 3, 2, 1], [9, 2, 4]]).save(counts)
    source = tmp_path/'source.jsonl'
    if not source.exists():
        write_rows(source, source_rows())
    return ['--output', str(tmp_path/output), '--counts', str(counts), '--synthetic',
            '--device', 'cpu', '--continuation-file', str(source), '--steps', '3',
            '--samples', str(count), '--batch-size', '3', '--warmup', '1',
            '--sample-offset', '10000', '--backend', backend, '--sampling', sampling]


@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
@pytest.mark.parametrize('backend', ['reference', 'gpu'])
@pytest.mark.parametrize('strength', [0., .25])
def test_distinct_prefixes_clamped_every_step_and_same_backend_draws(sampling, backend, strength):
    prefixes = [[0, 1], [2, 3], [1, 0]]
    model = RecordingBackbone()
    kwargs = dict(length=5, steps=3, batch_size=3, device='cpu', prefixes=prefixes,
                  sample_offset=17, sampling=sampling)
    tokens, timing = evaluator.generate_sparse_count(model, potential(backend, strength),
                                                     inference_backend=backend, **kwargs)
    expected, _ = evaluator.generate_sparse_count(RecordingBackbone(), potential('reference', strength),
                                                  **kwargs)
    assert torch.equal(tokens, expected)
    assert tokens.shape == (3, 7) and not tokens.eq(4).any()
    assert tokens[:, :2].tolist() == prefixes
    assert all(row[:, :2].tolist() == prefixes for row in model.history)
    assert timing['generated_tokens'] == 15 and timing['prefix_length'] == 2
    for before, after in zip(model.history, model.history[1:]):
        visible = before.ne(4)
        assert torch.equal(before[visible], after[visible])


@pytest.mark.parametrize('backend', ['reference', 'gpu'])
@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
def test_shared_prefix_equivalence_and_invalid_prefix_rejection(backend, sampling):
    kwargs = dict(length=4, steps=3, batch_size=2, device='cpu', sample_offset=91,
                  sampling=sampling, inference_backend=backend)
    for prefix in ([], [2, 1]):
        a, _ = evaluator.generate_sparse_count(RecordingBackbone(), potential(backend), prefix=prefix, **kwargs)
        b, _ = evaluator.generate_sparse_count(RecordingBackbone(), potential(backend), prefixes=[prefix]*2, **kwargs)
        assert torch.equal(a, b)
    for prefixes, error in [([[4], [1]], 'absorbing masks'), ([[True], [1]], 'integer'),
                            ([[1.], [1]], 'integer'), ([[5], [1]], 'out-of-vocabulary'),
                            ([[0], [1, 2]], 'equal lengths'), ([[0]], 'batch_size')]:
        model = RecordingBackbone()
        with pytest.raises(ValueError, match=error):
            evaluator.generate_sparse_count(model, potential(backend), prefixes=prefixes, **kwargs)
        assert not model.history
    with pytest.raises(ValueError, match='not both'):
        evaluator.generate_sparse_count(RecordingBackbone(), potential(backend),
                                        prefix=[0], prefixes=[[0], [0]], **kwargs)


@pytest.mark.parametrize('backend', ['reference', 'gpu'])
@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
def test_cli_exact_ids_source_identity_variable_shapes_and_suffix_statistics(tmp_path, monkeypatch, backend, sampling):
    original = evaluator.SyntheticBackbone

    class NoEncodingTokenizer:
        def encode(self, *args, **kwargs):
            raise AssertionError('Continuation mode must never tokenize text')
        def decode(self, ids):
            return 'display only'

    def model(**kwargs):
        result = original(**kwargs)
        result.tokenizer = NoEncodingTokenizer()
        return result

    monkeypatch.setattr(evaluator, 'SyntheticBackbone', model)
    evaluator.main(arguments(tmp_path, backend=backend, sampling=sampling))
    actual = records(tmp_path/'run')
    assert [r['draw_id'] for r in actual] == list(range(10000, 10006))
    assert [r['batch_id'] for r in actual] == [0, 0, 0, 3, 3, 5]
    assert [r['batch_size'] for r in actual] == [3, 3, 3, 2, 2, 1]
    digest = hashlib.sha256((tmp_path/'source.jsonl').read_bytes()).hexdigest()
    for index, (row, source) in enumerate(zip(actual, source_rows())):
        for key in ('document_id', 'chunk_id', 'source_split', 'source_start_row',
                    'source_stop_row', 'token_start', 'token_stop', 'prefix_input_ids',
                    'reference_continuation_ids'):
            assert row[key] == source[key]
        assert row['input_index'] == index
        assert row['continuation_file_sha256'] == digest
        assert row['token_ids'][:row['prefix_length']] == source['prefix_input_ids']
        assert len(row['token_ids']) == len(source['prefix_input_ids'])+len(source['reference_continuation_ids'])
        assert row['generated_length'] == len(source['reference_continuation_ids'])
    manifest = json.loads((tmp_path/'run/manifest.json').read_text())
    assert manifest['continuations']['file_sha256'] == digest
    assert manifest['source_sha256']['scripts/evaluate_chain_crf.py']
    assert manifest['prefix_token_ids'] == []
    metrics = json.loads((tmp_path/'run/metrics.json').read_text())
    assert metrics['tokens'] == 21
    assert metrics['generated_tokens_per_second'] == pytest.approx(21/metrics['elapsed_seconds'])


@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
def test_reference_values_never_influence_generation(tmp_path, sampling):
    args = arguments(tmp_path, backend='gpu', sampling=sampling)
    evaluator.main(args)
    first = records(tmp_path/'run')
    rows = source_rows()
    for row in rows:
        row['reference_continuation_ids'] = [(v+3)%16 for v in row['reference_continuation_ids']]
    write_rows(tmp_path/'source.jsonl', rows)
    evaluator.main(arguments(tmp_path, backend='gpu', sampling=sampling, output='other'))
    assert [r['token_ids'] for r in first] == [r['token_ids'] for r in records(tmp_path/'other')]


def test_selection_offset_does_not_change_draw_id_offset_and_alias(tmp_path):
    args = arguments(tmp_path, count=2)
    args[args.index('--continuation-file')] = '--continuations-jsonl'
    evaluator.main(args+['--one-per-document', '--continuation-offset', '1'])
    actual = records(tmp_path/'run')
    assert [r['input_index'] for r in actual] == [2, 4]
    assert [r['draw_id'] for r in actual] == [10000, 10001]
    assert [r['chunk_id'] for r in actual] == ['chunk-2', 'chunk-4']


@pytest.mark.parametrize('backend', ['reference', 'gpu'])
@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
@pytest.mark.parametrize('saved_count', [1, 4, 5, 6])
def test_partial_batch_resume_replays_original_variable_shapes(tmp_path, backend, sampling, saved_count):
    args = arguments(tmp_path, backend=backend, sampling=sampling)
    evaluator.main(args)
    output = tmp_path/'run'
    expected = records(output)
    manifest_bytes = (output/'manifest.json').read_bytes()
    write_rows(output/'samples.jsonl', expected[:saved_count])
    evaluator.main(args+['--resume'])
    actual = records(output)
    assert actual[:saved_count] == expected[:saved_count]
    keys = ('sample_id', 'draw_id', 'token_ids', 'prefix_length', 'generated_length',
            'batch_id', 'batch_size', 'document_id', 'chunk_id', 'input_index',
            'continuation_file_sha256', 'manifest_sha256')
    assert [[r[k] for k in keys] for r in actual] == [[r[k] for k in keys] for r in expected]
    assert (output/'manifest.json').read_bytes() == manifest_bytes


@pytest.mark.parametrize('field', ['document_id', 'prefix', 'generated_token', 'batch_id'])
def test_resume_rejects_changed_identity_prefix_or_replayed_tokens(tmp_path, field):
    args = arguments(tmp_path)
    evaluator.main(args)
    saved = records(tmp_path/'run')[:1]
    if field == 'document_id':
        saved[0]['document_id'] = 'other-article'
    elif field == 'batch_id':
        saved[0]['batch_id'] = 4
    else:
        index = 0 if field == 'prefix' else -1
        saved[0]['token_ids'][index] = (saved[0]['token_ids'][index]+1)%16
    path = write_rows(tmp_path/'run/samples.jsonl', saved)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Stored continuation|Partial-batch replay'):
        evaluator.main(args+['--resume'])
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['input_file', 'source_code'])
def test_changed_input_or_code_rejects_resume_without_appending(tmp_path, monkeypatch, change):
    args = arguments(tmp_path)
    evaluator.main(args)
    if change == 'input_file':
        rows = source_rows()
        rows[0]['reference_continuation_ids'][0] = 14
        write_rows(tmp_path/'source.jsonl', rows)
    else:
        original = evaluator.file_sha256
        monkeypatch.setattr(evaluator, 'file_sha256',
                            lambda path: 'changed-source' if str(path) == evaluator.__file__ else original(path))
    path = tmp_path/'run/samples.jsonl'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Resume manifest'):
        evaluator.main(args+['--resume'])
    assert path.read_bytes() == before


def test_shared_suffix_scorer_full_prefix_conditioning_and_score_only_source_validation(tmp_path, monkeypatch):
    assert evaluator.score_gpt2 is shared.score_gpt2
    seen = []

    class Scorer:
        config = SimpleNamespace(n_positions=1024, _commit_hash='fixture')
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, ids):
            seen.append(ids.tolist()[0])
            logits = torch.arange(ids.shape[1]*17, dtype=torch.float32).reshape(1, ids.shape[1], 17)/50
            return SimpleNamespace(logits=logits)

    def backbone(*args, **kwargs):
        result = evaluator.SyntheticBackbone(device=kwargs['device'])
        result.provenance = {**result.provenance, 'synthetic_only': False, 'mask_id': result.mask_id}
        return result

    monkeypatch.setattr(evaluator, 'FrozenMDLM', backbone)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Scorer())))
    args = arguments(tmp_path)
    args.remove('--synthetic')
    evaluator.main(args+['--score-gpt2'])
    output = tmp_path/'run'
    actual = records(output)
    assert seen == [r['token_ids'] for r in actual]
    scores = json.loads((output/'gpt2-large.json').read_text())
    assert scores['scored_tokens'] == 21
    assert [r['scored_tokens'] for r in scores['samples']] == [4, 4, 4, 2, 2, 5]
    for row, score in zip(actual, scores['samples']):
        length = len(row['token_ids'])
        logits = torch.arange(length*17, dtype=torch.float32).reshape(length, 17)/50
        start = row['prefix_length']-1
        expected = torch.nn.functional.cross_entropy(
            logits[start:-1], torch.tensor(row['token_ids'][row['prefix_length']:]), reduction='sum')
        assert score['nll'] == pytest.approx(float(expected))
    before = (output/'gpt2-large.json').read_bytes()
    rows = source_rows()
    rows[0]['document_id'] = 'changed-article'
    write_rows(tmp_path/'source.jsonl', rows)
    with pytest.raises(ValueError, match='Continuation source differs'):
        evaluator.main(['--output', str(output), '--device', 'cpu', '--score-only'])
    assert (output/'gpt2-large.json').read_bytes() == before


def test_invalid_continuation_options_rejected_before_output(tmp_path):
    args = arguments(tmp_path)
    with pytest.raises(SystemExit):
        evaluator.main(args+['--prefix-token-ids', '1'])
    with pytest.raises(ValueError, match='Continuation offset'):
        evaluator.main(args+['--continuation-offset', '-1'])
    assert not (tmp_path/'run').exists()
