"""Offline CPU integration checks; no checkpoint download or external scoring."""
import json

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf.generation import generate
from chain_crf.sparse_count import SparseCountPotential, sparse_chain_marginals
from scripts.evaluate_chain_sparse_count import (
    full_vocabulary_unary, generate_sparse_count, main,
)


class ToyBackbone:
    vocab_size = 5
    mask_id = 4

    def __init__(self):
        self.history = []

    def __call__(self, tokens, t):
        self.history.append(tokens.clone())
        logits = torch.tensor([0., -.4, -.8, -1.1, 3.])
        return {'log_probs': logits.log_softmax(-1).expand(*tokens.shape, 5),
                'hidden': torch.zeros(*tokens.shape, 2)}


def potential(strength=.5, mode='pmi'):
    head = CountBigramHead(5, mode, strength).fit([[0, 1, 2], [2, 0, 1]]*5)
    return SparseCountPotential.from_head(head)


def test_zero_strength_recovers_full_vocab_backbone_marginals_and_clamps():
    tokens = torch.tensor([[4, 1, 4, 2]])
    pred = ToyBackbone()(tokens, torch.ones(1))
    unary = full_vocabulary_unary(pred['log_probs'], tokens, 4, temperature=.8)
    marginals = sparse_chain_marginals(unary, potential(0.))
    torch.testing.assert_close(marginals, unary.softmax(-1), atol=2e-14, rtol=2e-14)
    assert torch.isneginf(unary[..., 4]).all()
    assert marginals[0, 1, 1] == 1 and marginals[0, 3, 2] == 1
    assert marginals[0, 0, :4].gt(0).all()  # Full support, no top-K.


@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
@pytest.mark.parametrize('strength', [0., .1])
def test_generation_schedule_prefix_and_reproducibility(sampling, strength):
    model = ToyBackbone()
    kwargs = dict(length=7, steps=4, batch_size=2, sampling=sampling,
                  prefix=[2, 1], device='cpu', sample_offset=103)
    tokens, timing = generate_sparse_count(model, potential(strength), **kwargs)
    assert tokens.shape == (2, 9) and not tokens.eq(4).any()
    assert tokens[:, :2].eq(torch.tensor([2, 1])).all()
    assert timing['backbone_calls'] == 4
    assert timing['elapsed_seconds'] >= timing['backbone_seconds']+timing['sampling_seconds']
    again, _ = generate_sparse_count(ToyBackbone(), potential(strength), **kwargs)
    assert torch.equal(tokens, again)
    reference = ToyBackbone()
    generate(reference, length=7, steps=4, batch_size=2, prefix=[2, 1],
             device='cpu', sample_offset=103)
    for current, previous in zip(model.history, reference.history):
        assert torch.equal(current.eq(4), previous.eq(4))
    for before, after in zip(model.history, model.history[1:]):
        visible = before.ne(4)
        assert torch.equal(before[visible], after[visible])


def test_skip_empty_steps_and_reject_bad_prefix():
    _, timing = generate_sparse_count(ToyBackbone(), potential(), length=3, steps=128, device='cpu')
    assert timing['backbone_calls'] == 3
    with pytest.raises(ValueError, match='Prefix'):
        generate_sparse_count(ToyBackbone(), potential(), prefix=[4], device='cpu')


def arguments(output, counts, offset=100):
    return ['--output', str(output), '--counts', str(counts), '--synthetic', '--device', 'cpu',
            '--length', '7', '--steps', '3', '--samples', '5', '--batch-size', '2',
            '--warmup', '0', '--sample-offset', str(offset), '--prefix-token-ids', '1', '2']


def make_counts(tmp_path):
    path = tmp_path/'counts.pt'
    CountBigramHead(17).fit([[1, 2, 3, 4, 5], [5, 3, 2, 1], [7, 8, 9]]).save(path)
    return path


def records(output):
    return [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]


def test_cli_manifest_ids_setup_and_partial_batch_resume(tmp_path):
    counts = make_counts(tmp_path)
    output = tmp_path/'run'
    args = arguments(output, counts)
    main(args)
    original = records(output)
    assert [row['sample_id'] for row in original] == list(range(5))
    assert [row['draw_id'] for row in original] == list(range(100, 105))
    assert [row['batch_id'] for row in original] == [0, 0, 2, 2, 4]
    manifest_bytes = (output/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['counts_sha256']
    assert manifest['source_sha256']['chain_crf/sparse_count.py']
    assert manifest['backbone_identity_sha256']
    assert manifest['method']['inference_dtype'] == 'float64'
    assert all(row['token_ids'][:2] == [1, 2] for row in original)
    # Simulate interruption after only the first record of a two-record batch.
    (output/'samples.jsonl').write_text(json.dumps(original[0])+'\n')
    main(args+['--resume'])
    resumed = records(output)
    assert [(r['draw_id'], r['token_ids']) for r in resumed] == [
        (r['draw_id'], r['token_ids']) for r in original]
    assert resumed[0] == original[0]
    assert (output/'manifest.json').read_bytes() == manifest_bytes
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['samples'] == 5 and metrics['tokens'] == 35
    assert metrics['setup_invocations'] == 2
    assert metrics['static_potential_setup_seconds_first_invocation'] >= 0
    assert metrics['seconds_per_sample'] == pytest.approx(sum(r['elapsed_seconds'] for r in resumed)/5)
    before = (output/'samples.jsonl').read_bytes()
    main(args+['--resume'])
    assert (output/'samples.jsonl').read_bytes() == before


def test_rejects_changed_manifest_and_corrupt_ids(tmp_path):
    counts = make_counts(tmp_path)
    output = tmp_path/'run'
    main(arguments(output, counts))
    with pytest.raises(ValueError, match='Resume manifest'):
        main(arguments(output, counts, 200)+['--resume'])
    with pytest.raises(ValueError, match='Resume manifest'):
        main(arguments(output, counts)+['--resume', '--mode', 'conditional'])
    rows = records(output)
    rows[0]['draw_id'] = 0
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='Stored draw IDs'):
        main(arguments(output, counts)+['--resume'])


def test_changed_counts_cannot_resume(tmp_path):
    counts = make_counts(tmp_path)
    output = tmp_path/'run'
    main(arguments(output, counts))
    CountBigramHead(17).fit([[1, 1, 1, 1]]).save(counts)
    with pytest.raises(ValueError, match='Resume manifest'):
        main(arguments(output, counts)+['--resume'])


@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
def test_cli_zero_strength_and_offset_disjointness(tmp_path, sampling):
    counts = make_counts(tmp_path)
    first, second = tmp_path/'a', tmp_path/'b'
    main(arguments(first, counts, 0)+['--sampling', sampling, '--strength', '0'])
    main(arguments(second, counts, 10000)+['--sampling', sampling, '--strength', '0'])
    assert not {r['draw_id'] for r in records(first)} & {r['draw_id'] for r in records(second)}
    assert [r['token_ids'] for r in records(first)] != [r['token_ids'] for r in records(second)]


def test_invalid_arguments_rejected_before_output_creation(tmp_path):
    counts = make_counts(tmp_path)
    output = tmp_path/'bad'
    with pytest.raises(ValueError, match='nonnegative'):
        main(arguments(output, counts, -1))
    assert not output.exists()
    with pytest.raises(ValueError, match='Synthetic fixtures'):
        main(arguments(output, counts)+['--score-gpt2'])
    assert not output.exists()
