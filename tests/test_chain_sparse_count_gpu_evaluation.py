"""CPU integration of explicit, manifest-identified exact backend selection."""

import json

import pytest
import torch

from chain_crf.counts import CountBigramHead
from chain_crf.sparse_count import SparseCountPotential
from chain_crf.sparse_count_gpu import GPUCountPotential
from scripts.evaluate_chain_sparse_count import generate_sparse_count, main


class ToyBackbone:
    vocab_size = 5
    mask_id = 4

    def __call__(self, tokens, t):
        logits = torch.tensor([0., -.4, -.8, -1.1, 3.])
        return {'log_probs': logits.log_softmax(-1).expand(*tokens.shape, 5)}


@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
@pytest.mark.parametrize('batch', [4, 8, 12])
def test_same_generation_for_both_explicit_backends(sampling, batch):
    counts = CountBigramHead(5, 'pmi', .25).fit([[0, 1, 2], [2, 0, 1]]*5)
    old = SparseCountPotential.from_head(counts)
    new = GPUCountPotential.from_reference(old)
    kwargs = dict(length=7, steps=4, batch_size=batch, sampling=sampling,
                  prefix=[2, 1], device='cpu', sample_offset=927)
    expected, _ = generate_sparse_count(ToyBackbone(), old, **kwargs)
    actual, _ = generate_sparse_count(ToyBackbone(), new, inference_backend='gpu', **kwargs)
    assert torch.equal(actual, expected)
    assert actual[:, :2].eq(torch.tensor([2, 1])).all()
    assert not actual.eq(4).any()


@pytest.mark.parametrize('sampling', ['joint', 'marginal'])
def test_gpu_backend_manifest_partial_resume_and_backend_change_rejection(tmp_path, sampling):
    counts = tmp_path/'counts.pt'
    CountBigramHead(17).fit([[1, 2, 3, 4], [4, 3, 1], [9, 2]]).save(counts)
    output = tmp_path/'gpu'
    args = ['--output', str(output), '--counts', str(counts), '--synthetic', '--device', 'cpu',
            '--length', '7', '--steps', '3', '--samples', '5', '--batch-size', '4',
            '--warmup', '0', '--sample-offset', '290', '--backend', 'gpu',
            '--sampling', sampling]
    main(args)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['config']['backend'] == manifest['method']['inference_backend'] == 'gpu'
    assert manifest['source_sha256']['chain_crf/sparse_count_gpu.py']
    rows = [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    (output/'samples.jsonl').write_text(json.dumps(rows[0])+'\n')
    main(args+['--resume'])
    resumed = [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    assert [(row['draw_id'], row['token_ids']) for row in resumed] == [
        (row['draw_id'], row['token_ids']) for row in rows]
    with pytest.raises(ValueError, match='Resume manifest'):
        main(args+['--resume', '--backend', 'reference'])
