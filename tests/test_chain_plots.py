import json

import pytest

from scripts.plot_chain_generation import collect, load_point


def write_run(directory, *, length=4, samples=2, steps=2):
    directory.mkdir()
    config = {'length': length, 'samples': samples, 'steps': steps, 'batch_size': 1}
    (directory/'manifest.json').write_text(json.dumps({'config': config, 'backbone': {'synthetic_only': False}}))
    (directory/'metrics.json').write_text(json.dumps({'samples': samples, 'seconds_per_sample': .5,
        'token_entropy_nats': 3., 'within_sample_repeat_4': 0.}))
    (directory/'gpt2-large.json').write_text(json.dumps({'model': 'gpt2-large', 'requested_revision': 'pinned',
        'perplexity': 100., 'scored_tokens': samples*(length-1),
        'samples': [{'sample_id': i, 'scored_tokens': length-1} for i in range(samples)]}))
    (directory/'samples.jsonl').write_text(''.join(json.dumps({'sample_id': i,
        'token_ids': list(range(length)), 'prefix_length': 0})+'\n' for i in range(samples)))


def test_measured_plot_data_excludes_local_paths(tmp_path):
    run = tmp_path/'run'
    write_run(run)
    result = collect([{'label': 'Model', 'runs': [str(run)]}])
    assert result[0]['points'][0]['generative_perplexity'] == 100.
    assert str(tmp_path) not in json.dumps(result)
    assert len(result[0]['points'][0]['samples_sha256']) == 64


def test_plot_refuses_incomplete_or_mixed_evaluation(tmp_path):
    first, second = tmp_path/'first', tmp_path/'second'
    write_run(first)
    write_run(second, length=8)
    with pytest.raises(ValueError, match='Mixing'):
        collect([{'label': 'A', 'runs': [first]}, {'label': 'B', 'runs': [second]}])
    (first/'samples.jsonl').write_text('')
    with pytest.raises(ValueError, match='sample counts'):
        load_point(first)
