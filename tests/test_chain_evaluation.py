"""Evaluation draw identities remain separate from local record indices."""
import json

import pytest

from scripts.evaluate_chain_crf import main


def arguments(output, offset=0):
    return ['--output', str(output), '--synthetic', '--device', 'cpu',
            '--length', '8', '--steps', '2', '--samples', '4',
            '--batch-size', '1', '--warmup', '0', '--sample-offset', str(offset)]


def read_records(output):
    return [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]


def test_final_draw_set_is_disjoint_and_resume_preserves_it(tmp_path):
    screen, final = tmp_path/'screen', tmp_path/'final'
    main(arguments(screen))
    main(arguments(final, 10000))
    screen_rows, final_rows = read_records(screen), read_records(final)
    assert [r['sample_id'] for r in final_rows] == list(range(4))
    assert [r['draw_id'] for r in final_rows] == list(range(10000, 10004))
    assert not {r['draw_id'] for r in final_rows} & {r['draw_id'] for r in screen_rows}
    assert [r['token_ids'] for r in final_rows] != [r['token_ids'] for r in screen_rows]
    assert json.loads((final/'manifest.json').read_text())['config']['sample_offset'] == 10000
    (final/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in final_rows[:2]))
    main(arguments(final, 10000)+['--resume'])
    resumed = read_records(final)
    assert [(r['draw_id'], r['token_ids']) for r in resumed] == [
        (r['draw_id'], r['token_ids']) for r in final_rows]


def test_resume_rejects_changed_or_corrupt_draw_identity(tmp_path):
    output = tmp_path/'run'
    main(arguments(output, 100))
    with pytest.raises(ValueError, match='Resume manifest'):
        main(arguments(output, 200)+['--resume'])
    rows = read_records(output)
    rows[0]['draw_id'] = 0
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='Stored draw IDs'):
        main(arguments(output, 100)+['--resume'])


def test_negative_offset_rejected_before_creating_output(tmp_path):
    output = tmp_path/'invalid'
    with pytest.raises(ValueError, match='nonnegative'):
        main(arguments(output, -1))
    assert not output.exists()
