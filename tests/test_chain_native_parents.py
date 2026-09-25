import pytest
import torch
from chain_crf.data import document_split
from scripts.prepare_chain_native_parents import select_parents


def doc(role):
    return next(str(i) for i in range(100000) if document_split(str(i)) == role)


def fixture():
    rows = [{'source_document_sha256': doc('train'), 'input_ids': [16, 2, 3, 16]},
            {'source_document_sha256': doc('dev'), 'input_ids': [16, 4, 5, 16]}]
    previous = {role: {'tokens': torch.tensor([row['input_ids'][:2], row['input_ids'][2:]]),
                       'document_ids': [row['source_document_sha256']] * 2}
                for role, row in zip(('train', 'dev'), rows)}
    return rows, previous


def select(rows, previous, **kwargs):
    return select_parents(rows, previous, length=4, chunk_length=2,
                          vocab_size=17, boundary_id=16, **kwargs)


def test_intact_parents_exact_tokens_and_source_indices():
    rows, previous = fixture()
    selected, report = select(rows, previous)
    assert selected['train'][0]['input_ids'] == rows[0]['input_ids']
    assert selected['dev'][0]['source_row_index'] == 1
    assert report['splits']['dev']['unmatched_previous_chunks'] == 0


def test_partial_parent_is_excluded_not_joined():
    rows, previous = fixture()
    rows.insert(1, {'source_document_sha256': doc('train'), 'input_ids': [16, 7, 8, 16]})
    previous['train']['tokens'] = torch.cat((previous['train']['tokens'], torch.tensor([[16, 7]])))
    previous['train']['document_ids'].append(doc('train'))
    selected, report = select(rows, previous)
    assert len(selected['train']) == 1
    assert report['splits']['train']['partial_parents_excluded'] == 1


def test_previous_exclusions_or_role_change_rejected():
    rows, previous = fixture()
    with pytest.raises(ValueError, match='exclusion'):
        select(rows, previous, exclusions=[doc('train')])
    previous['dev']['document_ids'][0] = doc('train')
    with pytest.raises(ValueError, match='role'):
        select(rows, previous)


@pytest.mark.parametrize('bad', [[16, 2, 3, 1], [16, -1, 3, 16], [16, True, 3, 16], [16, 3, 16]])
def test_invalid_or_boundary_changed_source_rejected(bad):
    rows, previous = fixture()
    rows[0]['input_ids'] = bad
    with pytest.raises(ValueError, match='Source'):
        select(rows, previous)


def test_missing_old_chunk_rejected():
    rows, previous = fixture()
    previous['train']['tokens'][0, 1] = 9
    with pytest.raises(ValueError, match='absent'):
        select(rows, previous)
