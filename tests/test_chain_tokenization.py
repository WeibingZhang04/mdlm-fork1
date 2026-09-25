"""Byte validity is separate from literal replacement and BPE canonicalization."""
import json

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

from scripts.evaluate_chain_tokenization import (
    ByteLevelAudit, evaluate_records, file_sha256, inverse_byte_alphabet, main,
)


@pytest.fixture
def fixture(tmp_path):
    inverse = inverse_byte_alphabet()
    alphabet = {byte: character for character, byte in inverse.items()}
    vocabulary = {character: byte for byte, character in alphabet.items()}
    vocabulary['<|endoftext|>'] = 256
    for token in ('he', 'hel', 'hell', 'hello'):
        vocabulary[token] = len(vocabulary)
    tokenizer = Tokenizer(models.BPE(vocabulary, [('h', 'e'), ('he', 'l'), ('hel', 'l'), ('hell', 'o')]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken('<|endoftext|>', special=True)])
    path = tmp_path/'tokenizer.json'
    tokenizer.save(str(path))
    return ByteLevelAudit(path, file_sha256(path)), path


def test_valid_multibyte_character_spanning_tokens(fixture):
    decoder, _ = fixture
    result = decoder.analyze([0xc3, 0xa9])
    assert result['strict_invalid_utf8_sequences'] == 0
    assert result['decoded_sequences_with_replacement_character'] == 0


def test_malformed_continuation_is_not_literal_replacement(fixture):
    decoder, _ = fixture
    result = decoder.analyze([0xc3, ord('x')])
    assert result['strict_invalid_utf8_sequences'] == 1
    assert result['decoded_sequences_with_replacement_character'] == 1
    assert result['sequences_with_literal_replacement_utf8_bytes'] == 0
    assert result['invalid_sequences_explained_only_by_final_utf8_truncation'] == 0


def test_literal_replacement_is_valid_utf8_even_across_tokens(fixture):
    decoder, _ = fixture
    result = decoder.analyze([0xef, 0xbf, 0xbd])
    assert result['strict_invalid_utf8_sequences'] == 0
    assert result['valid_utf8_sequences_with_literal_replacement'] == 1
    result = decoder.analyze([0xff, 0xef, 0xbf, 0xbd])
    assert result['invalid_utf8_sequences_also_containing_literal_replacement'] == 1


def test_valid_noncanonical_bpe_ids_are_not_invalid_text(fixture):
    decoder, _ = fixture
    result = decoder.analyze(list(b'hello'))
    assert result['valid_utf8_noncanonical_id_sequences'] == 1
    assert result['strict_invalid_utf8_sequences'] == 0
    assert result['roundtrip_tokens'] == 1
    assert decoder.analyze([260])['exact_id_roundtrips'] == 1


def test_final_truncation_after_boundary_removal_is_separate(fixture):
    decoder, _ = fixture
    result = decoder.analyze([256, ord('x'), 0xc3, 256])
    assert result['invalid_sequences_explained_only_by_final_utf8_truncation'] == 1
    # An earlier invalid byte makes this more than a final incomplete scalar.
    assert decoder.analyze([256, 0xff, ord('x'), 0xc3, 256])[
        'invalid_sequences_explained_only_by_final_utf8_truncation'] == 0
    assert decoder.analyze([256])['strict_invalid_utf8_sequences'] == 0


@pytest.mark.parametrize('ids', [[], [-1], [261], [True], [1.0], 'hello', None])
def test_invalid_ids_rejected(fixture, ids):
    decoder, _ = fixture
    with pytest.raises(ValueError, match='valid integer token IDs'):
        decoder.analyze(ids)


def test_tokenizer_pin_and_codec_are_validated(fixture):
    _, path = fixture
    with pytest.raises(ValueError, match='SHA256'):
        ByteLevelAudit(path, '0'*64)
    data = json.loads(path.read_text())
    data['decoder'] = {'type': 'WordPiece', 'prefix': '##', 'cleanup': True}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='ByteLevel'):
        ByteLevelAudit(path, file_sha256(path))


def test_empty_or_malformed_records_fail(fixture):
    decoder, _ = fixture
    with pytest.raises(ValueError, match='no retained samples'):
        evaluate_records([], decoder)
    with pytest.raises(ValueError, match='contain token_ids'):
        evaluate_records([{}], decoder)


def test_original_gpt2_legacy_json_without_model_type(fixture):
    _, path = fixture
    data = json.loads(path.read_text())
    data['model'].pop('type')
    path.write_text(json.dumps(data))
    decoder = ByteLevelAudit(path, file_sha256(path))
    assert decoder.analyze([0xc3, 0xa9])['strict_invalid_utf8_sequences'] == 0


@pytest.mark.parametrize('setting', ['padding', 'truncation', 'dropout'])
def test_noncanonicalizing_tokenizer_settings_rejected(fixture, setting):
    _, path = fixture
    data = json.loads(path.read_text())
    if setting == 'dropout':
        data['model']['dropout'] = 0.1
    else:
        data[setting] = {'enabled': True}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='truncation, padding'):
        ByteLevelAudit(path, file_sha256(path))


def test_cli_emits_only_aggregate_counts_and_hashes(fixture, tmp_path, capsys):
    _, tokenizer = fixture
    source, output = tmp_path/'samples.jsonl', tmp_path/'audit.json'
    source.write_text(json.dumps({'token_ids': list(b'hello')})+'\n'+
                      json.dumps({'token_ids': [256, 0xc3, 256]})+'\n')
    arguments = ['--input', str(source), '--tokenizer-json', str(tokenizer),
                 '--tokenizer-sha256', file_sha256(tokenizer), '--output', str(output)]
    main(arguments)
    report = json.loads(output.read_text())
    assert report['counts']['samples'] == 2
    assert report['counts']['strict_invalid_utf8_sequences'] == 1
    assert report['counts']['valid_utf8_noncanonical_id_sequences'] == 1
    assert report['input_sha256'] == file_sha256(source)
    assert report['tokenizer_json_sha256'] == file_sha256(tokenizer)
    assert 'hello' not in output.read_text() and str(tmp_path) not in output.read_text()
    assert 'hello' not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main(arguments)
