#!/usr/bin/env python3
"""Audit retained byte-level BPE token IDs without generating or scoring text.

Example:
    python scripts/evaluate_chain_tokenization.py --input samples.jsonl \
        --tokenizer-json tokenizer.json --tokenizer-sha256 KNOWN_SHA256 \
        --output tokenization.json

The output contains aggregate counts and provenance hashes, never decoded text.
An ID round-trip mismatch need not imply invalid UTF-8: distinct BPE segmentations
can decode to the same valid string. Literal U+FFFD is counted separately from
replacement characters introduced while decoding malformed UTF-8 bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer, models


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inverse_byte_alphabet():
    """Inverse of the reversible GPT-2 byte-to-Unicode alphabet."""
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = values.copy()
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte for byte, codepoint in zip(values, codepoints)}


def valid_utf8(data):
    try:
        data.decode('utf-8', errors='strict')
        return True
    except UnicodeDecodeError:
        return False


def suffix_truncation_only(data):
    """True iff the first strict-decoding failure is an incomplete final scalar."""
    try:
        data.decode('utf-8', errors='strict')
    except UnicodeDecodeError as error:
        return error.reason == 'unexpected end of data' and error.end == len(data)
    return False


class ByteLevelAudit:
    """Pinned ByteLevel/BPE decoder; only the specified boundary special is allowed."""

    def __init__(self, tokenizer_json, expected_sha256, boundary_token='<|endoftext|>'):
        path = Path(tokenizer_json)
        raw = path.read_bytes()
        if (not re.fullmatch('[0-9a-f]{64}', expected_sha256)
                or hashlib.sha256(raw).hexdigest() != expected_sha256):
            raise ValueError('Tokenizer JSON does not match the explicit SHA256 pin')
        specification = json.loads(raw)
        # Older GPT-2 tokenizer JSONs omit model.type; the loaded class is also checked.
        if (specification.get('model', {}).get('type') not in (None, 'BPE')
                or (specification.get('decoder') or {}).get('type') != 'ByteLevel'
                or specification.get('normalizer') is not None):
            raise ValueError('Require unnormalized ByteLevel BPE with the GPT-2 byte alphabet')
        if (specification.get('truncation') is not None or specification.get('padding') is not None
                or specification['model'].get('dropout') not in (None, 0, 0.0)):
            raise ValueError('Disable tokenizer truncation, padding and stochastic BPE dropout')
        if any(item['content'] != boundary_token for item in specification.get('added_tokens', [])):
            raise ValueError('Additional special/added tokens are unsupported')
        self.tokenizer = Tokenizer.from_str(raw.decode('utf-8'))
        if not isinstance(self.tokenizer.model, models.BPE):
            raise ValueError('Require a BPE tokenizer model')
        vocabulary = specification['model']['vocab']
        if boundary_token not in vocabulary:
            raise ValueError('Boundary token must be present in the model vocabulary')
        if (any(type(index) is not int or index < 0 for index in vocabulary.values())
                or len(set(vocabulary.values())) != len(vocabulary)):
            raise ValueError('Vocabulary IDs must be distinct nonnegative integers')
        inverse = inverse_byte_alphabet()
        try:
            self.token_bytes = {index: bytes(inverse[c] for c in token)
                                for token, index in vocabulary.items()}
        except KeyError as error:
            raise ValueError('Token contains a character outside the GPT-2 byte alphabet') from error
        if self.tokenizer.get_vocab(with_added_tokens=True) != vocabulary:
            raise ValueError('Tokenizer added-token IDs differ from the model vocabulary')
        self.boundary_id = vocabulary[boundary_token]
        self.sha256 = expected_sha256

    def analyze(self, token_ids):
        if (not isinstance(token_ids, list) or not token_ids
                or any(type(index) is not int or index not in self.token_bytes for index in token_ids)):
            raise ValueError('Each record needs a nonempty list of valid integer token IDs')
        raw = b''.join(self.token_bytes[index] for index in token_ids)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        if raw.decode('utf-8', errors='replace') != text:
            raise ValueError('Byte reconstruction disagrees with the pinned tokenizer decoder')
        strict_valid = valid_utf8(raw)
        literal = b'\xef\xbf\xbd' in raw
        replacement = '\ufffd' in text
        if replacement != (not strict_valid or literal):
            raise ValueError('Replacement-character accounting is inconsistent')
        canonical = self.tokenizer.encode(text, add_special_tokens=False).ids
        start = int(token_ids[0] == self.boundary_id)
        end = len(token_ids) - int(token_ids[-1] == self.boundary_id)
        payload = b''.join(self.token_bytes[index] for index in token_ids[start:end])
        return {
            'samples': 1,
            'original_tokens': len(token_ids),
            'roundtrip_tokens': len(canonical),
            'exact_id_roundtrips': int(canonical == token_ids),
            'sequences_shorter_after_roundtrip': int(len(canonical) < len(token_ids)),
            'sequences_longer_after_roundtrip': int(len(canonical) > len(token_ids)),
            'strict_invalid_utf8_sequences': int(not strict_valid),
            'sequences_with_literal_replacement_utf8_bytes': int(literal),
            'valid_utf8_sequences_with_literal_replacement': int(strict_valid and literal),
            'invalid_utf8_sequences_also_containing_literal_replacement': int(not strict_valid and literal),
            'decoded_sequences_with_replacement_character': int(replacement),
            'valid_utf8_noncanonical_id_sequences': int(strict_valid and canonical != token_ids),
            'invalid_sequences_explained_only_by_final_utf8_truncation': int(suffix_truncation_only(payload)),
        }


def evaluate_records(records, decoder):
    counts = Counter()
    for record in records:
        if not isinstance(record, dict) or 'token_ids' not in record:
            raise ValueError('Every JSONL record must contain token_ids')
        counts.update(decoder.analyze(record['token_ids']))
    if not counts['samples']:
        raise ValueError('Input contains no retained samples')
    return dict(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--tokenizer-json', type=Path, required=True)
    parser.add_argument('--tokenizer-sha256', required=True)
    parser.add_argument('--boundary-token', default='<|endoftext|>')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing audit')
    decoder = ByteLevelAudit(args.tokenizer_json, args.tokenizer_sha256, args.boundary_token)
    input_sha = file_sha256(args.input)
    with args.input.open() as handle:
        counts = evaluate_records((json.loads(line) for line in handle if line.strip()), decoder)
    if file_sha256(args.input) != input_sha:
        raise ValueError('Input changed during the audit; use immutable retained samples')
    report = {
        'schema': 'chain_tokenization_audit_v1',
        'input_sha256': input_sha,
        'tokenizer_json_sha256': decoder.sha256,
        'evaluator_sha256': file_sha256(__file__),
        'boundary_token_id': decoder.boundary_id,
        'counts': counts,
        'protocol': {
            'byte_reconstruction': 'Concatenate inverse-GPT2-alphabet bytes across the complete token sequence.',
            'decoding': 'Strict UTF8 for validity; lossy decode verified against ByteLevel decoder, special tokens retained.',
            'encoding': 'Pinned tokenizer encode(add_special_tokens=False), without HF whitespace cleanup.',
            'suffix_rule': 'Remove at most one initial and final boundary token; only a first error of unexpected end-of-data counts as suffix-only truncation.',
        },
        'interpretation': 'Structural diagnostics only. Noncanonical BPE IDs can decode valid text; these counts are not model scores or a causal explanation of generation quality.',
        'gpu_used': False,
        'inference_or_features_recomputed': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'samples': counts['samples'], 'audit_sha256': file_sha256(args.output)}))


if __name__ == '__main__':
    main()
