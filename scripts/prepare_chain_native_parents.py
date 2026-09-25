#!/usr/bin/env python3
"""Recover intact source rows fully represented by an existing short-row split.

This preserves train/dev roles, document exclusions and token content. Native
rows are selected from their authenticated source, never assembled by joining
short rows. An incompletely represented parent is excluded and counted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from chain_crf.backbone import file_sha256
from chain_crf.data import atomic_json, atomic_torch_save, canonical_hash, document_split
from scripts.prepare_chain_data import prior_document_ids, source_rows


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest()


def select_parents(rows, previous, *, exclusions=(), length=1024, chunk_length=256,
                   salt='chain-crf-20260925-v1', vocab_size=50257, boundary_id=50256):
    if length < 1 or chunk_length < 1 or length % chunk_length:
        raise ValueError('Native length must be a positive multiple of previous chunk length')
    excluded = set(exclusions)
    chunks, documents = {}, {}
    for split, old in previous.items():
        tokens, ids = old['tokens'], old['document_ids']
        if tokens.ndim != 2 or tokens.shape[1] != chunk_length or len(tokens) != len(ids):
            raise ValueError('Previous split shape or identity count differs')
        if tokens.dtype != torch.long or bool(((tokens < 0) | (tokens >= vocab_size)).any()):
            raise ValueError('Previous split contains invalid token IDs')
        documents[split] = set(ids)
        if any(document_split(doc, salt) != split or doc in excluded for doc in ids):
            raise ValueError('Previous document role or exclusion mismatch')
        chunks[split] = {(doc, token_hash(row)) for row, doc in zip(tokens.tolist(), ids)}
    if set(previous) != {'train', 'dev'} or documents['train'] & documents['dev']:
        raise ValueError('Need document-disjoint previous train and dev only')
    chosen = {split: [] for split in previous}
    matched = {split: set() for split in previous}
    partial = {split: 0 for split in previous}
    seen = set()
    scanned = 0
    for index, row in enumerate(rows):
        scanned += 1
        doc, ids = row.get('source_document_sha256'), row.get('input_ids')
        if (not isinstance(doc, str) or not isinstance(ids, list) or len(ids) != length
                or any(type(v) is not int or not 0 <= v < vocab_size for v in ids)
                or ids[0] != boundary_id or ids[-1] != boundary_id):
            raise ValueError('Source is not a valid intact, boundary-preserving native row')
        if doc in excluded:
            continue
        split = document_split(doc, salt)
        if split not in previous or doc not in documents[split]:
            continue
        digest = token_hash(ids)
        if digest in seen:
            continue
        seen.add(digest)
        pieces = [(doc, token_hash(ids[i:i+chunk_length])) for i in range(0, length, chunk_length)]
        present = [piece in chunks[split] for piece in pieces]
        matched[split].update(piece for piece in pieces if piece in chunks[split])
        if all(present):
            chosen[split].append({'input_ids': ids, 'document_id': doc,
                                  'source_row_index': index, 'source_row_tokens_sha256': digest})
        elif any(present):
            partial[split] += 1
    reports = {}
    for split, records in chosen.items():
        missing = chunks[split] - matched[split]
        if missing:
            raise ValueError(f'{split}: {len(missing)} previous chunks absent from original source')
        if not records:
            raise ValueError(f'{split}: no complete native parents')
        reports[split] = {'rows': len(records), 'documents': len({r['document_id'] for r in records}),
                          'partial_parents_excluded': partial[split], 'unmatched_previous_chunks': 0,
                          'source_indices_sha256': canonical_hash([r['source_row_index'] for r in records]),
                          'source_token_hashes_sha256': canonical_hash([r['source_row_tokens_sha256'] for r in records])}
    return chosen, {'source_rows_scanned': scanned, 'splits': reports}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-data', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-train', type=int, required=True)
    parser.add_argument('--expected-dev', type=int, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    previous_manifest_path = args.previous_data / 'manifest.json'
    previous_manifest = json.loads(previous_manifest_path.read_text())
    if previous_manifest['length'] != 256:
        raise ValueError('This CLI requires the audited 256-token previous protocol')
    excluded_sources = previous_manifest['excluded_sources']
    if any(file_sha256(Path(path)) != expected for path, expected in excluded_sources.items()):
        raise ValueError('Exclusion source checksum mismatch')
    exclusions = prior_document_ids([Path(path) for path in excluded_sources])
    if canonical_hash(sorted(exclusions)) != previous_manifest['excluded_ids_sha256']:
        raise ValueError('Excluded identity list changed')
    previous = {}
    for split in ('train', 'dev'):
        path = args.previous_data / f'{split}.pt'
        if file_sha256(path) != previous_manifest['files'][path.name]:
            raise ValueError(f'{split} checksum mismatch')
        previous[split] = torch.load(path, map_location='cpu', weights_only=True)
    rows, source = source_rows(args.source)
    if source.get('provenance_sha256') != previous_manifest['source'].get('provenance_sha256'):
        raise ValueError('Native cache differs from previous authenticated source')
    selected, report = select_parents(rows, previous, exclusions=exclusions,
                                     salt=previous_manifest['split_salt'])
    for split, count in (('train', args.expected_train), ('dev', args.expected_dev)):
        if report['splits'][split]['rows'] != count:
            raise ValueError(f'{split} native parent count differs from prior audit')
    provenance = {'schema': 'chain_crf_native_parents_v1', 'length': 1024,
                  'source': source, 'previous_manifest_sha256': file_sha256(previous_manifest_path),
                  'previous_files': {f'{s}.pt': previous_manifest['files'][f'{s}.pt'] for s in previous},
                  'split_salt': previous_manifest['split_salt'], 'document_disjoint': True,
                  'excluded_ids_sha256': previous_manifest['excluded_ids_sha256'],
                  'excluded_sources': excluded_sources, 'tokenizer': previous_manifest['tokenizer'],
                  'tokenizer_revision': previous_manifest['tokenizer_revision'],
                  'selection': 'intact source rows with every chunk present in previous same-document same-role split',
                  'boundary_policy': 'preserve native source BOS/EOS; no joins or re-tokenization',
                  'test_access': False, 'report': report,
                  'preparation_source_sha256': file_sha256(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=False)
    for split, records in selected.items():
        atomic_torch_save({'tokens': torch.tensor([r['input_ids'] for r in records], dtype=torch.long),
                           'document_ids': [r['document_id'] for r in records],
                           'source_row_indices': [r['source_row_index'] for r in records],
                           'provenance': {**provenance, 'split': split}}, args.output / f'{split}.pt')
        with (args.output / f'{split}.jsonl').open('x') as handle:
            for record in records:
                handle.write(json.dumps({**record, 'split': split}) + '\n')
    provenance['files'] = {p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()}
    atomic_json(provenance, args.output / 'manifest.json')
    print(json.dumps({'event': 'prepared_native_parents', **report, 'files': provenance['files']}), flush=True)


if __name__ == '__main__':
    main()
