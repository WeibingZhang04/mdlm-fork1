#!/usr/bin/env python3
"""Carve document-disjoint exploratory subsets from a pinned token cache.

These small subsets are for mechanism debugging, including deliberate fitting
of the debug-fit subset. They must be excluded from future benchmark claims.
No model score is used to select examples or assign splits.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cache', required=True, type=Path)
  parser.add_argument('--output-dir', required=True, type=Path)
  parser.add_argument('--examples-per-split', type=int, default=32)
  parser.add_argument('--length', type=int, default=128)
  args = parser.parse_args()
  if min(args.examples_per_split, args.length) < 1:
    raise ValueError('counts and length must be positive')
  from datasets import load_from_disk
  data = load_from_disk(str(args.cache))
  needed = {'input_ids', 'source_document_index', 'source_document_sha256'}
  if not needed.issubset(data.column_names):
    raise ValueError('cache must preserve source document identities')
  provenance = args.cache.with_suffix(args.cache.suffix + '.provenance.json')
  if not provenance.is_file():
    raise FileNotFoundError(provenance)
  rows, seen = [], set()
  split_names = ('debug-fit', 'dev', 'test')
  for index, row in enumerate(data):
    identity = str(row['source_document_sha256'])
    if identity in seen or len(row['input_ids']) < args.length:
      continue
    seen.add(identity)
    split = split_names[len(rows) // args.examples_per_split]
    rows.append({
      'id': f'owt-debug-{row["source_document_index"]}', 'split': split,
      'dataset': 'openwebtext-heldout-pinned',
      'document_id': identity, 'source_document_index': row['source_document_index'],
      'source_cache_row': index, 'input_ids': row['input_ids'][:args.length],
    })
    if len(rows) == 3 * args.examples_per_split:
      break
  if len(rows) != 3 * args.examples_per_split:
    raise ValueError('not enough distinct source documents')
  args.output_dir.mkdir(parents=True, exist_ok=False)
  files = {}
  for split in split_names + ('dependence',):
    selected = [row for row in rows if row['split'] == split or
                (split == 'dependence' and row['split'] in ('dev', 'test'))]
    path = args.output_dir / f'{split}.jsonl'
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in selected))
    files[path.name] = sha256(path)
  result = {
    'purpose': 'exploratory debugging only; debug-fit is deliberately trained on',
    'selection': 'first distinct documents in cache order; no outcome selection',
    'exclude_from_future_benchmark_document_ids': sorted(seen),
    'source_cache': str(args.cache), 'source_cache_rows': len(data),
    'source_provenance': json.loads(provenance.read_text()),
    'source_provenance_sha256': sha256(provenance),
    'source_arrow_sha256': {p.name: sha256(p) for p in sorted(args.cache.glob('*.arrow'))},
    'examples_per_split': args.examples_per_split, 'length': args.length,
    'file_sha256': files,
  }
  (args.output_dir / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({'output_dir': str(args.output_dir), 'file_sha256': files}), flush=True)


if __name__ == '__main__':
  main()
