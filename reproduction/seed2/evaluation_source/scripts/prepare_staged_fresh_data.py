#!/usr/bin/env python3
"""Select fresh document-disjoint pilot data from pinned OpenWebText caches.

Training uses a document-preserving cache inside the pinned training window.
Development and test use the separate held-out window and exclude every old
debug document. Selection follows cache/source order and never reads model
scores. Only the requested prefixes are collected in Python; Arrow caches
remain memory mapped and file hashes are computed in bounded-size chunks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import data_provenance


IDENTITY_FIELDS = (
  'dataset_name_or_path', 'dataset_config_name', 'source_revision',
  'source_split', 'source_num_rows', 'text_field', 'tokenizer_name_or_path',
  'tokenizer_revision', 'tokenizer_vocab_size', 'wrap',
)
REQUIRED_COLUMNS = frozenset({
  'input_ids', 'source_document_index', 'source_document_sha256',
  'source_chunk_index',
})
HASH_RE = re.compile(r'^[0-9a-f]{64}$')
REJECTION_REASONS = (
  'old_debug_document', 'not_first_chunk', 'record_too_short',
  'duplicate_source_index', 'duplicate_document_sha256',
  'duplicate_token_prefix',
)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _hash(value, field):
  if not isinstance(value, str) or not HASH_RE.fullmatch(value):
    raise ValueError(f'{field} must be a lowercase SHA256 digest')
  return value


def _integer(value, field):
  if type(value) is not int or value < 0:
    raise ValueError(f'{field} must be a nonnegative integer')
  return value


def read_debug_manifest(path: Path):
  payload = json.loads(path.read_text())
  hashes = payload.get('exclude_from_future_benchmark_document_ids')
  if not isinstance(hashes, list) or not hashes:
    raise ValueError('debug manifest must contain all excluded document hashes')
  excluded = {_hash(value, 'excluded debug document') for value in hashes}
  if len(excluded) != len(hashes):
    raise ValueError('debug exclusion list contains duplicate hashes')
  count = payload.get('examples_per_split')
  if count is not None and (type(count) is not int or len(excluded) != 3 * count):
    raise ValueError('debug exclusion count does not cover all three old subsets')
  source = payload.get('source_provenance', {})
  reference = source.get('specification', {})
  data_provenance.validate_manifest(source, expected_specification=reference)
  if reference.get('dataset_name_or_path') != 'Skylion007/openwebtext':
    raise ValueError('debug reference is not the pinned OpenWebText source')
  data_provenance.require_commit_revision(reference.get('source_revision'),
                                           field='source revision')
  data_provenance.require_commit_revision(reference.get('tokenizer_revision'),
                                           field='tokenizer revision')
  proof = reference.get('disjoint_window_proof', {})
  expected = data_provenance.disjoint_window_proof(
    dataset_name_or_path=reference['dataset_name_or_path'],
    dataset_config_name=reference.get('dataset_config_name'),
    split=reference['source_split'], revision=reference['source_revision'],
    source_num_rows=reference['source_num_rows'],
    train_window=proof.get('train_window'),
    heldout_window=proof.get('heldout_window'))
  if proof != expected:
    raise ValueError('debug reference has an invalid pinned-window proof')
  return payload, reference, excluded, expected


def open_cache(path: Path, role: str, reference: dict, proof: dict):
  from datasets import Dataset, load_from_disk
  sidecar = path.with_suffix(path.suffix + '.provenance.json')
  provenance = json.loads(sidecar.read_text())
  specification = provenance.get('specification', {})
  data_provenance.validate_manifest(provenance, expected_specification=specification)
  for key in IDENTITY_FIELDS:
    if key not in reference or specification.get(key) != reference[key]:
      raise ValueError(f'{role} cache differs from the pinned reference: {key}')
  if specification.get('document_boundary_mode') != 'source_document':
    raise ValueError(f'{role} cache must use source_document boundaries; '
                     'concatenated token streams have no usable document identity')
  window = data_provenance.normalize_window(
    specification.get('source_window'), field=f'{role} source window',
    source_num_rows=specification['source_num_rows'])
  allowed = proof[f'{role}_window']
  if window is None or not (allowed[0] <= window[0] < window[1] <= allowed[1]):
    raise ValueError(f'{role} cache window must lie inside the pinned {role} window')
  data = load_from_disk(str(path), keep_in_memory=False)
  if not isinstance(data, Dataset):
    raise ValueError(f'{role} cache must be one saved Dataset, not a DatasetDict')
  missing = REQUIRED_COLUMNS - set(data.column_names)
  if missing:
    raise ValueError(f'{role} cache lacks source metadata: {sorted(missing)}')
  observed = provenance.get('observed', {})
  if observed.get('processed_num_sequences') != len(data):
    raise ValueError(f'{role} cache row count differs from its provenance')
  if observed.get('processed_fingerprint') != data._fingerprint:
    raise ValueError(f'{role} cache fingerprint differs from its provenance')
  info = {'path': str(path), 'rows': len(data), 'fingerprint': data._fingerprint,
          'source_window': list(window), 'provenance_path': str(sidecar),
          'provenance_sha256': sha256_file(sidecar), 'provenance': provenance}
  return data.with_format(None), info


def cache_file_identity(data, path: Path):
  """Hash all persisted Arrow shards without reading them into Python lists."""
  files = sorted({Path(item['filename']).resolve() for item in data.cache_files})
  if not files:
    raise ValueError('cache has no persisted Arrow files')
  shards = []
  for filename in files:
    if not filename.is_relative_to(path):
      raise ValueError('saved Dataset references a file outside its cache directory')
    shards.append({'path': str(filename.relative_to(path)),
                   'bytes': filename.stat().st_size,
                   'sha256': sha256_file(filename)})
  metadata = {name: sha256_file(path / name)
               for name in ('state.json', 'dataset_info.json')
               if (path / name).is_file()}
  return {'arrow_files': shards, 'metadata_sha256': metadata}


def select_source(data, info, targets, length, excluded, used, minimum_source_index=None):
  """Fill requested splits in source order, retaining only selected prefixes."""
  selected = {name: [] for name, _ in targets}
  rejections = Counter({key: 0 for key in REJECTION_REASONS})
  if minimum_source_index is not None:
    start, stop = info['source_window']
    if type(minimum_source_index) is not int or not start <= minimum_source_index < stop:
      raise ValueError('minimum source index must be within the declared source window')
    rejections['before_minimum_source_index'] = 0
  split_cursor = 0
  previous_index = -1
  scanned = 0
  vocab_size = info['provenance']['specification']['tokenizer_vocab_size']
  for cache_row, row in enumerate(data):
    scanned += 1
    index = _integer(row['source_document_index'], 'source_document_index')
    document_hash = _hash(row['source_document_sha256'], 'source_document_sha256')
    chunk_index = _integer(row['source_chunk_index'], 'source_chunk_index')
    if index < previous_index:
      raise ValueError('cache source_document_index order is not monotonic')
    previous_index = index
    start, stop = info['source_window']
    if not start <= index < stop:
      raise ValueError('cached document index is outside its declared source window')
    if minimum_source_index is not None and index < minimum_source_index:
      rejections['before_minimum_source_index'] += 1
      continue
    if document_hash in excluded:
      rejections['old_debug_document'] += 1
      continue
    if chunk_index != 0:
      rejections['not_first_chunk'] += 1
      continue
    ids = row['input_ids']
    if not isinstance(ids, list):
      raise ValueError('input_ids must be an Arrow list of integers')
    if len(ids) < length:
      rejections['record_too_short'] += 1
      continue
    if index in used['source_document_index']:
      rejections['duplicate_source_index'] += 1
      continue
    if document_hash in used['source_document_sha256']:
      rejections['duplicate_document_sha256'] += 1
      continue
    prefix = ids[:length]
    if any(type(value) is not int or not 0 <= value < vocab_size for value in prefix):
      raise ValueError('selected token prefix contains an invalid token ID')
    prefix_hash = hashlib.sha256(json.dumps(
      prefix, separators=(',', ':')).encode('ascii')).hexdigest()
    if prefix_hash in used['prefix_token_ids_sha256']:
      rejections['duplicate_token_prefix'] += 1
      continue
    name, count = targets[split_cursor]
    record = {
      'id': f'owt-fresh-{name}-{index}', 'split': name,
      'dataset': info['provenance']['specification']['logical_dataset_name'],
      'document_id': document_hash, 'source_document_index': index,
      'source_document_sha256': document_hash, 'source_chunk_index': chunk_index,
      'source_cache_row': cache_row, 'prefix_token_ids_sha256': prefix_hash,
      'input_ids': prefix,
    }
    if 'source_document_token_count' in row:
      record['source_document_token_count'] = _integer(
        row['source_document_token_count'], 'source_document_token_count')
    selected[name].append(record)
    for key in used:
      used[key].add(record[key])
    if len(selected[name]) == count:
      split_cursor += 1
      if split_cursor == len(targets):
        break
  if split_cursor != len(targets):
    counts = {name: len(rows) for name, rows in selected.items()}
    raise ValueError(f'insufficient eligible source documents: selected {counts}, '
                     f'requested {dict(targets)}, rejections {dict(rejections)}')
  return selected, {'rows_scanned': scanned,
                    'selected_documents': sum(len(rows) for rows in selected.values()),
                    'rejections': dict(rejections)}


def verify_disjoint(selected, excluded):
  checks = {}
  for key in ('source_document_index', 'source_document_sha256',
               'prefix_token_ids_sha256'):
    sets = {split: {row[key] for row in rows} for split, rows in selected.items()}
    for split, rows in selected.items():
      if len(sets[split]) != len(rows):
        raise AssertionError(f'{split} repeats {key}')
    for first, second in itertools.combinations(selected, 2):
      overlap = len(sets[first] & sets[second])
      checks[f'{first}_{second}_{key}_overlap'] = overlap
      if overlap:
        raise AssertionError(f'{first}/{second} overlap on {key}')
  for split, rows in selected.items():
    overlap = len({row['source_document_sha256'] for row in rows} & excluded)
    checks[f'{split}_old_debug_document_overlap'] = overlap
    if overlap:
      raise AssertionError(f'{split} reuses an old debug document')
  return checks


def prepare(train_cache: Path, heldout_cache: Path, debug_manifest: Path,
             output_dir: Path, *, train_examples=2048, dev_examples=64,
             test_examples=128, length=128, heldout_min_source_index=None):
  if output_dir.exists() or output_dir.is_symlink():
    raise FileExistsError(output_dir)
  if any(type(value) is not int or value < 1 for value in (
      train_examples, dev_examples, test_examples, length)):
    raise ValueError('all document counts and prefix length must be positive integers')
  train_cache, heldout_cache = train_cache.resolve(), heldout_cache.resolve()
  if train_cache == heldout_cache:
    raise ValueError('training and held-out caches must be distinct')
  debug_manifest = debug_manifest.resolve()
  _, reference, excluded, proof = read_debug_manifest(debug_manifest)
  train, train_info = open_cache(train_cache, 'train', reference, proof)
  heldout, heldout_info = open_cache(heldout_cache, 'heldout', reference, proof)
  used = {key: set() for key in ('source_document_index',
                                'source_document_sha256', 'prefix_token_ids_sha256')}
  selected, train_stats = select_source(
    train, train_info, [('train', train_examples)], length, excluded, used)
  heldout_selected, heldout_stats = select_source(
    heldout, heldout_info, [('dev', dev_examples), ('test', test_examples)],
    length, excluded, used, minimum_source_index=heldout_min_source_index)
  selected.update(heldout_selected)
  checks = verify_disjoint(selected, excluded)
  train_info.update(cache_file_identity(train, train_cache))
  heldout_info.update(cache_file_identity(heldout, heldout_cache))
  result = {
    'artifact': 'fresh_staged_openwebtext_splits', 'schema_version': 1,
    'purpose': 'source-order pilot; not a random representative benchmark sample',
    'selection': 'first eligible source documents in each cache; no model outcomes',
    'prefix_policy': 'first length cached tokens from source chunk zero; '
                     'preserve the cache tokenizer and wrapping',
    'length': length,
    'heldout_min_source_index': heldout_min_source_index,
    'counts': {'train': train_examples, 'dev': dev_examples, 'test': test_examples},
    'source': {'train': train_info, 'heldout': heldout_info},
    'pinned_source_window_proof': proof,
    'old_debug_manifest': str(debug_manifest),
    'old_debug_manifest_sha256': sha256_file(debug_manifest),
    'excluded_old_debug_document_sha256': sorted(excluded),
    'exclusions_applied_to': ['train', 'dev', 'test'],
    'scan': {'train': train_stats, 'heldout': heldout_stats},
    'rejection_counting': 'one reason per rejected row, in the order listed by rejections',
    'splits': {name: {
      'source_cache_role': 'train' if name == 'train' else 'heldout',
      'examples': len(rows),
      **{key: [row[key] for row in rows] for key in (
        'source_document_index', 'source_document_sha256', 'prefix_token_ids_sha256')},
    } for name, rows in selected.items()},
    'disjointness_checks': checks,
    'script_sha256': sha256_file(Path(__file__)),
    'file_sha256': {},
  }
  # Selection and provenance validation finish before any output is created.
  # Exclusive creation prevents a second invocation from replacing a dataset.
  output_dir.mkdir(parents=True, exist_ok=False)
  for split, rows in selected.items():
    path = output_dir / f'{split}.jsonl'
    with path.open('x') as handle:
      for row in rows:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    result['file_sha256'][path.name] = sha256_file(path)
  result['manifest_sha256'] = data_provenance.canonical_sha256(result)
  with (output_dir / 'manifest.json').open('x') as handle:
    json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')
  return result


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--train-cache', type=Path, required=True)
  parser.add_argument('--heldout-cache', type=Path, required=True)
  parser.add_argument('--debug-manifest', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--train-examples', type=int, default=2048)
  parser.add_argument('--dev-examples', type=int, default=64)
  parser.add_argument('--test-examples', type=int, default=128)
  parser.add_argument('--length', type=int, default=128)
  parser.add_argument('--heldout-min-source-index', type=int,
                      help='Skip held-out source documents before this index, before selection.')
  return parser.parse_args(argv)


def main(argv=None):
  args = _args(argv)
  result = prepare(**vars(args))
  print(json.dumps({'output_dir': str(args.output_dir.resolve()),
                    'counts': result['counts'], 'scan': result['scan'],
                    'manifest_sha256': result['manifest_sha256']}, sort_keys=True), flush=True)


if __name__ == '__main__':
  main()
