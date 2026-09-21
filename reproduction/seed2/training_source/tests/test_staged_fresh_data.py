import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import datasets

import data_provenance
from scripts.prepare_staged_fresh_data import _args, prepare, sha256_file


def digest(text):
  return hashlib.sha256(text.encode()).hexdigest()


def row(index, text, tokens=None, chunk=0):
  return {'source_document_index': index, 'source_document_sha256': digest(text),
          'source_chunk_index': chunk,
          'input_ids': tokens if tokens is not None else [index * 10 + i for i in range(5)]}


class StagedFreshDataTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    datasets.disable_progress_bars()

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.proof = data_provenance.disjoint_window_proof(
      dataset_name_or_path='Skylion007/openwebtext', dataset_config_name='plain_text',
      split='train', revision='a' * 40, source_num_rows=200,
      train_window=[0, 100], heldout_window=[100, 200])
    self.reference = {
      'logical_dataset_name': 'openwebtext-heldout-pinned',
      'dataset_name_or_path': 'Skylion007/openwebtext',
      'dataset_config_name': 'plain_text', 'source_revision': 'a' * 40,
      'source_split': 'train', 'source_num_rows': 200, 'source_window': [100, 200],
      'text_field': 'text', 'tokenizer_name_or_path': 'openai-community/gpt2',
      'tokenizer_revision': 'b' * 40, 'tokenizer_vocab_size': 10000, 'wrap': True,
      'document_boundary_mode': 'source_document', 'disjoint_window_proof': self.proof,
    }
    reference_provenance = data_provenance.build_manifest(
      specification=self.reference, observed={})
    self.debug_manifest = self.root / 'old-debug-manifest.json'
    self.debug_manifest.write_text(json.dumps({
      'examples_per_split': 1, 'source_provenance': reference_provenance,
      'exclude_from_future_benchmark_document_ids': [digest(f'old-{i}') for i in range(3)]}))

  def cache(self, name, rows, *, train=False, specification=None):
    path = self.root / name
    data = datasets.Dataset.from_list(rows)
    data.save_to_disk(str(path))
    persisted = datasets.load_from_disk(str(path))
    spec = copy.deepcopy(specification or self.reference)
    if train:
      spec['source_window'] = [0, 20]  # A small subset of the pinned train window.
      spec['logical_dataset_name'] = 'openwebtext-train-pinned'
    payload = data_provenance.build_manifest(specification=spec, observed={
      'processed_fingerprint': persisted._fingerprint,
      'processed_num_sequences': len(persisted)})
    path.with_suffix(path.suffix + '.provenance.json').write_text(json.dumps(payload))
    return path

  def valid_caches(self):
    train = self.cache('train.dat', [row(1, 'train')], train=True)
    heldout = self.cache('heldout.dat', [row(110, 'dev'), row(111, 'test')])
    return train, heldout

  def run_prepare(self, train, heldout, output=None, **kwargs):
    return prepare(train, heldout, self.debug_manifest,
                    output or self.root / 'fresh', train_examples=kwargs.pop('train_examples', 1),
                    dev_examples=kwargs.pop('dev_examples', 1),
                    test_examples=kwargs.pop('test_examples', 1), length=4, **kwargs)

  def test_document_content_and_prefix_exclusions_are_disjoint(self):
    train = self.cache('train.dat', [
      row(0, 'short', [1, 2]), row(1, 'old-0'),
      row(2, 'shared-content', [11, 12, 13, 14, 15]),
      row(2, 'same-index'), row(3, 'shared-content'),
      row(4, 'same-prefix', [11, 12, 13, 14, 99]),
      row(5, 'later-chunk', chunk=1), row(6, 'train-second', [61, 62, 63, 64]),
      row(7, 'never-scanned'),
    ], train=True)
    heldout = self.cache('heldout.dat', [
      row(100, 'old-0'), row(101, 'old-1'), row(102, 'old-2'),
      row(103, 'shared-content'), row(104, 'train-prefix', [61, 62, 63, 64, 99]),
      row(105, 'fresh-dev'), row(105, 'same-dev-index'), row(106, 'fresh-dev'),
      row(107, 'short-heldout', [1]), row(108, 'fresh-test'), row(109, 'never-scanned'),
    ])
    result = self.run_prepare(train, heldout, train_examples=2)
    self.assertEqual(result['splits']['train']['source_document_index'], [2, 6])
    self.assertEqual(result['splits']['dev']['source_document_index'], [105])
    self.assertEqual(result['splits']['test']['source_document_index'], [108])
    self.assertTrue(all(value == 0 for value in result['disjointness_checks'].values()))
    self.assertEqual(result['scan']['train']['rows_scanned'], 8)
    self.assertEqual(result['scan']['heldout']['rows_scanned'], 10)
    self.assertEqual(result['scan']['train']['rejections'], {
      'old_debug_document': 1, 'not_first_chunk': 1, 'record_too_short': 1,
      'duplicate_source_index': 1, 'duplicate_document_sha256': 1,
      'duplicate_token_prefix': 1})
    self.assertEqual(result['scan']['heldout']['rejections']['old_debug_document'], 3)
    for split in ('train', 'dev', 'test'):
      path = self.root / 'fresh' / f'{split}.jsonl'
      rows = [json.loads(line) for line in path.read_text().splitlines()]
      self.assertTrue(all(len(r['input_ids']) == 4 and r['source_chunk_index'] == 0 for r in rows))
      self.assertEqual(result['file_sha256'][path.name], sha256_file(path))
    for role, source in result['source'].items():
      path = train if role == 'train' else heldout
      for shard in source['arrow_files']:
        self.assertEqual(shard['sha256'], sha256_file(path / shard['path']))
    unsigned = dict(result)
    manifest_hash = unsigned.pop('manifest_sha256')
    self.assertEqual(manifest_hash, data_provenance.canonical_sha256(unsigned))

  def test_repeat_selection_is_byte_deterministic_and_does_not_overwrite(self):
    train, heldout = self.valid_caches()
    first = self.run_prepare(train, heldout, self.root / 'first')
    second = self.run_prepare(train, heldout, self.root / 'second')
    self.assertEqual(first, second)
    for name in ('train.jsonl', 'dev.jsonl', 'test.jsonl', 'manifest.json'):
      self.assertEqual((self.root / 'first' / name).read_bytes(),
                        (self.root / 'second' / name).read_bytes())
    before = (self.root / 'first' / 'manifest.json').read_bytes()
    with self.assertRaises(FileExistsError):
      self.run_prepare(train, heldout, self.root / 'first')
    self.assertEqual(before, (self.root / 'first' / 'manifest.json').read_bytes())

  def test_missing_document_metadata_and_concatenation_are_rejected(self):
    _, heldout = self.valid_caches()
    missing = self.cache('missing.dat', [{'input_ids': [1, 2, 3, 4]}], train=True)
    with self.assertRaisesRegex(ValueError, 'source metadata'):
      self.run_prepare(missing, heldout)
    spec = dict(self.reference, document_boundary_mode='concatenate')
    joined = self.cache('concatenated.dat', [row(1, 'training')],
                         train=True, specification=spec)
    with self.assertRaisesRegex(ValueError, 'source_document'):
      self.run_prepare(joined, heldout)
    self.assertFalse((self.root / 'fresh').exists())

  def test_insufficient_documents_produce_no_output(self):
    train, heldout = self.valid_caches()
    with self.assertRaisesRegex(ValueError, 'insufficient eligible'):
      self.run_prepare(train, heldout, dev_examples=2)
    self.assertFalse((self.root / 'fresh').exists())

  def test_source_order_and_window_are_enforced(self):
    _, heldout = self.valid_caches()
    shuffled = self.cache('shuffled.dat', [row(2, 'first'), row(1, 'second')], train=True)
    with self.assertRaisesRegex(ValueError, 'not monotonic'):
      self.run_prepare(shuffled, heldout, train_examples=2)
    outside = self.cache('outside.dat', [row(110, 'wrong-source')], train=True)
    with self.assertRaisesRegex(ValueError, 'outside its declared'):
      self.run_prepare(outside, heldout)

  def test_pinned_identity_fingerprint_and_distinct_caches(self):
    train, heldout = self.valid_caches()
    with self.assertRaisesRegex(ValueError, 'must be distinct'):
      self.run_prepare(train, train)
    sidecar = train.with_suffix(train.suffix + '.provenance.json')
    provenance = json.loads(sidecar.read_text())
    provenance['observed']['processed_fingerprint'] = 'stale'
    sidecar.write_text(json.dumps(data_provenance.build_manifest(
      specification=provenance['specification'], observed=provenance['observed'])))
    with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
      self.run_prepare(train, heldout)
    changed = dict(self.reference, source_revision='c' * 40)
    different = self.cache('different.dat', [row(1, 'different')],
                            train=True, specification=changed)
    with self.assertRaisesRegex(ValueError, 'pinned reference: source_revision'):
      self.run_prepare(different, heldout)

  def test_defaults_match_pilot_document_counts(self):
    args = _args(['--train-cache', 'a', '--heldout-cache', 'b',
                  '--debug-manifest', 'old.json', '--output-dir', 'new'])
    self.assertEqual((args.train_examples, args.dev_examples, args.test_examples,
                       args.length), (2048, 64, 128, 128))

  def test_minimum_heldout_index_skips_old_benchmark_prefix(self):
    train = self.cache('train.dat', [row(1, 'train')], train=True)
    heldout = self.cache('heldout.dat', [row(i, f'doc-{i}') for i in range(110, 114)])
    result = self.run_prepare(train, heldout, heldout_min_source_index=112)
    self.assertEqual(result['heldout_min_source_index'], 112)
    self.assertEqual(result['splits']['dev']['source_document_index'], [112])
    self.assertEqual(result['splits']['test']['source_document_index'], [113])
    self.assertEqual(result['scan']['heldout']['rejections']['before_minimum_source_index'], 2)

  def test_invalid_or_exhausted_minimum_leaves_no_partial_output(self):
    train, heldout = self.valid_caches()
    for minimum in (-1, 99, 200, 110.5):
      with self.assertRaisesRegex(ValueError, 'within the declared source window'):
        self.run_prepare(train, heldout, heldout_min_source_index=minimum)
    with self.assertRaisesRegex(ValueError, 'insufficient eligible'):
      self.run_prepare(train, heldout, heldout_min_source_index=111)
    self.assertFalse((self.root / 'fresh').exists())


if __name__ == '__main__':
  unittest.main()
