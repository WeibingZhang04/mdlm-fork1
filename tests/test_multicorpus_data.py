import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import datasets
import fsspec
import yaml

import data_provenance


def _load_dataloader_module():
  fake_utils = types.ModuleType('utils')
  fake_utils.get_logger = logging.getLogger
  fake_utils.fsspec_exists = lambda path: fsspec.core.url_to_fs(path)[0].exists(
    fsspec.core.url_to_fs(path)[1])
  fake_utils.fsspec_mkdirs = lambda path, exist_ok=True: os.makedirs(
    path, exist_ok=exist_ok)
  module_path = Path(__file__).resolve().parents[1] / 'dataloader.py'
  spec = importlib.util.spec_from_file_location(
    '_multicorpus_test_dataloader', module_path)
  module = importlib.util.module_from_spec(spec)
  missing = object()
  previous = sys.modules.get('utils', missing)
  sys.modules['utils'] = fake_utils
  try:
    with mock.patch.object(
        os, 'sched_getaffinity', create=True, return_value={0}):
      spec.loader.exec_module(module)
  finally:
    if previous is missing:
      del sys.modules['utils']
    else:
      sys.modules['utils'] = previous
  return module


dataloader = _load_dataloader_module()


class _CharacterTokenizer:
  bos_token = '<bos>'
  eos_token = '<eos>'
  vocab_size = 256

  def encode(self, token):
    return [1 if token == self.bos_token else 2]

  def __call__(self, texts, **_kwargs):
    return {'input_ids': [
      [10 if character == 'a' else 20 for character in text]
      for text in texts
    ]}


class MulticorpusDataTest(unittest.TestCase):

  def test_exact_revisions_and_disjoint_half_open_windows(self):
    with self.assertRaisesRegex(ValueError, 'exact 40-character'):
      data_provenance.require_commit_revision('main', field='revision')
    proof = data_provenance.disjoint_window_proof(
      dataset_name_or_path='Skylion007/openwebtext',
      dataset_config_name='plain_text', split='train',
      revision='a' * 40, source_num_rows=100,
      train_window=[0, 90], heldout_window=[90, 100])
    self.assertEqual(proof['overlap_num_rows'], 0)
    self.assertEqual(len(proof['proof_sha256']), 64)
    with self.assertRaisesRegex(ValueError, 'overlaps'):
      data_provenance.disjoint_window_proof(
        dataset_name_or_path='Skylion007/openwebtext',
        dataset_config_name='plain_text', split='train',
        revision='a' * 40, source_num_rows=100,
        train_window=[0, 91], heldout_window=[90, 100])

  def test_document_windows_do_not_cross_source_rows_and_are_provenanced(self):
    source = datasets.Dataset.from_dict({
      'text': ['a' * 12, 'b' * 12],
    })
    tokenizer = _CharacterTokenizer()
    with tempfile.TemporaryDirectory() as directory:
      cache_dir = Path(directory) / 'cache'
      provenance_dir = Path(directory) / 'run-provenance'
      with mock.patch.object(
          dataloader.datasets, 'load_dataset', return_value=source) as load:
        result = dataloader.get_dataset(
          'fake-pinned', tokenizer=tokenizer, wrap=True,
          mode='validation', cache_dir=str(cache_dir), block_size=6,
          num_proc=1, streaming=False, revision='a' * 40,
          dataset_name_or_path='owner/corpus',
          dataset_config_name='default', source_split='validation',
          expected_source_num_rows=2, text_field='text',
          document_boundary_mode='source_document',
          require_pinned_provenance=True,
          tokenizer_name_or_path='owner/tokenizer',
          tokenizer_revision='b' * 40,
          provenance_dir=str(provenance_dir),
          provenance_role='valid')

      load.assert_called_once_with(
        'owner/corpus', name='default', split='validation',
        cache_dir=str(cache_dir), streaming=False, revision='a' * 40)
      self.assertEqual(len(result), 6)
      for row in result:
        interior = set(row['input_ids'][1:-1].tolist())
        self.assertIn(interior, ({10}, {20}))
        self.assertEqual(int(row['attention_mask'].sum()), 6)
      self.assertEqual(
        result[0]['source_document_sha256'],
        result[1]['source_document_sha256'])
      self.assertNotEqual(
        result[0]['source_document_sha256'],
        result[3]['source_document_sha256'])

      runtime_manifests = list(provenance_dir.glob('valid-*.json'))
      self.assertEqual(len(runtime_manifests), 1)
      manifest = json.loads(runtime_manifests[0].read_text())
      validated = data_provenance.validate_manifest(
        manifest, expected_specification=manifest['specification'])
      self.assertEqual(validated['observed']['source_num_rows'], 2)
      self.assertEqual(validated['observed']['processed_num_sequences'], 6)
      cache_manifests = list(cache_dir.glob('*.provenance.json'))
      self.assertEqual(len(cache_manifests), 1)

      # A second caller must validate and reuse the persisted Arrow dataset.
      # Dataset fingerprints can change across save/load, so this specifically
      # guards against recording the transient pre-save fingerprint.
      with mock.patch.object(
          dataloader.datasets, 'load_dataset',
          side_effect=AssertionError('source dataset should not be reloaded')):
        cached_result = dataloader.get_dataset(
          'fake-pinned', tokenizer=tokenizer, wrap=True,
          mode='validation', cache_dir=str(cache_dir), block_size=6,
          num_proc=1, streaming=False, revision='a' * 40,
          dataset_name_or_path='owner/corpus',
          dataset_config_name='default', source_split='validation',
          expected_source_num_rows=2, text_field='text',
          document_boundary_mode='source_document',
          require_pinned_provenance=True,
          tokenizer_name_or_path='owner/tokenizer',
          tokenizer_revision='b' * 40,
          provenance_dir=str(provenance_dir),
          provenance_role='valid')
      self.assertEqual(len(cached_result), len(result))
      cached_manifest = json.loads(cache_manifests[0].read_text())
      raw_cached_result = datasets.load_from_disk(
        str(cache_manifests[0]).removesuffix('.provenance.json'))
      self.assertEqual(
        cached_manifest['observed']['processed_fingerprint'],
        raw_cached_result._fingerprint)

      cache_manifests[0].unlink()
      with self.assertRaisesRegex(RuntimeError, 'without provenance'):
        dataloader.get_dataset(
          'fake-pinned', tokenizer=tokenizer, wrap=True,
          mode='validation', cache_dir=str(cache_dir), block_size=6,
          num_proc=1, streaming=False, revision='a' * 40,
          dataset_name_or_path='owner/corpus',
          dataset_config_name='default', source_split='validation',
          expected_source_num_rows=2, text_field='text',
          document_boundary_mode='source_document',
          require_pinned_provenance=True,
          tokenizer_name_or_path='owner/tokenizer',
          tokenizer_revision='b' * 40,
          provenance_dir=str(provenance_dir),
          provenance_role='valid')

  def test_wikitext_article_recovery_never_merges_level_one_headings(self):
    source = datasets.Dataset.from_dict({'text': [
      ' = First = \n', 'a' * 8 + '\n', 'a' * 8 + '\n',
      ' = = Section = = \n', 'a' * 8 + '\n',
      ' = Second = \n', 'b' * 8 + '\n', 'b' * 8 + '\n',
    ]})
    recovered = dataloader._coalesce_wikitext_articles(source)
    self.assertEqual(len(recovered), 2)
    self.assertIn('First', recovered[0]['text'])
    self.assertIn('Section', recovered[0]['text'])
    self.assertNotIn('Second', recovered[0]['text'])
    self.assertIn('Second', recovered[1]['text'])

  def test_submission_data_configs_pin_all_mutable_sources(self):
    config_dir = Path(__file__).resolve().parents[1] / 'configs' / 'data'
    names = (
      'train_openwebtext_pinned.yaml',
      'eval_wikitext103_pinned.yaml',
      'eval_openwebtext_heldout_pinned.yaml',
      'eval_scientific_papers_arxiv_pinned.yaml',
      'eval_scientific_papers_pubmed_pinned.yaml',
    )
    for name in names:
      with self.subTest(name=name):
        payload = yaml.safe_load((config_dir / name).read_text())
        self.assertTrue(payload['require_pinned_provenance'])
        self.assertFalse(payload['shuffle_validation'])
        self.assertRegex(payload['tokenizer_revision'], r'^[0-9a-f]{40}$')
        self.assertRegex(payload['train_revision'], r'^[0-9a-f]{40}$')
        self.assertRegex(payload['valid_revision'], r'^[0-9a-f]{40}$')
        self.assertNotEqual(
          payload['valid_document_boundary_mode'], 'concatenate')
        self.assertEqual(
          payload['train_document_boundary_mode'], 'concatenate')
        self.assertEqual(
          payload['train_dataset_name_or_path'],
          'Skylion007/openwebtext')


if __name__ == '__main__':
  unittest.main()
