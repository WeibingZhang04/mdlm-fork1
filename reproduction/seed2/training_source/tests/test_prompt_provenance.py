import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import data_provenance
from evaluation.infilling_prompts import (
  PROMPT_POLICY_ID,
  deterministic_span_start,
  serialize_prompt_jsonl,
)
from evaluation.prompt_provenance import validate_prompt_bundle


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class PromptProvenanceTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.config = self.root / 'eval_fixture.yaml'
    self.config.write_text('valid: wiki-pinned\n')
    self.prompt = self.root / 'prompts.jsonl'
    self.runtime = self.root / 'valid-runtime.json'
    self.manifest = self.root / 'prompts.jsonl.manifest.json'

    document_sha = hashlib.sha256(b'document-7').hexdigest()
    span_start = deterministic_span_start(
      dataset_id='wiki-pinned',
      document_sha256=document_sha,
      chunk_index=0,
      sequence_length=8,
      span_length=2,
      selection_seed=31)
    prompt = {
      'id': 'wiki-pinned/document-000000007/chunk-00000/span-0002',
      'input_ids': list(range(8)),
      'active_mask': [
        span_start <= index < span_start + 2 for index in range(8)],
      'reference_token_ids': list(range(8)),
      'metadata': {
        'prompt_policy_id': PROMPT_POLICY_ID,
        'dataset_id': 'wiki-pinned',
        'source_document_index': 7,
        'source_document_sha256': document_sha,
        'source_chunk_index': 0,
        'sequence_length': 8,
        'span_start': span_start,
        'span_stop': span_start + 2,
        'span_length': 2,
        'selection_seed': 31,
      },
    }
    self.prompt.write_bytes(serialize_prompt_jsonl([prompt]))
    specification = {
      'logical_dataset_name': 'wiki-pinned',
      'source_revision': 'a' * 40,
      'tokenizer_name_or_path': 'openai-community/gpt2',
      'tokenizer_revision': 'b' * 40,
      'block_size': 8,
      'document_boundary_mode': 'wikitext_articles',
    }
    self.runtime_payload = data_provenance.build_manifest(
      specification=specification,
      observed={'num_processed_rows': 1})
    self._write_runtime()
    self._write_manifest()

  def tearDown(self):
    self.temporary.cleanup()

  def _write_runtime(self):
    self.runtime.write_text(json.dumps(
      self.runtime_payload, indent=2, sort_keys=True) + '\n')

  def _manifest_payload(self):
    return {
      'schema_version': 2,
      'artifact': 'pinned_document_local_infilling_prompts',
      'created_utc': '2026-08-30T00:00:00+00:00',
      'command': ['build_infilling_prompts.py'],
      'repository': {'git_sha': 'c' * 40, 'clean': True},
      'data_config': {
        'name': 'eval_fixture',
        'path': str(self.config),
        'sha256': _sha(self.config),
        'logical_validation_dataset': 'wiki-pinned',
        'dataset_revision': 'a' * 40,
        'tokenizer_name_or_path': 'openai-community/gpt2',
        'tokenizer_revision': 'b' * 40,
      },
      'runtime_provenance': {
        'path': str(self.runtime),
        'sha256': _sha(self.runtime),
      },
      'policy': {
        'policy_id': PROMPT_POLICY_ID,
        'selection_seed': 31,
        'span_length': 2,
        'sequence_length': 8,
        'record_selection': 'first_n_in_pinned_validation_order',
        'boundary_policy': 'never_mask_first_or_last_token',
      },
      'output': {
        'path': str(self.prompt),
        'sha256': _sha(self.prompt),
        'size_bytes': self.prompt.stat().st_size,
        'num_prompts': 1,
      },
      'model_weights_loaded': False,
    }

  def _write_manifest(self):
    self.manifest.write_text(json.dumps(
      self._manifest_payload(), indent=2, sort_keys=True) + '\n')

  def _validate(self):
    return validate_prompt_bundle(
      self.prompt,
      self.manifest,
      expected_manifest_sha256=_sha(self.manifest),
      expected_data_config='eval_fixture',
      expected_sequence_length=8)

  def test_validates_complete_prompt_dataset_and_runtime_bundle(self):
    identity = self._validate()
    self.assertEqual(identity['output']['num_prompts'], 1)
    self.assertEqual(
      identity['data_config']['dataset_revision'], 'a' * 40)
    self.assertEqual(identity['manifest_sha256'], _sha(self.manifest))

  def test_rejects_semantic_prompt_forgery_even_after_rehash(self):
    record = json.loads(self.prompt.read_text())
    record['active_mask'][record['metadata']['span_start']] = False
    self.prompt.write_text(json.dumps(record, sort_keys=True) + '\n')
    self._write_manifest()
    with self.assertRaisesRegex(ValueError, 'deterministic committed span'):
      self._validate()

  def test_rejects_runtime_semantic_drift_even_after_rehash(self):
    specification = self.runtime_payload['specification']
    specification['source_revision'] = 'd' * 40
    self.runtime_payload['specification_sha256'] = (
      data_provenance.canonical_sha256(specification))
    body = dict(self.runtime_payload)
    body.pop('manifest_sha256')
    self.runtime_payload['manifest_sha256'] = (
      data_provenance.canonical_sha256(body))
    self._write_runtime()
    self._write_manifest()
    with self.assertRaisesRegex(ValueError, 'source_revision differs'):
      self._validate()

  def test_rejects_data_config_byte_drift(self):
    self.config.write_text('valid: forged\n')
    with self.assertRaisesRegex(ValueError, 'data config bytes'):
      self._validate()


if __name__ == '__main__':
  unittest.main()
