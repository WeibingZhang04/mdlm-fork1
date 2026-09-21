import hashlib
import json
import unittest

import torch

from evaluation.infilling_prompts import (
  PROMPT_POLICY_ID,
  build_infilling_prompts,
  deterministic_span_start,
  prompt_from_validation_record,
  serialize_prompt_jsonl,
)
from evaluation.generation_harness import prompt_from_record


def _record(document_index=12, chunk_index=3, length=16):
  return {
    'input_ids': torch.arange(length, dtype=torch.long),
    'source_document_index': torch.tensor(document_index),
    'source_document_sha256': hashlib.sha256(
      f'document-{document_index}'.encode()).hexdigest(),
    'source_chunk_index': torch.tensor(chunk_index),
  }


class InfillingPromptsTest(unittest.TestCase):

  def test_span_is_deterministic_and_never_masks_boundaries(self):
    kwargs = {
      'dataset_id': 'wiki-pinned',
      'document_sha256': 'a' * 64,
      'chunk_index': 2,
      'sequence_length': 16,
      'span_length': 5,
      'selection_seed': 101,
    }
    first = deterministic_span_start(**kwargs)
    second = deterministic_span_start(**kwargs)
    self.assertEqual(first, second)
    self.assertGreaterEqual(first, 1)
    self.assertLessEqual(first + 5, 15)

  def test_record_uses_explicit_reference_schema_and_document_provenance(self):
    prompt = prompt_from_validation_record(
      _record(),
      dataset_id='wiki-pinned',
      span_length=4,
      selection_seed=101)
    self.assertEqual(prompt['input_ids'], list(range(16)))
    self.assertEqual(prompt['reference_token_ids'], list(range(16)))
    self.assertEqual(sum(prompt['active_mask']), 4)
    self.assertFalse(prompt['active_mask'][0])
    self.assertFalse(prompt['active_mask'][-1])
    self.assertEqual(
      prompt['metadata']['prompt_policy_id'], PROMPT_POLICY_ID)
    self.assertEqual(prompt['metadata']['source_document_index'], 12)
    self.assertEqual(prompt['metadata']['source_chunk_index'], 3)
    parsed = prompt_from_record(
      prompt,
      tokenizer=None,
      mask_token_id=99,
      sequence_length=16,
      line_number=1)
    self.assertEqual(sum(parsed.active_mask), 4)
    self.assertEqual(
      sum(token == 99 for token in parsed.initial_token_ids), 4)
    self.assertEqual(parsed.reference_token_ids, tuple(range(16)))

  def test_builder_takes_exact_pinned_order_and_serializes_canonically(self):
    prompts = build_infilling_prompts(
      [_record(3, 0), _record(4, 0), _record(5, 0)],
      dataset_id='wiki-pinned',
      span_length=3,
      selection_seed=9,
      num_prompts=2)
    self.assertEqual(len(prompts), 2)
    self.assertIn('document-000000003', prompts[0]['id'])
    payload = serialize_prompt_jsonl(prompts)
    decoded = [json.loads(line) for line in payload.decode().splitlines()]
    self.assertEqual(decoded, prompts)
    self.assertEqual(payload, serialize_prompt_jsonl(prompts))

  def test_missing_or_non_document_local_records_fail_closed(self):
    with self.assertRaisesRegex(ValueError, 'document-local'):
      prompt_from_validation_record(
        {'input_ids': [1, 2, 3]},
        dataset_id='wiki-pinned', span_length=1, selection_seed=0)
    bad = _record()
    bad['source_document_sha256'] = 'not-a-hash'
    with self.assertRaisesRegex(ValueError, 'lowercase SHA256'):
      prompt_from_validation_record(
        bad,
        dataset_id='wiki-pinned', span_length=1, selection_seed=0)
    with self.assertRaisesRegex(ValueError, 'yielded 1 prompts'):
      build_infilling_prompts(
        [_record()],
        dataset_id='wiki-pinned', span_length=1,
        selection_seed=0, num_prompts=2)

  def test_duplicate_document_chunk_identity_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'duplicate'):
      build_infilling_prompts(
        [_record(), _record()],
        dataset_id='wiki-pinned', span_length=2,
        selection_seed=0, num_prompts=2)


if __name__ == '__main__':
  unittest.main()
