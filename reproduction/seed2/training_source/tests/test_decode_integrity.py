import json
from pathlib import Path
import tempfile
import unittest

from evaluation.decode_integrity import (
  audit_decode_integrity,
  audit_token_ids,
  canonical_sha256,
  resolve_sample_paths,
  tokenizer_identity,
)


class FakeTokenizer:
  """Small byte-failure fixture with exact and non-exact token mappings."""

  vocab_size = 10
  bos_token_id = None
  eos_token_id = 0
  pad_token_id = 0
  mask_token_id = 99

  _decode = {
    1: 'a',
    2: 'b',
    3: '\ufffd',
    4: 'xy',
    5: 'x',
    6: 'y',
    9: '\ufffd',
  }
  _encode = {
    'a': 1,
    'b': 2,
    'x': 5,
    'y': 6,
    '\ufffd': 9,
  }

  def decode(self, token_ids, **kwargs):
    if kwargs != {
        'skip_special_tokens': False,
        'clean_up_tokenization_spaces': False,
    }:
      raise AssertionError(f'unexpected decode policy: {kwargs}')
    return ''.join(self._decode[token] for token in token_ids)

  def __call__(self, text, **kwargs):
    if kwargs != {
        'add_special_tokens': False,
        'return_attention_mask': False,
    }:
      raise AssertionError(f'unexpected encode policy: {kwargs}')
    return {'input_ids': [self._encode[character] for character in text]}


def _record(
    *, sample_index, tokens, active_mask, mode='structured_joint', nfe=64,
):
  return {
    'sample_index': sample_index,
    'pair_key': f'pair-{sample_index}',
    'pair_seed': 700 + sample_index,
    'prompt_id': f'prompt-{sample_index}',
    'prompt_metadata': {'dataset_id': 'wiki-pinned'},
    'sampling_mode': mode,
    'requested_nfe_budget': nfe,
    'measured_nfe': nfe,
    'sample_token_ids': tokens,
    'active_mask': active_mask,
    'sample_active_token_ids': [
      token for token, active in zip(tokens, active_mask) if active
    ],
  }


class DecodeIntegrityTest(unittest.TestCase):

  def setUp(self):
    self.tokenizer = FakeTokenizer()
    self.tokenizer_identity = {
      'name_or_path': 'fake/tokenizer',
      'requested_revision': '1' * 40,
    }

  def test_single_scope_reports_replacement_and_length_mismatch(self):
    result = audit_token_ids(self.tokenizer, [3, 4])
    self.assertTrue(result['contains_replacement_character'])
    self.assertEqual(result['replacement_character_count'], 1)
    self.assertEqual(result['replacement_character_codepoint_rate'], 1 / 3)
    self.assertTrue(result['roundtrip_mismatch'])
    self.assertEqual(result['retokenized_token_count'], 3)
    self.assertEqual(result['token_length_delta'], 1)
    self.assertEqual(result['token_length_ratio'], 1.5)
    self.assertEqual(result['first_roundtrip_mismatch'], {
      'index': 0,
      'raw_token_id': 3,
      'retokenized_token_id': 9,
    })

  def test_directory_audit_groups_records_and_is_deterministic(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      first_dir = root / 'shard-b'
      second_dir = root / 'shard-a'
      first_dir.mkdir()
      second_dir.mkdir()
      records = [
        _record(
          sample_index=0, tokens=[1, 2], active_mask=[True, True],
          mode='structured_joint', nfe=64),
        _record(
          sample_index=1, tokens=[3, 4], active_mask=[True, True],
          mode='structured_marginal', nfe=32),
      ]
      (first_dir / 'samples.jsonl').write_text(
        json.dumps(records[1], sort_keys=True) + '\n', encoding='utf-8')
      (second_dir / 'samples.jsonl').write_text(
        json.dumps(records[0], sort_keys=True) + '\n', encoding='utf-8')

      result = audit_decode_integrity(
        [root], tokenizer=self.tokenizer,
        tokenizer_identity=self.tokenizer_identity)
      repeated = audit_decode_integrity(
        [root], tokenizer=self.tokenizer,
        tokenizer_identity=self.tokenizer_identity)

      self.assertEqual(result, repeated)
      self.assertEqual(result['aggregate']['num_records'], 2)
      full = result['aggregate']['scopes']['sample_token_ids']
      self.assertEqual(full['records_with_replacement_character'], 1)
      self.assertEqual(full['roundtrip_mismatch_records'], 1)
      self.assertEqual(full['roundtrip_mismatch_rate'], 0.5)
      self.assertEqual(full['token_length_delta_sum'], 1)
      self.assertEqual(len(result['groups']), 2)
      dimensions = [group['dimensions'] for group in result['groups']]
      self.assertIn({
        'dataset_id': 'wiki-pinned',
        'sampling_mode': 'structured_joint',
        'requested_nfe_budget': 64,
      }, dimensions)
      diagnostic = result['records'][0]
      self.assertEqual(len(diagnostic['source']['sha256']), 64)
      self.assertEqual(len(diagnostic['source']['record_sha256']), 64)
      self.assertEqual(len(diagnostic['diagnostic_key_sha256']), 64)
      self.assertEqual(len(result['audit_sha256']), 64)
      without_digest = dict(result)
      without_digest.pop('audit_sha256')
      self.assertEqual(result['audit_sha256'], canonical_sha256(without_digest))

  def test_input_paths_are_sorted_and_duplicate_paths_are_deduplicated(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      nested = root / 'nested'
      nested.mkdir()
      sample_path = nested / 'samples.jsonl'
      sample_path.write_text('{}\n', encoding='utf-8')
      self.assertEqual(
        resolve_sample_paths([sample_path, root]), [sample_path.resolve()])

  def test_missing_or_inconsistent_raw_ids_fail_loudly(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'samples.jsonl'
      malformed = _record(
        sample_index=0, tokens=[1, 2], active_mask=[True, False])
      malformed.pop('sample_active_token_ids')
      path.write_text(json.dumps(malformed) + '\n', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'missing required raw token IDs'):
        audit_decode_integrity(
          [path], tokenizer=self.tokenizer,
          tokenizer_identity=self.tokenizer_identity)

  def test_tokenizer_identity_requires_and_checks_exact_revision(self):
    self.tokenizer.init_kwargs = {'_commit_hash': 'a' * 40}
    identity = tokenizer_identity(
      self.tokenizer,
      name_or_path='fake/tokenizer',
      requested_revision='A' * 40)
    self.assertEqual(identity['requested_revision'], 'a' * 40)
    self.assertEqual(identity['resolved_revision'], 'a' * 40)
    with self.assertRaisesRegex(ValueError, 'exact 40-character'):
      tokenizer_identity(
        self.tokenizer,
        name_or_path='fake/tokenizer',
        requested_revision='main')
    with self.assertRaisesRegex(ValueError, 'differs'):
      tokenizer_identity(
        self.tokenizer,
        name_or_path='fake/tokenizer',
        requested_revision='b' * 40)

      malformed['sample_active_token_ids'] = [2]
      path.write_text(json.dumps(malformed) + '\n', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'is inconsistent'):
        audit_decode_integrity(
          [path], tokenizer=self.tokenizer,
          tokenizer_identity=self.tokenizer_identity)


if __name__ == '__main__':
  unittest.main()
