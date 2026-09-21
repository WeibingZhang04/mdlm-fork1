from pathlib import Path
import unittest
from unittest import mock

import torch

from scripts.replay_generation_trajectory import (
  SELECTION_POLICY,
  _batch_selection_sha256,
  paired_samples_from_records,
  resolve_device,
  select_outcome_independent_batch,
)


def _record(index, mode, nfe=64):
  return {
    'sample_index': index,
    'pair_key': f'pair-{index:03d}',
    'pair_seed': 9000 + index,
    'prompt_id': f'prompt-{index:03d}',
    'prompt_metadata': {'dataset_id': 'test'},
    'sampling_mode': mode,
    'requested_nfe_budget': nfe,
    'measured_nfe': nfe,
    'batch_seed': 123,
    'initial_token_ids': [1, 99, 2],
    'active_mask': [False, True, False],
    'reference_token_ids': [1, 7, 2],
    'sample_token_ids': [1, 7, 2],
    'sample_active_token_ids': [7],
    'metrics': {},
    'timing': {
      'batch_seed': 123,
      'batch_size': 2,
      'requested_nfe_budget': nfe,
      'measured_nfe': nfe,
    },
  }


def _loaded(shard_index, indices):
  marginal = [_record(index, 'structured_marginal') for index in indices]
  joint = [_record(index, 'structured_joint') for index in indices]
  return {
    'manifest_path': Path(f'/source/shard-{shard_index:02d}/manifest.json'),
    'manifest_sha256': f'{shard_index + 1:064x}',
    'manifest': {
      'pairing': {'batch_size': 2, 'shard_index': shard_index},
      'outputs': {'samples_jsonl': {'sha256': f'{shard_index + 9:064x}'}},
    },
    'groups': {
      ('structured_marginal', 64): marginal,
      ('structured_joint', 64): joint,
    },
  }


class ReplayGenerationTrajectoryCliTest(unittest.TestCase):

  def test_selection_is_hash_min_and_independent_of_mode_outcomes(self):
    shards = [_loaded(0, [0, 16, 32, 48]), _loaded(1, [1, 17, 33, 49])]
    selected = select_outcome_independent_batch(
      shards,
      modes=['structured_marginal', 'structured_joint'],
      nfe_budget=64)
    candidates = []
    for loaded in shards:
      records = loaded['groups'][('structured_marginal', 64)]
      for offset in (0, 2):
        batch = records[offset:offset + 2]
        candidates.append((
          _batch_selection_sha256(batch),
          [record['sample_index'] for record in batch]))
    expected_hash, expected_indices = min(candidates)
    self.assertEqual(selected['selection_policy'], SELECTION_POLICY)
    self.assertEqual(selected['selection_sha256'], expected_hash)
    self.assertEqual(selected['sample_indices'], expected_indices)
    self.assertEqual(selected['num_eligible_full_batches'], 4)
    self.assertEqual(
      [record['sample_index'] for record in
       selected['records_by_mode']['structured_joint']],
      expected_indices)

  def test_partial_final_batch_is_not_eligible(self):
    loaded = _loaded(0, [0, 16, 32])
    selected = select_outcome_independent_batch(
      [loaded], modes=['structured_joint'], nfe_budget=64)
    self.assertEqual(selected['num_eligible_full_batches'], 1)
    self.assertEqual(selected['sample_indices'], [0, 16])

  def test_missing_mode_or_duplicate_modes_fails(self):
    loaded = _loaded(0, [0, 16])
    with self.assertRaisesRegex(ValueError, 'unique'):
      select_outcome_independent_batch(
        [loaded], modes=['structured_joint', 'structured_joint'],
        nfe_budget=64)
    with self.assertRaisesRegex(ValueError, 'lacks replay group'):
      select_outcome_independent_batch(
        [loaded], modes=['structured_joint', 'factorized'], nfe_budget=64)

  def test_record_reconstruction_preserves_full_identity(self):
    records = [_record(0, 'structured_joint'), _record(16, 'structured_joint')]
    samples = paired_samples_from_records(records)
    self.assertEqual([sample.sample_index for sample in samples], [0, 16])
    self.assertEqual(samples[0].pair_key, 'pair-000')
    self.assertEqual(samples[0].prompt.initial_token_ids, (1, 99, 2))
    self.assertEqual(samples[0].prompt.active_mask, (False, True, False))
    self.assertEqual(samples[0].prompt.reference_token_ids, (1, 7, 2))
    self.assertEqual(samples[0].prompt.metadata, {'dataset_id': 'test'})

  def test_device_alias_is_canonicalized_for_strict_replay_check(self):
    self.assertEqual(resolve_device('cpu'), torch.device('cpu'))
    with mock.patch('torch.cuda.is_available', return_value=True), \
         mock.patch('torch.cuda.current_device', return_value=2):
      self.assertEqual(resolve_device('cuda'), torch.device('cuda:2'))
      self.assertEqual(resolve_device('cuda:1'), torch.device('cuda:1'))
    with mock.patch('torch.cuda.is_available', return_value=False):
      with self.assertRaisesRegex(RuntimeError, 'unavailable'):
        resolve_device('cuda')


if __name__ == '__main__':
  unittest.main()
