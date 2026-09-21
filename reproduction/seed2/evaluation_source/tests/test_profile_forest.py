import unittest
from pathlib import Path

import torch
import yaml

from scripts.profile_forest_inference import (
  DEFAULT_MEASURED_REPETITIONS,
  DEFAULT_WARMUP_REPETITIONS,
  _args,
  bounded_chain_edges,
  cuda_provenance,
  profile_shape,
  profile_protocol,
  verify_dense_low_rank_agreement,
)


class ForestProfileTest(unittest.TestCase):

  def test_reported_timing_defaults_are_explicit_and_config_aligned(self):
    args = _args(['--output', '/tmp/profile-unused.json'])
    self.assertEqual(DEFAULT_WARMUP_REPETITIONS, 3)
    self.assertEqual(DEFAULT_MEASURED_REPETITIONS, 10)
    self.assertEqual(args.warmup, 3)
    self.assertEqual(args.repetitions, 10)
    protocol = profile_protocol(
      args.warmup, args.repetitions, torch.device('cuda'))
    self.assertEqual(protocol['warmup_repetitions_per_backend'], 3)
    self.assertEqual(protocol['measured_repetitions_per_backend'], 10)
    self.assertEqual(
      protocol['cuda_synchronization'],
      'before_and_after_each_backend_timing_block')
    self.assertIn(
      'only its required steady-state inputs',
      protocol['peak_memory_scope'])
    self.assertIn(
      'released before dense warmup',
      protocol['dense_construction_memory'])

    config_path = (
      Path(__file__).resolve().parents[1]
      / 'configs/experiment/contextual-forest-g1.yaml')
    profile = yaml.safe_load(config_path.read_text())['profile']
    self.assertEqual(profile['warmup_repetitions'], 3)
    self.assertEqual(profile['measured_repetitions'], 10)
    self.assertEqual(
      profile['cuda_synchronization'],
      'before_and_after_each_backend_timing_block')
    self.assertIsNone(cuda_provenance(torch.device('cpu')))

  def test_profile_protocol_rejects_invalid_repetition_counts(self):
    with self.assertRaisesRegex(ValueError, 'nonnegative'):
      profile_protocol(-1, 10, torch.device('cpu'))
    with self.assertRaisesRegex(ValueError, 'positive'):
      profile_protocol(3, 0, torch.device('cpu'))

  def test_bounded_chain_splits_components(self):
    edges, mask = bounded_chain_edges(10, 3, torch.device('cpu'))
    selected = edges[mask].tolist()
    self.assertEqual(len(selected), 6)
    self.assertNotIn([2, 3], selected)
    self.assertNotIn([5, 6], selected)
    self.assertNotIn([8, 9], selected)

  def test_profile_reports_backend_specific_input_accounting(self):
    profile = profile_shape(
      length=6,
      candidate_count=4,
      rank=3,
      component_size=3,
      warmup=0,
      repetitions=1,
      device=torch.device('cpu'))
    self.assertIsNone(profile['low_rank_peak_bytes'])
    self.assertIsNone(profile['dense_peak_bytes'])
    # Low-rank: node + two endpoint-factor tensors + edges + masks.
    self.assertEqual(profile['low_rank_steady_state_input_bytes'], 715)
    # Dense: node + one (K+1)^2 log-factor tensor + edges + masks.  The
    # low-rank construction factors and raw (pre-log) dense tensor are absent.
    self.assertEqual(profile['dense_steady_state_input_bytes'], 735)

  def test_profile_reference_agreement(self):
    errors = verify_dense_low_rank_agreement()
    self.assertLess(errors['max_log_partition_error'], 1e-10)
    self.assertLess(errors['max_node_marginal_error'], 1e-10)


if __name__ == '__main__':
  unittest.main()
