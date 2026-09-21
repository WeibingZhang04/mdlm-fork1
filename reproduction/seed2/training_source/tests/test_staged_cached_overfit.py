import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.run_staged_cached_overfit import replay_error, verified_cache
from scripts.run_staged_real_overfit import CACHE_KEYS


class StagedCachedOverfitTest(unittest.TestCase):
  def test_cache_contents_and_active_count_are_verified(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      record = {key: torch.ones(2) for key in CACHE_KEYS}
      record['active'] = torch.tensor([True, False])
      digest = hashlib.sha256()
      for key in CACHE_KEYS:
        digest.update(key.encode())
        digest.update(record[key].numpy().tobytes())
      torch.save(record, path / 'example-0000.pt')
      expected = {'examples': 1, 'active_tokens': 1, 'cache_sha256': digest.hexdigest()}
      cache, report = verified_cache(path, expected)
      self.assertEqual(len(cache), 1)
      self.assertTrue(report['matches_original'])
      record['hidden'][0] = 2
      torch.save(record, path / 'example-0000.pt')
      with self.assertRaisesRegex(ValueError, 'tensor contents'):
        verified_cache(path, expected)

  def test_missing_cache_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'example count'):
        verified_cache(Path(directory), {'examples': 1})

  def test_prefix_replay_uses_predeclared_tolerance(self):
    scores = {key: 1.0 for key in (
      'backbone_nll_per_masked_token', 'joint_nll_per_masked_token',
      'marginal_nll_per_masked_token', 'dependence_gain_nats_per_masked_token')}
    row = {'step': 500, 'train': dict(scores), 'dev': dict(scores)}
    self.assertTrue(replay_error(row, row, 1e-4)['passed'])
    other = {'step': 500, 'train': dict(scores), 'dev': dict(scores)}
    other['dev']['joint_nll_per_masked_token'] += 0.01
    with self.assertRaisesRegex(AssertionError, 'failed to replay'):
      replay_error(other, row, 1e-4)


if __name__ == '__main__':
  unittest.main()
