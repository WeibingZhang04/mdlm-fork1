"""Tests for local-only candidates; never enables them in production."""
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

import structured_utils as utils
import models.structured_decoder as decoder
from scripts.audit_ccf_sampling_v3 import unchecked_rows, experiment, list_kruskal
from scripts.verify_ccf_sampling_optimization import verify
from scripts import verify_ccf_sampling_v3_gpu as gpu_runner
from evaluation import generation_harness as harness


class LocalSpeedCandidatesTest(unittest.TestCase):
  def test_full_generation_runner_checks_actual_sample_field_and_restores_patches(self):
    original = utils._sample_rows
    calls = []
    def fake_run(model, specs, **kwargs):
      calls.append((specs[0].pair_seed, kwargs['sampling_mode'], utils._sample_rows is original))
      self.assertEqual(kwargs['nfe_budget'], 1001)
      return [{'sample_token_ids': [7, 8], 'text': 'test'}], {'measured_nfe': 1000, 'wall_clock_seconds': 1.0}
    with TemporaryDirectory() as folder, \
         mock.patch.object(gpu_runner, 'load_model', return_value=(SimpleNamespace(mask_index=9), None, {})), \
         mock.patch.object(harness, 'run_sampling_group', side_effect=fake_run), \
         mock.patch.object(torch.cuda, 'synchronize'), \
         mock.patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([1], dtype=torch.uint8)):
      report = {}
      gpu_runner.full_samples('unchecked', Path(folder), report, lambda: None)
      self.assertEqual(len(report['full_samples']), 2)
      self.assertTrue(all(r['tokens_equal'] and r['rng_equal'] and r['nfe_equal'] for r in report['full_samples']))
    self.assertEqual(calls, [(91001, 'structured_joint', True), (91001, 'structured_joint', False),
                            (91002, 'structured_joint', False), (91002, 'structured_joint', True),
                            (91001, 'factorized', True)])
    self.assertIs(utils._sample_rows, original)

  def test_unchecked_draw_preserves_tokens_and_rng(self):
    for dtype in (torch.float32, torch.float64):
      for rows in (1, 3, 32):
        for width in (5, 33, 129, 257):
          for seed in (0, 1, 91001, 91002):
            torch.manual_seed(seed)
            logits = torch.randn(rows, width, dtype=dtype) * 20
            logits[:, 0] = -torch.inf
            old = torch.Generator().manual_seed(seed)
            new = torch.Generator().manual_seed(seed)
            self.assertTrue(torch.equal(utils._sample_rows(logits, old), unchecked_rows(logits, new)))
            self.assertTrue(torch.equal(old.get_state(), new.get_state()))

  def test_all_candidates_preserve_existing_cpu_cases_and_restore_functions(self):
    original_draw = utils._sample_rows
    original_infer = utils._single_low_rank_sum_product
    for mode in ('unchecked', 'unchecked_no_validation', 'unchecked_batched',
                 'unchecked_batched_no_validation', 'unchecked_batched_roots_kruskal'):
      with experiment(mode):
        self.assertEqual(verify(torch.device('cpu'))['equivalence_cases'], 54)
      self.assertIs(utils._sample_rows, original_draw)
      self.assertIs(utils._single_low_rank_sum_product, original_infer)

  def test_list_kruskal_preserves_ties_caps_masks_and_threshold(self):
    edges = torch.tensor([[[0, 1], [1, 2], [2, 3], [0, 3], [1, 3], [3, 4], [1, 0]]]).expand(2, -1, -1)
    scores = torch.tensor([[1., 1., 1., .5, .5, .1, 1.], [1., .5, .5, 1., 1., .1, 1.]])
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 3] = False
    active = torch.ones(2, 5, dtype=torch.bool)
    candidate = list_kruskal()
    for cap in (0, 2, 3, 32):
      for threshold in (None, .2, 1.1):
        args = (edges, scores, mask, active, cap, threshold)
        old = decoder._bounded_kruskal_indices(*args)
        new = candidate(*args)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, new)))


if __name__ == '__main__':
  torch.set_num_threads(1)
  unittest.main()
