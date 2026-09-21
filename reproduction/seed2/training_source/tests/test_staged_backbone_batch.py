"""Numerical batch audit detects support changes without tuning tolerances."""

import math
from types import SimpleNamespace
import unittest

import torch

from scripts.audit_staged_backbone_batch import audit_model, compare_outputs
from scripts.run_staged_real_overfit import fixed_masks
from tests.test_staged_real_overfit import TinyFrozenBackbone


class StagedBackboneBatchTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(48)
    self.hidden = torch.randn(2, 3, 6)
    self.logits = torch.tensor([
      [[4., 3., 2., 1., -torch.inf], [1., 2., 3., 4., -torch.inf], [4., 3., 2., 1., -torch.inf]],
      [[3., 2., 4., 1., -torch.inf], [1., 2., 3., 4., -torch.inf], [1., 2., 3., 4., -torch.inf]],
    ])
    self.targets = torch.tensor([[0, 1, 2], [1, 2, 3]])
    self.active = torch.tensor([[True, True, False], [True, False, True]])

  def test_identity_has_no_differences_despite_negative_infinity(self):
    report = compare_outputs(self.hidden, self.logits, self.hidden, self.logits,
                              self.targets, self.active, 2)
    self.assertEqual(report['finite_logits_max_absolute_difference'], 0)
    self.assertEqual(report['active_target_logprob_max_absolute_difference_nats'], 0)
    self.assertEqual(report['active_top_k_mean_overlap_fraction'], 1)
    self.assertEqual(report['active_top_k_set_disagreement_fraction'], 0)
    self.assertTrue(report['negative_infinity_masks_match_exactly'])

  def test_changes_to_argmax_and_candidate_support_are_reported(self):
    changed_logits = self.logits.clone()
    changed_logits[0, 0, 2] += 4
    changed_hidden = self.hidden.clone()
    changed_hidden[0, 1, 2] += 0.25
    report = compare_outputs(self.hidden, self.logits, changed_hidden, changed_logits,
                              self.targets, self.active, 2)
    self.assertAlmostEqual(report['hidden_max_absolute_difference'], 0.25)
    self.assertEqual(report['finite_logits_max_absolute_difference'], 4)
    self.assertEqual(report['active_argmax_disagreement_fraction'], 0.25)
    self.assertEqual(report['active_top_k_set_disagreement_fraction'], 0.25)
    self.assertEqual(report['active_top_k_mean_overlap_fraction'], 0.875)
    self.assertGreater(report['active_target_logprob_max_absolute_difference_nats'], 0)
    self.assertNotIn('passed', report)

  def test_audit_uses_same_masks_sigma_and_production_backbone_interface(self):
    model = TinyFrozenBackbone().eval()
    wrapper = SimpleNamespace(backbone=model, noise=model.noise, mask_index=model.mask_index,
                              eval=model.eval,
                              _structured_backbone_output=model._structured_backbone_output)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])
    active = fixed_masks(tokens, 0.5, 1001)
    report = audit_model(wrapper, tokens, active, 0.5, 'cpu', 4)
    self.assertAlmostEqual(report['diffusion_time'], 0.5 / 0.999)
    self.assertAlmostEqual(report['sigma'], math.log(2), places=6)
    self.assertLess(report['finite_logits_max_absolute_difference'], 1e-6)
    self.assertLess(report['hidden_max_absolute_difference'], 1e-6)
    self.assertEqual(report['active_tokens'], 9)
    self.assertTrue(report['comparison_only_no_posthoc_pass_threshold'])
    self.assertEqual(report['production_cuda_autocast_dtype'], 'disabled on CPU')


if __name__ == '__main__':
  unittest.main()
