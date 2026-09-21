import unittest

import torch

from crf_utils import factorized_confidence_gated_update
from structured_training import (
  uses_structured_token_distribution,
  validate_structured_sampling_mode,
)


def _probabilities():
  return torch.tensor(
    [[
      [0.55, 0.20, 0.15, 0.10, 0.00],
      [0.10, 0.80, 0.05, 0.05, 0.00],
      [0.25, 0.25, 0.25, 0.25, 0.00],
      [0.10, 0.10, 0.70, 0.10, 0.00],
    ]])


class FactorizedConfidenceGatedTest(unittest.TestCase):

  def test_mode_is_factorized_identity_sampling_not_structured_sampling(self):
    self.assertEqual(
      validate_structured_sampling_mode('factorized_confidence_gated'),
      'factorized_confidence_gated')
    self.assertFalse(
      uses_structured_token_distribution('factorized_confidence_gated'))
    self.assertTrue(
      uses_structured_token_distribution('structured_marginal'))
    self.assertTrue(uses_structured_token_distribution('structured_joint'))

  def test_update_reveals_highest_confidence_with_matched_count(self):
    mask_index = 4
    x = torch.full((1, 4), mask_index, dtype=torch.long)
    torch.manual_seed(123)
    updated = factorized_confidence_gated_update(
      x=x,
      probabilities=_probabilities(),
      mask_index=mask_index,
      move_chance_t=1 - torch.exp(torch.tensor(-1.0)),
      move_chance_s=1 - torch.exp(torch.tensor(-0.5)))

    # At sigma_t=1 and sigma_s=0.5, the schedule retains ceil(4 * 0.622...)
    # = 3 masks, hence reveals exactly one site. Position 1 has the highest
    # factorized confidence and must be the selected site.
    revealed = updated.ne(mask_index)
    self.assertEqual(int(revealed.sum()), 1)
    self.assertTrue(bool(revealed[0, 1]))

  def test_observed_tokens_are_never_changed(self):
    mask_index = 4
    x = torch.tensor([[3, mask_index, mask_index, 2]])
    torch.manual_seed(456)
    updated = factorized_confidence_gated_update(
      x=x,
      probabilities=_probabilities(),
      mask_index=mask_index,
      move_chance_t=1 - torch.exp(torch.tensor(-1.0)),
      move_chance_s=1 - torch.exp(torch.tensor(-0.5)))
    self.assertEqual(int(updated[0, 0]), 3)
    self.assertEqual(int(updated[0, 3]), 2)

  def test_singleton_batch_accepts_broadcast_diffusion_probabilities(self):
    mask_index = 4
    x = torch.full((1, 4), mask_index, dtype=torch.long)
    torch.manual_seed(789)
    updated = factorized_confidence_gated_update(
      x=x,
      probabilities=_probabilities(),
      mask_index=mask_index,
      move_chance_t=torch.full((1, 1, 1), 0.8),
      move_chance_s=torch.full((1, 1, 1), 0.4))
    self.assertEqual(int(updated.ne(mask_index).sum()), 2)


if __name__ == '__main__':
  unittest.main()
