"""Protocol checks that do not require downloading GPT-2 Large."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'score_generated_text_gpt2.py'
SPEC = importlib.util.spec_from_file_location('score_generated_text_gpt2', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScoreMaskTests(unittest.TestCase):

  def test_leading_bos_and_later_eos(self):
    # Original MDLM ignores the second EOS but counts later ordinary tokens.
    ids = [0, 7, 8, 0, 9]
    attention = [1] * len(ids)
    self.assertEqual(
      MODULE.mdlm_original_mask(ids, eos_id=0),
      [True, True, False, True])
    self.assertEqual(
      MODULE.stop_at_eos_mask(ids, attention, eos_id=0, bos_id=0),
      [True, True, True, False])

  def test_padding_is_excluded_only_by_stop_at_eos(self):
    ids = [7, 8, 0, 0]
    attention = [1, 1, 0, 0]
    self.assertEqual(
      MODULE.mdlm_original_mask(ids, eos_id=0),
      [True, True, False])
    self.assertEqual(
      MODULE.stop_at_eos_mask(ids, attention, eos_id=0, bos_id=0),
      [True, False, False])

  def test_no_eos_scores_all_real_next_tokens(self):
    ids = [7, 8, 9]
    self.assertEqual(MODULE.mdlm_original_mask(ids, eos_id=0), [True, True])
    self.assertEqual(
      MODULE.stop_at_eos_mask(ids, [1, 1, 1], eos_id=0, bos_id=0),
      [True, True])


if __name__ == '__main__':
  unittest.main()
