import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from scripts.render_generation_trajectory import (
  DISPLAY_CALLS, EXPECTED_MODES, _assert_text_boxes_fit,
  _decode_segments, _wrap_to_width, render,
)


class _Tokenizer:

  def decode(self, token_ids, **unused_kwargs):
    return ''.join(chr(96 + value) for value in token_ids)


class RenderGenerationTrajectoryTest(unittest.TestCase):

  def test_consecutive_masks_are_counted_and_resolved_runs_are_decoded(self):
    rendered = _decode_segments(
      _Tokenizer(), [1, 2, 3, 4, 5], [False, True, True, False, False])
    self.assertEqual(rendered, 'a ⟦2 masked⟧ de')

  def test_all_resolved_tokens_decode_as_one_run(self):
    rendered = _decode_segments(
      _Tokenizer(), [1, 2, 3], [False, False, False])
    self.assertEqual(rendered, 'abc')

  def test_wrapping_respects_measured_width_for_wide_glyphs_and_long_words(self):
    fig = plt.figure(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family='DejaVu Sans Mono', size=7.5)
    text = 'WWW $money$ ' + ('LongIdentifier' * 15) + ' ⟦12 masked⟧ end'
    wrapped = _wrap_to_width(text, renderer, font, 180)
    for line in wrapped.splitlines():
      width = renderer.get_text_width_height_descent(line, font, ismath=False)[0]
      self.assertLessEqual(width, 180)
    self.assertEqual(''.join(wrapped.split()), ''.join(text.split()))
    plt.close(fig)

  def test_bounding_box_assertion_detects_overflow(self):
    fig = plt.figure(figsize=(3, 2), dpi=100)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    label = ax.text(0.05, 0.9, 'call 0', va='top', transform=ax.transAxes)
    body = ax.text(0.05, 0.6, 'This long line extends beyond the panel boundary' * 3,
                   va='top', transform=ax.transAxes)
    with self.assertRaisesRegex(AssertionError, 'beyond its panel'):
      _assert_text_boxes_fit(fig, [(ax, label, body)], [], padding_inches=0.02)
    plt.close(fig)

  def test_renderer_sizes_rows_and_checks_all_text_boxes(self):
    class LongTokenizer:
      def decode(self, ids, **kwargs):
        return 'LongIdentifierWithoutSpaces' * 4 + ' word' * len(ids)

    payload = {
      'artifact': 'exact_generation_trajectory_replay', 'artifact_sha256': 'artifact',
      'selection': {'outcome_independent': True, 'policy': 'hash_min_full_source_batch_v1',
                    'sample_indices': [641], 'selection_sha256': 'selection'},
      'modes': list(EXPECTED_MODES), 'nfe_budget': 64,
      'trajectories': {}, 'source_records': {},
    }
    for mode in EXPECTED_MODES:
      payload['source_records'][mode] = [{'active_mask': [True] * 32}]
      payload['trajectories'][mode] = {
        'final_token_ids_match_expected': True, 'measured_nfe': 63,
        'requested_nfe_budget': 64,
        'batch_order': [{'sample_index': 641, 'pair_key': 'same-source-pair'}],
        'snapshots': [
          {'model_call_index': call, 'stage': 'final' if call == 63 else 'step',
           'token_ids': [list(range(32))],
           'unresolved_active_mask': [[position >= min(32, call // 2 + 1) for position in range(32)]]}
          for call in DISPLAY_CALLS
        ],
      }
    before = json.dumps(payload, sort_keys=True)
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'trajectory.pdf'
      with patch('scripts.render_generation_trajectory._clean_repository_identity',
                  return_value={'git_sha': 'test', 'dirty': False}):
        provenance = render(payload, LongTokenizer(), output, 0, 'source-file-sha256')
      self.assertTrue(output.is_file())
      self.assertEqual(provenance['sample_index'], 641)
      self.assertEqual(provenance['batch_row_index'], 0)
      self.assertEqual(len(provenance['panels']), 10)
      self.assertTrue(provenance['layout']['all_text_bounding_boxes_checked'])
      # The paper includes this 7.2-inch source at 0.96 * 5.5 inches.
      self.assertGreaterEqual(provenance['layout']['body_font_size_points'] * 0.96 * 5.5 / 7.2, 8)
      self.assertEqual(len(provenance['layout']['panel_bounding_boxes_pixels']), 10)
      self.assertGreater(max(provenance['layout']['row_heights_inches']), 0.5)
      self.assertEqual(before, json.dumps(payload, sort_keys=True))


if __name__ == '__main__':
  unittest.main()
