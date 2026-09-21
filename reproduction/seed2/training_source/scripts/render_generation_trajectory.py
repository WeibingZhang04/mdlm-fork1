#!/usr/bin/env python3
"""Render an outcome-independent exact generation trajectory for the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402


EXPECTED_MODES = ('structured_marginal', 'structured_joint')
DISPLAY_CALLS = (0, 16, 32, 48, 63)
ROW_SELECTION_POLICY = 'first_row_of_hash_min_full_source_batch_v1'
REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--trajectory', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--tokenizer', default='gpt2-large')
  parser.add_argument('--tokenizer-revision', required=True)
  parser.add_argument('--batch-row-index', type=int, default=0)
  parser.add_argument('--local-files-only', action='store_true')
  return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
  return hashlib.sha256(json.dumps(
    payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _clean_repository_identity() -> dict[str, Any]:
  git_sha = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
  status = subprocess.check_output(
    ['git', 'status', '--porcelain'], cwd=REPO_ROOT, text=True).splitlines()
  if status:
    raise RuntimeError('refusing to render from a dirty repository')
  return {'git_sha': git_sha, 'dirty': False}


def _validate_artifact(payload: Mapping[str, Any], row_index: int) -> None:
  if payload.get('artifact') != 'exact_generation_trajectory_replay':
    raise ValueError('input is not an exact trajectory replay artifact')
  selection = payload['selection']
  if selection.get('outcome_independent') is not True:
    raise ValueError('trajectory source selection is not outcome-independent')
  if selection.get('policy') != 'hash_min_full_source_batch_v1':
    raise ValueError('trajectory source selection policy is unexpected')
  if tuple(payload.get('modes', ())) != EXPECTED_MODES:
    raise ValueError('trajectory modes are unexpected')
  if not 0 <= row_index < len(selection['sample_indices']):
    raise ValueError('batch row index lies outside the selected batch')
  for mode in EXPECTED_MODES:
    trajectory = payload['trajectories'][mode]
    if trajectory.get('final_token_ids_match_expected') is not True:
      raise ValueError(f'{mode} final raw token IDs do not match source')
    if trajectory.get('measured_nfe') != 63:
      raise ValueError(f'{mode} measured NFE is not the expected 63')
    if trajectory.get('requested_nfe_budget') != 64:
      raise ValueError(f'{mode} requested NFE budget is not 64')


def _selected_snapshots(
    trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
  selected = []
  for call_index in DISPLAY_CALLS:
    candidates = [
      snapshot for snapshot in trajectory['snapshots']
      if snapshot['model_call_index'] == call_index]
    if not candidates:
      raise ValueError(f'missing required model-call snapshot {call_index}')
    if call_index == 63:
      candidates.sort(key=lambda item: item['stage'] != 'final')
    selected.append(candidates[0])
  return selected


def _active_values(values: Sequence[Any], active_mask: Sequence[bool]) -> list[Any]:
  return [value for value, active in zip(values, active_mask) if active]


def _decode_segments(
    tokenizer: Any,
    token_ids: Sequence[int],
    unresolved: Sequence[bool],
) -> str:
  """Decode consecutive resolved runs and mark unrevealed runs explicitly."""
  pieces = []
  index = 0
  while index < len(token_ids):
    stop = index + 1
    while stop < len(token_ids) and unresolved[stop] == unresolved[index]:
      stop += 1
    if unresolved[index]:
      count = stop - index
      pieces.append(f' ⟦{count} masked⟧ ')
    else:
      decoded = tokenizer.decode(
        list(token_ids[index:stop]),
        clean_up_tokenization_spaces=False,
        skip_special_tokens=False)
      pieces.append(decoded.replace('\n', '↵').replace('\t', '⇥'))
    index = stop
  return ''.join(pieces).strip()


def _wrap_to_width(text: str, renderer, font: FontProperties,
                   width_pixels: float) -> str:
  """Wrap using rendered glyph widths, including unusually long words.

  Inserting line breaks changes layout only. The source token IDs and decoded
  runs remain in the original artifact; no words are shortened or omitted.
  """
  if width_pixels <= 0:
    raise ValueError('available text width must be positive')

  def width(value):
    return renderer.get_text_width_height_descent(value, font, ismath=False)[0]

  lines, current = [], ''
  for word in text.split():
    combined = f'{current} {word}' if current else word
    if width(combined) <= width_pixels:
      current = combined
      continue
    if current:
      lines.append(current)
      current = ''
    # URLs and other unspaced strings need a line break within the string.
    # Find the largest fitting prefix using the actual font, not character
    # count; wide glyphs and masked-token markers need different widths.
    while word and width(word) > width_pixels:
      low, high = 1, len(word)
      while low < high:
        middle = (low + high + 1) // 2
        if width(word[:middle]) <= width_pixels:
          low = middle
        else:
          high = middle - 1
      if width(word[:low]) > width_pixels:
        raise ValueError('one glyph exceeds the available panel width')
      lines.append(word[:low])
      word = word[low:]
    current = word
  if current:
    lines.append(current)
  return '\n'.join(lines)


def _assert_text_boxes_fit(fig, panel_artists, outer_artists,
                           *, padding_inches: float, tolerance_pixels: float = 0.5):
  """Fail before saving if a text box leaves its allotted panel or overlaps."""
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  padding = padding_inches * fig.dpi
  boxes = []
  for ax, label, body in panel_artists:
    panel = ax.get_window_extent(renderer)
    label_box = label.get_window_extent(renderer)
    body_box = body.get_window_extent(renderer)
    for name, box in (('label', label_box), ('body', body_box)):
      if (box.x0 < panel.x0 + padding - tolerance_pixels
          or box.x1 > panel.x1 - padding + tolerance_pixels
          or box.y0 < panel.y0 + padding - tolerance_pixels
          or box.y1 > panel.y1 - padding + tolerance_pixels):
        raise AssertionError(f'{name} text extends beyond its panel')
    if label_box.overlaps(body_box):
      raise AssertionError('panel label overlaps its body text')
    boxes.append({'label_width_pixels': label_box.width,
                  'body_width_pixels': body_box.width,
                  'body_height_pixels': body_box.height,
                  'panel_width_pixels': panel.width,
                  'panel_height_pixels': panel.height})
  outer_boxes = [artist.get_window_extent(renderer) for artist in outer_artists]
  for index, box in enumerate(outer_boxes):
    if (box.x0 < -tolerance_pixels or box.y0 < -tolerance_pixels
        or box.x1 > fig.bbox.x1 + tolerance_pixels
        or box.y1 > fig.bbox.y1 + tolerance_pixels):
      raise AssertionError('title or footer extends beyond the figure')
    if any(box.overlaps(other) for other in outer_boxes[:index]):
      raise AssertionError('figure titles or footer overlap')
    if any(box.overlaps(ax.get_window_extent(renderer)) for ax, _, _ in panel_artists):
      raise AssertionError('figure title or footer overlaps a panel')
  return boxes


def render(
    payload: Mapping[str, Any],
    tokenizer: Any,
    output: Path,
    row_index: int,
    trajectory_file_sha256: str,
) -> dict[str, Any]:
  _validate_artifact(payload, row_index)
  repository_identity = _clean_repository_identity()
  output.parent.mkdir(parents=True, exist_ok=True)
  colors = {
    'structured_marginal': '#35618f',
    'structured_joint': '#ad542f',
  }
  titles = {
    'structured_marginal': 'Marginal sampling',
    'structured_joint': 'Joint sampling',
  }
  # Physical dimensions keep text size and margins stable when a row needs
  # more space. The final figure height follows the measured body text.
  figure_width = 7.2
  left_margin, right_margin, column_gap = 0.22, 0.12, 0.18
  panel_width = (figure_width - left_margin - right_margin - column_gap) / 2
  padding, label_body_gap, row_gap = 0.10, 0.085, 0.12
  top_band, bottom_band = 0.78, 0.55
  # At 0.96 times the ICLR text width, the body remains about 8 pt.
  # Row heights grow with the text rather than shrinking it to fit.
  body_font_size, label_font_size = 11.0, 10.5
  body_font = FontProperties(family='DejaVu Sans Mono', size=body_font_size)
  line_spacing = 1.25
  fig = plt.figure(figsize=(figure_width, 1.0), dpi=100)
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  available_width = (panel_width - 2 * padding) * fig.dpi
  panel_records = []
  panels = {}
  row_heights = [0.0] * len(DISPLAY_CALLS)

  def measure_height(text, **style):
    probe = fig.text(0, 0, text, va='top', parse_math=False, **style)
    height = probe.get_window_extent(renderer).height / fig.dpi
    probe.remove()
    return height

  for column, mode in enumerate(EXPECTED_MODES):
    trajectory = payload['trajectories'][mode]
    source_record = payload['source_records'][mode][row_index]
    active_mask = source_record['active_mask']
    active_count = sum(active_mask)
    for row, snapshot in enumerate(_selected_snapshots(trajectory)):
      active_ids = _active_values(snapshot['token_ids'][row_index], active_mask)
      unresolved = _active_values(
        snapshot['unresolved_active_mask'][row_index], active_mask)
      unresolved_count = sum(unresolved)
      resolved_percent = 100.0 * (active_count - unresolved_count) / active_count
      display = _decode_segments(tokenizer, active_ids, unresolved)
      label = (
        f"call {snapshot['model_call_index']}  |  "
        f'{resolved_percent:.0f}% revealed')
      if snapshot['stage'] == 'final':
        label += '  |  final'
      wrapped = _wrap_to_width(display, renderer, body_font, available_width)
      label_height = measure_height(label, fontsize=label_font_size, fontweight='bold')
      body_height = measure_height(wrapped, fontproperties=body_font, linespacing=line_spacing)
      panels[row, column] = (label, wrapped, label_height, colors[mode])
      row_heights[row] = max(row_heights[row],
                             2 * padding + label_height + label_body_gap + body_height)
      panel_records.append({
        'mode': mode,
        'model_call_index': snapshot['model_call_index'],
        'stage': snapshot['stage'],
        'active_token_count': active_count,
        'unresolved_active_count': unresolved_count,
        'resolved_percent': resolved_percent,
      })

  batch_row = payload['trajectories'][EXPECTED_MODES[0]]['batch_order'][row_index]
  figure_height = top_band + bottom_band + sum(row_heights) + row_gap * (len(row_heights) - 1)
  fig.set_size_inches(figure_width, figure_height)
  panel_artists = []
  panel_top = figure_height - top_band
  for row, height in enumerate(row_heights):
    for column in range(len(EXPECTED_MODES)):
      x = left_margin + column * (panel_width + column_gap)
      ax = fig.add_axes((x / figure_width, (panel_top - height) / figure_height,
                         panel_width / figure_width, height / figure_height))
      ax.set_facecolor('#f7f8fa')
      for spine in ax.spines.values():
        spine.set_edgecolor('#d8dce2')
      ax.set_xticks([])
      ax.set_yticks([])
      label, wrapped, label_height, color = panels[row, column]
      label_artist = ax.text(
        padding / panel_width, 1 - padding / height, label,
        transform=ax.transAxes, fontsize=label_font_size, fontweight='bold',
        color=color, va='top', parse_math=False)
      body_artist = ax.text(
        padding / panel_width, 1 - (padding + label_height + label_body_gap) / height,
        wrapped, transform=ax.transAxes, fontproperties=body_font,
        color='#20242a', va='top', linespacing=line_spacing, parse_math=False)
      panel_artists.append((ax, label_artist, body_artist))
    panel_top -= height + row_gap
  outer_artists = [fig.text(
    0.5, 1 - 0.06 / figure_height, 'Filling the masked span',
    ha='center', va='top', fontsize=15.0, fontweight='bold')]
  for column, mode in enumerate(EXPECTED_MODES):
    center = left_margin + column * (panel_width + column_gap) + panel_width / 2
    outer_artists.append(fig.text(
      center / figure_width, 1 - 0.40 / figure_height, titles[mode],
      ha='center', va='top', fontsize=12.0, fontweight='bold', color=colors[mode]))
  outer_artists.append(fig.text(
    0.5, 0.065 / figure_height,
    f"Sample {batch_row['sample_index']} · 64-call budget, 63 calls used.\n"
    '⟦k masked⟧ marks k unresolved tokens.',
    ha='center', va='bottom', fontsize=10.0, color='#4b5058', linespacing=1.2))
  layout_boxes = _assert_text_boxes_fit(
    fig, panel_artists, outer_artists, padding_inches=padding)
  fig.savefig(output)
  plt.close(fig)

  provenance = {
    'schema_version': 1,
    'artifact': 'generation_trajectory_figure_provenance',
    'figure_filename': output.name,
    'renderer_repository': repository_identity,
    'trajectory_input_file_sha256': trajectory_file_sha256,
    'trajectory_artifact_sha256': payload['artifact_sha256'],
    'source_selection_policy': payload['selection']['policy'],
    'source_selection_sha256': payload['selection']['selection_sha256'],
    'source_selection_outcome_independent': True,
    'row_selection_policy': ROW_SELECTION_POLICY,
    'batch_row_index': row_index,
    'sample_index': batch_row['sample_index'],
    'pair_key': batch_row['pair_key'],
    'requested_nfe_budget': payload['nfe_budget'],
    'measured_nfe': {
      mode: payload['trajectories'][mode]['measured_nfe']
      for mode in EXPECTED_MODES
    },
    'final_raw_token_ids_match_source': {
      mode: payload['trajectories'][mode]['final_token_ids_match_expected']
      for mode in EXPECTED_MODES
    },
    'panels': panel_records,
    'layout': {
      'body_font_size_points': body_font_size,
      'body_font_family': body_font.get_family(),
      'figure_size_inches': [figure_width, figure_height],
      'panel_width_inches': panel_width,
      'row_heights_inches': row_heights,
      'all_text_bounding_boxes_checked': True,
      'panel_bounding_boxes_pixels': layout_boxes,
    },
  }
  provenance['provenance_sha256'] = _canonical_sha256(provenance)
  sidecar = output.with_suffix(output.suffix + '.provenance.json')
  sidecar.write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  return provenance


def main(argv=None) -> int:
  args = _parse_args(argv)
  from transformers import AutoTokenizer
  payload = json.loads(args.trajectory.read_text(encoding='utf-8'))
  tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer,
    revision=args.tokenizer_revision,
    local_files_only=args.local_files_only)
  provenance = render(
    payload, tokenizer, args.output, args.batch_row_index,
    _sha256_file(args.trajectory))
  print(json.dumps({
    'event': 'generation_trajectory_figure_rendered',
    'output': str(args.output),
    'output_sha256': _sha256_file(args.output),
    'provenance_sha256': provenance['provenance_sha256'],
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
