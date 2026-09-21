#!/usr/bin/env python3
"""Plot completed fresh-training development curves and selected test contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = {
  'shared': ('Shared factors', '#697581', '--'),
  'directional': ('Separate factors', '#0072B2', '-'),
  'unary': ('Unary adapter', '#D55E00', '-.'),
}


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(run_dir, test_dir, selection_path):
  paths = {'training_results': run_dir / 'results.json',
           'test_results': test_dir / 'results.json', 'selection': selection_path}
  data = {key: json.loads(path.read_text()) for key, path in paths.items()}
  run, test, selection = (data[key] for key in paths)
  if not run.get('completed') or run['completed_steps'] != run['target_steps']:
    raise ValueError('training run is not complete')
  if test.get('artifact') != 'fresh_pair_heldout_statistics':
    raise ValueError('expected selected-checkpoint test statistics')
  unsigned = {key: value for key, value in selection.items() if key != 'selection_sha256'}
  canonical = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
  if canonical != selection.get('selection_sha256') or test['selection_sha256'] != canonical:
    raise ValueError('test results do not match the authenticated selection')
  if test['selected_checkpoints'] != selection['selected']:
    raise ValueError('test selected checkpoints differ')
  if len(selection['training_seeds']) != 1:
    raise ValueError('this development-curve plot requires a single training run')
  rows = run['evaluations']
  steps = [row['step'] for row in rows]
  if not steps or steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != run['target_steps']:
    raise ValueError('development curve is incomplete or unordered')
  chosen = {row['arm']: row for row in selection['selected']}
  if set(chosen) != set(ARMS):
    raise ValueError('selection must contain all three arms')
  for arm, choice in chosen.items():
    point = next((row for row in rows if row['step'] == choice['checkpoint_step']), None)
    if point is None or point['checkpoint_sha256'] != choice['checkpoint_sha256']:
      raise ValueError('selected checkpoint is not in the development curve')
    for row in rows:
      if not math.isfinite(row['arms'][arm]['joint_gain_nats_per_masked_token']):
        raise ValueError('nonfinite development gain')
  for group in [test['overall'], *test['by_mask_rate'].values()]:
    for key in ('directional_vs_unary', 'directional_dependence_gain'):
      metric = group['metrics_nats_per_masked_token'][key]
      if (not all(math.isfinite(value) for value in [metric['estimate'], *metric['ci95']])
          or metric['ci95'][0] > metric['ci95'][1]):
        raise ValueError('invalid paired test interval')
  return run, test, selection, {key: sha(path) for key, path in paths.items()}


def draw(run_dir, test_dir, selection_path, output):
  run, test, selection, sources = read_inputs(run_dir, test_dir, selection_path)
  sources['plot_script'] = sha(Path(__file__))
  plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                       'axes.labelsize': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42})
  fig = plt.figure(figsize=(5.5, 6.2))
  grid = fig.add_gridspec(2, 2, left=.17, right=.965, top=.94, bottom=.13,
                          hspace=.75, wspace=.58, height_ratios=[1.1, 1])
  ax = fig.add_subplot(grid[0, :])
  selected = {row['arm']: row['checkpoint_step'] for row in selection['selected']}
  for arm, (label, color, style) in ARMS.items():
    points = run['evaluations']
    values = [row['arms'][arm]['joint_gain_nats_per_masked_token'] for row in points]
    ax.plot([row['step'] for row in points], values, color=color, linestyle=style,
            linewidth=1.6, label=label)
    chosen = next(row for row in points if row['step'] == selected[arm])
    ax.plot(selected[arm], chosen['arms'][arm]['joint_gain_nats_per_masked_token'],
            marker='o', markersize=5, color=color, markeredgecolor='white', zorder=4)
  ax.axhline(0, color='#333333', linewidth=.8)
  ax.set(title='Checkpoint selection on development documents',
         xlabel='Training updates', ylabel='Gain over backbone\n(nats per masked token)')
  ax.legend(loc='center right', bbox_to_anchor=(1, .4), frameon=True, facecolor='white',
             edgecolor='none', framealpha=1, fontsize=8)
  ax.text(1, -.32, 'Dots mark selected checkpoints. Higher is better.',
           transform=ax.transAxes, ha='right', fontsize=8, color='#555555')
  rates = sorted(test['by_mask_rate'], key=float)
  groups = [test['overall']] + [test['by_mask_rate'][rate] for rate in rates]
  labels = ['Pooled'] + [f'{float(rate):.0%}' for rate in rates]
  for index, (key, title) in enumerate((
      ('directional_vs_unary', 'Separate factors vs. unary'),
      ('directional_dependence_gain', 'Joint vs. its own marginals'))):
    panel = fig.add_subplot(grid[1, index])
    for y, group in enumerate(groups):
      metric = group['metrics_nats_per_masked_token'][key]
      lo, hi = metric['ci95']
      color = '#0072B2' if y == 0 else '#697581'
      panel.hlines(y, lo, hi, color=color, linewidth=1.6)
      panel.plot(metric['estimate'], y, 'o', color=color, markersize=4)
    panel.axvline(0, color='#555555', linewidth=.8)
    panel.set_yticks(range(len(labels)), labels)
    panel.invert_yaxis()
    panel.set_title(title, fontsize=9)
    panel.set_xlabel('Test log-score gain\n(nats per masked token)', fontsize=8)
    panel.grid(axis='x', alpha=.16)
    panel.tick_params(axis='both', labelsize=8)
    panel.ticklabel_format(axis='x', style='plain', useOffset=False)
  for panel in fig.axes:
    panel.spines[['top', 'right']].set_visible(False)
  fig.text(.5, .015, 'Test intervals: 95% paired document bootstrap; one training seed.',
            ha='center', fontsize=8, color='#555555')
  output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output)
  fig.savefig(output.with_suffix('.png'), dpi=180)
  plt.close(fig)
  sidecar = {
    'artifact': 'fresh_training_and_selected_test_figure', 'source_sha256': sources,
    'selection_sha256': selection['selection_sha256'],
    'training_seed': selection['training_seeds'][0],
    'selected_checkpoints': selection['selected'],
    'scope': 'source-order conditional-likelihood pilot; no generation claim',
    'test_overall': test['overall'], 'test_by_mask_rate': test['by_mask_rate'],
    'bootstrap': test['bootstrap'], 'layout_inches': [5.5, 6.2],
  }
  output.with_suffix('.json').write_text(json.dumps(sidecar, indent=2, allow_nan=False) + '\n')
  return sidecar


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('run-dir', 'test-dir', 'selection', 'output'):
    parser.add_argument(f'--{name}', type=Path, required=True)
  args = parser.parse_args(argv)
  draw(args.run_dir, args.test_dir, args.selection, args.output)


if __name__ == '__main__':
  main()
