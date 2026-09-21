#!/usr/bin/env python3
"""Plot every completed development curve and the crossed-seed test contrasts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation.fresh_pair_statistics import canonical_sha256
from scripts.plot_staged_fresh import ARMS, sha


def read_inputs(run_dirs, test_dir, selection_path):
  selection = json.loads(selection_path.read_text())
  unsigned = {key: value for key, value in selection.items() if key != 'selection_sha256'}
  digest = canonical_sha256(unsigned)
  test = json.loads((test_dir / 'results.json').read_text())
  if (selection.get('selection_sha256') != digest
      or test.get('artifact') != 'fresh_pair_heldout_statistics'
      or test.get('selection_sha256') != digest
      or test.get('selected_checkpoints') != selection['selected']):
    raise ValueError('test results do not match the authenticated selection')
  seeds = selection['training_seeds']
  if not 2 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
    raise ValueError('this figure requires two or three distinct training seeds')
  if (test['overall']['training_seeds'] != len(seeds)
      or test['bootstrap']['training_seed_uncertainty_estimated'] is not True):
    raise ValueError('test intervals must include training-seed variation')
  if set(test['by_mask_rate']) != {str(rate) for rate in selection['mask_rates']}:
    raise ValueError('test mask-rate grid differs from the selection')
  sources = {'selection': sha(selection_path), 'test_results': sha(test_dir / 'results.json')}
  runs = {}
  for directory in run_dirs:
    run_path, protocol_path = directory / 'results.json', directory / 'protocol.json'
    run, protocol = json.loads(run_path.read_text()), json.loads(protocol_path.read_text())
    seed = protocol['training_config']['seed']
    if seed in runs or seed not in seeds:
      raise ValueError('development runs duplicate or add a training seed')
    final, interval = protocol['arguments']['steps'], protocol['arguments']['eval_every']
    if (run.get('completed') is not True or run['completed_steps'] != final
        or run['target_steps'] != final or run['protocol_sha256'] != sha(protocol_path)):
      raise ValueError('development run is incomplete or differs from its protocol')
    expected = sorted({0, final, *range(interval, final + 1, interval)})
    points = run['evaluations']
    if [point['step'] for point in points] != expected:
      raise ValueError('development curve omits, repeats, or reorders a planned checkpoint')
    chosen = {row['arm']: row for row in selection['selected'] if row['training_seed'] == seed}
    if set(chosen) != set(ARMS):
      raise ValueError('selection must contain all three arms per seed')
    for arm, choice in chosen.items():
      point = next((row for row in points if row['step'] == choice['checkpoint_step']), None)
      if point is None or point['checkpoint_sha256'] != choice['checkpoint_sha256']:
        raise ValueError('selected checkpoint is not in its development curve')
    for point in points:
      if set(point['arms']) != set(ARMS) or any(not math.isfinite(
          point['arms'][arm]['joint_gain_nats_per_masked_token']) for arm in ARMS):
        raise ValueError('development curve has missing arms or nonfinite gains')
    runs[seed] = run
    sources[f'training_results_seed{seed}'] = sha(run_path)
    sources[f'training_protocol_seed{seed}'] = sha(protocol_path)
  if set(runs) != set(seeds):
    raise ValueError('development curves omit a selected training seed')
  for group in [test['overall'], *test['by_mask_rate'].values()]:
    for key in ('directional_vs_unary', 'directional_dependence_gain'):
      metric = group['metrics_nats_per_masked_token'][key]
      values = [metric['estimate'], *metric['ci95']]
      if len(values) != 3 or not all(map(math.isfinite, values)) or values[1] > values[2]:
        raise ValueError('invalid paired test interval')
  return runs, test, selection, sources


def draw(run_dirs, test_dir, selection_path, output):
  runs, test, selection, sources = read_inputs(run_dirs, test_dir, selection_path)
  sources['plot_script'] = sha(Path(__file__))
  # The 6.6-inch source is printed at 5.5 inches in the manuscript. Keep the
  # smallest native type at 9.6 points so the final figure remains >=8 points.
  plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.6,
                       'axes.labelsize': 9.6, 'axes.titlesize': 10.5, 'pdf.fonttype': 42})
  fig = plt.figure(figsize=(6.6, 6.3))
  outer = fig.add_gridspec(2, 1, left=.13, right=.97, top=.84, bottom=.14, hspace=.82)
  development = outer[0].subgridspec(1, len(runs), wspace=.18)
  values = [point['arms'][arm]['joint_gain_nats_per_masked_token']
            for run in runs.values() for point in run['evaluations'] for arm in ARMS]
  span = max(max(values) - min(values), 1e-3)
  limits = (min(min(values), 0) - .1 * span, max(max(values), 0) + .1 * span)
  handles = []
  for index, (seed, run) in enumerate(sorted(runs.items())):
    panel = fig.add_subplot(development[0, index])
    selected = {row['arm']: row['checkpoint_step'] for row in selection['selected']
                if row['training_seed'] == seed}
    for arm, (label, color, style) in ARMS.items():
      points = run['evaluations']
      line, = panel.plot([point['step'] for point in points],
                         [point['arms'][arm]['joint_gain_nats_per_masked_token'] for point in points],
                         color=color, linestyle=style, linewidth=1.5, label=label)
      chosen = next(point for point in points if point['step'] == selected[arm])
      panel.plot(selected[arm], chosen['arms'][arm]['joint_gain_nats_per_masked_token'],
                  marker='o', markersize=4, color=color, markeredgecolor='white', zorder=4)
      if index == 0:
        handles.append(line)
    panel.axhline(0, color='#555555', linewidth=.7)
    panel.set(title=f'Seed {seed}', xlabel='Updates', ylim=limits,
               xticks=[0, run['target_steps'] // 2, run['target_steps']])
    panel.tick_params(labelsize=9.6, labelleft=index == 0)
    if index == 0:
      panel.set_ylabel('Gain over backbone\n(nats per masked token)')
  fig.suptitle('Development checkpoint selection', y=.96, fontsize=11)
  fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .925),
               ncol=3, frameon=False, fontsize=9.6)
  fig.text(.55, .49, 'Dots mark selected checkpoints. Higher is better.',
             ha='center', color='#555555', fontsize=9.6)
  panels = outer[1].subgridspec(1, 2, wspace=.48)
  rates = sorted(test['by_mask_rate'], key=float)
  groups = [test['overall']] + [test['by_mask_rate'][rate] for rate in rates]
  labels = ['Pooled'] + [f'{float(rate):.0%}' for rate in rates]
  for index, (key, title) in enumerate((
      ('directional_vs_unary', 'Separate factors vs. unary'),
      ('directional_dependence_gain', 'Joint vs. its own marginals'))):
    panel = fig.add_subplot(panels[0, index])
    for y, group in enumerate(groups):
      metric = group['metrics_nats_per_masked_token'][key]
      color = '#0072B2' if y == 0 else '#697581'
      panel.hlines(y, *metric['ci95'], color=color, linewidth=1.6)
      panel.plot(metric['estimate'], y, 'o', color=color, markersize=4)
    panel.axvline(0, color='#555555', linewidth=.8)
    panel.set_yticks(range(len(labels)), labels)
    panel.invert_yaxis()
    panel.set_title(title, fontsize=10)
    panel.set_xlabel('Test log-score gain\n(nats per masked token)', fontsize=9.6)
    panel.grid(axis='x', alpha=.16)
    panel.tick_params(labelsize=9.6)
    panel.ticklabel_format(axis='x', style='plain', useOffset=False)
  for panel in fig.axes:
    panel.spines[['top', 'right']].set_visible(False)
  fig.text(.55, .016, f'95% crossed seed/document-bootstrap intervals; {len(runs)} training seeds.',
             ha='center', fontsize=9.6, color='#555555')
  output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output)
  fig.savefig(output.with_suffix('.png'), dpi=180)
  plt.close(fig)
  sidecar = {'artifact': 'fresh_multiseed_training_and_selected_test_figure',
             'source_sha256': sources, 'selection_sha256': selection['selection_sha256'],
             'training_seeds': sorted(runs), 'selected_checkpoints': selection['selected'],
             'test_overall': test['overall'], 'test_by_mask_rate': test['by_mask_rate'],
             'bootstrap': test['bootstrap'], 'layout_inches': [6.6, 6.3]}
  output.with_suffix('.json').write_text(json.dumps(sidecar, indent=2, allow_nan=False) + '\n')
  return sidecar


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, action='append', required=True)
  for name in ('test-dir', 'selection', 'output'):
    parser.add_argument(f'--{name}', type=Path, required=True)
  args = parser.parse_args(argv)
  draw(args.run_dir, args.test_dir, args.selection, args.output)


if __name__ == '__main__':
  main()
