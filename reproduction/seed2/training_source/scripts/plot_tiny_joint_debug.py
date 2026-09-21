#!/usr/bin/env python3
"""Plot the two-token diagnostic from measured probabilities, without fitting."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', type=Path, default=Path(
    'artifacts/paper/staged-debugging-v1/tiny/results.json'))
  parser.add_argument('--output', type=Path, default=Path(
    'artifacts/paper/staged-debugging-v1/tiny/figure.pdf'))
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--directional-input', type=Path)
  parser.add_argument('--shared-variant', default='shared_contextual_wide')
  parser.add_argument('--directional-variant', default='directional_contextual_wide')
  args = parser.parse_args(argv)
  data = json.loads(args.input.read_text())

  def row(name, report=data):
    matches = [r for r in report['runs'] if r['task'] == 'opposite'
               and r['variant'] == name and r['seed'] == args.seed]
    if len(matches) != 1:
      raise ValueError(f'expected exactly one {name}/seed-{args.seed} row')
    return matches[0]

  def matrix(values):
    return np.array([values[x] for x in ('AA', 'AB', 'BA', 'BB')]).reshape(2, 2)

  shared = row(args.shared_variant)
  directional_data = (json.loads(args.directional_input.read_text())
                      if args.directional_input else data)
  directional = row(args.directional_variant, directional_data)
  probability = matrix(directional['probabilities'])
  independent = matrix(directional['independent_marginal_probabilities'])
  panels = [np.array([[0., 0.5], [0.5, 0.]]),
            matrix(shared['probabilities']), probability, independent]
  labels = ['Target', 'Shared factors', 'Separate factors',
            'Independent draws\nfrom the same marginals']
  plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                       'pdf.fonttype': 42, 'ps.fonttype': 42})
  fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.45))
  for i, (ax, panel, label) in enumerate(zip(axes, panels, labels)):
    ax.imshow(panel, vmin=0, vmax=0.5, cmap='Blues')
    ax.set_title(label, fontsize=10, pad=9)
    ax.set_xticks([0, 1], ['A', 'B'])
    ax.set_yticks([0, 1], ['A', 'B'])
    ax.set_xlabel('Second token', labelpad=2)
    if i == 0:
      ax.set_ylabel('First token', labelpad=2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
      spine.set_visible(False)
    for first in range(2):
      for second in range(2):
        value = panel[first, second]
        label = (f'{100 * value:.2f}%' if 0 < value < 0.001
                 else f'{100 * value:.1f}%')
        ax.text(second, first, label, ha='center', va='center',
                 color='white' if value > 0.28 else '#17324d', fontsize=10)
    invalid = np.trace(panel)
    ax.text(0.5, -0.43, f'{100 * invalid:.2f}% invalid',
             transform=ax.transAxes, ha='center', fontsize=9)
  fig.subplots_adjust(left=0.065, right=0.99, bottom=0.34, top=0.78, wspace=0.31)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output, bbox_inches='tight', pad_inches=0.04)
  fig.savefig(args.output.with_suffix('.png'), dpi=180,
              bbox_inches='tight', pad_inches=0.04)
  plt.close(fig)
  args.output.with_suffix('.json').write_text(json.dumps({
    'source': str(args.input), 'seed': args.seed, 'task': 'opposite',
    'directional_source': str(args.directional_input or args.input),
    'shared_variant': shared['variant'],
    'directional_variant': directional['variant'],
    'shared_config': shared['variant_config'],
    'directional_config': directional['variant_config'],
    'shared_gradient_active_parameter_count': shared['gradient_active_parameter_count'],
    'directional_gradient_active_parameter_count': directional[
      'gradient_active_parameter_count'],
    'panel_order': labels,
    'panels': [panel.tolist() for panel in panels],
    'caption_details': 'AA and BB are invalid. Both learned models use Gaussian '
      'factor initialization std=0.25, uniform frozen unaries, full vocabulary, '
      'one fixed edge, and exact likelihood training. The final panel is the '
      'exact distribution of independent draws from the directional model '
      'marginals. The seed was fixed to 1 before plotting; no examples selected.',
  }, indent=2, allow_nan=False) + '\n')
  print(args.output.resolve())


if __name__ == '__main__':
  main()
