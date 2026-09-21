#!/usr/bin/env python3
"""Plot measured learning curves from the fixed-corruption overfit diagnostic.

The figure describes a small training/development diagnostic. It does not
estimate generation quality, benchmark performance, or uncertainty across
training seeds. Source hashes and exact plotted values accompany the figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ARMS = {
  'shared': ('Shared factors', '#697581', (0, (2, 1.6))),
  'directional': ('Separate factors', '#0072B2', '-'),
  'unary': ('Unary adapter', '#D55E00', (0, (5, 1.6, 1, 1.6))),
}
NLL = 'joint_nll_per_masked_token'
DEPENDENCE = 'dependence_gain_nats_per_masked_token'


def file_sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value, description):
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f'{description} must be finite')
  return result


def load_run(run_dir: Path):
  """Read current curve files, falling back to completed arm records."""
  run_dir = run_dir.resolve()
  sources = {}

  def read(path):
    sources[str(path)] = file_sha256(path)
    return json.loads(path.read_text())

  protocol = read(run_dir / 'protocol.json')
  result_path = run_dir / 'results.json'
  results = read(result_path) if result_path.exists() else {}
  curves = {}
  for name in ARMS:
    path = run_dir / f'{name}-curve.jsonl'
    recorded = results.get('arms', {}).get(name, {}).get('curve', [])
    if path.exists():
      sources[str(path)] = file_sha256(path)
      rows = [json.loads(line) for line in path.read_text().splitlines()
              if line.strip()]
    else:
      rows = recorded
    if not rows:
      continue
    prior = -1
    compact = []
    for row in rows:
      step = row['step']
      if type(step) is not int or step < 0 or step <= prior:
        raise ValueError(f'{name}: updates must be distinct increasing integers')
      prior = step
      value = {'step': step}
      for split in ('train', 'dev'):
        value[split] = {
          key: _finite(row[split][key], f'{name}/{split}/{key}')
          for key in (NLL, DEPENDENCE, 'backbone_nll_per_masked_token')}
      compact.append(value)
    # Detect stale or mixed copies when both storage forms are available.
    by_step = {row['step']: row for row in compact}
    for row in recorded:
      if row['step'] not in by_step:
        continue
      for split in ('train', 'dev'):
        for key in (NLL, DEPENDENCE):
          if not math.isclose(row[split][key], by_step[row['step']][split][key],
                              rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f'{name}/{split}: curve and results disagree')
    curves[name] = compact
  if not curves:
    raise ValueError('no nonempty shared, directional, or unary curves found')
  floors = {}
  for split in ('train', 'dev'):
    path = run_dir / f'support-floor-{split}.json'
    if path.exists():
      report = read(path)
      top_k = protocol.get('arguments', {}).get('top_k')
      if top_k is not None and report.get('top_k') != top_k:
        raise ValueError(f'{split}: support bound uses a different candidate count')
      floors[split] = {
        'value': _finite(report['nll_floor_per_masked_token'],
                         f'{split} support bound'),
        'applies_to': report.get('applies_to'),
      }
  return protocol, results, curves, floors, sources


def backbone_value(protocol, curves, split):
  cache = protocol.get('cache', {}).get(split, {})
  value = cache.get('backbone_nll_per_masked_token')
  values = [row[split]['backbone_nll_per_masked_token']
            for rows in curves.values() for row in rows]
  baseline = _finite(value if value is not None else values[0],
                      f'{split} backbone NLL')
  if any(not math.isclose(number, baseline, rel_tol=1e-5, abs_tol=1e-5)
         for number in values):
    raise ValueError(f'{split}: curves do not share a fixed backbone baseline')
  return baseline


def draw(run_dir: Path, output: Path):
  protocol, results, curves, floors, sources = load_run(run_dir)
  sources[str(Path(__file__).resolve())] = file_sha256(Path(__file__))
  plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                       'axes.titlesize': 10, 'axes.labelsize': 9,
                       'pdf.fonttype': 42, 'ps.fonttype': 42})
  # Export at the paper's actual 5.5-inch text width. A wide three-panel
  # source shrinks nominal 8pt labels below 5pt when included in the paper.
  fig = plt.figure(figsize=(5.5, 6.0))
  grid = fig.add_gridspec(2, 2, left=0.16, right=0.98, bottom=0.185,
                          top=0.95, wspace=0.42, hspace=0.60)
  axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
          fig.add_subplot(grid[1, :])]
  maximum_step = max(rows[-1]['step'] for rows in curves.values())
  baselines, floor_display = {}, {}
  for ax, split, title in zip(axes[:2], ('train', 'dev'),
                              ('Training examples', 'Held-out examples')):
    values = []
    for name, rows in curves.items():
      _, color, style = ARMS[name]
      y = [row[split][NLL] for row in rows]
      values.extend(y)
      ax.plot([row['step'] for row in rows], y, color=color, linestyle=style,
               linewidth=1.8, marker='o', markersize=2.8)
    baseline = backbone_value(protocol, curves, split)
    baselines[split] = baseline
    values.append(baseline)
    lower, upper = min(values), max(values)
    span = max(upper - lower, 0.05)
    limits = (lower - 0.10 * span, upper + 0.13 * span)
    ax.set_ylim(*limits)
    ax.axhline(baseline, color='#252525', linestyle=(0, (5, 3)), linewidth=1.0)
    if split in floors:
      floor = floors[split]['value']
      inside = limits[0] <= floor <= limits[1]
      floor_display[split] = {'value': floor, 'drawn_on_axis': inside,
                              'axis_limits': list(limits)}
      if inside:
        ax.axhline(floor, color='#979797', linestyle=':', linewidth=1.2)
        ax.annotate(f'Lower bound {floor:.2f}', (0.025, floor),
                     xycoords=('axes fraction', 'data'), xytext=(0, 3),
                     textcoords='offset points', ha='left', va='bottom',
                     fontsize=8, color='#666666')
      # Bounds outside the plotted loss range remain in the sidecar. Drawing
      # an extra label inside this axis can obscure the backbone reference.
    ax.set_title(title)
    ax.set_ylabel('NLL per masked token')
  dependence_values = [0.0]
  for name, rows in curves.items():
    _, color, style = ARMS[name]
    y = [row['dev'][DEPENDENCE] for row in rows]
    dependence_values.extend(y)
    axes[2].plot([row['step'] for row in rows], y, color=color, linestyle=style,
                  linewidth=1.8, marker='o', markersize=2.8)
  lower, upper = min(dependence_values), max(dependence_values)
  span = max(upper - lower, 0.001)
  limits = (lower - 0.12 * span, upper + 0.15 * span)
  axes[2].set_ylim(*limits)
  axes[2].axhspan(limits[0], 0, facecolor='#f7eeee', zorder=-5)
  axes[2].axhline(0, color='#777777', linewidth=0.8, zorder=-3)
  axes[2].set_title('Do token relationships help?')
  axes[2].set_ylabel('Joint − independent log probability\n(nats per held-out masked token)')
  axes[2].text(0.98, 0.03, 'Below zero: joint prediction hurts',
                transform=axes[2].transAxes, ha='right', va='bottom', fontsize=8,
                color='#7d4444', bbox={'facecolor': 'white', 'alpha': 0.8,
                                       'edgecolor': 'none', 'pad': 1.0})
  for ax in axes:
    ax.set_xlim(0, max(maximum_step, 1))
    # Keep the final update label within the fixed-width PDF, including
    # four-digit endpoints in longer diagnostic runs.
    ticks = [value for value in ax.get_xticks() if 0 <= value <= maximum_step]
    ax.set_xticks(ticks)
    if ticks and ticks[-1] == maximum_step:
      ax.get_xticklabels()[-1].set_horizontalalignment('right')
    ax.set_xlabel('Training updates')
    ax.grid(axis='y', alpha=0.18, linewidth=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=8)
  legend = [Line2D([0], [0], color=ARMS[name][1], linestyle=ARMS[name][2],
                    linewidth=1.8, label=ARMS[name][0]) for name in curves]
  legend.append(Line2D([0], [0], color='#252525', linestyle=(0, (5, 3)),
                         linewidth=1, label='Frozen backbone (NLL panels)'))
  fig.legend(handles=legend, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=8.5, handlelength=2.8)
  output = output.resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  # Avoid bbox_inches='tight': it changes the PDF width and thus the final
  # font size under LaTeX's width=\linewidth scaling.
  fig.savefig(output)
  fig.savefig(output.with_suffix('.png'), dpi=180)
  plt.close(fig)
  arguments = protocol.get('arguments', {})
  metadata = {
    'artifact': 'staged_overfit_learning_curves', 'schema_version': 1,
    'purpose': protocol.get('purpose'), 'protocol': protocol.get('protocol'),
    'run_dir': str(run_dir.resolve()), 'source_sha256': sources,
    'run_marked_completed': bool(results.get('completed', False)),
    'train_examples': protocol.get('train_source', {}).get('examples',
                                                          arguments.get('examples')),
    'dev_examples': protocol.get('dev_source', {}).get('examples',
                                                      arguments.get('dev_examples')),
    'seed': arguments.get('seed'), 'planned_updates': arguments.get('steps'),
    'backbone_nll': baselines, 'support_floor_display': floor_display,
    'plotted_curves': curves,
    'dependence_definition': 'held-out log joint probability minus sum of '
      'log singleton marginals from that same model, divided by masked tokens',
    'sign': 'positive helps; negative hurts relative to the same singleton marginals',
    'uncertainty': 'single-run diagnostic curves; no confidence intervals estimated',
    'note': 'support bounds do not expand the observed NLL axes',
    'layout': {'width_inches': 5.5, 'height_inches': 6.0,
               'minimum_font_size_pt': 8,
               'panels': 'two NLL panels above one full-width dependence panel'},
  }
  output.with_suffix('.json').write_text(json.dumps(metadata, indent=2,
                                                   allow_nan=False) + '\n')
  return output


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path,
                      help='default: RUN_DIR/learning-curves.pdf')
  args = parser.parse_args(argv)
  print(draw(args.run_dir, args.output or args.run_dir / 'learning-curves.pdf'))


if __name__ == '__main__':
  main()
