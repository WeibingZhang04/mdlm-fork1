#!/usr/bin/env python3
"""Reproduce the paper's joint-prediction figures from immutable artifacts.

No models are fit and no examples, seeds, checkpoints, or mask rates are selected
here. The city-name distribution is a labeled illustration, not an observation.
The new between-model decomposition is computed from paired, saved test scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.fresh_pair_statistics import _paired_test_rows, summarize_test
from scripts.plot_staged_fresh import sha
from scripts.plot_staged_fresh_multiseed import read_inputs

BASE = REPO_ROOT / 'artifacts/paper/staged-debugging-v1'
PINS = {
  'fresh-three-seeds/test/test-records.jsonl':
    'e35c8b45b06a67f5ba10d08169e3123419af38c7254d8e13b7af3220a564f1a0',
  'fresh-three-seeds/test/results.json':
    '4aefe030903e42746c79ae9baa67e04758d5bb3bb7247bd7fd67fe88e049be7f',
  'fresh-three-seeds/selection/selection.json':
    '05deaa0a7287d6d722e67d0e45b53ba6bd7148ce153afd354296d1b0d0d1954e',
  'tiny/results.json':
    'aff5a94c6069982855d94e8e2bf6fb8c0d8fc1cbc458fe5a6aae670f10d2832b',
  'tiny/rank2_results.json':
    '3151c713fb083f04e28597837aa6b0bee50c662eb727f245e1a656918d560efd',
  'tiny/figure_matched.json':
    'e4d779676a573fdc5b87cdf681dceb0e627ec94d4451e181fcf9010442ed4aeb',
}
ARMS = {
  'shared': ('Tied factors', '#697581', '--'),
  'directional': ('Separate endpoints', '#0072B2', '-'),
  'unary': ('Independent adapter', '#D55E00', '-.'),
}


def save_figure(fig, stem, figure_dir, artifact_dir, metadata):
  """Keep a fixed 5.5-inch PDF canvas; do not shrink text with tight cropping."""
  if not np.isclose(fig.get_size_inches()[0], 5.5):
    raise ValueError('paper figures must have native width 5.5 inches')
  sizes = [obj.get_fontsize() for obj in fig.findobj(plt.Text) if obj.get_text()]
  if not sizes or min(sizes) < 8:
    raise ValueError('figure contains text smaller than 8 points')
  pdf, png = figure_dir / f'{stem}.pdf', artifact_dir / f'{stem}.png'
  fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
  fig.savefig(png, dpi=180)
  metadata.update({
    'plot_script_sha256': sha(Path(__file__)),
    'layout_inches': fig.get_size_inches().tolist(),
    'minimum_font_points': min(sizes),
    'output_pdf_sha256': sha(pdf), 'output_png_sha256': sha(png),
  })
  (artifact_dir / f'{stem}.json').write_text(
    json.dumps(metadata, indent=2, allow_nan=False) + '\n')
  plt.close(fig)


def motivation(figure_dir, artifact_dir):
  fig = plt.figure(figsize=(5.5, 3.05))
  fig.text(.5, .966, 'Same individual-token probabilities',
           ha='center', va='top', fontsize=10)
  fig.text(.26, .874, 'First position: New 50%, San 50%', ha='center', fontsize=8.5)
  fig.text(.76, .874, 'Second position: York 50%, Diego 50%', ha='center', fontsize=8.5)
  matrices = [np.full((2, 2), .25), np.diag([.5, .5])]
  labels = ['Independent sampling', 'Intended joint distribution']
  for index, (matrix, title) in enumerate(zip(matrices, labels)):
    ax = fig.add_axes([.15 + index * .50, .27, .30, .45])
    ax.imshow(matrix, vmin=0, vmax=.5, cmap='Blues')
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_xticks([0, 1], ['York', 'Diego'])
    ax.set_yticks([0, 1], ['New', 'San'])
    ax.tick_params(length=0, labelsize=9)
    ax.spines[:].set_visible(False)
    for row in range(2):
      for col in range(2):
        crossed = row != col
        if crossed:
          ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, fill=False,
                                 edgecolor='#b74b3d', linewidth=1.2, hatch='//'))
        ax.text(col, row, f'{matrix[row, col]:.0%}', ha='center', va='center',
                fontsize=11, color='white' if matrix[row, col] > .3 else '#17324d',
                bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=1)
                if crossed else None)
    fig.text(.30 + index * .50, .145,
             f'Crossed combinations: {matrix[0, 1] + matrix[1, 0]:.0%}',
             ha='center', fontsize=9, color='#a33b30')
  fig.text(.5, .036, 'Illustrative distribution; hatched cells are New Diego and San York.',
           ha='center', fontsize=8)
  save_figure(fig, 'joint-prediction-example', figure_dir, artifact_dir, {
    'artifact': 'illustrative_joint_prediction_distribution',
    'empirical': False, 'row_tokens': ['New', 'San'],
    'column_tokens': ['York', 'Diego'],
    'independent_probability': matrices[0].tolist(),
    'intended_joint_probability': matrices[1].tolist(),
    'first_position_probability': [.5, .5], 'second_position_probability': [.5, .5],
    'crossed_combinations': ['New Diego', 'San York'],
  })


def mask_rates(test, sources, figure_dir, artifact_dir):
  fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9))
  fig.subplots_adjust(left=.115, right=.98, bottom=.28, top=.83, wspace=.42)
  rates = sorted(test['by_mask_rate'], key=float)
  groups = [test['overall']] + [test['by_mask_rate'][rate] for rate in rates]
  labels = ['Pooled'] + [f'{float(rate):.0%}' for rate in rates]
  for ax, key, title in zip(axes,
      ['directional_vs_unary', 'directional_dependence_gain'],
      ['Separate endpoints vs.\nindependent adapter', 'Joint vs.\nown marginals']):
    for y, group in enumerate(groups):
      metric = group['metrics_nats_per_masked_token'][key]
      color = '#0072B2' if y == 0 else '#697581'
      ax.hlines(y, *metric['ci95'], color=color, linewidth=1.6)
      ax.plot(metric['estimate'], y, 'o', color=color, markersize=4)
    ax.axvline(0, color='#555555', linewidth=.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(4.4, -.4)
    ax.set_title(title, fontsize=9, pad=9)
    ax.set_xlabel('Log-score gain\n(nats per masked token)', fontsize=8.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.grid(axis='x', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
  fig.text(.5, .045, '95% crossed seed/document-bootstrap intervals; three training seeds.',
           ha='center', fontsize=8, color='#555555')
  save_figure(fig, 'fresh-mask-rate-comparison', figure_dir, artifact_dir, {
    'artifact': 'fresh_mask_rate_comparison', 'source_sha256': sources,
    'test_overall': test['overall'], 'test_by_mask_rate': test['by_mask_rate'],
    'bootstrap': test['bootstrap'],
  })


def development(runs, selection, sources, figure_dir, artifact_dir):
  fig, axes = plt.subplots(1, len(runs), figsize=(5.5, 2.9), sharey=True)
  fig.subplots_adjust(left=.15, right=.98, bottom=.28, top=.77, wspace=.18)
  handles = []
  for index, ((seed, run), ax) in enumerate(zip(sorted(runs.items()), axes)):
    chosen = {row['arm']: row['checkpoint_step'] for row in selection['selected']
              if row['training_seed'] == seed}
    for arm, (label, color, style) in ARMS.items():
      points = run['evaluations']
      line, = ax.plot([point['step'] for point in points],
                     [point['arms'][arm]['joint_gain_nats_per_masked_token'] for point in points],
                     color=color, linestyle=style, linewidth=1.5, label=label)
      point = next(point for point in points if point['step'] == chosen[arm])
      ax.plot(chosen[arm], point['arms'][arm]['joint_gain_nats_per_masked_token'],
              marker='o', markersize=4, color=color, markeredgecolor='white', zorder=4)
      if index == 0:
        handles.append(line)
    ax.axhline(0, color='#555555', linewidth=.7)
    ax.set(title=f'Seed {seed}', xlabel='Updates', xticks=[0, 500, 1000])
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
  axes[0].set_ylabel('Gain over backbone\n(nats per masked token)', fontsize=8.5)
  fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .975),
             ncol=3, frameon=False, fontsize=8, columnspacing=1, handlelength=2)
  fig.text(.5, .06, 'Development set; dots mark checkpoints selected by lowest loss.',
           ha='center', fontsize=8, color='#555555')
  save_figure(fig, 'fresh-development-curves', figure_dir, artifact_dir, {
    'artifact': 'fresh_development_checkpoint_curves', 'source_sha256': sources,
    'training_seeds': sorted(runs), 'selected_checkpoints': selection['selected'],
  })


def tiny(figure_dir, artifact_dir):
  original = json.loads((BASE / 'tiny/figure_matched.json').read_text())
  rows = []
  for filename, variant in [('results.json', original['shared_variant']),
                            ('rank2_results.json', original['directional_variant'])]:
    data = json.loads((BASE / 'tiny' / filename).read_text())
    matches = [row for row in data['runs'] if row['task'] == 'opposite'
               and row['variant'] == variant and row['seed'] == original['seed']]
    if len(matches) != 1:
      raise ValueError('tiny-model artifact must contain exactly one prespecified fit')
    rows.append(matches[0])
  def matrix(values):
    return np.array([values[key] for key in ['AA', 'AB', 'BA', 'BB']]).reshape(2, 2)
  panels = [np.array([[0., .5], [.5, 0.]]), matrix(rows[0]['probabilities']),
            matrix(rows[1]['probabilities']), matrix(rows[1]['independent_marginal_probabilities'])]
  np.testing.assert_array_equal(panels, original['panels'])
  labels = ['Target', 'Tied factors', 'Separate\nendpoints', 'Same marginals,\nindependent draws']
  fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.4))
  fig.subplots_adjust(left=.085, right=.985, bottom=.32, top=.74, wspace=.28)
  for index, (ax, panel, label) in enumerate(zip(axes, panels, labels)):
    ax.imshow(panel, vmin=0, vmax=.5, cmap='Blues')
    ax.set_title(label, fontsize=8.5, pad=8)
    ax.set_xticks([0, 1], ['A', 'B'])
    ax.set_yticks([0, 1], ['A', 'B'])
    ax.tick_params(length=0, labelsize=8)
    ax.spines[:].set_visible(False)
    if index == 0:
      ax.set_ylabel('First token', fontsize=8)
    for first in range(2):
      for second in range(2):
        value = panel[first, second]
        text = f'{100 * value:.3f}%' if 0 < value < .001 else f'{100 * value:.1f}%'
        ax.text(second, first, text, ha='center', va='center',
                fontsize=8, color='white' if value > .28 else '#17324d')
    ax.text(.5, -.40, f'{100 * np.trace(panel):.3f}% invalid',
            transform=ax.transAxes, ha='center', fontsize=8)
  fig.text(.54, .06, 'Second token (AA and BB are invalid)', ha='center', fontsize=8)
  save_figure(fig, 'tiny-joint-comparison', figure_dir, artifact_dir, {
    'artifact': 'matched_parameter_two_token_joint_comparison',
    'source_sha256': {key: value for key, value in PINS.items() if key.startswith('tiny/')},
    'seed': original['seed'], 'task': 'opposite', 'panel_order': labels,
    'panels': [panel.tolist() for panel in panels],
    'shared_variant': original['shared_variant'],
    'directional_variant': original['directional_variant'],
    'shared_config': original['shared_config'], 'directional_config': original['directional_config'],
    'shared_gradient_active_parameter_count': rows[0]['gradient_active_parameter_count'],
    'directional_gradient_active_parameter_count': rows[1]['gradient_active_parameter_count'],
    'caption_details': original['caption_details'],
  })


def decomposition(records, selection, test, sources, artifact_dir):
  groups = _paired_test_rows(records, selection)
  names = ['tied_marginal_product_nll', 'separate_marginal_product_nll',
           'marginal_improvement', 'dependence_improvement', 'joint_improvement']
  def summarize(subset):
    seeds = sorted({seed for seed, _ in subset})
    docs = sorted({context[:2] for _, context in subset})
    si, di = {v: i for i, v in enumerate(seeds)}, {v: i for i, v in enumerate(docs)}
    values = np.zeros((len(seeds), len(docs), len(names)), dtype=np.float64)
    counts = np.zeros((len(seeds), len(docs)), dtype=np.float64)
    max_identity_error = 0.
    for (seed, context), arms in subset.items():
      tied, separate = arms['shared'], arms['directional']
      tied_m, separate_m = tied['marginal_log_probability'], separate['marginal_log_probability']
      marginal = separate_m - tied_m
      dependence = ((separate['joint_log_probability'] - separate_m)
                    - (tied['joint_log_probability'] - tied_m))
      joint = separate['joint_log_probability'] - tied['joint_log_probability']
      max_identity_error = max(max_identity_error, abs(joint - marginal - dependence))
      index = si[seed], di[context[:2]]
      values[index] += [-tied_m, -separate_m, marginal, dependence, joint]
      counts[index] += tied['active_token_count']
    if not np.all(counts > 0):
      raise ValueError('incomplete seed/document crossing')
    estimate = values.sum(axis=(0, 1)) / counts.sum()
    strata = [np.array([index for index, doc in enumerate(docs) if doc[0] == dataset])
              for dataset in sorted({doc[0] for doc in docs})]
    rng = np.random.default_rng(test['bootstrap']['seed'])
    draws = np.empty((test['bootstrap']['replicates'], len(names)))
    for index in range(len(draws)):
      sampled_seeds = rng.integers(0, len(seeds), len(seeds))
      sampled_docs = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
      sampled = np.ix_(sampled_seeds, sampled_docs)
      draws[index] = values[sampled].sum(axis=(0, 1)) / counts[sampled].sum()
    bounds = np.quantile(draws, [.025, .975], axis=0)
    return {
      'metrics_nats_per_masked_token': {
        name: {'estimate': float(estimate[i]), 'ci95': bounds[:, i].tolist()}
        for i, name in enumerate(names)},
      'dependence_fraction_of_mean_joint_improvement': float(estimate[3] / estimate[4]),
      'max_per_observation_identity_error_nats': max_identity_error,
      'paired_observations': len(subset), 'active_tokens_per_arm': int(counts.sum()),
    }
  overall = summarize(groups)
  by_rate = {str(rate): summarize({key: value for key, value in groups.items() if key[1][2] == rate})
             for rate in selection['mask_rates']}
  for new, old in [(overall, test['overall']),
                   *[(by_rate[rate], test['by_mask_rate'][rate]) for rate in by_rate]]:
    calculated = new['metrics_nats_per_masked_token']['joint_improvement']
    prior = old['metrics_nats_per_masked_token']['directional_vs_shared']
    np.testing.assert_allclose([calculated['estimate'], *calculated['ci95']],
                                [prior['estimate'], *prior['ci95']], rtol=0, atol=1e-14)
  output = {
    'artifact': 'tied_to_separate_likelihood_decomposition', 'source_sha256': sources,
    'plot_script_sha256': sha(Path(__file__)),
    'definition': 'log q_separate - log q_tied = difference of marginal-product log scores '
                  '+ difference of joint-versus-own-marginal log scores',
    'positive_improvement_is_better': True, 'bootstrap': test['bootstrap'],
    'fraction_scope': 'ratio of pooled mean dependence improvement to pooled mean joint improvement; '
                      'descriptive only, not a significance claim',
    'overall': overall, 'by_mask_rate': by_rate,
  }
  (artifact_dir / 'between-model-decomposition.json').write_text(
    json.dumps(output, indent=2, allow_nan=False) + '\n')
  return output


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--figure-dir', type=Path, required=True)
  parser.add_argument('--artifact-dir', type=Path,
                      default=REPO_ROOT / 'artifacts/paper/expert-revision-v1')
  args = parser.parse_args(argv)
  for name, expected in PINS.items():
    if sha(BASE / name) != expected:
      raise ValueError(f'immutable artifact hash mismatch: {name}')
  args.figure_dir.mkdir(parents=True, exist_ok=True)
  args.artifact_dir.mkdir(parents=True, exist_ok=True)
  plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                       'axes.labelsize': 8.5, 'axes.titlesize': 9,
                       'xtick.labelsize': 8, 'ytick.labelsize': 8,
                       'pdf.fonttype': 42, 'ps.fonttype': 42})
  root = BASE / 'fresh-three-seeds'
  runs, test, selection, sources = read_inputs(
    [BASE / 'fresh-pilot/training', root / 'seed2/training', root / 'seed3/training'],
    root / 'test', root / 'selection/selection.json')
  records = [json.loads(line) for line in (root / 'test/test-records.jsonl').read_text().splitlines()]
  if len(records) != 4608:
    raise ValueError('expected every completed three-seed test record')
  reproduced = summarize_test(records, selection,
    bootstrap_replicates=test['bootstrap']['replicates'], bootstrap_seed=test['bootstrap']['seed'])
  if any(test.get(key) != value for key, value in reproduced.items()):
    raise ValueError('saved test report did not reproduce exactly from immutable records')
  sources['test_records'] = PINS['fresh-three-seeds/test/test-records.jsonl']
  sources['statistics_implementation'] = sha(REPO_ROOT / 'evaluation/fresh_pair_statistics.py')
  sources['input_validation_implementation'] = sha(REPO_ROOT / 'scripts/plot_staged_fresh_multiseed.py')
  motivation(args.figure_dir, args.artifact_dir)
  mask_rates(test, sources, args.figure_dir, args.artifact_dir)
  development(runs, selection, sources, args.figure_dir, args.artifact_dir)
  tiny(args.figure_dir, args.artifact_dir)
  analysis = decomposition(records, selection, test, sources, args.artifact_dir)
  print(json.dumps(analysis['overall'], indent=2))


if __name__ == '__main__':
  main()
