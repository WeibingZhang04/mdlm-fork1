#!/usr/bin/env python3
"""Run the full-support two-token dependence learning experiment on CPU."""

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from synthetic.tiny_joint_debug import TARGETS, VARIANTS, oracle_result, train_tiny


def aggregate_report(report):
  groups = {}
  fields = ('kl_target_to_model', 'invalid_mass', 'total_variation',
            'max_marginal_error', 'mutual_information', 'steps_run',
            'joint_sample_invalid_rate', 'marginal_sample_invalid_rate')
  for row in report['runs']:
    groups.setdefault((row['task'], row['variant']), []).append(row)
  aggregate = []
  for (task, variant), rows in groups.items():
    result = {'task': task, 'variant': variant, 'seeds': [r['seed'] for r in rows],
              'target_reached_count': sum(r['stop_reason'] ==
                'distribution_target_reached' for r in rows),
              'parameter_count': rows[0]['parameter_count'],
              'gradient_active_parameter_count': rows[0][
                'gradient_active_parameter_count']}
    for field in fields:
      if field not in rows[0]:
        continue
      values = [r[field] for r in rows]
      mean = sum(values) / len(values)
      result[field] = {'mean': mean, 'min': min(values), 'max': max(values),
                       'sample_std': (math.sqrt(sum((x - mean) ** 2
                         for x in values) / (len(values) - 1))
                         if len(values) > 1 else None)}
    result['mean_probabilities'] = {outcome: sum(
      row['probabilities'][outcome] for row in rows) / len(rows)
      for outcome in ('AA', 'AB', 'BA', 'BB')}
    aggregate.append(result)
  return {'protocol': report['protocol'], 'source_sha256': report['source_sha256'],
          'run_count': len(report['runs']), 'groups': aggregate,
          'oracles': report['oracles'],
          'max_backend_probability_error': max(row[
            'max_dense_low_rank_probability_error'] for row in report['runs']),
          'max_raw_normalization_error': max(row['normalization_error']
                                             for row in report['runs']),
          'shared_head_bound': report['shared_head_bound']}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=Path(
    'artifacts/paper/staged-debugging-v1/tiny/results.json'))
  parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
  parser.add_argument('--tasks', nargs='+', choices=tuple(TARGETS),
                      default=list(TARGETS))
  parser.add_argument('--variants', nargs='+', choices=[v.name for v in VARIANTS])
  parser.add_argument('--steps', type=int, default=600)
  parser.add_argument('--samples', type=int, default=20000)
  parser.add_argument('--learning-rate', type=float, default=0.03)
  args = parser.parse_args(argv)
  torch.set_num_threads(1)
  torch.use_deterministic_algorithms(True)
  variants = [v for v in VARIANTS if not args.variants or v.name in args.variants]
  tracked_sources = ('synthetic/tiny_joint_debug.py', 'models/directional_forest.py',
                     'models/structured_decoder.py', 'structured_objective.py',
                     'structured_utils.py', 'scripts/run_tiny_joint_debug.py')
  report = {
    'protocol': 'two_token_full_vocabulary_staged_debugging_v1',
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'git_head': subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
    'source_sha256': {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()
                      for p in tracked_sources},
    'torch_version': torch.__version__, 'device': 'cpu',
    'training': 'exact expectation over all four outcomes; frozen uniform unaries',
    'metric_precision': 'enumerated FP32 probabilities renormalized in FP64; '
                        'raw normalization error recorded separately',
    'same_visible_context_for_all_targets': True, 'full_vocabulary': 2,
    'rank': 'per variant_config.rank (table and oracle use rank 2)',
    'time': 0.5, 'fixed_edge': [0, 1],
    'target_criterion': {'kl_max': 0.01, 'invalid_max': 0.01,
                         'max_marginal_error': 0.02},
    'shared_head_bound': {
      'premises': 'two tokens, uniform unaries, full vocabulary, one edge',
      'formula': 'F_AA+F_BB-F_AB-F_BA=sum_r (u_A-u_B)(v_A-v_B)>=0',
      'reason': 'shared token factors and positive contextual scale preserve order',
      'opposite_invalid_mass_lower_bound': 0.5,
      'opposite_kl_lower_bound': 0.6931471805599453,
    },
    'oracles': [], 'runs': [],
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for task in args.tasks:
    report['oracles'].append(oracle_result(task, args.samples, seed=8675309))
    for variant in variants:
      for seed in args.seeds:
        result = train_tiny(variant, task, seed, args.steps,
                            args.learning_rate, args.samples)
        report['runs'].append(result)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({key: result[key] for key in (
          'task', 'variant', 'seed', 'steps_run', 'stop_reason',
          'kl_target_to_model', 'invalid_mass', 'probabilities')}), flush=True)
  summary_path = args.output.with_name(args.output.stem + '_summary.json')
  summary_path.write_text(json.dumps(aggregate_report(report), indent=2,
                                    allow_nan=False) + '\n')
  print(f'Results: {args.output.resolve()}', flush=True)
  print(f'Summary: {summary_path.resolve()}', flush=True)


if __name__ == '__main__':
  main()
