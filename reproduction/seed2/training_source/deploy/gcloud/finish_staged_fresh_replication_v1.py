#!/usr/bin/env python3
"""Seal complete replications, score only new seeds, and combine original shards.

Run from the authenticated analysis checkout after seeds 2 and 3 finish, or
use --single-seed to export one completed replica without combining shards.
All output namespaces are exclusive. A failed stage leaves its artifacts in
place and exits; it never replaces an earlier seal or test export.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_staged_fresh_tests import aggregate
from scripts.evaluate_staged_fresh_selected import main as evaluate
from scripts.seal_staged_fresh_selection import file_sha256, seal


DATA_SHA = '70d989988c6318a52960791667804fde30e794f324628186df8d010438439753'
TEST_SHA = '2e22dae402fe61c6febf2a0d48c8161c1c8bb44a757061730c11205d2b6bffbb'
EXPECTATIONS_SHA = 'a978c17befc2788b0b773eee7cd608ac0200379947a63a0ba8699a0b9818c956'
PILOT_MANIFEST_SHA = '7d425ba57cf1b5f419c8e90a3d7ddfe1e8afec6e2995ec46773b9a30e1bb5644'
PILOT_TEST_SHA = '538bf373dddcba7c0ed03a80f6f3bf594ccf1293f31895d1be98483a2f284cad'


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--experiment-root', type=Path, required=True)
  parser.add_argument('--run-tag', choices=('v1', 'retry1-v1'), default='retry1-v1')
  parser.add_argument('--seed2-resumed', action='store_true')
  parser.add_argument('--single-seed', type=int, choices=(2, 3),
                      help='seal and evaluate only this completed replica; do not combine or aggregate')
  parser.add_argument('--output-tag', choices=('v1',), default='v1')
  args = parser.parse_args(argv)
  root, tag = args.experiment_root.resolve(), args.output_tag
  release = root.parent / 'releases/contextual-forest-adapter-cf8b808-20260831T131607Z'
  manifest = root / 'fresh-data-v1/manifest.json'
  pilot_run = root / 'fresh-pilot-resume-v1'
  replica_runs = [root / f'fresh-replication-seed{seed}-{args.run_tag}' for seed in (2, 3)]
  if args.seed2_resumed:
    replica_runs[0] = root / 'fresh-replication-seed2-resume-v1'
  if args.single_seed is not None:
    replica_runs = [replica_runs[args.single_seed - 2]]
  new_seeds_tag = str(args.single_seed) if args.single_seed is not None else '23'
  selection_new = root / f'fresh-selection-seed{new_seeds_tag}-{tag}'
  selection123 = root / f'fresh-selection-seed123-{tag}'
  test_new = root / f'fresh-test-seed{new_seeds_tag}-{tag}'
  test123 = root / f'fresh-test-seed123-{tag}'
  outputs = ((selection_new, test_new) if args.single_seed is not None
             else (selection_new, selection123, test_new, test123))
  for path in outputs:
    if path.exists() or path.is_symlink():
      raise FileExistsError(path)
  os.environ.update(
    HF_HOME=str(root.parent / 'hf-home'),
    HUGGINGFACE_HUB_CACHE=str(root.parent / 'huggingface/hub'),
    HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

  # These functions require final ledgers and the complete development grid.
  _, sealed_new = seal(replica_runs, manifest, DATA_SHA, selection_new)
  if args.single_seed is None:
    seal([pilot_run, *replica_runs], manifest, DATA_SHA, selection123)
  elif {item['training_seed'] for item in sealed_new['selected_checkpoint_files']} != {args.single_seed}:
    raise ValueError('sealed checkpoint identities do not match --single-seed')
  print(json.dumps({'event': 'replication_development_choices_sealed',
                    'new_selected': sealed_new['selected_checkpoint_files'],
                    'test_tokens_read': False}), flush=True)
  command = [
    '--selection', str(selection_new / 'selection.json'),
    '--expected-selection-sha256', sealed_new['selection_file_sha256'],
    '--data-manifest', str(manifest), '--expected-data-manifest-sha256', DATA_SHA,
    '--test-jsonl', str(root / 'fresh-data-v1/test.jsonl'), '--expected-test-jsonl-sha256', TEST_SHA,
    '--expectations', str(release / 'expectations/production-expectations-v2.json'),
    '--expected-expectations-sha256', EXPECTATIONS_SHA,
    '--backbone-checkpoint', str(release / 'inputs/mdlm-owt-backbone-schema-v2.pt'),
    '--output-dir', str(test_new),
  ]
  for path in sorted({item['path'] for item in sealed_new['selected_checkpoint_files']}):
    command.extend(['--selected-checkpoint', path])
  evaluate(command)
  if args.single_seed is not None:
    print(json.dumps({'event': 'replication_seed_test_complete', 'output_dir': str(test_new),
                      'training_seeds': [args.single_seed]}), flush=True)
    return
  summary, lineage = aggregate(
    selection123 / 'selection-manifest.json',
    file_sha256(selection123 / 'selection-manifest.json'), manifest, DATA_SHA,
    [
      [root / 'fresh-selection-v1/selection-manifest.json', PILOT_MANIFEST_SHA,
       root / 'fresh-test-v1', PILOT_TEST_SHA],
      [selection_new / 'selection-manifest.json', file_sha256(selection_new / 'selection-manifest.json'),
       test_new, file_sha256(test_new / 'results.json')],
    ], test123)
  print(json.dumps({'event': 'replication_test_complete', 'output_dir': str(test123),
                    'training_seeds': lineage['training_seeds'], 'records': lineage['records'],
                    'overall': summary['overall']}), flush=True)


if __name__ == '__main__':
  main()
