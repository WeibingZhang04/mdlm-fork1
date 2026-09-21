#!/usr/bin/env python3
"""Reproduce the completed three-seed study from its immutable local backups.

This is a fixed result recipe, not a new evaluation or a model-selection sweep.
Every original test seal/result is pinned by its observed cloud file digest.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.aggregate_staged_fresh_tests import aggregate
from scripts.seal_staged_fresh_selection import file_sha256, seal

DATA_SHA = '70d989988c6318a52960791667804fde30e794f324628186df8d010438439753'
SHARDS = (
  (1, 'fresh-pilot-resume-v1', 'fresh-selection-v1',
   '7d425ba57cf1b5f419c8e90a3d7ddfe1e8afec6e2995ec46773b9a30e1bb5644',
   'fresh-test-v1', '538bf373dddcba7c0ed03a80f6f3bf594ccf1293f31895d1be98483a2f284cad'),
  (2, 'fresh-replication-seed2-resume-v1', 'fresh-selection-seed2-v1',
   '28206c9350d62ac410884c7b8d7c5c62e7e20307bd5b06055576ef8e92259939',
   'fresh-test-seed2-v1', '94959cc0d6605ff14becbddc21290596162ddb0efb7f7f8142412a90c19ef3af'),
  (3, 'fresh-replication-seed3-retry1-v1', 'fresh-selection-seed3-v1',
   '0169542df43d66926f2da63eec5f018aebbdaf5b8e57f010d84d3b8499a10b7c',
   'fresh-test-seed3-v1', 'c477a654dbc8efcb73ae5303563554573f9ade826277783d5e754e1a4ed2aa24'),
)


def reproduce(experiment_root, output_dir):
  if output_dir.exists():
    raise FileExistsError(output_dir)
  run_dirs, shards = {}, []
  for seed, run, selection, manifest_sha, test, result_sha in SHARDS:
    run_dir = experiment_root / run
    manifest = experiment_root / selection / 'selection-manifest.json'
    if file_sha256(manifest) != manifest_sha:
      raise ValueError(f'seed {seed}: original selection manifest changed')
    original_seal = json.loads(manifest.read_text())
    ledger_path = run_dir / 'hashes.json'
    if file_sha256(ledger_path) != original_seal['runs'][0]['hash_ledger_file_sha256']:
      raise ValueError(f'seed {seed}: final training ledger changed')
    ledger = json.loads(ledger_path.read_text())
    for name, expected in ledger.items():
      target = (run_dir / name).resolve()
      if not target.is_relative_to(run_dir.resolve()) or file_sha256(target) != expected:
        raise ValueError(f'seed {seed}: backup file differs: {name}')
    run_dirs[seed] = run_dir
    shards.append([manifest, manifest_sha, experiment_root / test, result_sha])

  data = experiment_root / 'fresh-data-v1' / 'manifest.json'
  selection_dir = output_dir / 'selection'
  seal(list(run_dirs.values()), data, DATA_SHA, selection_dir)
  combined_manifest = selection_dir / 'selection-manifest.json'
  summary, lineage = aggregate(
    combined_manifest, file_sha256(combined_manifest), data, DATA_SHA, shards,
    output_dir / 'test', run_dirs=run_dirs,
    bootstrap_replicates=5000, bootstrap_seed=1701)
  from scripts.plot_staged_fresh_multiseed import draw
  draw(list(run_dirs.values()), output_dir / 'test', selection_dir / 'selection.json',
       output_dir / 'learning-and-test.pdf')
  print(json.dumps({'event': 'three_seed_result_reproduced',
                    'records': lineage['records'], 'training_seeds': lineage['training_seeds'],
                    'results_sha256': file_sha256(output_dir / 'test' / 'results.json'),
                    'selection_sha256': summary['selection_sha256'],
                    'output_dir': str(output_dir.resolve())}, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--experiment-root', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()
  reproduce(args.experiment_root, args.output_dir)


if __name__ == '__main__':
  main()
