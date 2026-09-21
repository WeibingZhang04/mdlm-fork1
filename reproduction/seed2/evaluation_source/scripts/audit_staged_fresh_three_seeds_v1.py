#!/usr/bin/env python3
"""Independently reproduce the issued three-seed conditional-likelihood report.

Only Python's standard library and NumPy are used: no production statistics,
selection, model, or aggregation code is imported. Original input files are
byte-pinned. The 5,000 crossed seed/document bootstrap draws use multiplicity
weights instead of the production bootstrap's repeated-array indexing.

Example (paths may be anywhere; input directory names are protocol-specific)::

  python scripts/audit_staged_fresh_three_seeds_v1.py \
    --experiment-root /path/to/staged-debugging-v1 \
    --combined-results /path/to/fresh-three-seeds-v1/test/results.json

This read-only arithmetic audit does not replace checkpoint/development-ledger
authentication by aggregate_staged_fresh_tests.py. It neither opens checkpoints
nor reads clean test tokens. Failure raises an exception and exits nonzero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ARMS = ('unary', 'shared', 'directional')
SEEDS = (1, 2, 3)
RATE_COUNTS = {0.25: 32, 0.5: 64, 0.75: 96, 0.9: 115}
REPLICATES = 5000
BOOTSTRAP_SEED = 1701
TOLERANCE = 1e-12
COMBINED_SELECTION_SHA256 = 'c0f1525b1722c3f02bb0b8be5c4cce744f07c3ed5d0009e6f5ef660db0f02107'
INPUT_SHA256 = {
  1: {
    'results.json': '538bf373dddcba7c0ed03a80f6f3bf594ccf1293f31895d1be98483a2f284cad',
    'test-records.jsonl': '863bc44caa063e6a9e76568392682ee8b185218d617bd7f83beaff5fb8305314',
    'selection.json': 'f05a08df0de9bd81e7b782eae44bd2672f8290288137fe6de77b9fe6f6f68ad0',
  },
  2: {
    'results.json': '94959cc0d6605ff14becbddc21290596162ddb0efb7f7f8142412a90c19ef3af',
    'test-records.jsonl': 'bdb8ab926d61cb16855149911b2b5208c75eb8e831cc226d20e91f69f11ab9f7',
    'selection.json': '6c206a23b3e0cd9c47fdae6ef589c54e4753fc1db4c7d8430433efcb3d2b9823',
  },
  3: {
    'results.json': 'c477a654dbc8efcb73ae5303563554573f9ade826277783d5e754e1a4ed2aa24',
    'test-records.jsonl': '854f0b33d82091840e011c43cc995739678a25d3c3bb1c72af5b8c2986417824',
    'selection.json': '28a0c86f007b5c17e65ebf25ad90e2dc6a9c84aa3cb62bafe92c73424ade4270',
  },
}
METRICS = ['backbone_nll']
for _arm in ARMS:
  METRICS.extend(_arm + suffix for suffix in (
    '_nll', '_joint_gain', '_singleton_gain', '_dependence_gain'))
METRICS.extend(('directional_vs_unary', 'directional_vs_shared', 'shared_vs_unary'))
PAIR_FIELDS = (
  'clean_token_sha256', 'mask_sha256', 'corrupted_token_sha256',
  'candidate_ids_sha256', 'active_token_count', 'backbone_log_probability',
  'corruption_seed', 'sigma', 'diffusion_time', 'example_index',
  'explicit_target_count',
)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def canonical_sha256(value):
  return hashlib.sha256(json.dumps(
    value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pinned_bytes(path, expected):
  payload = path.read_bytes()
  require(hashlib.sha256(payload).hexdigest() == expected, f'input digest differs: {path}')
  return payload


def context(row):
  return row['dataset'], row['document_id'], float(row['mask_rate']), row['corruption_seed']


def load_inputs(root):
  records, choices = [], []
  for seed in SEEDS:
    suffix = '' if seed == 1 else f'-seed{seed}'
    test_dir = root / f'fresh-test{suffix}-v1'
    selection_dir = root / f'fresh-selection{suffix}-v1'
    pins = INPUT_SHA256[seed]
    original = json.loads(pinned_bytes(test_dir / 'results.json', pins['results.json']))
    selection = json.loads(pinned_bytes(selection_dir / 'selection.json', pins['selection.json']))
    rows = [json.loads(line) for line in pinned_bytes(
      test_dir / 'test-records.jsonl', pins['test-records.jsonl']).splitlines() if line.strip()]
    require(len(rows) == 1536, f'seed {seed}: expected 1536 records')
    require(selection['training_seeds'] == [seed], 'selection seed differs')
    require(selection['selection_split'] == 'dev', 'selection must use development data')
    digest = selection['selection_sha256']
    require(canonical_sha256({k: v for k, v in selection.items() if k != 'selection_sha256'})
            == digest, 'selection canonical digest differs')
    require(original['selection_sha256'] == digest, 'original results selection differs')
    require(canonical_sha256(sorted(rows, key=lambda r: (r['training_seed'], context(r), r['arm'])))
            == original['test_records_sha256'], 'original records canonical digest differs')
    selected = {row['arm']: row for row in selection['selected']}
    require(set(selected) == set(ARMS) and len(selection['selected']) == 3,
            'selection does not contain exactly three arms')
    forbidden = set(selection['excluded_test_document_ids'])
    dev_windows = set(selection['dev_clean_token_sha256s'])
    for row in rows:
      require(row['split'] == 'test' and row['training_seed'] == seed, 'record split/seed differs')
      require(row['selection_sha256'] == digest, 'record selection binding differs')
      require(row['arm'] in selected, 'record arm is unknown')
      chosen = selected[row['arm']]
      require(all(row[key] == chosen[key] for key in ('checkpoint_step', 'checkpoint_sha256')),
              'record does not use its selected checkpoint')
      require(row['document_id'] not in forbidden and row['clean_token_sha256'] not in dev_windows,
              'test record reuses an excluded document/window')
    records.extend(rows)
    choices.extend(selection['selected'])
  return records, choices


def validate_grid(records):
  require(len(records) == 4608, 'expected 4608 records')
  groups, identities = {}, {}
  for row in records:
    seed, dataset, document, rate = (row[key] for key in (
      'training_seed', 'dataset', 'document_id', 'mask_rate'))
    require(type(seed) is int and seed in SEEDS, 'unexpected training seed')
    require(dataset == 'openwebtext' and rate in RATE_COUNTS, 'unexpected dataset/rate')
    require(type(row['active_token_count']) is int
            and row['active_token_count'] == RATE_COUNTS[rate], 'masked-token count differs')
    scores = [row[name] for name in ('backbone_log_probability', 'joint_log_probability',
                                   'marginal_log_probability', 'dependence_log_probability')]
    require(all(type(value) in (float, int) and math.isfinite(value) for value in scores),
            'nonfinite/non-numeric score')
    require(abs(scores[1] - scores[2] - scores[3]) <= 1e-8, 'score decomposition differs')
    if row['arm'] == 'unary':
      require(scores[1] == scores[2] and scores[3] == 0, 'unary dependence is nonzero')
    identity = row['clean_token_sha256'], row['example_index']
    require(identities.setdefault((dataset, document), identity) == identity,
            'document identity changes across seeds/rates')
    key = seed, dataset, document, float(rate)
    paired = groups.setdefault(key, {})
    require(row['arm'] in ARMS and row['arm'] not in paired, 'unknown/duplicate paired arm')
    paired[row['arm']] = row
  docs = set(identities)
  require(len(docs) == 128, 'expected 128 shared test documents')
  require({index for _, index in identities.values()} == set(range(128)), 'document indices differ')
  expected = {(seed, dataset, doc, rate) for seed in SEEDS for dataset, doc in docs for rate in RATE_COUNTS}
  require(set(groups) == expected, 'incomplete seed/document/rate grid')
  for paired in groups.values():
    require(set(paired) == set(ARMS), 'missing paired arm')
    reference = tuple(paired['unary'][field] for field in PAIR_FIELDS)
    require(all(tuple(row[field] for field in PAIR_FIELDS) == reference for row in paired.values()),
            'paired context, candidate support, or backbone score differs')
  return groups


def independent_stats(groups, rate=None):
  groups = {key: arms for key, arms in groups.items() if rate is None or key[3] == rate}
  docs = sorted({key[1:3] for key in groups})
  seed_index = {value: index for index, value in enumerate(SEEDS)}
  doc_index = {value: index for index, value in enumerate(docs)}
  values = np.zeros((3, len(docs), len(METRICS)), dtype=np.float64)
  tokens = np.zeros((3, len(docs)), dtype=np.float64)
  for (seed, dataset, document, _), arms in groups.items():
    base = arms['unary']['backbone_log_probability']
    measured = [-base]
    for arm in ARMS:
      joint, marginal = arms[arm]['joint_log_probability'], arms[arm]['marginal_log_probability']
      measured.extend((-joint, joint - base, marginal - base, joint - marginal))
    measured.extend(arms[a]['joint_log_probability'] - arms[b]['joint_log_probability']
                    for a, b in [('directional', 'unary'), ('directional', 'shared'), ('shared', 'unary')])
    index = seed_index[seed], doc_index[(dataset, document)]
    values[index] += measured
    tokens[index] += arms['unary']['active_token_count']
  require(bool(np.all(tokens > 0)), 'a seed/document has no observations')
  point = values.sum((0, 1)) / tokens.sum()
  strata = [np.array([i for i, doc in enumerate(docs) if doc[0] == dataset])
            for dataset in sorted({doc[0] for doc in docs})]
  rng = np.random.default_rng(BOOTSTRAP_SEED)
  draws = np.empty((REPLICATES, len(METRICS)))
  for iteration in range(REPLICATES):
    # One document draw is shared by all sampled seeds, rather than making
    # independent document draws inside seeds. All rates stay with a document.
    seed_counts = np.bincount(rng.integers(0, 3, 3), minlength=3)
    sampled_docs = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
    doc_counts = np.bincount(sampled_docs, minlength=len(docs))
    weights = seed_counts[:, None] * doc_counts[None, :]
    draws[iteration] = np.einsum('sd,sdm->m', weights, values) / (weights * tokens).sum()
  require(bool(np.isfinite(draws).all()), 'nonfinite bootstrap draw')
  intervals = np.quantile(draws, [0.025, 0.975], axis=0)
  return {
    'documents': len(docs), 'training_seeds': 3, 'paired_observations': len(groups),
    'active_tokens_per_arm': int(tokens.sum()),
    'metrics_nats_per_masked_token': {
      name: {'estimate': float(point[k]), 'ci95': intervals[:, k].tolist()}
      for k, name in enumerate(METRICS)},
  }


def compare_summary(expected, actual):
  for key in ('documents', 'training_seeds', 'paired_observations', 'active_tokens_per_arm'):
    require(type(actual[key]) is int and actual[key] == expected[key], f'count differs: {key}')
  metrics = actual['metrics_nats_per_masked_token']
  require(set(metrics) == set(METRICS), 'reported metric names differ')
  error, count = 0.0, 0
  for name, reference in expected['metrics_nats_per_masked_token'].items():
    reported = metrics[name]
    require(isinstance(reported['ci95'], list) and len(reported['ci95']) == 2, 'malformed interval')
    for a, b in zip([reference['estimate'], *reference['ci95']], [reported['estimate'], *reported['ci95']]):
      require(type(b) in (int, float) and math.isfinite(b), f'nonfinite reported value: {name}')
      difference = abs(a - b)
      require(math.isfinite(difference) and difference <= TOLERANCE, f'reported value differs: {name}')
      error, count = max(error, difference), count + 1
  return count, error


def audit(experiment_root, combined_results):
  records, selected = load_inputs(experiment_root)
  groups = validate_grid(records)
  payload = combined_results.read_bytes()
  result = json.loads(payload)
  require(result['artifact'] == 'fresh_pair_heldout_statistics', 'wrong combined artifact')
  require(result['selection_sha256'] == COMBINED_SELECTION_SHA256, 'combined selection binding differs')
  require(result['selected_checkpoints'] == selected, 'combined selected checkpoints differ')
  require(result['primary_contrast'] == 'directional_vs_unary', 'primary comparison differs')
  require(result['bootstrap']['replicates'] == REPLICATES
          and result['bootstrap']['seed'] == BOOTSTRAP_SEED
          and result['bootstrap']['masks_kept_with_document'] is True
          and result['bootstrap']['training_seed_uncertainty_estimated'] is True,
          'combined bootstrap protocol differs')
  require(set(result['by_mask_rate']) == {str(rate) for rate in RATE_COUNTS}, 'combined mask rates differ')
  counts, maximum_error, independent = 0, 0.0, {}
  for rate in (None, *RATE_COUNTS):
    label = 'overall' if rate is None else str(rate)
    measured = independent_stats(groups, rate)
    observed = result['overall'] if rate is None else result['by_mask_rate'][label]
    n_values, error = compare_summary(measured, observed)
    counts, maximum_error = counts + n_values, max(maximum_error, error)
    independent[label] = measured
  require(counts == 240, 'expected exactly 240 compared estimates/interval endpoints')
  return {
    'artifact': 'independent_fresh_three_seed_arithmetic_audit_v1', 'passed': True,
    'scope': 'arithmetic and pinned-shard consistency; not a replacement for checkpoint/ledger authentication',
    'experiment_root': str(experiment_root.resolve()), 'combined_results': str(combined_results.resolve()),
    'combined_results_file_sha256': hashlib.sha256(payload).hexdigest(),
    'original_input_sha256': INPUT_SHA256, 'records': len(records), 'paired_contexts': len(groups),
    'compared_values': counts, 'maximum_absolute_difference': maximum_error, 'tolerance': TOLERANCE,
    'all_count_fields_match_exactly': True, 'bootstrap_replicates': REPLICATES,
    'bootstrap_seed': BOOTSTRAP_SEED, 'numpy_version': np.__version__,
    'independent_statistics': independent,
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--experiment-root', type=Path, required=True)
  parser.add_argument('--combined-results', type=Path, required=True)
  args = parser.parse_args(argv)
  print(json.dumps(audit(args.experiment_root, args.combined_results), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
  main()
