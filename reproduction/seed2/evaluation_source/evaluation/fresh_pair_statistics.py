"""Paired held-out statistics for freshly trained fixed-graph output heads.

Select checkpoints from development records first. Test reporting then checks
that each arm uses its selected state, all arms score the same corruptions,
and test documents exclude development and explicitly forbidden documents.
Uncertainty resamples source documents (with all their masks kept together)
and training seeds. These are likelihood comparisons, not generation results.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ARMS = ('unary', 'shared', 'directional')
SHA_FIELDS = ('checkpoint_sha256', 'clean_token_sha256', 'mask_sha256', 'corrupted_token_sha256')
SCORE_FIELDS = ('backbone_log_probability', 'marginal_log_probability',
                'joint_log_probability', 'dependence_log_probability')


def canonical_sha256(value) -> str:
  payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
  return hashlib.sha256(payload.encode()).hexdigest()


def _validate_row(row, split):
  if row.get('split') != split or row.get('arm') not in ARMS:
    raise ValueError(f'expected split={split} and arm in {ARMS}')
  for field in ('dataset', 'document_id'):
    if not isinstance(row.get(field), str) or not row[field]:
      raise ValueError(f'{field} must be a nonempty string')
  for field in ('training_seed', 'checkpoint_step', 'corruption_seed', 'active_token_count'):
    value = row.get(field)
    if type(value) is not int or value < (1 if field == 'active_token_count' else 0):
      raise ValueError(f'invalid {field}')
  if type(row.get('mask_rate')) not in (int, float) or not 0 < row['mask_rate'] < 1:
    raise ValueError('mask_rate must lie strictly between zero and one')
  for field in SHA_FIELDS:
    value = row.get(field)
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
      raise ValueError(f'invalid {field}')
  for field in SCORE_FIELDS:
    if type(row.get(field)) not in (int, float) or not math.isfinite(row[field]):
      raise ValueError(f'{field} must be finite')
    if field != 'dependence_log_probability' and row[field] > 1e-6:
      raise ValueError(f'{field} cannot be positive for a normalized distribution')
  observed = row['joint_log_probability'] - row['marginal_log_probability']
  if not math.isclose(observed, row['dependence_log_probability'], rel_tol=1e-5, abs_tol=5e-3):
    raise ValueError('joint/marginal/dependence scores do not decompose')
  if row['arm'] == 'unary' and not math.isclose(observed, 0.0, abs_tol=1e-8):
    raise ValueError('unary arm must have zero dependence')


def _context(row):
  return (row['dataset'], row['document_id'], float(row['mask_rate']), row['corruption_seed'])


def _signature(row):
  return tuple(row[field] for field in (
    'clean_token_sha256', 'mask_sha256', 'corrupted_token_sha256',
    'active_token_count', 'backbone_log_probability'))


def select_checkpoints(dev_records: list[dict], *, excluded_test_document_ids=()) -> dict:
  """Seal per-arm/per-seed choices using total dev joint NLL only.

  All arms and checkpoints must see the same development corruptions within
  each training seed. Exact ties choose the earlier checkpoint. Pass training
  document IDs and all previously examined debug document IDs in
  ``excluded_test_document_ids``; development IDs are added automatically.
  """
  if not dev_records:
    raise ValueError('development records are empty')
  groups, contexts = defaultdict(dict), {}
  checkpoint_steps = defaultdict(set)
  for row in dev_records:
    _validate_row(row, 'dev')
    group = (row['training_seed'], row['arm'], row['checkpoint_step'])
    key = _context(row)
    if key in groups[group]:
      raise ValueError('duplicate development observation')
    groups[group][key] = row
    checkpoint_steps[(row['training_seed'], row['arm'])].add(row['checkpoint_step'])
    signature_key = (row['training_seed'], key)
    signature = _signature(row)
    if contexts.setdefault(signature_key, signature) != signature:
      raise ValueError('development masks, targets, counts or backbone scores drifted')
  seeds = sorted({row['training_seed'] for row in dev_records})
  rates = {float(row['mask_rate']) for row in dev_records}
  dev_doc_rates = defaultdict(set)
  for seed, context in contexts:
    dev_doc_rates[(seed, context[0], context[1])].add(context[2])
  if any(observed != rates for observed in dev_doc_rates.values()):
    raise ValueError('each development document must cover the mask-rate grid')
  selected = []
  for seed in seeds:
    expected_steps = checkpoint_steps[(seed, ARMS[0])]
    expected_contexts = {key for context_seed, key in contexts if context_seed == seed}
    if not expected_steps:
      raise ValueError('a development arm is missing')
    for arm in ARMS:
      if checkpoint_steps[(seed, arm)] != expected_steps:
        raise ValueError('arms must use the same development checkpoint grid')
      candidates = []
      for step in sorted(expected_steps):
        rows = groups[(seed, arm, step)]
        if set(rows) != expected_contexts:
          raise ValueError('development checkpoints do not cover identical observations')
        hashes = {row['checkpoint_sha256'] for row in rows.values()}
        if len(hashes) != 1:
          raise ValueError('one checkpoint step names multiple checkpoint files')
        count = sum(row['active_token_count'] for row in rows.values())
        nll = -sum(row['joint_log_probability'] for row in rows.values())
        candidates.append({'training_seed': seed, 'arm': arm, 'checkpoint_step': step,
                           'checkpoint_sha256': next(iter(hashes)), 'dev_nll_sum': nll,
                           'dev_active_tokens': count, 'dev_nll_per_masked_token': nll / count})
      selected.append(min(candidates, key=lambda row: (row['dev_nll_sum'], row['checkpoint_step'])))
  dev_ids = {row['document_id'] for row in dev_records}
  excluded = set(excluded_test_document_ids)
  if any(not isinstance(value, str) or not value for value in excluded):
    raise ValueError('excluded document IDs must be nonempty strings')
  result = {
    'artifact': 'fresh_pair_checkpoint_selection', 'schema_version': 1,
    'selection_split': 'dev', 'selection_rule': 'lowest total dev joint NLL; earlier checkpoint breaks ties',
    'arms': list(ARMS), 'training_seeds': seeds,
    'mask_rates': sorted(rates),
    'dev_document_ids': sorted(dev_ids),
    'dev_clean_token_sha256s': sorted({row['clean_token_sha256'] for row in dev_records}),
    'excluded_test_document_ids': sorted(dev_ids | excluded),
    'external_exclusion_document_count': len(excluded),
    'dev_records_sha256': canonical_sha256(sorted(dev_records, key=lambda row: (
      row['training_seed'], row['arm'], row['checkpoint_step'], _context(row)))),
    'selected': selected,
  }
  result['selection_sha256'] = canonical_sha256(result)
  return result


def _paired_test_rows(test_records, selection):
  payload = {key: value for key, value in selection.items() if key != 'selection_sha256'}
  if canonical_sha256(payload) != selection.get('selection_sha256'):
    raise ValueError('checkpoint selection digest mismatch')
  if selection.get('artifact') != 'fresh_pair_checkpoint_selection' or selection.get('selection_split') != 'dev':
    raise ValueError('expected a development-only checkpoint selection')
  expected = {(row['training_seed'], row['arm']): row for row in selection['selected']}
  if set(expected) != {(seed, arm) for seed in selection['training_seeds'] for arm in ARMS}:
    raise ValueError('selection does not cover all arms and seeds')
  forbidden = set(selection['excluded_test_document_ids'])
  seen_dev_windows = set(selection['dev_clean_token_sha256s'])
  groups = defaultdict(dict)
  for row in test_records:
    _validate_row(row, 'test')
    if row.get('selection_sha256') != selection['selection_sha256']:
      raise ValueError('test record is not bound to the selected checkpoints')
    chosen = expected.get((row['training_seed'], row['arm']))
    if chosen is None or any(row[field] != chosen[field] for field in ('checkpoint_step', 'checkpoint_sha256')):
      raise ValueError('test record uses an unselected checkpoint')
    if row['document_id'] in forbidden or row['clean_token_sha256'] in seen_dev_windows:
      raise ValueError('test reuses a development, training or excluded debug document/window')
    key = (row['training_seed'], _context(row))
    if row['arm'] in groups[key]:
      raise ValueError('duplicate test observation')
    groups[key][row['arm']] = row
  if not groups:
    raise ValueError('test records are empty')
  doc_sets = defaultdict(set)
  doc_rates = defaultdict(set)
  identities = {}
  for (seed, context), arms in groups.items():
    if set(arms) != set(ARMS):
      raise ValueError('test observation is missing a paired arm')
    reference = arms[ARMS[0]]
    if any(_signature(row) != _signature(reference) for row in arms.values()):
      raise ValueError('paired test masks, targets, counts or backbone scores differ')
    dataset, document, rate, _ = context
    doc_sets[seed].add((dataset, document))
    doc_rates[(seed, dataset, document)].add(rate)
    identity = (dataset, reference['clean_token_sha256'])
    if identities.setdefault(document, identity) != identity:
      raise ValueError('document identity or clean window changes across observations')
  if set(doc_sets) != set(selection['training_seeds']):
    raise ValueError('test records omit a selected training seed')
  reference_docs = next(iter(doc_sets.values()))
  if any(documents != reference_docs for documents in doc_sets.values()):
    raise ValueError('training seeds must evaluate the same test documents')
  if any(rates != set(selection['mask_rates']) for rates in doc_rates.values()):
    raise ValueError('each test document must cover the development mask-rate grid')
  return groups


def _metric_values(arms):
  base = arms[ARMS[0]]['backbone_log_probability']
  values = {'backbone_nll': -base}
  for arm in ARMS:
    row = arms[arm]
    values[f'{arm}_nll'] = -row['joint_log_probability']
    values[f'{arm}_joint_gain'] = row['joint_log_probability'] - base
    values[f'{arm}_singleton_gain'] = row['marginal_log_probability'] - base
    values[f'{arm}_dependence_gain'] = row['joint_log_probability'] - row['marginal_log_probability']
  for better, reference in [('directional', 'unary'), ('directional', 'shared'), ('shared', 'unary')]:
    values[f'{better}_vs_{reference}'] = arms[better]['joint_log_probability'] - arms[reference]['joint_log_probability']
  return values


def _summarize(groups, *, bootstrap_replicates, bootstrap_seed):
  seeds = sorted({seed for seed, _ in groups})
  docs = sorted({context[:2] for _, context in groups})
  seed_index, doc_index = {value: i for i, value in enumerate(seeds)}, {value: i for i, value in enumerate(docs)}
  names = list(_metric_values(next(iter(groups.values()))))
  numerators = np.zeros((len(seeds), len(docs), len(names)), dtype=np.float64)
  counts = np.zeros((len(seeds), len(docs)), dtype=np.float64)
  for (seed, context), arms in groups.items():
    index = seed_index[seed], doc_index[context[:2]]
    values = _metric_values(arms)
    numerators[index] += [values[name] for name in names]
    counts[index] += arms[ARMS[0]]['active_token_count']
  if not np.all(counts > 0):
    raise ValueError('every sampled document needs observations in every training seed')
  point = numerators.sum(axis=(0, 1)) / counts.sum()
  # Same document draw across all sampled seeds preserves their crossed
  # evaluation structure. Dataset strata preserve corpus composition.
  strata = [np.array([index for index, doc in enumerate(docs) if doc[0] == dataset])
            for dataset in sorted({doc[0] for doc in docs})]
  rng = np.random.default_rng(bootstrap_seed)
  draws = np.empty((bootstrap_replicates, len(names)))
  for iteration in range(bootstrap_replicates):
    sampled_seeds = rng.integers(0, len(seeds), len(seeds))
    sampled_docs = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
    selected = np.ix_(sampled_seeds, sampled_docs)
    draws[iteration] = numerators[selected].sum(axis=(0, 1)) / counts[selected].sum()
  lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
  return {
    'documents': len(docs), 'training_seeds': len(seeds), 'paired_observations': len(groups),
    'active_tokens_per_arm': int(counts.sum()),
    'metrics_nats_per_masked_token': {
      name: {'estimate': float(point[index]), 'ci95': [float(lower[index]), float(upper[index])]}
      for index, name in enumerate(names)},
  }


def summarize_test(test_records: list[dict], selection: dict, *,
                   bootstrap_replicates: int = 5000, bootstrap_seed: int = 1701) -> dict:
  """Report paired likelihood contrasts; test data cannot change selection."""
  if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
    raise ValueError('bootstrap_replicates must be positive')
  groups = _paired_test_rows(test_records, selection)
  return {
    'artifact': 'fresh_pair_heldout_statistics', 'schema_version': 1,
    'selection_sha256': selection['selection_sha256'], 'selected_checkpoints': selection['selected'],
    'test_records_sha256': canonical_sha256(sorted(test_records, key=lambda row: (
      row['training_seed'], _context(row), row['arm']))),
    'scope': 'held-out conditional likelihood; does not establish generation quality',
    'primary_contrast': 'directional_vs_unary',
    'dependence_check': 'directional_dependence_gain compares joint with its own marginals',
    'positive_gain_is_better': True, 'nll_lower_is_better': True,
    'bootstrap': {'replicates': bootstrap_replicates, 'seed': bootstrap_seed,
                  'unit': 'crossed training seeds and source documents; documents stratified by dataset',
                  'masks_kept_with_document': True,
                  'intervals': '95% percentile intervals; secondary contrasts unadjusted',
                  'training_seed_uncertainty_estimated': len(selection['training_seeds']) > 1},
    'overall': _summarize(groups, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed),
    'by_mask_rate': {
      str(rate): _summarize({key: value for key, value in groups.items() if key[1][2] == rate},
                           bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed)
      for rate in selection['mask_rates']},
  }


def _read_records(path):
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest='command', required=True)
  for command in ('select', 'report'):
    subparser = subparsers.add_parser(command)
    subparser.add_argument('--records', type=Path, required=True)
    subparser.add_argument('--output', type=Path, required=True)
    if command == 'select':
      subparser.add_argument('--excluded-document-ids', type=Path, required=True,
                             help='JSON array of training and all previously examined debug document IDs')
    else:
      subparser.add_argument('--selection', type=Path, required=True)
      subparser.add_argument('--bootstrap-replicates', type=int, default=5000)
      subparser.add_argument('--bootstrap-seed', type=int, default=1701)
  args = parser.parse_args(argv)
  if args.output.exists():
    raise FileExistsError(args.output)
  records = _read_records(args.records)
  if args.command == 'select':
    excluded = json.loads(args.excluded_document_ids.read_text())
    if not isinstance(excluded, list):
      raise ValueError('excluded-document-ids must contain a JSON array')
    result = select_checkpoints(records, excluded_test_document_ids=excluded)
  else:
    result = summarize_test(records, json.loads(args.selection.read_text()),
                            bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('x') as handle:
    json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
    handle.write('\n')
  print(json.dumps({'artifact': result['artifact'], 'output': str(args.output)}))


if __name__ == '__main__':
  main()
