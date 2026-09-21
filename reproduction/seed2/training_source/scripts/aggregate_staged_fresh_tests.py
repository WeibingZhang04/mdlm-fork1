#!/usr/bin/env python3
"""Authenticate and aggregate independently sealed test shards without rescoring.

Each original selection manifest and test result has an externally supplied
byte digest. Original records remain bound to their original seals. New rows
change only that binding and add explicit source-file/row lineage; all scores,
checkpoints, documents and corruptions are preserved. No checkpoints or test
tokens are loaded, and no model is constructed.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.fresh_pair_statistics import ARMS, canonical_sha256, select_checkpoints, summarize_test
from scripts.evaluate_staged_fresh_selected import (
  TEST_SEED_OFFSET, _is_sha, load_selection, load_split_manifest, read_authenticated,
)
from scripts.run_staged_real_overfit import file_sha256
from scripts.seal_staged_fresh_selection import _validate_run


LINEAGE_FIELDS = (
  'source_selection_sha256', 'source_test_records_file_sha256',
  'source_record_sha256', 'source_record_bytes_sha256', 'source_record_line',
)
# Issued seed-1 v1 sealer, whose integer checkpoint keys require the narrow
# canonical-digest compatibility path above. Its complete output bytes still
# need the externally supplied manifest hash and reproducible dev selection.
LEGACY_SEALER_SHA256 = '3229d66ca78b18c315b4d9acf92cb3de3df46a7411d8b3fa91e513288a8c8aed'


def _json(path, digest):
  return json.loads(read_authenticated(Path(path), digest))


def _jsonl(path, digest):
  payload = read_authenticated(Path(path), digest)
  if not payload.endswith(b'\n'):
    raise ValueError('JSONL is truncated or lacks its final record terminator')
  rows, lineage = [], []
  for number, line in enumerate(payload.splitlines(keepends=True), 1):
    if not line.strip():
      raise ValueError('authenticated JSONL contains an empty record')
    row = json.loads(line)
    canonical_sha256(row)  # Reject NaN and infinity, including extra fields.
    rows.append(row)
    lineage.append((number, hashlib.sha256(line).hexdigest()))
  return rows, lineage


def _same(actual, expected, message):
  if canonical_sha256(actual) != canonical_sha256(expected):
    raise ValueError(message)


def _manifest_digest_mode(sidecar):
  """Verify JSON-key canonicalization, including already issued v1 seals.

  The v1 sealer hashes checkpoint step dictionaries with integer keys before
  JSON converts their keys to strings. Numeric 1000 sorts after 900, whereas
  the serialized string sorts before 200. Only that known field is restored;
  the caller has already authenticated the complete original file by bytes.
  """
  body = {key: value for key, value in sidecar.items() if key != 'manifest_sha256'}
  if canonical_sha256(body) == sidecar.get('manifest_sha256'):
    return 'serialized_json_keys'
  restored = copy.deepcopy(body)
  for report in restored.get('runs', []):
    checkpoints = report.get('checkpoint_hashes', {})
    if not checkpoints or any(not isinstance(key, str) or not key.isdecimal()
                              or str(int(key)) != key for key in checkpoints):
      raise ValueError('selection manifest canonical digest mismatch')
    report['checkpoint_hashes'] = {int(key): value for key, value in checkpoints.items()}
  if canonical_sha256(restored) != sidecar.get('manifest_sha256'):
    raise ValueError('selection manifest canonical digest mismatch')
  return 'legacy_v1_integer_checkpoint_step_keys'


def _load_seal(path, digest, data_manifest, data_digest, run_dirs):
  path = Path(path)
  sidecar = _json(path, digest)
  if (sidecar.get('artifact') != 'fresh_pair_completed_run_selection_manifest'
      or sidecar.get('schema_version') != 1 or sidecar.get('test_tokens_read') is not False):
    raise ValueError('expected a completed development-only selection manifest')
  digest_mode = _manifest_digest_mode(sidecar)
  selection = load_selection(path.parent / 'selection.json', sidecar['selection_file_sha256'])
  if selection['selection_sha256'] != sidecar['selection_sha256']:
    raise ValueError('selection manifest names another selection')
  if sidecar['data_manifest_file_sha256'] != data_digest:
    raise ValueError('selection uses a different prepared data manifest')
  manifest = load_split_manifest(data_manifest, data_digest, selection)
  if sidecar['data_manifest_sha256'] != manifest['manifest_sha256']:
    raise ValueError('selection names another canonical data manifest')
  sealer_source, statistics_source = 'scripts/seal_staged_fresh_selection.py', 'evaluation/fresh_pair_statistics.py'
  source = sidecar['source_sha256']
  if (set(source) != {sealer_source, statistics_source}
      or source[statistics_source] != file_sha256(REPO_ROOT / statistics_source)
      or source[sealer_source] not in {LEGACY_SEALER_SHA256, file_sha256(REPO_ROOT / sealer_source)}):
    raise ValueError('selection implementation differs from the authenticated implementation')
  if (digest_mode == 'legacy_v1_integer_checkpoint_step_keys'
      and source[sealer_source] != LEGACY_SEALER_SHA256):
    raise ValueError('legacy checkpoint-key digest requires the issued legacy sealer')
  protocols, reports, protocol_paths, records = {}, {}, {}, []
  for report in sidecar['runs']:
    seed = report['training_seed']
    if seed in protocols:
      raise ValueError('selection manifest repeats a training seed')
    run_dir = Path(run_dirs.get(seed, report['run_dir']))
    # Recheck the entire planned step/document/rate grid and final hash ledger,
    # not just the selected rows. A uniformly omitted group can otherwise
    # reproduce a selection while silently changing the experiment.
    _, verified_report = _validate_run(run_dir, manifest)
    _same({key: value for key, value in verified_report.items() if key != 'run_dir'},
          {key: value for key, value in report.items() if key != 'run_dir'},
          'sealed run differs from the authenticated complete development grid')
    protocol = _json(run_dir / 'protocol.json', report['protocol_file_sha256'])
    if (protocol.get('protocol') != 'paired_fresh_corruption_frozen_backbone_v1'
        or protocol.get('schema_version') != 1 or protocol.get('test_data_access') is not False
        or protocol.get('document_disjoint') is not True or protocol.get('step_zero_included') is not True):
      raise ValueError('training protocol is not the sealed development-only experiment')
    identity, config = protocol['identity'], protocol['training_config']
    if (config['seed'] != seed or identity['head_config']['seed'] != seed
        or identity['eval_every'] != protocol['arguments']['eval_every']):
      raise ValueError('training seed or evaluation schedule differs from its protocol')
    _same(config, report['training_config'], 'sealed training configuration differs from protocol')
    _same(canonical_sha256(identity), report['identity_sha256'], 'sealed training identity mismatch')
    _same(protocol['identity_sha256'], report['identity_sha256'], 'protocol identity digest mismatch')
    dev_rows, _ = _jsonl(run_dir / 'dev-records.jsonl', report['dev_records_file_sha256'])
    if len(dev_rows) != report['dev_observations'] or any(row['training_seed'] != seed for row in dev_rows):
      raise ValueError('sealed development record count or seed differs')
    records.extend(dev_rows)
    protocols[seed], reports[seed] = protocol, report
    protocol_paths[seed] = str(run_dir.resolve())
  if set(protocols) != set(selection['training_seeds']):
    raise ValueError('selection manifest omits or adds a training seed')
  recomputed = select_checkpoints(records, excluded_test_document_ids=selection['excluded_test_document_ids'])
  _same(recomputed, selection, 'selection is not reproducible from its authenticated development records')
  return {'selection': selection, 'sidecar': sidecar, 'protocols': protocols, 'reports': reports,
          'path': str(path.resolve()), 'file_sha256': digest, 'data_manifest': manifest,
          'manifest_digest_mode': digest_mode, 'protocol_paths': protocol_paths}


def _seed_invariant(protocol):
  # Different seeds deliberately have different dev masks/cache hashes and
  # head initializations. Every other identity field must remain unchanged.
  identity = dict(protocol['identity'])
  if not identity.pop('dev_cache_sha256', None):
    raise ValueError('training identity lacks development cache hashes')
  head_config = dict(identity['head_config'])
  head_config.pop('seed')
  identity['head_config'] = head_config
  config = dict(protocol['training_config'])
  config.pop('seed')
  return {'identity': identity, 'training_config': config,
          'target_steps': protocol['arguments']['steps']}


def _validate_test_grid(rows, selection, manifest, protocols):
  for protocol in protocols.values():
    if sorted(selection['mask_rates']) != sorted(protocol['training_config']['mask_rates']):
      raise ValueError('selection mask-rate grid differs from the training protocol')
  documents = manifest['splits']['test']['source_document_sha256']
  expected = {(seed, arm, document, rate)
              for seed in selection['training_seeds'] for arm in ARMS
              for document in documents for rate in selection['mask_rates']}
  seen, support = set(), {}
  for row in rows:
    seed, arm, document, rate = (row[key] for key in ('training_seed', 'arm', 'document_id', 'mask_rate'))
    key = seed, arm, document, rate
    if key not in expected or key in seen:
      raise ValueError('test grid has an extra, duplicate, or unselected observation')
    seen.add(key)
    identity, config = protocols[seed]['identity'], protocols[seed]['training_config']
    if (row['example_index'] != documents.index(document) or row['dataset'] != identity['dataset']
        or row['corruption_seed'] != seed + TEST_SEED_OFFSET + config['mask_rates'].index(rate)
        or row['active_token_count'] != min(manifest['length'], max(2, round(manifest['length'] * rate)))):
      raise ValueError('test document, corruption, count, or dataset differs from protocol')
    paired = seed, document, rate
    if (not _is_sha(row['candidate_ids_sha256'])
        or support.setdefault(paired, row['candidate_ids_sha256']) != row['candidate_ids_sha256']):
      raise ValueError('paired test arms have different candidate support')
  if seen != expected:
    raise ValueError('test shard is missing a seed, document, arm, or mask-rate observation')


def _load_shard(spec, combined, data_manifest, data_digest, run_dirs):
  seal_path, seal_digest, test_dir, result_digest = spec
  test_dir = Path(test_dir)
  seal = _load_seal(seal_path, seal_digest, data_manifest, data_digest, run_dirs)
  selection = seal['selection']
  results = _json(test_dir / 'results.json', result_digest)
  commitment = _json(test_dir / 'commitment.json', results['commitment_file_sha256'])
  rows, row_lines = _jsonl(test_dir / 'test-records.jsonl', results['test_records_file_sha256'])
  if any(any(key in row for key in LINEAGE_FIELDS) for row in rows):
    raise ValueError('supply original evaluator shards, not previously rebound derivative rows')
  if (commitment.get('artifact') != 'fresh_pair_test_commitment' or commitment.get('schema_version') != 1
      or commitment.get('test_tokens_read') is not False
      or commitment['selection_sha256'] != selection['selection_sha256']
      or commitment['selection_file_sha256'] != seal['sidecar']['selection_file_sha256']
      or commitment['data_manifest_file_sha256'] != data_digest):
    raise ValueError('test commitment is not bound to the original seal and data')
  _same(commitment['selected'], selection['selected'], 'commitment changed the sealed checkpoints')
  manifest = combined['data_manifest']
  if (commitment['test_file_sha256'] != manifest['file_sha256']['test.jsonl']
      or commitment['test_examples'] != manifest['counts']['test']
      or commitment['test_seed_offset'] != TEST_SEED_OFFSET
      or commitment['seed_mapping'] != 'training seed + offset + training rate-list index'
      or commitment['backbone_batch_size'] != 1 or commitment['head_forward_dtype'] != 'float32'
      or commitment['forest_inference_dtype'] != 'float64'
      or commitment['residual_normalization_dtype'] != 'float64'):
    raise ValueError('test commitment changes the full test data or scoring protocol')
  sources = ('scripts/evaluate_staged_fresh_selected.py', 'evaluation/fresh_pair_statistics.py')
  _same(commitment['source_sha256'], {name: file_sha256(REPO_ROOT / name) for name in sources},
        'test scoring implementation differs')
  expected_identities = {str(seed): canonical_sha256({
    'identity': protocol['identity'], 'config': protocol['training_config']})
    for seed, protocol in seal['protocols'].items()}
  _same(commitment['training_identity_sha256_by_seed'], expected_identities,
        'test commitment changes the sealed training identities')
  for seed, protocol in seal['protocols'].items():
    _same(commitment['backbone_provenance'], protocol['identity']['backbone_provenance'],
          'test commitment uses another backbone')
    if seed not in combined['reports']:
      raise ValueError('test shard seed is absent from the combined seal')
    # A new combined seal may be made from authenticated local backups. The
    # location is not training content; both copies have already passed the
    # complete grid and final-ledger checks above. Keep every other field exact.
    _same({key: value for key, value in seal['reports'][seed].items() if key != 'run_dir'},
          {key: value for key, value in combined['reports'][seed].items() if key != 'run_dir'},
          'combined seal changes an original sealed training run')
  source = results['test_source']
  if (source['file_sha256'] != commitment['test_file_sha256']
      or source['examples'] != manifest['counts']['test'] or source['length'] != manifest['length']
      or source['document_ids_sha256'] != canonical_sha256(manifest['splits']['test']['source_document_sha256'])):
    raise ValueError('test results cover a different source window')
  _validate_test_grid(rows, selection, manifest, seal['protocols'])
  recomputed = summarize_test(rows, selection, bootstrap_replicates=commitment['bootstrap_replicates'],
                              bootstrap_seed=commitment['bootstrap_seed'])
  _same({key: results.get(key) for key in recomputed}, recomputed,
        'recomputed original test statistics differ from the authenticated published results')
  if results['max_decomposition_error_nats'] != max(row['decomposition_error_nats'] for row in rows):
    raise ValueError('published decomposition error differs from original records')
  return {'seal': seal, 'rows': rows, 'row_lines': row_lines, 'test_source': source,
          'path': str(test_dir.resolve()), 'results_file_sha256': result_digest,
          'records_file_sha256': results['test_records_file_sha256'],
          'commitment_file_sha256': results['commitment_file_sha256']}


def aggregate(combined_selection_manifest, expected_combined_sha256, data_manifest,
              expected_data_sha256, shards, output_dir, *, run_dirs=None,
              bootstrap_replicates=5000, bootstrap_seed=1701):
  output_dir, run_dirs = Path(output_dir), run_dirs or {}
  if output_dir.exists() or output_dir.is_symlink():
    raise FileExistsError(output_dir)
  if not shards:
    raise ValueError('at least one original test shard is required')
  combined = _load_seal(combined_selection_manifest, expected_combined_sha256,
                        data_manifest, expected_data_sha256, run_dirs)
  selection = combined['selection']
  if not set(run_dirs) <= set(selection['training_seeds']):
    raise ValueError('training run relocation names an unselected seed')
  invariants = [_seed_invariant(protocol) for protocol in combined['protocols'].values()]
  for invariant in invariants[1:]:
    _same(invariant, invariants[0], 'training seeds differ in a seed-invariant identity or configuration')
  loaded, observed_seeds, selected_union = [], set(), []
  for spec in shards:
    shard = _load_shard(spec, combined, data_manifest, expected_data_sha256, run_dirs)
    shard_selection = shard['seal']['selection']
    seeds = set(shard_selection['training_seeds'])
    if observed_seeds & seeds:
      raise ValueError('test shards overlap in training seeds')
    observed_seeds.update(seeds)
    for key in ('arms', 'mask_rates', 'dev_document_ids', 'dev_clean_token_sha256s',
                'excluded_test_document_ids', 'selection_rule'):
      _same(shard_selection[key], selection[key], f'test shards differ in selection field {key}')
    if loaded:
      _same(shard['test_source'], loaded[0]['test_source'], 'test shards use different held-out sources')
    selected_union.extend(shard_selection['selected'])
    loaded.append(shard)
  if observed_seeds != set(selection['training_seeds']):
    raise ValueError('test shards omit a combined selected training seed')
  key = lambda row: (row['training_seed'], row['arm'])
  _same(sorted(selected_union, key=key), sorted(selection['selected'], key=key),
        'combined seal is not the exact union of original per-seed selections')
  derivative = []
  for shard in loaded:
    for row, (line_number, line_digest) in zip(shard['rows'], shard['row_lines']):
      derived = {**row, 'selection_sha256': selection['selection_sha256'],
                 'source_selection_sha256': row['selection_sha256'],
                 'source_test_records_file_sha256': shard['records_file_sha256'],
                 'source_record_sha256': canonical_sha256(row),
                 'source_record_bytes_sha256': line_digest, 'source_record_line': line_number}
      derivative.append(derived)
  _validate_test_grid(derivative, selection, combined['data_manifest'], combined['protocols'])
  summary = summarize_test(derivative, selection, bootstrap_replicates=bootstrap_replicates,
                           bootstrap_seed=bootstrap_seed)
  # All validation, including recomputation of original reports, precedes any
  # output creation. The input seals, results and rows are never rewritten.
  output_dir.mkdir(parents=True, exist_ok=False)
  with (output_dir / 'test-records.jsonl').open('x') as handle:
    for row in derivative:
      handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
  lineage = {
    'artifact': 'fresh_pair_test_shard_aggregation_lineage', 'schema_version': 1,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'combined_selection_manifest_path': str(Path(combined_selection_manifest).resolve()),
    'combined_selection_manifest_file_sha256': expected_combined_sha256,
    'combined_selection_manifest_digest_mode': combined['manifest_digest_mode'],
    'combined_selection_sha256': selection['selection_sha256'],
    'data_manifest_file_sha256': expected_data_sha256,
    'seed_invariant_identity_sha256': canonical_sha256(invariants[0]),
    'training_runs': [{
      'training_seed': seed, 'authenticated_run_dir': combined['protocol_paths'][seed],
      'protocol_file_sha256': report['protocol_file_sha256'],
      'dev_records_file_sha256': report['dev_records_file_sha256'],
      'identity_sha256': report['identity_sha256'],
    } for seed, report in sorted(combined['reports'].items())],
    'training_seeds': sorted(observed_seeds), 'records': len(derivative),
    'backbone_rescored': False, 'test_tokens_read': False, 'source_files_modified': False,
    'transformation': 'copy every original field unchanged except selection_sha256; add explicit source lineage',
    'changed_original_fields': ['selection_sha256'], 'added_fields': list(LINEAGE_FIELDS),
    'original_statistics_recomputed_exactly': True,
    'test_records_file_sha256': file_sha256(output_dir / 'test-records.jsonl'),
    'shards': [{
      'selection_manifest_path': shard['seal']['path'],
      'selection_manifest_file_sha256': shard['seal']['file_sha256'],
      'selection_manifest_digest_mode': shard['seal']['manifest_digest_mode'],
      'selection_file_sha256': shard['seal']['sidecar']['selection_file_sha256'],
      'selection_sha256': shard['seal']['selection']['selection_sha256'],
      'training_seeds': shard['seal']['selection']['training_seeds'],
      'test_dir': shard['path'], 'results_file_sha256': shard['results_file_sha256'],
      'commitment_file_sha256': shard['commitment_file_sha256'],
      'test_records_file_sha256': shard['records_file_sha256'], 'records': len(shard['rows']),
    } for shard in loaded],
    'source_sha256': {name: file_sha256(REPO_ROOT / name) for name in (
      'scripts/aggregate_staged_fresh_tests.py', 'evaluation/fresh_pair_statistics.py')},
  }
  lineage['manifest_sha256'] = canonical_sha256(lineage)
  _write_json(output_dir / 'lineage.json', lineage)
  summary.update(lineage_file_sha256=file_sha256(output_dir / 'lineage.json'),
                 test_records_file_sha256=lineage['test_records_file_sha256'],
                 aggregation_without_rescoring=True, test_source=loaded[0]['test_source'])
  _write_json(output_dir / 'results.json', summary)
  return summary, lineage


def _write_json(path, value):
  with path.open('x') as handle:
    json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
    handle.write('\n')


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--combined-selection-manifest', type=Path, required=True)
  parser.add_argument('--expected-combined-selection-manifest-sha256', required=True)
  parser.add_argument('--data-manifest', type=Path, required=True)
  parser.add_argument('--expected-data-manifest-sha256', required=True)
  parser.add_argument('--shard', nargs=4, action='append', required=True,
                      metavar=('SELECTION_MANIFEST', 'MANIFEST_SHA256', 'TEST_DIR', 'RESULTS_SHA256'))
  parser.add_argument('--run-dir', nargs=2, action='append', default=[], metavar=('SEED', 'DIRECTORY'),
                      help='optional authenticated relocation of a training run directory')
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--bootstrap-replicates', type=int, default=5000)
  parser.add_argument('--bootstrap-seed', type=int, default=1701)
  args = parser.parse_args(argv)
  run_dirs = {int(seed): Path(path) for seed, path in args.run_dir}
  if len(run_dirs) != len(args.run_dir):
    raise ValueError('a training run relocation repeats a seed')
  summary, lineage = aggregate(
    args.combined_selection_manifest, args.expected_combined_selection_manifest_sha256,
    args.data_manifest, args.expected_data_manifest_sha256, args.shard, args.output_dir,
    run_dirs=run_dirs, bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
  print(json.dumps({'event': 'authenticated_test_shards_aggregated',
                    'output_dir': str(args.output_dir), 'training_seeds': lineage['training_seeds'],
                    'records': lineage['records'], 'selection_sha256': summary['selection_sha256']}))


if __name__ == '__main__':
  main()
