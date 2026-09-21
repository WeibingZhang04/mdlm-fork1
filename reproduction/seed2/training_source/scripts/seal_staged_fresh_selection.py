#!/usr/bin/env python3
"""Seal development-only choices after complete paired fresh-training runs.

Every planned checkpoint must contain every arm/document/mask observation.
The final hash ledger is required: results.completed is written before final
exports finish and alone does not prove that a run finished. No test token
file or model checkpoint is deserialized by this command.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.fresh_pair_statistics import ARMS, canonical_sha256, select_checkpoints
from scripts.run_staged_fresh_training import SOURCE_FILES


def file_sha256(path):
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _sha(value):
  return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _read_json(path, expected=None):
  payload = path.read_bytes()
  digest = hashlib.sha256(payload).hexdigest()
  if expected is not None and (not _sha(expected) or digest != expected):
    raise ValueError(f'{path.name}: SHA256 mismatch')
  return json.loads(payload), digest


def read_data_manifest(path, expected):
  manifest, digest = _read_json(path, expected)
  if manifest.get('artifact') != 'fresh_staged_openwebtext_splits' or manifest.get('schema_version') != 1:
    raise ValueError('expected the prepared fresh source-document split manifest')
  body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
  if canonical_sha256(body) != manifest.get('manifest_sha256'):
    raise ValueError('data manifest canonical digest mismatch')
  if type(manifest['length']) is not int or manifest['length'] < 2:
    raise ValueError('invalid prepared sequence length')
  groups = {}
  for split in ('train', 'dev', 'test'):
    group = manifest['splits'][split]['source_document_sha256']
    if (not group or len(group) != manifest['counts'][split]
        or len(set(group)) != len(group) or any(not _sha(value) for value in group)):
      raise ValueError('invalid prepared document identities')
    groups[split] = set(group)
  old = manifest['excluded_old_debug_document_sha256']
  if not old or len(set(old)) != len(old) or any(not _sha(value) for value in old):
    raise ValueError('invalid old-debug exclusion list')
  for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
    if groups[left] & groups[right]:
      raise ValueError('prepared source documents overlap across splits')
  if any(group & set(old) for group in groups.values()):
    raise ValueError('prepared split reuses an old-debug document')
  exclusions = sorted(groups['train'] | groups['dev'] | set(old))
  return manifest, digest, exclusions


def _validate_run(run_dir, manifest):
  # hashes.json is emitted only after the final development evaluation and
  # all final adapter exports. Authenticate every lightweight selection input
  # against that final ledger, which the sidecar itself then binds by bytes.
  ledger, ledger_sha = _read_json(run_dir / 'hashes.json')
  required = ('protocol.json', 'results.json', 'dev-records.jsonl')
  if any(not _sha(ledger.get(name)) for name in required):
    raise ValueError('final run hash ledger lacks required selection inputs')
  protocol, protocol_sha = _read_json(run_dir / 'protocol.json', ledger['protocol.json'])
  results, result_sha = _read_json(run_dir / 'results.json', ledger['results.json'])
  payload = (run_dir / 'dev-records.jsonl').read_bytes()
  if hashlib.sha256(payload).hexdigest() != ledger['dev-records.jsonl']:
    raise ValueError('dev-records.jsonl: SHA256 mismatch')
  if not payload.endswith(b'\n'):
    raise ValueError('development JSONL is truncated or lacks its final record terminator')
  records = [json.loads(line) for line in payload.splitlines() if line.strip()]
  if (protocol.get('protocol') != 'paired_fresh_corruption_frozen_backbone_v1'
      or protocol.get('schema_version') != 1
      or protocol.get('test_data_access') is not False
      or protocol.get('document_disjoint') is not True
      or protocol.get('step_zero_included') is not True):
    raise ValueError('run protocol is not a development-only paired fresh experiment')
  identity, config, arguments = protocol['identity'], protocol['training_config'], protocol['arguments']
  if canonical_sha256(identity) != protocol['identity_sha256']:
    raise ValueError('run identity canonical digest mismatch')
  source = identity['source_sha256']
  if set(source) != set(SOURCE_FILES):
    raise ValueError('run implementation source manifest is incomplete')
  for name in SOURCE_FILES:
    if file_sha256(REPO_ROOT / name) != source[name]:
      raise ValueError(f'run implementation source changed: {name}')
  final_step, interval, seed = arguments['steps'], arguments['eval_every'], config['seed']
  if any(type(value) is not int or value < 1 for value in (final_step, interval)):
    raise ValueError('invalid planned final step/evaluation interval')
  if type(seed) is not int or seed < 0 or seed != arguments['seed']:
    raise ValueError('inconsistent training seed')
  if identity['eval_every'] != interval:
    raise ValueError('evaluation interval differs from the training identity')
  if (results.get('completed') is not True or results.get('completed_steps') != final_step
      or results.get('target_steps') != final_step or results.get('protocol_sha256') != protocol_sha):
    raise ValueError('run is incomplete or final results do not match its protocol')
  rates = config['mask_rates']
  if (not rates or len(set(rates)) != len(rates)
      or any(type(rate) not in (int, float) or not 0 < rate < 1 for rate in rates)
      or rates != arguments['mask_rates'] or config['backbone_batch_size'] != 1):
    raise ValueError('invalid rate grid or non-serial frozen-backbone protocol')
  histogram = results['training_mask_rate_histogram']
  if (set(histogram) != {str(rate) for rate in rates}
      or any(type(count) is not int or count < 0 for count in histogram.values())
      or sum(histogram.values()) != final_step):
    raise ValueError('training mask-rate histogram does not cover every planned step')
  for split in ('train', 'dev'):
    selected = identity[f'{split}_source']
    count = selected['examples']
    if type(count) is not int or count < 1:
      raise ValueError('invalid training/development example count')
    documents = manifest['splits'][split]['source_document_sha256'][:count]
    if (len(documents) != count or selected['length'] != manifest['length']
        or selected['file_sha256'] != manifest['file_sha256'][f'{split}.jsonl']
        or selected['document_ids_sha256'] != canonical_sha256(documents)
        or arguments[f'{split}_examples'] != count):
      raise ValueError(f'{split} source identity differs from prepared data')
  dev_documents = manifest['splits']['dev']['source_document_sha256'][:identity['dev_source']['examples']]
  expected_steps = sorted({0, final_step, *range(interval, final_step + 1, interval)})
  expected_keys = {(step, arm, document, rate) for step in expected_steps
                   for arm in ARMS for document in dev_documents for rate in rates}
  observed, groups, checkpoint_hashes = set(), defaultdict(list), {}
  document_windows = {}
  for row in records:
    key = row['checkpoint_step'], row['arm'], row['document_id'], row['mask_rate']
    if key in observed:
      raise ValueError('duplicate development observation')
    observed.add(key)
    rate = row['mask_rate']
    if (key not in expected_keys or row['training_seed'] != seed
        or row['dataset'] != identity['dataset'] or row['split'] != 'dev'):
      raise ValueError('development record is outside the planned grid/data identity')
    if (row['example_index'] != dev_documents.index(row['document_id'])
        or row['corruption_seed'] != seed + 900000 + rates.index(rate)
        or row['active_token_count'] != min(manifest['length'], max(2, round(manifest['length'] * rate)))):
      raise ValueError('development mask count/seed/document order differs from protocol')
    if document_windows.setdefault(row['document_id'], row['clean_token_sha256']) != row['clean_token_sha256']:
      raise ValueError('development document window changes across rates/checkpoints')
    step = row['checkpoint_step']
    if checkpoint_hashes.setdefault(step, row['checkpoint_sha256']) != row['checkpoint_sha256']:
      raise ValueError('arms or observations at one step name different checkpoints')
    groups[(step, row['arm'])].append(row)
  if observed != expected_keys:
    raise ValueError('development records are incomplete: expected every planned step/arm/document/rate')
  evaluations = results['evaluations']
  if sorted(row['step'] for row in evaluations) != expected_steps:
    raise ValueError('final evaluation summaries omit, duplicate or add planned checkpoints')
  for evaluation in evaluations:
    step = evaluation['step']
    digest = checkpoint_hashes[step]
    filename = f'checkpoints/step-{step:06d}.pt'
    if evaluation['checkpoint_sha256'] != digest or ledger.get(filename) != digest:
      raise ValueError('checkpoint hashes differ across records/results/final ledger')
    if set(evaluation['arms']) != set(ARMS):
      raise ValueError('final development summary omits an arm')
    for arm in ARMS:
      rows = groups[(step, arm)]
      count = sum(row['active_token_count'] for row in rows)
      measured = -sum(row['joint_log_probability'] for row in rows) / count
      saved = evaluation['arms'][arm]
      if (saved['active_token_count'] != count
          or not math.isclose(measured, saved['joint_nll_per_masked_token'], rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError('final NLL summary differs from complete development records')
  # The shared reporter also validates normalized probabilities, exact pairing,
  # fixed mask/target hashes and the joint = marginal + dependence identity.
  select_checkpoints(records)
  return records, {
    'run_dir': str(run_dir.resolve()), 'training_seed': seed, 'completed_steps': final_step,
    'expected_checkpoint_steps': expected_steps, 'dev_observations': len(records),
    'dev_documents': dev_documents, 'mask_rates': rates,
    'protocol_file_sha256': protocol_sha, 'results_file_sha256': result_sha,
    'dev_records_file_sha256': ledger['dev-records.jsonl'], 'hash_ledger_file_sha256': ledger_sha,
    'identity_sha256': protocol['identity_sha256'],
    # JSON object keys are strings. Canonicalize before hashing so numeric
    # step ordering cannot change to lexicographic ordering after a roundtrip.
    'checkpoint_hashes': {str(step): digest for step, digest in checkpoint_hashes.items()},
    'best_dev_checkpoints': results['best_dev_checkpoints'],
    'training_config': config,
  }


def seal(run_dirs, data_manifest, expected_data_manifest_sha256, output_dir):
  if output_dir.exists() or output_dir.is_symlink():
    raise FileExistsError(output_dir)
  manifest, manifest_sha, exclusions = read_data_manifest(data_manifest, expected_data_manifest_sha256)
  if not run_dirs or len({path.resolve() for path in run_dirs}) != len(run_dirs):
    raise ValueError('supply distinct completed run directories')
  records, reports = [], []
  for run_dir in run_dirs:
    rows, report = _validate_run(run_dir, manifest)
    records.extend(rows)
    reports.append(report)
  if len({report['training_seed'] for report in reports}) != len(reports):
    raise ValueError('completed runs must have distinct training seeds')
  reference = reports[0]
  for report in reports[1:]:
    for field in ('completed_steps', 'expected_checkpoint_steps', 'dev_documents', 'mask_rates'):
      if report[field] != reference[field]:
        raise ValueError('training seeds must share the planned checkpoint grid and development data')
    without_seed = lambda config: {key: value for key, value in config.items() if key != 'seed'}
    if without_seed(report['training_config']) != without_seed(reference['training_config']):
      raise ValueError('training seeds use different optimization configurations')
  selection = select_checkpoints(records, excluded_test_document_ids=exclusions)
  selected_files = []
  by_seed = {report['training_seed']: report for report in reports}
  for chosen in selection['selected']:
    report = by_seed[chosen['training_seed']]
    best = report['best_dev_checkpoints'][chosen['arm']]
    if (best['step'] != chosen['checkpoint_step']
        or best['checkpoint_sha256'] != chosen['checkpoint_sha256']
        or not math.isclose(best['dev_joint_nll_per_masked_token'], chosen['dev_nll_per_masked_token'],
                            rel_tol=1e-10, abs_tol=1e-10)):
      raise ValueError('recomputed selection differs from final run best-dev summary')
    path = Path(report['run_dir']) / 'checkpoints' / f'step-{chosen["checkpoint_step"]:06d}.pt'
    if file_sha256(path) != chosen['checkpoint_sha256']:
      raise ValueError('selected checkpoint bytes differ from development records')
    selected_files.append({'training_seed': chosen['training_seed'], 'arm': chosen['arm'],
                           'path': str(path), 'sha256': chosen['checkpoint_sha256']})
  sidecar = {
    'artifact': 'fresh_pair_completed_run_selection_manifest', 'schema_version': 1,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'selection_sha256': selection['selection_sha256'],
    'data_manifest_path': str(data_manifest.resolve()), 'data_manifest_file_sha256': manifest_sha,
    'data_manifest_sha256': manifest['manifest_sha256'],
    'excluded_document_count': len(exclusions),
    'exclusion_counts': {'prepared_train': manifest['counts']['train'],
                         'prepared_dev': manifest['counts']['dev'],
                         'old_debug': len(manifest['excluded_old_debug_document_sha256'])},
    'test_tokens_read': False, 'runs': reports, 'selected_checkpoint_files': selected_files,
    'source_sha256': {name: file_sha256(REPO_ROOT / name) for name in (
      'scripts/seal_staged_fresh_selection.py', 'evaluation/fresh_pair_statistics.py')},
  }
  # Validate completely before exclusive creation; never replace a prior seal.
  output_dir.mkdir(parents=True, exist_ok=False)
  with (output_dir / 'selection.json').open('x') as handle:
    json.dump(selection, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')
  sidecar['selection_file_sha256'] = file_sha256(output_dir / 'selection.json')
  sidecar['manifest_sha256'] = canonical_sha256(sidecar)
  with (output_dir / 'selection-manifest.json').open('x') as handle:
    json.dump(sidecar, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')
  return selection, sidecar


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, action='append', required=True,
                      help='repeat for independent training seeds under the same protocol')
  parser.add_argument('--data-manifest', type=Path, required=True)
  parser.add_argument('--expected-data-manifest-sha256', required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args(argv)
  selection, sidecar = seal(args.run_dir, args.data_manifest, args.expected_data_manifest_sha256, args.output_dir)
  print(json.dumps({'event': 'development_selection_sealed', 'output_dir': str(args.output_dir),
                    'selection_sha256': selection['selection_sha256'],
                    'selection_file_sha256': sidecar['selection_file_sha256'],
                    'test_tokens_read': False}), flush=True)


if __name__ == '__main__':
  main()
