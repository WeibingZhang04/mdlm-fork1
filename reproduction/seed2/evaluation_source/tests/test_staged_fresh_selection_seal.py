"""A complete development grid must precede immutable checkpoint selection."""

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.fresh_pair_statistics import ARMS, canonical_sha256
from scripts.evaluate_staged_fresh_selected import load_selection
from scripts.seal_staged_fresh_selection import REPO_ROOT, SOURCE_FILES, file_sha256, seal
from tests.test_fresh_pair_statistics import observation, sha


class StagedFreshSelectionSealTest(unittest.TestCase):

  @staticmethod
  def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

  def fixture(self, root, *, seed=3, final_step=3, interval=2):
    run = root / f'run-{seed}'
    (run / 'checkpoints').mkdir(parents=True)
    manifest = {
      'artifact': 'fresh_staged_openwebtext_splits', 'schema_version': 1,
      'length': 8, 'counts': {'train': 2, 'dev': 2, 'test': 2},
      'splits': {split: {'source_document_sha256': [sha((split, i)) for i in range(2)]}
                 for split in ('train', 'dev', 'test')},
      'excluded_old_debug_document_sha256': [sha('old-a'), sha('old-b')],
      'file_sha256': {f'{split}.jsonl': sha(split) for split in ('train', 'dev', 'test')},
    }
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    manifest_path = root / 'data-manifest.json'
    self.write_json(manifest_path, manifest)
    rates = [0.25, 0.5]
    identity = {
      'dataset': 'corpus', 'eval_every': interval,
      'source_sha256': {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES},
      **{f'{split}_source': {'examples': 2, 'length': 8, 'file_sha256': sha(split),
                            'document_ids_sha256': canonical_sha256(manifest['splits'][split]['source_document_sha256'])}
         for split in ('train', 'dev')},
    }
    protocol = {
      'protocol': 'paired_fresh_corruption_frozen_backbone_v1', 'schema_version': 1,
      'test_data_access': False, 'document_disjoint': True, 'step_zero_included': True,
      'identity': identity, 'identity_sha256': canonical_sha256(identity),
      'training_config': {'seed': seed, 'mask_rates': rates, 'backbone_batch_size': 1, 'batch_size': 2},
      'arguments': {'steps': final_step, 'eval_every': interval, 'seed': seed, 'mask_rates': rates,
                    'train_examples': 2, 'dev_examples': 2},
    }
    self.write_json(run / 'protocol.json', protocol)
    rows, evaluations = [], []
    for step in sorted({0, final_step, *range(interval, final_step + 1, interval)}):
      checkpoint = run / 'checkpoints' / f'step-{step:06d}.pt'
      # The sealer hashes checkpoint bytes but must not deserialize them.
      checkpoint.write_bytes(f'non-pickle fixture: seed={seed}, step={step}'.encode())
      checkpoint_hash = file_sha256(checkpoint)
      summary = {'step': step, 'checkpoint_sha256': checkpoint_hash, 'arms': {}}
      for arm in ARMS:
        group = []
        for index, document in enumerate(manifest['splits']['dev']['source_document_sha256']):
          for rate_index, rate in enumerate(rates):
            gain = (0.6 if step == interval else 0.4) if arm == 'directional' else 0.1 * step / (final_step / 3)
            row = observation(arm, seed, step, document, rate, gain=gain)
            row.update(checkpoint_sha256=checkpoint_hash, example_index=index,
                       corruption_seed=seed + 900000 + rate_index)
            rows.append(row)
            group.append(row)
        count = sum(row['active_token_count'] for row in group)
        summary['arms'][arm] = {'active_token_count': count,
                                'joint_nll_per_masked_token': -sum(row['joint_log_probability'] for row in group) / count}
      evaluations.append(summary)
    (run / 'dev-records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    best = {}
    for arm in ARMS:
      choice = min(evaluations, key=lambda row: (row['arms'][arm]['joint_nll_per_masked_token'], row['step']))
      best[arm] = {'step': choice['step'], 'checkpoint_sha256': choice['checkpoint_sha256'],
                   'dev_joint_nll_per_masked_token': choice['arms'][arm]['joint_nll_per_masked_token']}
    results = {'completed': True, 'completed_steps': final_step, 'target_steps': final_step,
               'protocol_sha256': file_sha256(run / 'protocol.json'),
               'training_mask_rate_histogram': {'0.25': final_step // 2, '0.5': final_step - final_step // 2},
               'evaluations': evaluations, 'best_dev_checkpoints': best}
    self.write_json(run / 'results.json', results)
    self.rehash(run)
    return run, manifest_path, manifest

  def rehash(self, run):
    paths = [run / name for name in ('protocol.json', 'results.json', 'dev-records.jsonl')]
    paths.extend(sorted((run / 'checkpoints').glob('*.pt')))
    self.write_json(run / 'hashes.json', {str(path.relative_to(run)): file_sha256(path) for path in paths})

  def call_seal(self, root, run, manifest):
    return seal([run], manifest, file_sha256(manifest), root / 'selection')

  def test_complete_run_seals_compatible_selection_and_full_exclusions(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, manifest_path, manifest = self.fixture(root)
      selection, sidecar = self.call_seal(root, run, manifest_path)
      self.assertEqual([row['checkpoint_step'] for row in selection['selected']], [3, 3, 2])
      excluded = set(manifest['excluded_old_debug_document_sha256'])
      for split in ('train', 'dev'):
        excluded.update(manifest['splits'][split]['source_document_sha256'])
      self.assertEqual(set(selection['excluded_test_document_ids']), excluded)
      self.assertFalse(sidecar['test_tokens_read'])
      self.assertEqual(sidecar['runs'][0]['expected_checkpoint_steps'], [0, 2, 3])
      path = root / 'selection' / 'selection.json'
      self.assertEqual(load_selection(path, sidecar['selection_file_sha256']), selection)
      self.assertEqual(sidecar['manifest_sha256'], canonical_sha256(
        {key: value for key, value in sidecar.items() if key != 'manifest_sha256'}))
      with self.assertRaises(FileExistsError):
        self.call_seal(root, run, manifest_path)

  def test_sidecar_digest_survives_json_roundtrip_with_step_1000(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, manifest_path, _ = self.fixture(root, final_step=1000, interval=100)
      selection, sidecar = self.call_seal(root, run, manifest_path)
      loaded = json.loads((root / 'selection' / 'selection-manifest.json').read_text())
      expected = {str(step) for step in range(0, 1001, 100)}
      self.assertEqual(set(sidecar['runs'][0]['checkpoint_hashes']), expected)
      self.assertEqual(sidecar, loaded)
      self.assertEqual(loaded['manifest_sha256'], canonical_sha256(
        {key: value for key, value in loaded.items() if key != 'manifest_sha256'}))
      self.assertEqual(load_selection(root / 'selection' / 'selection.json',
                                      loaded['selection_file_sha256']), selection)

  def test_entire_missing_step_document_or_rate_fails_even_with_updated_ledger(self):
    filters = [lambda row: row['checkpoint_step'] != 0,
               lambda row: row['example_index'] != 1,
               lambda row: row['mask_rate'] != 0.5]
    for keep in filters:
      with self.subTest(keep=keep), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run, manifest_path, _ = self.fixture(root)
        path = run / 'dev-records.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows if keep(row)))
        self.rehash(run)
        with self.assertRaisesRegex(ValueError, 'records are incomplete'):
          self.call_seal(root, run, manifest_path)
        self.assertFalse((root / 'selection').exists())

  def test_incomplete_run_and_unfinished_final_exports_fail(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, manifest_path, _ = self.fixture(root)
      result_path = run / 'results.json'
      results = json.loads(result_path.read_text())
      results['completed'] = False
      self.write_json(result_path, results)
      self.rehash(run)
      with self.assertRaisesRegex(ValueError, 'run is incomplete'):
        self.call_seal(root, run, manifest_path)
      (run / 'hashes.json').unlink()
      with self.assertRaises(FileNotFoundError):
        self.call_seal(root, run, manifest_path)

  def test_stale_hashes_truncated_lines_and_duplicate_records_fail(self):
    for mutation, pattern, rehash in (
      (lambda payload: payload[:-1], 'SHA256 mismatch', False),
      (lambda payload: payload[:-1], 'truncated', True),
      (lambda payload: payload + payload.splitlines(keepends=True)[0], 'duplicate', True),
    ):
      with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run, manifest_path, _ = self.fixture(root)
        path = run / 'dev-records.jsonl'
        path.write_bytes(mutation(path.read_bytes()))
        if rehash:
          self.rehash(run)
        with self.assertRaisesRegex(ValueError, pattern):
          self.call_seal(root, run, manifest_path)

  def test_selected_checkpoint_changed_after_completion_fails(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, manifest_path, _ = self.fixture(root)
      (run / 'checkpoints' / 'step-000003.pt').write_bytes(b'different state')
      with self.assertRaisesRegex(ValueError, 'selected checkpoint bytes'):
        self.call_seal(root, run, manifest_path)

  def test_protocol_source_hash_is_checked_even_when_internal_digests_are_consistent(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, manifest_path, _ = self.fixture(root)
      path = run / 'protocol.json'
      protocol = json.loads(path.read_text())
      protocol['identity']['source_sha256']['models/dit.py'] = sha('wrong implementation')
      protocol['identity_sha256'] = canonical_sha256(protocol['identity'])
      self.write_json(path, protocol)
      result_path = run / 'results.json'
      results = json.loads(result_path.read_text())
      results['protocol_sha256'] = file_sha256(path)
      self.write_json(result_path, results)
      self.rehash(run)
      with self.assertRaisesRegex(ValueError, 'implementation source changed'):
        self.call_seal(root, run, manifest_path)

  def test_inconsistent_summary_source_and_seed_metadata_fail(self):
    mutations = [
      (lambda results: results['evaluations'][1]['arms']['unary'].update(joint_nll_per_masked_token=0.2), 'NLL summary'),
      (lambda results: results['best_dev_checkpoints']['unary'].update(step=0), 'recomputed selection'),
      (lambda results: results['training_mask_rate_histogram'].update({'0.25': 0}), 'every planned step'),
      (lambda results: results['evaluations'].pop(), 'omit, duplicate or add'),
    ]
    for mutation, pattern in mutations:
      with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run, manifest_path, _ = self.fixture(root)
        path = run / 'results.json'
        results = json.loads(path.read_text())
        mutation(results)
        self.write_json(path, results)
        self.rehash(run)
        with self.assertRaisesRegex(ValueError, pattern):
          self.call_seal(root, run, manifest_path)

  def test_multiple_independent_seeds_share_one_seal(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      first, manifest, _ = self.fixture(root, seed=3)
      second, _, _ = self.fixture(root, seed=4)
      selection, sidecar = seal([first, second], manifest, file_sha256(manifest), root / 'selection')
      self.assertEqual(selection['training_seeds'], [3, 4])
      self.assertEqual(len(selection['selected']), 6)
      self.assertEqual(len(sidecar['runs']), 2)

  def test_manifest_bytes_are_authenticated_without_reading_test_file(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, manifest, _ = self.fixture(root)
      self.assertFalse((root / 'test.jsonl').exists())
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        seal([run], manifest, '0' * 64, root / 'selection')
      self.call_seal(root, run, manifest)


if __name__ == '__main__':
  unittest.main()
