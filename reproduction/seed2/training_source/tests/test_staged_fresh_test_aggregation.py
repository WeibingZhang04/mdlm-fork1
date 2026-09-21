"""Authenticated incremental replication without rescoring or rewriting a shard."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.fresh_pair_statistics import ARMS, canonical_sha256, summarize_test
from scripts.aggregate_staged_fresh_tests import (
  LINEAGE_FIELDS, _manifest_digest_mode, _validate_test_grid, aggregate,
)
from scripts.evaluate_staged_fresh_selected import TEST_SEED_OFFSET
from scripts.seal_staged_fresh_selection import REPO_ROOT, file_sha256, seal
from tests.test_fresh_pair_statistics import observation, sha
from tests import test_staged_fresh_selection_seal as seal_test_helpers


def write_json(path, value):
  path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def write_rows(path, rows):
  path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


class StagedFreshTestAggregationTest(unittest.TestCase):

  def test_test_grid_cannot_omit_a_protocol_mask_rate(self):
    with self.assertRaisesRegex(ValueError, 'mask-rate grid differs'):
      _validate_test_grid([], {'mask_rates': [.25, .5]}, {},
                          {1: {'training_config': {'mask_rates': [.25, .5, .75]}}})

  def test_fixed_writer_cannot_use_legacy_integer_key_digest(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = self.fixture(Path(directory))
      sidecar = json.loads(fixture['combined'].read_text())
      del sidecar['manifest_sha256']
      sidecar['runs'][0]['checkpoint_hashes']['1000'] = sha('extra')
      for run in sidecar['runs']:
        run['checkpoint_hashes'] = {int(key): value for key, value in run['checkpoint_hashes'].items()}
      sidecar['manifest_sha256'] = canonical_sha256(sidecar)
      write_json(fixture['combined'], sidecar)
      with self.assertRaisesRegex(ValueError, 'requires the issued legacy sealer'):
        self.call(fixture)

  def test_issued_integer_step_manifest_digest_is_verified_without_modifying_it(self):
    sidecar = {'runs': [{'checkpoint_hashes': {0: sha(0), 100: sha(100), 200: sha(200), 1000: sha(1000)}}]}
    sidecar['manifest_sha256'] = canonical_sha256(sidecar)
    parsed = json.loads(json.dumps(sidecar, sort_keys=True))
    before = copy.deepcopy(parsed)
    self.assertEqual(_manifest_digest_mode(parsed), 'legacy_v1_integer_checkpoint_step_keys')
    self.assertEqual(parsed, before)
    parsed['runs'][0]['checkpoint_hashes']['1000'] = sha('changed')
    with self.assertRaisesRegex(ValueError, 'canonical digest mismatch'):
      _manifest_digest_mode(parsed)
    parsed['manifest_sha256'] = canonical_sha256({
      key: value for key, value in parsed.items() if key != 'manifest_sha256'})
    self.assertEqual(_manifest_digest_mode(parsed), 'serialized_json_keys')

  def fixture(self, root, identity_mutation=None):
    helper = seal_test_helpers.StagedFreshSelectionSealTest()
    runs = []
    for seed in (1, 2, 3):
      run, manifest_path, manifest = helper.fixture(root, seed=seed)
      protocol = json.loads((run / 'protocol.json').read_text())
      identity = protocol['identity']
      identity.update(
        dev_cache_sha256=[sha(('dev-cache', seed, rate)) for rate in (0.25, 0.5)],
        head_config={'seed': seed, 'rank': 16, 'directional_rank': 8, 'unary_rank': 16,
                     'top_k': 64, 'init_std': 0.25, 'component_size_cap': 0},
        backbone_provenance={'release': 'tiny-released-backbone'},
        backbone_runtime_config={'noise': {'type': 'loglinear', 'eps': 0.001}},
        runtime={'torch_version': 'fixture', 'device_type': 'cuda', 'gpu_name': 'L4'})
      if identity_mutation is not None and seed == 2:
        identity_mutation(identity)
      protocol['identity_sha256'] = canonical_sha256(identity)
      write_json(run / 'protocol.json', protocol)
      result = json.loads((run / 'results.json').read_text())
      result['protocol_sha256'] = file_sha256(run / 'protocol.json')
      write_json(run / 'results.json', result)
      helper.rehash(run)
      runs.append(run)
    for split in ('train', 'dev', 'test'):
      manifest['splits'][split]['prefix_token_ids_sha256'] = [
        sha(('prefix', document)) for document in manifest['splits'][split]['source_document_sha256']]
    del manifest['manifest_sha256']
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    write_json(manifest_path, manifest)
    sealed = []
    for name, selected_runs in [('seed1', runs[:1]), ('seed23', runs[1:]), ('combined', runs)]:
      directory = root / f'selection-{name}'
      selection, sidecar = seal(selected_runs, manifest_path, file_sha256(manifest_path), directory)
      sealed.append((selection, sidecar, directory))
    shards = []
    for number, (selection, sidecar, directory) in enumerate(sealed[:2]):
      test_dir = root / f'test-{number}'
      test_dir.mkdir()
      rows = []
      for chosen in selection['selected']:
        seed, arm = chosen['training_seed'], chosen['arm']
        for index, document in enumerate(manifest['splits']['test']['source_document_sha256']):
          for rate_index, rate in enumerate(selection['mask_rates']):
            gain = {'unary': 0.1, 'shared': 0.2, 'directional': 0.3 + 0.01 * seed + 0.05 * index}[arm]
            row = observation(arm, seed, chosen['checkpoint_step'], document, rate, split='test',
                              gain=gain, dependence=0.0 if arm == 'unary' else 0.05)
            row.update(checkpoint_sha256=chosen['checkpoint_sha256'],
                       selection_sha256=selection['selection_sha256'], example_index=index,
                       corruption_seed=seed + TEST_SEED_OFFSET + rate_index,
                       candidate_ids_sha256=sha(('support', seed, document, rate)),
                       decomposition_error_nats=0.0)
            rows.append(row)
      write_rows(test_dir / 'test-records.jsonl', rows)
      protocols = {seed: json.loads((runs[seed - 1] / 'protocol.json').read_text())
                   for seed in selection['training_seeds']}
      commitment = {
        'artifact': 'fresh_pair_test_commitment', 'schema_version': 1, 'test_tokens_read': False,
        'selection_sha256': selection['selection_sha256'],
        'selection_file_sha256': sidecar['selection_file_sha256'],
        'data_manifest_file_sha256': file_sha256(manifest_path),
        'test_file_sha256': manifest['file_sha256']['test.jsonl'], 'test_examples': 2,
        'selected': selection['selected'], 'test_seed_offset': TEST_SEED_OFFSET,
        'seed_mapping': 'training seed + offset + training rate-list index',
        'backbone_batch_size': 1, 'head_forward_dtype': 'float32',
        'forest_inference_dtype': 'float64', 'residual_normalization_dtype': 'float64',
        'backbone_provenance': protocols[selection['training_seeds'][0]]['identity']['backbone_provenance'],
        'source_sha256': {name: file_sha256(REPO_ROOT / name) for name in (
          'scripts/evaluate_staged_fresh_selected.py', 'evaluation/fresh_pair_statistics.py')},
        'training_identity_sha256_by_seed': {str(seed): canonical_sha256({
          'identity': protocol['identity'], 'config': protocol['training_config']})
          for seed, protocol in protocols.items()},
        'bootstrap_replicates': 100, 'bootstrap_seed': 1701,
      }
      write_json(test_dir / 'commitment.json', commitment)
      results = summarize_test(rows, selection, bootstrap_replicates=100)
      results.update(
        commitment_file_sha256=file_sha256(test_dir / 'commitment.json'),
        test_records_file_sha256=file_sha256(test_dir / 'test-records.jsonl'),
        max_decomposition_error_nats=0.0,
        test_source={'file_sha256': manifest['file_sha256']['test.jsonl'], 'examples': 2,
                     'length': manifest['length'], 'selected_token_sha256': sha('all-test-tokens'),
                     'document_ids_sha256': canonical_sha256(manifest['splits']['test']['source_document_sha256'])})
      write_json(test_dir / 'results.json', results)
      manifest_file = directory / 'selection-manifest.json'
      shards.append([manifest_file, file_sha256(manifest_file), test_dir, file_sha256(test_dir / 'results.json')])
    combined = sealed[2][2] / 'selection-manifest.json'
    return {'combined': combined, 'data': manifest_path, 'shards': shards, 'runs': runs,
            'selection': sealed[2][0], 'root': root}

  def call(self, fixture, shards=None, **kwargs):
    return aggregate(fixture['combined'], file_sha256(fixture['combined']),
                     fixture['data'], file_sha256(fixture['data']),
                     fixture['shards'] if shards is None else shards, fixture['root'] / 'aggregate',
                     bootstrap_replicates=100, **kwargs)

  def rehash_result(self, shard, *, recompute=False):
    path = shard[2] / 'results.json'
    result = json.loads(path.read_text())
    result['commitment_file_sha256'] = file_sha256(shard[2] / 'commitment.json')
    result['test_records_file_sha256'] = file_sha256(shard[2] / 'test-records.jsonl')
    if recompute:
      rows = [json.loads(line) for line in (shard[2] / 'test-records.jsonl').read_text().splitlines()]
      selection = json.loads((shard[0].parent / 'selection.json').read_text())
      result.update(summarize_test(rows, selection, bootstrap_replicates=100))
    write_json(path, result)
    shard[3] = file_sha256(path)

  def test_original_shards_unchanged_and_combined_rows_have_verifiable_lineage(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = self.fixture(Path(directory))
      before = {path: path.read_bytes() for path in Path(directory).rglob('*') if path.is_file()}
      summary, lineage = self.call(fixture)
      self.assertEqual(lineage['training_seeds'], [1, 2, 3])
      self.assertEqual(lineage['records'], 36)
      self.assertFalse(lineage['backbone_rescored'])
      self.assertTrue(lineage['original_statistics_recomputed_exactly'])
      self.assertEqual(summary['overall']['training_seeds'], 3)
      self.assertTrue(summary['bootstrap']['training_seed_uncertainty_estimated'])
      self.assertEqual(summary['overall']['documents'], 2)
      for path, payload in before.items():
        self.assertEqual(path.read_bytes(), payload)
      originals = {}
      for shard in fixture['shards']:
        for index, line in enumerate((shard[2] / 'test-records.jsonl').read_bytes().splitlines(keepends=True), 1):
          originals[shard[3], index] = line
      rows = [json.loads(line) for line in (Path(directory) / 'aggregate/test-records.jsonl').read_text().splitlines()]
      by_records_hash = {file_sha256(shard[2] / 'test-records.jsonl'): shard for shard in fixture['shards']}
      for row in rows:
        shard = by_records_hash[row['source_test_records_file_sha256']]
        original_line = originals[shard[3], row['source_record_line']]
        original = json.loads(original_line)
        self.assertEqual(row['source_record_bytes_sha256'], hashlib.sha256(original_line).hexdigest())
        self.assertEqual(row['source_record_sha256'], canonical_sha256(original))
        restored = {key: value for key, value in row.items() if key not in LINEAGE_FIELDS}
        restored['selection_sha256'] = row['source_selection_sha256']
        self.assertEqual(restored, original)
      self.assertEqual(lineage['manifest_sha256'], canonical_sha256(
        {key: value for key, value in lineage.items() if key != 'manifest_sha256'}))
      with self.assertRaises(FileExistsError):
        self.call(fixture)

  def test_record_bytes_and_external_result_hashes_are_authenticated(self):
    for target in ('test-records.jsonl', 'results.json', 'commitment.json'):
      with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
        fixture = self.fixture(Path(directory))
        path = fixture['shards'][0][2] / target
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
          self.call(fixture)
        self.assertFalse((Path(directory) / 'aggregate').exists())

  def test_overlapping_and_missing_seed_shards_fail(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = self.fixture(Path(directory))
      with self.assertRaisesRegex(ValueError, 'overlap'):
        self.call(fixture, shards=[fixture['shards'][0], fixture['shards'][0]])
      with self.assertRaisesRegex(ValueError, 'omit a combined'):
        self.call(fixture, shards=fixture['shards'][:1])

  def test_changed_head_or_backbone_configuration_fails_before_aggregation(self):
    mutations = [lambda identity: identity['head_config'].update(rank=32),
                 lambda identity: identity['backbone_runtime_config']['noise'].update(eps=0.01),
                 lambda identity: identity['train_source'].update(selected_token_sha256=sha('different'))]
    for mutation in mutations:
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
        fixture = self.fixture(Path(directory), identity_mutation=mutation)
        with self.assertRaisesRegex(ValueError, 'seed-invariant'):
          self.call(fixture)

  def test_changed_combined_selection_is_not_accepted_as_new_authority(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = self.fixture(Path(directory))
      path = fixture['combined'].parent / 'selection.json'
      selection = json.loads(path.read_text())
      selection['selected'][0]['checkpoint_step'] = 0
      del selection['selection_sha256']
      selection['selection_sha256'] = canonical_sha256(selection)
      write_json(path, selection)
      sidecar = json.loads(fixture['combined'].read_text())
      sidecar['selection_sha256'], sidecar['selection_file_sha256'] = selection['selection_sha256'], file_sha256(path)
      del sidecar['manifest_sha256']
      sidecar['manifest_sha256'] = canonical_sha256(sidecar)
      write_json(fixture['combined'], sidecar)
      with self.assertRaisesRegex(ValueError, 'not reproducible'):
        self.call(fixture)

  def test_missing_document_and_extra_corruption_fail_despite_recomputed_shard_results(self):
    for mutation in ('missing_document', 'extra_corruption'):
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
        fixture = self.fixture(Path(directory))
        shard = fixture['shards'][0]
        path = shard[2] / 'test-records.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == 'missing_document':
          rows = [row for row in rows if row['example_index'] != 1]
        else:
          extra = copy.deepcopy(rows)
          for row in extra:
            row['corruption_seed'] += 777
          rows.extend(extra)
        write_rows(path, rows)
        self.rehash_result(shard, recompute=True)
        with self.assertRaisesRegex(ValueError, 'missing|extra, duplicate'):
          self.call(fixture)

  def test_published_statistics_and_scoring_precision_must_match(self):
    for target in ('statistics', 'precision'):
      with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
        fixture = self.fixture(Path(directory))
        shard = fixture['shards'][0]
        if target == 'statistics':
          path = shard[2] / 'results.json'
          result = json.loads(path.read_text())
          result['overall']['metrics_nats_per_masked_token']['directional_vs_unary']['estimate'] += 0.001
          write_json(path, result)
          shard[3] = file_sha256(path)
        else:
          path = shard[2] / 'commitment.json'
          commitment = json.loads(path.read_text())
          commitment['forest_inference_dtype'] = 'float32'
          write_json(path, commitment)
          self.rehash_result(shard)
        with self.assertRaisesRegex(ValueError, 'statistics differ|scoring protocol'):
          self.call(fixture)

  def test_training_relocation_is_authenticated_by_original_protocol_hash(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = self.fixture(Path(directory))
      moved = Path(directory) / 'relocated-seed1'
      fixture['runs'][0].rename(moved)
      summary, _ = self.call(fixture, run_dirs={1: moved})
      self.assertEqual(summary['overall']['training_seeds'], 3)

  def test_combined_seal_can_be_created_from_verified_relocated_runs(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = self.fixture(Path(directory))
      original = {shard[0]: shard[0].read_bytes() for shard in fixture['shards']}
      moved = Path(directory) / 'relocated-seed1'
      fixture['runs'][0].rename(moved)
      new_seal = Path(directory) / 'combined-after-relocation'
      seal([moved, *fixture['runs'][1:]], fixture['data'], file_sha256(fixture['data']), new_seal)
      fixture['combined'] = new_seal / 'selection-manifest.json'
      summary, lineage = self.call(fixture, run_dirs={1: moved})
      self.assertEqual(summary['overall']['training_seeds'], 3)
      self.assertEqual(lineage['training_runs'][0]['authenticated_run_dir'], str(moved.resolve()))
      for path, payload in original.items():
        self.assertEqual(path.read_bytes(), payload)


if __name__ == '__main__':
  unittest.main()
