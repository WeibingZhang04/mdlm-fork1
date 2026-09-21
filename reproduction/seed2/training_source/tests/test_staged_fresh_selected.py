"""Sealed selection, exact checkpoint loading and paired FP64 test scoring."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation.fresh_pair_statistics import ARMS, canonical_sha256, select_checkpoints, summarize_test
from scripts.evaluate_staged_fresh_selected import (
  REPO_ROOT, TEST_SEED_OFFSET, evaluate_selected, load_selected_heads,
  load_selection, load_split_manifest, main, score_head_float64,
  validate_test_tokens, verify_backbone,
)
from scripts.run_staged_fresh_training import SOURCE_FILES, online_backbone_batch, read_documents
from scripts.run_staged_real_overfit import file_sha256, fixed_masks, make_head, score_batch, token_digest
from tests.test_fresh_pair_statistics import observation, sha
from tests.test_staged_real_overfit import TinyFrozenBackbone


class StagedFreshSelectedTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(32)
    torch.set_num_threads(1)
    self.model = TinyFrozenBackbone().eval()
    self.tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    self.documents = [sha('test-a'), sha('test-b')]
    self.head_config = dict(hidden_size=8, vocab_size=11, seed=3, rank=4,
                            directional_rank=2, unary_rank=4, top_k=4,
                            time_embed_dim=6, init_std=0.25, component_size_cap=0)
    self.heads = {(3, arm): make_head(arm, **self.head_config).requires_grad_(False).eval() for arm in ARMS}
    rows = [observation(arm, 3, 5, sha('dev-a'), rate)
            for arm in ARMS for rate in (0.25, 0.5)]
    self.selection = select_checkpoints(rows, excluded_test_document_ids=[sha('train-a'), sha('debug-a')])
    self.manifest = {
      'artifact': 'fresh_staged_openwebtext_splits', 'schema_version': 1,
      'length': 6, 'counts': {'train': 1, 'dev': 1, 'test': 2},
      'splits': {split: {'source_document_sha256': documents,
                        'prefix_token_ids_sha256': [sha(('prefix', document)) for document in documents]}
                 for split, documents in (
        ('train', [sha('train-a')]), ('dev', [sha('dev-a')]), ('test', self.documents))},
      'excluded_old_debug_document_sha256': [sha('debug-a')],
      'file_sha256': {f'{split}.jsonl': sha(split) for split in ('train', 'dev', 'test')},
    }
    self.manifest['splits']['test']['prefix_token_ids_sha256'] = [canonical_sha256(row.tolist()) for row in self.tokens]
    self.manifest['manifest_sha256'] = canonical_sha256(self.manifest)
    self.identity = {
      'head_config': self.head_config, 'dataset': 'corpus', 'backbone_provenance': {'release': 'test'},
      'backbone_runtime_config': {'noise': {'type': 'loglinear', 'eps': 0.001}},
      'runtime': {'torch_version': str(torch.__version__), 'device_type': 'cpu',
                  'gpu_name': None, 'cached_output_dtype': 'float32', 'backbone_cuda_autocast': 'bfloat16'},
      'source_sha256': {path: file_sha256(REPO_ROOT / path) for path in SOURCE_FILES},
      **{f'{split}_source': {'examples': 1, 'length': 6, 'file_sha256': sha(split),
                            'document_ids_sha256': canonical_sha256([sha(f'{split}-a')])}
         for split in ('train', 'dev')},
    }
    self.config = {'seed': 3, 'batch_size': 2, 'backbone_batch_size': 1,
                   'mask_rates': [0.25, 0.5], 'pair_warmup_steps': 1}
    self.metadata = {3: {'identity': self.identity, 'config': self.config}}

  @staticmethod
  def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
    return file_sha256(path)

  def write_checkpoint(self, path, mutate=None):
    state = {'schema_version': 1, 'identity': copy.deepcopy(self.identity),
             'config': copy.deepcopy(self.config), 'completed_steps': 5,
             'heads': {arm: copy.deepcopy(self.heads[(3, arm)].state_dict()) for arm in ARMS}}
    if mutate is not None:
      mutate(state)
    torch.save(state, path)
    digest = file_sha256(path)
    selection = copy.deepcopy(self.selection)
    for row in selection['selected']:
      row['checkpoint_sha256'] = digest
    del selection['selection_sha256']
    selection['selection_sha256'] = canonical_sha256(selection)
    return selection

  def test_selection_authentication_and_duplicate_arms_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'selection.json'
      digest = self.write_json(path, self.selection)
      self.assertEqual(load_selection(path, digest), self.selection)
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        load_selection(path, '0' * 64)
      duplicate = copy.deepcopy(self.selection)
      duplicate['selected'].append(duplicate['selected'][0])
      del duplicate['selection_sha256']
      duplicate['selection_sha256'] = canonical_sha256(duplicate)
      digest = self.write_json(path, duplicate)
      with self.assertRaisesRegex(ValueError, 'exactly once'):
        load_selection(path, digest)

  def test_manifest_requires_complete_training_and_debug_exclusions(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'data.json'
      digest = self.write_json(path, self.manifest)
      self.assertEqual(load_split_manifest(path, digest, self.selection), self.manifest)
      for document in (sha('train-a'), sha('debug-a')):
        selection = copy.deepcopy(self.selection)
        selection['excluded_test_document_ids'].remove(document)
        with self.assertRaisesRegex(ValueError, 'exclusions'):
          load_split_manifest(path, digest, selection)

  def test_strict_checkpoint_load_and_torchversion_compatibility(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'head.pt'
      selection = self.write_checkpoint(path, lambda state: state['identity']['runtime'].update(torch_version=torch.__version__))
      loaded, metadata = load_selected_heads([path], selection, self.manifest)
      for key, head in loaded.items():
        self.assertFalse(head.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in head.parameters()))
        for name, tensor in head.state_dict().items():
          torch.testing.assert_close(tensor, self.heads[key].state_dict()[name], atol=0, rtol=0)
      self.assertEqual(metadata[3]['config'], self.config)
      with self.assertRaisesRegex(ValueError, 'exactly once'):
        load_selected_heads([path, path], selection, self.manifest)

  def test_checkpoint_rejects_dtype_nonfinite_config_data_and_source_drift(self):
    mutations = [
      (lambda state: state['heads']['unary'].update(token_embedding_weight=torch.ones(1)), 'key mismatch'),
      (lambda state: state['heads']['unary'].__setitem__('token_embedding.weight',
         state['heads']['unary']['token_embedding.weight'].double()), 'dtype mismatch'),
      (lambda state: state['heads']['unary']['token_embedding.weight'].fill_(torch.nan), 'nonfinite'),
      (lambda state: state['config'].update(backbone_batch_size=2), 'serial backbone'),
      (lambda state: state['identity']['train_source'].update(document_ids_sha256=sha('wrong')), 'train data identity'),
      (lambda state: state['identity']['source_sha256'].update({'models/dit.py': sha('wrong')}), 'source changed'),
    ]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'head.pt'
      for mutation, message in mutations:
        with self.subTest(message=message):
          selection = self.write_checkpoint(path, mutation)
          with self.assertRaisesRegex(ValueError, message):
            load_selected_heads([path], selection, self.manifest)

  def test_float64_scoring_matches_existing_score_and_bounds_decomposition(self):
    active = fixed_masks(self.tokens, 0.5, 32)
    batch = online_backbone_batch(self.model, self.tokens, active, 0.5, 'cpu', 1)
    self.assertTrue(torch.isneginf(batch['logits'][..., 10]).all())
    for arm in ARMS:
      with self.subTest(arm=arm):
        head = self.heads[(3, arm)]
        exact, original = score_head_float64(head, batch), score_batch(head, batch)
        for key in ('joint_log_probability', 'marginal_log_probability', 'dependence_log_probability'):
          self.assertEqual(exact[key].dtype, torch.float64)
          torch.testing.assert_close(exact[key], original[key].double(), atol=1e-4, rtol=1e-5)
        self.assertLess(float(exact['decomposition_error_nats'].max()), 1e-10)
        self.assertLess(int(exact['explicit_target_count'].sum()), int(active.sum()))

  def test_every_arm_sees_identical_inputs_serial_encoder_and_records_pair(self):
    seen, calls = [], []
    original_encode = self.model._structured_backbone_output

    def encode(tokens, *args, **kwargs):
      calls.append(len(tokens))
      return original_encode(tokens, *args, **kwargs)

    def score(head, batch):
      seen.append({key: value.data_ptr() for key, value in batch.items()})
      return score_head_float64(head, batch)

    with patch.object(self.model, '_structured_backbone_output', side_effect=encode), \
         patch('scripts.evaluate_staged_fresh_selected.score_head_float64', side_effect=score):
      records = evaluate_selected(self.model, self.heads, self.metadata, self.tokens,
                                    self.documents, self.selection, device='cpu')
    self.assertEqual(calls, [1, 1, 1, 1])
    self.assertEqual(len(records), 12)
    for offset in (0, 3):
      self.assertEqual(seen[offset], seen[offset + 1])
      self.assertEqual(seen[offset], seen[offset + 2])
    for row in records:
      self.assertEqual(row['selection_sha256'], self.selection['selection_sha256'])
      self.assertIn(row['corruption_seed'], (3 + TEST_SEED_OFFSET, 4 + TEST_SEED_OFFSET))
      self.assertLess(row['decomposition_error_nats'], 1e-10)
    summary = summarize_test(records, self.selection, bootstrap_replicates=20)
    self.assertEqual(summary['overall']['documents'], 2)
    self.assertFalse(summary['bootstrap']['training_seed_uncertainty_estimated'])

  def test_dev_window_and_document_exclusions_checked_before_scoring(self):
    validate_test_tokens(self.tokens, self.documents, self.selection, self.manifest)
    selection = copy.deepcopy(self.selection)
    selection['dev_clean_token_sha256s'].append(token_digest(self.tokens[0]))
    with self.assertRaisesRegex(ValueError, 'development token window'):
      validate_test_tokens(self.tokens, self.documents, selection, self.manifest)
    selection = copy.deepcopy(self.selection)
    selection['excluded_test_document_ids'].append(self.documents[0])
    with self.assertRaisesRegex(ValueError, 'excluded document'):
      validate_test_tokens(self.tokens, self.documents, selection, self.manifest)
    with self.assertRaisesRegex(ValueError, 'full prespecified'):
      validate_test_tokens(self.tokens[:1], self.documents[:1], self.selection, self.manifest)

  def test_backbone_runtime_config_and_provenance_must_match(self):
    self.model.structured_backbone_provenance = {'release': 'test'}
    # Tiny oracle lacks the full production Hydra config and structured head.
    self.model.structured_head = self.heads[(3, 'unary')]
    config = self.identity['backbone_runtime_config']
    with patch('scripts.evaluate_staged_fresh_selected.backbone_runtime_identity', return_value=config):
      verify_backbone(self.model, self.metadata, 'cpu')
      changed = copy.deepcopy(self.metadata)
      changed[3]['identity']['backbone_runtime_config']['noise']['eps'] = 0.1
      with self.assertRaisesRegex(ValueError, 'resolved backbone/noise'):
        verify_backbone(self.model, changed, 'cpu')
      changed = copy.deepcopy(self.metadata)
      changed[3]['identity']['backbone_provenance']['release'] = 'wrong'
      with self.assertRaisesRegex(ValueError, 'backbone provenance'):
        verify_backbone(self.model, changed, 'cpu')

  def test_invalid_selection_prevents_any_test_data_access(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      path = root / 'selection.json'
      self.write_json(path, self.selection)
      argv = ['--selection', str(path), '--expected-selection-sha256', '0' * 64,
              '--data-manifest', 'missing-manifest', '--expected-data-manifest-sha256', '0' * 64,
              '--test-jsonl', 'must-not-be-read', '--expected-test-jsonl-sha256', '0' * 64,
              '--expectations', 'missing', '--expected-expectations-sha256', '0' * 64,
              '--selected-checkpoint', 'missing', '--backbone-checkpoint', 'missing',
              '--output-dir', str(root / 'output')]
      with patch('scripts.evaluate_staged_fresh_selected.read_documents') as reader:
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
          main(argv)
        reader.assert_not_called()

  def test_cli_writes_commitment_before_test_read_and_records_bind_to_it(self):
    self.model.structured_backbone_provenance = self.identity['backbone_provenance']
    self.model.structured_head = self.heads[(3, 'unary')]
    self.model.tokenizer = None
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'checkpoint.pt'
      selection = self.write_checkpoint(checkpoint)
      selection_path = root / 'selection.json'
      selection_hash = self.write_json(selection_path, selection)
      test_path = root / 'test.jsonl'
      test_path.write_text(''.join(json.dumps({'document_id': doc, 'input_ids': tokens.tolist()}) + '\n'
                                   for doc, tokens in zip(self.documents, self.tokens)))
      manifest = copy.deepcopy(self.manifest)
      manifest['file_sha256']['test.jsonl'] = file_sha256(test_path)
      del manifest['manifest_sha256']
      manifest['manifest_sha256'] = canonical_sha256(manifest)
      manifest_path = root / 'manifest.json'
      manifest_hash = self.write_json(manifest_path, manifest)
      output = root / 'result'
      argv = ['--selection', str(selection_path), '--expected-selection-sha256', selection_hash,
              '--data-manifest', str(manifest_path), '--expected-data-manifest-sha256', manifest_hash,
              '--test-jsonl', str(test_path), '--expected-test-jsonl-sha256', file_sha256(test_path),
              '--expectations', 'mock-expectations', '--expected-expectations-sha256', '0' * 64,
              '--selected-checkpoint', str(checkpoint), '--backbone-checkpoint', 'mock-backbone',
              '--output-dir', str(output), '--device', 'cpu', '--threads', '1', '--bootstrap-replicates', '20']

      def guarded_read(*args, **kwargs):
        commitment = json.loads((output / 'commitment.json').read_text())
        self.assertFalse(commitment['test_tokens_read'])
        self.assertEqual(commitment['selection_sha256'], selection['selection_sha256'])
        self.assertEqual(commitment['selection_file_sha256'], selection_hash)
        return read_documents(*args, **kwargs)

      with patch('scripts.export_contextual_forest_adapter.build_production_model', return_value=self.model), \
           patch('scripts.export_contextual_forest_adapter.load_production_expectations', return_value={}), \
           patch('scripts.evaluate_staged_fresh_selected.backbone_runtime_identity',
                 return_value=self.identity['backbone_runtime_config']), \
           patch('scripts.evaluate_staged_fresh_selected.read_documents', side_effect=guarded_read) as reader:
        main(argv)
        reader.assert_called_once()
        with self.assertRaises(FileExistsError):
          main(argv)
      result = json.loads((output / 'results.json').read_text())
      self.assertEqual(result['test_records_file_sha256'], file_sha256(output / 'test-records.jsonl'))
      self.assertEqual(result['commitment_file_sha256'], file_sha256(output / 'commitment.json'))
      self.assertLess(result['max_decomposition_error_nats'], 1e-10)


if __name__ == '__main__':
  unittest.main()
