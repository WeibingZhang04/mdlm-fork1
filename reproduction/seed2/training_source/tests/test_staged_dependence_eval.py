import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from models.structured_decoder import ContextualCouplingForestHead
from scripts.run_staged_dependence_eval import (
  canonical_bytes,
  deterministic_mask,
  evaluate_example,
  identity_overrides,
  load_examples,
  read_authenticated,
  summarize_records,
)


class _Tokenizer:
  def __len__(self):
    return 10

  def encode(self, text, add_special_tokens=False):
    return [int(value) for value in text.split()]


class StagedDependenceEvalTest(unittest.TestCase):

  def test_input_explicit_splits_and_tokenizer_path(self):
    rows = [
      {'id': 'd1', 'split': 'dev', 'input_ids': [1, 2, 3, 4]},
      {'id': 'd2', 'split': 'dev', 'input_ids': [2, 3, 4, 5]},
      {'id': 't1', 'split': 'test', 'text': '4 3 2 1'},
    ]
    payload = b'\n'.join(canonical_bytes(row) for row in rows)
    examples = load_examples(payload, _Tokenizer(), 4, 1, 9)
    self.assertEqual([row['id'] for row in examples], ['d1', 't1'])
    self.assertEqual(examples[1]['input_ids'], [4, 3, 2, 1])
    self.assertEqual(len(examples[0]['input_ids_sha256']), 64)

  def test_input_rejects_split_leakage_duplicate_ids_and_mask_tokens(self):
    dev = {'id': 'd', 'split': 'dev', 'input_ids': [1, 2, 3, 4]}
    test = {'id': 't', 'split': 'test', 'input_ids': [4, 3, 2, 1]}
    cases = [
      [dev, {**test, 'input_ids': dev['input_ids']}],
      [dev, {**test, 'id': 'd'}],
      [dev, {**test, 'input_ids': [9, 3, 2, 1]}],
      [{**dev, 'document_id': 'same'}, {**test, 'document_id': 'same'}],
      [dev, {**test, 'split': 'train'}],
      [dev, {**test, 'input_ids': [1, 2]}],
    ]
    for rows in cases:
      with self.assertRaises(ValueError):
        load_examples(b'\n'.join(canonical_bytes(row) for row in rows), _Tokenizer(), 4, 32, 9)

  def test_masks_are_nested_replayable_and_order_independent(self):
    low = deterministic_mask('doc-3', 128, 0.25, 17)
    high = deterministic_mask('doc-3', 128, 0.9, 17)
    self.assertTrue(bool((~low | high).all()))
    torch.testing.assert_close(low, deterministic_mask('doc-3', 128, 0.25, 17))
    self.assertFalse(torch.equal(low, deterministic_mask('doc-3', 128, 0.25, 18)))

  def test_config_overrides_restore_all_manifest_semantics(self):
    identity = {
      'candidate_top_k': 32, 'topology_mode': 'fixed', 'factor_mode': 'dynamic',
      'independent_mode': False, 'topology_weight': 0.0,
      'head_semantics': {'rank': 8, 'fixed_edges': [[0, 1]], 'fixed_edge_path': None},
      'training_semantics': {'objective_name': 'conditional', 'factorized_aux_weight': 0.2},
    }
    overrides = identity_overrides(identity, 128)
    values = {key: json.loads(value) for key, value in (item.split('=', 1) for item in overrides)}
    self.assertEqual(values['model.length'], 128)
    self.assertEqual(values['model.structured_decoder.topology_mode'], 'fixed')
    self.assertEqual(values['model.structured_decoder.top_k'], 32)
    self.assertEqual(values['model.structured_decoder.training.factorized_aux_weight'], 0.2)
    self.assertEqual(values['model.structured_decoder.fixed_edges'], [[0, 1]])

  def test_test_scores_cannot_change_selected_lambda(self):
    def record(split, scores):
      return {'id': split, 'split': split, 'active_token_count': 10, 'mask_rate': 0.5,
              'backbone_log_probability': -21, 'marginal_log_probability': scores[0],
              'joint_log_probability': scores[2],
              'lambda_log_probabilities': dict(zip(['0.0', '0.5', '1.0'], scores))}
    dev = record('dev', [-20, -10, -15])
    first = summarize_records([dev, record('test', [-50, -80, -20])], [0.0, 0.5, 1.0])
    second = summarize_records([dev, record('test', [-1, -2000, -999])], [0.0, 0.5, 1.0])
    self.assertEqual(first['selected_lambda'], 0.5)
    self.assertEqual(second['selected_lambda'], 0.5)
    self.assertEqual(first['splits']['test']['selected_lambda_nll'], 8.0)
    self.assertEqual(second['splits']['test']['selected_lambda_nll'], 200.0)

  def test_authentication_rejects_changed_bytes(self):
    import hashlib
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'input.jsonl'
      payload = b'{"id":"original"}\n'
      path.write_bytes(payload)
      sha = hashlib.sha256(payload).hexdigest()
      self.assertEqual(read_authenticated(path, sha), payload)
      path.write_bytes(payload + b'\n')
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        read_authenticated(path, sha)

  def test_evaluate_example_uses_sigma_conditioning_and_exact_scores(self):
    torch.manual_seed(2)
    head = ContextualCouplingForestHead(
      hidden_size=8, vocab_size=5, top_k=3, rank=2, time_embed_dim=4,
      topology_dim=6, local_window=2, num_anchor_slots=2,
      contextual_neighbors=1, component_size_cap=4, topology_mode='fixed')

    class Model:
      mask_index = 4
      config = SimpleNamespace(noise=SimpleNamespace(type='loglinear'))

      class Noise:
        eps = 1e-3

        def __call__(self, value):
          return -torch.log1p(-(1 - self.eps) * value), torch.ones_like(value)

      noise = Noise()

      def _structured_head_output(self, *, tokens, conditioning, active_mask, force_no_grad_backbone):
        self.conditioning = conditioning
        logits = torch.randn(*tokens.shape, 5)
        logits[..., self.mask_index] = -torch.inf
        hidden = torch.randn(*tokens.shape, 8)
        return head(hidden, logits, conditioning[:, 0], active_mask), logits

    model = Model()
    example = {'id': 'smoke', 'split': 'dev', 'input_ids': [0, 1, 2, 3] * 4,
               'input_ids_sha256': 'a' * 64, 'dataset': None, 'document_id': None}
    result = evaluate_example(model, example, 0.5, [0.0, 0.5, 1.0], 11, 'cpu')
    self.assertAlmostEqual(result['conditioning_sigma'], -__import__('math').log(0.5), places=6)
    self.assertAlmostEqual(result['lambda_log_probabilities']['0.0'], result['marginal_log_probability'])
    self.assertAlmostEqual(result['lambda_log_probabilities']['1.0'], result['joint_log_probability'])
    self.assertAlmostEqual(sum(result['edge_log_dependence']), result['joint_log_probability'] - result['marginal_log_probability'])
    self.assertEqual(len(result['selected_edges']), len(result['edge_target_is_residual']))


if __name__ == '__main__':
  unittest.main()
