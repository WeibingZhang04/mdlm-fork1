"""Authenticated old-cache-only coupling fit and finite CPU gradient smoke."""

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from models.contextual_unary import ContextualUnaryAdapter
from scripts import run_staged_centered_overfit as runner


class StagedCenteredOverfitTest(unittest.TestCase):

  def setUp(self):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.source = self.root / 'old-debug'
    self.source.mkdir()
    self.checkpoint = self.root / 'step-001000.pt'
    self.output = self.root / 'new-diagnostic'
    unary = ContextualUnaryAdapter(2, 3, rank=2, time_embed_dim=2, top_k=2)
    identity = {
      'head_config': dict(hidden_size=2, vocab_size=3, unary_rank=2, time_embed_dim=2, top_k=2, seed=1),
      'source_sha256': {name: runner.file_sha256(runner.ROOT / name) for name in
                       ('models/contextual_unary.py', 'models/structured_decoder.py')},
      'backbone_provenance': {'verified_fixture': 'no backbone exists or runs'},
    }
    self.state = {'schema_version': 1, 'completed_steps': 1000,
                  'config': {'seed': 1, 'backbone_batch_size': 1}, 'identity': identity,
                  'heads': {'unary': unary.state_dict(), 'shared': {}, 'directional': {}}}
    torch.save(self.state, self.checkpoint)
    self.cache_reports, self.cache_pins = {}, {}
    for split, targets in (('train', [[0, 1], [1, 0]]), ('dev', [[0, 0], [1, 1]])):
      directory = self.source / 'cache' / split
      directory.mkdir(parents=True)
      digest = hashlib.sha256()
      for index, target in enumerate(targets):
        row = {'hidden': torch.zeros(2, 2), 'logits': torch.tensor([[0., 0., -torch.inf]]).expand(2, 3).clone(),
               'targets': torch.tensor(target), 'active': torch.ones(2, dtype=torch.bool),
               'sigma': torch.tensor(0.4), 'corrupted': torch.full((2,), 2, dtype=torch.long)}
        for key in runner.CACHE_KEYS:
          digest.update(key.encode())
          digest.update(row[key].numpy().tobytes())
        torch.save(row, directory / f'example-{index:04d}.pt')
      self.cache_pins[split] = digest.hexdigest()
      self.cache_reports[split] = {'cache_sha256': digest.hexdigest(), 'examples': 2, 'active_tokens': 4}
    self.source_result = {'completed': True, 'backbone_frozen': True, 'hidden_size': 2, 'vocab_size': 3,
                          'arguments': {'length': 2}, 'cache': self.cache_reports,
                          'backbone_provenance': identity['backbone_provenance']}
    (self.source / 'results.json').write_text(json.dumps(self.source_result))

  def arguments(self):
    return ['--source-run-dir', str(self.source), '--source-results-sha256',
            runner.file_sha256(self.source / 'results.json'), '--unary-checkpoint', str(self.checkpoint),
            '--output-dir', str(self.output), '--steps', '30', '--eval-every', '15',
            '--batch-size', '2', '--feature-dim', '1', '--learning-rate', '.08',
            '--device', 'cpu', '--threads', '1']

  def run_fixture(self, extra=()):
    with mock.patch.object(runner, 'UNARY_CHECKPOINT_SHA256', runner.file_sha256(self.checkpoint)), \
         mock.patch.object(runner, 'OLD_CACHE_SHA256', self.cache_pins), redirect_stdout(io.StringIO()):
      runner.main(self.arguments() + list(extra))

  def test_cpu_coupling_only_fit_records_full_artifacts_and_frozen_invariants(self):
    original = self.checkpoint.read_bytes()
    source_bytes = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
    self.run_fixture(['--include-dev'])
    result = json.loads((self.output / 'results.json').read_text())
    protocol = json.loads((self.output / 'protocol.json').read_text())
    self.assertTrue(result['completed'])
    self.assertEqual(result['completed_steps'], 30)
    self.assertEqual(result['backbone_calls'], 0)
    self.assertFalse(result['reserved_test_access'])
    self.assertTrue(result['frozen_unary_unchanged'])
    self.assertTrue(result['input_caches_unchanged'])
    self.assertGreater(result['train_nll_reduction'], 0.05)
    self.assertEqual([row['step'] for row in result['curve']], [0, 15, 30])
    for row in result['curve']:
      self.assertEqual(row['frozen_unary_sha256'], protocol['unary']['unary_state_sha256'])
      for split in ('train', 'dev'):
        self.assertLess(row[split]['max_marginal_error'], 1e-12)
        self.assertLess(row[split]['max_direct_inferred_score_error_nats'], 1e-12)
        self.assertAlmostEqual(row[split]['frozen_unary_nll_per_masked_token'], 0.6931471805599453)
    self.assertLess(abs(result['curve'][0]['train']['dependence_gain_nats_per_masked_token']), 1e-12)
    steps = [json.loads(line) for line in (self.output / 'training-steps.jsonl').read_text().splitlines()]
    self.assertEqual(len(steps), 30)
    self.assertGreater(steps[0]['gradient_norm_before_clip'], 0)
    self.assertNotEqual(protocol['initial_head_sha256'], result['final_head_sha256'])
    checkpoint = torch.load(self.output / 'final-checkpoint.pt', weights_only=True)
    frozen = {key.removeprefix('_frozen_unary.'): value for key, value in checkpoint['head'].items()
              if key.startswith('_frozen_unary.')}
    self.assertEqual(runner.tensor_state_sha256(frozen), protocol['unary']['unary_state_sha256'])
    for name, digest in json.loads((self.output / 'hashes.json').read_text()).items():
      self.assertEqual(runner.file_sha256(self.output / name), digest)
    self.assertEqual(self.checkpoint.read_bytes(), original)
    self.assertEqual({path: path.read_bytes() for path in source_bytes}, source_bytes)

  def test_default_never_opens_dev_or_any_reserved_test_files(self):
    with mock.patch.object(runner, 'load_old_cache', wraps=runner.load_old_cache) as loader:
      self.run_fixture(['--steps', '1'])
    self.assertEqual([call.args[2] for call in loader.call_args_list], ['train'])
    protocol = json.loads((self.output / 'protocol.json').read_text())
    self.assertEqual(set(protocol['cache']), {'train'})
    self.assertFalse(protocol['reserved_test_access'])

  def test_checkpoint_hash_is_checked_before_deserialization(self):
    with mock.patch.object(runner.torch, 'load') as load:
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        runner.load_unary(self.checkpoint)
    load.assert_not_called()

  def test_unary_schema_and_source_code_are_strict(self):
    for change in ('step', 'source', 'tensor'):
      with self.subTest(change=change):
        state = dict(self.state)
        if change == 'step':
          state['completed_steps'] = 900
        elif change == 'source':
          state['identity'] = {**state['identity'], 'source_sha256': {}}
        else:
          state['heads'] = {**state['heads'], 'unary': {**state['heads']['unary'], 'unknown': torch.zeros(1)}}
        torch.save(state, self.checkpoint)
        with mock.patch.object(runner, 'UNARY_CHECKPOINT_SHA256', runner.file_sha256(self.checkpoint)):
          with self.assertRaises(ValueError):
            runner.load_unary(self.checkpoint)

  def test_non_debug_cache_hash_and_tampered_tensor_bytes_are_rejected(self):
    with self.assertRaisesRegex(ValueError, 'pinned OLD'):
      runner.load_old_cache(self.source / 'cache/train', self.cache_reports['train'], 'train',
                            hidden_size=2, vocab_size=3, length=2)
    path = self.source / 'cache/train/example-0000.pt'
    row = torch.load(path, weights_only=True)
    row['hidden'][0, 0] = 1
    torch.save(row, path)
    with self.assertRaisesRegex(ValueError, 'tensor contents differ'):
      self.run_fixture()
    self.assertFalse(self.output.exists())

  def test_serialized_cache_metadata_is_checked_even_if_tensor_bytes_match(self):
    path = self.source / 'cache/train/example-0000.pt'
    row = torch.load(path, weights_only=True)
    row['hidden'] = row['hidden'].flatten()
    torch.save(row, path)
    with self.assertRaisesRegex(ValueError, 'shape/dtype differs'):
      self.run_fixture()

  def test_output_is_exclusive_and_invalid_inputs_fail_before_training(self):
    self.output.mkdir()
    with self.assertRaises(FileExistsError):
      self.run_fixture()
    with self.assertRaisesRegex(ValueError, 'configuration'):
      self.run_fixture(['--learning-rate', 'nan'])

  def audit_problem(self):
    unary = ContextualUnaryAdapter(2, 3, rank=2, time_embed_dim=2, top_k=2)
    head = runner.FrozenUnaryCenteredForestHead(unary, feature_dim=1)
    records = [torch.load(path, weights_only=True) for path in sorted((self.source / 'cache/train').glob('*.pt'))]
    for row in records:
      row['logits'][..., 2] = -2.0  # A real, nonempty residual for the audit tests.
    return head, runner.FrozenCache(records=records)

  def test_audit_compares_to_independent_frozen_reference_not_output_declarations(self):
    for changed in ('unaries', 'candidate_ids', 'tail'):
      with self.subTest(changed=changed):
        head, cache = self.audit_problem()
        original = head.forward

        def tampered(*args, **kwargs):
          output = original(*args, **kwargs)
          if changed == 'unaries':
            potentials = output.unary_log_potentials.clone()
            potentials[..., 0] += 0.3
            return replace(output, unary_log_potentials=potentials.log_softmax(-1))
          if changed == 'candidate_ids':
            return replace(output, candidate_ids=output.candidate_ids.flip(-1))
          return replace(output, base_tail_log_mass=output.base_tail_log_mass + 0.1)

        with mock.patch.object(head, 'forward', side_effect=tampered):
          with self.assertRaises(AssertionError):
            runner.evaluate_debug(head, cache, 2, 'cpu')

  def test_nan_inferred_marginals_fail_before_python_max(self):
    head, cache = self.audit_problem()
    original = runner.infer_structured_distribution

    def nan_marginals(*args, **kwargs):
      inference = original(*args, **kwargs)
      marginals = replace(inference.marginals, node_log_marginals=torch.full_like(
        inference.marginals.node_log_marginals, torch.nan))
      return replace(inference, marginals=marginals)  # The partition remains finite.

    with mock.patch.object(runner, 'infer_structured_distribution', side_effect=nan_marginals):
      with self.assertRaisesRegex(FloatingPointError, 'marginal audit'):
        runner.evaluate_debug(head, cache, 2, 'cpu')

  def test_nonfinite_scores_and_overflowed_score_errors_are_rejected(self):
    head, cache = self.audit_problem()
    for inferred in (float('nan'), -1e308):
      with self.subTest(inferred=inferred), \
           mock.patch.object(runner, 'centered_forest_log_probability', return_value=torch.full((2,), 1e308, dtype=torch.float64)), \
           mock.patch.object(runner, 'structured_token_log_probability', return_value=torch.full((2,), inferred, dtype=torch.float64)):
        with self.assertRaises(FloatingPointError):
          runner.evaluate_debug(head, cache, 2, 'cpu')


if __name__ == '__main__':
  unittest.main()
