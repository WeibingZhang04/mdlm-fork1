import copy
import json
from types import SimpleNamespace
import unittest

import torch

from evaluation.generation_harness import (
  DEFAULT_SAMPLING_MODES,
  PromptSpec,
  expand_paired_samples,
  replay_sampling_group_trajectory,
  run_sampling_group,
  sample_from_initial_state,
  sample_trajectory_from_initial_state,
  seed_everything,
)
from scripts.run_generation_pilot import _parse_args


class ToyTokenizer:

  def batch_decode(self, rows):
    return [' '.join(str(value) for value in row) for row in rows]


class CountingBackbone(torch.nn.Module):

  def forward(self, tokens):
    batch, length = tokens.shape
    logits = torch.zeros(batch, length, 128, device=tokens.device)
    logits[..., 7] = 1.0
    return logits

  def encode(self, tokens):
    batch, length = tokens.shape
    return torch.ones(batch, length, 3, device=tokens.device)


class TrajectoryModel:

  def __init__(self, *, stochastic=False):
    self.device = torch.device('cpu')
    self.mask_index = 99
    self.sampler = 'ddpm'
    self.structured_enabled = True
    self.structured_sampling_mode = 'factorized'
    self.backbone = CountingBackbone()
    self.config = SimpleNamespace(
      backbone='dit',
      sampling=SimpleNamespace(noise_removal=True))
    self.stochastic = stochastic

  def _ddpm_update(self, x, unused_t, unused_dt):
    if self.structured_sampling_mode in {
        'structured_marginal', 'structured_joint'}:
      return self._structured_update(x)
    self.backbone(x)
    result = x.clone()
    for row in range(x.shape[0]):
      positions = torch.nonzero(x[row].eq(self.mask_index)).flatten()
      if positions.numel():
        token = (
          int(torch.randint(1, 31, ()).item())
          if self.stochastic else 7)
        result[row, positions[0]] = token
    return result

  def _structured_update(self, x):
    if not bool(x.eq(self.mask_index).any().item()):
      return x
    self._structured_backbone_output(x)
    return torch.where(x.eq(self.mask_index), torch.full_like(x, 7), x)

  def _structured_clean_sample(self, x, unused_conditioning):
    return self._structured_update(x)

  def _structured_backbone_output(self, x):
    return self.backbone.encode(x)

  def noise(self, t):
    return t, None

  def forward(self, x, unused_conditioning):
    return self.backbone(x)


class GenerationTrajectoryTest(unittest.TestCase):

  def test_generation_cli_default_grid_remains_the_original_three_modes(self):
    args = _parse_args([
      '--backbone-checkpoint', 'backbone.ckpt',
      '--backbone-sha256', '0' * 64,
      '--adapter', 'adapter.pt',
      '--adapter-sha256', '1' * 64,
      '--adapter-manifest', 'adapter.json',
      '--adapter-manifest-sha256', '2' * 64,
      '--output-dir', 'output',
    ])
    self.assertEqual(DEFAULT_SAMPLING_MODES, (
      'factorized', 'structured_marginal', 'structured_joint'))
    self.assertEqual(args.modes, list(DEFAULT_SAMPLING_MODES))

  def test_fixed_call_indices_capture_one_single_64_nfe_run(self):
    model = TrajectoryModel()
    initial = torch.tensor([
      [5, 99, 99, 99, 6],
      [8, 99, 99, 99, 9],
    ])
    requested = [0, 16, 32, 48, 63, 64]
    generated, measured_nfe, trajectory = (
      sample_trajectory_from_initial_state(
        model,
        initial,
        nfe_budget=64,
        snapshot_call_indices=requested))

    self.assertEqual(measured_nfe, 64)
    self.assertEqual(trajectory['trajectory_scope'], 'single_nfe_run')
    self.assertEqual(
      trajectory['snapshot_call_indices_captured'], requested)
    self.assertEqual(trajectory['snapshot_call_indices_missing'], [])
    snapshots = trajectory['snapshots']
    self.assertEqual(
      [snapshot['model_call_index'] for snapshot in snapshots], requested)
    self.assertEqual(snapshots[0]['stage'], 'initial')
    self.assertIsNone(snapshots[0]['timestep'])
    self.assertEqual(snapshots[0]['unresolved_active_count'], 6)
    self.assertEqual(snapshots[-1]['stage'], 'final')
    self.assertEqual(snapshots[-1]['token_ids'], generated.tolist())
    self.assertEqual(snapshots[-1]['unresolved_active_count'], 0)
    self.assertEqual(snapshots[-1]['active_mask'][0], [
      False, True, True, True, False])
    self.assertEqual(snapshots[-1]['unresolved_active_mask'][0], [
      False, False, False, False, False])
    self.assertEqual(len(snapshots[-1]['timestep']), 2)
    json.dumps(trajectory)

  def test_capture_does_not_change_sampling_rng_or_final_tokens(self):
    initial = torch.tensor([[5, 99, 99, 99, 6]])
    ordinary_model = TrajectoryModel(stochastic=True)
    seed_everything(12345, torch.device('cpu'))
    ordinary, ordinary_nfe = sample_from_initial_state(
      ordinary_model, initial, nfe_budget=5)
    ordinary_next_random = torch.rand(4)

    trajectory_model = TrajectoryModel(stochastic=True)
    seed_everything(12345, torch.device('cpu'))
    captured, captured_nfe, unused_trajectory = (
      sample_trajectory_from_initial_state(
        trajectory_model,
        initial,
        nfe_budget=5,
        snapshot_call_indices=[0, 2, 5]))
    captured_next_random = torch.rand(4)

    self.assertTrue(torch.equal(captured, ordinary))
    self.assertEqual(captured_nfe, ordinary_nfe)
    self.assertTrue(torch.equal(captured_next_random, ordinary_next_random))

  def test_early_structured_completion_reports_unreached_calls(self):
    model = TrajectoryModel()
    model.structured_sampling_mode = 'structured_joint'
    initial = torch.tensor([[5, 99, 6]])
    generated, measured_nfe, trajectory = (
      sample_trajectory_from_initial_state(
        model,
        initial,
        nfe_budget=64,
        snapshot_call_indices=[0, 1, 2, 64]))

    self.assertEqual(generated.tolist(), [[5, 7, 6]])
    self.assertEqual(measured_nfe, 1)
    self.assertEqual(
      trajectory['snapshot_call_indices_captured'], [0, 1])
    self.assertEqual(trajectory['snapshot_call_indices_missing'], [2, 64])
    self.assertEqual(
      [snapshot['stage'] for snapshot in trajectory['snapshots']],
      ['initial', 'denoising', 'final'])
    self.assertEqual(
      [snapshot['model_call_index'] for snapshot in trajectory['snapshots']],
      [0, 1, 1])

  def test_confidence_gated_control_uses_factorized_nfe_counter_path(self):
    model = TrajectoryModel()
    model.structured_sampling_mode = 'factorized_confidence_gated'
    initial = torch.tensor([[5, 99, 99, 6]])
    generated, measured_nfe, trajectory = (
      sample_trajectory_from_initial_state(
        model,
        initial,
        nfe_budget=4,
        snapshot_call_indices=[0, 2, 4]))
    self.assertEqual(generated.tolist(), [[5, 7, 7, 6]])
    self.assertEqual(measured_nfe, 4)
    self.assertEqual(
      trajectory['snapshot_call_indices_captured'], [0, 2, 4])

  def test_exact_replay_requires_and_verifies_complete_ordered_batch(self):
    prompts = [
      PromptSpec('first', (5, 99, 99, 6), (False, True, True, False)),
      PromptSpec('second', (8, 99, 99, 9), (False, True, True, False)),
    ]
    samples = expand_paired_samples(prompts, num_samples=2, base_seed=30)
    records, unused_batch = run_sampling_group(
      TrajectoryModel(),
      samples,
      sampling_mode='factorized',
      nfe_budget=4,
      tokenizer=ToyTokenizer(),
      device=torch.device('cpu'))
    model = TrajectoryModel()
    replay = replay_sampling_group_trajectory(
      model,
      samples,
      sampling_mode='factorized',
      nfe_budget=4,
      expected_records=records,
      snapshot_call_indices=[0, 2, 4],
      device=torch.device('cpu'))

    self.assertTrue(replay['final_token_ids_match_expected'])
    self.assertEqual(
      replay['final_token_ids'],
      [record['sample_token_ids'] for record in records])
    self.assertEqual(replay['batch_size'], 2)
    self.assertEqual(
      [row['pair_key'] for row in replay['batch_order']],
      [sample.pair_key for sample in samples])
    self.assertEqual(model.structured_sampling_mode, 'factorized')
    json.dumps(replay)

    with self.assertRaisesRegex(ValueError, 'every record'):
      replay_sampling_group_trajectory(
        TrajectoryModel(), samples,
        sampling_mode='factorized', nfe_budget=4,
        expected_records=records[:1], snapshot_call_indices=[0, 4],
        device=torch.device('cpu'))
    with self.assertRaisesRegex(ValueError, 'ordered source batch'):
      replay_sampling_group_trajectory(
        TrajectoryModel(), samples,
        sampling_mode='factorized', nfe_budget=4,
        expected_records=list(reversed(records)),
        snapshot_call_indices=[0, 4], device=torch.device('cpu'))

    mismatched = copy.deepcopy(records)
    mismatched[1]['sample_token_ids'][1] = 12
    with self.assertRaisesRegex(AssertionError, 'token-id mismatch'):
      replay_sampling_group_trajectory(
        TrajectoryModel(), samples,
        sampling_mode='factorized', nfe_budget=4,
        expected_records=mismatched, snapshot_call_indices=[0, 4],
        device=torch.device('cpu'))

  def test_invalid_snapshot_indices_fail_closed(self):
    model = TrajectoryModel()
    initial = torch.tensor([[99, 99]])
    with self.assertRaisesRegex(ValueError, 'non-negative'):
      sample_trajectory_from_initial_state(
        model, initial, nfe_budget=4, snapshot_call_indices=[-1])
    with self.assertRaisesRegex(TypeError, 'integers'):
      sample_trajectory_from_initial_state(
        model, initial, nfe_budget=4, snapshot_call_indices=[True])


if __name__ == '__main__':
  unittest.main()
