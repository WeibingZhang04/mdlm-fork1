import unittest
from pathlib import Path
import tempfile

import torch
import yaml

from scripts.train_contextual_forest_g1 import (
  REPORTED_MODELS,
  _args as training_args,
  _validate_resumed_job,
)
from synthetic.distributions import ContextSwitchingMatching
from synthetic.neural_g1 import (
  NeuralTrainConfig,
  REPORTED_FACTOR_INIT_SEED,
  REPORTED_FACTOR_INIT_STD,
  REPORTED_FACTOR_WARMUP_STEPS,
  REPORTED_HELDOUT_SEEDS,
  REPORTED_TRAINING_STEPS,
  SyntheticForestAdapter,
  dependency_adjacency,
  dependency_loss,
  dependency_targets,
  model_specs,
  sample_training_batch,
  train_adapter,
)
from structured_objective import structured_token_log_probability


class NeuralG1Test(unittest.TestCase):

  def test_reported_heldout_defaults_match_named_config_and_cli(self):
    defaults = NeuralTrainConfig()
    self.assertEqual(defaults.steps, 900)
    self.assertEqual(defaults.factor_warmup_steps, 300)
    self.assertEqual(defaults.factor_init_std, 0.25)
    self.assertEqual(defaults.factor_init_seed, 1729)
    self.assertEqual(REPORTED_TRAINING_STEPS, 900)
    self.assertEqual(REPORTED_FACTOR_WARMUP_STEPS, 300)
    self.assertEqual(REPORTED_FACTOR_INIT_STD, 0.25)
    self.assertEqual(REPORTED_FACTOR_INIT_SEED, 1729)
    self.assertEqual(REPORTED_HELDOUT_SEEDS, (9, 10, 11, 12, 13))

    args = training_args(['--output-dir', '/tmp/g1-unused'])
    self.assertEqual(args.seeds, list(REPORTED_HELDOUT_SEEDS))
    self.assertEqual(args.models, list(REPORTED_MODELS))
    self.assertEqual(args.steps, REPORTED_TRAINING_STEPS)
    self.assertEqual(
      args.factor_warmup_steps, REPORTED_FACTOR_WARMUP_STEPS)
    self.assertEqual(args.factor_init_std, REPORTED_FACTOR_INIT_STD)
    self.assertEqual(args.factor_init_seed, REPORTED_FACTOR_INIT_SEED)

    config_path = (
      Path(__file__).resolve().parents[1]
      / 'configs/experiment/contextual-forest-g1.yaml')
    named = yaml.safe_load(config_path.read_text())
    learned = named['learned_adapter']
    self.assertEqual(named['seeds'], list(REPORTED_HELDOUT_SEEDS))
    self.assertEqual(learned['steps'], REPORTED_TRAINING_STEPS)
    self.assertEqual(
      learned['factor_warmup_steps'], REPORTED_FACTOR_WARMUP_STEPS)
    self.assertEqual(learned['factor_init_std'], REPORTED_FACTOR_INIT_STD)
    self.assertEqual(learned['factor_init_seed'], REPORTED_FACTOR_INIT_SEED)

  def test_factor_initialization_scale_must_be_positive(self):
    task = ContextSwitchingMatching(vocab_size=3)
    with self.assertRaisesRegex(ValueError, 'factor_init_std'):
      SyntheticForestAdapter(
        task, model_specs(task)['contextual_forest'], factor_init_std=0.0)

  def test_factor_warmup_steps_must_be_nonnegative(self):
    task = ContextSwitchingMatching(vocab_size=3)
    with self.assertRaisesRegex(ValueError, 'factor_warmup_steps'):
      train_adapter(
        task, model_specs(task)['contextual_forest'], seed=1,
        config=NeuralTrainConfig(steps=1, factor_warmup_steps=-1),
        device=torch.device('cpu'))

  def test_factor_basis_initialization_is_seed_independent(self):
    task = ContextSwitchingMatching(vocab_size=3)
    spec = model_specs(task)['contextual_forest']
    torch.manual_seed(11)
    first = SyntheticForestAdapter(task, spec, factor_init_seed=1729)
    torch.manual_seed(97)
    second = SyntheticForestAdapter(task, spec, factor_init_seed=1729)
    torch.testing.assert_close(
      first.head.token_factor_embedding.weight,
      second.head.token_factor_embedding.weight,
      rtol=0.0, atol=0.0)

  def test_batch_respects_context_matching(self):
    task = ContextSwitchingMatching(vocab_size=4)
    contexts, tokens, _ = sample_training_batch(
      task, 200, torch.Generator().manual_seed(5), torch.device('cpu'))
    for row in range(tokens.shape[0]):
      for first, second in task.true_edges(int(contexts[row])):
        self.assertEqual(int(tokens[row, first]), int(tokens[row, second]))

  def test_contextual_adapter_has_finite_gradients(self):
    task = ContextSwitchingMatching(vocab_size=3)
    spec = model_specs(task)['contextual_forest']
    model, history = train_adapter(
      task, spec, seed=2,
      config=NeuralTrainConfig(
        steps=2, batch_size=6, eval_samples=20, log_every=1,
        factor_warmup_steps=0,
        inference_backend='low_rank'),
      device=torch.device('cpu'))
    self.assertTrue(all(torch.isfinite(torch.tensor([
      row['loss'], row['structured_nll'], row['dependency_loss'],
      row['gradient_norm']])).all() for row in history))
    contexts, tokens, timestep = sample_training_batch(
      task, 4, torch.Generator().manual_seed(8), torch.device('cpu'))
    output, logits, active = model(contexts, timestep)
    log_probability = structured_token_log_probability(
      output, logits, tokens, active)
    self.assertTrue(bool(torch.isfinite(log_probability).all()))
    self.assertGreater(
      float(dependency_loss(task, contexts, output).detach()), 0.0)

  def test_cached_dependency_adjacency_matches_direct_targets(self):
    task = ContextSwitchingMatching(vocab_size=3)
    contexts = torch.tensor([0, 1, 2])
    edges = torch.tensor([
      [[0, 1], [0, 2], [1, 4]],
      [[0, 1], [0, 2], [1, 4]],
      [[0, 1], [0, 2], [1, 4]],
    ])
    direct = dependency_targets(task, contexts, edges)
    cached = dependency_targets(
      task, contexts, edges, dependency_adjacency(task))
    torch.testing.assert_close(cached, direct)

  def test_resume_validation_rejects_mismatched_checkpoint_config(self):
    task = ContextSwitchingMatching(vocab_size=4)
    spec = model_specs(task)['contextual_forest']
    expected = NeuralTrainConfig(steps=2, eval_samples=10)
    rows = [
      {'seed': 9, 'model': 'contextual_forest', 'context': context}
      for context in range(task.num_contexts)
    ]
    history = [{'step': 2.0}]
    with tempfile.TemporaryDirectory() as directory:
      checkpoint_path = Path(directory) / 'model.pt'
      torch.save({
        'model_state_dict': {'weight': torch.ones(1)},
        'model_spec': spec,
        'train_config': NeuralTrainConfig(steps=3, eval_samples=10),
        'seed': 9,
      }, checkpoint_path)
      with self.assertRaisesRegex(ValueError, 'train config mismatch'):
        _validate_resumed_job(
          checkpoint_path, rows, history,
          seed=9,
          model_name='contextual_forest',
          expected_spec=spec,
          expected_config=expected,
          num_contexts=task.num_contexts)

  def test_resume_validation_accepts_exact_checkpoint_provenance(self):
    task = ContextSwitchingMatching(vocab_size=4)
    spec = model_specs(task)['contextual_forest']
    config = NeuralTrainConfig(steps=2, eval_samples=10)
    rows = [
      {'seed': 9, 'model': 'contextual_forest', 'context': context}
      for context in range(task.num_contexts)
    ]
    with tempfile.TemporaryDirectory() as directory:
      checkpoint_path = Path(directory) / 'model.pt'
      torch.save({
        'model_state_dict': {'weight': torch.ones(1)},
        'model_spec': spec,
        'train_config': config,
        'seed': 9,
      }, checkpoint_path)
      _validate_resumed_job(
        checkpoint_path, rows, [{'step': 2.0}],
        seed=9,
        model_name='contextual_forest',
        expected_spec=spec,
        expected_config=config,
        num_contexts=task.num_contexts)


if __name__ == '__main__':
  unittest.main()
