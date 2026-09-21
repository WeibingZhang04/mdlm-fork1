from pathlib import Path
import unittest

from scripts.run_wikitext_reveal_control_queue import (
  BASE_SEED,
  BATCH_SIZE,
  MODES,
  NFE_BUDGETS,
  NUM_SAMPLES,
  NUM_SHARDS,
  Paths,
  _absolute_path_without_resolving_symlinks,
  build_tasks,
  expected_shard_samples,
  launch_plan_sha256,
)


class WikitextRevealControlQueueTest(unittest.TestCase):

  def setUp(self):
    self.paths = Paths(
      runner_repo=Path('/runner'),
      python=Path('/venv/python'),
      experiment_root=Path('/experiments/paper'),
      expansion_root=Path('/experiments/expansion'),
      backbone=Path('/checkpoints/backbone.pt'))

  def test_frozen_grid_and_record_counts(self):
    tasks = build_tasks(self.paths)
    self.assertEqual(len(tasks), NUM_SHARDS)
    self.assertEqual(sum(
      expected_shard_samples(index) for index in range(NUM_SHARDS)),
      NUM_SAMPLES)
    command = tasks[0].command
    modes_at = command.index('--modes')
    self.assertEqual(command[modes_at + 1:modes_at + 3], MODES)
    nfe_at = command.index('--nfe-budgets')
    self.assertEqual(
      command[nfe_at + 1:nfe_at + 5],
      tuple(str(value) for value in NFE_BUDGETS))
    self.assertIn(str(BASE_SEED), command)
    self.assertIn(str(BATCH_SIZE), command)

  def test_paths_are_isolated_from_primary_outputs(self):
    tasks = build_tasks(self.paths)
    for index, task in enumerate(tasks):
      self.assertEqual(task.shard_index, index)
      self.assertIn('reveal-policy-control-v2', str(task.output_dir))
      self.assertIn('reveal-policy-control-v2', str(task.log_path))
      self.assertNotIn('dynamic_dynamic/shard-', str(task.output_dir))

  def test_launch_plan_is_deterministic_and_path_bound(self):
    tasks = build_tasks(self.paths)
    self.assertEqual(launch_plan_sha256(tasks), launch_plan_sha256(tasks))
    changed = Paths(
      runner_repo=Path('/other-runner'),
      python=self.paths.python,
      experiment_root=self.paths.experiment_root,
      expansion_root=self.paths.expansion_root,
      backbone=self.paths.backbone)
    self.assertNotEqual(
      launch_plan_sha256(tasks), launch_plan_sha256(build_tasks(changed)))

  def test_invalid_shard_index_fails(self):
    with self.assertRaisesRegex(ValueError, 'outside'):
      expected_shard_samples(-1)
    with self.assertRaisesRegex(ValueError, 'outside'):
      expected_shard_samples(NUM_SHARDS)

  def test_python_path_preserves_virtual_environment_symlink(self):
    path = Path('/mnt/contextual-forest/venv/bin/python')
    self.assertEqual(_absolute_path_without_resolving_symlinks(path), path)


if __name__ == '__main__':
  unittest.main()
