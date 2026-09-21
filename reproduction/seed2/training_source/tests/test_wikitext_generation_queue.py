import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.run_wikitext_generation_queue import (
  ADAPTER_ORIGIN_EVIDENCE_SHA256,
  BACKBONE_SHA256,
  IMMUTABLE_PROTOCOL_SHA256,
  IMMUTABLE_RUNNER_GIT_SHA,
  IMMUTABLE_RUNNER_SHA256,
  PROMPT_MANIFEST_SHA256,
  WIKITEXT_LAUNCH_PLAN_SHA256,
  ProcessSnapshot,
  QueueController,
  QueueFailure,
  QueuePaths,
  QueuePlan,
  _arm_specs,
  _default_completion_validator,
  _normalize_runner_arguments,
  frozen_queue_plan,
  launch_plan_sha256,
)


def _paths(root: Path) -> QueuePaths:
  return QueuePaths(
    runner_repo=root / 'runner',
    python=root / 'venv' / 'bin' / 'python',
    experiment_root=root / 'generation',
    expansion_root=root / 'expansion',
    backbone=root / 'backbone.pt',
  )


def _write_complete(task) -> None:
  task.output_dir.mkdir(parents=True, exist_ok=True)
  (task.output_dir / 'manifest.json').write_text(json.dumps({
    'command': [task.command[1], *task.command[2:]],
  }))
  for name in ('samples.jsonl', 'summary.json', 'resolved_config.yaml'):
    (task.output_dir / name).write_text('fixture\n')
  task.log_path.parent.mkdir(parents=True, exist_ok=True)
  task.log_path.touch(exist_ok=True)


def _snapshot(task, paths, *, start_ticks=123, state='R'):
  return ProcessSnapshot(
    pid=task.adopted_pid or 999,
    start_ticks=start_ticks,
    state=state,
    cwd=paths.runner_repo,
    command=task.command,
  )


class FrozenPlanTest(unittest.TestCase):

  def test_exact_live_pid_and_32_shard_plan(self):
    with tempfile.TemporaryDirectory() as directory:
      plan = frozen_queue_plan(_paths(Path(directory)))

    self.assertEqual(
      [(task.shard_index, task.adopted_pid) for task in plan.initial_tasks],
      [(0, 5226), (1, 5958)])
    self.assertEqual(
      [(task.arm.name, task.shard_index) for task in plan.phases[0]],
      [('dynamic_dynamic', index) for index in range(2, 16)])
    self.assertEqual(
      [(task.arm.name, task.shard_index) for task in plan.phases[1]],
      [('static_static', index) for index in range(16)])
    dynamic_command = plan.initial_tasks[0].command
    static_command = plan.phases[1][0].command
    self.assertIn(IMMUTABLE_RUNNER_GIT_SHA[:7], '09f89c0')
    self.assertEqual(
      _normalize_runner_arguments(dynamic_command[2:])['--modes'],
      ('factorized', 'structured_marginal', 'structured_joint'))
    self.assertEqual(
      _normalize_runner_arguments(static_command[2:])['--modes'],
      ('structured_joint',))
    self.assertEqual(
      _normalize_runner_arguments(dynamic_command[2:])['--nfe-budgets'],
      ('8', '16', '32', '64'))
    self.assertEqual(
      launch_plan_sha256(plan), WIKITEXT_LAUNCH_PLAN_SHA256)

  def test_command_parser_rejects_dirty_unknown_and_duplicate_arguments(self):
    with self.assertRaisesRegex(QueueFailure, 'allow-dirty'):
      _normalize_runner_arguments(['--allow-dirty'])
    with self.assertRaisesRegex(QueueFailure, 'unknown runner argument'):
      _normalize_runner_arguments(['--invented', 'true'])
    with self.assertRaisesRegex(QueueFailure, 'repeats --device'):
      _normalize_runner_arguments([
        '--device', 'cuda', '--device', 'cpu'])
    with self.assertRaisesRegex(QueueFailure, 'repeats a value'):
      _normalize_runner_arguments([
        '--override', 'trainer.devices=1',
        '--override', 'trainer.devices=1'])
    with self.assertRaisesRegex(QueueFailure, 'repeats a Hydra override key'):
      _normalize_runner_arguments([
        '--override', 'trainer.devices=1',
        '--override', 'trainer.devices=2'])


class QueueControllerTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.paths = _paths(self.root)
    self.full_plan = frozen_queue_plan(self.paths)

  def tearDown(self):
    self.temporary.cleanup()

  def _controller(self, plan=None, **kwargs):
    controller = QueueController(
      plan or self.full_plan,
      poll_seconds=0.001,
      completion_validator=kwargs.pop(
        'completion_validator', lambda task, paths: None),
      **kwargs)
    controller.verify_environment = lambda: None
    return controller

  def test_nonempty_or_empty_incomplete_directory_is_never_reused(self):
    task = self.full_plan.phases[0][0]
    controller = self._controller()
    task.output_dir.mkdir(parents=True)
    with self.assertRaisesRegex(QueueFailure, 'incomplete directory'):
      controller._pending_or_complete(task)
    (task.output_dir / 'partial.txt').write_text('preserve me')
    with self.assertRaisesRegex(QueueFailure, 'incomplete directory'):
      controller._pending_or_complete(task)
    self.assertEqual(
      (task.output_dir / 'partial.txt').read_text(), 'preserve me')

  def test_existing_log_without_complete_shard_is_never_overwritten(self):
    task = self.full_plan.phases[0][0]
    task.log_path.parent.mkdir(parents=True)
    task.log_path.write_text('old log\n')
    controller = self._controller()
    with self.assertRaisesRegex(QueueFailure, 'log already exists'):
      controller._pending_or_complete(task)
    self.assertEqual(task.log_path.read_text(), 'old log\n')

  def test_existing_complete_shard_is_revalidated_and_skipped(self):
    tasks = self.full_plan.phases[0][:2]
    validated = []
    for task in tasks:
      _write_complete(task)
    controller = self._controller(
      completion_validator=lambda task, paths: validated.append(task.task_id),
      popen_factory=mock.Mock(side_effect=AssertionError('must not launch')))

    controller._launch_pair(tasks)

    self.assertEqual(validated, [task.task_id for task in tasks])
    controller.popen_factory.assert_not_called()

  def test_manifest_command_substitution_fails_after_crypto_validator(self):
    task = self.full_plan.phases[0][0]
    _write_complete(task)
    manifest_path = task.output_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    position = manifest['command'].index('--shard-index') + 1
    manifest['command'][position] = '15'
    manifest_path.write_text(json.dumps(manifest))
    controller = self._controller()

    with self.assertRaisesRegex(QueueFailure, 'manifest command differs'):
      controller._validate_completed(task)

  def test_adopted_pid_is_waited_then_completed_directory_validated(self):
    task = self.full_plan.initial_tasks[0]
    task.output_dir.mkdir(parents=True)
    task.log_path.parent.mkdir(parents=True)
    task.log_path.write_text('current process log\n')
    snapshots = [_snapshot(task, self.paths), None]
    validated = []

    def read_process(pid):
      return snapshots.pop(0)

    def sleep(_):
      _write_complete(task)

    controller = self._controller(
      process_reader=read_process,
      sleep=sleep,
      completion_validator=lambda item, paths: validated.append(item.task_id))
    controller._wait_initial_task(task)

    self.assertEqual(validated, [task.task_id])

  def test_vanished_initial_pid_without_manifest_fails_closed(self):
    task = self.full_plan.initial_tasks[0]
    task.output_dir.mkdir(parents=True)
    task.log_path.parent.mkdir(parents=True)
    task.log_path.write_text('current process log\n')
    controller = self._controller(process_reader=lambda pid: None)

    with self.assertRaisesRegex(QueueFailure, 'is gone'):
      controller._wait_initial_task(task)

  def test_pid_reuse_or_command_drift_fails_closed(self):
    task = self.full_plan.initial_tasks[0]
    task.output_dir.mkdir(parents=True)
    task.log_path.parent.mkdir(parents=True)
    task.log_path.write_text('current process log\n')
    snapshots = [
      _snapshot(task, self.paths, start_ticks=123),
      _snapshot(task, self.paths, start_ticks=124),
    ]
    controller = self._controller(
      process_reader=lambda pid: snapshots.pop(0), sleep=lambda _: None)

    with self.assertRaisesRegex(QueueFailure, 'was reused'):
      controller._wait_initial_task(task)

  def test_two_worker_pairs_validate_before_static_phase(self):
    initial = self.full_plan.initial_tasks
    dynamic = self.full_plan.phases[0][:2]
    static = self.full_plan.phases[1][:2]
    plan = QueuePlan(
      paths=self.paths,
      initial_tasks=initial,
      phases=(dynamic, static),
    )
    for task in initial:
      _write_complete(task)
    task_by_output = {str(task.output_dir): task for task in (*dynamic, *static)}
    events = []
    next_pid = iter(range(1000, 1004))

    class FakeProcess:
      def __init__(self, task):
        self.task = task
        self.pid = next(next_pid)

      def wait(self):
        events.append(('wait', self.task.task_id))
        _write_complete(self.task)
        return 0

    def popen(command, **kwargs):
      args = _normalize_runner_arguments(command[2:])
      task = task_by_output[args['--output-dir']]
      events.append(('launch', task.task_id))
      return FakeProcess(task)

    def validate(task, paths):
      events.append(('validate', task.task_id))

    controller = self._controller(
      plan,
      completion_validator=validate,
      popen_factory=popen)
    controller.run()

    first_static_launch = next(
      index for index, event in enumerate(events)
      if event == ('launch', static[0].task_id))
    for task in dynamic:
      self.assertLess(
        events.index(('validate', task.task_id)), first_static_launch)
    self.assertEqual(
      [event for event in events if event[0] == 'launch'],
      [('launch', task.task_id) for task in (*dynamic, *static)])
    completion_path = self.paths.experiment_root / 'wikitext' / 'queue-complete.json'
    completion = json.loads(completion_path.read_text())
    self.assertEqual(completion['dataset_slug'], 'wikitext')
    self.assertEqual(completion['logical_dataset'], 'wikitext103-pinned')
    self.assertEqual(completion['num_tasks'], 6)
    self.assertEqual(len(completion['tasks']), 6)
    self.assertFalse(
      (self.paths.experiment_root / 'generation-queue.lock').exists())

  def test_failed_child_drains_and_validates_peer_then_stops(self):
    dynamic = self.full_plan.phases[0][:2]
    static = self.full_plan.phases[1][:2]
    task_by_output = {str(task.output_dir): task for task in dynamic}
    events = []

    class FakeProcess:
      def __init__(self, task, returncode):
        self.task = task
        self.returncode = returncode
        self.pid = 2000 + task.shard_index

      def wait(self):
        events.append(('wait', self.task.task_id))
        if self.returncode == 0:
          _write_complete(self.task)
        return self.returncode

    def popen(command, **kwargs):
      task = task_by_output[
        _normalize_runner_arguments(command[2:])['--output-dir']]
      events.append(('launch', task.task_id))
      return FakeProcess(task, 1 if task is dynamic[0] else 0)

    controller = self._controller(
      completion_validator=lambda task, paths: events.append(
        ('validate', task.task_id)),
      popen_factory=popen)

    with self.assertRaisesRegex(QueueFailure, 'exited with status 1'):
      controller._launch_pair(dynamic)

    self.assertIn(('wait', dynamic[1].task_id), events)
    self.assertIn(('validate', dynamic[1].task_id), events)
    self.assertFalse(static[0].output_dir.exists())
    self.assertTrue(dynamic[0].output_dir.is_dir())
    self.assertTrue(dynamic[0].log_path.is_file())

  def test_validation_failure_halts_before_next_pair(self):
    dynamic = self.full_plan.phases[0][:4]
    task_by_output = {str(task.output_dir): task for task in dynamic}
    launched = []

    class FakeProcess:
      def __init__(self, task):
        self.task = task
        self.pid = 3000 + task.shard_index

      def wait(self):
        _write_complete(self.task)
        return 0

    def popen(command, **kwargs):
      task = task_by_output[
        _normalize_runner_arguments(command[2:])['--output-dir']]
      launched.append(task.task_id)
      return FakeProcess(task)

    def validate(task, paths):
      if task is dynamic[0]:
        raise ValueError('forged manifest')

    controller = self._controller(
      completion_validator=validate, popen_factory=popen)
    with self.assertRaisesRegex(QueueFailure, 'validation failed'):
      controller._launch_pair(dynamic[:2])

    self.assertEqual(launched, [task.task_id for task in dynamic[:2]])
    self.assertFalse(dynamic[2].output_dir.exists())

  def test_log_reservation_race_never_overwrites_competitor(self):
    task = self.full_plan.phases[0][0]
    controller = self._controller()
    task.log_path.parent.mkdir(parents=True)
    task.log_path.write_text('competitor\n')

    with self.assertRaisesRegex(QueueFailure, 'log appeared'):
      controller._reserve_task(task)

    self.assertEqual(task.log_path.read_text(), 'competitor\n')
    self.assertTrue(task.output_dir.is_dir())


class EnvironmentAndValidatorTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.paths = _paths(self.root)
    self.plan = frozen_queue_plan(self.paths)
    arms = _arm_specs(self.paths)
    files = [
      self.paths.runner_repo / 'scripts/run_generation_pilot.py',
      self.paths.runner_repo / 'configs/experiment'
      / 'contextual-forest-generation-paper-v1.yaml',
      self.paths.backbone,
      self.paths.experiment_root / 'adapter-pair-origin.json',
      self.paths.experiment_root / 'prompts'
      / 'wikitext-span32.jsonl.manifest.json',
      self.paths.experiment_root / 'prompts' / 'wikitext-span32.jsonl',
      *(arm.adapter for arm in arms.values()),
      *(arm.adapter_manifest for arm in arms.values()),
      self.paths.python,
    ]
    for path in files:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text('fixture')
    self.paths.python.chmod(0o755)

  def tearDown(self):
    self.temporary.cleanup()

  def _hashes(self):
    arms = _arm_specs(self.paths)
    return {
      self.paths.runner_repo / 'scripts/run_generation_pilot.py':
        IMMUTABLE_RUNNER_SHA256,
      self.paths.runner_repo / 'configs/experiment'
      / 'contextual-forest-generation-paper-v1.yaml':
        IMMUTABLE_PROTOCOL_SHA256,
      self.paths.backbone: BACKBONE_SHA256,
      self.paths.experiment_root / 'adapter-pair-origin.json':
        ADAPTER_ORIGIN_EVIDENCE_SHA256,
      self.paths.experiment_root / 'prompts'
      / 'wikitext-span32.jsonl.manifest.json': PROMPT_MANIFEST_SHA256,
      **{arm.adapter: arm.adapter_sha256 for arm in arms.values()},
      **{
        arm.adapter_manifest: arm.adapter_manifest_sha256
        for arm in arms.values()},
    }

  def test_environment_requires_exact_clean_09f_checkout(self):
    controller = QueueController(
      self.plan, completion_validator=lambda task, paths: None)
    hashes = self._hashes()
    with mock.patch(
        'scripts.run_wikitext_generation_queue._sha256_file',
        side_effect=lambda path: hashes[path]), mock.patch(
        'scripts.run_wikitext_generation_queue.subprocess.check_output',
        side_effect=[IMMUTABLE_RUNNER_GIT_SHA + '\n', '']):
      controller.verify_environment()

    with mock.patch(
        'scripts.run_wikitext_generation_queue._sha256_file',
        side_effect=lambda path: hashes[path]), mock.patch(
        'scripts.run_wikitext_generation_queue.subprocess.check_output',
        side_effect=[IMMUTABLE_RUNNER_GIT_SHA + '\n', '?? injected.py\n']):
      with self.assertRaisesRegex(QueueFailure, 'checkout is dirty'):
        controller.verify_environment()

  def test_default_validator_uses_immutable_checkout_and_logs_failure(self):
    task = self.plan.phases[0][0]
    task.log_path.parent.mkdir(parents=True)
    task.log_path.write_text('runner log\n')
    result = subprocess.CompletedProcess(
      args=[], returncode=7, stdout='validator stdout\n',
      stderr='validator stderr\n')
    with mock.patch(
        'scripts.run_wikitext_generation_queue.subprocess.run',
        return_value=result) as run:
      with self.assertRaisesRegex(QueueFailure, 'failed cryptographic'):
        _default_completion_validator(task, self.paths)

    self.assertEqual(run.call_args.kwargs['cwd'], self.paths.runner_repo)
    self.assertNotIn('PYTHONPATH', run.call_args.kwargs['env'])
    log = task.log_path.read_text()
    self.assertIn('returncode=7', log)
    self.assertIn('validator stdout', log)
    self.assertIn('validator stderr', log)


if __name__ == '__main__':
  unittest.main()
