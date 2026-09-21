import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.run_cross_domain_generation_queue import (
  DATASETS,
  CROSS_DOMAIN_LAUNCH_PLAN_SHA256S,
  CrossDomainQueueController,
  _completion_validator,
  frozen_cross_domain_plan,
  launch_plan_sha256,
)
from scripts.run_wikitext_generation_queue import (
  ADAPTER_ORIGIN_EVIDENCE_SHA256,
  BACKBONE_SHA256,
  IMMUTABLE_PROTOCOL_SHA256,
  IMMUTABLE_RUNNER_GIT_SHA,
  IMMUTABLE_RUNNER_SHA256,
  QueueFailure,
  QueuePlan,
  _require_exact_launch_plan,
  QueuePaths,
  _arm_specs,
  _normalize_runner_arguments,
)


def _paths(root: Path) -> QueuePaths:
  return QueuePaths(
    runner_repo=root / 'runner',
    python=root / 'venv' / 'bin' / 'python',
    experiment_root=root / 'generation',
    expansion_root=root / 'expansion',
    backbone=root / 'backbone.pt',
  )


class FrozenCrossDomainPlanTest(unittest.TestCase):

  def test_direct_script_invocation_loads_shared_controller(self):
    result = subprocess.run(
      [sys.executable, 'scripts/run_cross_domain_generation_queue.py', '--help'],
      text=True, capture_output=True, check=False)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('--wikitext-gate-sha256', result.stdout)

  def test_exact_arxiv_and_pubmed_grids(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = _paths(Path(directory))
      for slug, base_seed in [('arxiv', '92001'), ('pubmed', '93001')]:
        plan = frozen_cross_domain_plan(slug, paths)
        self.assertEqual(plan.initial_tasks, ())
        self.assertEqual(len(plan.phases), 2)
        self.assertEqual([len(phase) for phase in plan.phases], [16, 16])
        self.assertEqual(
          [task.task_id for task in plan.phases[0]][:2],
          [
            f'{slug}-dynamic_dynamic-shard-00',
            f'{slug}-dynamic_dynamic-shard-01',
          ])
        dynamic = _normalize_runner_arguments(plan.phases[0][0].command[2:])
        static = _normalize_runner_arguments(plan.phases[1][0].command[2:])
        self.assertEqual(dynamic['--base-seed'], base_seed)
        self.assertEqual(dynamic['--num-samples'], '1024')
        self.assertEqual(dynamic['--modes'], (
          'factorized', 'structured_marginal', 'structured_joint'))
        self.assertEqual(static['--modes'], ('structured_joint',))
        self.assertEqual(
          dynamic['--prompt-manifest-sha256'],
          DATASETS[slug].prompt_manifest_sha256)
        self.assertEqual(
          launch_plan_sha256(plan), CROSS_DOMAIN_LAUNCH_PLAN_SHA256S[slug])
        self.assertEqual(
          [(task.arm.name, task.shard_index) for task in plan.phases[0]],
          [('dynamic_dynamic', index) for index in range(16)])
        self.assertEqual(
          [(task.arm.name, task.shard_index) for task in plan.phases[1]],
          [('static_static', index) for index in range(16)])
        dynamic_records = sum(
          ((1024 - 1 - task.shard_index) // 16 + 1) * 3 * 4
          for task in plan.phases[0])
        static_records = sum(
          ((1024 - 1 - task.shard_index) // 16 + 1) * 1 * 4
          for task in plan.phases[1])
        self.assertEqual(dynamic_records, 12_288)
        self.assertEqual(static_records, 4_096)

  def test_unknown_dataset_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(QueueFailure, 'unsupported'):
        frozen_cross_domain_plan('invented', _paths(Path(directory)))


class CrossDomainEnvironmentTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.paths = _paths(self.root)
    self.dataset = DATASETS['arxiv']
    self.prompt = (
      self.paths.experiment_root / 'prompts' / self.dataset.prompt_name)
    self.prompt_manifest = Path(f'{self.prompt}.manifest.json')
    self.prompt_provenance = (
      self.paths.experiment_root / 'prompts'
      / self.dataset.prompt_provenance_relative_path)
    self.gate = self.paths.experiment_root / 'wikitext' / 'gate.json'
    self.gate_sha256 = 'a' * 64
    arms = _arm_specs(self.paths)
    files = [
      self.paths.runner_repo / 'scripts/run_generation_pilot.py',
      self.paths.runner_repo / 'configs/experiment'
      / 'contextual-forest-generation-paper-v1.yaml',
      self.paths.backbone,
      self.paths.experiment_root / 'adapter-pair-origin.json',
      self.prompt,
      self.prompt_provenance,
      self.paths.runner_repo / 'configs' / 'data'
      / f'{self.dataset.data_config}.yaml',
      self.gate,
      *(arm.adapter for arm in arms.values()),
      *(arm.adapter_manifest for arm in arms.values()),
      self.paths.python,
    ]
    for path in files:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text('fixture')
    self.paths.python.chmod(0o755)
    self.prompt_manifest.write_text(json.dumps({
      'schema_version': 2,
      'repository': {
        'git_sha': IMMUTABLE_RUNNER_GIT_SHA,
        'clean': True,
      },
      'data_config': {
        'name': self.dataset.data_config,
        'logical_validation_dataset': self.dataset.logical_dataset,
        'sha256': self.dataset.data_config_sha256,
      },
      'output': {
        'num_prompts': self.dataset.num_prompts,
        'sha256': self.dataset.prompt_sha256,
      },
      'policy': {
        'policy_id': 'document-local-contiguous-span-v1',
        'record_selection': 'first_n_in_pinned_validation_order',
        'boundary_policy': 'never_mask_first_or_last_token',
        'selection_seed': 31001,
        'sequence_length': 256,
        'span_length': 32,
      },
      'runtime_provenance': {
        'path': str(self.prompt_provenance.resolve()),
        'sha256': self.dataset.prompt_provenance_sha256,
      },
    }))

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
      self.prompt: self.dataset.prompt_sha256,
      self.prompt_manifest: self.dataset.prompt_manifest_sha256,
      self.prompt_provenance: self.dataset.prompt_provenance_sha256,
      self.paths.runner_repo / 'configs' / 'data'
      / f'{self.dataset.data_config}.yaml': self.dataset.data_config_sha256,
      **{arm.adapter: arm.adapter_sha256 for arm in arms.values()},
      **{
        arm.adapter_manifest: arm.adapter_manifest_sha256
        for arm in arms.values()},
    }

  def _controller(self):
    return CrossDomainQueueController(
      self.dataset, self.paths, reviewed_gate_sha256=self.gate_sha256,
      reviewed_gate_path=self.gate)

  def _verify(self, controller=None):
    controller = controller or self._controller()
    hashes = self._hashes()
    with mock.patch(
        'scripts.run_cross_domain_generation_queue._sha256_file',
        side_effect=lambda path: hashes[path]), mock.patch(
        'scripts.run_cross_domain_generation_queue.subprocess.check_output',
        side_effect=[IMMUTABLE_RUNNER_GIT_SHA + '\n', '']), mock.patch(
        'scripts.run_cross_domain_generation_queue.'
        'validate_reviewed_wikitext_gate',
        return_value={
          'path': str(self.gate), 'sha256': self.gate_sha256,
          'decision': 'proceed'}):
      controller.verify_environment()
    return controller

  def test_environment_authenticates_prompt_and_clean_runner(self):
    self._verify()

  def test_manifest_dataset_substitution_is_rejected(self):
    manifest = json.loads(self.prompt_manifest.read_text())
    manifest['data_config']['logical_validation_dataset'] = 'wrong-dataset'
    self.prompt_manifest.write_text(json.dumps(manifest))
    with self.assertRaisesRegex(QueueFailure, 'logical_validation_dataset'):
      self._verify()

  def test_missing_referenced_provenance_is_rejected_prelaunch(self):
    self.prompt_provenance.unlink()
    with self.assertRaisesRegex(QueueFailure, 'required immutable input'):
      self._verify()

  def test_tampered_referenced_provenance_is_rejected_prelaunch(self):
    controller = self._controller()
    hashes = self._hashes()
    hashes[self.prompt_provenance] = '0' * 64
    with mock.patch(
        'scripts.run_cross_domain_generation_queue._sha256_file',
        side_effect=lambda path: hashes[path]):
      with self.assertRaisesRegex(QueueFailure, 'SHA256 mismatch'):
        controller.verify_environment()

  def test_wrong_or_unreviewed_gate_is_rejected_prelaunch(self):
    controller = self._controller()
    controller.popen_factory = mock.Mock()
    hashes = self._hashes()
    with mock.patch(
        'scripts.run_cross_domain_generation_queue._sha256_file',
        side_effect=lambda path: hashes[path]), mock.patch(
        'scripts.run_cross_domain_generation_queue.subprocess.check_output',
        side_effect=[IMMUTABLE_RUNNER_GIT_SHA + '\n', '']), mock.patch(
        'scripts.run_cross_domain_generation_queue.'
        'validate_reviewed_wikitext_gate',
        side_effect=ValueError('decision is hold')):
      with self.assertRaisesRegex(QueueFailure, 'launch gate failed'):
        controller._run_locked()
    controller.popen_factory.assert_not_called()

  def test_every_command_token_and_path_is_bound_by_launch_plan(self):
    original = self._controller()
    task = original.plan.phases[0][0]
    for position in range(len(task.command)):
      command = list(task.command)
      command[position] = f'{command[position]}-drift'
      changed = replace(task, command=tuple(command))
      phases = (
        (changed, *original.plan.phases[0][1:]),
        original.plan.phases[1],
      )
      controller = self._controller()
      controller.plan = QueuePlan(
        paths=self.paths, initial_tasks=(), phases=phases)
      with self.assertRaisesRegex(QueueFailure, 'launch plan differs'):
        _require_exact_launch_plan(
          controller.plan, CROSS_DOMAIN_LAUNCH_PLAN_SHA256S['arxiv'])

    for field in ('output_dir', 'log_path'):
      changed = replace(
        task, **{field: getattr(task, field).with_name('drifted')})
      controller = self._controller()
      controller.plan = QueuePlan(
        paths=self.paths,
        initial_tasks=(),
        phases=((changed, *controller.plan.phases[0][1:]),
                controller.plan.phases[1]))
      with self.assertRaisesRegex(QueueFailure, 'launch plan differs'):
        _require_exact_launch_plan(
          controller.plan, CROSS_DOMAIN_LAUNCH_PLAN_SHA256S['arxiv'])

  def test_duplicate_extra_or_swapped_tasks_fail_exact_plan(self):
    for mutation in ('duplicate', 'swap'):
      controller = self._controller()
      dynamic = controller.plan.phases[0]
      if mutation == 'duplicate':
        dynamic = (*dynamic, dynamic[-1])
      else:
        dynamic = (dynamic[1], dynamic[0], *dynamic[2:])
      controller.plan = QueuePlan(
        paths=self.paths, initial_tasks=(),
        phases=(dynamic, controller.plan.phases[1]))
      with self.assertRaisesRegex(QueueFailure, 'launch plan differs'):
        _require_exact_launch_plan(
          controller.plan, CROSS_DOMAIN_LAUNCH_PLAN_SHA256S['arxiv'])

  def test_incomplete_cross_domain_shard_is_preserved(self):
    controller = CrossDomainQueueController(
      self.dataset, self.paths,
      reviewed_gate_sha256=self.gate_sha256,
      reviewed_gate_path=self.gate)
    controller.verify_environment = lambda: None
    task = controller.plan.phases[0][0]
    task.output_dir.mkdir(parents=True)
    (task.output_dir / 'partial.txt').write_text('preserve')
    with self.assertRaisesRegex(QueueFailure, 'incomplete directory'):
      controller._pending_or_complete(task)
    self.assertEqual(
      (task.output_dir / 'partial.txt').read_text(), 'preserve')

  def test_completion_validator_binds_dataset_and_logs_failure(self):
    controller = CrossDomainQueueController(
      self.dataset, self.paths,
      reviewed_gate_sha256=self.gate_sha256,
      reviewed_gate_path=self.gate)
    task = controller.plan.phases[0][0]
    task.log_path.parent.mkdir(parents=True)
    task.log_path.write_text('runner log\n')
    result = subprocess.CompletedProcess(
      args=[], returncode=9, stdout='bad manifest\n', stderr='')
    with mock.patch(
        'scripts.run_cross_domain_generation_queue.subprocess.run',
        return_value=result) as run:
      with self.assertRaisesRegex(QueueFailure, 'failed cryptographic'):
        _completion_validator(task, self.paths, self.dataset)
    arguments = run.call_args.args[0]
    self.assertIn(self.dataset.logical_dataset, arguments)
    self.assertIn(self.dataset.prompt_sha256, arguments)
    self.assertIn('returncode=9', task.log_path.read_text())


if __name__ == '__main__':
  unittest.main()
