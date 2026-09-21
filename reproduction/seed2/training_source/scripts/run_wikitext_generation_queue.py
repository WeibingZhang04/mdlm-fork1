#!/usr/bin/env python3
"""Run the frozen two-worker WikiText paper-generation shard queue.

This controller is deliberately specific to the active K=128 paper run.  It
adopts the two already-running dynamic shards, then runs strict two-job pairs
for dynamic shards 02--15 followed by static shards 00--15.  A task is released
only after the previous pair has finished and every successful directory has
passed the repository's cryptographic shard and paper-protocol validators.

The controller must live outside the immutable runner checkout when deployed.
Copying it into that checkout would dirty the tree and the runner will refuse
to start.  It never removes, reuses, or overwrites an output directory or log.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.generation_queue_artifacts import (  # noqa: E402
  SharedQueueLock,
  atomic_write_new,
  canonical_sha256,
  load_strict_json,
  sha256_file,
)


IMMUTABLE_RUNNER_GIT_SHA = '09f89c00bbf8c65f679cd40b92609754608817b8'
IMMUTABLE_RUNNER_SHA256 = (
  '95b20ab4d7cba502f79aeaf7ce069b994e64d49167c45fdb445c874e538619ab')
IMMUTABLE_PROTOCOL_SHA256 = (
  '8b48305568495434836b39f1770342e1b9366b49746e45f7af85cf4233d0b837')

BACKBONE_SHA256 = (
  '1b7c724d0228b1a2c825185c96642ffd706bd828b237f84512f0e1c7b5765573')
ADAPTER_ORIGIN_EVIDENCE_SHA256 = (
  '8a3580f7d40c0139f6e57603917ecf923a9b62d9ef878c284e170bce280f8455')
PROMPT_MANIFEST_SHA256 = (
  'e6f9cc313b6296cbb7692c450e1d58ef1ccf223d13eff6507b6d509b717015b8')

GLOBAL_NUM_SAMPLES = 788
NUM_SHARDS = 16
BASE_SEED = 91001
SEQUENCE_LENGTH = 256
BATCH_SIZE = 8
NFE_BUDGETS = (8, 16, 32, 64)
CANDIDATE_TOP_K = 128
LAUNCH_PLAN_SCHEMA_VERSION = 1
LAUNCH_PLAN_POLICY_ID = 'frozen-generation-queue-launch-plan-v1'
# Filled from the path-normalized, complete plan payload.  Unlike a comparison
# against a task reconstructed by the same mutable object, this commitment
# catches changes to ordering, paths, interpreter/script tokens, and every argv
# value before any child is reserved or launched.
WIKITEXT_LAUNCH_PLAN_SHA256 = (
  '15b75cca776085608e9aa087d040544c598b793dd661bb17aa8086059d017dce')


@dataclass(frozen=True)
class QueuePaths:
  """Operational paths; the default instance is the exact active run."""

  runner_repo: Path
  python: Path
  experiment_root: Path
  expansion_root: Path
  backbone: Path


DEFAULT_PATHS = QueuePaths(
  runner_repo=Path('/mnt/contextual-forest/mdlm-paper-09f89c0-v3'),
  python=Path('/mnt/contextual-forest/venv/bin/python'),
  experiment_root=Path(
    '/mnt/contextual-forest/experiments/'
    'contextual-forest-generation-paper-v1'),
  expansion_root=Path(
    '/mnt/contextual-forest/experiments/'
    'contextual-forest-expansion-v1'),
  backbone=Path('/mnt/contextual-forest/checkpoints/mdlm-owt-backbone.pt'),
)


@dataclass(frozen=True)
class ArmSpec:
  name: str
  adapter: Path
  adapter_sha256: str
  adapter_manifest: Path
  adapter_manifest_sha256: str
  modes: tuple[str, ...]
  topology_mode: str
  factor_mode: str
  topology_weight: str


@dataclass(frozen=True)
class ShardTask:
  arm: ArmSpec
  shard_index: int
  output_dir: Path
  log_path: Path
  command: tuple[str, ...]
  adopted_pid: int | None = None
  dataset_slug: str = 'wikitext'

  @property
  def task_id(self) -> str:
    return f'{self.dataset_slug}-{self.arm.name}-shard-{self.shard_index:02d}'


@dataclass(frozen=True)
class QueuePlan:
  paths: QueuePaths
  initial_tasks: tuple[ShardTask, ...]
  phases: tuple[tuple[ShardTask, ...], ...]


@dataclass(frozen=True)
class ProcessSnapshot:
  pid: int
  start_ticks: int
  state: str
  cwd: Path
  command: tuple[str, ...]


class QueueFailure(RuntimeError):
  """A fail-closed queue invariant was violated."""


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _emit(event: str, **fields: object) -> None:
  print(json.dumps({'event': event, **fields}, sort_keys=True), flush=True)


def _arm_specs(paths: QueuePaths) -> dict[str, ArmSpec]:
  runs = paths.expansion_root / 'runs'
  dynamic_root = (
    runs / 'export--dynamic_dynamic--s001--k128'
    / 'attempts' / 'attempt-0001')
  static_root = (
    runs / 'export--static_static--s001--k128'
    / 'attempts' / 'attempt-0001')
  return {
    'dynamic_dynamic': ArmSpec(
      name='dynamic_dynamic',
      adapter=dynamic_root / 'adapter.safetensors',
      adapter_sha256=(
        '037817d874af0c8c60b1a8eaf5bab3506fc07ca70234911498445f7b8c12c769'),
      adapter_manifest=dynamic_root / 'adapter-manifest.json',
      adapter_manifest_sha256=(
        'b98a563bbc52ab501197956d424e4ab496b51841268c0556fbc9b7e134e0a61f'),
      modes=('factorized', 'structured_marginal', 'structured_joint'),
      topology_mode='dynamic',
      factor_mode='dynamic',
      topology_weight='0.1',
    ),
    'static_static': ArmSpec(
      name='static_static',
      adapter=static_root / 'adapter.safetensors',
      adapter_sha256=(
        '3d7169686ccddca0819d8f0b929372597f5efdbe3c8e341ca7c439a20e483206'),
      adapter_manifest=static_root / 'adapter-manifest.json',
      adapter_manifest_sha256=(
        '12942b50fd1c20f8e6a59e8e875bb339267201080801ad9ff6a9f08252522b81'),
      modes=('structured_joint',),
      topology_mode='fixed',
      factor_mode='fixed',
      topology_weight='0.0',
    ),
  }


def _task_command(
    paths: QueuePaths,
    arm: ArmSpec,
    shard_index: int,
    output_dir: Path,
) -> tuple[str, ...]:
  root = paths.experiment_root
  return (
    str(paths.python),
    'scripts/run_generation_pilot.py',
    '--backbone-checkpoint', str(paths.backbone),
    '--backbone-sha256', BACKBONE_SHA256,
    '--adapter', str(arm.adapter),
    '--adapter-sha256', arm.adapter_sha256,
    '--adapter-manifest', str(arm.adapter_manifest),
    '--adapter-manifest-sha256', arm.adapter_manifest_sha256,
    '--adapter-origin-evidence', str(root / 'adapter-pair-origin.json'),
    '--adapter-origin-evidence-sha256', ADAPTER_ORIGIN_EVIDENCE_SHA256,
    '--adapter-origin-arm', arm.name,
    '--output-dir', str(output_dir),
    '--prompt-jsonl', str(root / 'prompts' / 'wikitext-span32.jsonl'),
    '--prompt-manifest', str(
      root / 'prompts' / 'wikitext-span32.jsonl.manifest.json'),
    '--prompt-manifest-sha256', PROMPT_MANIFEST_SHA256,
    '--num-samples', str(GLOBAL_NUM_SAMPLES),
    '--num-shards', str(NUM_SHARDS),
    '--shard-index', str(shard_index),
    '--sequence-length', str(SEQUENCE_LENGTH),
    '--batch-size', str(BATCH_SIZE),
    '--base-seed', str(BASE_SEED),
    '--modes', *arm.modes,
    '--nfe-budgets', *(str(value) for value in NFE_BUDGETS),
    '--device', 'cuda',
    '--model-config', 'contextual-forest-small',
    '--data-config', 'eval_wikitext103_pinned',
    '--reference-lm', 'gpt2-large',
    '--reference-lm-revision',
    '32b71b12589c2f8d625668d2335a01cac3249519',
    '--reference-lm-device', 'cuda',
    '--reference-lm-batch-size', '8',
    '--reference-lm-max-length', '256',
    '--reference-lm-dtype', 'float32',
    '--override', 'model.structured_decoder.top_k=128',
    '--override', 'trainer.devices=1',
    '--override', 'loader.num_workers=8',
    '--override', (
      'checkpointing.save_dir='
      '/mnt/contextual-forest/mdlm-generation-paper-v1'),
    '--override',
    f'model.structured_decoder.topology_mode={arm.topology_mode}',
    '--override',
    f'model.structured_decoder.factor_mode={arm.factor_mode}',
    '--override', (
      'model.structured_decoder.training.topology_weight='
      f'{arm.topology_weight}'),
  )


def _make_task(
    paths: QueuePaths,
    arm: ArmSpec,
    shard_index: int,
    *,
    adopted_pid: int | None = None,
) -> ShardTask:
  output_dir = (
    paths.experiment_root / 'wikitext' / arm.name
    / f'shard-{shard_index:02d}')
  arm_log_name = 'dynamic' if arm.name == 'dynamic_dynamic' else 'static'
  log_path = (
    paths.experiment_root / 'logs'
    / f'wikitext-{arm_log_name}-shard{shard_index:02d}.log')
  return ShardTask(
    arm=arm,
    shard_index=shard_index,
    output_dir=output_dir,
    log_path=log_path,
    command=_task_command(paths, arm, shard_index, output_dir),
    adopted_pid=adopted_pid,
    dataset_slug='wikitext',
  )


def frozen_queue_plan(paths: QueuePaths = DEFAULT_PATHS) -> QueuePlan:
  """Return the exact two-worker queue requested for the active run."""
  arms = _arm_specs(paths)
  dynamic = arms['dynamic_dynamic']
  static = arms['static_static']
  initial = (
    _make_task(paths, dynamic, 0, adopted_pid=5226),
    _make_task(paths, dynamic, 1, adopted_pid=5958),
  )
  dynamic_pending = tuple(
    _make_task(paths, dynamic, index) for index in range(2, NUM_SHARDS))
  static_pending = tuple(
    _make_task(paths, static, index) for index in range(NUM_SHARDS))
  return QueuePlan(
    paths=paths,
    initial_tasks=initial,
    phases=(dynamic_pending, static_pending),
  )


def _plan_path_substitutions(paths: QueuePaths):
  raw_substitutions = (
    (paths.python, '<PYTHON>'),
    (paths.backbone, '<BACKBONE>'),
    (paths.runner_repo, '<RUNNER_REPO>'),
    (paths.experiment_root, '<EXPERIMENT_ROOT>'),
    (paths.expansion_root, '<EXPANSION_ROOT>'),
  )
  return tuple(
    (
      str(Path(raw_path).expanduser()),
      str(Path(raw_path).expanduser().resolve()),
      replacement,
    )
    for raw_path, replacement in raw_substitutions)


def _portable_plan_token(token: str, substitutions) -> str:
  for raw_value, resolved_value, replacement in substitutions:
    for value in (raw_value, resolved_value):
      if token == value:
        return replacement
      prefix = f'{value}{os.sep}'
      if token.startswith(prefix):
        return f'{replacement}/{token[len(prefix):]}'
  return token


def launch_plan_payload(plan: QueuePlan) -> dict[str, object]:
  """Return the complete path-portable plan committed before launch."""
  paths = plan.paths
  substitutions = _plan_path_substitutions(paths)

  def task_payload(task: ShardTask) -> dict[str, object]:
    return {
      'task_id': task.task_id,
      'dataset_slug': task.dataset_slug,
      'arm': task.arm.name,
      'shard_index': task.shard_index,
      'adopted_pid': task.adopted_pid,
      'output_dir': _portable_plan_token(
        str(task.output_dir.resolve()), substitutions),
      'log_path': _portable_plan_token(
        str(task.log_path.resolve()), substitutions),
      'command': [
        _portable_plan_token(token, substitutions) for token in task.command
      ],
    }

  return {
    'schema_version': LAUNCH_PLAN_SCHEMA_VERSION,
    'policy_id': LAUNCH_PLAN_POLICY_ID,
    'initial_tasks': [task_payload(task) for task in plan.initial_tasks],
    'phases': [
      [task_payload(task) for task in phase] for phase in plan.phases
    ],
  }


def launch_plan_sha256(plan: QueuePlan) -> str:
  return canonical_sha256(launch_plan_payload(plan))


def _require_exact_launch_plan(plan: QueuePlan, expected_sha256: str) -> str:
  observed = launch_plan_sha256(plan)
  if observed != expected_sha256:
    raise QueueFailure(
      'queue launch plan differs from its independent frozen commitment: '
      f'expected {expected_sha256}, found {observed}')
  return observed


_SCALAR_OPTIONS = {
  '--backbone-checkpoint', '--backbone-sha256', '--adapter',
  '--adapter-sha256', '--adapter-manifest', '--adapter-manifest-sha256',
  '--adapter-origin-evidence', '--adapter-origin-evidence-sha256',
  '--adapter-origin-arm', '--output-dir', '--prompt-jsonl',
  '--prompt-manifest', '--prompt-manifest-sha256', '--num-samples',
  '--num-shards', '--shard-index', '--sequence-length', '--batch-size',
  '--base-seed', '--device', '--model-config', '--data-config',
  '--reference-lm', '--reference-lm-revision', '--reference-lm-device',
  '--reference-lm-batch-size', '--reference-lm-max-length',
  '--reference-lm-dtype',
}
_LIST_OPTIONS = {'--modes', '--nfe-budgets'}
_REPEATED_OPTIONS = {'--override'}


def _normalize_runner_arguments(arguments: Sequence[str]) -> dict[str, object]:
  """Parse the frozen runner CLI without accepting unknown or dirty flags."""
  result: dict[str, object] = {}
  repeated: dict[str, list[str]] = {name: [] for name in _REPEATED_OPTIONS}
  index = 0
  while index < len(arguments):
    option = arguments[index]
    if option == '--allow-dirty':
      raise QueueFailure('the immutable queue never permits --allow-dirty')
    if option in _SCALAR_OPTIONS or option in _REPEATED_OPTIONS:
      if index + 1 >= len(arguments) or arguments[index + 1].startswith('--'):
        raise QueueFailure(f'{option} lacks one value')
      value = arguments[index + 1]
      if option in _REPEATED_OPTIONS:
        repeated[option].append(value)
      else:
        if option in result:
          raise QueueFailure(f'runner command repeats {option}')
        result[option] = value
      index += 2
      continue
    if option in _LIST_OPTIONS:
      if option in result:
        raise QueueFailure(f'runner command repeats {option}')
      values = []
      index += 1
      while index < len(arguments) and not arguments[index].startswith('--'):
        values.append(arguments[index])
        index += 1
      if not values:
        raise QueueFailure(f'{option} has no values')
      result[option] = tuple(values)
      continue
    raise QueueFailure(f'unknown runner argument {option!r}')
  for option, values in repeated.items():
    if values:
      if len(values) != len(set(values)):
        raise QueueFailure(f'runner command repeats a value for {option}')
      if option == '--override':
        keys = []
        for value in values:
          key, separator, _ = value.partition('=')
          if not separator or not key:
            raise QueueFailure(f'runner override lacks key=value: {value!r}')
          keys.append(key)
        if len(keys) != len(set(keys)):
          raise QueueFailure('runner command repeats a Hydra override key')
      # Override order is not semantically relevant because all frozen keys
      # are distinct.  Sorting permits adoption of the existing shell argv.
      result[option] = tuple(sorted(values))
  return result


def _runner_script_path(token: str, cwd: Path) -> Path:
  path = Path(token)
  return (path if path.is_absolute() else cwd / path).resolve()


def _validate_process_snapshot(
    snapshot: ProcessSnapshot,
    task: ShardTask,
    paths: QueuePaths,
) -> None:
  if snapshot.state == 'Z':
    raise QueueFailure(f'{task.task_id} adopted PID is a zombie')
  if snapshot.cwd.resolve() != paths.runner_repo.resolve():
    raise QueueFailure(
      f'{task.task_id} PID cwd differs from the immutable runner checkout')
  if len(snapshot.command) < 3:
    raise QueueFailure(f'{task.task_id} PID command is truncated')
  if Path(snapshot.command[0]).resolve() != paths.python.resolve():
    raise QueueFailure(f'{task.task_id} PID uses an unexpected Python')
  expected_script = (paths.runner_repo / 'scripts/run_generation_pilot.py') \
    .resolve()
  if _runner_script_path(snapshot.command[1], snapshot.cwd) != expected_script:
    raise QueueFailure(f'{task.task_id} PID uses an unexpected runner script')
  observed = _normalize_runner_arguments(snapshot.command[2:])
  expected = _normalize_runner_arguments(task.command[2:])
  if observed != expected:
    raise QueueFailure(
      f'{task.task_id} PID scientific argv differs from the frozen task')


def read_linux_process(pid: int) -> ProcessSnapshot | None:
  """Read one Linux process atomically enough to detect PID reuse."""
  proc = Path('/proc') / str(pid)
  try:
    raw_stat = (proc / 'stat').read_text()
    command = tuple(
      item.decode('utf-8', errors='strict')
      for item in (proc / 'cmdline').read_bytes().split(b'\0') if item)
    cwd = (proc / 'cwd').resolve(strict=True)
  except FileNotFoundError:
    return None
  except (OSError, UnicodeError) as error:
    raise QueueFailure(f'cannot inspect adopted PID {pid}: {error}') from error
  _, separator, tail = raw_stat.rpartition(') ')
  if not separator:
    raise QueueFailure(f'/proc/{pid}/stat has an invalid format')
  fields = tail.split()
  # tail starts at field 3 (state); starttime is Linux proc stat field 22.
  if len(fields) <= 19 or not fields[19].isdigit():
    raise QueueFailure(f'/proc/{pid}/stat lacks a valid start time')
  return ProcessSnapshot(
    pid=pid,
    start_ticks=int(fields[19]),
    state=fields[0],
    cwd=cwd,
    command=command,
  )


_VALIDATION_PROGRAM = r'''
import json
from pathlib import Path
import sys

from evaluation.generation_protocol import validate_generation_protocol
from evaluation.generation_shard_aggregation import load_generation_shard

path = Path(sys.argv[1]).resolve()
expected_arm = sys.argv[2]
expected_index = int(sys.argv[3])
expected_repo = sys.argv[4]
expected_adapter = str(Path(sys.argv[5]).resolve())
expected_adapter_sha = sys.argv[6]
expected_manifest = str(Path(sys.argv[7]).resolve())
expected_manifest_sha = sys.argv[8]
expected_origin_sha = sys.argv[9]

shard = load_generation_shard(path)
manifest = shard['manifest']
validate_generation_protocol(
    shard['config_path'], manifest, candidate_top_k=128,
    expected_control=expected_arm)
repository = manifest['repository']
if repository['git_sha'] != expected_repo or repository['dirty'] is not False:
  raise ValueError('shard repository is not the immutable clean 09f checkout')
pairing = manifest['pairing']
expected_pairing = {
  'shard_index': expected_index,
  'num_shards': 16,
  'global_num_samples': 788,
  'base_seed': 91001,
  'batch_size': 8,
  'sequence_length': 256,
}
for field, expected in expected_pairing.items():
  if pairing.get(field) != expected or type(pairing.get(field)) is not type(expected):
    raise ValueError(f'shard pairing {field} differs from the queue')
adapter = manifest['artifacts']['structured_adapter']
expected_adapter_fields = {
  'path': expected_adapter,
  'sha256': expected_adapter_sha,
  'manifest_path': expected_manifest,
  'manifest_sha256': expected_manifest_sha,
}
for field, expected in expected_adapter_fields.items():
  if adapter.get(field) != expected:
    raise ValueError(f'shard adapter {field} differs from the queue')
origin_file = manifest['adapter_origin_evidence']['evidence_file']
if origin_file.get('sha256') != expected_origin_sha:
  raise ValueError('shard adapter-origin file differs from the queue')
expected_modes = (
  ['factorized', 'structured_marginal', 'structured_joint']
  if expected_arm == 'dynamic_dynamic' else ['structured_joint'])
matrix = manifest['matrix']
if matrix.get('sampling_modes') != expected_modes:
  raise ValueError('shard modes differ from the queue')
if matrix.get('nfe_budgets') != [8, 16, 32, 64]:
  raise ValueError('shard NFE budgets differ from the queue')
expected_samples = (787 - expected_index) // 16 + 1
expected_records = expected_samples * len(expected_modes) * 4
if matrix.get('num_output_records') != expected_records:
  raise ValueError('shard record count differs from complete modulo coverage')
print(json.dumps({
  'manifest_sha256': shard['manifest_sha256'],
  'num_output_records': matrix['num_output_records'],
  'shard_index': expected_index,
  'arm': expected_arm,
}, sort_keys=True))
'''


def _append_log(path: Path, content: str) -> None:
  with path.open('a', encoding='utf-8') as handle:
    handle.write(content)
    if content and not content.endswith('\n'):
      handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())


def _default_completion_validator(task: ShardTask, paths: QueuePaths) -> None:
  environment = dict(os.environ)
  environment.pop('PYTHONPATH', None)
  result = subprocess.run(
    [
      str(paths.python), '-c', _VALIDATION_PROGRAM,
      str(task.output_dir), task.arm.name, str(task.shard_index),
      IMMUTABLE_RUNNER_GIT_SHA, str(task.arm.adapter),
      task.arm.adapter_sha256, str(task.arm.adapter_manifest),
      task.arm.adapter_manifest_sha256, ADAPTER_ORIGIN_EVIDENCE_SHA256,
    ],
    cwd=paths.runner_repo,
    env=environment,
    text=True,
    capture_output=True,
    check=False,
  )
  validation_log = (
    f'\n[queue-validator] returncode={result.returncode}\n'
    f'{result.stdout}{result.stderr}')
  _append_log(task.log_path, validation_log)
  if result.returncode != 0:
    raise QueueFailure(
      f'{task.task_id} failed cryptographic/protocol validation; '
      f'see {task.log_path}')


def _validate_manifest_command(task: ShardTask, paths: QueuePaths) -> None:
  try:
    manifest = json.loads((task.output_dir / 'manifest.json').read_text())
  except (OSError, json.JSONDecodeError) as error:
    raise QueueFailure(
      f'{task.task_id} manifest cannot be read after validation') from error
  command = manifest.get('command')
  if (not isinstance(command, list) or len(command) < 2
      or any(not isinstance(token, str) or not token for token in command)):
    raise QueueFailure(f'{task.task_id} manifest command is invalid')
  expected_script = (paths.runner_repo / 'scripts/run_generation_pilot.py') \
    .resolve()
  if _runner_script_path(command[0], paths.runner_repo) != expected_script:
    raise QueueFailure(
      f'{task.task_id} manifest names a different runner script')
  observed = _normalize_runner_arguments(command[1:])
  expected = _normalize_runner_arguments(task.command[2:])
  if observed != expected:
    raise QueueFailure(
      f'{task.task_id} manifest command differs from the frozen task')


CompletionValidator = Callable[[ShardTask, QueuePaths], None]
ProcessReader = Callable[[int], ProcessSnapshot | None]


class QueueController:
  """Fail-closed phase/barrier scheduler with two workers."""

  def __init__(
      self,
      plan: QueuePlan,
      *,
      poll_seconds: float = 30.0,
      completion_validator: CompletionValidator = _default_completion_validator,
      process_reader: ProcessReader = read_linux_process,
      sleep: Callable[[float], None] = time.sleep,
      popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
      expected_launch_plan_sha256: str = WIKITEXT_LAUNCH_PLAN_SHA256,
      logical_dataset: str = 'wikitext103-pinned',
      launch_authorization: dict[str, object] | None = None,
      recover_stale_lock: bool = False,
      lock_factory: Callable[..., SharedQueueLock] = SharedQueueLock,
  ) -> None:
    if poll_seconds <= 0:
      raise ValueError('poll_seconds must be positive')
    self.plan = plan
    self.poll_seconds = poll_seconds
    self.completion_validator = completion_validator
    self.process_reader = process_reader
    self.sleep = sleep
    self.popen_factory = popen_factory
    self.expected_launch_plan_sha256 = expected_launch_plan_sha256
    self.launch_plan_sha256 = launch_plan_sha256(plan)
    self.logical_dataset = logical_dataset
    self.launch_authorization = launch_authorization
    self.recover_stale_lock = recover_stale_lock
    self.lock_factory = lock_factory

  def _all_tasks(self) -> tuple[ShardTask, ...]:
    return (
      *self.plan.initial_tasks,
      *(task for phase in self.plan.phases for task in phase),
    )

  def _dataset_slug(self) -> str:
    slugs = {task.dataset_slug for task in self._all_tasks()}
    if len(slugs) != 1:
      raise QueueFailure(
        f'queue plan must carry exactly one dataset slug, found {sorted(slugs)}')
    return next(iter(slugs))

  def _completion_evidence_path(self) -> Path:
    return (
      self.plan.paths.experiment_root / self._dataset_slug()
      / 'queue-complete.json')

  def _completion_evidence_payload(self) -> dict[str, object]:
    tasks = []
    for task in self._all_tasks():
      manifest_path = task.output_dir / 'manifest.json'
      if not manifest_path.is_file():
        raise QueueFailure(
          f'{task.task_id} lacks a manifest for completion evidence')
      tasks.append({
        'task_id': task.task_id,
        'dataset_slug': task.dataset_slug,
        'arm': task.arm.name,
        'shard_index': task.shard_index,
        'output_dir': str(task.output_dir.resolve()),
        'manifest_sha256': sha256_file(manifest_path),
      })
    return {
      'schema_version': 1,
      'artifact': 'frozen_generation_queue_completion',
      'dataset_slug': self._dataset_slug(),
      'logical_dataset': self.logical_dataset,
      'immutable_runner_git_sha': IMMUTABLE_RUNNER_GIT_SHA,
      'launch_plan_sha256': self.launch_plan_sha256,
      'launch_authorization': self.launch_authorization,
      'num_tasks': len(tasks),
      'tasks': tasks,
      'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }

  def _write_or_validate_completion_evidence(self) -> tuple[Path, str]:
    path = self._completion_evidence_path()
    expected = self._completion_evidence_payload()
    if path.exists():
      try:
        observed = load_strict_json(path)
      except (OSError, TypeError, ValueError) as error:
        raise QueueFailure(
          f'cannot validate existing queue completion evidence {path}') \
          from error
      if not isinstance(observed, dict):
        raise QueueFailure('queue completion evidence must be a JSON object')
      observed_without_time = dict(observed)
      observed_time = observed_without_time.pop('completed_utc', None)
      expected_without_time = dict(expected)
      expected_without_time.pop('completed_utc')
      if (not isinstance(observed_time, str) or not observed_time
          or observed_without_time != expected_without_time):
        raise QueueFailure(
          'existing queue completion evidence differs from revalidated shards')
    else:
      atomic_write_new(
        path, json.dumps(expected, indent=2, sort_keys=True) + '\n')
    return path, sha256_file(path)

  def verify_environment(self) -> None:
    _require_exact_launch_plan(
      self.plan, self.expected_launch_plan_sha256)
    paths = self.plan.paths
    required_files = {
      paths.runner_repo / 'scripts/run_generation_pilot.py':
        IMMUTABLE_RUNNER_SHA256,
      paths.runner_repo / 'configs/experiment'
      / 'contextual-forest-generation-paper-v1.yaml':
        IMMUTABLE_PROTOCOL_SHA256,
      paths.backbone: BACKBONE_SHA256,
      paths.experiment_root / 'adapter-pair-origin.json':
        ADAPTER_ORIGIN_EVIDENCE_SHA256,
      paths.experiment_root / 'prompts'
      / 'wikitext-span32.jsonl.manifest.json': PROMPT_MANIFEST_SHA256,
    }
    for arm in _arm_specs(paths).values():
      required_files[arm.adapter] = arm.adapter_sha256
      required_files[arm.adapter_manifest] = arm.adapter_manifest_sha256
    for path, expected_sha in required_files.items():
      if not path.is_file():
        raise QueueFailure(f'required immutable input is missing: {path}')
      actual_sha = _sha256_file(path)
      if actual_sha != expected_sha:
        raise QueueFailure(
          f'immutable input SHA256 mismatch for {path}: '
          f'expected {expected_sha}, found {actual_sha}')
    prompt = (
      paths.experiment_root / 'prompts' / 'wikitext-span32.jsonl')
    if not prompt.is_file():
      raise QueueFailure(f'pinned prompt JSONL is missing: {prompt}')
    if not paths.python.is_file() or not os.access(paths.python, os.X_OK):
      raise QueueFailure(f'frozen Python is not executable: {paths.python}')
    try:
      revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=paths.runner_repo,
        text=True, stderr=subprocess.STDOUT).strip()
      status = subprocess.check_output(
        ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
        cwd=paths.runner_repo, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
      raise QueueFailure('cannot verify immutable runner repository') from error
    if revision != IMMUTABLE_RUNNER_GIT_SHA:
      raise QueueFailure(
        f'runner checkout is {revision}, expected {IMMUTABLE_RUNNER_GIT_SHA}')
    if status:
      raise QueueFailure('immutable runner checkout is dirty')
    observed_signature = (
      tuple(
        (task.dataset_slug, task.arm.name, task.shard_index, task.adopted_pid)
        for task in self.plan.initial_tasks),
      tuple(
        tuple(
          (task.dataset_slug, task.arm.name, task.shard_index, task.adopted_pid)
          for task in phase)
        for phase in self.plan.phases),
    )
    expected_signature = (
      (
        ('wikitext', 'dynamic_dynamic', 0, 5226),
        ('wikitext', 'dynamic_dynamic', 1, 5958),
      ),
      (
        tuple(
          ('wikitext', 'dynamic_dynamic', index, None)
          for index in range(2, NUM_SHARDS)),
        tuple(
          ('wikitext', 'static_static', index, None)
          for index in range(NUM_SHARDS)),
      ),
    )
    if observed_signature != expected_signature:
      raise QueueFailure(
        'queue task ordering or PID mapping differs from the exact grid')
    for task in self._all_tasks():
      _normalize_runner_arguments(task.command[2:])

  def _validate_completed(self, task: ShardTask) -> None:
    if not task.output_dir.is_dir():
      raise QueueFailure(
        f'{task.task_id} has no completed output directory')
    expected_names = {
      'manifest.json', 'samples.jsonl', 'summary.json', 'resolved_config.yaml'}
    actual_names = {path.name for path in task.output_dir.iterdir()}
    if actual_names != expected_names:
      raise QueueFailure(
        f'{task.task_id} output directory is not an exact atomic shard: '
        f'expected={sorted(expected_names)}, found={sorted(actual_names)}')
    if not task.log_path.is_file():
      raise QueueFailure(f'{task.task_id} is missing its per-task log')
    self.completion_validator(task, self.plan.paths)
    _validate_manifest_command(task, self.plan.paths)
    _emit(
      'generation_queue_task_validated', task_id=task.task_id,
      output_dir=str(task.output_dir), log=str(task.log_path))

  def _pending_or_complete(self, task: ShardTask) -> bool:
    """Return True if pending; validate and return False if completed."""
    if task.output_dir.exists():
      if not task.output_dir.is_dir():
        raise QueueFailure(
          f'{task.task_id} output path exists and is not a directory')
      if not (task.output_dir / 'manifest.json').is_file():
        raise QueueFailure(
          f'{task.task_id} has a pre-existing incomplete directory; '
          'preserving it and refusing reuse')
      self._validate_completed(task)
      return False
    if task.log_path.exists():
      raise QueueFailure(
        f'{task.task_id} log already exists without a complete shard; '
        'preserving it and refusing overwrite')
    return True

  def _wait_initial_task(self, task: ShardTask) -> None:
    if task.adopted_pid is None:
      raise QueueFailure(f'{task.task_id} has no adopted PID')
    if (task.output_dir / 'manifest.json').is_file():
      self._validate_completed(task)
      return
    if not task.log_path.is_file():
      raise QueueFailure(f'{task.task_id} adopted process log is missing')
    first = self.process_reader(task.adopted_pid)
    if first is None:
      raise QueueFailure(
        f'{task.task_id} PID {task.adopted_pid} is gone and the shard is '
        'not complete')
    _validate_process_snapshot(first, task, self.plan.paths)
    _emit(
      'generation_queue_adopted_pid', task_id=task.task_id,
      pid=task.adopted_pid, start_ticks=first.start_ticks)
    while True:
      self.sleep(self.poll_seconds)
      current = self.process_reader(task.adopted_pid)
      if current is None or current.state == 'Z':
        break
      if current.start_ticks != first.start_ticks:
        raise QueueFailure(
          f'{task.task_id} PID {task.adopted_pid} was reused')
      _validate_process_snapshot(current, task, self.plan.paths)
    self._validate_completed(task)

  def _reserve_task(self, task: ShardTask):
    task.output_dir.parent.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
      task.output_dir.mkdir()
    except FileExistsError as error:
      raise QueueFailure(
        f'{task.task_id} output directory appeared during reservation') \
        from error
    try:
      log_handle = task.log_path.open('x', encoding='utf-8', buffering=1)
    except FileExistsError as error:
      raise QueueFailure(
        f'{task.task_id} log appeared during reservation; preserving the '
        'reserved incomplete directory') from error
    log_handle.write(json.dumps({
      'event': 'generation_queue_task_launch',
      'task_id': task.task_id,
      'command': list(task.command),
      'cwd': str(self.plan.paths.runner_repo),
      'immutable_runner_git_sha': IMMUTABLE_RUNNER_GIT_SHA,
    }, sort_keys=True) + '\n')
    log_handle.flush()
    os.fsync(log_handle.fileno())
    return log_handle

  def _launch_pair(self, pair: Sequence[ShardTask]) -> None:
    # Complete shards are revalidated.  Every other path is checked before
    # either peer is launched, so a stale second task cannot strand a new first.
    pending = [task for task in pair if self._pending_or_complete(task)]
    if not pending:
      return
    reservations = []
    try:
      for task in pending:
        reservations.append((task, self._reserve_task(task)))
    except Exception:
      for _, handle in reservations:
        handle.close()
      raise

    processes = []
    launch_errors = []
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment['PYTHONUNBUFFERED'] = '1'
    for task, handle in reservations:
      try:
        process = self.popen_factory(
          list(task.command),
          cwd=self.plan.paths.runner_repo,
          env=environment,
          stdout=handle,
          stderr=subprocess.STDOUT,
          text=True,
          start_new_session=True,
        )
        processes.append((task, process))
        _emit(
          'generation_queue_task_started', task_id=task.task_id,
          pid=process.pid, output_dir=str(task.output_dir),
          log=str(task.log_path))
      except Exception as error:  # preserve reservation, then drain the peer
        launch_errors.append((task, error))
      finally:
        handle.close()

    results = []
    for task, process in processes:
      try:
        returncode = process.wait()
      except Exception as error:
        results.append((task, None, error))
      else:
        results.append((task, returncode, None))

    errors = [
      f'{task.task_id} could not launch: {error}'
      for task, error in launch_errors]
    for task, returncode, wait_error in results:
      if wait_error is not None:
        errors.append(f'{task.task_id} wait failed: {wait_error}')
      elif returncode != 0:
        errors.append(f'{task.task_id} exited with status {returncode}')
      else:
        try:
          self._validate_completed(task)
        except Exception as error:  # retain peer validation before halting
          errors.append(f'{task.task_id} validation failed: {error}')
    if errors:
      raise QueueFailure('; '.join(errors))

  def _run_locked(self) -> None:
    self.verify_environment()
    _emit(
      'generation_queue_verified',
      immutable_runner_git_sha=IMMUTABLE_RUNNER_GIT_SHA,
      initial_pids=[task.adopted_pid for task in self.plan.initial_tasks],
      max_workers=2)
    initial_errors = []
    with ThreadPoolExecutor(max_workers=2) as executor:
      futures = {
        executor.submit(self._wait_initial_task, task): task
        for task in self.plan.initial_tasks}
      for future, task in futures.items():
        try:
          future.result()
        except Exception as error:
          initial_errors.append(f'{task.task_id}: {error}')
    if initial_errors:
      raise QueueFailure('; '.join(initial_errors))

    for phase_number, phase in enumerate(self.plan.phases, start=1):
      if len(phase) % 2:
        raise QueueFailure('each frozen queue phase must contain task pairs')
      _emit(
        'generation_queue_phase_started', phase=phase_number,
        arm=phase[0].arm.name if phase else None, num_tasks=len(phase))
      for offset in range(0, len(phase), 2):
        self._launch_pair(phase[offset:offset + 2])
      _emit(
        'generation_queue_phase_completed', phase=phase_number,
        arm=phase[0].arm.name if phase else None)
    completion_path, completion_sha256 = \
      self._write_or_validate_completion_evidence()
    _emit(
      'generation_queue_complete',
      dataset_slug=self._dataset_slug(),
      logical_dataset=self.logical_dataset,
      num_tasks=len(self._all_tasks()),
      launch_plan_sha256=self.launch_plan_sha256,
      completion_evidence=str(completion_path),
      completion_evidence_sha256=completion_sha256)

  def run(self) -> None:
    dataset_slug = self._dataset_slug()
    lock = self.lock_factory(
      self.plan.paths.experiment_root / 'generation-queue.lock',
      queue_id=f'{dataset_slug}-generation-queue',
      dataset_slug=dataset_slug,
      launch_plan_sha256=self.launch_plan_sha256,
      recover_stale=self.recover_stale_lock)
    with lock:
      self._run_locked()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--poll-seconds', type=float, default=30.0,
    help='seconds between immutable /proc checks for adopted PIDs')
  parser.add_argument(
    '--recover-stale-lock', action='store_true',
    help=(
      'preserve and replace a reviewed stale/invalid shared queue lock; '
      'never use for an active owner'))
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  controller = QueueController(
    frozen_queue_plan(), poll_seconds=args.poll_seconds,
    recover_stale_lock=args.recover_stale_lock)
  try:
    controller.run()
  except Exception as error:
    _emit(
      'generation_queue_failed', error_type=type(error).__name__,
      error=str(error))
    return 1
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
