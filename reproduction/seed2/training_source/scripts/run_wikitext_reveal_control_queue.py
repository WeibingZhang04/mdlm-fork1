#!/usr/bin/env python3
"""Run the frozen WikiText factorized reveal-policy control.

This diagnostic reruns the same dynamic adapter and paired prompts under two
factorized token laws: the original independent reveal kernel and the
confidence-gated reveal-position control.  It is deliberately separate from
the frozen primary marginal-versus-joint experiment.
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
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.generation_shard_aggregation import (  # noqa: E402
  load_generation_shard,
  sha256_file,
)


RUNNER_GIT_SHA = '8b309fccbddfa0661f89ecd59275d5b3719e9d44'
RUNNER_SCRIPT_SHA256 = (
  'd65b5a26615377454a43aa76e78db4a329d0368c2ffeaadd173dac25ec48b910')
BACKBONE_SHA256 = (
  '1b7c724d0228b1a2c825185c96642ffd706bd828b237f84512f0e1c7b5765573')
ADAPTER_SHA256 = (
  '037817d874af0c8c60b1a8eaf5bab3506fc07ca70234911498445f7b8c12c769')
ADAPTER_MANIFEST_SHA256 = (
  'b98a563bbc52ab501197956d424e4ab496b51841268c0556fbc9b7e134e0a61f')
ADAPTER_ORIGIN_SHA256 = (
  '8a3580f7d40c0139f6e57603917ecf923a9b62d9ef878c284e170bce280f8455')
PROMPT_MANIFEST_SHA256 = (
  'e6f9cc313b6296cbb7692c450e1d58ef1ccf223d13eff6507b6d509b717015b8')
REFERENCE_LM_REVISION = '32b71b12589c2f8d625668d2335a01cac3249519'
MODES = ('factorized', 'factorized_confidence_gated')
NFE_BUDGETS = (8, 16, 32, 64)
NUM_SAMPLES = 788
NUM_SHARDS = 16
BATCH_SIZE = 8
BASE_SEED = 91001
OUTPUT_NAMESPACE = 'reveal-policy-control-v2'


@dataclass(frozen=True)
class Paths:
  runner_repo: Path
  python: Path
  experiment_root: Path
  expansion_root: Path
  backbone: Path


DEFAULT_PATHS = Paths(
  runner_repo=Path('/mnt/contextual-forest/mdlm-nina-replay-cd138c7'),
  python=Path('/mnt/contextual-forest/venv/bin/python'),
  experiment_root=Path(
    '/mnt/contextual-forest/experiments/'
    'contextual-forest-generation-paper-v1'),
  expansion_root=Path(
    '/mnt/contextual-forest/experiments/contextual-forest-expansion-v1'),
  backbone=Path('/mnt/contextual-forest/checkpoints/mdlm-owt-backbone.pt'),
)


@dataclass(frozen=True)
class Task:
  shard_index: int
  output_dir: Path
  log_path: Path
  command: tuple[str, ...]


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runner-repo', type=Path,
                      default=DEFAULT_PATHS.runner_repo)
  parser.add_argument('--python', type=Path, default=DEFAULT_PATHS.python)
  parser.add_argument('--experiment-root', type=Path,
                      default=DEFAULT_PATHS.experiment_root)
  parser.add_argument('--expansion-root', type=Path,
                      default=DEFAULT_PATHS.expansion_root)
  parser.add_argument('--backbone', type=Path, default=DEFAULT_PATHS.backbone)
  parser.add_argument('--plan-only', action='store_true')
  return parser.parse_args(argv)


def _absolute_path_without_resolving_symlinks(path: Path) -> Path:
  """Make a path absolute while preserving virtual-environment symlinks."""
  return Path(os.path.abspath(path))


def expected_shard_samples(shard_index: int) -> int:
  if not 0 <= shard_index < NUM_SHARDS:
    raise ValueError('shard_index lies outside the frozen grid')
  return len(range(shard_index, NUM_SAMPLES, NUM_SHARDS))


def build_tasks(paths: Paths) -> list[Task]:
  root = paths.experiment_root
  output_root = root / 'wikitext' / OUTPUT_NAMESPACE
  adapter_root = (
    paths.expansion_root / 'runs'
    / 'export--dynamic_dynamic--s001--k128'
    / 'attempts' / 'attempt-0001')
  tasks = []
  for shard_index in range(NUM_SHARDS):
    suffix = f'{shard_index:02d}'
    output_dir = output_root / f'shard-{suffix}'
    log_path = output_root / 'logs' / f'shard-{suffix}.log'
    command = (
      str(paths.python),
      str(paths.runner_repo / 'scripts' / 'run_generation_pilot.py'),
      '--backbone-checkpoint', str(paths.backbone),
      '--backbone-sha256', BACKBONE_SHA256,
      '--adapter', str(adapter_root / 'adapter.safetensors'),
      '--adapter-sha256', ADAPTER_SHA256,
      '--adapter-manifest', str(adapter_root / 'adapter-manifest.json'),
      '--adapter-manifest-sha256', ADAPTER_MANIFEST_SHA256,
      '--adapter-origin-evidence', str(root / 'adapter-pair-origin.json'),
      '--adapter-origin-evidence-sha256', ADAPTER_ORIGIN_SHA256,
      '--adapter-origin-arm', 'dynamic_dynamic',
      '--output-dir', str(output_dir),
      '--prompt-jsonl', str(root / 'prompts' / 'wikitext-span32.jsonl'),
      '--prompt-manifest', str(
        root / 'prompts' / 'wikitext-span32.jsonl.manifest.json'),
      '--prompt-manifest-sha256', PROMPT_MANIFEST_SHA256,
      '--num-samples', str(NUM_SAMPLES),
      '--num-shards', str(NUM_SHARDS),
      '--shard-index', str(shard_index),
      '--sequence-length', '256',
      '--batch-size', str(BATCH_SIZE),
      '--base-seed', str(BASE_SEED),
      '--modes', *MODES,
      '--nfe-budgets', *(str(value) for value in NFE_BUDGETS),
      '--device', 'cuda',
      '--model-config', 'contextual-forest-small',
      '--data-config', 'eval_wikitext103_pinned',
      '--reference-lm', 'gpt2-large',
      '--reference-lm-revision', REFERENCE_LM_REVISION,
      '--reference-lm-device', 'cuda',
      '--reference-lm-batch-size', '8',
      '--reference-lm-max-length', '256',
      '--reference-lm-dtype', 'float32',
      '--override', 'model.structured_decoder.top_k=128',
      '--override', 'model.structured_decoder.topology_mode=dynamic',
      '--override', 'model.structured_decoder.factor_mode=dynamic',
      '--override', 'model.structured_decoder.training.topology_weight=0.1',
      '--override', 'trainer.devices=1',
      '--override', 'loader.num_workers=8',
      '--override',
      'checkpointing.save_dir=/mnt/contextual-forest/'
      'mdlm-generation-paper-v1',
    )
    tasks.append(Task(shard_index, output_dir, log_path, command))
  return tasks


def _canonical_sha256(payload: Any) -> str:
  return hashlib.sha256(json.dumps(
    payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def launch_plan_sha256(tasks: Sequence[Task]) -> str:
  return _canonical_sha256({
    'policy': 'wikitext-reveal-policy-control-v2',
    'runner_git_sha': RUNNER_GIT_SHA,
    'tasks': [{
      'shard_index': task.shard_index,
      'output_dir': str(task.output_dir),
      'log_path': str(task.log_path),
      'command': list(task.command),
    } for task in tasks],
  })


def _verify_runner(paths: Paths) -> None:
  if subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=paths.runner_repo,
      text=True).strip() != RUNNER_GIT_SHA:
    raise RuntimeError('runner checkout has the wrong Git revision')
  if subprocess.check_output(
      ['git', 'status', '--porcelain'], cwd=paths.runner_repo,
      text=True).strip():
    raise RuntimeError('runner checkout is dirty')
  runner = paths.runner_repo / 'scripts' / 'run_generation_pilot.py'
  if sha256_file(runner) != RUNNER_SCRIPT_SHA256:
    raise RuntimeError('generation runner SHA256 mismatch')
  checks = (
    (paths.backbone, BACKBONE_SHA256),
    (paths.experiment_root / 'adapter-pair-origin.json',
     ADAPTER_ORIGIN_SHA256),
    (paths.experiment_root / 'prompts'
     / 'wikitext-span32.jsonl.manifest.json', PROMPT_MANIFEST_SHA256),
  )
  adapter_root = (
    paths.expansion_root / 'runs'
    / 'export--dynamic_dynamic--s001--k128'
    / 'attempts' / 'attempt-0001')
  checks += (
    (adapter_root / 'adapter.safetensors', ADAPTER_SHA256),
    (adapter_root / 'adapter-manifest.json', ADAPTER_MANIFEST_SHA256),
  )
  for path, expected in checks:
    if not path.is_file() or sha256_file(path) != expected:
      raise RuntimeError(f'frozen input missing or hash-mismatched: {path}')


def validate_task(task: Task) -> dict[str, Any]:
  loaded = load_generation_shard(task.output_dir)
  manifest = loaded['manifest']
  pairing = manifest['pairing']
  matrix = manifest['matrix']
  expected_records = (
    expected_shard_samples(task.shard_index)
    * len(MODES) * len(NFE_BUDGETS))
  checks = (
    (manifest['repository']['git_sha'] == RUNNER_GIT_SHA,
     'runner Git SHA'),
    (pairing['shard_index'] == task.shard_index, 'shard index'),
    (pairing['num_shards'] == NUM_SHARDS, 'shard count'),
    (pairing['global_num_samples'] == NUM_SAMPLES, 'sample count'),
    (pairing['batch_size'] == BATCH_SIZE, 'batch size'),
    (pairing['base_seed'] == BASE_SEED, 'base seed'),
    (matrix['sampling_modes'] == list(MODES), 'sampling modes'),
    (matrix['nfe_budgets'] == list(NFE_BUDGETS), 'NFE budgets'),
    (matrix['num_output_records'] == expected_records, 'record count'),
    (manifest['adapter_origin_evidence']['arm'] == 'dynamic_dynamic',
     'adapter arm'),
  )
  failed = [label for passed, label in checks if not passed]
  if failed:
    raise RuntimeError(
      f'shard {task.shard_index} validation failed: {failed}')
  return {
    'shard_index': task.shard_index,
    'manifest_path': str(loaded['manifest_path']),
    'manifest_sha256': loaded['manifest_sha256'],
    'samples_sha256': manifest['outputs']['samples_jsonl']['sha256'],
    'num_records': expected_records,
  }


def _run_task(task: Task, runner_repo: Path) -> dict[str, Any]:
  if task.output_dir.exists():
    if not (task.output_dir / 'manifest.json').is_file():
      raise RuntimeError(
        f'refusing incomplete existing output {task.output_dir}')
    return validate_task(task)
  if task.log_path.exists():
    raise RuntimeError(f'refusing existing orphan log {task.log_path}')
  task.log_path.parent.mkdir(parents=True, exist_ok=True)
  with task.log_path.open('x', encoding='utf-8') as log:
    result = subprocess.run(
      task.command,
      cwd=runner_repo,
      stdout=log,
      stderr=subprocess.STDOUT,
      text=True,
      check=False)
  if result.returncode:
    raise RuntimeError(
      f'shard {task.shard_index} failed with return code '
      f'{result.returncode}; see {task.log_path}')
  return validate_task(task)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
  if path.exists():
    raise FileExistsError(f'refusing to overwrite {path}')
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    temporary.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def main(argv=None) -> int:
  args = _parse_args(argv)
  paths = Paths(
    args.runner_repo.resolve(),
    _absolute_path_without_resolving_symlinks(args.python),
    args.experiment_root.resolve(), args.expansion_root.resolve(),
    args.backbone.resolve())
  tasks = build_tasks(paths)
  plan_sha = launch_plan_sha256(tasks)
  if args.plan_only:
    print(json.dumps({
      'launch_plan_sha256': plan_sha,
      'tasks': [list(task.command) for task in tasks],
    }, indent=2, sort_keys=True))
    return 0
  _verify_runner(paths)
  output_root = (
    paths.experiment_root / 'wikitext' / OUTPUT_NAMESPACE)
  completion = output_root / 'queue-complete.json'
  if completion.exists():
    raise FileExistsError(f'queue already completed: {completion}')
  lock = output_root / 'queue.lock'
  lock.parent.mkdir(parents=True, exist_ok=True)
  descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  os.write(descriptor, json.dumps({
    'pid': os.getpid(),
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'launch_plan_sha256': plan_sha,
  }, sort_keys=True).encode())
  os.close(descriptor)
  completed = []
  try:
    for offset in range(0, len(tasks), 2):
      pair = tasks[offset:offset + 2]
      with ThreadPoolExecutor(max_workers=len(pair)) as executor:
        completed.extend(executor.map(
          lambda task: _run_task(task, paths.runner_repo), pair))
    completed.sort(key=lambda item: item['shard_index'])
    payload = {
      'schema_version': 1,
      'artifact': 'wikitext_reveal_policy_control_queue_completion',
      'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
      'launch_plan_sha256': plan_sha,
      'runner_git_sha': RUNNER_GIT_SHA,
      'modes': list(MODES),
      'nfe_budgets': list(NFE_BUDGETS),
      'shards': completed,
    }
    payload['completion_sha256'] = _canonical_sha256(payload)
    _atomic_json(completion, payload)
    print(json.dumps({
      'event': 'wikitext_reveal_policy_control_complete',
      'completion': str(completion),
      'completion_sha256': payload['completion_sha256'],
      'launch_plan_sha256': plan_sha,
    }, indent=2, sort_keys=True))
  finally:
    if lock.exists():
      lock.unlink()
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
