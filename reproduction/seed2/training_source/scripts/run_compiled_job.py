#!/usr/bin/env python3
"""Run compiled experiment jobs without a shell and record durable success.

Jobs are skipped only when their success marker, job-spec digest, and every
required output hash still match. Every job uses a new attempt directory.
This is essential for streaming OpenWebText: its loader has no restorable row
cursor, so a preempted fit must restart from row zero instead of loading a
model checkpoint while silently replaying the start of the corpus.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.compile_experiment_matrix import (  # noqa: E402
  JOB_SCHEMA_VERSION,
  PLAN_SCHEMA_VERSION,
  SLUG_PATTERN,
  _canonical_json,
  _git_metadata,
  sha256_file,
)


SUCCESS_MARKER = '_job_success.json'
LOCK_FILE = '_job.lock'
PLACEHOLDER = re.compile(
  r'\$\{(artifact|sha256):([a-z0-9][a-z0-9_-]*):([^{}]+)\}')


def _read_json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  with path.open() as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise TypeError(f'{path} must contain a JSON object')
  return payload


def _job_digest(job: Mapping[str, Any]) -> str:
  return hashlib.sha256(_canonical_json(job).encode()).hexdigest()


def _job_execution_digest(job: Mapping[str, Any]) -> str:
  """Hash fields that affect execution, permitting cross-suite reuse.

  ``plan_id`` and ``suites`` are orchestration metadata. A promoted plan may
  therefore reuse an already successful prerequisite only when every other
  job field—including the clean source repository SHA, argv, input hashes,
  output contract, and frozen source manifest—remains byte-for-byte identical.
  """
  projected = {
    key: value for key, value in job.items()
    if key not in {'plan_id', 'suites'}}
  return hashlib.sha256(_canonical_json(projected).encode()).hexdigest()


def _safe_relative(value: str, *, context: str) -> PurePosixPath:
  path = PurePosixPath(value)
  if path.is_absolute() or not path.parts \
      or any(part in {'', '.', '..'} for part in path.parts):
    raise ValueError(f'{context} must be a safe relative path: {value!r}')
  return path


def _within(path: Path, root: Path, *, context: str) -> Path:
  resolved = path.expanduser().resolve(strict=False)
  root = root.expanduser().resolve(strict=False)
  try:
    resolved.relative_to(root)
  except ValueError as error:
    raise ValueError(f'{context} {resolved} is outside {root}') from error
  if resolved == root:
    raise ValueError(f'{context} must not equal {root}')
  return resolved


def _load_plan(plan_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  plan_dir = plan_dir.expanduser().resolve()
  plan = _read_json(plan_dir / 'compiled-plan.json')
  if plan.get('schema_version') != PLAN_SCHEMA_VERSION:
    raise ValueError('unsupported compiled plan schema')
  if (not isinstance(plan.get('plan_id'), str)
      or len(plan['plan_id']) != 64):
    raise ValueError('invalid plan_id')
  job_ids = plan.get('job_ids')
  if not isinstance(job_ids, list) or not job_ids:
    raise ValueError('compiled plan has no jobs')
  if len(job_ids) != len(set(job_ids)):
    raise ValueError('compiled plan repeats job IDs')
  committed_job_digests = plan.get('job_spec_sha256')
  if (not isinstance(committed_job_digests, dict)
      or set(committed_job_digests) != set(job_ids)):
    raise ValueError('compiled plan has an invalid job-spec commitment')
  jobs = {}
  for job_id in job_ids:
    if not isinstance(job_id, str) or not SLUG_PATTERN.fullmatch(job_id):
      raise ValueError(f'invalid job ID in plan: {job_id!r}')
    job = _read_json(plan_dir / 'jobs' / f'{job_id}.json')
    expected_keys = {
      'schema_version', 'protocol_id', 'source_manifest_sha256', 'plan_id',
      'source_repository_sha',
      'job_id', 'kind', 'artifact_dir', 'suites', 'dependencies', 'identity',
      'argv', 'execution_mode', 'external_inputs', 'required_outputs',
    }
    if set(job) != expected_keys:
      raise ValueError(
        f'{job_id} job schema mismatch: '
        f'missing={sorted(expected_keys - set(job))}, '
        f'unknown={sorted(set(job) - expected_keys)}')
    if job['schema_version'] != JOB_SCHEMA_VERSION:
      raise ValueError(f'{job_id} has unsupported job schema')
    if job['job_id'] != job_id or job['plan_id'] != plan['plan_id']:
      raise ValueError(f'{job_id} identity does not match compiled plan')
    if job['protocol_id'] != plan.get('protocol_id'):
      raise ValueError(f'{job_id} protocol does not match plan')
    if job['source_manifest_sha256'] != plan.get(
        'source_manifest_sha256'):
      raise ValueError(f'{job_id} source manifest does not match plan')
    repository = plan.get('repository')
    if (not isinstance(repository, Mapping)
        or job['source_repository_sha'] != repository.get('sha')):
      raise ValueError(f'{job_id} source repository does not match plan')
    if _job_digest(job) != committed_job_digests[job_id]:
      raise ValueError(f'{job_id} differs from its compiled plan commitment')
    if job['kind'] not in {'train', 'export', 'eval'}:
      raise ValueError(f'{job_id} has invalid kind')
    if job['execution_mode'] not in {'resume_in_place', 'fresh_attempt'}:
      raise ValueError(f'{job_id} has invalid execution_mode')
    if not isinstance(job['argv'], list) or not job['argv'] \
        or any(not isinstance(token, str) or not token for token in job['argv']):
      raise ValueError(f'{job_id} argv must contain non-empty strings')
    jobs[job_id] = job
  if set(jobs) != set(job_ids):
    raise ValueError('compiled job set differs from job_ids')
  for job_id, job in jobs.items():
    dependencies = job['dependencies']
    if not isinstance(dependencies, list) \
        or len(dependencies) != len(set(dependencies)):
      raise ValueError(f'{job_id} dependencies are invalid')
    missing = sorted(set(dependencies) - set(jobs))
    if missing:
      raise ValueError(f'{job_id} has missing dependencies {missing}')
    if job_id in dependencies:
      raise ValueError(f'{job_id} depends on itself')
  return plan, jobs


def _validate_repository_checkout(
    plan: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
  expected = plan.get('repository')
  if not isinstance(expected, Mapping) or set(expected) != {'sha', 'dirty'}:
    raise ValueError('compiled plan lacks exact repository metadata')
  if expected['sha'] is None or expected['dirty'] is not False:
    raise ValueError(
      'compiled jobs are executable only from a clean committed repository; '
      'recompile the plan after committing the experiment code')
  actual = _git_metadata(repo_root)
  if actual != dict(expected):
    raise ValueError(
      f'repository checkout differs from compiled plan: expected '
      f'{dict(expected)}, found {actual}')


def _output_records(run_dir: Path, required: object) -> list[dict[str, Any]]:
  if not isinstance(required, list) or not required:
    raise ValueError('required_outputs must be a non-empty list')
  seen_names: set[str] = set()
  records = []
  for index, raw in enumerate(required):
    if not isinstance(raw, Mapping) or set(raw) != {
        'name', 'pattern', 'exactly_one'}:
      raise ValueError(f'required_outputs[{index}] has invalid schema')
    name = raw['name']
    if not isinstance(name, str) or not SLUG_PATTERN.fullmatch(name):
      raise ValueError(f'invalid required output name: {name!r}')
    if name in seen_names:
      raise ValueError(f'duplicate required output name: {name}')
    seen_names.add(name)
    pattern = raw['pattern']
    if not isinstance(pattern, str):
      raise TypeError(f'output pattern for {name} must be a string')
    _safe_relative(pattern.replace('**/', ''), context=f'{name} pattern')
    if raw['exactly_one'] is not True:
      raise ValueError('only exactly_one output commitments are supported')
    matches = sorted(path for path in run_dir.glob(pattern) if path.is_file())
    if len(matches) != 1:
      raise ValueError(
        f'{name} expected exactly one {pattern!r} below {run_dir}; '
        f'found {len(matches)}')
    path = matches[0].resolve()
    _within(path, run_dir, context=f'{name} output')
    records.append({
      'name': name,
      'relative_path': path.relative_to(run_dir.resolve()).as_posix(),
      'size_bytes': path.stat().st_size,
      'sha256': sha256_file(path),
    })
  return records


def _marker_path(job: Mapping[str, Any]) -> Path:
  return Path(job['artifact_dir']).resolve() / SUCCESS_MARKER


def _validated_marker(
    job: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
  marker_path = _marker_path(job)
  if not marker_path.exists():
    if required:
      raise FileNotFoundError(
        f'dependency {job["job_id"]} is incomplete: {marker_path}')
    return None
  marker = _read_json(marker_path)
  expected_keys = {
    'schema_version', 'artifact', 'job_id', 'originating_plan_id',
    'source_repository_sha', 'job_execution_sha256',
    'run_dir', 'argv', 'start_time_utc',
    'end_time_utc', 'outputs',
  }
  if set(marker) != expected_keys:
    raise ValueError(f'invalid success marker schema: {marker_path}')
  if marker['schema_version'] != 2 \
      or marker['artifact'] != 'compiled_experiment_job_success':
    raise ValueError(f'invalid success marker identity: {marker_path}')
  if (marker['job_id'] != job['job_id']
      or marker['source_repository_sha'] != job['source_repository_sha']
      or marker['job_execution_sha256'] != _job_execution_digest(job)):
    raise ValueError(f'success marker does not match job spec: {marker_path}')
  run_dir = _within(
    Path(marker['run_dir']), Path(job['artifact_dir']),
    context='marker run_dir')
  actual_outputs = _output_records(run_dir, job['required_outputs'])
  if marker['outputs'] != actual_outputs:
    raise ValueError(f'completed job outputs drifted: {job["job_id"]}')
  return marker


def _dependency_value(
    kind: str,
    dependency_id: str,
    relative_path: str,
    *,
    jobs: Mapping[str, Mapping[str, Any]],
) -> str:
  if dependency_id not in jobs:
    raise ValueError(f'placeholder names unknown job {dependency_id}')
  relative = _safe_relative(relative_path, context='artifact placeholder')
  marker = _validated_marker(jobs[dependency_id], required=True)
  assert marker is not None
  run_dir = Path(marker['run_dir']).resolve()
  candidate = _within(
    run_dir / Path(*relative.parts), run_dir,
    context='dependency artifact')
  if not candidate.is_file():
    raise FileNotFoundError(candidate)
  committed = {
    output['relative_path']: output for output in marker['outputs']}
  record = committed.get(relative.as_posix())
  if record is None:
    raise ValueError(
      f'{dependency_id} did not commit output {relative.as_posix()}')
  actual_sha = sha256_file(candidate)
  if actual_sha != record['sha256']:
    raise ValueError(f'dependency artifact drifted: {candidate}')
  return str(candidate) if kind == 'artifact' else actual_sha


def _resolve_token(
    token: str,
    *,
    run_dir: Path,
    jobs: Mapping[str, Mapping[str, Any]],
) -> str:
  token = token.replace('{python}', sys.executable)
  token = token.replace('{artifact_dir}', str(run_dir))

  def replacement(match: re.Match[str]) -> str:
    return _dependency_value(
      match.group(1), match.group(2), match.group(3), jobs=jobs)

  resolved = PLACEHOLDER.sub(replacement, token)
  if '${' in resolved or '{artifact_dir}' in resolved or '{python}' in resolved:
    raise ValueError(f'unresolved placeholder in argv token: {resolved!r}')
  return resolved


def _validate_external_inputs(raw_inputs: object) -> list[dict[str, Any]]:
  if not isinstance(raw_inputs, list):
    raise TypeError('external_inputs must be a list')
  records = []
  for index, raw in enumerate(raw_inputs):
    if not isinstance(raw, Mapping) or set(raw) != {'role', 'path', 'sha256'}:
      raise ValueError(f'external_inputs[{index}] has invalid schema')
    path = Path(raw['path']).expanduser().resolve()
    if not path.is_file():
      raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != raw['sha256']:
      raise ValueError(
        f'external input SHA256 mismatch for {raw["role"]}: '
        f'expected {raw["sha256"]}, found {actual}')
    records.append({
      'role': raw['role'], 'path': str(path), 'sha256': actual,
      'size_bytes': path.stat().st_size,
    })
  return records


def _boot_id() -> str | None:
  path = Path('/proc/sys/kernel/random/boot_id')
  return path.read_text().strip() if path.is_file() else None


def _acquire_lock(artifact_dir: Path, job_id: str) -> Path:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  lock_path = artifact_dir / LOCK_FILE
  payload = {
    'job_id': job_id,
    'hostname': socket.gethostname(),
    'pid': os.getpid(),
    'boot_id': _boot_id(),
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
  }
  try:
    descriptor = os.open(
      lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  except FileExistsError as error:
    stale = _read_json(lock_path)
    same_host = stale.get('hostname') == socket.gethostname()
    different_boot = (
      stale.get('boot_id') is not None
      and _boot_id() is not None
      and stale.get('boot_id') != _boot_id())
    alive = False
    if same_host and not different_boot:
      try:
        os.kill(int(stale.get('pid')), 0)
        alive = True
      except (OSError, TypeError, ValueError):
        alive = False
    if same_host and (different_boot or not alive):
      lock_path.unlink()
      return _acquire_lock(artifact_dir, job_id)
    raise RuntimeError(
      f'job lock exists and cannot be proven stale: {lock_path}') from error
  with os.fdopen(descriptor, 'w') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())
  return lock_path


def _next_attempt(artifact_dir: Path) -> Path:
  attempts = artifact_dir / 'attempts'
  attempts.mkdir(parents=True, exist_ok=True)
  observed = []
  for child in attempts.iterdir():
    match = re.fullmatch(r'attempt-(\d{4})', child.name)
    if match:
      observed.append(int(match.group(1)))
  run_dir = attempts / f'attempt-{max(observed, default=0) + 1:04d}'
  run_dir.mkdir()
  return run_dir.resolve()


def run_job(
    job_id: str,
    *,
    plan: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    dry_run: bool = False,
) -> str:
  """Execute one job, returning ``skipped``, ``dry-run``, or ``completed``."""
  if job_id not in jobs:
    raise ValueError(f'unknown job: {job_id}')
  job = jobs[job_id]
  artifact_root = Path(plan['artifact_root']).resolve()
  artifact_dir = _within(
    Path(job['artifact_dir']), artifact_root, context='job artifact_dir')
  completed = _validated_marker(job, required=False)
  if completed is not None:
    return 'skipped'
  _validate_external_inputs(job['external_inputs'])
  if dry_run:
    # A whole-plan dry run must be possible before any dependency has run.
    # Preserve dependency-output placeholders (whose paths and hashes do not
    # exist yet), while still showing the exact interpreter and attempt path.
    preview_dir = (
      artifact_dir if job['execution_mode'] == 'resume_in_place'
      else artifact_dir / 'attempts' / 'attempt-NNNN')
    argv = [
      token.replace('{python}', sys.executable).replace(
        '{artifact_dir}', str(preview_dir))
      for token in job['argv']]
    if any('{python}' in token or '{artifact_dir}' in token for token in argv):
      raise ValueError('dry-run preview contains an unresolved local token')
    print(json.dumps({'job_id': job_id, 'argv': argv}, indent=2))
    return 'dry-run'

  for dependency_id in job['dependencies']:
    _validated_marker(jobs[dependency_id], required=True)

  lock_path = _acquire_lock(artifact_dir, job_id)
  start = dt.datetime.now(dt.timezone.utc)
  try:
    # Check again after acquiring the lock in case another process completed
    # the job between the first marker check and lock acquisition.
    if _validated_marker(job, required=False) is not None:
      return 'skipped'
    run_dir = (
      artifact_dir if job['execution_mode'] == 'resume_in_place'
      else _next_attempt(artifact_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = [
      _resolve_token(token, run_dir=run_dir, jobs=jobs)
      for token in job['argv']]
    subprocess.run(argv, cwd=REPO_ROOT, check=True)
    outputs = _output_records(run_dir, job['required_outputs'])
    end = dt.datetime.now(dt.timezone.utc)
    marker = {
      'schema_version': 2,
      'artifact': 'compiled_experiment_job_success',
      'job_id': job_id,
      'originating_plan_id': job['plan_id'],
      'source_repository_sha': job['source_repository_sha'],
      'job_execution_sha256': _job_execution_digest(job),
      'run_dir': str(run_dir),
      'argv': argv,
      'start_time_utc': start.isoformat(),
      'end_time_utc': end.isoformat(),
      'outputs': outputs,
    }
    marker_path = artifact_dir / SUCCESS_MARKER
    temporary = marker_path.with_name(
      f'.{marker_path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, marker_path)
    return 'completed'
  finally:
    if lock_path.exists():
      lock_path.unlink()


def _topological_order(
    selected: Sequence[str],
    jobs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
  wanted: set[str] = set()

  def add(job_id: str) -> None:
    if job_id in wanted:
      return
    if job_id not in jobs:
      raise ValueError(f'unknown job: {job_id}')
    for dependency in jobs[job_id]['dependencies']:
      add(dependency)
    wanted.add(job_id)

  for job_id in selected:
    add(job_id)
  ordered = []
  remaining = set(wanted)
  while remaining:
    ready = sorted(
      job_id for job_id in remaining
      if set(jobs[job_id]['dependencies']).issubset(set(ordered)))
    if not ready:
      raise ValueError('job graph contains a dependency cycle')
    ordered.extend(ready)
    remaining.difference_update(ready)
  return ordered


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan-dir', type=Path, required=True)
  selection = parser.add_mutually_exclusive_group(required=True)
  selection.add_argument('--job-id', action='append')
  selection.add_argument('--all', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args(argv)


def main() -> int:
  args = _parse_args()
  plan, jobs = _load_plan(args.plan_dir)
  _validate_repository_checkout(plan)
  selected = list(jobs) if args.all else args.job_id
  statuses = {}
  for job_id in _topological_order(selected, jobs):
    statuses[job_id] = run_job(
      job_id, plan=plan, jobs=jobs, dry_run=args.dry_run)
  print(json.dumps({
    'host': platform.node(),
    'plan_id': plan['plan_id'],
    'statuses': statuses,
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
