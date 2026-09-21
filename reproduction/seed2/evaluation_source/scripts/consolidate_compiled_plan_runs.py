#!/usr/bin/env python3
"""Consolidate a complete compiled-plan run from read-only partition disks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


SUCCESS_MARKER = '_job_success.json'


def _load_object(path: Path) -> dict:
  with path.open() as handle:
    value = json.load(handle)
  if not isinstance(value, dict):
    raise TypeError(f'{path} must contain a JSON object')
  return value


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _marker_fingerprint(path: Path, job_id: str) -> dict:
  marker = _load_object(path)
  if marker.get('job_id') != job_id:
    raise ValueError(f'{path} belongs to {marker.get("job_id")!r}, not {job_id}')
  outputs = marker.get('outputs')
  if not isinstance(outputs, list) or not outputs:
    raise ValueError(f'{path} has no committed outputs')
  normalized = []
  for output in outputs:
    if not isinstance(output, dict):
      raise TypeError(f'{path} has a non-object output commitment')
    required = ('name', 'relative_path', 'sha256', 'size_bytes')
    if any(key not in output for key in required):
      raise ValueError(f'{path} has an incomplete output commitment')
    normalized.append({key: output[key] for key in required})
  return {
    'job_execution_sha256': marker.get('job_execution_sha256'),
    'outputs': sorted(normalized, key=lambda item: item['name']),
  }


def _mapped_artifact_dir(
    artifact_dir: Path,
    *,
    canonical_root: Path,
    disk_root: Path,
) -> Path:
  try:
    relative = artifact_dir.relative_to(canonical_root)
  except ValueError as error:
    raise ValueError(
      f'compiled artifact directory escapes canonical root: {artifact_dir}') \
      from error
  return disk_root / relative


def _copy_tree_atomic(source: Path, destination: Path) -> None:
  if destination.exists():
    raise FileExistsError(destination)
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
  if temporary.exists():
    raise FileExistsError(temporary)
  shutil.copytree(source, temporary, symlinks=True)
  os.replace(temporary, destination)


def _relative_plan_dir(
    plan_dir: Path,
    *,
    canonical_root: Path,
    source_roots: list[Path],
) -> Path:
  candidates = []
  for root in [canonical_root, *source_roots]:
    try:
      candidates.append(plan_dir.relative_to(root))
    except ValueError:
      continue
  if len(set(candidates)) != 1:
    raise ValueError(
      'plan directory must lie under exactly one canonical/source root')
  return candidates[0]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan-dir', type=Path, required=True)
  parser.add_argument('--source-root', type=Path, action='append', required=True)
  parser.add_argument('--destination-root', type=Path, required=True)
  parser.add_argument(
    '--canonical-root', type=Path, default=Path('/mnt/contextual-forest'))
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()

  plan_dir = args.plan_dir.expanduser().resolve()
  source_roots = [path.expanduser().resolve() for path in args.source_root]
  destination_root = args.destination_root.expanduser().resolve()
  canonical_root = args.canonical_root.expanduser().resolve()
  output = args.output.expanduser().resolve()
  if len(source_roots) != len(set(source_roots)):
    raise ValueError('source roots must be unique')
  if destination_root in source_roots:
    raise ValueError('destination root must not also be a source root')
  if output.exists():
    raise FileExistsError(output)

  plan_path = plan_dir / 'compiled-plan.json'
  plan_relative = _relative_plan_dir(
    plan_dir, canonical_root=canonical_root, source_roots=source_roots)
  plan = _load_object(plan_path)
  job_ids = plan.get('job_ids')
  if (not isinstance(job_ids, list) or not job_ids
      or len(job_ids) != len(set(job_ids))):
    raise ValueError('compiled plan has invalid job IDs')

  selected = {}
  duplicates = {}
  for job_id in job_ids:
    job_path = plan_dir / 'jobs' / f'{job_id}.json'
    job = _load_object(job_path)
    if job.get('job_id') != job_id:
      raise ValueError(f'job specification mismatch: {job_path}')
    artifact_dir = Path(job['artifact_dir']).expanduser().resolve()
    candidates = []
    for source_root in source_roots:
      candidate = _mapped_artifact_dir(
        artifact_dir, canonical_root=canonical_root, disk_root=source_root)
      marker_path = candidate / SUCCESS_MARKER
      if marker_path.is_file():
        candidates.append((source_root, candidate, _marker_fingerprint(
          marker_path, job_id)))
    if not candidates:
      raise FileNotFoundError(f'no successful partition contains {job_id}')
    fingerprint = candidates[0][2]
    conflicts = [item for item in candidates[1:] if item[2] != fingerprint]
    if conflicts:
      roots = [str(item[0]) for item in candidates]
      raise ValueError(f'conflicting successful copies of {job_id}: {roots}')
    selected[job_id] = candidates[0]
    duplicates[job_id] = len(candidates)

  copied = 0
  reused = 0
  job_records = {}
  for job_id in job_ids:
    source_root, source_dir, fingerprint = selected[job_id]
    job = _load_object(plan_dir / 'jobs' / f'{job_id}.json')
    canonical_artifact = Path(job['artifact_dir']).expanduser().resolve()
    destination = _mapped_artifact_dir(
      canonical_artifact,
      canonical_root=canonical_root,
      disk_root=destination_root)
    destination_marker = destination / SUCCESS_MARKER
    if destination.exists():
      if (not destination_marker.is_file()
          or _marker_fingerprint(destination_marker, job_id) != fingerprint):
        raise ValueError(f'destination contains conflicting job {job_id}')
      reused += 1
      status = 'reused'
    else:
      _copy_tree_atomic(source_dir, destination)
      if _marker_fingerprint(destination_marker, job_id) != fingerprint:
        raise RuntimeError(f'copied marker verification failed for {job_id}')
      copied += 1
      status = 'copied'
    job_records[job_id] = {
      'source_root': str(source_root),
      'source_success_marker_sha256': _sha256(source_dir / SUCCESS_MARKER),
      'duplicate_successful_copies': duplicates[job_id],
      'status': status,
    }

  destination_plan = destination_root / plan_relative
  if destination_plan.exists():
    if _sha256(destination_plan / 'compiled-plan.json') != _sha256(plan_path):
      raise ValueError('destination contains a different compiled plan')
  else:
    _copy_tree_atomic(plan_dir, destination_plan)

  result = {
    'schema_version': 1,
    'artifact': 'compiled_plan_partition_consolidation',
    'compiled_plan_sha256': _sha256(plan_path),
    'canonical_root': str(canonical_root),
    'destination_root': str(destination_root),
    'num_jobs': len(job_ids),
    'num_copied': copied,
    'num_reused': reused,
    'jobs': job_records,
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
  os.replace(temporary, output)
  print(json.dumps({
    'output': str(output),
    'num_jobs': len(job_ids),
    'num_copied': copied,
    'num_reused': reused,
  }, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
