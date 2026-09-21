#!/usr/bin/env python3
"""Build a disjoint, provenance-recorded view of partitioned plan runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.consolidate_compiled_plan_runs import (
  SUCCESS_MARKER,
  _load_object,
  _mapped_artifact_dir,
  _marker_fingerprint,
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _copy_tree_atomic(source: Path, destination: Path) -> None:
  if destination.exists():
    raise FileExistsError(destination)
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
  if temporary.exists():
    raise FileExistsError(temporary)
  shutil.copytree(source, temporary, symlinks=True)
  os.replace(temporary, destination)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan-dir', type=Path, required=True)
  parser.add_argument('--base-root', type=Path, required=True)
  parser.add_argument('--source-root', type=Path, action='append', required=True)
  parser.add_argument('--destination-root', type=Path, required=True)
  parser.add_argument(
    '--canonical-root', type=Path, default=Path('/mnt/contextual-forest'))
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()

  plan_dir = args.plan_dir.expanduser().resolve()
  base_root = args.base_root.expanduser().resolve()
  source_roots = [path.expanduser().resolve() for path in args.source_root]
  destination_root = args.destination_root.expanduser().resolve()
  canonical_root = args.canonical_root.expanduser().resolve()
  output = args.output.expanduser().resolve()
  all_roots = [base_root, *source_roots]
  if len(all_roots) != len(set(all_roots)):
    raise ValueError('base and source roots must be unique')
  if destination_root in all_roots:
    raise ValueError('destination root must not also be an input root')
  if output.exists():
    raise FileExistsError(output)

  plan_path = plan_dir / 'compiled-plan.json'
  plan = _load_object(plan_path)
  job_ids = plan.get('job_ids')
  if (not isinstance(job_ids, list) or not job_ids
      or len(job_ids) != len(set(job_ids))):
    raise ValueError('compiled plan has invalid job IDs')

  selected_from_base = 0
  copied = 0
  records = {}
  for job_id in job_ids:
    job = _load_object(plan_dir / 'jobs' / f'{job_id}.json')
    if job.get('job_id') != job_id:
      raise ValueError(f'job specification mismatch: {job_id}')
    artifact_dir = Path(job['artifact_dir']).expanduser().resolve()
    candidates = []
    for root in all_roots:
      candidate = _mapped_artifact_dir(
        artifact_dir, canonical_root=canonical_root, disk_root=root)
      marker_path = candidate / SUCCESS_MARKER
      if marker_path.is_file():
        candidates.append({
          'source_root': root,
          'source_dir': candidate,
          'marker_path': marker_path,
          'marker_sha256': _sha256(marker_path),
          'fingerprint': _marker_fingerprint(marker_path, job_id),
        })
    if not candidates:
      raise FileNotFoundError(f'no successful partition contains {job_id}')

    selected = candidates[0]
    selected_is_base = selected['source_root'] == base_root
    destination = _mapped_artifact_dir(
      artifact_dir,
      canonical_root=canonical_root,
      disk_root=destination_root)
    if selected_is_base:
      selected_from_base += 1
      status = 'retained_in_base'
    else:
      if destination.exists():
        destination_marker = destination / SUCCESS_MARKER
        if (not destination_marker.is_file()
            or _marker_fingerprint(destination_marker, job_id)
            != selected['fingerprint']):
          raise ValueError(f'destination contains conflicting job {job_id}')
        status = 'reused'
      else:
        _copy_tree_atomic(selected['source_dir'], destination)
        if (_marker_fingerprint(destination / SUCCESS_MARKER, job_id)
            != selected['fingerprint']):
          raise RuntimeError(f'copied marker verification failed for {job_id}')
        copied += 1
        status = 'copied'

    records[job_id] = {
      'selected_source_root': str(selected['source_root']),
      'selected_success_marker_sha256': selected['marker_sha256'],
      'status': status,
      'candidate_count': len(candidates),
      'conflicting_candidate_count': sum(
        candidate['fingerprint'] != selected['fingerprint']
        for candidate in candidates[1:]),
      'candidates': [{
        'source_root': str(candidate['source_root']),
        'success_marker_sha256': candidate['marker_sha256'],
        'matches_selected_fingerprint': (
          candidate['fingerprint'] == selected['fingerprint']),
      } for candidate in candidates],
    }

  result = {
    'schema_version': 1,
    'artifact': 'compiled_plan_disjoint_partition_selection',
    'selection_rule': 'base_then_ordered_source_roots_first_success',
    'compiled_plan_sha256': _sha256(plan_path),
    'canonical_root': str(canonical_root),
    'base_root': str(base_root),
    'ordered_source_roots': [str(root) for root in source_roots],
    'destination_root': str(destination_root),
    'num_jobs': len(job_ids),
    'num_selected_from_base': selected_from_base,
    'num_copied': copied,
    'num_jobs_with_conflicting_candidates': sum(
      record['conflicting_candidate_count'] > 0 for record in records.values()),
    'jobs': records,
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
  os.replace(temporary, output)
  print(json.dumps({
    'output': str(output),
    'num_jobs': len(job_ids),
    'num_selected_from_base': selected_from_base,
    'num_copied': copied,
    'num_jobs_with_conflicting_candidates': result[
      'num_jobs_with_conflicting_candidates'],
  }, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
