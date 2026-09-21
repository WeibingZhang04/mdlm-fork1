#!/usr/bin/env python3
"""Build one atomic verified arXiv or PubMed generation-analysis bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.generation_adapter_comparison import (  # noqa: E402
  compare_generation_adapters,
)
from evaluation.generation_analysis_artifacts import (  # noqa: E402
  DATASET_CONTRACTS,
  POST_BUNDLE_ARTIFACT,
  POST_BUNDLE_SCHEMA_VERSION,
  validate_analysis_triplet,
  validate_cross_domain_post_bundle,
  validate_cross_domain_queue_completion,
)
from evaluation.generation_queue_artifacts import (  # noqa: E402
  SharedQueueLock,
  atomic_rename_directory_new,
  atomic_write_new,
  load_strict_json,
  sha256_file,
)
from evaluation.generation_shard_aggregation import (  # noqa: E402
  aggregate_generation_shards,
)
from scripts.run_cross_domain_generation_queue import (  # noqa: E402
  CROSS_DOMAIN_LAUNCH_PLAN_SHA256S,
  DATASETS,
  CrossDomainQueueController,
)
from scripts.run_wikitext_generation_queue import (  # noqa: E402
  DEFAULT_PATHS,
  IMMUTABLE_RUNNER_GIT_SHA,
  QueueFailure,
)


POST_BUNDLE_DIRECTORY = 'verified-analysis-v1'
POST_STAGING_DIRECTORY = f'.{POST_BUNDLE_DIRECTORY}.staging'
DYNAMIC_UNION_NAME = 'verified-dynamic-union.json'
STATIC_UNION_NAME = 'verified-static-union.json'
PAIRED_COMPARISON_NAME = 'paired-adapter-comparison.json'
FINAL_BUNDLE_NAME = 'bundle.json'
BOOTSTRAP_RESAMPLES = 20_000
UNION_BOOTSTRAP_SEED = 91_017
COMPARISON_BOOTSTRAP_SEED = 94_001
BOOTSTRAP_CONFIDENCE = 0.95


def _all_tasks(controller: CrossDomainQueueController):
  return tuple(
    task for phase in controller.plan.phases for task in phase)


def validate_queue_completion_evidence(
    path: Path, controller: CrossDomainQueueController,
) -> dict[str, Any]:
  dataset = controller.dataset
  tasks = _all_tasks(controller)
  gate = controller.reviewed_gate_identity
  if not isinstance(gate, Mapping):
    raise QueueFailure('reviewed WikiText gate is unavailable')
  expected_tasks = []
  for task in tasks:
    manifest_path = task.output_dir / 'manifest.json'
    expected_tasks.append({
      'task_id': task.task_id,
      'dataset_slug': dataset.slug,
      'arm': task.arm.name,
      'shard_index': task.shard_index,
      'output_dir': str(task.output_dir.resolve()),
      'manifest_sha256': (
        sha256_file(manifest_path) if manifest_path.is_file() else None),
    })
  try:
    return validate_cross_domain_queue_completion(
      path,
      contract=DATASET_CONTRACTS[dataset.slug],
      reviewed_gate=gate,
      expected_tasks=expected_tasks)
  except (OSError, TypeError, ValueError) as error:
    raise QueueFailure(str(error)) from error


def _write_json(path: Path, payload: object) -> None:
  atomic_write_new(path, json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _artifact_reference(final_root: Path, staged_path: Path) -> dict[str, str]:
  return {
    'path': str((final_root / staged_path.name).resolve()),
    'sha256': sha256_file(staged_path),
  }


def _preserve_stale_staging(staging: Path) -> Path:
  preserved = staging.with_name(
    f'{staging.name}.preserved-'
    f'{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex}')
  os.rename(staging, preserved)
  return preserved


def build_post_bundle(
    controller: CrossDomainQueueController,
    *,
    recover_stale_staging: bool = False,
    aggregate_fn: Callable[..., dict[str, Any]] = aggregate_generation_shards,
    compare_fn: Callable[..., dict[str, Any]] = compare_generation_adapters,
) -> tuple[Path, str]:
  """Revalidate all raw shards and atomically publish one analysis directory."""
  dataset = controller.dataset
  contract = DATASET_CONTRACTS[dataset.slug]
  root = controller.plan.paths.experiment_root / dataset.slug
  final_root = root / POST_BUNDLE_DIRECTORY
  staging = root / POST_STAGING_DIRECTORY
  controller.verify_environment()
  for task in _all_tasks(controller):
    controller._validate_completed(task)
  completion = validate_queue_completion_evidence(
    root / 'queue-complete.json', controller)
  if final_root.exists():
    raise QueueFailure(
      f'refusing to overwrite existing cross-domain analysis {final_root}')
  if staging.exists():
    if not recover_stale_staging:
      raise QueueFailure(
        f'stale post-analysis staging directory is preserved: {staging}')
    _preserve_stale_staging(staging)
  staging.mkdir(parents=True)

  try:
    dynamic_shards = [task.output_dir for task in controller.plan.phases[0]]
    static_shards = [task.output_dir for task in controller.plan.phases[1]]
    dynamic = aggregate_fn(
      dynamic_shards,
      baseline_mode='factorized',
      bootstrap_resamples=BOOTSTRAP_RESAMPLES,
      bootstrap_seed=UNION_BOOTSTRAP_SEED,
      bootstrap_confidence=BOOTSTRAP_CONFIDENCE)
    static = aggregate_fn(
      static_shards,
      baseline_mode='structured_joint',
      bootstrap_resamples=BOOTSTRAP_RESAMPLES,
      bootstrap_seed=UNION_BOOTSTRAP_SEED,
      bootstrap_confidence=BOOTSTRAP_CONFIDENCE)
    comparison = compare_fn(
      static_shards,
      dynamic_shards,
      baseline_union=static,
      treatment_union=dynamic,
      bootstrap_resamples=BOOTSTRAP_RESAMPLES,
      bootstrap_seed=COMPARISON_BOOTSTRAP_SEED,
      bootstrap_confidence=BOOTSTRAP_CONFIDENCE)
    validated = validate_analysis_triplet(
      dynamic, static, comparison, contract=contract)

    dynamic_path = staging / DYNAMIC_UNION_NAME
    static_path = staging / STATIC_UNION_NAME
    comparison_path = staging / PAIRED_COMPARISON_NAME
    _write_json(dynamic_path, dynamic)
    _write_json(static_path, static)
    _write_json(comparison_path, comparison)
    replayed = validate_analysis_triplet(
      load_strict_json(dynamic_path),
      load_strict_json(static_path),
      load_strict_json(comparison_path),
      contract=contract)
    if replayed != validated:
      raise QueueFailure('serialized analysis differs from validated results')
    artifacts = {
      'dynamic_union': _artifact_reference(final_root, dynamic_path),
      'static_union': _artifact_reference(final_root, static_path),
      'paired_comparison': _artifact_reference(final_root, comparison_path),
    }
    gate = controller.reviewed_gate_identity
    if not isinstance(gate, Mapping):
      raise QueueFailure('reviewed WikiText gate was not retained after preflight')
    bundle = {
      'schema_version': POST_BUNDLE_SCHEMA_VERSION,
      'artifact': POST_BUNDLE_ARTIFACT,
      'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
      'dataset_slug': dataset.slug,
      'logical_dataset': dataset.logical_dataset,
      'immutable_runner_git_sha': IMMUTABLE_RUNNER_GIT_SHA,
      'launch_plan_sha256': CROSS_DOMAIN_LAUNCH_PLAN_SHA256S[dataset.slug],
      'queue_completion_evidence': {
        'path': completion['path'],
        'sha256': completion['sha256'],
      },
      'reviewed_wikitext_gate': {
        'path': gate['path'],
        'sha256': gate['sha256'],
        'decision': gate['decision'],
        'controller_repository': gate['controller_repository'],
      },
      'artifacts': artifacts,
      'validated_analysis': validated,
    }
    bundle_path = staging / FINAL_BUNDLE_NAME
    _write_json(bundle_path, bundle)
    if load_strict_json(bundle_path) != bundle:
      raise QueueFailure('serialized final analysis bundle differs in replay')
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
    try:
      atomic_rename_directory_new(staging, final_root)
    except FileExistsError as error:
      raise QueueFailure(
        f'analysis destination appeared during staging: {final_root}') \
        from error
    final_bundle = final_root / FINAL_BUNDLE_NAME
    final_sha256 = sha256_file(final_bundle)
    validate_cross_domain_post_bundle(
      final_bundle,
      expected_sha256=final_sha256,
      controller_repo_root=getattr(controller, 'controller_repo_root', REPO_ROOT))
    return final_bundle, final_sha256
  except Exception:
    # A fixed staging name is deliberate: any partial analysis is preserved
    # and blocks an unreviewed retry.
    raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', choices=tuple(DATASETS), required=True)
  parser.add_argument('--wikitext-gate-sha256', required=True)
  parser.add_argument('--wikitext-gate', type=Path)
  parser.add_argument('--recover-stale-lock', action='store_true')
  parser.add_argument('--recover-stale-staging', action='store_true')
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  dataset = DATASETS[args.dataset]
  controller = CrossDomainQueueController(
    dataset,
    reviewed_gate_sha256=args.wikitext_gate_sha256,
    reviewed_gate_path=args.wikitext_gate,
    recover_stale_lock=args.recover_stale_lock)
  try:
    lock = SharedQueueLock(
      DEFAULT_PATHS.experiment_root / 'generation-queue.lock',
      queue_id=f'{dataset.slug}-generation-post',
      dataset_slug=dataset.slug,
      launch_plan_sha256=CROSS_DOMAIN_LAUNCH_PLAN_SHA256S[dataset.slug],
      recover_stale=args.recover_stale_lock)
    with lock:
      output, sha256 = build_post_bundle(
        controller, recover_stale_staging=args.recover_stale_staging)
  except Exception as error:
    print(json.dumps({
      'event': 'cross_domain_generation_post_failed',
      'dataset': dataset.slug,
      'error_type': type(error).__name__,
      'error': str(error),
    }, sort_keys=True), flush=True)
    return 1
  print(json.dumps({
    'event': 'cross_domain_generation_analysis_complete',
    'dataset': dataset.slug,
    'logical_dataset': dataset.logical_dataset,
    'bundle': str(output),
    'bundle_sha256': sha256,
  }, sort_keys=True), flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
