#!/usr/bin/env python3
"""Run one non-paper exact-environment Tensor-Train compatibility sample."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.tensor_train_baseline import (  # noqa: E402
  EXPECTED_CHECKPOINT_CONFIG_SHA256,
  EXPECTED_CHECKPOINT_STATE_KEYS,
  EXPECTED_CHECKPOINT_STEPS,
  SUBMISSION_GPU_LOCK,
  canonical_sha256,
  clean_git_identity,
  load_compiled_plan,
  sha256_file,
)
from scripts.run_tensor_train_feasibility import (  # noqa: E402
  _ForeignPidMonitor,
  _PositionScheduleRecorder,
  _checkpoint_config_identity,
  _clean_identity_matches,
  _compose_official_config,
  _exclusive_gpu_lock,
  _gpu_identity,
  _import_official_source,
  _install_pinned_transformer_proxies,
  _package_versions,
  _prepare_determinism,
  _prepare_offline_cache,
  _reseed_sampling,
)


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      'Load the exact released model and generate one non-paper sequence.'))
  parser.add_argument('--plan', type=Path, required=True)
  parser.add_argument('--job-id', required=True)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args(argv)


def _write_exclusive(path: Path, payload: dict) -> None:
  path = path.expanduser().resolve(strict=False)
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  with os.fdopen(descriptor, 'w') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())


def _descriptive_resource_probe(
    *,
    generation_seconds: float,
    generation_peak_allocated_bytes: int,
    generation_peak_reserved_bytes: int,
) -> dict:
  if not math.isfinite(generation_seconds) or generation_seconds <= 0.0:
    raise RuntimeError('preflight generation timing must be finite and positive')
  for name, value in (
      ('allocated', generation_peak_allocated_bytes),
      ('reserved', generation_peak_reserved_bytes)):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      raise RuntimeError(f'preflight CUDA peak {name} bytes are invalid')
  if generation_peak_reserved_bytes < generation_peak_allocated_bytes:
    raise RuntimeError('preflight CUDA reserved memory is inconsistent')
  return {
    'scope': 'one_sample_generation_only_excludes_model_load_and_evaluator',
    'generation_seconds': generation_seconds,
    'generation_peak_allocated_bytes': generation_peak_allocated_bytes,
    'generation_peak_reserved_bytes': generation_peak_reserved_bytes,
  }


def run_preflight(plan_path: Path, job_id: str) -> dict:
  plan_path = plan_path.expanduser().resolve()
  plan, jobs = load_compiled_plan(plan_path)
  if job_id not in jobs:
    raise KeyError(f'unknown compiled job: {job_id}')
  job = jobs[job_id]
  runtime = job['runtime']
  if '.'.join(platform.python_version_tuple()[:2]) \
      != runtime['python_major_minor']:
    raise RuntimeError('preflight Python differs from compiled job')
  packages = _package_versions(runtime['critical_packages'])
  cache_identity = _prepare_offline_cache(plan, job)
  with _exclusive_gpu_lock(SUBMISSION_GPU_LOCK) as acquired_lock:
    with _ForeignPidMonitor(
        runtime['gpu_monitor_interval_seconds']) as monitor:
      source_root = Path(job['source']['path']).resolve()
      _clean_identity_matches(
        clean_git_identity(source_root), job['source'], context='source')
      _clean_identity_matches(
        clean_git_identity(REPO_ROOT), job['harness_repository'],
        context='harness')
      checkpoint_path = Path(job['checkpoint']['path']).resolve()
      if sha256_file(checkpoint_path) != job['checkpoint']['sha256']:
        raise RuntimeError('preflight checkpoint hash mismatch')
      torch = _prepare_determinism(job['generation']['generation_seed'])
      if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('preflight requires exactly one visible CUDA GPU')
      torch.cuda.set_device(0)
      config, resolved_yaml = _compose_official_config(job, source_root)
      upstream_generate, upstream_mdlm = _import_official_source(source_root)
      _install_pinned_transformer_proxies(upstream_mdlm, job=job)
      model = upstream_mdlm.MDLM(config=config).to('cuda')
      payload = torch.load(
        checkpoint_path, map_location='cuda', weights_only=True)
      if sorted(payload) != ['config', 'model', 'optimizer', 'scheduler', 'step'] \
          or payload['step'] != EXPECTED_CHECKPOINT_STEPS[job['arm']] \
          or sorted(payload['model']) \
          != list(EXPECTED_CHECKPOINT_STATE_KEYS[job['arm']]):
        raise RuntimeError('preflight checkpoint structure mismatch')
      checkpoint_config_identity = _checkpoint_config_identity(
        payload['config'], job=job)
      if canonical_sha256(payload['config']) \
          != EXPECTED_CHECKPOINT_CONFIG_SHA256[job['arm']]:
        raise RuntimeError('preflight checkpoint config hash mismatch')
      missing_keys, unexpected_keys = model.load_state_dict(
        payload['model'], strict=False)
      if unexpected_keys \
          or set(EXPECTED_CHECKPOINT_STATE_KEYS[job['arm']]) & set(missing_keys):
        raise RuntimeError('preflight released learned state did not load')
      model.eval()
      smoke_generation = copy.deepcopy(job['generation'])
      smoke_generation['num_samples'] = 1
      config.generation.total_samples = 1
      config.generation.batch_size = 1
      recorder = _PositionScheduleRecorder(smoke_generation)
      original_pick = upstream_generate.pick_tokens_to_unmask

      def recorded_pick(ordering, x, K, mask_id, logprobs=None):
        selected = original_pick(ordering, x, K, mask_id, logprobs=logprobs)
        return recorder.record(
          selected, ordering=ordering, requested_k=K, logprobs=logprobs)

      upstream_generate.pick_tokens_to_unmask = recorded_pick
      _reseed_sampling(torch, smoke_generation['generation_seed'])
      torch.cuda.reset_peak_memory_stats()
      torch.cuda.synchronize()
      generation_started = time.perf_counter()
      try:
        with torch.inference_mode():
          generated, observed_steps = upstream_generate.sample(
            model, config, model.mask_id)
      finally:
        upstream_generate.pick_tokens_to_unmask = original_pick
      torch.cuda.synchronize()
      generation_seconds = time.perf_counter() - generation_started
      generation_peak_allocated = torch.cuda.max_memory_allocated()
      generation_peak_reserved = torch.cuda.max_memory_reserved()
      resource_probe = _descriptive_resource_probe(
        generation_seconds=generation_seconds,
        generation_peak_allocated_bytes=generation_peak_allocated,
        generation_peak_reserved_bytes=generation_peak_reserved)
      schedule = recorder.finalize(job={**job, 'generation': smoke_generation})
      if tuple(generated.shape) != (1, smoke_generation['sequence_length']) \
          or not math.isclose(
            observed_steps, float(smoke_generation['nfe_steps']),
            abs_tol=1e-6):
        raise RuntimeError('preflight sample shape/step count mismatch')
      tokens = generated[0].detach().cpu().tolist()
      gpu_evidence = monitor.snapshot(lock_path=acquired_lock)
      if gpu_evidence['foreign_pid_observations'] \
          or gpu_evidence['monitor_errors']:
        raise RuntimeError('preflight GPU was not exclusive')
      result = {
        'schema_version': 1,
        'artifact': 'tensor_train_exact_environment_preflight',
        'claim_scope': 'non-paper compatibility sample only',
        'plan_id': plan['plan_id'],
        'compiled_plan_sha256': sha256_file(plan_path),
        'job_id': job_id,
        'job_spec_sha256': job['job_spec_sha256'],
        'source': job['source'],
        'harness_repository': job['harness_repository'],
        'checkpoint': job['checkpoint'],
        'model_inputs': job['model_inputs'],
        'cache_identity_sha256': cache_identity['identity_sha256'],
        'resolved_config_sha256': hashlib.sha256(
          resolved_yaml.encode('utf-8')).hexdigest(),
        'checkpoint_config_identity': checkpoint_config_identity,
        'checkpoint_state_keys': sorted(payload['model']),
        'missing_keys': sorted(missing_keys),
        'runtime_packages': packages,
        'gpu': _gpu_identity(),
        'gpu_exclusivity': gpu_evidence,
        'descriptive_resource_probe': resource_probe,
        'sample_token_ids_sha256': canonical_sha256(tokens),
        'position_schedule_sha256': schedule['records'][0][
          'position_schedule_sha256'],
        'observed_steps': observed_steps,
      }
  return result


def main(argv=None) -> int:
  args = _parse_args(argv)
  output_path = args.output.expanduser().resolve(strict=False)
  if output_path.exists():
    raise FileExistsError(f'refusing to overwrite preflight output {output_path}')
  result = run_preflight(args.plan, args.job_id)
  _write_exclusive(output_path, result)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
