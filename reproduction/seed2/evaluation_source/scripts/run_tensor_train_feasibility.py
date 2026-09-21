#!/usr/bin/env python3
"""Execute one compiled Tensor-Train feasibility job atomically.

The official source checkout is imported without modification.  This wrapper
adds immutable input checks, pinned model/tokenizer loading, deterministic RNG,
complete sample persistence, a shared reference-LM evaluator, and replayable
resource/output manifests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from typing import Any
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.tensor_train_baseline import (  # noqa: E402
  CACHE_POLICY,
  EXPECTED_CHECKPOINT_CONFIG_SHA256,
  EXPECTED_CHECKPOINT_STATE_KEYS,
  EXPECTED_CHECKPOINT_STEPS,
  GPU_EXCLUSIVITY_POLICY,
  SUBMISSION_GPU_LOCK,
  canonical_sha256,
  cached_model_identities,
  clean_git_identity,
  load_compiled_plan,
  load_protocol,
  sha256_file,
  validate_completed_run,
)


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Run one frozen Tensor-Train OWT feasibility job.')
  parser.add_argument('--plan', type=Path, required=True)
  parser.add_argument('--job-id', required=True)
  parser.add_argument(
    '--resume', action='store_true',
    help=(
      'Reuse an existing complete run only after full validation. Partial or '
      'invalid outputs still fail closed and are never modified.'))
  return parser.parse_args(argv)


def _atomic_write(path: Path, content: str) -> None:
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    with temporary.open('x') as handle:
      handle.write(content)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def _output_descriptor(path: Path) -> dict[str, Any]:
  if not path.is_file() or path.stat().st_size <= 0:
    raise RuntimeError(f'expected nonempty output is missing: {path}')
  return {
    'path': path.name,
    'sha256': sha256_file(path),
    'size_bytes': path.stat().st_size,
  }


def _package_versions(expected: dict[str, str]) -> dict[str, str]:
  observed = {}
  for name, version in expected.items():
    try:
      actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
      raise RuntimeError(f'required package is missing: {name}') from error
    if actual != version:
      raise RuntimeError(
        f'{name} version mismatch: expected {version}, found {actual}')
    observed[name] = actual
  return observed


def _other_compute_pids() -> list[int]:
  try:
    output = subprocess.check_output([
      'nvidia-smi', '--query-compute-apps=pid',
      '--format=csv,noheader,nounits',
    ], text=True, stderr=subprocess.STDOUT)
  except (OSError, subprocess.CalledProcessError) as error:
    raise RuntimeError('nvidia-smi compute-process query failed') from error
  pids = []
  for line in output.splitlines():
    stripped = line.strip()
    if not stripped or stripped == '[N/A]':
      continue
    try:
      pid = int(stripped)
    except ValueError as error:
      raise RuntimeError(
        f'unexpected nvidia-smi compute PID row: {stripped!r}') from error
    if pid != os.getpid():
      pids.append(pid)
  return sorted(set(pids))


def _gpu_identity() -> dict[str, Any]:
  fields = ('index', 'name', 'uuid', 'driver_version', 'memory.total')
  try:
    output = subprocess.check_output([
      'nvidia-smi', f'--query-gpu={",".join(fields)}',
      '--format=csv,noheader,nounits',
    ], text=True, stderr=subprocess.STDOUT)
  except (OSError, subprocess.CalledProcessError) as error:
    raise RuntimeError('nvidia-smi GPU identity query failed') from error
  rows = [line.strip() for line in output.splitlines() if line.strip()]
  if len(rows) != 1:
    raise RuntimeError(
      f'feasibility protocol requires exactly one visible GPU, found {len(rows)}')
  values = [item.strip() for item in rows[0].split(',')]
  if len(values) != len(fields):
    raise RuntimeError(f'unexpected nvidia-smi identity row: {rows[0]!r}')
  return {
    'index': int(values[0]),
    'name': values[1],
    'uuid': values[2],
    'driver_version': values[3],
    'memory_total_mib': int(values[4]),
  }


@contextmanager
def _exclusive_gpu_lock(lock_path: Path):
  lock_path = lock_path.expanduser().resolve(strict=False)
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
  try:
    try:
      fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise RuntimeError(
        f'another Tensor-Train job holds the GPU lock: {lock_path}') from error
    yield lock_path
  finally:
    try:
      fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
      os.close(descriptor)


class _ForeignPidMonitor:
  """Continuously sample compute PIDs while the exclusive lock is held."""

  def __init__(self, interval_seconds: float):
    self.interval_seconds = float(interval_seconds)
    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None
    self._mutex = threading.Lock()
    self._sample_count = 0
    self._foreign_observations: list[dict[str, Any]] = []
    self._errors: list[str] = []
    self.preflight: list[int] = []
    self.postflight: list[int] = []

  def _sample(self) -> list[int]:
    try:
      pids = _other_compute_pids()
    except BaseException as error:
      with self._mutex:
        self._sample_count += 1
        self._errors.append(f'{type(error).__name__}: {error}')
      return []
    with self._mutex:
      self._sample_count += 1
      if pids:
        self._foreign_observations.append({
          'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
          'pids': pids,
        })
    return pids

  def _run(self) -> None:
    while not self._stop_event.wait(self.interval_seconds):
      self._sample()

  def __enter__(self):
    self.preflight = self._sample()
    if self.preflight:
      raise RuntimeError(
        f'GPU is not exclusive; active compute PIDs: {self.preflight}')
    with self._mutex:
      if self._errors:
        raise RuntimeError('GPU process monitor failed during preflight')
    self._thread = threading.Thread(
      target=self._run, name='tensor-train-gpu-pid-monitor', daemon=True)
    self._thread.start()
    return self

  def snapshot(self, *, lock_path: Path) -> dict[str, Any]:
    self.postflight = self._sample()
    with self._mutex:
      return {
        'required': True,
        'policy': GPU_EXCLUSIVITY_POLICY,
        'lock_path': str(lock_path),
        'lock_acquired': True,
        'monitor_interval_seconds': self.interval_seconds,
        'monitor_samples': self._sample_count,
        'preflight_other_compute_pids': list(self.preflight),
        'postflight_other_compute_pids': list(self.postflight),
        'foreign_pid_observations': list(self._foreign_observations),
        'monitor_errors': list(self._errors),
      }

  def __exit__(self, exc_type, exc_value, traceback) -> bool:
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=max(2.0, self.interval_seconds * 2.0))
      if self._thread.is_alive():
        with self._mutex:
          self._errors.append('GPU process monitor thread did not stop')
    if exc_type is None:
      evidence = self.snapshot(lock_path=Path('/nonfinal-monitor-snapshot'))
      if evidence['foreign_pid_observations'] or evidence['monitor_errors']:
        raise RuntimeError(
          'GPU exclusivity monitor observed a foreign process or query failure')
    return False


def _prepare_offline_cache(
    plan: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
  cache_root = Path(job['cache']['root']).expanduser().resolve()
  os.environ['HF_HOME'] = str(cache_root)
  os.environ['HF_HUB_CACHE'] = str(cache_root / 'hub')
  os.environ['HF_HUB_OFFLINE'] = '1'
  os.environ['TRANSFORMERS_OFFLINE'] = '1'
  os.environ['HF_DATASETS_OFFLINE'] = '1'
  if job['cache']['policy'] != CACHE_POLICY:
    raise RuntimeError('compiled job does not require the offline cache policy')
  protocol = load_protocol(Path(plan['protocol_path']).expanduser().resolve())
  observed = cached_model_identities(cache_root, protocol)
  if observed != job['cache']:
    raise RuntimeError('offline cached model bytes differ from compiled plan')
  return observed


def _clean_identity_matches(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    context: str,
) -> None:
  for field in ('revision', 'clean', 'origin'):
    if observed[field] != expected[field]:
      raise RuntimeError(
        f'{context} {field} differs from compiled plan: expected '
        f'{expected[field]!r}, found {observed[field]!r}')


def _prepare_determinism(seed: int):
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
  import numpy as np
  import torch

  random.seed(seed)
  np.random.seed(seed % (2 ** 32))
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(True, warn_only=False)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_float32_matmul_precision('highest')
  return torch


def _reseed_sampling(torch, seed: int) -> None:
  """Reset RNG after arm-dependent module construction and state loading."""
  import numpy as np

  random.seed(seed)
  np.random.seed(seed % (2 ** 32))
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


class _PositionScheduleRecorder:

  def __init__(self, generation: dict[str, Any]):
    self.generation = generation
    self.records: list[list[list[int]]] = [
      [] for _ in range(generation['num_samples'])]
    self.call_index = 0

  def record(
      self,
      selected_indices,
      *,
      ordering: str,
      requested_k: int,
      logprobs,
  ):
    if ordering != 'random' or logprobs is not None \
        or requested_k != self.generation['tokens_per_step']:
      raise RuntimeError('official sampler left the frozen random schedule path')
    rows = selected_indices.detach().cpu().tolist()
    expected_batch = self.generation['batch_size']
    if len(rows) != expected_batch:
      raise RuntimeError('position schedule batch size differs from job')
    step = self.call_index % self.generation['nfe_steps']
    batch_index = self.call_index // self.generation['nfe_steps']
    first_sample = batch_index * expected_batch
    if first_sample >= self.generation['num_samples']:
      raise RuntimeError('position schedule produced too many samples')
    if step != len(self.records[first_sample]):
      raise RuntimeError('position schedule call order is not batch-major')
    for row_index, positions in enumerate(rows):
      sample_id = batch_index * expected_batch + row_index
      if sample_id >= self.generation['num_samples']:
        raise RuntimeError('position schedule produced too many samples')
      values = [int(position) for position in positions]
      if len(values) != self.generation['tokens_per_step'] \
          or any(position < 0 for position in values):
        raise RuntimeError('position schedule contains a partial/invalid chunk')
      self.records[sample_id].append(values)
    self.call_index += 1
    return selected_indices

  def finalize(self, *, job: dict[str, Any]) -> dict[str, Any]:
    expected_calls = (
      self.generation['num_samples'] // self.generation['batch_size']
      * self.generation['nfe_steps'])
    if self.call_index != expected_calls:
      raise RuntimeError(
        f'position schedule call count {self.call_index} != {expected_calls}')
    records = []
    expected_positions = list(range(self.generation['sequence_length']))
    for sample_id, chunks in enumerate(self.records):
      if len(chunks) != self.generation['nfe_steps'] \
          or sorted(position for chunk in chunks for position in chunk) \
          != expected_positions:
        raise RuntimeError(
          f'position schedule for sample {sample_id} is not a permutation')
      records.append({
        'sample_id': sample_id,
        'chunks': chunks,
        'position_schedule_sha256': canonical_sha256(chunks),
      })
    return {
      'schema_version': 1,
      'artifact': 'tensor_train_owt_position_schedules',
      'job_id': job['job_id'],
      'arm': job['arm'],
      'nfe_steps': self.generation['nfe_steps'],
      'generation_seed': self.generation['generation_seed'],
      'schedule_policy': self.generation['schedule_policy'],
      'num_samples': self.generation['num_samples'],
      'sequence_length': self.generation['sequence_length'],
      'records': records,
    }


def _checkpoint_config_identity(
    config: dict[str, Any],
    *,
    job: dict[str, Any],
) -> dict[str, Any]:
  try:
    algo = config['algo']
    model_type = config['type']
    decomposition_name = algo['decomp']
    decomposition = algo[decomposition_name]
    identity = {
      'name': config['name'],
      'rank_arch': config['rank_arch'],
      'type_model': model_type['model'],
      'type_mdlm': model_type['mdlm'],
      'decomposition': decomposition_name,
      'rank': decomposition['rank'],
      'rank_weights': (
        decomposition.get('rank_weights', False)
        if decomposition_name == 'cp' else False),
      'marginal_head': (
        decomposition.get('marginal_head', False)
        if decomposition_name == 'tt' else False),
      'output_dtype': algo['output_dtype'],
      'causal_attention': algo['causal_attention'],
    }
  except (KeyError, TypeError) as error:
    raise RuntimeError('checkpoint embedded config is malformed') from error
  expected = job['decomposition']
  if identity != {
      'name': 'mdlm',
      'rank_arch': '2-layer',
      'type_model': 'mdlm',
      'type_mdlm': 'owt',
      'decomposition': expected['decomposition'],
      'rank': expected['rank'],
      'rank_weights': expected['rank_weights'],
      'marginal_head': expected['marginal_head'],
      'output_dtype': 'float32',
      'causal_attention': False,
  }:
    raise RuntimeError('checkpoint embedded config differs from audited arm')
  return identity


def _compose_official_config(job: dict[str, Any], source_root: Path):
  from hydra import compose, initialize_config_dir
  from omegaconf import OmegaConf

  generation = job['generation']
  decomposition = job['decomposition']
  overrides = [
    'name=mdlm',
    'training=mdlm',
    'generation=mdlm',
    'type.model=mdlm',
    'type.mdlm=owt',
    'rank_arch=2-layer',
    f'algo.decomp={decomposition["decomposition"]}',
    f'algo.cp.rank={decomposition["rank"] if job["arm"] == "marginal" else 1}',
    f'algo.cp.rank_weights={str(decomposition["rank_weights"]).lower()}',
    f'algo.tt.rank={decomposition["rank"] if job["arm"] != "marginal" else 4}',
    f'algo.tt.marginal_head={str(decomposition["marginal_head"]).lower()}',
    f'generation.total_samples={generation["num_samples"]}',
    f'generation.batch_size={generation["batch_size"]}',
    f'generation.length={generation["sequence_length"]}',
    f'generation.temperature={generation["temperature"]}',
    f'generation.ckpt_path={job["checkpoint"]["path"]}',
    f'generation.sampling={generation["sampling"]}',
    f'generation.k={generation["tokens_per_step"]}',
    f'generation.ordering={generation["ordering"]}',
  ]
  with initialize_config_dir(
      version_base=None, config_dir=str((source_root / 'conf').resolve())):
    config = compose(config_name='config', overrides=overrides)
  resolved = OmegaConf.to_container(config, resolve=True)
  expected = {
    'total_samples': generation['num_samples'],
    'batch_size': generation['batch_size'],
    'length': generation['sequence_length'],
    'temperature': generation['temperature'],
    'ckpt_path': job['checkpoint']['path'],
    'sampling': generation['sampling'],
    'k': generation['tokens_per_step'],
    'ordering': generation['ordering'],
  }
  for field, value in expected.items():
    if resolved['generation'][field] != value:
      raise RuntimeError(
        f'resolved official config {field} differs from compiled job')
  if resolved['algo']['decomp'] != decomposition['decomposition']:
    raise RuntimeError('resolved decomposition differs from compiled job')
  return config, OmegaConf.to_yaml(config, resolve=True)


def _import_official_source(source_root: Path):
  source_text = str(source_root)
  if source_text in sys.path:
    sys.path.remove(source_text)
  sys.path.insert(0, source_text)
  for module_name in ('generate', 'mdlm'):
    existing = sys.modules.get(module_name)
    if existing is not None:
      module_path = Path(getattr(existing, '__file__', '')).resolve()
      try:
        module_path.relative_to(source_root)
      except ValueError as error:
        raise RuntimeError(
          f'module name collision for official {module_name}: {module_path}') \
          from error
  upstream_generate = importlib.import_module('generate')
  upstream_mdlm = importlib.import_module('mdlm')
  for module in (upstream_generate, upstream_mdlm):
    module_path = Path(module.__file__).resolve()
    try:
      module_path.relative_to(source_root)
    except ValueError as error:
      raise RuntimeError(
        f'imported module is outside official source: {module_path}') from error
  return upstream_generate, upstream_mdlm


def _install_pinned_transformer_proxies(
    upstream_mdlm,
    *,
    job: dict[str, Any],
) -> None:
  from transformers import AutoModelForMaskedLM, AutoTokenizer

  tokenizer_spec = job['model_inputs']['tokenizer']
  backbone_spec = job['model_inputs']['backbone']

  class PinnedTokenizer:

    @staticmethod
    def from_pretrained(identifier, *args, **kwargs):
      if identifier != 'gpt2' or args or 'revision' in kwargs:
        raise RuntimeError('official tokenizer request differs from audited call')
      return AutoTokenizer.from_pretrained(
        tokenizer_spec['repository'],
        revision=tokenizer_spec['revision'],
        local_files_only=True,
        **kwargs)

  class PinnedMaskedLM:

    @staticmethod
    def from_pretrained(identifier, *args, **kwargs):
      if identifier != backbone_spec['repository'] or args \
          or 'revision' in kwargs:
        raise RuntimeError('official backbone request differs from audited call')
      return AutoModelForMaskedLM.from_pretrained(
        backbone_spec['repository'],
        revision=backbone_spec['revision'],
        local_files_only=True,
        **kwargs)

  upstream_mdlm.AutoTokenizer = PinnedTokenizer
  upstream_mdlm.AutoModelForMaskedLM = PinnedMaskedLM


def _token_entropy(tokens: list[list[int]]) -> float:
  counts = Counter(token for sequence in tokens for token in sequence)
  total = sum(counts.values())
  if total <= 0:
    raise ValueError('cannot compute entropy for empty tokens')
  return math.fsum(
    -(count / total) * math.log(count / total)
    for count in counts.values())


def _max_rss_bytes() -> int:
  value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  # Linux reports KiB; macOS reports bytes. Actual runs use Linux, while this
  # branch keeps local schema tests interpretable.
  return int(value * 1024 if sys.platform.startswith('linux') else value)


def _reference_scores(
    texts: list[str],
    *,
    evaluator: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  from evaluation.generation_metrics import TransformersReferenceLMScorer

  scorer = TransformersReferenceLMScorer(
    evaluator['model_name_or_path'],
    revision=evaluator['revision'],
    device='cuda',
    batch_size=evaluator['batch_size'],
    max_length=evaluator['max_length'],
    dtype=evaluator['dtype'])
  runtime_identity = scorer.runtime_identity()
  expected = {
    'model_name_or_path': evaluator['model_name_or_path'],
    'model_revision': evaluator['revision'],
    'batch_size': evaluator['batch_size'],
    'max_length': evaluator['max_length'],
    'requested_dtype': evaluator['dtype'],
    'sequence_policy': evaluator['sequence_policy'],
    'device': 'cuda',
  }
  for field, value in expected.items():
    if runtime_identity.get(field) != value:
      raise RuntimeError(
        f'reference-LM runtime {field} differs from frozen evaluator')
  scores = scorer.score(texts)
  rows = []
  total_tokens = 0
  contributions = []
  for score in scores:
    if score.token_count <= 0 or score.mean_nll_nats is None \
        or score.perplexity is None:
      raise RuntimeError('reference LM produced an unscored sample')
    rows.append({
      'model_name_or_path': evaluator['model_name_or_path'],
      'revision': evaluator['revision'],
      'sequence_policy': evaluator['sequence_policy'],
      'token_count': score.token_count,
      'mean_nll_nats': score.mean_nll_nats,
      'perplexity': score.perplexity,
    })
    total_tokens += score.token_count
    contributions.append(score.mean_nll_nats * score.token_count)
  mean_nll = math.fsum(contributions) / total_tokens
  return rows, {
    'model_name_or_path': evaluator['model_name_or_path'],
    'revision': evaluator['revision'],
    'sequence_policy': evaluator['sequence_policy'],
    'runtime_identity': runtime_identity,
    'num_scored_sequences': len(rows),
    'num_scored_tokens': total_tokens,
    'mean_nll_nats': mean_nll,
    'perplexity': math.exp(min(mean_nll, 80.0)),
  }


def _execute_monitored(
    *,
    plan_path: Path,
    plan: dict[str, Any],
    job: dict[str, Any],
    temporary_dir: Path,
    lock_path: Path,
    gpu_monitor: _ForeignPidMonitor,
    cache_identity: dict[str, Any],
) -> None:
  started = dt.datetime.now(dt.timezone.utc)
  total_started = time.perf_counter()
  runtime = job['runtime']
  if '.'.join(platform.python_version_tuple()[:2]) \
      != runtime['python_major_minor']:
    raise RuntimeError(
      f'Python must equal {runtime["python_major_minor"]}, found '
      f'{platform.python_version()}')
  packages = _package_versions(runtime['critical_packages'])
  gpu_identity = _gpu_identity()

  source_root = Path(job['source']['path']).resolve()
  source_identity = clean_git_identity(source_root)
  _clean_identity_matches(source_identity, job['source'], context='source')
  harness_identity = clean_git_identity(REPO_ROOT)
  _clean_identity_matches(
    harness_identity, job['harness_repository'], context='harness')
  checkpoint_path = Path(job['checkpoint']['path']).resolve()
  if not checkpoint_path.is_file() \
      or sha256_file(checkpoint_path) != job['checkpoint']['sha256'] \
      or checkpoint_path.stat().st_size != job['checkpoint']['size_bytes']:
    raise RuntimeError('checkpoint bytes differ from compiled plan')

  torch = _prepare_determinism(job['generation']['generation_seed'])
  if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required but unavailable')
  if torch.cuda.device_count() != 1:
    raise RuntimeError('exactly one CUDA device must be visible')
  torch.cuda.set_device(0)
  config, resolved_config = _compose_official_config(job, source_root)
  config_path = temporary_dir / 'resolved_config.yaml'
  _atomic_write(config_path, resolved_config)
  upstream_generate, upstream_mdlm = _import_official_source(source_root)
  _install_pinned_transformer_proxies(upstream_mdlm, job=job)

  torch.cuda.reset_peak_memory_stats()
  load_started = time.perf_counter()
  model = upstream_mdlm.MDLM(config=config).to('cuda')
  payload = torch.load(
    checkpoint_path, map_location='cuda', weights_only=True)
  if not isinstance(payload, dict) \
      or sorted(payload) != ['config', 'model', 'optimizer', 'scheduler', 'step']:
    raise RuntimeError('checkpoint payload keys differ from audited release')
  if payload['step'] != EXPECTED_CHECKPOINT_STEPS[job['arm']]:
    raise RuntimeError('checkpoint step differs from audited release')
  if not isinstance(payload['config'], dict):
    raise RuntimeError('checkpoint embedded config must be an object')
  checkpoint_config_identity = _checkpoint_config_identity(
    payload['config'], job=job)
  checkpoint_config_sha256 = canonical_sha256(payload['config'])
  if checkpoint_config_sha256 != EXPECTED_CHECKPOINT_CONFIG_SHA256[job['arm']]:
    raise RuntimeError('checkpoint embedded config hash differs from release')
  checkpoint_state = payload['model']
  expected_state_keys = list(EXPECTED_CHECKPOINT_STATE_KEYS[job['arm']])
  if not isinstance(checkpoint_state, dict) \
      or sorted(checkpoint_state) != expected_state_keys:
    raise RuntimeError('checkpoint learned state keys differ from audited release')
  missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint_state, strict=False)
  if unexpected_keys:
    raise RuntimeError(
      f'checkpoint contains unexpected model keys: {unexpected_keys[:10]}')
  if set(expected_state_keys) & set(missing_keys):
    raise RuntimeError('a released learned state key was not loaded')
  model.eval()
  torch.cuda.synchronize()
  model_load_seconds = time.perf_counter() - load_started
  model_load_peak_allocated = torch.cuda.max_memory_allocated()
  model_load_peak_reserved = torch.cuda.max_memory_reserved()
  model_identity = {
    'class': f'{type(model).__module__}.{type(model).__qualname__}',
    'backbone_class': (
      f'{type(model.backbone).__module__}.{type(model.backbone).__qualname__}'),
    'tokenizer_class': (
      f'{type(model.tokenizer).__module__}.{type(model.tokenizer).__qualname__}'),
    'mask_token_id': model.mask_id,
    'tokenizer_vocab_size': len(model.tokenizer),
    'ema_used': False,
    'checkpoint_step': payload['step'],
    'checkpoint_payload_keys': sorted(payload),
    'checkpoint_state_keys': sorted(checkpoint_state),
    'checkpoint_config_sha256': checkpoint_config_sha256,
    'checkpoint_config_identity': checkpoint_config_identity,
    'missing_keys': sorted(missing_keys),
    'unexpected_keys': [],
    'parameter_dtypes': sorted({
      str(parameter.dtype) for parameter in model.parameters()
    }),
    'decomposition': job['decomposition'],
    'backbone_input': job['model_inputs']['backbone'],
    'tokenizer_input': job['model_inputs']['tokenizer'],
    'cache_identity_sha256': cache_identity['identity_sha256'],
  }
  if model.mask_id != 50257 or len(model.tokenizer) != 50257:
    raise RuntimeError('loaded tokenizer/model mask identity is unexpected')

  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()
  _reseed_sampling(torch, job['generation']['generation_seed'])
  schedule_recorder = _PositionScheduleRecorder(job['generation'])
  original_pick_tokens = upstream_generate.pick_tokens_to_unmask

  def recorded_pick_tokens(ordering, x, K, mask_id, logprobs=None):
    selected = original_pick_tokens(
      ordering, x, K, mask_id, logprobs=logprobs)
    return schedule_recorder.record(
      selected,
      ordering=ordering,
      requested_k=K,
      logprobs=logprobs)

  upstream_generate.pick_tokens_to_unmask = recorded_pick_tokens
  generation_started = time.perf_counter()
  try:
    with torch.inference_mode():
      generated, observed_steps = upstream_generate.sample(
        model, config, model.mask_id)
  finally:
    upstream_generate.pick_tokens_to_unmask = original_pick_tokens
  torch.cuda.synchronize()
  generation_seconds = time.perf_counter() - generation_started
  generation_peak_allocated = torch.cuda.max_memory_allocated()
  generation_peak_reserved = torch.cuda.max_memory_reserved()
  expected_shape = (
    job['generation']['num_samples'], job['generation']['sequence_length'])
  if tuple(generated.shape) != expected_shape:
    raise RuntimeError(
      f'generated tensor shape mismatch: expected {expected_shape}, '
      f'found {tuple(generated.shape)}')
  if not math.isclose(
      float(observed_steps), float(job['generation']['nfe_steps']),
      abs_tol=1e-6):
    raise RuntimeError(
      f'observed {observed_steps} steps for requested '
      f'{job["generation"]["nfe_steps"]}')
  token_rows = generated.detach().cpu().tolist()
  if any(token < 0 or token >= model.mask_id
         for row in token_rows for token in row):
    raise RuntimeError('generated output contains invalid or masked tokens')
  texts = model.tokenizer.batch_decode(token_rows, skip_special_tokens=True)
  if len(texts) != job['generation']['num_samples']:
    raise RuntimeError('decoded sample count mismatch')
  position_schedules = schedule_recorder.finalize(job=job)
  position_schedules_path = temporary_dir / 'position-schedules.json'
  _atomic_write(
    position_schedules_path,
    json.dumps(position_schedules, indent=2, sort_keys=True) + '\n')
  del generated
  del model
  del payload
  torch.cuda.empty_cache()

  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()
  evaluator_started = time.perf_counter()
  score_rows, score_summary = _reference_scores(
    texts, evaluator=job['evaluator'])
  torch.cuda.synchronize()
  evaluator_seconds = time.perf_counter() - evaluator_started
  evaluator_peak_allocated = torch.cuda.max_memory_allocated()
  evaluator_peak_reserved = torch.cuda.max_memory_reserved()

  sample_records = []
  for index, (tokens, text, score) in enumerate(
      zip(token_rows, texts, score_rows)):
    sample_records.append({
      'schema_version': 1,
      'sample_id': index,
      'job_id': job['job_id'],
      'arm': job['arm'],
      'nfe_steps': job['generation']['nfe_steps'],
      'generation_seed': job['generation']['generation_seed'],
      'token_ids': tokens,
      'token_ids_sha256': canonical_sha256(tokens),
      'text': text,
      'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
      'position_schedule_sha256': position_schedules['records'][index][
        'position_schedule_sha256'],
      'reference_lm': score,
    })
  samples_path = temporary_dir / 'samples.jsonl'
  _atomic_write(
    samples_path,
    ''.join(json.dumps(row, sort_keys=True) + '\n' for row in sample_records))
  metrics = {
    'schema_version': 1,
    'artifact': 'tensor_train_owt_feasibility_metrics',
    'job_id': job['job_id'],
    'arm': job['arm'],
    'nfe_steps': job['generation']['nfe_steps'],
    'num_samples': job['generation']['num_samples'],
    'sequence_length': job['generation']['sequence_length'],
    'generation_seed': job['generation']['generation_seed'],
    'token_entropy_nats': _token_entropy(token_rows),
    'reference_lm': score_summary,
  }
  metrics_path = temporary_dir / 'metrics.json'
  _atomic_write(
    metrics_path, json.dumps(metrics, indent=2, sort_keys=True) + '\n')

  gpu_exclusivity = gpu_monitor.snapshot(lock_path=lock_path)
  if gpu_exclusivity['foreign_pid_observations'] \
      or gpu_exclusivity['monitor_errors']:
    raise RuntimeError('GPU exclusivity was lost during the measured run')
  total_seconds = time.perf_counter() - total_started
  resource_metrics = {
    'schema_version': 1,
    'artifact': 'tensor_train_owt_resource_metrics',
    'job_id': job['job_id'],
    'measurement_scope': 'single_job_uncontended_end_to_end_v1',
    'host': {
      'hostname': platform.node(),
      'platform': platform.platform(),
      'python': platform.python_version(),
      'torch': torch.__version__,
      'cuda_runtime': torch.version.cuda,
      'gpu': gpu_identity,
      'critical_packages': packages,
      'precision_policy': runtime['precision_policy'],
    },
    'timing_seconds': {
      'model_load': model_load_seconds,
      'generation': generation_seconds,
      'evaluator_load_and_scoring': evaluator_seconds,
      'total': total_seconds,
    },
    'throughput': {
      'generation_samples_per_second': (
        job['generation']['num_samples'] / generation_seconds),
      'generation_tokens_per_second': (
        job['generation']['num_samples']
        * job['generation']['sequence_length'] / generation_seconds),
      'evaluator_samples_per_second': (
        job['generation']['num_samples'] / evaluator_seconds),
    },
    'cuda_memory_bytes': {
      'model_load_peak_allocated': model_load_peak_allocated,
      'model_load_peak_reserved': model_load_peak_reserved,
      'generation_peak_allocated': generation_peak_allocated,
      'generation_peak_reserved': generation_peak_reserved,
      'evaluator_peak_allocated': evaluator_peak_allocated,
      'evaluator_peak_reserved': evaluator_peak_reserved,
    },
    'process': {
      'pid': os.getpid(),
      'max_rss_bytes': _max_rss_bytes(),
    },
    'generation': {
      'requested_nfe_steps': job['generation']['nfe_steps'],
      'observed_mean_steps': observed_steps,
      'tokens_per_step': job['generation']['tokens_per_step'],
      'num_samples': job['generation']['num_samples'],
      'sequence_length': job['generation']['sequence_length'],
      'batch_size': job['generation']['batch_size'],
    },
    'gpu_exclusivity': gpu_exclusivity,
  }
  resource_path = temporary_dir / 'resource-metrics.json'
  _atomic_write(
    resource_path,
    json.dumps(resource_metrics, indent=2, sort_keys=True) + '\n')

  ended = dt.datetime.now(dt.timezone.utc)
  manifest = {
    'schema_version': 1,
    'artifact': 'tensor_train_owt_feasibility_run',
    'scientific_scope': plan['scientific_scope'],
    'job_id': job['job_id'],
    'arm': job['arm'],
    'nfe_steps': job['generation']['nfe_steps'],
    'job_spec_sha256': job['job_spec_sha256'],
    'plan_id': plan['plan_id'],
    'plan_file_sha256': sha256_file(plan_path),
    'start_time_utc': started.isoformat(),
    'end_time_utc': ended.isoformat(),
    'runtime': resource_metrics['host'],
    'source': job['source'],
    'checkpoint': job['checkpoint'],
    'harness_repository': job['harness_repository'],
    'model_inputs': job['model_inputs'],
    'cache': cache_identity,
    'model_load': model_identity,
    'evaluator': score_summary,
    'outputs': {
      'samples_jsonl': _output_descriptor(samples_path),
      'metrics_json': _output_descriptor(metrics_path),
      'resource_metrics_json': _output_descriptor(resource_path),
      'resolved_config_yaml': _output_descriptor(config_path),
      'position_schedules_json': _output_descriptor(position_schedules_path),
    },
    'interruption_policy': runtime['interruption_policy'],
  }
  manifest_path = temporary_dir / 'manifest.json'
  _atomic_write(
    manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  success = {
    'schema_version': 1,
    'artifact': 'tensor_train_owt_feasibility_success',
    'job_id': job['job_id'],
    'job_spec_sha256': job['job_spec_sha256'],
    'manifest_sha256': sha256_file(manifest_path),
  }
  _atomic_write(
    temporary_dir / '_SUCCESS.json',
    json.dumps(success, indent=2, sort_keys=True) + '\n')


def _execute(
    *,
    plan_path: Path,
    plan: dict[str, Any],
    job: dict[str, Any],
    temporary_dir: Path,
    lock_path: Path,
) -> None:
  cache_identity = _prepare_offline_cache(plan, job)
  interval = float(job['runtime']['gpu_monitor_interval_seconds'])
  with _ForeignPidMonitor(interval) as gpu_monitor:
    _execute_monitored(
      plan_path=plan_path,
      plan=plan,
      job=job,
      temporary_dir=temporary_dir,
      lock_path=lock_path,
      gpu_monitor=gpu_monitor,
      cache_identity=cache_identity)


def run_job(plan_path: Path, job_id: str, *, resume: bool) -> dict[str, Any]:
  plan_path = plan_path.expanduser().resolve()
  plan, jobs = load_compiled_plan(plan_path)
  if job_id not in jobs:
    raise KeyError(f'unknown compiled job: {job_id}')
  job = jobs[job_id]
  output_dir = Path(job['artifact_dir']).resolve()
  if output_dir.exists():
    if not resume:
      raise FileExistsError(
        f'refusing to overwrite existing job output {output_dir}')
    validated = validate_completed_run(
      output_dir,
      plan=plan,
      job=job,
      expected_plan_file_sha256=sha256_file(plan_path))
    return {
      'event': 'tensor_train_feasibility_job_reused',
      'job_id': job_id,
      'output_dir': str(output_dir),
      'manifest_sha256': validated['manifest_sha256'],
    }
  output_dir.parent.mkdir(parents=True, exist_ok=True)
  temporary_dir = output_dir.parent / (
    f'.{job_id}.partial-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-'
    f'{uuid.uuid4().hex[:12]}')
  temporary_dir.mkdir(parents=False, exist_ok=False)
  try:
    with _exclusive_gpu_lock(SUBMISSION_GPU_LOCK) as acquired_lock_path:
      _execute(
        plan_path=plan_path,
        plan=plan,
        job=job,
        temporary_dir=temporary_dir,
        lock_path=acquired_lock_path)
    os.rename(temporary_dir, output_dir)
  except BaseException:
    # Intentionally preserve partial attempts. Never append or silently retry.
    raise
  validated = validate_completed_run(
    output_dir,
    plan=plan,
    job=job,
    expected_plan_file_sha256=sha256_file(plan_path))
  return {
    'event': 'tensor_train_feasibility_job_complete',
    'job_id': job_id,
    'output_dir': str(output_dir),
    'manifest_sha256': validated['manifest_sha256'],
  }


def main(argv=None) -> int:
  args = _parse_args(argv)
  result = run_job(args.plan, args.job_id, resume=args.resume)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
