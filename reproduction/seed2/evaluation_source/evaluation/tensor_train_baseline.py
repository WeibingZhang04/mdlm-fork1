"""Fail-closed planning and artifact validation for the Tensor-Train baseline.

This module deliberately has no Torch or Transformers import.  Plan compilation
and completed-run replay therefore remain usable on a laptop without the
official baseline environment or checkpoints installed.
"""

from __future__ import annotations

from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping

import yaml


PROTOCOL_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
JOB_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
PROTOCOL_ID = 'tensor-train-owt-feasibility-v1'
OFFICIAL_SOURCE_REVISION = '9d0087afd3771ac3e94898ed842858fcc81fb3b0'
OFFICIAL_CHECKPOINT_REVISION = '09c0042df4be9608de81302d79c16d08a95889db'
REFERENCE_LM_SEQUENCE_POLICY = (
  'retokenize_decoded_text_score_through_first_nonleading_eos_v1')
REFERENCE_LM_PRECISION_POLICY = (
  'explicit_checkpoint_dtype_no_autocast_float32_cross_entropy_v1')
REFERENCE_LM_TOKENIZATION_POLICY = (
  'fast_tokenizer_right_padding_right_truncation_add_special_tokens_v1')
SCHEDULE_POLICY = 'recorded_selected_position_chunks_v1'
CACHE_POLICY = 'pinned_snapshot_offline_warm_cache_v1'
GPU_EXCLUSIVITY_POLICY = (
  'nonblocking_flock_and_continuous_pid_monitor_v1')
SUBMISSION_GPU_LOCK = Path(
  '/mnt/contextual-forest/experiments/.submission-gpu.lock')
EXPECTED_CHECKPOINT_STEPS = {
  'marginal': 599999,
  'tensor_train_rank4': 149999,
}
EXPECTED_CHECKPOINT_CONFIG_SHA256 = {
  'marginal': (
    '0834225cb5049ebd56c1e088e74b925624a76e5dc1a2cf8482702ed1148e70d0'),
  'tensor_train_rank4': (
    '405fd1c9d02237ce34028bc3833b2ff04a4726f269bcaebfbb94f69ae231d0fd'),
}
EXPECTED_CHECKPOINT_STATE_KEYS = {
  'marginal': (
    'rank_layer.bias',
    'rank_layer.weight',
  ),
  'tensor_train_rank4': (
    'marginal_layer.0.bias',
    'marginal_layer.0.weight',
    'marginal_layer.2.bias',
    'marginal_layer.2.weight',
    'marginal_layer.4.bias',
    'marginal_layer.4.weight',
    'rank_layer.bias',
    'rank_layer.weight',
  ),
}
HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
JOB_ID = re.compile(
  r'^owt--(marginal|tensor_train_rank4)--nfe(008|016|032)--s260703$')


def canonical_json(value: object) -> str:
  return json.dumps(
    value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def canonical_sha256(value: object) -> str:
  return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(chunk_size), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _exact_keys(
    value: object,
    expected: Iterable[str],
    *,
    context: str,
) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise TypeError(f'{context} must be an object')
  expected_set = set(expected)
  actual = set(value)
  missing = sorted(expected_set - actual)
  unknown = sorted(actual - expected_set)
  if missing or unknown:
    raise ValueError(
      f'{context} schema mismatch: missing={missing}, unknown={unknown}')
  return value


def _positive_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f'{context} must be a positive integer')
  return value


def _nonnegative_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f'{context} must be a nonnegative integer')
  return value


def _finite_float(value: object, *, context: str) -> float:
  if (not isinstance(value, (int, float)) or isinstance(value, bool)
      or not math.isfinite(float(value))):
    raise ValueError(f'{context} must be finite')
  return float(value)


def _hex(value: object, length: int, *, context: str) -> str:
  pattern = HEX40 if length == 40 else HEX64
  if not isinstance(value, str) or pattern.fullmatch(value) is None:
    raise ValueError(
      f'{context} must be {length} lowercase hexadecimal digits')
  return value


def _nonempty_string(value: object, *, context: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f'{context} must be a non-empty string')
  return value


def _read_yaml_or_json(path: Path) -> object:
  if not path.is_file():
    raise FileNotFoundError(path)
  with path.open() as handle:
    return yaml.safe_load(handle)


def _read_json(path: Path) -> object:
  if not path.is_file():
    raise FileNotFoundError(path)
  try:
    with path.open() as handle:
      return json.load(handle)
  except json.JSONDecodeError as error:
    raise ValueError(f'invalid JSON at {path}: {error}') from error


def _safe_absolute_child(path: Path, root: Path, *, context: str) -> Path:
  if not path.is_absolute() or not root.is_absolute():
    raise ValueError(f'{context} and its root must be absolute')
  resolved = path.resolve(strict=False)
  resolved_root = root.resolve(strict=False)
  if resolved == Path(resolved.anchor) or resolved == resolved_root:
    raise ValueError(f'{context} is too broad: {resolved}')
  try:
    resolved.relative_to(resolved_root)
  except ValueError as error:
    raise ValueError(
      f'{context} {resolved} is outside root {resolved_root}') from error
  return resolved


def _validate_unique_int_list(
    value: object,
    *,
    context: str,
) -> list[int]:
  if not isinstance(value, list) or not value:
    raise ValueError(f'{context} must be a non-empty list')
  values = [
    _positive_int(item, context=f'{context}[{index}]')
    for index, item in enumerate(value)]
  if len(set(values)) != len(values):
    raise ValueError(f'{context} contains duplicates')
  return values


def load_protocol(path: Path) -> dict[str, Any]:
  """Load and strictly validate the frozen feasibility protocol."""
  protocol = _exact_keys(_read_yaml_or_json(path), {
    'schema_version', 'protocol_id', 'protocol_status', 'scientific_scope',
    'source', 'checkpoints', 'model_inputs', 'generation', 'evaluator',
    'runtime',
  }, context='protocol')
  if protocol['schema_version'] != PROTOCOL_SCHEMA_VERSION:
    raise ValueError('protocol.schema_version must equal 1')
  if protocol['protocol_id'] != PROTOCOL_ID:
    raise ValueError(f'protocol_id must equal {PROTOCOL_ID!r}')
  if protocol['protocol_status'] != 'frozen_before_execution':
    raise ValueError('protocol_status must be frozen_before_execution')
  _nonempty_string(protocol['scientific_scope'], context='scientific_scope')

  source = _exact_keys(protocol['source'], {
    'repository', 'revision', 'require_clean_checkout',
  }, context='source')
  if source['repository'] != 'https://github.com/ssamt/tensor-train.git':
    raise ValueError('source.repository is not the audited official repository')
  if _hex(source['revision'], 40, context='source.revision') \
      != OFFICIAL_SOURCE_REVISION:
    raise ValueError('source.revision is not the audited official revision')
  if source['require_clean_checkout'] is not True:
    raise ValueError('source.require_clean_checkout must be true')

  checkpoints = _exact_keys(protocol['checkpoints'], {
    'repository', 'revision', 'arms',
  }, context='checkpoints')
  if checkpoints['repository'] != 'ssamt/tensor-train':
    raise ValueError('checkpoints.repository is not official')
  if _hex(checkpoints['revision'], 40, context='checkpoints.revision') \
      != OFFICIAL_CHECKPOINT_REVISION:
    raise ValueError('checkpoints.revision is not the audited revision')
  arms = _exact_keys(
    checkpoints['arms'], {'marginal', 'tensor_train_rank4'},
    context='checkpoints.arms')
  expected_arms = {
    'marginal': {
      'relative_path': 'owt/marginal.pt',
      'sha256': (
        '84fc03cacd818df293602987d4367b8ead7c96539b9b536b748bc86a6cd7079c'),
      'decomposition': 'cp',
      'rank': 1,
      'rank_weights': False,
      'marginal_head': False,
    },
    'tensor_train_rank4': {
      'relative_path': 'owt/ttd_4_marg.pt',
      'sha256': (
        '8ad8d956af127795686489e9f3496e7a634da18ecf79464df1319668a2a3a7a2'),
      'decomposition': 'tt',
      'rank': 4,
      'rank_weights': False,
      'marginal_head': True,
    },
  }
  for arm_name, expected in expected_arms.items():
    arm = _exact_keys(arms[arm_name], expected, context=f'arms.{arm_name}')
    if dict(arm) != expected:
      raise ValueError(f'arms.{arm_name} differs from the audited checkpoint')

  model_inputs = _exact_keys(
    protocol['model_inputs'], {'backbone', 'tokenizer'},
    context='model_inputs')
  expected_model_inputs = {
    'backbone': {
      'repository': 'kuleshov-group/mdlm-owt',
      'revision': 'd0958fa851335ece6c15260ce0025f030673c0fb',
    },
    'tokenizer': {
      'repository': 'openai-community/gpt2',
      'revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
    },
  }
  for name, expected in expected_model_inputs.items():
    observed = _exact_keys(
      model_inputs[name], {'repository', 'revision'},
      context=f'model_inputs.{name}')
    _hex(observed['revision'], 40, context=f'model_inputs.{name}.revision')
    if dict(observed) != expected:
      raise ValueError(f'model_inputs.{name} differs from the frozen input')

  generation = _exact_keys(protocol['generation'], {
    'dataset_label', 'num_samples', 'sequence_length', 'nfe_steps',
    'ordering', 'temperature', 'sampling', 'batch_size', 'generation_seed',
    'deterministic_algorithms', 'require_exclusive_gpu', 'schedule_policy',
  }, context='generation')
  if generation['dataset_label'] != 'openwebtext':
    raise ValueError('generation.dataset_label must equal openwebtext')
  if _positive_int(generation['num_samples'], context='num_samples') != 256:
    raise ValueError('generation.num_samples must equal 256')
  length = _positive_int(generation['sequence_length'], context='sequence_length')
  if length != 1024:
    raise ValueError('generation.sequence_length must equal 1024')
  steps = _validate_unique_int_list(
    generation['nfe_steps'], context='generation.nfe_steps')
  if steps != [8, 16, 32]:
    raise ValueError('generation.nfe_steps must equal [8, 16, 32]')
  if any(length % step for step in steps):
    raise ValueError('every NFE budget must divide the sequence length')
  if generation['ordering'] != 'random':
    raise ValueError('generation.ordering must equal random')
  if _finite_float(generation['temperature'], context='temperature') != 1.0:
    raise ValueError('generation.temperature must equal 1.0')
  if generation['sampling'] != 'top-k':
    raise ValueError('generation.sampling must equal top-k')
  batch_size = _positive_int(generation['batch_size'], context='batch_size')
  if batch_size != 1:
    raise ValueError('generation.batch_size must equal 1 for the L4 protocol')
  if generation['generation_seed'] != 260703:
    raise ValueError('generation.generation_seed must equal 260703')
  if generation['deterministic_algorithms'] is not True:
    raise ValueError('generation.deterministic_algorithms must be true')
  if generation['require_exclusive_gpu'] is not True:
    raise ValueError('generation.require_exclusive_gpu must be true')
  if generation['schedule_policy'] != SCHEDULE_POLICY:
    raise ValueError('generation.schedule_policy differs from the frozen policy')

  evaluator = _exact_keys(protocol['evaluator'], {
    'model_name_or_path', 'revision', 'batch_size', 'max_length', 'dtype',
    'sequence_policy',
  }, context='evaluator')
  if evaluator['model_name_or_path'] != 'gpt2-large':
    raise ValueError('evaluator model must equal gpt2-large')
  _hex(evaluator['revision'], 40, context='evaluator.revision')
  if evaluator['revision'] != \
      '32b71b12589c2f8d625668d2335a01cac3249519':
    raise ValueError('evaluator.revision differs from the frozen revision')
  if _positive_int(evaluator['batch_size'], context='evaluator.batch_size') != 8:
    raise ValueError('evaluator.batch_size must equal 8')
  if _positive_int(evaluator['max_length'], context='evaluator.max_length') \
      != 1024:
    raise ValueError('evaluator.max_length must equal 1024')
  if evaluator['dtype'] != 'float32':
    raise ValueError('evaluator.dtype must equal float32')
  if evaluator['sequence_policy'] != REFERENCE_LM_SEQUENCE_POLICY:
    raise ValueError('evaluator.sequence_policy differs from the frozen policy')

  runtime = _exact_keys(protocol['runtime'], {
    'python_major_minor', 'critical_packages', 'device', 'precision_policy',
    'cache_policy', 'gpu_exclusivity_policy',
    'gpu_monitor_interval_seconds', 'interruption_policy',
  }, context='runtime')
  if runtime['python_major_minor'] != '3.12':
    raise ValueError('runtime.python_major_minor must equal 3.12')
  expected_packages = {
    'torch': '2.3.1',
    'transformers': '4.46.2',
    'tokenizers': '0.20.3',
    'flash-attn': '2.7.4.post1',
    'packaging': '23.2',
    'lightning': '2.2.1',
    'pytorch-lightning': '2.2.1',
    'torchmetrics': '1.3.2',
    'lightning-utilities': '0.15.3',
    'fsspec': '2024.2.0',
    'setuptools': '69.5.1',
    'timm': '0.9.16',
    'torchvision': '0.18.1',
  }
  packages = _exact_keys(
    runtime['critical_packages'], expected_packages,
    context='runtime.critical_packages')
  if dict(packages) != expected_packages:
    raise ValueError('runtime.critical_packages differs from audited pins')
  if runtime['device'] != 'cuda':
    raise ValueError('runtime.device must equal cuda')
  if runtime['precision_policy'] != 'upstream_float32_no_autocast':
    raise ValueError('runtime.precision_policy differs from the frozen policy')
  if runtime['cache_policy'] != CACHE_POLICY:
    raise ValueError('runtime.cache_policy differs from the frozen policy')
  if runtime['gpu_exclusivity_policy'] != GPU_EXCLUSIVITY_POLICY:
    raise ValueError(
      'runtime.gpu_exclusivity_policy differs from the frozen policy')
  if _finite_float(
      runtime['gpu_monitor_interval_seconds'],
      context='runtime.gpu_monitor_interval_seconds') != 1.0:
    raise ValueError('runtime.gpu_monitor_interval_seconds must equal 1.0')
  _nonempty_string(
    runtime['interruption_policy'], context='runtime.interruption_policy')
  return dict(protocol)


def _git_output(repo: Path, *args: str) -> str:
  try:
    return subprocess.check_output(
      ['git', *args], cwd=repo, text=True,
      stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.CalledProcessError) as error:
    raise ValueError(f'cannot inspect Git checkout at {repo}') from error


def clean_git_identity(
    repo: Path,
    *,
    expected_revision: str | None = None,
    expected_remote: str | None = None,
) -> dict[str, Any]:
  repo = repo.expanduser().resolve()
  if not repo.is_dir():
    raise FileNotFoundError(repo)
  revision = _git_output(repo, 'rev-parse', 'HEAD')
  _hex(revision, 40, context='Git revision')
  if expected_revision is not None and revision != expected_revision:
    raise ValueError(
      f'Git revision mismatch: expected {expected_revision}, found {revision}')
  if _git_output(repo, 'status', '--porcelain=v1'):
    raise ValueError(f'Git checkout is dirty: {repo}')
  remote = _git_output(repo, 'remote', 'get-url', 'origin')
  normalized_remote = remote[:-4] if remote.endswith('.git') else remote
  if normalized_remote.startswith('git@github.com:'):
    normalized_remote = 'https://github.com/' + normalized_remote.split(':', 1)[1]
  if expected_remote is not None:
    normalized_expected = (
      expected_remote[:-4]
      if expected_remote.endswith('.git') else expected_remote)
    if normalized_remote != normalized_expected:
      raise ValueError(
        f'Git origin mismatch: expected {expected_remote}, found {remote}')
  return {
    'path': str(repo),
    'revision': revision,
    'clean': True,
    'origin': remote,
  }


def checkpoint_identities(
    checkpoint_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
  checkpoint_root = checkpoint_root.expanduser().resolve()
  if not checkpoint_root.is_dir():
    raise FileNotFoundError(checkpoint_root)
  result = {}
  for arm_name, specification in protocol['checkpoints']['arms'].items():
    path = (checkpoint_root / specification['relative_path']).resolve()
    try:
      path.relative_to(checkpoint_root)
    except ValueError as error:
      raise ValueError('checkpoint path escapes checkpoint root') from error
    if not path.is_file():
      raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != specification['sha256']:
      raise ValueError(
        f'{arm_name} checkpoint SHA256 mismatch: expected '
        f'{specification["sha256"]}, found {actual}')
    result[arm_name] = {
      'path': str(path),
      'sha256': actual,
      'size_bytes': path.stat().st_size,
      'huggingface_repository': protocol['checkpoints']['repository'],
      'huggingface_revision': protocol['checkpoints']['revision'],
      'relative_path': specification['relative_path'],
    }
  return result


def _cache_specs(protocol: Mapping[str, Any]) -> dict[str, dict[str, str]]:
  return {
    'backbone': dict(protocol['model_inputs']['backbone']),
    'tokenizer': dict(protocol['model_inputs']['tokenizer']),
    'evaluator': {
      'repository': protocol['evaluator']['model_name_or_path'],
      'revision': protocol['evaluator']['revision'],
    },
  }


def _snapshot_file_manifest(
    snapshot_path: Path,
    *,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], str]:
  files = []
  for path in sorted(snapshot_path.rglob('*')):
    if not path.is_file():
      continue
    try:
      path.resolve().relative_to(cache_root)
    except ValueError as error:
      raise ValueError(
        f'cached snapshot file escapes cache root: {path}') from error
    relative = path.relative_to(snapshot_path).as_posix()
    files.append({
      'path': relative,
      'size_bytes': path.stat().st_size,
      'sha256': sha256_file(path),
    })
  if not files:
    raise ValueError(f'cached snapshot contains no files: {snapshot_path}')
  return files, canonical_sha256(files)


def cached_model_identities(
    cache_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
  """Resolve and hash all three pinned HF snapshots without network access."""
  cache_root = cache_root.expanduser().resolve()
  if not cache_root.is_dir() or cache_root == Path(cache_root.anchor):
    raise ValueError('cache_root must be an existing absolute non-root directory')
  try:
    from huggingface_hub import snapshot_download
  except ImportError as error:
    raise RuntimeError('huggingface_hub is required to inspect the cache') from error
  snapshots = {}
  for name, spec in _cache_specs(protocol).items():
    try:
      raw_snapshot_path = snapshot_download(
        repo_id=spec['repository'],
        revision=spec['revision'],
        cache_dir=str(cache_root / 'hub'),
        local_files_only=True)
    except Exception as error:
      raise RuntimeError(
        f'pinned {name} snapshot is not fully cached offline') from error
    snapshot_path = Path(raw_snapshot_path).expanduser().resolve()
    try:
      snapshot_path.relative_to(cache_root)
    except ValueError as error:
      raise ValueError(
        f'{name} snapshot escapes cache root: {snapshot_path}') from error
    snapshot_revision = snapshot_path.name
    if snapshot_revision != spec['revision']:
      raise ValueError(
        f'{name} cache resolved revision {snapshot_revision!r}, expected '
        f'{spec["revision"]!r}')
    files, files_manifest_sha256 = _snapshot_file_manifest(
      snapshot_path, cache_root=cache_root)
    snapshots[name] = {
      'repository': spec['repository'],
      'revision': spec['revision'],
      'snapshot_path': str(snapshot_path),
      'snapshot_revision': snapshot_revision,
      'files': files,
      'files_manifest_sha256': files_manifest_sha256,
    }
  identity = {
    'root': str(cache_root),
    'policy': CACHE_POLICY,
    'snapshots': snapshots,
  }
  identity['identity_sha256'] = canonical_sha256(identity)
  return identity


def validate_cache_identity(
    value: object,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
  cache = _exact_keys(value, {
    'root', 'policy', 'snapshots', 'identity_sha256',
  }, context='cache identity')
  root = Path(_nonempty_string(cache['root'], context='cache.root')).expanduser()
  if not root.is_absolute() or root == Path(root.anchor):
    raise ValueError('cache.root must be an absolute non-root directory')
  if cache['policy'] != CACHE_POLICY:
    raise ValueError('cache policy differs from the frozen protocol')
  snapshots = _exact_keys(
    cache['snapshots'], {'backbone', 'tokenizer', 'evaluator'},
    context='cache.snapshots')
  expected_specs = _cache_specs(protocol)
  for name, expected in expected_specs.items():
    snapshot = _exact_keys(snapshots[name], {
      'repository', 'revision', 'snapshot_path', 'snapshot_revision',
      'files', 'files_manifest_sha256',
    }, context=f'cache.snapshots.{name}')
    if snapshot['repository'] != expected['repository'] \
        or snapshot['revision'] != expected['revision'] \
        or snapshot['snapshot_revision'] != expected['revision']:
      raise ValueError(f'cache {name} input identity differs from the protocol')
    snapshot_path = Path(_nonempty_string(
      snapshot['snapshot_path'], context=f'cache.{name}.snapshot_path'))
    if not snapshot_path.is_absolute():
      raise ValueError(f'cache {name} snapshot path must be absolute')
    try:
      snapshot_path.expanduser().resolve(strict=False).relative_to(
        root.expanduser().resolve(strict=False))
    except ValueError as error:
      raise ValueError(f'cache {name} snapshot path escapes cache root') \
        from error
    files = snapshot['files']
    if not isinstance(files, list) or not files:
      raise ValueError(f'cache {name} files must be a non-empty list')
    seen_paths = set()
    for index, raw_file in enumerate(files):
      item = _exact_keys(raw_file, {'path', 'size_bytes', 'sha256'},
                         context=f'cache.{name}.files[{index}]')
      relative = Path(_nonempty_string(
        item['path'], context=f'cache.{name}.files[{index}].path'))
      if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'cache {name} contains an unsafe file path')
      if relative.as_posix() in seen_paths:
        raise ValueError(f'cache {name} contains duplicate file paths')
      seen_paths.add(relative.as_posix())
      _nonnegative_int(
        item['size_bytes'], context=f'cache.{name}.files[{index}].size_bytes')
      _hex(item['sha256'], 64, context=f'cache.{name}.files[{index}].sha256')
    if canonical_sha256(files) != _hex(
        snapshot['files_manifest_sha256'], 64,
        context=f'cache.{name}.files_manifest_sha256'):
      raise ValueError(f'cache {name} file manifest hash mismatch')
  body = {key: value for key, value in cache.items()
          if key != 'identity_sha256'}
  if canonical_sha256(body) != _hex(
      cache['identity_sha256'], 64, context='cache.identity_sha256'):
    raise ValueError('cache identity self-hash mismatch')
  return dict(cache)


def _job_without_commitment(job: Mapping[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in job.items()
          if key != 'job_spec_sha256'}


def _plan_without_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in plan.items()
          if key not in {'plan_id', 'plan_sha256'}}


def compile_plan(
    protocol_path: Path,
    *,
    source_root: Path,
    checkpoint_root: Path,
    artifact_root: Path,
    harness_repo_root: Path,
    cache_root: Path,
    source_identity: Mapping[str, Any] | None = None,
    checkpoint_identity: Mapping[str, Mapping[str, Any]] | None = None,
    harness_identity: Mapping[str, Any] | None = None,
    cache_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  """Compile the two-arm, six-job feasibility matrix without model execution."""
  protocol_path = protocol_path.expanduser().resolve()
  protocol = load_protocol(protocol_path)
  source_root = source_root.expanduser().resolve()
  checkpoint_root = checkpoint_root.expanduser().resolve()
  cache_root = cache_root.expanduser().resolve()
  artifact_root = artifact_root.expanduser().resolve(strict=False)
  if not artifact_root.is_absolute() or artifact_root == Path(artifact_root.anchor):
    raise ValueError('artifact_root must be an absolute non-root directory')
  if source_identity is None:
    source_identity = clean_git_identity(
      source_root,
      expected_revision=protocol['source']['revision'],
      expected_remote=protocol['source']['repository'])
  if checkpoint_identity is None:
    checkpoint_identity = checkpoint_identities(checkpoint_root, protocol)
  if harness_identity is None:
    harness_identity = clean_git_identity(harness_repo_root)
  if cache_identity is None:
    cache_identity = cached_model_identities(cache_root, protocol)
  _exact_keys(source_identity, {'path', 'revision', 'clean', 'origin'},
              context='source_identity')
  if source_identity['revision'] != OFFICIAL_SOURCE_REVISION \
      or source_identity['clean'] is not True:
    raise ValueError('source identity is not the clean audited checkout')
  _exact_keys(harness_identity, {'path', 'revision', 'clean', 'origin'},
              context='harness_identity')
  _hex(harness_identity['revision'], 40, context='harness revision')
  if harness_identity['clean'] is not True:
    raise ValueError('harness repository must be clean')
  validated_cache = validate_cache_identity(cache_identity, protocol=protocol)
  if Path(validated_cache['root']).expanduser().resolve() != cache_root:
    raise ValueError('cache identity root differs from requested cache root')

  checkpoint_copy: dict[str, dict[str, Any]] = {}
  for arm_name in ('marginal', 'tensor_train_rank4'):
    identity = _exact_keys(checkpoint_identity[arm_name], {
      'path', 'sha256', 'size_bytes', 'huggingface_repository',
      'huggingface_revision', 'relative_path',
    }, context=f'checkpoint_identity.{arm_name}')
    expected = protocol['checkpoints']['arms'][arm_name]
    if _hex(identity['sha256'], 64, context=f'{arm_name}.sha256') \
        != expected['sha256']:
      raise ValueError(f'{arm_name} checkpoint identity differs from protocol')
    if identity['huggingface_repository'] != protocol['checkpoints']['repository'] \
        or identity['huggingface_revision'] \
        != protocol['checkpoints']['revision'] \
        or identity['relative_path'] != expected['relative_path']:
      raise ValueError(f'{arm_name} checkpoint origin differs from protocol')
    _positive_int(identity['size_bytes'], context=f'{arm_name}.size_bytes')
    checkpoint_copy[arm_name] = dict(identity)

  jobs: dict[str, dict[str, Any]] = {}
  generation = protocol['generation']
  for arm_name in ('marginal', 'tensor_train_rank4'):
    for nfe_steps in generation['nfe_steps']:
      tokens_per_step = generation['sequence_length'] // nfe_steps
      job_id = (
        f'owt--{arm_name}--nfe{nfe_steps:03d}--'
        f's{generation["generation_seed"]:06d}')
      if JOB_ID.fullmatch(job_id) is None:
        raise AssertionError(f'internal invalid job ID: {job_id}')
      artifact_dir = _safe_absolute_child(
        artifact_root / 'runs' / job_id,
        artifact_root,
        context=f'{job_id}.artifact_dir')
      job = {
        'schema_version': JOB_SCHEMA_VERSION,
        'artifact': 'tensor_train_owt_feasibility_job',
        'job_id': job_id,
        'protocol_id': PROTOCOL_ID,
        'arm': arm_name,
        'checkpoint': checkpoint_copy[arm_name],
        'decomposition': dict(protocol['checkpoints']['arms'][arm_name]),
        'model_inputs': {
          name: dict(specification)
          for name, specification in protocol['model_inputs'].items()
        },
        'cache': validated_cache,
        'generation': {
          'dataset_label': generation['dataset_label'],
          'num_samples': generation['num_samples'],
          'sequence_length': generation['sequence_length'],
          'nfe_steps': nfe_steps,
          'tokens_per_step': tokens_per_step,
          'ordering': generation['ordering'],
          'temperature': generation['temperature'],
          'sampling': generation['sampling'],
          'batch_size': generation['batch_size'],
          'generation_seed': generation['generation_seed'],
          'deterministic_algorithms': generation['deterministic_algorithms'],
          'require_exclusive_gpu': generation['require_exclusive_gpu'],
          'schedule_policy': generation['schedule_policy'],
        },
        'evaluator': dict(protocol['evaluator']),
        'runtime': dict(protocol['runtime']),
        'source': dict(source_identity),
        'harness_repository': dict(harness_identity),
        'artifact_dir': str(artifact_dir),
      }
      job['job_spec_sha256'] = canonical_sha256(job)
      jobs[job_id] = job

  plan = {
    'schema_version': PLAN_SCHEMA_VERSION,
    'artifact': 'tensor_train_owt_feasibility_plan',
    'protocol_id': PROTOCOL_ID,
    'protocol_path': str(protocol_path),
    'protocol_sha256': sha256_file(protocol_path),
    'scientific_scope': protocol['scientific_scope'],
    'source': dict(source_identity),
    'harness_repository': dict(harness_identity),
    'checkpoint_revision': protocol['checkpoints']['revision'],
    'checkpoints': checkpoint_copy,
    'model_inputs': {
      name: dict(specification)
      for name, specification in protocol['model_inputs'].items()
    },
    'cache': validated_cache,
    'artifact_root': str(artifact_root),
    'job_ids': list(jobs),
    'job_spec_sha256': {
      job_id: job['job_spec_sha256'] for job_id, job in jobs.items()
    },
    'num_jobs': len(jobs),
  }
  plan['plan_id'] = canonical_sha256(plan)
  plan['plan_sha256'] = canonical_sha256(_plan_without_identity(plan))
  return plan, jobs


def _atomic_write_exclusive(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
  descriptor = os.open(path, flags, 0o644)
  try:
    with os.fdopen(descriptor, 'w') as handle:
      handle.write(content)
      handle.flush()
      os.fsync(handle.fileno())
  except BaseException:
    try:
      path.unlink()
    except FileNotFoundError:
      pass
    raise


def write_plan(
    plan: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> Path:
  output_dir = output_dir.expanduser().resolve(strict=False)
  if output_dir.exists():
    raise FileExistsError(
      f'refusing to overwrite existing plan directory {output_dir}')
  output_dir.mkdir(parents=True, exist_ok=False)
  try:
    for job_id, job in jobs.items():
      _atomic_write_exclusive(
        output_dir / f'{job_id}.json',
        json.dumps(job, indent=2, sort_keys=True) + '\n')
    _atomic_write_exclusive(
      output_dir / 'compiled-plan.json',
      json.dumps(plan, indent=2, sort_keys=True) + '\n')
  except BaseException:
    # Preserve a failed compilation directory as evidence. A retry must use a
    # fresh directory; callers never see a silently replaced commitment.
    raise
  return output_dir / 'compiled-plan.json'


def load_compiled_plan(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  path = path.expanduser().resolve()
  plan = _exact_keys(_read_json(path), {
    'schema_version', 'artifact', 'protocol_id', 'protocol_path',
    'protocol_sha256', 'scientific_scope', 'source', 'harness_repository',
    'checkpoint_revision', 'checkpoints', 'model_inputs', 'cache',
    'artifact_root', 'job_ids', 'job_spec_sha256', 'num_jobs', 'plan_id',
    'plan_sha256',
  }, context='compiled plan')
  if plan['schema_version'] != PLAN_SCHEMA_VERSION \
      or plan['artifact'] != 'tensor_train_owt_feasibility_plan' \
      or plan['protocol_id'] != PROTOCOL_ID:
    raise ValueError('unsupported compiled plan identity')
  _hex(plan['protocol_sha256'], 64, context='plan.protocol_sha256')
  _hex(plan['plan_id'], 64, context='plan.plan_id')
  _hex(plan['plan_sha256'], 64, context='plan.plan_sha256')
  body = _plan_without_identity(plan)
  if canonical_sha256(body) != plan['plan_sha256']:
    raise ValueError('compiled plan self-hash mismatch')
  # plan_id was computed before either identity field was attached.
  if canonical_sha256(body) != plan['plan_id']:
    raise ValueError('compiled plan ID mismatch')
  if not isinstance(plan['job_ids'], list) or len(plan['job_ids']) != 6 \
      or len(set(plan['job_ids'])) != 6 or plan['num_jobs'] != 6:
    raise ValueError('compiled plan must contain exactly six unique jobs')
  if set(plan['job_spec_sha256']) != set(plan['job_ids']):
    raise ValueError('compiled plan job commitments are incomplete')
  protocol_path = Path(plan['protocol_path']).expanduser().resolve()
  if sha256_file(protocol_path) != plan['protocol_sha256']:
    raise ValueError('frozen protocol bytes differ from the compiled plan')
  protocol = load_protocol(protocol_path)
  if plan['scientific_scope'] != protocol['scientific_scope'] \
      or plan['checkpoint_revision'] \
      != protocol['checkpoints']['revision']:
    raise ValueError('compiled plan differs from its frozen protocol')
  if plan['source'].get('revision') != protocol['source']['revision'] \
      or plan['source'].get('clean') is not True:
    raise ValueError('compiled plan source is not the frozen clean revision')
  if set(plan['checkpoints']) != {'marginal', 'tensor_train_rank4'}:
    raise ValueError('compiled plan checkpoint identities are incomplete')
  if plan['model_inputs'] != protocol['model_inputs']:
    raise ValueError('compiled plan model inputs differ from its protocol')
  validate_cache_identity(plan['cache'], protocol=protocol)
  artifact_root = Path(plan['artifact_root']).expanduser().resolve()
  if not artifact_root.is_absolute() \
      or artifact_root == Path(artifact_root.anchor):
    raise ValueError('compiled plan artifact root is unsafe')
  plan_dir = path.parent
  jobs = {}
  for job_id in plan['job_ids']:
    if not isinstance(job_id, str) or JOB_ID.fullmatch(job_id) is None:
      raise ValueError(f'invalid compiled job ID: {job_id!r}')
    job = _exact_keys(_read_json(plan_dir / f'{job_id}.json'), {
      'schema_version', 'artifact', 'job_id', 'protocol_id', 'arm',
      'checkpoint', 'decomposition', 'model_inputs', 'cache', 'generation',
      'evaluator', 'runtime', 'source', 'harness_repository', 'artifact_dir',
      'job_spec_sha256',
    }, context=f'{job_id} spec')
    if job.get('job_id') != job_id:
      raise ValueError(f'{job_id} spec identity mismatch')
    expected = _hex(
      plan['job_spec_sha256'][job_id], 64,
      context=f'{job_id} commitment')
    if job.get('job_spec_sha256') != expected \
        or canonical_sha256(_job_without_commitment(job)) != expected:
      raise ValueError(f'{job_id} job-spec commitment mismatch')
    if job['schema_version'] != JOB_SCHEMA_VERSION \
        or job['artifact'] != 'tensor_train_owt_feasibility_job' \
        or job['protocol_id'] != PROTOCOL_ID:
      raise ValueError(f'{job_id} has an unsupported job identity')
    match = JOB_ID.fullmatch(job_id)
    assert match is not None
    arm_name = match.group(1)
    nfe_steps = int(match.group(2))
    if job['arm'] != arm_name \
        or job['checkpoint'] != plan['checkpoints'][arm_name] \
        or job['model_inputs'] != plan['model_inputs'] \
        or job['cache'] != plan['cache'] \
        or job['source'] != plan['source'] \
        or job['harness_repository'] != plan['harness_repository']:
      raise ValueError(f'{job_id} input identity differs from its plan')
    if job['decomposition'] != protocol['checkpoints']['arms'][arm_name]:
      raise ValueError(f'{job_id} decomposition differs from its protocol')
    expected_generation = {
      'dataset_label': 'openwebtext',
      'num_samples': 256,
      'sequence_length': 1024,
      'nfe_steps': nfe_steps,
      'tokens_per_step': 1024 // nfe_steps,
      'ordering': 'random',
      'temperature': 1.0,
      'sampling': 'top-k',
      'batch_size': 1,
      'generation_seed': 260703,
      'deterministic_algorithms': True,
      'require_exclusive_gpu': True,
      'schedule_policy': SCHEDULE_POLICY,
    }
    if job['generation'] != expected_generation \
        or job['evaluator'] != protocol['evaluator'] \
        or job['runtime'] != protocol['runtime']:
      raise ValueError(f'{job_id} execution policy differs from its protocol')
    expected_artifact_dir = _safe_absolute_child(
      artifact_root / 'runs' / job_id,
      artifact_root,
      context=f'{job_id}.artifact_dir')
    if Path(job['artifact_dir']).expanduser().resolve() \
        != expected_artifact_dir:
      raise ValueError(f'{job_id} artifact directory differs from its plan')
    jobs[job_id] = dict(job)
  return dict(plan), jobs


def validate_resolved_official_config(
    value: object,
    *,
    job: Mapping[str, Any],
) -> dict[str, Any]:
  """Replay every execution-relevant Hydra value from the saved YAML."""
  if not isinstance(value, Mapping):
    raise TypeError('resolved official config must be an object')
  config = value
  required_top_level = {
    'name', 'model', 'training', 'generation', 'type', 'rank_arch', 'algo',
    'dataset', 'is_di4c',
  }
  missing_top_level = sorted(required_top_level - set(config))
  if missing_top_level:
    raise ValueError(
      f'resolved official config is missing fields: {missing_top_level}')
  if config['name'] != 'mdlm' or config['rank_arch'] != '2-layer' \
      or config['is_di4c'] is not False:
    raise ValueError('resolved official model identity mismatch')
  model_type = _exact_keys(
    config['type'], {'model', 'mdlm'}, context='resolved config.type')
  if dict(model_type) != {'model': 'mdlm', 'mdlm': 'owt'}:
    raise ValueError('resolved official type differs from MDLM OWT')
  algo = _exact_keys(config['algo'], {
    'decomp', 'cp', 'tt', 'output_dtype', 'causal_attention',
  }, context='resolved config.algo')
  decomposition = job['decomposition']
  if algo['decomp'] != decomposition['decomposition'] \
      or algo['output_dtype'] != 'float32' \
      or algo['causal_attention'] is not False:
    raise ValueError('resolved official decomposition identity mismatch')
  cp = _exact_keys(algo['cp'], {'rank', 'rank_weights'},
                   context='resolved config.algo.cp')
  tt = _exact_keys(algo['tt'], {'rank', 'marginal_head'},
                   context='resolved config.algo.tt')
  expected_cp = {
    'rank': decomposition['rank'] if job['arm'] == 'marginal' else 1,
    'rank_weights': decomposition['rank_weights'],
  }
  expected_tt = {
    'rank': decomposition['rank'] if job['arm'] != 'marginal' else 4,
    'marginal_head': decomposition['marginal_head'],
  }
  if dict(cp) != expected_cp or dict(tt) != expected_tt:
    raise ValueError('resolved official rank configuration mismatch')
  generation = _exact_keys(config['generation'], {
    'length', 'ckpt_path', 'total_samples', 'eval_model_name', 'temperature',
    'batch_size', 'sampling', 'k', 'gamma', 'ordering',
  }, context='resolved config.generation')
  expected_generation = {
    'length': job['generation']['sequence_length'],
    'ckpt_path': job['checkpoint']['path'],
    'total_samples': job['generation']['num_samples'],
    'eval_model_name': 'gpt2-large',
    'temperature': job['generation']['temperature'],
    'batch_size': job['generation']['batch_size'],
    'sampling': job['generation']['sampling'],
    'k': job['generation']['tokens_per_step'],
    'gamma': 0.1,
    'ordering': job['generation']['ordering'],
  }
  if dict(generation) != expected_generation:
    raise ValueError('resolved official generation config differs from job')
  training = config['training']
  if not isinstance(training, Mapping) \
      or _finite_float(
        training.get('weight_noise_coeff'),
        context='resolved training.weight_noise_coeff') != 0.01:
    raise ValueError('resolved official initialization policy mismatch')
  return dict(config)


def validate_model_load_identity(
    value: object,
    *,
    job: Mapping[str, Any],
) -> dict[str, Any]:
  identity = _exact_keys(value, {
    'class', 'backbone_class', 'tokenizer_class', 'mask_token_id',
    'tokenizer_vocab_size', 'ema_used', 'checkpoint_step',
    'checkpoint_payload_keys', 'checkpoint_state_keys',
    'checkpoint_config_sha256', 'checkpoint_config_identity', 'missing_keys',
    'unexpected_keys', 'parameter_dtypes', 'decomposition', 'backbone_input',
    'tokenizer_input', 'cache_identity_sha256',
  }, context='model_load')
  for field in ('class', 'backbone_class', 'tokenizer_class'):
    _nonempty_string(identity[field], context=f'model_load.{field}')
  if identity['mask_token_id'] != 50257 \
      or identity['tokenizer_vocab_size'] != 50257 \
      or identity['ema_used'] is not False:
    raise ValueError('loaded tokenizer/model boundary identity mismatch')
  if identity['checkpoint_step'] != EXPECTED_CHECKPOINT_STEPS[job['arm']]:
    raise ValueError('loaded checkpoint step differs from audited release')
  if identity['checkpoint_payload_keys'] \
      != ['config', 'model', 'optimizer', 'scheduler', 'step']:
    raise ValueError('loaded checkpoint payload keys differ from release')
  expected_state_keys = list(EXPECTED_CHECKPOINT_STATE_KEYS[job['arm']])
  if identity['checkpoint_state_keys'] != expected_state_keys:
    raise ValueError('loaded learned state keys differ from audited release')
  if _hex(
      identity['checkpoint_config_sha256'], 64,
      context='model_load.checkpoint_config_sha256') \
      != EXPECTED_CHECKPOINT_CONFIG_SHA256[job['arm']]:
    raise ValueError('checkpoint embedded config hash differs from release')
  config_identity = _exact_keys(identity['checkpoint_config_identity'], {
    'name', 'rank_arch', 'type_model', 'type_mdlm', 'decomposition', 'rank',
    'rank_weights', 'marginal_head', 'output_dtype', 'causal_attention',
  }, context='model_load.checkpoint_config_identity')
  expected_decomposition = job['decomposition']
  expected_config_identity = {
    'name': 'mdlm',
    'rank_arch': '2-layer',
    'type_model': 'mdlm',
    'type_mdlm': 'owt',
    'decomposition': expected_decomposition['decomposition'],
    'rank': expected_decomposition['rank'],
    'rank_weights': expected_decomposition['rank_weights'],
    'marginal_head': expected_decomposition['marginal_head'],
    'output_dtype': 'float32',
    'causal_attention': False,
  }
  if dict(config_identity) != expected_config_identity:
    raise ValueError('checkpoint embedded config differs from audited arm')
  missing_keys = identity['missing_keys']
  if not isinstance(missing_keys, list) \
      or any(not isinstance(item, str) or not item for item in missing_keys):
    raise ValueError('model_load.missing_keys must be a string list')
  if set(expected_state_keys) & set(missing_keys):
    raise ValueError('a released learned state key was not loaded')
  if identity['unexpected_keys'] != []:
    raise ValueError('released checkpoint has unexpected model keys')
  if identity['parameter_dtypes'] != ['torch.float32']:
    raise ValueError('loaded model parameter dtype differs from float32')
  if identity['decomposition'] != job['decomposition'] \
      or identity['backbone_input'] != job['model_inputs']['backbone'] \
      or identity['tokenizer_input'] != job['model_inputs']['tokenizer'] \
      or identity['cache_identity_sha256'] \
      != job['cache']['identity_sha256']:
    raise ValueError('model-load input identity differs from compiled job')
  return dict(identity)


def validate_evaluator_runtime_identity(
    value: object,
    *,
    evaluator: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
  identity = _exact_keys(value, {
    'schema_version', 'model_name_or_path', 'model_revision', 'model_class',
    'model_config_class', 'tokenizer_name_or_path', 'tokenizer_revision',
    'tokenizer_class', 'tokenizer_vocab_size', 'tokenizer_bos_token_id',
    'tokenizer_eos_token_id', 'tokenizer_pad_token_id',
    'tokenizer_padding_side', 'tokenizer_truncation_side',
    'tokenization_policy', 'sequence_policy', 'add_special_tokens',
    'batch_size', 'max_length', 'requested_dtype', 'parameter_dtypes',
    'precision_policy', 'device', 'python', 'torch', 'cuda_runtime',
    'transformers', 'tokenizers',
  }, context='reference-LM runtime identity')
  expected = {
    'schema_version': 1,
    'model_name_or_path': evaluator['model_name_or_path'],
    'model_revision': evaluator['revision'],
    'tokenizer_name_or_path': evaluator['model_name_or_path'],
    'tokenizer_revision': evaluator['revision'],
    'tokenizer_padding_side': 'right',
    'tokenizer_truncation_side': 'right',
    'tokenization_policy': REFERENCE_LM_TOKENIZATION_POLICY,
    'sequence_policy': evaluator['sequence_policy'],
    'add_special_tokens': True,
    'batch_size': evaluator['batch_size'],
    'max_length': evaluator['max_length'],
    'requested_dtype': evaluator['dtype'],
    'parameter_dtypes': ['torch.float32'],
    'precision_policy': REFERENCE_LM_PRECISION_POLICY,
    'device': 'cuda',
    'transformers': runtime['critical_packages']['transformers'],
    'tokenizers': runtime['critical_packages']['tokenizers'],
  }
  for field, expected_value in expected.items():
    if identity[field] != expected_value:
      raise ValueError(
        f'reference-LM runtime {field} differs from compiled job')
  for field in (
      'model_class', 'model_config_class', 'tokenizer_class', 'python', 'torch',
      'cuda_runtime'):
    _nonempty_string(identity[field], context=f'reference runtime.{field}')
  if not identity['python'].startswith(runtime['python_major_minor'] + '.') \
      or identity['torch'].split('+', 1)[0] \
      != runtime['critical_packages']['torch']:
    raise ValueError('reference-LM Python/Torch runtime differs from job')
  if _positive_int(
      identity['tokenizer_vocab_size'], context='reference tokenizer vocab size') \
      != 50257:
    raise ValueError('reference tokenizer vocabulary differs from GPT-2')
  for field in (
      'tokenizer_bos_token_id', 'tokenizer_eos_token_id',
      'tokenizer_pad_token_id'):
    value = identity[field]
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
      raise ValueError(f'reference runtime {field} must be an integer or null')
  if any(identity[field] != 50256 for field in (
      'tokenizer_bos_token_id', 'tokenizer_eos_token_id',
      'tokenizer_pad_token_id')):
    raise ValueError('reference tokenizer special-token identity differs')
  return dict(identity)


def _token_entropy(tokens: Iterable[Iterable[int]]) -> float:
  counts = Counter(token for sequence in tokens for token in sequence)
  total = sum(counts.values())
  if total <= 0:
    raise ValueError('cannot compute entropy for empty token rows')
  return math.fsum(
    -(count / total) * math.log(count / total)
    for count in counts.values())


def validate_position_schedules(
    value: object,
    *,
    job: Mapping[str, Any],
) -> list[dict[str, Any]]:
  artifact = _exact_keys(value, {
    'schema_version', 'artifact', 'job_id', 'arm', 'nfe_steps',
    'generation_seed', 'schedule_policy', 'num_samples', 'sequence_length',
    'records',
  }, context='position schedule artifact')
  generation = job['generation']
  if artifact['schema_version'] != RUN_SCHEMA_VERSION \
      or artifact['artifact'] != 'tensor_train_owt_position_schedules' \
      or artifact['job_id'] != job['job_id'] \
      or artifact['arm'] != job['arm'] \
      or artifact['nfe_steps'] != generation['nfe_steps'] \
      or artifact['generation_seed'] != generation['generation_seed'] \
      or artifact['schedule_policy'] != generation['schedule_policy'] \
      or artifact['num_samples'] != generation['num_samples'] \
      or artifact['sequence_length'] != generation['sequence_length']:
    raise ValueError('position schedule artifact identity mismatch')
  records = artifact['records']
  if not isinstance(records, list) or len(records) != generation['num_samples']:
    raise ValueError('position schedule record count mismatch')
  validated = []
  expected_positions = list(range(generation['sequence_length']))
  for index, record in enumerate(records):
    row = _exact_keys(record, {
      'sample_id', 'chunks', 'position_schedule_sha256',
    }, context=f'position schedule[{index}]')
    if row['sample_id'] != index:
      raise ValueError('position schedule sample ID differs from row order')
    chunks = row['chunks']
    if not isinstance(chunks, list) \
        or len(chunks) != generation['nfe_steps']:
      raise ValueError(f'position schedule[{index}] chunk count mismatch')
    flattened = []
    for step, chunk in enumerate(chunks):
      if not isinstance(chunk, list) \
          or len(chunk) != generation['tokens_per_step'] \
          or any(not isinstance(position, int) or isinstance(position, bool)
                 for position in chunk) \
          or chunk != sorted(chunk) \
          or len(set(chunk)) != len(chunk):
        raise ValueError(
          f'position schedule[{index}] has invalid chunk {step}')
      flattened.extend(chunk)
    if sorted(flattened) != expected_positions:
      raise ValueError(f'position schedule[{index}] is not a full permutation')
    if canonical_sha256(chunks) != _hex(
        row['position_schedule_sha256'], 64,
        context=f'position schedule[{index}].sha256'):
      raise ValueError(f'position schedule[{index}] hash mismatch')
    validated.append(dict(row))
  return validated


def _validate_sample_record(
    record: object,
    *,
    job: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
  row = _exact_keys(record, {
    'schema_version', 'sample_id', 'job_id', 'arm', 'nfe_steps',
    'generation_seed', 'token_ids', 'token_ids_sha256', 'text',
    'text_sha256', 'position_schedule_sha256', 'reference_lm',
  }, context=f'sample[{index}]')
  if row['schema_version'] != RUN_SCHEMA_VERSION:
    raise ValueError(f'sample[{index}] schema_version must equal 1')
  if row['sample_id'] != index:
    raise ValueError(f'sample[{index}] sample_id differs from line order')
  if row['job_id'] != job['job_id'] or row['arm'] != job['arm'] \
      or row['nfe_steps'] != job['generation']['nfe_steps'] \
      or row['generation_seed'] != job['generation']['generation_seed']:
    raise ValueError(f'sample[{index}] job identity mismatch')
  tokens = row['token_ids']
  if not isinstance(tokens, list) \
      or len(tokens) != job['generation']['sequence_length']:
    raise ValueError(f'sample[{index}] has invalid token length')
  if any(not isinstance(token, int) or isinstance(token, bool)
         or not 0 <= token < 50257 for token in tokens):
    raise ValueError(f'sample[{index}] contains an invalid or masked token')
  if canonical_sha256(tokens) != row['token_ids_sha256']:
    raise ValueError(f'sample[{index}] token hash mismatch')
  if not isinstance(row['text'], str):
    raise TypeError(f'sample[{index}].text must be a string')
  if hashlib.sha256(row['text'].encode('utf-8')).hexdigest() \
      != row['text_sha256']:
    raise ValueError(f'sample[{index}] text hash mismatch')
  _hex(
    row['position_schedule_sha256'], 64,
    context=f'sample[{index}].position_schedule_sha256')
  score = _exact_keys(row['reference_lm'], {
    'model_name_or_path', 'revision', 'sequence_policy', 'token_count',
    'mean_nll_nats', 'perplexity',
  }, context=f'sample[{index}].reference_lm')
  evaluator = job['evaluator']
  if score['model_name_or_path'] != evaluator['model_name_or_path'] \
      or score['revision'] != evaluator['revision'] \
      or score['sequence_policy'] != evaluator['sequence_policy']:
    raise ValueError(f'sample[{index}] evaluator identity mismatch')
  if _positive_int(score['token_count'], context='reference token_count') > \
      evaluator['max_length'] - 1:
    raise ValueError(f'sample[{index}] reference token count is too large')
  mean_nll = _finite_float(score['mean_nll_nats'], context='mean_nll_nats')
  perplexity = _finite_float(score['perplexity'], context='perplexity')
  if mean_nll < 0.0 or perplexity <= 0.0:
    raise ValueError(f'sample[{index}] has invalid reference-LM score')
  expected_perplexity = math.exp(min(mean_nll, 80.0))
  if not math.isclose(perplexity, expected_perplexity, rel_tol=1e-6):
    raise ValueError(f'sample[{index}] perplexity is inconsistent with NLL')
  return dict(row)


def validate_completed_run(
    run_dir: Path,
    *,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    expected_plan_file_sha256: str | None = None,
) -> dict[str, Any]:
  """Replay hashes and schemas for one atomic run directory."""
  run_dir = run_dir.expanduser().resolve()
  if run_dir != Path(job['artifact_dir']).expanduser().resolve():
    raise ValueError('run directory differs from the compiled job')
  success = _exact_keys(_read_json(run_dir / '_SUCCESS.json'), {
    'schema_version', 'artifact', 'job_id', 'job_spec_sha256',
    'manifest_sha256',
  }, context='success marker')
  if success['schema_version'] != RUN_SCHEMA_VERSION \
      or success['artifact'] != 'tensor_train_owt_feasibility_success' \
      or success['job_id'] != job['job_id'] \
      or success['job_spec_sha256'] != job['job_spec_sha256']:
    raise ValueError('success marker identity mismatch')
  manifest_path = run_dir / 'manifest.json'
  if sha256_file(manifest_path) != success['manifest_sha256']:
    raise ValueError('manifest hash differs from success marker')
  manifest = _exact_keys(_read_json(manifest_path), {
    'schema_version', 'artifact', 'scientific_scope', 'job_id', 'arm',
    'nfe_steps', 'job_spec_sha256', 'plan_id', 'plan_file_sha256',
    'start_time_utc', 'end_time_utc', 'runtime', 'source', 'checkpoint',
    'harness_repository', 'model_inputs', 'cache', 'model_load', 'evaluator',
    'outputs', 'interruption_policy',
  }, context='run manifest')
  if manifest['schema_version'] != RUN_SCHEMA_VERSION \
      or manifest['artifact'] != 'tensor_train_owt_feasibility_run' \
      or manifest['job_id'] != job['job_id'] \
      or manifest['arm'] != job['arm'] \
      or manifest['nfe_steps'] != job['generation']['nfe_steps'] \
      or manifest['job_spec_sha256'] != job['job_spec_sha256'] \
      or manifest['plan_id'] != plan['plan_id'] \
      or manifest['scientific_scope'] != plan['scientific_scope']:
    raise ValueError('run manifest identity mismatch')
  try:
    start_time = dt.datetime.fromisoformat(manifest['start_time_utc'])
    end_time = dt.datetime.fromisoformat(manifest['end_time_utc'])
  except (TypeError, ValueError) as error:
    raise ValueError('run manifest timestamps must be ISO-8601') from error
  if start_time.tzinfo is None or end_time.tzinfo is None \
      or end_time < start_time:
    raise ValueError('run manifest timestamps are unordered or timezone-naive')
  plan_file_sha256 = _hex(
    manifest['plan_file_sha256'], 64, context='plan_file_sha256')
  if expected_plan_file_sha256 is not None \
      and plan_file_sha256 != _hex(
        expected_plan_file_sha256, 64,
        context='expected_plan_file_sha256'):
    raise ValueError('run manifest compiled-plan file hash mismatch')
  if manifest['source'] != job['source'] \
      or manifest['checkpoint'] != job['checkpoint'] \
      or manifest['harness_repository'] != job['harness_repository'] \
      or manifest['model_inputs'] != job['model_inputs'] \
      or manifest['cache'] != job['cache']:
    raise ValueError('run manifest input identity mismatch')
  validate_cache_identity(manifest['cache'], protocol=load_protocol(
    Path(plan['protocol_path']).expanduser().resolve()))
  validate_model_load_identity(manifest['model_load'], job=job)
  if manifest['interruption_policy'] != job['runtime']['interruption_policy']:
    raise ValueError('run manifest interruption policy mismatch')
  outputs = _exact_keys(manifest['outputs'], {
    'samples_jsonl', 'metrics_json', 'resource_metrics_json',
    'resolved_config_yaml', 'position_schedules_json',
  }, context='manifest.outputs')
  expected_output_names = {
    'samples_jsonl': 'samples.jsonl',
    'metrics_json': 'metrics.json',
    'resource_metrics_json': 'resource-metrics.json',
    'resolved_config_yaml': 'resolved_config.yaml',
    'position_schedules_json': 'position-schedules.json',
  }
  resolved_outputs = {}
  for name, descriptor in outputs.items():
    item = _exact_keys(
      descriptor, {'path', 'sha256', 'size_bytes'},
      context=f'manifest.outputs.{name}')
    relative = Path(_nonempty_string(item['path'], context=f'{name}.path'))
    if relative.as_posix() != expected_output_names[name]:
      raise ValueError(f'{name} output filename differs from the protocol')
    if relative.is_absolute() or '..' in relative.parts:
      raise ValueError(f'{name} output path must be a safe relative path')
    output_path = (run_dir / relative).resolve()
    try:
      output_path.relative_to(run_dir)
    except ValueError as error:
      raise ValueError(f'{name} output escapes run directory') from error
    if not output_path.is_file():
      raise FileNotFoundError(output_path)
    if sha256_file(output_path) != _hex(
        item['sha256'], 64, context=f'{name}.sha256'):
      raise ValueError(f'{name} output hash mismatch')
    if output_path.stat().st_size != _positive_int(
        item['size_bytes'], context=f'{name}.size_bytes'):
      raise ValueError(f'{name} output size mismatch')
    resolved_outputs[name] = output_path
  if len(set(resolved_outputs.values())) != len(resolved_outputs):
    raise ValueError('manifest output paths must be unique')
  expected_run_entries = set(expected_output_names.values()) | {
    'manifest.json', '_SUCCESS.json',
  }
  actual_run_entries = {path.name for path in run_dir.iterdir()}
  if actual_run_entries != expected_run_entries:
    raise ValueError(
      'run directory output set mismatch: '
      f'missing={sorted(expected_run_entries - actual_run_entries)}, '
      f'unknown={sorted(actual_run_entries - expected_run_entries)}')
  validate_resolved_official_config(
    _read_yaml_or_json(resolved_outputs['resolved_config_yaml']), job=job)

  metrics = _exact_keys(_read_json(resolved_outputs['metrics_json']), {
    'schema_version', 'artifact', 'job_id', 'arm', 'nfe_steps',
    'num_samples', 'sequence_length', 'generation_seed', 'token_entropy_nats',
    'reference_lm',
  }, context='metrics')
  if metrics['schema_version'] != RUN_SCHEMA_VERSION \
      or metrics['artifact'] != 'tensor_train_owt_feasibility_metrics' \
      or metrics['job_id'] != job['job_id'] \
      or metrics['arm'] != job['arm'] \
      or metrics['nfe_steps'] != job['generation']['nfe_steps'] \
      or metrics['num_samples'] != job['generation']['num_samples'] \
      or metrics['sequence_length'] != job['generation']['sequence_length'] \
      or metrics['generation_seed'] != job['generation']['generation_seed']:
    raise ValueError('metrics identity mismatch')
  entropy = _finite_float(
    metrics['token_entropy_nats'], context='metrics.token_entropy_nats')
  if entropy <= 0.0:
    raise ValueError('token entropy must be positive')
  summary_score = _exact_keys(metrics['reference_lm'], {
    'model_name_or_path', 'revision', 'sequence_policy', 'runtime_identity',
    'num_scored_sequences', 'num_scored_tokens', 'mean_nll_nats',
    'perplexity',
  }, context='metrics.reference_lm')
  if summary_score['num_scored_sequences'] != job['generation']['num_samples']:
    raise ValueError('reference-LM sequence count mismatch')
  evaluator = job['evaluator']
  if summary_score['model_name_or_path'] != evaluator['model_name_or_path'] \
      or summary_score['revision'] != evaluator['revision'] \
      or summary_score['sequence_policy'] != evaluator['sequence_policy']:
    raise ValueError('reference-LM summary identity differs from compiled job')
  runtime_identity = summary_score['runtime_identity']
  validate_evaluator_runtime_identity(
    runtime_identity, evaluator=evaluator, runtime=job['runtime'])
  _positive_int(summary_score['num_scored_tokens'], context='num_scored_tokens')
  summary_nll = _finite_float(
    summary_score['mean_nll_nats'], context='summary mean NLL')
  summary_ppl = _finite_float(
    summary_score['perplexity'], context='summary perplexity')
  if not math.isclose(
      summary_ppl, math.exp(min(summary_nll, 80.0)), rel_tol=1e-6):
    raise ValueError('summary perplexity is inconsistent with NLL')
  if manifest['evaluator'] != dict(summary_score):
    raise ValueError('manifest evaluator differs from metrics evaluator')

  resource_metrics = _exact_keys(
    _read_json(resolved_outputs['resource_metrics_json']), {
      'schema_version', 'artifact', 'job_id', 'measurement_scope', 'host',
      'timing_seconds', 'throughput', 'cuda_memory_bytes', 'process',
      'generation', 'gpu_exclusivity',
    }, context='resource metrics')
  if resource_metrics['schema_version'] != RUN_SCHEMA_VERSION \
      or resource_metrics['artifact'] \
      != 'tensor_train_owt_resource_metrics' \
      or resource_metrics['job_id'] != job['job_id'] \
      or resource_metrics['measurement_scope'] \
      != 'single_job_uncontended_end_to_end_v1':
    raise ValueError('resource metrics identity mismatch')
  host = _exact_keys(resource_metrics['host'], {
    'hostname', 'platform', 'python', 'torch', 'cuda_runtime', 'gpu',
    'critical_packages', 'precision_policy',
  }, context='resource host')
  for name in ('hostname', 'platform', 'python', 'torch', 'cuda_runtime'):
    _nonempty_string(host[name], context=f'resource host.{name}')
  if not host['python'].startswith(job['runtime']['python_major_minor'] + '.'):
    raise ValueError('resource host Python differs from compiled job')
  if host['torch'].split('+', 1)[0] \
      != job['runtime']['critical_packages']['torch'] \
      or host['critical_packages'] != job['runtime']['critical_packages'] \
      or host['precision_policy'] != job['runtime']['precision_policy']:
    raise ValueError('resource host runtime differs from compiled job')
  gpu = _exact_keys(host['gpu'], {
    'index', 'name', 'uuid', 'driver_version', 'memory_total_mib',
  }, context='resource host.gpu')
  _nonnegative_int(gpu['index'], context='gpu.index')
  for name in ('name', 'uuid', 'driver_version'):
    _nonempty_string(gpu[name], context=f'gpu.{name}')
  _positive_int(gpu['memory_total_mib'], context='gpu.memory_total_mib')
  if manifest['runtime'] != host:
    raise ValueError('run manifest and resource host identities differ')
  timing = _exact_keys(resource_metrics['timing_seconds'], {
    'model_load', 'generation', 'evaluator_load_and_scoring', 'total',
  }, context='resource timing')
  for name, value in timing.items():
    if _finite_float(value, context=f'timing.{name}') <= 0.0:
      raise ValueError(f'timing.{name} must be positive')
  if timing['total'] + 1e-6 < sum(
      timing[name]
      for name in ('model_load', 'generation', 'evaluator_load_and_scoring')):
    raise ValueError('total timing is shorter than measured phases')
  throughput = _exact_keys(resource_metrics['throughput'], {
    'generation_samples_per_second', 'generation_tokens_per_second',
    'evaluator_samples_per_second',
  }, context='resource throughput')
  for name, value in throughput.items():
    if _finite_float(value, context=f'throughput.{name}') <= 0.0:
      raise ValueError(f'throughput.{name} must be positive')
  expected_sample_rate = job['generation']['num_samples'] / timing['generation']
  expected_token_rate = (
    job['generation']['num_samples'] * job['generation']['sequence_length']
    / timing['generation'])
  expected_evaluator_rate = (
    job['generation']['num_samples']
    / timing['evaluator_load_and_scoring'])
  for field, expected in (
      ('generation_samples_per_second', expected_sample_rate),
      ('generation_tokens_per_second', expected_token_rate),
      ('evaluator_samples_per_second', expected_evaluator_rate)):
    if not math.isclose(throughput[field], expected, rel_tol=1e-9):
      raise ValueError(f'throughput.{field} is inconsistent with timing')
  memory = _exact_keys(resource_metrics['cuda_memory_bytes'], {
    'model_load_peak_allocated', 'model_load_peak_reserved',
    'generation_peak_allocated', 'generation_peak_reserved',
    'evaluator_peak_allocated', 'evaluator_peak_reserved',
  }, context='resource CUDA memory')
  for name, value in memory.items():
    _nonnegative_int(value, context=f'cuda_memory_bytes.{name}')
  if memory['model_load_peak_reserved'] < memory['model_load_peak_allocated'] \
      or memory['generation_peak_reserved'] \
      < memory['generation_peak_allocated'] \
      or memory['evaluator_peak_reserved'] \
      < memory['evaluator_peak_allocated']:
    raise ValueError('CUDA reserved memory is smaller than allocated memory')
  process = _exact_keys(
    resource_metrics['process'], {'pid', 'max_rss_bytes'},
    context='resource process')
  _positive_int(process['pid'], context='process.pid')
  _positive_int(process['max_rss_bytes'], context='process.max_rss_bytes')
  generation_resource = _exact_keys(resource_metrics['generation'], {
    'requested_nfe_steps', 'observed_mean_steps', 'tokens_per_step',
    'num_samples', 'sequence_length', 'batch_size',
  }, context='resource generation')
  if generation_resource['requested_nfe_steps'] \
      != job['generation']['nfe_steps'] \
      or generation_resource['tokens_per_step'] \
      != job['generation']['tokens_per_step'] \
      or generation_resource['num_samples'] \
      != job['generation']['num_samples'] \
      or generation_resource['sequence_length'] \
      != job['generation']['sequence_length'] \
      or generation_resource['batch_size'] \
      != job['generation']['batch_size']:
    raise ValueError('resource generation identity mismatch')
  if not math.isclose(
      _finite_float(
        generation_resource['observed_mean_steps'],
        context='observed_mean_steps'),
      float(job['generation']['nfe_steps']), abs_tol=1e-6):
    raise ValueError('observed generation steps differ from requested NFE')
  exclusivity = _exact_keys(resource_metrics['gpu_exclusivity'], {
    'required', 'policy', 'lock_path', 'lock_acquired',
    'monitor_interval_seconds', 'monitor_samples',
    'preflight_other_compute_pids', 'postflight_other_compute_pids',
    'foreign_pid_observations', 'monitor_errors',
  }, context='gpu exclusivity')
  if exclusivity['required'] is not True \
      or exclusivity['policy'] != GPU_EXCLUSIVITY_POLICY \
      or exclusivity['lock_acquired'] is not True \
      or _finite_float(
        exclusivity['monitor_interval_seconds'],
        context='gpu monitor interval') != 1.0 \
      or _positive_int(
        exclusivity['monitor_samples'], context='gpu monitor samples') < 2 \
      or exclusivity['preflight_other_compute_pids'] != [] \
      or exclusivity['postflight_other_compute_pids'] != [] \
      or exclusivity['foreign_pid_observations'] != [] \
      or exclusivity['monitor_errors'] != []:
    raise ValueError('resource measurement was not GPU-exclusive')
  lock_path = Path(_nonempty_string(
    exclusivity['lock_path'], context='gpu exclusivity.lock_path'))
  if lock_path.expanduser().resolve() != SUBMISSION_GPU_LOCK:
    raise ValueError('GPU lock path differs from the submission-wide lock')

  schedules = validate_position_schedules(
    _read_json(resolved_outputs['position_schedules_json']), job=job)
  records = []
  with resolved_outputs['samples_jsonl'].open() as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        raise ValueError(f'blank samples line at {line_number}')
      try:
        raw = json.loads(line)
      except json.JSONDecodeError as error:
        raise ValueError(
          f'invalid samples JSON at line {line_number}: {error}') from error
      records.append(_validate_sample_record(
        raw, job=job, index=line_number - 1))
  if len(records) != job['generation']['num_samples']:
    raise ValueError('sample record count differs from the compiled job')
  for index, (record, schedule) in enumerate(zip(records, schedules)):
    if record['position_schedule_sha256'] \
        != schedule['position_schedule_sha256']:
      raise ValueError(f'sample[{index}] position schedule hash mismatch')
  replayed_entropy = _token_entropy(
    record['token_ids'] for record in records)
  if not math.isclose(
      replayed_entropy, entropy, rel_tol=1e-12, abs_tol=1e-12):
    raise ValueError('token entropy does not reproduce from samples')
  total_tokens = sum(
    record['reference_lm']['token_count'] for record in records)
  weighted_nll = math.fsum(
    record['reference_lm']['mean_nll_nats']
    * record['reference_lm']['token_count']
    for record in records) / total_tokens
  if total_tokens != summary_score['num_scored_tokens'] \
      or not math.isclose(
        weighted_nll, summary_score['mean_nll_nats'], rel_tol=1e-10):
    raise ValueError('reference-LM summary does not reproduce from samples')
  return {
    'run_dir': str(run_dir),
    'manifest_sha256': success['manifest_sha256'],
    'manifest': dict(manifest),
    'metrics': dict(metrics),
    'resource_metrics': dict(resource_metrics),
    'samples': records,
    'position_schedules': schedules,
  }


def verify_complete_matrix(
    plan_path: Path,
) -> dict[str, Any]:
  """Validate all six jobs and require paired evaluator/sample identities."""
  plan, jobs = load_compiled_plan(plan_path)
  validated = {
    job_id: validate_completed_run(
      Path(job['artifact_dir']),
      plan=plan,
      job=job,
      expected_plan_file_sha256=sha256_file(Path(plan_path).resolve()))
    for job_id, job in jobs.items()
  }
  evaluator_identities = {
    canonical_json(run['metrics']['reference_lm']['runtime_identity'])
    for run in validated.values()
  }
  if len(evaluator_identities) != 1:
    raise ValueError('reference-LM runtime identity differs across jobs')
  resource_host_identities = {
    canonical_json(run['resource_metrics']['host'])
    for run in validated.values()
  }
  if len(resource_host_identities) != 1:
    raise ValueError('resource host identity differs across jobs')
  cells = []
  for job_id in plan['job_ids']:
    run = validated[job_id]
    job = jobs[job_id]
    cells.append({
      'job_id': job_id,
      'arm': job['arm'],
      'nfe_steps': job['generation']['nfe_steps'],
      'manifest_sha256': run['manifest_sha256'],
      'mean_nll_nats': run['metrics']['reference_lm']['mean_nll_nats'],
      'perplexity': run['metrics']['reference_lm']['perplexity'],
      'token_entropy_nats': run['metrics']['token_entropy_nats'],
      'position_schedule_set_sha256': canonical_sha256([
        record['position_schedule_sha256']
        for record in run['samples']]),
      'generation_seconds': (
        run['resource_metrics']['timing_seconds']['generation']),
      'samples_per_second': (
        run['resource_metrics']['throughput']['generation_samples_per_second']),
    })
  for nfe_steps in (8, 16, 32):
    pair = [cell for cell in cells if cell['nfe_steps'] == nfe_steps]
    if {cell['arm'] for cell in pair} \
        != {'marginal', 'tensor_train_rank4'}:
      raise ValueError(f'incomplete arm pair for NFE={nfe_steps}')
    pair_runs = [validated[cell['job_id']] for cell in pair]
    for index in range(256):
      observed = {
        (run['samples'][index]['sample_id'],
         run['samples'][index]['generation_seed'],
         run['samples'][index]['position_schedule_sha256'])
        for run in pair_runs
      }
      if len(observed) != 1:
        raise ValueError(
          f'paired sample/schedule identity mismatch at row {index}')
      observed_identity = next(iter(observed))
      if observed_identity[:2] != (index, 260703):
        raise ValueError(f'paired sample identity mismatch at row {index}')
  return {
    'schema_version': 1,
    'artifact': 'verified_tensor_train_owt_feasibility_matrix',
    'protocol_id': PROTOCOL_ID,
    'plan_id': plan['plan_id'],
    'compiled_plan_sha256': sha256_file(Path(plan_path).resolve()),
    'num_jobs': len(cells),
    'num_samples_per_job': 256,
    'paired_generation_seed': 260703,
    'evaluator_runtime_identity': json.loads(next(iter(evaluator_identities))),
    'resource_host_identity': json.loads(next(iter(resource_host_identities))),
    'cells': cells,
    'claim_scope': (
      'matched feasibility comparison between the two released baseline arms; '
      'this is not yet a matched comparison to Contextual Coupling Forests, '
      'and runtime is valid only for the recorded exclusive host'),
  }
