#!/usr/bin/env python3
"""Run the authenticated schema-v5 seed-201 adapter replay.

This is deliberately separate from ``main.py``.  The legacy generic runtime
is a schema-v4 loader and must continue to reject schema-v5 release artifacts.
Here the released expectations are authenticated first, the actual runtime
backbone is attested, and the schema-v5 head is structurally verified and
strict-loaded before the existing validation sequence sees the model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.export_contextual_forest_adapter import (  # noqa: E402
  AuthenticatedProductionExpectations,
  PRODUCTION_ADAPTER_PARAMETER_COUNT,
  PRODUCTION_ADAPTER_TENSOR_BYTES,
  PRODUCTION_ADAPTER_TENSOR_COUNT,
  attest_runtime_backbone,
  build_production_model,
  load_production_expectations,
)
from scripts.export_structured_adapter import canonical_sha256  # noqa: E402
from scripts.verify_contextual_forest_adapter import (  # noqa: E402
  verify_contextual_forest_adapter,
)


REPLAY_SCHEMA_VERSION = 1
REPLAY_REPORT_FILENAME = 'authenticated-seed-201-replay.json'
MODEL_CONFIG_NAME = 'contextual-forest-small'
MODEL_CONFIG_PATH = REPO_ROOT / 'configs/model/contextual-forest-small.yaml'
MODEL_CONFIG_SHA256 = (
  'd19f76c3305f7005de4f3a626ef918a8f731d72ebc45a164d2f8af08e6043a1e')
DATA_CONFIG_NAME = 'openwebtext-streaming'
DATA_CONFIG_PATH = REPO_ROOT / 'configs/data/openwebtext-streaming.yaml'
DATA_CONFIG_SHA256 = (
  '96ea95014d9cf79768a34e4e5a6ef50e9f88675f9d2634f5fea43c1bd1fdb827')
SEED = 201
EXPECTED_PAIRING_DIGEST_SHA256 = (
  'b1de8e71d5d578422f2fac0962ab175eeffcf0725e3c9b70f7dd3ea9cf6a295c')
EXPECTED_PAIRING_INVENTORY = {
  'epoch': 0,
  'step': 0,
  'sanity_checking': False,
  'world_size': 1,
  'num_batches': 32,
  'num_examples': 128,
  'num_token_slots': 131_072,
  'num_active_tokens': 66_827,
}
HISTORICAL_REFERENCE = {
  'conditional_nll_per_masked_token': 6.033883870619285,
  'candidate_recall': 0.5665225133553803,
  'retained_unary_mass': 0.5701100346586382,
  'active_fraction': 0.5098495483398438,
}
EXPECTED_PYTHON_VERSION = '3.10.12'
EXPECTED_PLATFORM = 'Linux-6.8.0-1066-gcp-x86_64-with-glibc2.35'
EXPECTED_TORCH_VERSION = '2.5.1+cu121'
EXPECTED_TORCH_CUDA_BUILD = '12.1'
EXPECTED_NVIDIA_DRIVER_VERSION = '580.173.02'
EXPECTED_PACKAGE_VERSIONS = {
  'lightning': '2.2.1',
  'torchmetrics': '1.3.2',
  'transformers': '4.38.2',
  'datasets': '2.18.0',
  'hydra-core': '1.3.2',
  'omegaconf': '2.3.0',
  'safetensors': '0.4.2',
  'tokenizers': '0.15.2',
  'fsspec': '2024.2.0',
}

# These are the historical seed-201 overrides that affect evaluation.  Input
# and output paths are assigned directly after Hydra composition so path text
# cannot be interpreted as override syntax.
SEED_201_OVERRIDES = (
  'seed=201',
  'model.structured_decoder.training.use_ema_backbone=false',
  'model.structured_decoder.training.topology_on_validation=false',
  'model.structured_decoder.factor_mode=dynamic',
  'model.structured_decoder.topology_mode=dynamic',
  'model.structured_decoder.independent_mode=false',
  'trainer.accelerator=cuda',
  'trainer.devices=1',
  'trainer.num_nodes=1',
  'trainer.precision=bf16',
  'trainer.limit_val_batches=32',
  'trainer.num_sanity_val_steps=0',
  'loader.global_batch_size=4',
  'loader.eval_global_batch_size=4',
  'loader.batch_size=4',
  'loader.eval_batch_size=4',
  'loader.num_workers=0',
  'training.ema=0',
  'eval.disable_ema=true',
  'eval.generate_samples=false',
  'eval.compute_generative_perplexity=false',
  'eval.gen_ppl_eval_model_name_or_path=gpt2',
  'checkpointing.resume_from_ckpt=false',
  'wandb=null',
)

# A closed projection catches semantic drift after composition.  The four
# adapter fields are intentionally blank: schema-v5 loading happens only after
# actual-backbone attestation in this entry point, never in Diffusion.__init__.
EXPECTED_RUNTIME_CONFIG = {
  'mode': 'ppl_eval',
  'seed': SEED,
  'backbone': 'dit',
  'parameterization': 'subs',
  'T': 0,
  'model.name': MODEL_CONFIG_NAME,
  'model.length': 1024,
  'model.structured_decoder.enabled': True,
  'model.structured_decoder.top_k': 64,
  'model.structured_decoder.rank': 16,
  'model.structured_decoder.topology_mode': 'dynamic',
  'model.structured_decoder.factor_mode': 'dynamic',
  'model.structured_decoder.independent_mode': False,
  'model.structured_decoder.training.backbone_mode': 'frozen',
  'model.structured_decoder.training.backbone_checkpoint': None,
  'model.structured_decoder.training.require_pretrained_backbone': False,
  'model.structured_decoder.training.strict_backbone_checkpoint': True,
  'model.structured_decoder.training.use_ema_backbone': False,
  'model.structured_decoder.training.deterministic_backbone': True,
  'model.structured_decoder.training.topology_on_validation': False,
  'data.train': 'openwebtext',
  'data.valid': 'wikitext103',
  'data.tokenizer_name_or_path': 'gpt2',
  'data.tokenizer_revision': (
    '607a30d783dfa663caf39e06633721c8d4cfcd7e'),
  'data.train_revision': (
    '79d93d786212f7344586290adb811d4ae6a1762c'),
  'data.valid_revision': (
    'b08601e04326c79dfdd32d625aee71d232d685c3'),
  'data.wrap': True,
  'data.streaming': True,
  'trainer.accelerator': 'cuda',
  'trainer.devices': 1,
  'trainer.num_nodes': 1,
  'trainer.precision': 'bf16',
  'trainer.limit_val_batches': 32,
  'trainer.num_sanity_val_steps': 0,
  'trainer.accumulate_grad_batches': 1,
  'loader.global_batch_size': 4,
  'loader.eval_global_batch_size': 4,
  'loader.batch_size': 4,
  'loader.eval_batch_size': 4,
  'loader.num_workers': 0,
  'training.ema': 0.0,
  'eval.disable_ema': True,
  'eval.generate_samples': False,
  'eval.compute_generative_perplexity': False,
  'eval.gen_ppl_eval_model_name_or_path': 'gpt2',
  'eval.structured_mask_rate': None,
  'eval.corruption_seed': None,
  'eval.conditional_records.enabled': False,
  'eval.checkpoint_path': '',
  'eval.adapter_checkpoint': '',
  'eval.adapter_sha256': '',
  'eval.adapter_manifest': '',
  'eval.adapter_manifest_sha256': '',
  'checkpointing.resume_from_ckpt': False,
  'wandb': None,
}
RUNTIME_CONFIG_IDENTITY_SHA256 = (
  '08fde8bd02c4c43a158e92da0b3f3c167698fbd5e9070bd24990999a77a4359a')


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _lower_hex(value: object, length: int, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != length
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(
      f'{context} must be {length} lowercase hexadecimal digits')
  return value


def _git_output(*arguments: str) -> str:
  try:
    return subprocess.check_output(
      ['git', *arguments], cwd=REPO_ROOT, text=True,
      stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.CalledProcessError) as error:
    raise ValueError('cannot authenticate the repository checkout') from error


def attest_repository(expected_revision: str) -> dict[str, object]:
  """Authenticate an exact clean checkout without retaining local paths."""
  expected_revision = _lower_hex(
    expected_revision, 40, context='expected repository revision')
  revision = _git_output('rev-parse', 'HEAD')
  _lower_hex(revision, 40, context='repository revision')
  if revision != expected_revision:
    raise ValueError(
      f'repository revision mismatch: expected {expected_revision}, '
      f'found {revision}')
  if _git_output('status', '--porcelain=v1', '--untracked-files=all'):
    raise ValueError('repository checkout is not clean')
  return {'revision': revision, 'clean': True}


def attest_replay_sources() -> dict[str, object]:
  """Hash-pin the two Hydra source files used by this replay."""
  result: dict[str, object] = {}
  for role, path, expected_sha256 in (
      ('model_config', MODEL_CONFIG_PATH, MODEL_CONFIG_SHA256),
      ('data_config', DATA_CONFIG_PATH, DATA_CONFIG_SHA256)):
    if not path.is_file():
      raise FileNotFoundError(path)
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
      raise ValueError(
        f'{role} SHA256 mismatch: expected {expected_sha256}, '
        f'found {actual_sha256}')
    result[role] = {
      'filename': path.name,
      'sha256': actual_sha256,
      'size_bytes': path.stat().st_size,
    }
  return result


def _config_value(config: object, dotted_path: str) -> object:
  current = config
  for field in dotted_path.split('.'):
    if isinstance(current, Mapping):
      if field not in current:
        raise ValueError(f'runtime config is missing {dotted_path}')
      current = current[field]
    else:
      try:
        current = getattr(current, field)
      except (AttributeError, KeyError) as error:
        raise ValueError(f'runtime config is missing {dotted_path}') from error
  return current


def validate_seed_201_runtime_config(
    config: object,
    *,
    data_cache_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
  """Validate and identify the path-free semantic replay configuration."""
  observed = {
    field: _config_value(config, field)
    for field in EXPECTED_RUNTIME_CONFIG
  }
  if observed != EXPECTED_RUNTIME_CONFIG:
    drift = {
      field: {
        'expected': EXPECTED_RUNTIME_CONFIG[field],
        'observed': observed.get(field),
      }
      for field in EXPECTED_RUNTIME_CONFIG
      if observed.get(field) != EXPECTED_RUNTIME_CONFIG[field]
    }
    raise ValueError(f'seed-201 runtime configuration drifted: {drift}')
  identity_sha256 = canonical_sha256(observed)
  if identity_sha256 != RUNTIME_CONFIG_IDENTITY_SHA256:
    raise AssertionError('seed-201 runtime config identity constant drifted')
  expected_cache = str(data_cache_dir.expanduser().resolve())
  expected_output = str(output_dir.expanduser().resolve())
  if _config_value(config, 'data.cache_dir') != expected_cache:
    raise ValueError('runtime data cache path drifted')
  if _config_value(config, 'checkpointing.save_dir') != expected_output:
    raise ValueError('runtime replay output path drifted')
  return {
    'identity_sha256': identity_sha256,
    'semantic_config': dict(observed),
    'data_cache_path_sha256': hashlib.sha256(
      expected_cache.encode('utf-8')).hexdigest(),
  }


def authenticate_and_load_replay_model(
    *,
    model: torch.nn.Module,
    adapter_path: Path,
    manifest_path: Path,
    expected_adapter_sha256: str,
    expected_manifest_sha256: str,
    expectations: AuthenticatedProductionExpectations,
    expected_adapter_tensor_count: int = PRODUCTION_ADAPTER_TENSOR_COUNT,
    expected_adapter_parameter_count: int = PRODUCTION_ADAPTER_PARAMETER_COUNT,
    expected_adapter_tensor_bytes: int = PRODUCTION_ADAPTER_TENSOR_BYTES,
) -> dict[str, Any]:
  """Attest the actual backbone, then strict-load one authenticated v5 head."""
  before = attest_runtime_backbone(model.backbone, expectations)
  verification = verify_contextual_forest_adapter(
    adapter_path,
    manifest_path,
    expected_adapter_sha256=expected_adapter_sha256,
    expected_manifest_sha256=expected_manifest_sha256,
    model=model,
    expected_adapter_tensor_count=expected_adapter_tensor_count,
    expected_adapter_parameter_count=expected_adapter_parameter_count,
    expected_adapter_tensor_bytes=expected_adapter_tensor_bytes,
    expectations=expectations)
  after = attest_runtime_backbone(model.backbone, expectations)
  if before != after:
    raise AssertionError('runtime backbone changed while loading the adapter')
  if (verification['backbone_tensor_content_sha256']
      != before['backbone_tensor_content_sha256']
      or verification['backbone_tensor_schema_sha256']
      != before['backbone_tensor_schema_sha256']):
    raise AssertionError('verifier report does not bind the attested backbone')
  return verification


def _normalize_json(value: object) -> object:
  if torch.is_tensor(value):
    if value.numel() != 1:
      raise ValueError('evaluation returned a non-scalar tensor metric')
    return _normalize_json(value.detach().cpu().item())
  if isinstance(value, Mapping):
    return {str(key): _normalize_json(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_normalize_json(item) for item in value]
  if isinstance(value, float) and not math.isfinite(value):
    raise ValueError('evaluation returned a non-finite metric')
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  raise TypeError(f'evaluation returned non-JSON metric type {type(value)}')


def _run_existing_validation(model: torch.nn.Module) -> object:
  """Run the same callback/trainer/dataloader ordering as main._ppl_eval."""
  import dataloader  # noqa: PLC0415
  import hydra  # noqa: PLC0415

  if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(
      'the authoritative seed-201 replay requires exactly one visible CUDA GPU')
  config = model.config
  if not bool(config.eval.disable_ema) or model.ema is not None:
    raise ValueError('authenticated replay requires EMA to be disabled')
  if config.get('wandb', None) is not None:
    raise ValueError('authenticated replay forbids external experiment logging')
  callbacks = []
  if 'callbacks' in config:
    callbacks = [
      hydra.utils.instantiate(callback)
      for callback in config.callbacks.values()
    ]
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=str(Path(config.checkpointing.save_dir).resolve()),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=None)
  _, valid_ds = dataloader.get_dataloaders(
    config, model.tokenizer, skip_train=True, valid_seed=config.seed)
  model.to('cuda')
  try:
    return trainer.validate(model, valid_ds)
  finally:
    model.cpu()


def _validate_pairing_digest(output_dir: Path) -> dict[str, object]:
  path = output_dir / 'validation_pairing_digest.json'
  if not path.is_file():
    raise FileNotFoundError(
      'validation did not produce validation_pairing_digest.json')
  payload_bytes = path.read_bytes()
  try:
    payload = json.loads(payload_bytes)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError('validation pairing digest is not valid JSON') from error
  if (not isinstance(payload, Mapping)
      or payload.get('schema_version') != 1
      or payload.get('artifact') != 'structured_validation_pairing_digest'
      or payload.get('algorithm') != 'sha256'):
    raise ValueError('validation pairing digest schema mismatch')
  if payload.get('sha256') != EXPECTED_PAIRING_DIGEST_SHA256:
    raise ValueError(
      'seed-201 validation pairing digest differs from the historical replay')
  for field, expected in EXPECTED_PAIRING_INVENTORY.items():
    if payload.get(field) != expected:
      raise ValueError(
        f'seed-201 pairing {field} mismatch: expected {expected!r}, '
        f'found {payload.get(field)!r}')
  return {
    'filename': path.name,
    'file_sha256': hashlib.sha256(payload_bytes).hexdigest(),
    'stream_sha256': payload['sha256'],
    **EXPECTED_PAIRING_INVENTORY,
  }


def attest_runtime_environment() -> dict[str, object]:
  """Fail closed unless the historical Python/CUDA software stack is active."""
  packages = {}
  for name, expected in EXPECTED_PACKAGE_VERSIONS.items():
    try:
      observed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
      raise RuntimeError(f'required replay package is missing: {name}') \
        from error
    if observed != expected:
      raise RuntimeError(
        f'replay package {name} drifted: expected {expected}, '
        f'found {observed}')
    packages[name] = observed
  environment = {
    'python': platform.python_version(),
    'platform': platform.platform(),
    'torch': str(torch.__version__),
    'torch_cuda_build': torch.version.cuda,
    'packages': packages,
  }
  expected_fields = {
    'python': EXPECTED_PYTHON_VERSION,
    'platform': EXPECTED_PLATFORM,
    'torch': EXPECTED_TORCH_VERSION,
    'torch_cuda_build': EXPECTED_TORCH_CUDA_BUILD,
  }
  for field, expected in expected_fields.items():
    if environment[field] != expected:
      raise RuntimeError(
        f'replay environment {field} drifted: expected {expected}, '
        f'found {environment[field]}')
  try:
    gpu_row = subprocess.check_output([
      'nvidia-smi', '--query-gpu=name,driver_version,memory.total',
      '--format=csv,noheader,nounits',
    ], text=True, stderr=subprocess.STDOUT).strip()
  except (OSError, subprocess.CalledProcessError) as error:
    raise RuntimeError('cannot authenticate the replay GPU') from error
  rows = [row.strip() for row in gpu_row.splitlines() if row.strip()]
  if len(rows) != 1:
    raise RuntimeError('authoritative replay requires exactly one visible GPU')
  fields = [value.strip() for value in rows[0].split(',')]
  if (len(fields) != 3 or fields[0] != 'NVIDIA L4'
      or fields[1] != EXPECTED_NVIDIA_DRIVER_VERSION
      or fields[2] != '23034'):
    raise RuntimeError(
      'replay GPU identity drifted from the 23,034 MiB NVIDIA L4 with '
      f'driver {EXPECTED_NVIDIA_DRIVER_VERSION}: {rows[0]}')
  environment['gpu'] = {
    'name': fields[0],
    'driver_version': fields[1],
    'memory_total_mib': int(fields[2]),
  }
  return environment


def _safe_new_output_dir(path: Path) -> Path:
  output = path.expanduser().resolve()
  if output in {Path(output.anchor), Path.home().resolve(), REPO_ROOT}:
    raise ValueError('replay output directory is too broad')
  if output.exists():
    raise FileExistsError(
      f'{output} already exists; replay attempts are append-only')
  return output


def _run_authenticated_replay_unlocked(
    args: argparse.Namespace,
) -> tuple[dict[str, object], Path]:
  repository = attest_repository(args.expected_repository_sha)
  replay_sources = attest_replay_sources()
  runtime_environment = attest_runtime_environment()
  output_dir = _safe_new_output_dir(args.output_dir)
  data_cache_dir = args.data_cache_dir.expanduser().resolve()
  if not data_cache_dir.is_dir():
    raise FileNotFoundError(data_cache_dir)

  expectations = load_production_expectations(
    args.expectations,
    expected_sha256=args.expected_expectations_sha256)
  import lightning as L  # noqa: PLC0415
  L.seed_everything(SEED)
  model = build_production_model(
    model_config=MODEL_CONFIG_NAME,
    data_config=DATA_CONFIG_NAME,
    backbone_checkpoint=args.backbone_checkpoint,
    expectations=expectations,
    overrides=SEED_201_OVERRIDES,
    runtime_mode='ppl_eval',
    data_cache_dir=data_cache_dir,
    checkpoint_save_dir=output_dir)
  config_identity = validate_seed_201_runtime_config(
    model.config, data_cache_dir=data_cache_dir, output_dir=output_dir)
  verification = authenticate_and_load_replay_model(
    model=model,
    adapter_path=args.adapter,
    manifest_path=args.manifest,
    expected_adapter_sha256=args.expected_adapter_sha256,
    expected_manifest_sha256=args.expected_manifest_sha256,
    expectations=expectations)

  # Authentication and strict loading are complete before the append-only
  # attempt exists.  Any subsequent failure leaves a preserved partial attempt.
  output_dir.mkdir(parents=True, exist_ok=False)
  metrics = _normalize_json(_run_existing_validation(model))
  post_evaluation_backbone = attest_runtime_backbone(
    model.backbone, expectations)
  pairing = _validate_pairing_digest(output_dir)

  if attest_repository(args.expected_repository_sha) != repository:
    raise ValueError('repository identity changed during replay')
  if attest_replay_sources() != replay_sources:
    raise ValueError('replay config sources changed during replay')
  validate_seed_201_runtime_config(
    model.config, data_cache_dir=data_cache_dir, output_dir=output_dir)
  if (post_evaluation_backbone['backbone_tensor_content_sha256']
      != verification['backbone_tensor_content_sha256']):
    raise ValueError('runtime backbone changed during evaluation')

  payload: dict[str, object] = {
    'artifact_role': 'authenticated_contextual_forest_seed_201_replay',
    'schema_version': REPLAY_SCHEMA_VERSION,
    'status': 'complete',
    'repository': repository,
    'replay_sources': replay_sources,
    'runtime_config': config_identity,
    'adapter': {
      'filename': args.adapter.name,
      'sha256': verification['adapter_sha256'],
      'manifest_filename': args.manifest.name,
      'manifest_sha256': verification['manifest_sha256'],
      'strict_load': verification['strict_load'],
    },
    'authenticated_expectations': {
      'filename': args.expectations.name,
      'file_sha256': verification[
        'production_expectations_file_sha256'],
      'identity_sha256': verification[
        'production_expectations_identity_sha256'],
    },
    'backbone_attestation': {
      'wrapper_filename': args.backbone_checkpoint.name,
      'wrapper_sha256': verification['backbone_wrapper_sha256'],
      'wrapper_metadata_sha256': verification[
        'backbone_wrapper_metadata_sha256'],
      'tensor_schema_sha256': verification[
        'backbone_tensor_schema_sha256'],
      'tensor_content_sha256': verification[
        'backbone_tensor_content_sha256'],
    },
    'verifier_report': verification,
    'validation_metrics': metrics,
    'validation_pairing': pairing,
    'historical_reference': HISTORICAL_REFERENCE,
    'runtime_environment': runtime_environment,
  }
  return payload, output_dir


def _write_report_exclusive(
    output_dir: Path,
    payload: Mapping[str, object],
) -> Path:
  path = output_dir / REPLAY_REPORT_FILENAME
  rendered = json.dumps(
    dict(payload), indent=2, sort_keys=True, allow_nan=False) + '\n'
  with path.open('x', encoding='utf-8') as handle:
    handle.write(rendered)
    handle.flush()
    os.fsync(handle.fileno())
  return path


def run_authenticated_replay(args: argparse.Namespace) -> dict[str, object]:
  """Run one authoritative replay under the submission-wide CUDA lock."""
  from evaluation.tensor_train_baseline import (  # noqa: PLC0415
    GPU_EXCLUSIVITY_POLICY,
    SUBMISSION_GPU_LOCK,
  )
  from scripts.run_tensor_train_feasibility import (  # noqa: PLC0415
    _ForeignPidMonitor,
    _exclusive_gpu_lock,
  )

  with _exclusive_gpu_lock(SUBMISSION_GPU_LOCK) as lock_path:
    monitor = _ForeignPidMonitor(1.0)
    with monitor:
      payload, output_dir = _run_authenticated_replay_unlocked(args)
    evidence = monitor.snapshot(lock_path=lock_path)
    if evidence['foreign_pid_observations'] or evidence['monitor_errors']:
      raise RuntimeError('GPU exclusivity was lost during adapter replay')
  payload = {
    **payload,
    'gpu_exclusivity': {
      **evidence,
      'lock_path': Path(evidence['lock_path']).name,
      'policy': GPU_EXCLUSIVITY_POLICY,
    },
  }
  report_path = _write_report_exclusive(output_dir, payload)
  return {
    'status': 'complete',
    'report_filename': report_path.name,
    'report_sha256': _sha256_file(report_path),
    'adapter_sha256': args.expected_adapter_sha256,
    'pairing_digest_sha256': EXPECTED_PAIRING_DIGEST_SHA256,
  }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--adapter', type=Path, required=True)
  parser.add_argument('--expected-adapter-sha256', required=True)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--expected-manifest-sha256', required=True)
  parser.add_argument('--expectations', type=Path, required=True)
  parser.add_argument('--expected-expectations-sha256', required=True)
  parser.add_argument('--backbone-checkpoint', type=Path, required=True)
  parser.add_argument('--data-cache-dir', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--expected-repository-sha', required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  result = run_authenticated_replay(_parse_args(argv))
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
