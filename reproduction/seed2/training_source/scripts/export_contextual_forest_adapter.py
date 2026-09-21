#!/usr/bin/env python3
"""Export the anonymous contextual-forest head with runtime-schema checks.

Production artifacts use the repository's backward-compatible schema-v5
safetensors plus JSON manifest; explicit development exports retain schema v4.
This submission-facing entry point adds three fail-closed
conditions: the complete checkpoint is authenticated before pickle loading,
the extracted tensors exactly match ``model.structured_head`` by key, shape,
and dtype, and the audited production inventory is pinned by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.export_structured_adapter import (  # noqa: E402
  CONTROL_BY_MODES,
  RELEASED_BACKBONE_IDENTITY,
  RELEASE_TENSOR_COUNT,
  canonical_sha256,
  export_adapter,
  structured_decoder_identity_from_config,
  tensor_state_content_sha256,
  tensor_state_schema,
)
from scripts.prepare_released_mdlm_owt import (  # noqa: E402
  RELEASE_FILENAME,
  RELEASE_REPOSITORY,
  RELEASE_REVISION,
  RELEASE_SHA256,
  RELEASE_SIZE_BYTES,
)


PRODUCTION_ADAPTER_TENSOR_COUNT = 21
PRODUCTION_ADAPTER_PARAMETER_COUNT = 984_417
PRODUCTION_ADAPTER_TENSOR_BYTES = 3_937_668
PRODUCTION_ADAPTER_KEYS = frozenset({
  'edge_proposer.anchor_projection.bias',
  'edge_proposer.anchor_projection.weight',
  'edge_proposer.edge_scorer.0.bias',
  'edge_proposer.edge_scorer.0.weight',
  'edge_proposer.edge_scorer.2.bias',
  'edge_proposer.edge_scorer.2.weight',
  'edge_proposer.slot_projection.bias',
  'edge_proposer.slot_projection.weight',
  'factor_hidden_projection.bias',
  'factor_hidden_projection.weight',
  'factor_time_projection.weight',
  'hidden_norm.bias',
  'hidden_norm.weight',
  'time_embedding.mlp.0.bias',
  'time_embedding.mlp.0.weight',
  'time_embedding.mlp.2.bias',
  'time_embedding.mlp.2.weight',
  'token_factor_embedding.weight',
  'topology_hidden_projection.bias',
  'topology_hidden_projection.weight',
  'topology_time_projection.weight',
})
EXPECTATIONS_FIELDS = {
  'artifact_role',
  'schema_version',
  'source_checkpoint_sha256',
  'source_checkpoint_global_step',
  'adapter_tensor_count',
  'adapter_parameter_count',
  'adapter_tensor_bytes',
  'structured_decoder_identity_sha256',
  'released_backbone',
  'backbone_wrapper',
}
BACKBONE_WRAPPER_FIELDS = {
  'sha256',
  'size_bytes',
  'envelope_keys',
  'state_namespace',
  'tensor_count',
  'parameter_count',
  'tensor_bytes',
  'tensor_schema_sha256',
  'tensor_content_sha256',
  'metadata',
  'metadata_sha256',
}
BACKBONE_WRAPPER_METADATA = {
  'format_version': 1,
  'artifact_role': 'released_raw_mdlm_owt_backbone',
  'source_repository': RELEASE_REPOSITORY,
  'source_revision': RELEASE_REVISION,
  'source_filename': RELEASE_FILENAME,
  'source_sha256': RELEASE_SHA256,
  'source_size_bytes': RELEASE_SIZE_BYTES,
  'weight_namespace': 'backbone.*',
  'tensor_count': RELEASE_TENSOR_COUNT,
  'ema_available': False,
  'ema_used': False,
  'required_loader_setting': 'use_ema_backbone=false',
}


@dataclass(frozen=True)
class AuthenticatedProductionExpectations:
  """Closed expectations plus both byte-level and canonical identities."""

  payload: dict[str, Any]
  file_sha256: str
  identity_sha256: str


def _validated_authenticated_expectations(
    expectations: object,
) -> AuthenticatedProductionExpectations:
  if not isinstance(expectations, AuthenticatedProductionExpectations):
    raise TypeError('authenticated production expectations are required')
  payload = validate_production_expectations_payload(expectations.payload)
  _lower_sha256(
    expectations.file_sha256, context='production expectations file SHA256')
  identity_sha256 = canonical_sha256(payload)
  if expectations.identity_sha256 != identity_sha256:
    raise ValueError('production expectations canonical identity mismatch')
  return expectations


def _lower_sha256(value: object, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != 64
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(f'{context} must be 64 lowercase hexadecimal digits')
  return value


def _positive_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f'{context} must be a positive integer')
  return value


def _nonnegative_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f'{context} must be a non-negative integer')
  return value


def _canonical_json(payload: object) -> str:
  try:
    return json.dumps(
      payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
  except (TypeError, ValueError) as error:
    raise ValueError('expectations contain non-canonical JSON data') from error


def load_production_expectations(
    path: Path,
    *,
    expected_sha256: str,
) -> AuthenticatedProductionExpectations:
  """Load a separately authenticated, path-free export expectation record."""
  path = path.expanduser().resolve()
  expected_sha256 = _lower_sha256(
    expected_sha256, context='expected expectations SHA256')
  if not path.is_file():
    raise FileNotFoundError(path)
  payload_bytes = path.read_bytes()
  actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
  if actual_sha256 != expected_sha256:
    raise ValueError(
      f'expectations SHA256 mismatch: expected {expected_sha256}, '
      f'found {actual_sha256}')
  try:
    payload = json.loads(payload_bytes)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError('expectations are not valid JSON') from error
  validated = validate_production_expectations_payload(payload)
  return AuthenticatedProductionExpectations(
    payload=validated,
    file_sha256=actual_sha256,
    identity_sha256=canonical_sha256(validated))


def validate_production_expectations_payload(
    payload: object,
) -> dict[str, Any]:
  """Validate the closed schema shared by file and programmatic callers."""
  if not isinstance(payload, Mapping) or set(payload) != EXPECTATIONS_FIELDS:
    observed = set(payload) if isinstance(payload, Mapping) else set()
    raise ValueError(
      'contextual-forest expectations schema mismatch: '
      f'missing={sorted(EXPECTATIONS_FIELDS - observed)}, '
      f'unknown={sorted(observed - EXPECTATIONS_FIELDS)}')
  result = dict(payload)
  if (result['artifact_role'] != 'contextual_forest_adapter_expectations'
      or type(result['schema_version']) is not int
      or result['schema_version'] != 2):
    raise ValueError('unsupported contextual-forest expectations identity')
  _lower_sha256(
    result['source_checkpoint_sha256'],
    context='expectations source_checkpoint_sha256')
  _nonnegative_int(
    result['source_checkpoint_global_step'],
    context='expectations source_checkpoint_global_step')
  for field in (
      'adapter_tensor_count', 'adapter_parameter_count',
      'adapter_tensor_bytes'):
    _positive_int(result[field], context=f'expectations {field}')
  _lower_sha256(
    result['structured_decoder_identity_sha256'],
    context='expectations structured_decoder_identity_sha256')
  if (not isinstance(result['released_backbone'], Mapping)
      or _canonical_json(result['released_backbone'])
      != _canonical_json(RELEASED_BACKBONE_IDENTITY)):
    raise ValueError(
      'expectations released_backbone differs from the pinned release')
  wrapper = result['backbone_wrapper']
  if not isinstance(wrapper, Mapping) or set(wrapper) != \
      BACKBONE_WRAPPER_FIELDS:
    observed = set(wrapper) if isinstance(wrapper, Mapping) else set()
    raise ValueError(
      'backbone-wrapper expectations schema mismatch: '
      f'missing={sorted(BACKBONE_WRAPPER_FIELDS - observed)}, '
      f'unknown={sorted(observed - BACKBONE_WRAPPER_FIELDS)}')
  wrapper = dict(wrapper)
  _lower_sha256(wrapper['sha256'], context='backbone wrapper SHA256')
  _positive_int(wrapper['size_bytes'], context='backbone wrapper size')
  if wrapper['envelope_keys'] != ['metadata', 'state_dict']:
    raise ValueError('backbone wrapper envelope keys are not canonical')
  if wrapper['state_namespace'] != 'backbone.*':
    raise ValueError('backbone wrapper state namespace must be backbone.*')
  for field in ('tensor_count', 'parameter_count', 'tensor_bytes'):
    _positive_int(wrapper[field], context=f'backbone wrapper {field}')
  if wrapper['tensor_count'] != RELEASE_TENSOR_COUNT:
    raise ValueError('backbone wrapper tensor count differs from release')
  for field in (
      'tensor_schema_sha256', 'tensor_content_sha256', 'metadata_sha256'):
    _lower_sha256(wrapper[field], context=f'backbone wrapper {field}')
  if (not isinstance(wrapper['metadata'], Mapping)
      or _canonical_json(wrapper['metadata'])
      != _canonical_json(BACKBONE_WRAPPER_METADATA)):
    raise ValueError('backbone wrapper metadata differs from pinned release')
  if wrapper['metadata_sha256'] != canonical_sha256(wrapper['metadata']):
    raise ValueError('backbone wrapper metadata digest mismatch')
  result['backbone_wrapper'] = wrapper
  return result


def _backbone_inventory(
    state: Mapping[str, torch.Tensor],
) -> dict[str, int]:
  return {
    'tensor_count': len(state),
    'parameter_count': sum(value.numel() for value in state.values()),
    'tensor_bytes': sum(
      value.numel() * value.element_size() for value in state.values()),
  }


def load_authenticated_backbone_wrapper(
    path: Path,
    *,
    expectations: AuthenticatedProductionExpectations,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
  """Authenticate and load one exact wrapper byte string without reopening."""
  if not isinstance(expectations, AuthenticatedProductionExpectations):
    raise TypeError('authenticated production expectations are required')
  expectations = _validated_authenticated_expectations(expectations)
  path = path.expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  wrapper_expectation = expectations.payload['backbone_wrapper']
  wrapper_bytes = path.read_bytes()
  actual_sha256 = hashlib.sha256(wrapper_bytes).hexdigest()
  if actual_sha256 != wrapper_expectation['sha256']:
    raise ValueError(
      'backbone wrapper SHA256 mismatch: expected '
      f'{wrapper_expectation["sha256"]}, found {actual_sha256}')
  if len(wrapper_bytes) != wrapper_expectation['size_bytes']:
    raise ValueError(
      'backbone wrapper size differs from authenticated expectations')

  # This wrapper is tensor state plus primitive metadata.  The exact bytes
  # authenticated above are the bytes deserialized below; the path is never
  # reopened, and unsafe pickle fallback is intentionally forbidden.
  try:
    wrapper = torch.load(
      io.BytesIO(wrapper_bytes), map_location='cpu', weights_only=True)
  except Exception as error:
    raise ValueError(
      'authenticated backbone wrapper is not weights-only compatible') \
      from error
  if not isinstance(wrapper, Mapping) or set(wrapper) != \
      {'metadata', 'state_dict'}:
    raise ValueError('backbone wrapper envelope schema mismatch')
  metadata = wrapper['metadata']
  if (not isinstance(metadata, Mapping)
      or _canonical_json(metadata) != _canonical_json(
        wrapper_expectation['metadata'])
      or _canonical_json(metadata) != _canonical_json(
        BACKBONE_WRAPPER_METADATA)):
    raise ValueError('backbone wrapper metadata mismatch')
  if canonical_sha256(metadata) != wrapper_expectation['metadata_sha256']:
    raise ValueError('backbone wrapper metadata SHA256 mismatch')

  raw_state = wrapper['state_dict']
  if not isinstance(raw_state, Mapping) or not raw_state:
    raise ValueError('backbone wrapper state_dict must be non-empty')
  invalid_values = [
    key for key, value in raw_state.items() if not torch.is_tensor(value)]
  invalid_names = [
    key for key in raw_state
    if not isinstance(key, str) or not key.startswith('backbone.')
    or key == 'backbone.']
  if invalid_values:
    raise TypeError(
      f'backbone wrapper contains non-tensors: {invalid_values[:5]}')
  if invalid_names:
    raise ValueError(
      f'backbone wrapper contains invalid tensor names: {invalid_names[:5]}')
  state = {
    key.removeprefix('backbone.'): value.detach().cpu().contiguous()
    for key, value in sorted(raw_state.items())
  }
  if any(
      not key or key.startswith(('backbone.', 'structured_head.'))
      for key in state):
    raise ValueError('backbone wrapper retains a forbidden stripped namespace')
  inventory = _backbone_inventory(state)
  for field, value in inventory.items():
    if wrapper_expectation[field] != value:
      raise ValueError(
        f'backbone wrapper {field} mismatch: expected '
        f'{wrapper_expectation[field]}, found {value}')
  schema_sha256 = canonical_sha256(tensor_state_schema(state))
  content_sha256 = tensor_state_content_sha256(state)
  if schema_sha256 != wrapper_expectation['tensor_schema_sha256']:
    raise ValueError('backbone wrapper tensor schema SHA256 mismatch')
  if content_sha256 != wrapper_expectation['tensor_content_sha256']:
    raise ValueError('backbone wrapper tensor content SHA256 mismatch')

  attestation = {
    'attestation_type': 'authenticated_released_backbone_wrapper_v1',
    'production_expectations_file_sha256': expectations.file_sha256,
    'production_expectations_identity_sha256': expectations.identity_sha256,
    'backbone_wrapper_sha256': actual_sha256,
    'backbone_wrapper_size_bytes': len(wrapper_bytes),
    'backbone_wrapper_metadata_sha256': canonical_sha256(metadata),
    'backbone_tensor_count': inventory['tensor_count'],
    'backbone_parameter_count': inventory['parameter_count'],
    'backbone_tensor_bytes': inventory['tensor_bytes'],
    'backbone_tensor_schema_sha256': schema_sha256,
    'backbone_tensor_content_sha256': content_sha256,
    'released_backbone_identity_sha256': canonical_sha256(
      expectations.payload['released_backbone']),
    'released_backbone': dict(expectations.payload['released_backbone']),
  }
  return state, attestation


def production_provenance_from_expectations(
    expectations: AuthenticatedProductionExpectations,
) -> dict[str, object]:
  """Build the expected manifest attestation without reading a path."""
  if not isinstance(expectations, AuthenticatedProductionExpectations):
    raise TypeError('authenticated production expectations are required')
  expectations = _validated_authenticated_expectations(expectations)
  wrapper = expectations.payload['backbone_wrapper']
  return {
    'attestation_type': 'authenticated_released_backbone_wrapper_v1',
    'production_expectations_file_sha256': expectations.file_sha256,
    'production_expectations_identity_sha256': expectations.identity_sha256,
    'backbone_wrapper_sha256': wrapper['sha256'],
    'backbone_wrapper_size_bytes': wrapper['size_bytes'],
    'backbone_wrapper_metadata_sha256': wrapper['metadata_sha256'],
    'backbone_tensor_count': wrapper['tensor_count'],
    'backbone_parameter_count': wrapper['parameter_count'],
    'backbone_tensor_bytes': wrapper['tensor_bytes'],
    'backbone_tensor_schema_sha256': wrapper['tensor_schema_sha256'],
    'backbone_tensor_content_sha256': wrapper['tensor_content_sha256'],
    'released_backbone_identity_sha256': canonical_sha256(
      expectations.payload['released_backbone']),
    'released_backbone': dict(expectations.payload['released_backbone']),
  }


def _structured_components(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module, Mapping[str, object]]:
  head = getattr(model, 'structured_head', None)
  backbone = getattr(model, 'backbone', None)
  structured_config = getattr(model, 'structured_config', None)
  if not isinstance(head, torch.nn.Module):
    raise ValueError('model.structured_head must be an initialized module')
  if not isinstance(backbone, torch.nn.Module):
    raise ValueError('model.backbone must be an initialized module')
  if not isinstance(structured_config, Mapping):
    raise ValueError('model.structured_config must be a mapping')
  return head, backbone, structured_config


def attest_runtime_backbone(
    backbone: torch.nn.Module,
    expectations: AuthenticatedProductionExpectations,
) -> dict[str, object]:
  """Prove the actual runtime module equals the authenticated wrapper state."""
  if not isinstance(backbone, torch.nn.Module):
    raise TypeError('runtime backbone must be a torch.nn.Module')
  if not isinstance(expectations, AuthenticatedProductionExpectations):
    raise TypeError('authenticated production expectations are required')
  expectations = _validated_authenticated_expectations(expectations)
  state = {
    key: value.detach().cpu().contiguous()
    for key, value in sorted(backbone.state_dict().items())
  }
  wrapper = expectations.payload['backbone_wrapper']
  inventory = _backbone_inventory(state)
  for field, value in inventory.items():
    if wrapper[field] != value:
      raise ValueError(
        f'runtime backbone {field} differs from authenticated wrapper')
  schema_sha256 = canonical_sha256(tensor_state_schema(state))
  content_sha256 = tensor_state_content_sha256(state)
  if schema_sha256 != wrapper['tensor_schema_sha256']:
    raise ValueError(
      'runtime backbone tensor schema differs from authenticated wrapper')
  if content_sha256 != wrapper['tensor_content_sha256']:
    raise ValueError(
      'runtime backbone tensor content differs from authenticated wrapper')
  return production_provenance_from_expectations(expectations)


def export_contextual_forest_adapter(
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_global_step: int,
    model: torch.nn.Module,
    expected_adapter_tensor_count: int = PRODUCTION_ADAPTER_TENSOR_COUNT,
    expected_adapter_parameter_count: int = (
      PRODUCTION_ADAPTER_PARAMETER_COUNT),
    expected_adapter_tensor_bytes: int = PRODUCTION_ADAPTER_TENSOR_BYTES,
    expectations: AuthenticatedProductionExpectations | None = None,
    development_mode: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
  """Export a head that exactly fits the authenticated runtime model."""
  expected_checkpoint_sha256 = _lower_sha256(
    expected_checkpoint_sha256, context='expected checkpoint SHA256')
  _nonnegative_int(expected_global_step, context='expected global step')
  for field, value in (
      ('adapter_tensor_count', expected_adapter_tensor_count),
      ('adapter_parameter_count', expected_adapter_parameter_count),
      ('adapter_tensor_bytes', expected_adapter_tensor_bytes)):
    _positive_int(value, context=f'expected {field}')
  if output_path.suffix != '.safetensors':
    raise ValueError('released adapter output must end in .safetensors')
  if manifest_path.suffix != '.json':
    raise ValueError('released adapter manifest must end in .json')
  if expectations is None and not development_mode:
    raise ValueError(
      'authenticated production expectations are required; only explicit '
      'programmatic development_mode may export without them')
  if expectations is not None and development_mode:
    raise ValueError(
      'development_mode cannot be combined with production expectations')

  head, backbone, structured_config = _structured_components(model)
  production_inventory = (
    expected_adapter_tensor_count == PRODUCTION_ADAPTER_TENSOR_COUNT
    and expected_adapter_parameter_count == PRODUCTION_ADAPTER_PARAMETER_COUNT
    and expected_adapter_tensor_bytes == PRODUCTION_ADAPTER_TENSOR_BYTES)
  if production_inventory:
    head_state = head.state_dict()
    if set(head_state) != PRODUCTION_ADAPTER_KEYS:
      raise ValueError(
        'production structured_head key mismatch: '
        f'missing={sorted(PRODUCTION_ADAPTER_KEYS - set(head_state))}, '
        f'unexpected={sorted(set(head_state) - PRODUCTION_ADAPTER_KEYS)}')
    non_fp32 = [
      key for key, value in head_state.items()
      if value.dtype != torch.float32]
    if non_fp32:
      raise ValueError(
        f'production structured_head must be FP32: {non_fp32[:5]}')
  structured_identity, structured_identity_sha256 = (
    structured_decoder_identity_from_config(structured_config))
  if structured_identity['head_semantics']['fixed_edge_path'] is not None:
    raise ValueError(
      'anonymous adapters require inline fixed_edges or no fixed topology; '
      'fixed_edge_path would disclose a local path')
  modes = (
    structured_identity['topology_mode'], structured_identity['factor_mode'])
  control_identity = CONTROL_BY_MODES.get(modes)
  if control_identity != structured_identity['control_identity']:
    raise ValueError('runtime structured control identity is inconsistent')

  production_provenance = None
  if expectations is not None:
    if not isinstance(expectations, AuthenticatedProductionExpectations):
      raise TypeError(
        'production export requires expectations loaded from an '
        'authenticated file')
    expectations = _validated_authenticated_expectations(expectations)
    validated_expectations = expectations.payload
    expectation_checks = {
      'source_checkpoint_sha256': expected_checkpoint_sha256,
      'source_checkpoint_global_step': expected_global_step,
      'adapter_tensor_count': expected_adapter_tensor_count,
      'adapter_parameter_count': expected_adapter_parameter_count,
      'adapter_tensor_bytes': expected_adapter_tensor_bytes,
      'structured_decoder_identity_sha256': structured_identity_sha256,
    }
    for field, observed in expectation_checks.items():
      if validated_expectations.get(field) != observed:
        raise ValueError(
          f'production expectations {field} mismatch: expected '
          f'{validated_expectations.get(field)!r}, found {observed!r}')
    if _canonical_json(validated_expectations.get('released_backbone')) != \
        _canonical_json(RELEASED_BACKBONE_IDENTITY):
      raise ValueError('production expectations backbone identity mismatch')
    production_provenance = attest_runtime_backbone(backbone, expectations)

  manifest = export_adapter(
    checkpoint_path,
    output_path,
    manifest_path,
    expected_checkpoint_sha256=expected_checkpoint_sha256,
    control_identity=str(control_identity),
    topology_mode=str(structured_identity['topology_mode']),
    factor_mode=str(structured_identity['factor_mode']),
    candidate_k=int(structured_identity['candidate_top_k']),
    independent_mode=bool(structured_identity['independent_mode']),
    topology_weight=float(structured_identity['topology_weight']),
    expected_global_step=expected_global_step,
    expected_backbone_tensors=RELEASE_TENSOR_COUNT,
    expected_structured_head=head,
    expected_frozen_backbone=backbone,
    expected_adapter_tensor_count=expected_adapter_tensor_count,
    expected_adapter_parameter_count=expected_adapter_parameter_count,
    expected_adapter_tensor_bytes=expected_adapter_tensor_bytes,
    production_provenance=production_provenance,
    overwrite=overwrite)
  if manifest['structured_decoder_identity_sha256'] != \
      structured_identity_sha256:
    raise AssertionError('exported structured identity digest drifted')
  return manifest


def _register_resolvers() -> None:
  from omegaconf import OmegaConf

  resolvers = {
    'cwd': os.getcwd,
    'device_count': torch.cuda.device_count,
    'eval': eval,
    'div_up': lambda left, right: (left + right - 1) // right,
  }
  for name, resolver in resolvers.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)


def build_production_model(
    *,
    model_config: str,
    data_config: str,
    backbone_checkpoint: Path,
    expectations: AuthenticatedProductionExpectations,
    overrides: Sequence[str],
    runtime_mode: str = 'train',
    data_cache_dir: Path | None = None,
    checkpoint_save_dir: Path | None = None,
) -> torch.nn.Module:
  """Construct the release target through the repository's Diffusion path."""
  if runtime_mode not in {'train', 'ppl_eval'}:
    raise ValueError('runtime_mode must be train or ppl_eval')
  backbone_state, loaded_attestation = load_authenticated_backbone_wrapper(
    backbone_checkpoint, expectations=expectations)
  expected_attestation = production_provenance_from_expectations(expectations)
  if _canonical_json(loaded_attestation) != _canonical_json(
      expected_attestation):
    raise AssertionError('authenticated backbone attestation drifted')

  import dataloader
  import diffusion
  import hydra
  from omegaconf import open_dict

  _register_resolvers()
  with hydra.initialize_config_dir(
      config_dir=str(REPO_ROOT / 'configs'), version_base=None):
    config = hydra.compose(
      config_name='config',
      overrides=[
        f'model={model_config}',
        f'data={data_config}',
        *overrides,
      ])
  with open_dict(config):
    config.mode = runtime_mode
    if data_cache_dir is not None:
      config.data.cache_dir = str(data_cache_dir.expanduser().resolve())
    if checkpoint_save_dir is not None:
      config.checkpointing.save_dir = str(
        checkpoint_save_dir.expanduser().resolve())
    # The authenticated bytes above are loaded directly below.  Leaving this
    # unset prevents Diffusion from reopening a replaceable filesystem path.
    config.model.structured_decoder.training.backbone_checkpoint = None
    config.model.structured_decoder.training.require_pretrained_backbone = False
    config.model.structured_decoder.training.use_ema_backbone = False
    config.model.structured_decoder.training.strict_backbone_checkpoint = True
    config.eval.checkpoint_path = ''
    config.eval.adapter_checkpoint = ''
    config.eval.adapter_sha256 = ''
    config.eval.adapter_manifest = ''
    config.eval.adapter_manifest_sha256 = ''
    config.eval.disable_ema = True
    config.eval.compute_generative_perplexity = False
    config.eval.generate_samples = False
    config.training.ema = 0.0
    config.checkpointing.resume_from_ckpt = False
  tokenizer = dataloader.get_tokenizer(config)
  with warnings.catch_warnings():
    warnings.filterwarnings(
      'ignore', message='structured head is freezing a backbone without.*')
    model = diffusion.Diffusion(config, tokenizer=tokenizer).cpu().eval()
  if model.structured_head is None:
    raise ValueError('production config did not initialize structured_head')
  try:
    incompatible = model.backbone.load_state_dict(
      backbone_state, strict=True)
  except RuntimeError as error:
    raise ValueError(
      f'authenticated backbone strict-load mismatch: {error}') from error
  if incompatible.missing_keys or incompatible.unexpected_keys:
    raise ValueError(
      'authenticated backbone strict load returned incompatible keys')
  model.backbone.requires_grad_(False)
  attest_runtime_backbone(model.backbone, expectations)
  model.structured_backbone_provenance = loaded_attestation
  return model


def _args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      'Export a model-checked anonymous contextual-forest adapter.'))
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--expected-checkpoint-sha256', required=True)
  parser.add_argument('--expected-global-step', type=int, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--model-config', default='contextual-forest-small')
  parser.add_argument('--data-config', default='train_openwebtext_pinned')
  parser.add_argument('--backbone-checkpoint', type=Path, required=True)
  parser.add_argument('--override', action='append', default=[])
  parser.add_argument('--expectations', type=Path, required=True)
  parser.add_argument('--expected-expectations-sha256', required=True)
  parser.add_argument('--force', action='store_true')
  return parser.parse_args(argv)


def build_export_cli_result(
    output_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
  """Build a path-free JSON result safe to retain in anonymous logs."""
  output_path = output_path.expanduser().resolve()
  manifest_path = manifest_path.expanduser().resolve()
  return {
    'adapter': {
      'filename': output_path.name,
      'sha256': manifest['adapter_sha256'],
      'size_bytes': manifest['adapter_size_bytes'],
    },
    'manifest': {
      'filename': manifest_path.name,
      'sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
      'size_bytes': manifest_path.stat().st_size,
    },
    'inventory': {
      field: manifest[field]
      for field in (
        'adapter_tensor_count', 'adapter_parameter_count',
        'adapter_tensor_bytes')
    },
    'structured_decoder_identity_sha256': (
      manifest['structured_decoder_identity_sha256']),
    'released_backbone_identity_sha256': canonical_sha256(
      manifest['released_backbone']),
  }


def main(argv=None) -> int:
  args = _args(argv)
  expectations = load_production_expectations(
    args.expectations,
    expected_sha256=args.expected_expectations_sha256)
  model = build_production_model(
    model_config=args.model_config,
    data_config=args.data_config,
    backbone_checkpoint=args.backbone_checkpoint,
    expectations=expectations,
    overrides=args.override)
  manifest = export_contextual_forest_adapter(
    args.checkpoint,
    args.output,
    args.manifest,
    expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    expected_global_step=args.expected_global_step,
    model=model,
    expectations=expectations,
    overwrite=args.force)
  result = build_export_cli_result(args.output, args.manifest, manifest)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
