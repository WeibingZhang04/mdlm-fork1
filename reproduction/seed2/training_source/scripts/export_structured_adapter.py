#!/usr/bin/env python3
"""Export a portable structured head from a frozen-backbone checkpoint.

The training checkpoint contains both the immutable released backbone and the
small learned structured head.  This exporter fails closed on unexpected state
names, removes the redundant frozen tensors, and writes the learned tensors in
the non-executable safetensors format plus a provenance manifest. Both files
cryptographically bind the canonical control, topology/factor modes, and
candidate-set size used to fit the head.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from safetensors.torch import load as load_safetensors
from safetensors.torch import save_file
import torch

# Keep both ``python -m scripts.export_structured_adapter`` and direct script
# invocation usable from an arbitrary working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_released_mdlm_owt import (  # noqa: E402
  RELEASE_REPOSITORY,
  RELEASE_REVISION,
  RELEASE_SHA256,
  RELEASE_SIZE_BYTES,
  RELEASE_TENSOR_COUNT,
  sha256_file,
)


BACKBONE_PREFIX = 'backbone.'
ADAPTER_PREFIX = 'structured_head.'
CONTROL_MODES = {
  'dynamic_dynamic': ('dynamic', 'dynamic'),
  'fixed_dynamic': ('fixed', 'dynamic'),
  'dynamic_fixed': ('dynamic', 'fixed'),
  'static_static': ('fixed', 'fixed'),
}
CONTROL_BY_MODES = {modes: name for name, modes in CONTROL_MODES.items()}
STRUCTURED_IDENTITY_FIELDS = {
  'control_identity', 'topology_mode', 'factor_mode', 'candidate_top_k',
  'independent_mode', 'topology_weight', 'head_semantics',
  'training_semantics',
}
PRODUCTION_PROVENANCE_FIELDS = {
  'attestation_type',
  'production_expectations_file_sha256',
  'production_expectations_identity_sha256',
  'backbone_wrapper_sha256',
  'backbone_wrapper_size_bytes',
  'backbone_wrapper_metadata_sha256',
  'backbone_tensor_count',
  'backbone_parameter_count',
  'backbone_tensor_bytes',
  'backbone_tensor_schema_sha256',
  'backbone_tensor_content_sha256',
  'released_backbone_identity_sha256',
  'released_backbone',
}
HEAD_INTEGER_FIELDS = (
  'rank', 'time_embed_dim', 'topology_dim', 'local_window',
  'num_anchor_slots', 'contextual_neighbors', 'component_size_cap',
)
TRAINING_SEMANTIC_FIELDS = (
  'objective_name', 'factorized_aux_weight', 'topology_strategy',
  'topology_temperature', 'topology_minimum_choices',
  'topology_edge_weight', 'topology_anchor_weight',
  'topology_slot_weight', 'topology_on_validation',
)
RELEASED_BACKBONE_IDENTITY = {
  'repository': RELEASE_REPOSITORY,
  'revision': RELEASE_REVISION,
  'source_sha256': RELEASE_SHA256,
  'source_size_bytes': RELEASE_SIZE_BYTES,
  'tensor_count': RELEASE_TENSOR_COUNT,
}


def _canonical_json(payload: object) -> str:
  try:
    return json.dumps(
      payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
  except (TypeError, ValueError) as error:
    raise ValueError('payload is not canonical JSON data') from error


def canonical_sha256(payload: object) -> str:
  return hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()


def tensor_state_schema(
    state: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
  """Return a canonical key/shape/dtype schema for a tensor state."""
  if not isinstance(state, Mapping) or not state:
    raise ValueError('tensor state must be a non-empty mapping')
  invalid = [
    key for key, value in state.items()
    if not isinstance(key, str) or not key or not torch.is_tensor(value)]
  if invalid:
    raise ValueError(f'tensor state contains invalid entries: {invalid[:5]}')
  return {
    key: {'shape': list(state[key].shape), 'dtype': str(state[key].dtype)}
    for key in sorted(state)
  }


def tensor_state_content_sha256(
    state: Mapping[str, torch.Tensor],
) -> str:
  """Hash canonical tensor schema plus exact contiguous CPU tensor bytes."""
  schema = tensor_state_schema(state)
  digest = hashlib.sha256()
  digest.update(b'contextual-forest-tensor-state-v1\0')
  digest.update(_canonical_json(schema).encode('utf-8'))
  digest.update(b'\0')
  for key in sorted(state):
    encoded_key = key.encode('utf-8')
    digest.update(len(encoded_key).to_bytes(8, byteorder='little'))
    digest.update(encoded_key)
    value = state[key].detach().cpu().contiguous()
    byte_view = value.view(torch.uint8).numpy()
    digest.update(memoryview(byte_view))
  return digest.hexdigest()


def validate_production_provenance_payload(
    payload: object,
) -> dict[str, object]:
  """Validate the closed provenance record embedded in schema-v5 exports."""
  if not isinstance(payload, Mapping) or set(payload) != \
      PRODUCTION_PROVENANCE_FIELDS:
    observed = set(payload) if isinstance(payload, Mapping) else set()
    raise ValueError(
      'production provenance schema mismatch: '
      f'missing={sorted(PRODUCTION_PROVENANCE_FIELDS - observed)}, '
      f'unknown={sorted(observed - PRODUCTION_PROVENANCE_FIELDS)}')
  result = dict(payload)
  if result['attestation_type'] != \
      'authenticated_released_backbone_wrapper_v1':
    raise ValueError('unsupported production backbone attestation')
  for field in (
      'production_expectations_file_sha256',
      'production_expectations_identity_sha256',
      'backbone_wrapper_sha256',
      'backbone_wrapper_metadata_sha256',
      'backbone_tensor_schema_sha256',
      'backbone_tensor_content_sha256',
      'released_backbone_identity_sha256'):
    _lower_sha256(result[field], context=field)
  for field in (
      'backbone_wrapper_size_bytes', 'backbone_tensor_count',
      'backbone_parameter_count', 'backbone_tensor_bytes'):
    _positive_int(result[field], context=field)
  released = result['released_backbone']
  if (not isinstance(released, Mapping)
      or _canonical_json(released) != _canonical_json(
        RELEASED_BACKBONE_IDENTITY)):
    raise ValueError(
      'production provenance does not establish the pinned raw release')
  if result['released_backbone_identity_sha256'] != canonical_sha256(released):
    raise ValueError('production raw-release identity digest mismatch')
  return result


def safetensors_metadata_from_bytes(payload: bytes) -> dict[str, str]:
  """Read the non-executable safetensors header metadata fail-closed."""
  if len(payload) < 8:
    raise ValueError('structured adapter is not a valid safetensors file')
  header_size = int.from_bytes(payload[:8], byteorder='little', signed=False)
  if header_size <= 0 or 8 + header_size > len(payload):
    raise ValueError('structured adapter has an invalid safetensors header')
  try:
    header = json.loads(payload[8:8 + header_size])
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(
      'structured adapter has an invalid safetensors JSON header') from error
  metadata = header.get('__metadata__') if isinstance(header, Mapping) else None
  if (not isinstance(metadata, Mapping)
      or any(not isinstance(key, str) or not isinstance(value, str)
             for key, value in metadata.items())):
    raise ValueError('structured adapter lacks valid safetensors metadata')
  return dict(metadata)


def canonicalize_safetensors_bytes(payload: bytes) -> bytes:
  """Return equivalent safetensors bytes with a canonical sorted header.

  The safetensors writer accepts a Python metadata mapping but does not promise
  stable JSON member order.  Tensor offsets are relative to the data section,
  so canonicalizing and re-padding only the header preserves every tensor byte
  while making repeated exports byte-identical.
  """
  if len(payload) < 8:
    raise ValueError('structured adapter is not a valid safetensors file')
  header_size = int.from_bytes(payload[:8], byteorder='little', signed=False)
  if header_size <= 0 or 8 + header_size > len(payload):
    raise ValueError('structured adapter has an invalid safetensors header')
  try:
    header = json.loads(payload[8:8 + header_size])
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(
      'structured adapter has an invalid safetensors JSON header') from error
  if not isinstance(header, Mapping):
    raise ValueError('structured adapter header must be a JSON object')
  canonical = json.dumps(
    header, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    allow_nan=False).encode('utf-8')
  padding = (-len(canonical)) % 8
  canonical += b' ' * padding
  tensor_bytes = payload[8 + header_size:]
  return len(canonical).to_bytes(8, byteorder='little') + canonical + tensor_bytes


def _finite_float(value: object, *, context: str) -> float:
  if (not isinstance(value, (int, float)) or isinstance(value, bool)
      or not math.isfinite(float(value))):
    raise ValueError(f'{context} must be finite')
  return float(value)


def _positive_int(value: object, *, context: str) -> int:
  if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
    raise ValueError(f'{context} must be a positive integer')
  return value


def _nonnegative_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f'{context} must be a non-negative integer')
  return value


def _lower_sha256(value: object, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != 64
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(f'{context} must be 64 lowercase hexadecimal digits')
  return value


def _metadata_integer(
    metadata: Mapping[str, str],
    field: str,
    *,
    positive: bool,
) -> int:
  value = metadata.get(field)
  if (not isinstance(value, str) or not value or not value.isascii()
      or not value.isdecimal()
      or (len(value) > 1 and value.startswith('0'))
      or (positive and value == '0')):
    qualifier = 'positive' if positive else 'non-negative'
    raise ValueError(
      f'safetensors metadata {field} must be a canonical {qualifier} integer')
  return int(value)


def _canonical_fixed_edges(value: object) -> list[list[int]] | None:
  if value is None:
    return None
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise ValueError('structured_decoder.fixed_edges must be null or pairs')
  edges = []
  for index, edge in enumerate(value):
    if (isinstance(edge, (str, bytes)) or not isinstance(edge, Sequence)
        or len(edge) != 2):
      raise ValueError(
        f'structured_decoder.fixed_edges[{index}] must be an integer pair')
    endpoints = []
    for endpoint in edge:
      if (not isinstance(endpoint, int) or isinstance(endpoint, bool)
          or endpoint < 0):
        raise ValueError(
          f'structured_decoder.fixed_edges[{index}] has an invalid endpoint')
      endpoints.append(endpoint)
    edges.append(endpoints)
  return edges


def structured_decoder_identity_from_config(
    structured: Mapping[str, object],
    *,
    control_identity: str | None = None,
) -> tuple[dict[str, object], str]:
  """Build the canonical semantic identity required to load an adapter.

  The digest binds architecture, active/disabled factor behavior, candidate
  support, and the training semantics that can change learned adapter bytes.
  Paths to the separately pinned backbone are intentionally excluded.
  """
  if not isinstance(structured, Mapping):
    raise ValueError('model.structured_decoder must be a mapping')
  topology_mode = structured.get('topology_mode')
  factor_mode = structured.get('factor_mode')
  inferred_control = CONTROL_BY_MODES.get((topology_mode, factor_mode))
  if inferred_control is None:
    raise ValueError(
      'structured topology/factor modes do not name a frozen control')
  if control_identity is None:
    control_identity = inferred_control
  elif control_identity != inferred_control:
    raise ValueError(
      f'control {control_identity!r} requires topology/factor modes '
      f'{CONTROL_MODES.get(control_identity)}, found '
      f'{(topology_mode, factor_mode)}')

  candidate_top_k = _positive_int(
    structured.get('top_k'), context='structured_decoder.top_k')
  independent_mode = structured.get('independent_mode')
  if not isinstance(independent_mode, bool):
    raise ValueError('structured_decoder.independent_mode must be boolean')
  training = structured.get('training')
  if not isinstance(training, Mapping):
    raise ValueError('structured_decoder.training must be a mapping')
  topology_weight = _finite_float(
    training.get('topology_weight'),
    context='structured_decoder.training.topology_weight')
  if topology_weight < 0:
    raise ValueError(
      'structured_decoder.training.topology_weight must be non-negative')

  head_semantics: dict[str, object] = {}
  for field in HEAD_INTEGER_FIELDS:
    head_semantics[field] = _positive_int(
      structured.get(field), context=f'structured_decoder.{field}')
  min_edge_score = structured.get('min_edge_score')
  head_semantics['min_edge_score'] = (
    None if min_edge_score is None else _finite_float(
      min_edge_score, context='structured_decoder.min_edge_score'))
  head_semantics['fixed_edges'] = _canonical_fixed_edges(
    structured.get('fixed_edges'))
  fixed_edge_path = structured.get('fixed_edge_path')
  if fixed_edge_path in (None, ''):
    fixed_edge_path = None
  elif not isinstance(fixed_edge_path, str):
    raise ValueError('structured_decoder.fixed_edge_path must be a string')
  head_semantics['fixed_edge_path'] = fixed_edge_path

  training_semantics: dict[str, object] = {}
  for field in TRAINING_SEMANTIC_FIELDS:
    value = training.get(field)
    if field in {
        'factorized_aux_weight', 'topology_temperature',
        'topology_edge_weight', 'topology_anchor_weight',
        'topology_slot_weight'}:
      value = _finite_float(
        value, context=f'structured_decoder.training.{field}')
    elif field == 'topology_minimum_choices':
      value = _positive_int(
        value, context=f'structured_decoder.training.{field}')
    elif field == 'topology_on_validation':
      if not isinstance(value, bool):
        raise ValueError(
          'structured_decoder.training.topology_on_validation must be '
          'boolean')
    elif not isinstance(value, str) or not value:
      raise ValueError(
        f'structured_decoder.training.{field} must be non-empty')
    training_semantics[field] = value

  identity = {
    'control_identity': control_identity,
    'topology_mode': topology_mode,
    'factor_mode': factor_mode,
    'candidate_top_k': candidate_top_k,
    'independent_mode': independent_mode,
    'topology_weight': topology_weight,
    'head_semantics': head_semantics,
    'training_semantics': training_semantics,
  }
  return identity, canonical_sha256(identity)


def _validated_adapter_identity(
    *,
    control_identity: str,
    topology_mode: str,
    factor_mode: str,
    candidate_k: int,
    independent_mode: bool,
    topology_weight: float,
) -> tuple[dict[str, object], str]:
  expected_modes = CONTROL_MODES.get(control_identity)
  if expected_modes is None:
    raise ValueError(
      f'unknown structured control identity: {control_identity!r}')
  actual_modes = (topology_mode, factor_mode)
  if actual_modes != expected_modes:
    raise ValueError(
      f'control {control_identity!r} requires topology/factor modes '
      f'{expected_modes}, found {actual_modes}')
  _positive_int(candidate_k, context='candidate_k')
  if not isinstance(independent_mode, bool):
    raise ValueError('independent_mode must be boolean')
  topology_weight = _finite_float(
    topology_weight, context='topology_weight')
  if topology_weight < 0:
    raise ValueError('topology_weight must be non-negative')
  identity = {
    'control_identity': control_identity,
    'topology_mode': topology_mode,
    'factor_mode': factor_mode,
    'candidate_top_k': candidate_k,
    'independent_mode': independent_mode,
    'topology_weight': topology_weight,
  }
  return identity, canonical_sha256(identity)


def _validate_checkpoint_adapter_identity(
    checkpoint: Mapping[str, object],
    *,
    expected_identity: Mapping[str, object],
) -> tuple[dict[str, object], str]:
  hyperparameters = checkpoint.get('hyper_parameters')
  if not isinstance(hyperparameters, Mapping):
    raise ValueError(
      'checkpoint lacks the saved hyper_parameters needed to verify its '
      'structured-decoder identity')
  config = hyperparameters.get('config')
  model = config.get('model') if isinstance(config, Mapping) else None
  structured = (
    model.get('structured_decoder') if isinstance(model, Mapping) else None)
  if not isinstance(structured, Mapping):
    raise ValueError(
      'checkpoint hyper_parameters lack model.structured_decoder')
  training = structured.get('training')
  observed = {
    'topology_mode': structured.get('topology_mode'),
    'factor_mode': structured.get('factor_mode'),
    'candidate_top_k': structured.get('top_k'),
    'independent_mode': structured.get('independent_mode'),
    'topology_weight': (
      training.get('topology_weight') if isinstance(training, Mapping)
      else None),
  }
  expected = {
    field: expected_identity[field]
    for field in (
      'topology_mode', 'factor_mode', 'candidate_top_k',
      'independent_mode', 'topology_weight')}
  if observed != expected:
    raise ValueError(
      'checkpoint structured-decoder identity mismatch: '
      f'expected {expected}, found {observed}')
  full_identity, full_digest = structured_decoder_identity_from_config(
    structured,
    control_identity=str(expected_identity['control_identity']))
  return full_identity, full_digest


def _validated_state_dict(
    checkpoint: Mapping[str, object],
    *,
    expected_backbone_tensors: int | None,
) -> tuple[dict[str, torch.Tensor], int]:
  state = checkpoint.get('state_dict')
  if not isinstance(state, Mapping) or not state:
    raise ValueError('checkpoint has no non-empty state_dict mapping')
  invalid_values = [
    key for key, value in state.items() if not torch.is_tensor(value)]
  if invalid_values:
    raise TypeError(
      f'checkpoint state contains non-tensors: {invalid_values[:5]}')
  invalid_names = [
    key for key in state if not isinstance(key, str) or not key]
  if invalid_names:
    raise ValueError(
      f'checkpoint state contains invalid tensor names: {invalid_names[:5]}')
  unexpected = [
    key for key in state
    if not key.startswith((BACKBONE_PREFIX, ADAPTER_PREFIX))]
  if unexpected:
    raise ValueError(
      'checkpoint state contains tensors outside backbone.* and '
      f'structured_head.*: {unexpected[:5]}')
  empty_namespaces = [
    key for key in state if key in {BACKBONE_PREFIX, ADAPTER_PREFIX}]
  if empty_namespaces:
    raise ValueError(
      f'checkpoint state contains empty tensor names: {empty_namespaces}')

  backbone_count = sum(key.startswith(BACKBONE_PREFIX) for key in state)
  if (expected_backbone_tensors is not None
      and backbone_count != expected_backbone_tensors):
    raise ValueError(
      f'backbone tensor-count mismatch: expected '
      f'{expected_backbone_tensors}, found {backbone_count}')
  adapter = {
    key.removeprefix(ADAPTER_PREFIX): value.detach().cpu().contiguous()
    for key, value in state.items() if key.startswith(ADAPTER_PREFIX)
  }
  if not adapter:
    raise ValueError('checkpoint contains no structured_head.* tensors')
  if any(not key for key in adapter):
    raise ValueError('invalid empty adapter tensor name after prefix removal')
  forbidden = [
    key for key in adapter
    if key.startswith((BACKBONE_PREFIX, ADAPTER_PREFIX))]
  if forbidden:
    raise ValueError(
      'prefix-stripped adapter keys retain a forbidden namespace: '
      f'{forbidden[:5]}')
  return dict(sorted(adapter.items())), backbone_count


def validate_tensor_state_against_module(
    state: Mapping[str, torch.Tensor],
    module: torch.nn.Module,
    *,
    context: str,
    compare_values: bool = False,
) -> dict[str, dict[str, object]]:
  """Require an exact key/shape/dtype match with a runtime module.

  ``torch.nn.Module.load_state_dict(strict=True)`` checks names and shapes but
  permits dtype conversion.  Released adapters must not depend on an implicit
  cast, so this check runs before every strict load.  ``compare_values`` is
  useful for proving that the frozen backbone saved in a training checkpoint
  is byte-for-byte the same tensor state as the separately authenticated
  runtime backbone.
  """
  if not isinstance(module, torch.nn.Module):
    raise TypeError(f'{context} target must be a torch.nn.Module')
  if not isinstance(state, Mapping) or not state:
    raise ValueError(f'{context} state must be a non-empty mapping')
  invalid_values = [
    key for key, value in state.items() if not torch.is_tensor(value)]
  if invalid_values:
    raise TypeError(f'{context} state contains non-tensors: {invalid_values[:5]}')
  invalid_names = [
    key for key in state if not isinstance(key, str) or not key]
  if invalid_names:
    raise ValueError(
      f'{context} state contains invalid tensor names: {invalid_names[:5]}')

  expected = module.state_dict()
  observed_keys = set(state)
  expected_keys = set(expected)
  if observed_keys != expected_keys:
    raise ValueError(
      f'{context} key mismatch: '
      f'missing={sorted(expected_keys - observed_keys)}, '
      f'unexpected={sorted(observed_keys - expected_keys)}')

  schema = {}
  for key in sorted(expected):
    observed = state[key]
    reference = expected[key]
    if tuple(observed.shape) != tuple(reference.shape):
      raise ValueError(
        f'{context} shape mismatch for {key}: expected '
        f'{tuple(reference.shape)}, found {tuple(observed.shape)}')
    if observed.dtype != reference.dtype:
      raise ValueError(
        f'{context} dtype mismatch for {key}: expected '
        f'{reference.dtype}, found {observed.dtype}')
    if compare_values and not torch.equal(
        observed.detach().cpu(), reference.detach().cpu()):
      raise ValueError(f'{context} value mismatch for {key}')
    schema[key] = {
      'shape': list(observed.shape),
      'dtype': str(observed.dtype),
    }
  return schema


def validate_adapter_inventory(
    state: Mapping[str, torch.Tensor],
    *,
    expected_tensor_count: int | None = None,
    expected_parameter_count: int | None = None,
    expected_tensor_bytes: int | None = None,
) -> dict[str, int]:
  """Derive and optionally pin the complete adapter tensor inventory."""
  if not isinstance(state, Mapping) or not state:
    raise ValueError('structured adapter state must be a non-empty mapping')
  invalid_values = [
    key for key, value in state.items() if not torch.is_tensor(value)]
  if invalid_values:
    raise TypeError(
      f'structured adapter contains non-tensors: {invalid_values[:5]}')
  inventory = {
    'adapter_tensor_count': len(state),
    'adapter_parameter_count': sum(value.numel() for value in state.values()),
    'adapter_tensor_bytes': sum(
      value.numel() * value.element_size() for value in state.values()),
  }
  expected = {
    'adapter_tensor_count': expected_tensor_count,
    'adapter_parameter_count': expected_parameter_count,
    'adapter_tensor_bytes': expected_tensor_bytes,
  }
  for field, expected_value in expected.items():
    if expected_value is None:
      continue
    _positive_int(expected_value, context=f'expected {field}')
    if inventory[field] != expected_value:
      raise ValueError(
        f'{field} mismatch: expected {expected_value}, '
        f'found {inventory[field]}')
  return inventory


def load_adapter_state(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, torch.Tensor]:
  """Low-level tensor reader; inference callers must validate the manifest."""
  path = path.resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = path.read_bytes()
  if expected_sha256 is not None:
    normalized_expected = (
      expected_sha256.lower()
      if isinstance(expected_sha256, str) else expected_sha256)
    expected_sha256 = _lower_sha256(
      normalized_expected, context='expected adapter SHA256')
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
      raise ValueError(
        f'adapter SHA256 mismatch: expected {expected_sha256}, '
        f'found {actual_sha256}')
  state = load_safetensors(payload)
  if not state:
    raise ValueError('adapter file contains no tensors')
  invalid = [
    key for key in state
    if key.startswith((BACKBONE_PREFIX, ADAPTER_PREFIX))]
  if invalid:
    raise ValueError(
      f'adapter file must use prefix-stripped keys: {invalid[:5]}')
  return dict(sorted(state.items()))


def load_adapter_into_head(
    head: torch.nn.Module,
    path: Path,
    *,
    manifest_path: Path,
    expected_identity: Mapping[str, object],
    expected_sha256: str,
    expected_manifest_sha256: str,
) -> None:
  """Strictly load a legacy schema-v4 adapter into a head-only target.

  Schema v5 binds an actual runtime backbone and authenticated expectations;
  a head-only API cannot establish either.  Call
  ``verify_contextual_forest_adapter`` for production artifacts.
  """
  validated_manifest = load_and_validate_adapter_manifest(
    manifest_path, path,
    expected_identity=expected_identity,
    expected_adapter_sha256=expected_sha256,
    expected_manifest_sha256=expected_manifest_sha256)
  if validated_manifest['schema_version'] != 4:
    raise ValueError(
      'head-only loading supports schema-v4 adapters only; use '
      'verify_contextual_forest_adapter for schema-v5 artifacts')
  state = load_adapter_state(path, expected_sha256=expected_sha256)
  validate_tensor_state_against_module(
    state, head, context='structured adapter')
  try:
    incompatible = head.load_state_dict(state, strict=True)
  except RuntimeError as error:
    raise ValueError(
      f'structured adapter strict-load mismatch: {error}') from error
  if incompatible.missing_keys or incompatible.unexpected_keys:
    raise ValueError(
      'adapter state mismatch: '
      f'missing={incompatible.missing_keys}, '
      f'unexpected={incompatible.unexpected_keys}')


def validate_adapter_manifest_payload(
    payload: object,
    *,
    adapter_filename: str,
    adapter_sha256: str,
    adapter_size_bytes: int,
    adapter_metadata: Mapping[str, str],
    adapter_state: Mapping[str, torch.Tensor],
    expected_identity: Mapping[str, object],
    expected_production_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
  """Structurally validate manifest claims against bytes and semantics.

  For schema v5 this function alone is not runtime authorization: callers must
  independently attest the actual backbone and expectations, then use the
  dedicated production verifier.
  """
  if not isinstance(payload, Mapping):
    raise TypeError('structured adapter manifest must be a JSON object')
  base_required = {
    'artifact_role', 'schema_version', 'format', 'adapter_file',
    'adapter_sha256', 'adapter_size_bytes', 'adapter_tensor_count',
    'adapter_parameter_count', 'adapter_tensor_bytes',
    'adapter_namespace_in_source', 'adapter_namespace_in_file',
    'structured_decoder_identity',
    'structured_decoder_identity_sha256', 'tensor_schema',
    'source_checkpoint_sha256', 'source_checkpoint_size_bytes',
    'source_checkpoint_global_step', 'source_state_dict_tensor_count',
    'omitted_frozen_backbone_tensor_count', 'ema_available', 'ema_used',
    'required_loader', 'required_loader_strict', 'released_backbone',
  }
  schema_version = payload.get('schema_version')
  if type(schema_version) is not int or schema_version not in {4, 5}:
    raise ValueError('unsupported structured adapter manifest schema version')
  required = set(base_required)
  if schema_version == 5:
    required.add('production_provenance')
  if set(payload) != required:
    raise ValueError(
      'structured adapter manifest schema mismatch: '
      f'missing={sorted(required - set(payload))}, '
      f'unknown={sorted(set(payload) - required)}')
  result = dict(payload)
  if (not isinstance(result['artifact_role'], str)
      or result['artifact_role'] != 'contextual_forest_structured_adapter'
      or type(result['schema_version']) is not int
      or result['schema_version'] != schema_version
      or not isinstance(result['format'], str)
      or result['format'] != 'safetensors'):
    raise ValueError('unsupported structured adapter manifest identity')
  if result['adapter_file'] != adapter_filename:
    raise ValueError(
      'structured adapter filename differs from its manifest: '
      f'expected {result["adapter_file"]!r}, found {adapter_filename!r}')
  manifest_adapter_sha256 = _lower_sha256(
    result['adapter_sha256'], context='manifest adapter SHA256')
  actual_adapter_sha256 = _lower_sha256(
    adapter_sha256, context='actual adapter SHA256')
  if manifest_adapter_sha256 != actual_adapter_sha256:
    raise ValueError(
      'structured adapter SHA256 differs from its manifest: '
      f'expected {result["adapter_sha256"]}, found {adapter_sha256}')
  manifest_adapter_size = _positive_int(
    result['adapter_size_bytes'], context='adapter_size_bytes')
  actual_adapter_size = _positive_int(
    adapter_size_bytes, context='actual adapter_size_bytes')
  if manifest_adapter_size != actual_adapter_size:
    raise ValueError('structured adapter size differs from its manifest')
  if not isinstance(adapter_state, Mapping) or not adapter_state:
    raise ValueError('structured adapter contains no tensors')
  invalid_values = [
    key for key, value in adapter_state.items() if not torch.is_tensor(value)]
  if invalid_values:
    raise TypeError(
      f'structured adapter contains non-tensors: {invalid_values[:5]}')
  invalid_names = [
    key for key in adapter_state
    if (not isinstance(key, str) or not key
        or key.startswith((BACKBONE_PREFIX, ADAPTER_PREFIX)))]
  if invalid_names:
    raise ValueError(
      'structured adapter must use non-empty prefix-stripped head keys: '
      f'{invalid_names[:5]}')
  tensor_schema = {
    key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
    for key, value in sorted(adapter_state.items())}
  tensor_count = len(adapter_state)
  parameter_count = sum(value.numel() for value in adapter_state.values())
  tensor_bytes = sum(
    value.numel() * value.element_size()
    for value in adapter_state.values())
  derived_adapter_fields = {
    'adapter_tensor_count': tensor_count,
    'adapter_parameter_count': parameter_count,
    'adapter_tensor_bytes': tensor_bytes,
    'adapter_namespace_in_source': f'{ADAPTER_PREFIX}*',
    'adapter_namespace_in_file': 'prefix-stripped',
    'tensor_schema': tensor_schema,
  }
  for field, expected_value in derived_adapter_fields.items():
    if _canonical_json(result[field]) != _canonical_json(expected_value):
      raise ValueError(
        f'structured adapter manifest {field} differs from adapter bytes')
  identity = result['structured_decoder_identity']
  if not isinstance(identity, Mapping) or set(identity) != \
      STRUCTURED_IDENTITY_FIELDS:
    raise ValueError('structured adapter identity schema mismatch')
  identity = dict(identity)
  identity_sha256 = canonical_sha256(identity)
  if result['structured_decoder_identity_sha256'] != identity_sha256:
    raise ValueError('structured adapter identity digest mismatch')
  if not isinstance(expected_identity, Mapping):
    raise ValueError('runtime structured adapter identity must be a mapping')
  expected = dict(expected_identity)
  if set(expected) != STRUCTURED_IDENTITY_FIELDS:
    raise ValueError('runtime structured adapter identity schema mismatch')
  expected_identity_sha256 = canonical_sha256(expected)
  if (identity_sha256 != expected_identity_sha256
      or _canonical_json(identity) != _canonical_json(expected)):
    differing = {
      field: {'adapter': identity.get(field), 'runtime': expected.get(field)}
      for field in sorted(STRUCTURED_IDENTITY_FIELDS)
      if identity.get(field) != expected.get(field)}
    raise ValueError(
      f'structured adapter identity differs from runtime config: {differing}')
  expected_metadata = {
    'artifact_role': 'contextual_forest_structured_head',
    'source_namespace': ADAPTER_PREFIX,
    'file_namespace': 'prefix-stripped',
    'control_identity': str(identity['control_identity']),
    'topology_mode': str(identity['topology_mode']),
    'factor_mode': str(identity['factor_mode']),
    'candidate_k': str(identity['candidate_top_k']),
    'independent_mode': json.dumps(identity['independent_mode']),
    'topology_weight': json.dumps(identity['topology_weight']),
    'structured_decoder_identity_sha256': identity_sha256,
    'source_checkpoint_sha256': str(result['source_checkpoint_sha256']),
    'source_checkpoint_size_bytes': str(
      result['source_checkpoint_size_bytes']),
    'source_checkpoint_global_step': str(
      result['source_checkpoint_global_step']),
    'source_state_dict_tensor_count': str(
      result['source_state_dict_tensor_count']),
    'omitted_frozen_backbone_tensor_count': str(
      result['omitted_frozen_backbone_tensor_count']),
  }
  production_provenance = None
  if schema_version == 5:
    production_provenance = validate_production_provenance_payload(
      result['production_provenance'])
    provenance_metadata = {
      'production_expectations_file_sha256': (
        production_provenance['production_expectations_file_sha256']),
      'production_expectations_identity_sha256': (
        production_provenance['production_expectations_identity_sha256']),
      'backbone_wrapper_sha256': (
        production_provenance['backbone_wrapper_sha256']),
      'backbone_wrapper_metadata_sha256': (
        production_provenance['backbone_wrapper_metadata_sha256']),
      'backbone_tensor_schema_sha256': (
        production_provenance['backbone_tensor_schema_sha256']),
      'backbone_tensor_content_sha256': (
        production_provenance['backbone_tensor_content_sha256']),
    }
    expected_metadata.update(provenance_metadata)
    if expected_production_provenance is not None:
      expected_provenance = validate_production_provenance_payload(
        expected_production_provenance)
      if _canonical_json(production_provenance) != _canonical_json(
          expected_provenance):
        raise ValueError(
          'adapter production provenance differs from authenticated '
          'expectations')
  elif expected_production_provenance is not None:
    raise ValueError(
      'schema-v4 adapter cannot satisfy authenticated production provenance')
  if dict(adapter_metadata) != expected_metadata:
    raise ValueError(
      'safetensors metadata differs from the validated adapter identity')
  source_sha = _lower_sha256(
    result['source_checkpoint_sha256'],
    context='source_checkpoint_sha256')
  source_size = _positive_int(
    result['source_checkpoint_size_bytes'],
    context='source_checkpoint_size_bytes')
  source_step = _nonnegative_int(
    result['source_checkpoint_global_step'],
    context='source_checkpoint_global_step')
  source_tensor_count = _positive_int(
    result['source_state_dict_tensor_count'],
    context='source_state_dict_tensor_count')
  omitted_count = _positive_int(
    result['omitted_frozen_backbone_tensor_count'],
    context='omitted_frozen_backbone_tensor_count')
  if source_tensor_count != omitted_count + tensor_count:
    raise ValueError(
      'source state tensor count does not equal omitted backbone plus adapter')
  if omitted_count != RELEASE_TENSOR_COUNT:
    raise ValueError(
      'omitted frozen-backbone tensor count differs from the pinned release')
  expected_released_backbone = (
    RELEASED_BACKBONE_IDENTITY if production_provenance is None
    else production_provenance['released_backbone'])
  if (not isinstance(result['released_backbone'], Mapping)
      or _canonical_json(dict(result['released_backbone']))
      != _canonical_json(expected_released_backbone)):
    raise ValueError(
      'structured adapter released-backbone identity is not the pinned release')
  header_source_fields = {
    'source_checkpoint_sha256': source_sha,
    'source_checkpoint_size_bytes': source_size,
    'source_checkpoint_global_step': source_step,
    'source_state_dict_tensor_count': source_tensor_count,
    'omitted_frozen_backbone_tensor_count': omitted_count,
  }
  for field, expected_value in header_source_fields.items():
    observed_value = (
      adapter_metadata[field]
      if field == 'source_checkpoint_sha256'
      else _metadata_integer(
        adapter_metadata, field,
        positive=(field != 'source_checkpoint_global_step')))
    if observed_value != expected_value:
      raise ValueError(
        f'safetensors metadata {field} differs from adapter manifest')
  expected_loader = (
    'manifest_validated_strict_structured_head_loader_v2'
    if schema_version == 4
    else 'manifest_validated_strict_structured_head_loader_v3')
  if result['required_loader'] != expected_loader:
    raise ValueError('structured adapter manifest names an invalid loader')
  if result['required_loader_strict'] is not True:
    raise ValueError('structured adapter manifest does not require strict load')
  if result['ema_available'] is not False or result['ema_used'] is not False:
    raise ValueError('structured adapter manifest has an invalid EMA policy')
  return result


def _load_and_structurally_validate_adapter_manifest(
    manifest_path: Path,
    adapter_path: Path,
    *,
    expected_identity: Mapping[str, object],
    expected_adapter_sha256: str,
    expected_manifest_sha256: str,
    expected_production_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
  """Low-level byte/schema validation, not schema-v5 runtime authorization.

  This private helper permits schema v5 only when the caller supplies exact
  production provenance.  It does not attest a runtime backbone itself; only
  ``verify_contextual_forest_adapter`` may call it for schema-v5 artifacts,
  after authenticating expectations and attesting the actual model backbone.
  """
  manifest_path = manifest_path.expanduser().resolve()
  adapter_path = adapter_path.expanduser().resolve()
  if not manifest_path.is_file():
    raise FileNotFoundError(manifest_path)
  if not adapter_path.is_file():
    raise FileNotFoundError(adapter_path)
  normalized_adapter_sha256 = (
    expected_adapter_sha256.lower()
    if isinstance(expected_adapter_sha256, str)
    else expected_adapter_sha256)
  normalized_manifest_sha256 = (
    expected_manifest_sha256.lower()
    if isinstance(expected_manifest_sha256, str)
    else expected_manifest_sha256)
  expected_adapter_sha256 = _lower_sha256(
    normalized_adapter_sha256, context='expected adapter SHA256')
  expected_manifest_sha256 = _lower_sha256(
    normalized_manifest_sha256, context='expected manifest SHA256')
  manifest_bytes = manifest_path.read_bytes()
  actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
  if actual_manifest_sha256 != expected_manifest_sha256:
    raise ValueError(
      f'structured adapter manifest SHA256 mismatch: expected '
      f'{expected_manifest_sha256}, found {actual_manifest_sha256}')
  try:
    payload = json.loads(manifest_bytes)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError('structured adapter manifest is not valid JSON') from error
  if (isinstance(payload, Mapping)
      and payload.get('schema_version') == 5
      and expected_production_provenance is None):
    raise ValueError(
      'schema-v5 validation requires authenticated expectations and actual '
      'backbone attestation through verify_contextual_forest_adapter')
  adapter_bytes = adapter_path.read_bytes()
  actual_sha256 = hashlib.sha256(adapter_bytes).hexdigest()
  if actual_sha256 != expected_adapter_sha256:
    raise ValueError(
      f'structured adapter SHA256 mismatch: expected '
      f'{expected_adapter_sha256}, found {actual_sha256}')
  adapter_state = load_safetensors(adapter_bytes)
  result = validate_adapter_manifest_payload(
    payload,
    adapter_filename=adapter_path.name,
    adapter_sha256=actual_sha256,
    adapter_size_bytes=len(adapter_bytes),
    adapter_metadata=safetensors_metadata_from_bytes(adapter_bytes),
    adapter_state=adapter_state,
    expected_identity=expected_identity,
    expected_production_provenance=expected_production_provenance)
  return result


def load_and_validate_adapter_manifest(
    manifest_path: Path,
    adapter_path: Path,
    *,
    expected_identity: Mapping[str, object],
    expected_adapter_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
  """Validate a legacy schema-v4 adapter; reject production schema v5.

  Head-only and analysis callers cannot attest the actual runtime backbone.
  Production schema-v5 callers must instead use
  ``verify_contextual_forest_adapter``.
  """
  return _load_and_structurally_validate_adapter_manifest(
    manifest_path,
    adapter_path,
    expected_identity=expected_identity,
    expected_adapter_sha256=expected_adapter_sha256,
    expected_manifest_sha256=expected_manifest_sha256)


def export_adapter(
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    control_identity: str,
    topology_mode: str,
    factor_mode: str,
    candidate_k: int,
    independent_mode: bool,
    topology_weight: float,
    expected_global_step: int | None = None,
    expected_backbone_tensors: int | None = RELEASE_TENSOR_COUNT,
    expected_structured_head: torch.nn.Module | None = None,
    expected_frozen_backbone: torch.nn.Module | None = None,
    expected_adapter_tensor_count: int | None = None,
    expected_adapter_parameter_count: int | None = None,
    expected_adapter_tensor_bytes: int | None = None,
    production_provenance: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
  checkpoint_path = checkpoint_path.resolve()
  output_path = output_path.resolve()
  manifest_path = manifest_path.resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
  if output_path == manifest_path:
    raise ValueError('adapter and manifest paths must differ')
  existing = [path for path in (output_path, manifest_path) if path.exists()]
  if existing and not overwrite:
    raise FileExistsError(
      f'{existing[0]} already exists; pass --force to replace outputs')

  structured_identity, structured_identity_sha256 = (
    _validated_adapter_identity(
      control_identity=control_identity,
      topology_mode=topology_mode,
      factor_mode=factor_mode,
      candidate_k=candidate_k,
      independent_mode=independent_mode,
      topology_weight=topology_weight))

  normalized_checkpoint_sha256 = (
    expected_checkpoint_sha256.lower()
    if isinstance(expected_checkpoint_sha256, str)
    else expected_checkpoint_sha256)
  expected_checkpoint_sha256 = _lower_sha256(
    normalized_checkpoint_sha256, context='expected checkpoint SHA256')
  if expected_backbone_tensors != RELEASE_TENSOR_COUNT:
    raise ValueError(
      'structured-adapter export requires the pinned released-backbone '
      f'tensor count {RELEASE_TENSOR_COUNT}')
  if expected_global_step is not None:
    _nonnegative_int(
      expected_global_step, context='expected checkpoint global_step')
  validated_production_provenance = (
    None if production_provenance is None
    else validate_production_provenance_payload(production_provenance))
  if validated_production_provenance is not None:
    if not isinstance(expected_frozen_backbone, torch.nn.Module):
      raise ValueError(
        'production provenance requires an authenticated runtime backbone')
    runtime_backbone_state = {
      key: value.detach().cpu().contiguous()
      for key, value in sorted(expected_frozen_backbone.state_dict().items())
    }
    runtime_backbone_inventory = {
      'backbone_tensor_count': len(runtime_backbone_state),
      'backbone_parameter_count': sum(
        value.numel() for value in runtime_backbone_state.values()),
      'backbone_tensor_bytes': sum(
        value.numel() * value.element_size()
        for value in runtime_backbone_state.values()),
      'backbone_tensor_schema_sha256': canonical_sha256(
        tensor_state_schema(runtime_backbone_state)),
      'backbone_tensor_content_sha256': tensor_state_content_sha256(
        runtime_backbone_state),
    }
    for field, observed in runtime_backbone_inventory.items():
      if validated_production_provenance[field] != observed:
        raise ValueError(
          f'production provenance {field} differs from runtime backbone')
  checkpoint_bytes = checkpoint_path.read_bytes()
  source_checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
  if source_checkpoint_sha256 != expected_checkpoint_sha256:
    raise ValueError(
      f'source checkpoint SHA256 mismatch: expected '
      f'{expected_checkpoint_sha256}, '
      f'found {source_checkpoint_sha256}')

  # Lightning checkpoints use pickle.  Byte identity is therefore checked
  # against a caller-supplied trusted digest before any deserialization.
  checkpoint = torch.load(
    io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
  if not isinstance(checkpoint, Mapping):
    raise ValueError('checkpoint payload is not a mapping')
  if checkpoint.get('ema') is not None:
    raise ValueError('checkpoint unexpectedly contains EMA state')
  global_step = _nonnegative_int(
    checkpoint.get('global_step'), context='source checkpoint global_step')
  if expected_global_step is not None and global_step != expected_global_step:
    raise ValueError(
      f'global-step mismatch: expected {expected_global_step}, '
      f'found {global_step}')
  structured_identity, structured_identity_sha256 = (
    _validate_checkpoint_adapter_identity(
      checkpoint, expected_identity=structured_identity))
  adapter, backbone_count = _validated_state_dict(
    checkpoint,
    expected_backbone_tensors=expected_backbone_tensors)
  if expected_structured_head is not None:
    validate_tensor_state_against_module(
      adapter,
      expected_structured_head,
      context='checkpoint structured_head')
  validate_adapter_inventory(
    adapter,
    expected_tensor_count=expected_adapter_tensor_count,
    expected_parameter_count=expected_adapter_parameter_count,
    expected_tensor_bytes=expected_adapter_tensor_bytes)
  if expected_frozen_backbone is not None:
    backbone = {
      key.removeprefix(BACKBONE_PREFIX): value
      for key, value in checkpoint['state_dict'].items()
      if key.startswith(BACKBONE_PREFIX)
    }
    validate_tensor_state_against_module(
      backbone,
      expected_frozen_backbone,
      context='checkpoint frozen backbone',
      compare_values=True)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_adapter = output_path.with_name(
    f'.{output_path.name}.tmp-{os.getpid()}')
  temporary_manifest = manifest_path.with_name(
    f'.{manifest_path.name}.tmp-{os.getpid()}')
  try:
    adapter_metadata = {
      'artifact_role': 'contextual_forest_structured_head',
      'source_namespace': ADAPTER_PREFIX,
      'file_namespace': 'prefix-stripped',
      'control_identity': str(
        structured_identity['control_identity']),
      'topology_mode': str(structured_identity['topology_mode']),
      'factor_mode': str(structured_identity['factor_mode']),
      'candidate_k': str(structured_identity['candidate_top_k']),
      'independent_mode': json.dumps(
        structured_identity['independent_mode']),
      'topology_weight': json.dumps(
        structured_identity['topology_weight']),
      'structured_decoder_identity_sha256': structured_identity_sha256,
      'source_checkpoint_sha256': source_checkpoint_sha256,
      'source_checkpoint_size_bytes': str(len(checkpoint_bytes)),
      'source_checkpoint_global_step': str(global_step),
      'source_state_dict_tensor_count': str(
        backbone_count + len(adapter)),
      'omitted_frozen_backbone_tensor_count': str(backbone_count),
    }
    if validated_production_provenance is not None:
      adapter_metadata.update({
        'production_expectations_file_sha256': (
          validated_production_provenance[
            'production_expectations_file_sha256']),
        'production_expectations_identity_sha256': (
          validated_production_provenance[
            'production_expectations_identity_sha256']),
        'backbone_wrapper_sha256': (
          validated_production_provenance['backbone_wrapper_sha256']),
        'backbone_wrapper_metadata_sha256': (
          validated_production_provenance[
            'backbone_wrapper_metadata_sha256']),
        'backbone_tensor_schema_sha256': (
          validated_production_provenance[
            'backbone_tensor_schema_sha256']),
        'backbone_tensor_content_sha256': (
          validated_production_provenance[
            'backbone_tensor_content_sha256']),
      })
    save_file(
      adapter,
      str(temporary_adapter),
      metadata=adapter_metadata)
    temporary_adapter.write_bytes(
      canonicalize_safetensors_bytes(temporary_adapter.read_bytes()))
    adapter_sha256 = sha256_file(temporary_adapter)
    tensor_schema = {
      key: {
        'shape': list(value.shape),
        'dtype': str(value.dtype),
      }
      for key, value in adapter.items()
    }
    schema_version = 4 if validated_production_provenance is None else 5
    released_backbone_identity = (
      RELEASED_BACKBONE_IDENTITY
      if validated_production_provenance is None
      else validated_production_provenance['released_backbone'])
    manifest = {
      'artifact_role': 'contextual_forest_structured_adapter',
      'schema_version': schema_version,
      'format': 'safetensors',
      'adapter_file': output_path.name,
      'adapter_sha256': adapter_sha256,
      'adapter_size_bytes': temporary_adapter.stat().st_size,
      'adapter_tensor_count': len(adapter),
      'adapter_parameter_count': sum(
        value.numel() for value in adapter.values()),
      'adapter_tensor_bytes': sum(
        value.numel() * value.element_size() for value in adapter.values()),
      'adapter_namespace_in_source': f'{ADAPTER_PREFIX}*',
      'adapter_namespace_in_file': 'prefix-stripped',
      'structured_decoder_identity': structured_identity,
      'structured_decoder_identity_sha256': structured_identity_sha256,
      'tensor_schema': tensor_schema,
      'source_checkpoint_sha256': source_checkpoint_sha256,
      'source_checkpoint_size_bytes': len(checkpoint_bytes),
      'source_checkpoint_global_step': global_step,
      'source_state_dict_tensor_count': backbone_count + len(adapter),
      'omitted_frozen_backbone_tensor_count': backbone_count,
      'ema_available': False,
      'ema_used': False,
      'required_loader': (
        'manifest_validated_strict_structured_head_loader_v2'
        if schema_version == 4 else
        'manifest_validated_strict_structured_head_loader_v3'),
      'required_loader_strict': True,
      'released_backbone': dict(released_backbone_identity),
    }
    if validated_production_provenance is not None:
      manifest['production_provenance'] = dict(
        validated_production_provenance)
    temporary_manifest.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(temporary_adapter, output_path)
    os.replace(temporary_manifest, manifest_path)
  finally:
    if temporary_adapter.exists():
      temporary_adapter.unlink()
    if temporary_manifest.exists():
      temporary_manifest.unlink()
  return manifest


def _args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Export the learned structured head from a frozen checkpoint.')
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--expected-checkpoint-sha256', required=True)
  parser.add_argument(
    '--control-identity', choices=sorted(CONTROL_MODES), required=True)
  parser.add_argument(
    '--topology-mode', choices=('dynamic', 'fixed'), required=True)
  parser.add_argument(
    '--factor-mode', choices=('dynamic', 'fixed'), required=True)
  parser.add_argument('--candidate-k', type=int, required=True)
  parser.add_argument(
    '--independent-mode', choices=('true', 'false'), required=True)
  parser.add_argument('--topology-weight', type=float, required=True)
  parser.add_argument('--expected-global-step', type=int)
  parser.add_argument(
    '--expected-backbone-tensors', type=int, default=RELEASE_TENSOR_COUNT)
  parser.add_argument('--force', action='store_true')
  return parser.parse_args(argv)


def main() -> int:
  args = _args()
  manifest = export_adapter(
    args.checkpoint,
    args.output,
    args.manifest,
    expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    control_identity=args.control_identity,
    topology_mode=args.topology_mode,
    factor_mode=args.factor_mode,
    candidate_k=args.candidate_k,
    independent_mode=args.independent_mode == 'true',
    topology_weight=args.topology_weight,
    expected_global_step=args.expected_global_step,
    expected_backbone_tensors=args.expected_backbone_tensors,
    overwrite=args.force)
  print(json.dumps(manifest, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
