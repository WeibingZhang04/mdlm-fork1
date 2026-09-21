#!/usr/bin/env python3
"""Strictly verify and load an anonymous contextual-forest adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.export_contextual_forest_adapter import (  # noqa: E402
  AuthenticatedProductionExpectations,
  PRODUCTION_ADAPTER_PARAMETER_COUNT,
  PRODUCTION_ADAPTER_KEYS,
  PRODUCTION_ADAPTER_TENSOR_BYTES,
  PRODUCTION_ADAPTER_TENSOR_COUNT,
  attest_runtime_backbone,
  build_production_model,
  load_production_expectations,
  production_provenance_from_expectations,
)
from scripts.export_structured_adapter import (  # noqa: E402
  RELEASED_BACKBONE_IDENTITY,
  _load_and_structurally_validate_adapter_manifest,
  canonical_sha256,
  load_adapter_state,
  structured_decoder_identity_from_config,
  validate_adapter_inventory,
  validate_tensor_state_against_module,
)


def _canonical_json(payload: object) -> str:
  try:
    return json.dumps(
      payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
  except (TypeError, ValueError) as error:
    raise ValueError('payload contains non-canonical JSON data') from error


def verify_contextual_forest_adapter(
    adapter_path: Path,
    manifest_path: Path,
    *,
    expected_adapter_sha256: str,
    expected_manifest_sha256: str,
    model: torch.nn.Module,
    expected_adapter_tensor_count: int = PRODUCTION_ADAPTER_TENSOR_COUNT,
    expected_adapter_parameter_count: int = (
      PRODUCTION_ADAPTER_PARAMETER_COUNT),
    expected_adapter_tensor_bytes: int = PRODUCTION_ADAPTER_TENSOR_BYTES,
    expectations: AuthenticatedProductionExpectations | None = None,
    development_mode: bool = False,
) -> dict[str, Any]:
  """Authenticate, schema-check, and strict-load a structured-head export."""
  head = getattr(model, 'structured_head', None)
  structured_config = getattr(model, 'structured_config', None)
  if not isinstance(head, torch.nn.Module):
    raise ValueError('model.structured_head must be an initialized module')
  if not isinstance(structured_config, Mapping):
    raise ValueError('model.structured_config must be a mapping')
  if expectations is None and not development_mode:
    raise ValueError(
      'authenticated production expectations are required; only explicit '
      'programmatic development_mode may verify without them')
  if expectations is not None and development_mode:
    raise ValueError(
      'development_mode cannot be combined with production expectations')
  runtime_identity, runtime_identity_sha256 = (
    structured_decoder_identity_from_config(structured_config))
  if runtime_identity['head_semantics']['fixed_edge_path'] is not None:
    raise ValueError(
      'anonymous adapters require inline fixed_edges or no fixed topology; '
      'fixed_edge_path would disclose a local path')

  expected_provenance = None
  if expectations is not None:
    if not isinstance(expectations, AuthenticatedProductionExpectations):
      raise TypeError(
        'production verification requires expectations loaded from an '
        'authenticated file')
    expected_provenance = attest_runtime_backbone(
      getattr(model, 'backbone', None), expectations)
    canonical_expected = production_provenance_from_expectations(expectations)
    if _canonical_json(expected_provenance) != _canonical_json(
        canonical_expected):
      raise AssertionError('runtime backbone attestation drifted')
  manifest = _load_and_structurally_validate_adapter_manifest(
    manifest_path,
    adapter_path,
    expected_identity=runtime_identity,
    expected_adapter_sha256=expected_adapter_sha256,
    expected_manifest_sha256=expected_manifest_sha256,
    expected_production_provenance=expected_provenance)
  if expectations is not None and manifest['schema_version'] != 5:
    raise ValueError('production verification requires a schema-v5 adapter')
  if manifest['structured_decoder_identity_sha256'] != \
      runtime_identity_sha256:
    raise AssertionError('validated runtime identity digest drifted')
  if (_canonical_json(manifest['released_backbone'])
      != _canonical_json(RELEASED_BACKBONE_IDENTITY)):
    raise ValueError('adapter backbone identity differs from pinned release')

  state = load_adapter_state(
    adapter_path, expected_sha256=expected_adapter_sha256)
  production_inventory = (
    expected_adapter_tensor_count == PRODUCTION_ADAPTER_TENSOR_COUNT
    and expected_adapter_parameter_count == PRODUCTION_ADAPTER_PARAMETER_COUNT
    and expected_adapter_tensor_bytes == PRODUCTION_ADAPTER_TENSOR_BYTES)
  if production_inventory:
    if set(state) != PRODUCTION_ADAPTER_KEYS:
      raise ValueError(
        'production structured adapter key mismatch: '
        f'missing={sorted(PRODUCTION_ADAPTER_KEYS - set(state))}, '
        f'unexpected={sorted(set(state) - PRODUCTION_ADAPTER_KEYS)}')
    non_fp32 = [
      key for key, value in state.items() if value.dtype != torch.float32]
    if non_fp32:
      raise ValueError(
        f'production structured adapter must be FP32: {non_fp32[:5]}')
  schema = validate_tensor_state_against_module(
    state, head, context='structured adapter')
  inventory = validate_adapter_inventory(
    state,
    expected_tensor_count=expected_adapter_tensor_count,
    expected_parameter_count=expected_adapter_parameter_count,
    expected_tensor_bytes=expected_adapter_tensor_bytes)

  if expectations is not None:
    validated_expectations = expectations.payload
    expectation_checks = {
      'source_checkpoint_sha256': manifest['source_checkpoint_sha256'],
      'source_checkpoint_global_step': (
        manifest['source_checkpoint_global_step']),
      **inventory,
      'structured_decoder_identity_sha256': runtime_identity_sha256,
    }
    for field, observed in expectation_checks.items():
      if validated_expectations.get(field) != observed:
        raise ValueError(
          f'production expectations {field} mismatch: expected '
          f'{validated_expectations.get(field)!r}, found {observed!r}')
    if (_canonical_json(validated_expectations.get('released_backbone'))
        != _canonical_json(manifest['released_backbone'])):
      raise ValueError('production expectations backbone identity mismatch')

  try:
    incompatible = head.load_state_dict(state, strict=True)
  except RuntimeError as error:
    raise ValueError(
      f'structured adapter strict-load mismatch: {error}') from error
  if incompatible.missing_keys or incompatible.unexpected_keys:
    raise ValueError(
      'structured adapter strict load returned incompatible keys: '
      f'missing={incompatible.missing_keys}, '
      f'unexpected={incompatible.unexpected_keys}')
  loaded = head.state_dict()
  for key in sorted(state):
    if (loaded[key].dtype != state[key].dtype
        or tuple(loaded[key].shape) != tuple(state[key].shape)
        or not torch.equal(
          loaded[key].detach().cpu(), state[key].detach().cpu())):
      raise AssertionError(
        f'structured adapter changed while strict-loading tensor {key}')

  return {
    'artifact_role': 'verified_contextual_forest_structured_adapter',
    'schema_version': 1,
    'adapter_sha256': manifest['adapter_sha256'],
    'manifest_sha256': expected_manifest_sha256,
    'source_checkpoint_sha256': manifest['source_checkpoint_sha256'],
    'source_checkpoint_global_step': (
      manifest['source_checkpoint_global_step']),
    **inventory,
    'tensor_schema': schema,
    'structured_decoder_identity_sha256': runtime_identity_sha256,
    'released_backbone_identity_sha256': canonical_sha256(
      manifest['released_backbone']),
    'production_expectations_file_sha256': (
      None if expectations is None else expectations.file_sha256),
    'production_expectations_identity_sha256': (
      None if expectations is None else expectations.identity_sha256),
    'backbone_tensor_content_sha256': (
      None if expected_provenance is None else
      expected_provenance['backbone_tensor_content_sha256']),
    'backbone_tensor_schema_sha256': (
      None if expected_provenance is None else
      expected_provenance['backbone_tensor_schema_sha256']),
    'backbone_wrapper_sha256': (
      None if expected_provenance is None else
      expected_provenance['backbone_wrapper_sha256']),
    'backbone_wrapper_metadata_sha256': (
      None if expected_provenance is None else
      expected_provenance['backbone_wrapper_metadata_sha256']),
    'strict_load': True,
  }


def _args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      'Authenticate and strict-load an anonymous contextual-forest adapter.'))
  parser.add_argument('--adapter', type=Path, required=True)
  parser.add_argument('--expected-adapter-sha256', required=True)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--expected-manifest-sha256', required=True)
  parser.add_argument('--model-config', default='contextual-forest-small')
  parser.add_argument('--data-config', default='train_openwebtext_pinned')
  parser.add_argument('--backbone-checkpoint', type=Path, required=True)
  parser.add_argument('--override', action='append', default=[])
  parser.add_argument('--expectations', type=Path, required=True)
  parser.add_argument('--expected-expectations-sha256', required=True)
  parser.add_argument('--output', type=Path)
  return parser.parse_args(argv)


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
  result = verify_contextual_forest_adapter(
    args.adapter,
    args.manifest,
    expected_adapter_sha256=args.expected_adapter_sha256,
    expected_manifest_sha256=args.expected_manifest_sha256,
    model=model,
    expectations=expectations)
  rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
  if args.output is not None:
    output = args.output.expanduser().resolve()
    if output.exists():
      raise FileExistsError(
        f'{output} already exists; verification reports are append-only')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
  print(rendered, end='')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
