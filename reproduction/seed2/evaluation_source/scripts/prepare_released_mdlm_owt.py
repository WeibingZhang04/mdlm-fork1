#!/usr/bin/env python3
"""Download, verify, and wrap the immutable released MDLM-OWT backbone.

The Hugging Face artifact is safetensors, while the structured-backbone loader
accepts the repository's Lightning-style ``{"state_dict": ...}`` envelope.
This script performs only that packaging conversion.  It never imports or
executes remote model code, and it fails closed if the released bytes or state
schema differ from the audited artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch


RELEASE_REPOSITORY = 'kuleshov-group/mdlm-owt'
RELEASE_REVISION = 'd0958fa851335ece6c15260ce0025f030673c0fb'
RELEASE_FILENAME = 'model.safetensors'
RELEASE_SHA256 = (
  '47149e73f7552f39ea9776dbe74d925d25237bcf2ed2e2ec03cdff9d51c82aa4')
RELEASE_SIZE_BYTES = 678_522_728
RELEASE_TENSOR_COUNT = 131


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    while True:
      chunk = handle.read(chunk_size)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def verify_release_file(
    path: Path,
    expected_sha256: str = RELEASE_SHA256,
    expected_size_bytes: int | None = RELEASE_SIZE_BYTES) -> dict[str, object]:
  if not path.is_file():
    raise FileNotFoundError(path)
  size_bytes = path.stat().st_size
  if expected_size_bytes is not None and size_bytes != expected_size_bytes:
    raise ValueError(
      f'released file size mismatch: expected {expected_size_bytes}, '
      f'found {size_bytes} at {path}')
  actual_sha256 = sha256_file(path)
  if actual_sha256 != expected_sha256:
    raise ValueError(
      f'released file SHA256 mismatch: expected {expected_sha256}, '
      f'found {actual_sha256} at {path}')
  return {'sha256': actual_sha256, 'size_bytes': size_bytes}


def validate_backbone_state(
    tensors: Mapping[str, torch.Tensor],
    expected_tensor_count: int | None = RELEASE_TENSOR_COUNT,
) -> dict[str, torch.Tensor]:
  if not tensors:
    raise ValueError('released safetensors file contains no tensors')
  if (expected_tensor_count is not None
      and len(tensors) != expected_tensor_count):
    raise ValueError(
      f'released tensor-count mismatch: expected {expected_tensor_count}, '
      f'found {len(tensors)}')
  invalid_values = [
    key for key, value in tensors.items() if not torch.is_tensor(value)]
  if invalid_values:
    raise TypeError(
      f'released state contains non-tensors: {invalid_values[:5]}')
  invalid_keys = [key for key in tensors if not key.startswith('backbone.')]
  if invalid_keys:
    raise ValueError(
      'released state must contain only backbone.* keys; found '
      f'{invalid_keys[:5]}')
  return {key: tensors[key].detach().cpu() for key in sorted(tensors)}


def convert_release(
    source: Path,
    output: Path,
    *,
    expected_sha256: str = RELEASE_SHA256,
    expected_size_bytes: int | None = RELEASE_SIZE_BYTES,
    expected_tensor_count: int | None = RELEASE_TENSOR_COUNT,
    overwrite: bool = False,
) -> dict[str, object]:
  source = source.resolve()
  output = output.resolve()
  if source == output:
    raise ValueError('source and output paths must differ')
  if output.exists() and not overwrite:
    raise FileExistsError(
      f'{output} already exists; pass --force to replace it')
  verification = verify_release_file(
    source,
    expected_sha256=expected_sha256,
    expected_size_bytes=expected_size_bytes)

  # Import only after the byte-level identity check.  Safetensors parses data
  # without the arbitrary-code execution surface of pickle checkpoints.
  from safetensors.torch import load_file
  state_dict = validate_backbone_state(
    load_file(str(source), device='cpu'),
    expected_tensor_count=expected_tensor_count)
  metadata = {
    'format_version': 1,
    'artifact_role': 'released_raw_mdlm_owt_backbone',
    'source_repository': RELEASE_REPOSITORY,
    'source_revision': RELEASE_REVISION,
    'source_filename': RELEASE_FILENAME,
    'source_sha256': verification['sha256'],
    'source_size_bytes': verification['size_bytes'],
    'weight_namespace': 'backbone.*',
    'tensor_count': len(state_dict),
    'ema_available': False,
    'ema_used': False,
    'required_loader_setting': 'use_ema_backbone=false',
  }
  wrapper = {'state_dict': state_dict, 'metadata': metadata}

  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
  try:
    torch.save(wrapper, temporary)
    os.replace(temporary, output)
  finally:
    if temporary.exists():
      temporary.unlink()
  return {'output': str(output), **metadata}


def download_release(cache_dir: Path | None = None) -> Path:
  from huggingface_hub import hf_hub_download
  kwargs = {
    'repo_id': RELEASE_REPOSITORY,
    'revision': RELEASE_REVISION,
    'filename': RELEASE_FILENAME,
    'repo_type': 'model',
  }
  if cache_dir is not None:
    kwargs['cache_dir'] = str(cache_dir.resolve())
  return Path(hf_hub_download(**kwargs))


def _args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Prepare the pinned raw MDLM-OWT backbone for this repo.')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument(
    '--source', type=Path,
    help=('optional already-downloaded model.safetensors; its exact pinned '
          'SHA256 and size are still required'))
  parser.add_argument('--cache-dir', type=Path)
  parser.add_argument(
    '--force', action='store_true',
    help='atomically replace an existing output wrapper')
  return parser.parse_args(argv)


def main() -> int:
  args = _args()
  source = args.source if args.source is not None else download_release(
    args.cache_dir)
  result = convert_release(source, args.output, overwrite=args.force)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
