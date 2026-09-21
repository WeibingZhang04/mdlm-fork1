#!/usr/bin/env python3
"""Prefetch and attest the immutable Hugging Face inputs for Tensor-Train."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.tensor_train_baseline import (  # noqa: E402
  cached_model_identities,
  load_protocol,
)
from scripts.compile_tensor_train_feasibility import DEFAULT_PROTOCOL  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Download pinned baseline inputs, then hash the offline cache.')
  parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
  parser.add_argument('--cache-root', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args(argv)


def _write_exclusive(path: Path, content: str) -> None:
  path = path.expanduser().resolve(strict=False)
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  with os.fdopen(descriptor, 'w') as handle:
    handle.write(content)
    handle.flush()
    os.fsync(handle.fileno())


def main(argv=None) -> int:
  args = _parse_args(argv)
  output_path = args.output.expanduser().resolve(strict=False)
  if output_path.exists():
    raise FileExistsError(f'refusing to overwrite cache identity {output_path}')
  protocol = load_protocol(args.protocol)
  cache_root = args.cache_root.expanduser().resolve(strict=False)
  if cache_root == Path(cache_root.anchor):
    raise ValueError('cache root must not be a filesystem root')
  cache_root.mkdir(parents=True, exist_ok=True)
  from huggingface_hub import snapshot_download

  specifications = {
    'backbone': protocol['model_inputs']['backbone'],
    'tokenizer': protocol['model_inputs']['tokenizer'],
    'evaluator': {
      'repository': protocol['evaluator']['model_name_or_path'],
      'revision': protocol['evaluator']['revision'],
    },
  }
  for specification in specifications.values():
    snapshot_download(
      repo_id=specification['repository'],
      revision=specification['revision'],
      cache_dir=str(cache_root / 'hub'))
  identity = cached_model_identities(cache_root, protocol)
  _write_exclusive(
    output_path, json.dumps(identity, indent=2, sort_keys=True) + '\n')
  print(json.dumps({
    'event': 'tensor_train_cache_prepared',
    'cache_root': str(cache_root),
    'identity_sha256': identity['identity_sha256'],
    'output': str(output_path),
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
