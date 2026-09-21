#!/usr/bin/env python3
"""Audit raw generation samples for decode/retokenize integrity failures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.decode_integrity import (  # noqa: E402
  audit_decode_integrity,
  tokenizer_identity,
)


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=(
    'Fail-closed U+FFFD and raw-token decode/retokenize audit for one or '
    'more generation samples.jsonl files.'))
  parser.add_argument(
    '--input', action='append', type=Path, required=True,
    help=(
      'Exact JSONL file or directory recursively containing samples.jsonl; '
      'repeat for multiple inputs.'))
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--tokenizer-name-or-path', required=True)
  parser.add_argument(
    '--tokenizer-revision', required=True,
    help='Pinned tokenizer commit/revision used for generation.')
  parser.add_argument(
    '--local-files-only', action='store_true',
    help='Refuse tokenizer downloads and use only the local cache.')
  return parser.parse_args(argv)


def _atomic_write(path: Path, content: str) -> None:
  path = path.expanduser().resolve()
  if path.exists():
    raise FileExistsError(f'refusing to overwrite audit artifact {path}')
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def main(argv=None) -> int:
  args = _parse_args(argv)
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name_or_path,
    revision=args.tokenizer_revision,
    use_fast=True,
    trust_remote_code=False,
    local_files_only=args.local_files_only)
  result = audit_decode_integrity(
    args.input,
    tokenizer=tokenizer,
    tokenizer_identity=tokenizer_identity(
      tokenizer,
      name_or_path=args.tokenizer_name_or_path,
      requested_revision=args.tokenizer_revision))
  output = args.output.expanduser().resolve()
  _atomic_write(output, json.dumps(
    result, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
  print(json.dumps({
    'event': 'generation_decode_integrity_audited',
    'output': str(output),
    'audit_sha256': result['audit_sha256'],
    'num_inputs': len(result['inputs']),
    'num_records': result['aggregate']['num_records'],
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
