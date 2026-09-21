#!/usr/bin/env python3
"""Materialize document-preserving training windows from the pinned OWT cache.

This creates a new processed cache. It does not change the legacy concatenated
training protocol or the existing held-out cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cache-dir', type=Path, required=True)
  parser.add_argument('--provenance-dir', type=Path, required=True)
  parser.add_argument('--source-stop', type=int, default=32768)
  parser.add_argument('--num-proc', type=int, default=4)
  args = parser.parse_args(argv)
  if not 0 < args.source_stop <= 7913769 or args.num_proc < 1:
    raise ValueError('training window must stay inside the pinned training partition')
  from transformers import AutoTokenizer
  from dataloader import get_dataset
  tokenizer_name = 'openai-community/gpt2'
  tokenizer_revision = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
  tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name, revision=tokenizer_revision, local_files_only=True)
  dataset = get_dataset(
    dataset_name='openwebtext-staged-train-documents', tokenizer=tokenizer,
    wrap=True, mode='train', cache_dir=str(args.cache_dir), block_size=1024,
    num_proc=args.num_proc, streaming=False,
    revision='79d93d786212f7344586290adb811d4ae6a1762c',
    dataset_name_or_path='Skylion007/openwebtext', dataset_config_name='plain_text',
    source_split='train', source_window=[0, args.source_stop],
    expected_source_num_rows=8013769, text_field='text',
    document_boundary_mode='source_document', trust_remote_code=False,
    require_pinned_provenance=True, tokenizer_name_or_path=tokenizer_name,
    tokenizer_revision=tokenizer_revision, provenance_dir=str(args.provenance_dir),
    provenance_role='staged_train')
  required = {'source_document_index', 'source_document_sha256', 'source_chunk_index'}
  if not required <= set(dataset.column_names):
    raise AssertionError('processed training cache lost document identity')
  print(json.dumps({'event': 'document_training_cache_ready', 'windows': len(dataset),
                    'columns': dataset.column_names, 'cache_files': dataset.cache_files,
                    'source_window': [0, args.source_stop], 'block_size': 1024}), flush=True)


if __name__ == '__main__':
  main()
