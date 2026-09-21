#!/usr/bin/env python3
"""Build deterministic text-infilling prompts from a pinned validation split.

This script loads only the pinned tokenizer and document-local dataset cache;
it never constructs or loads diffusion-model weights.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.infilling_prompts import (  # noqa: E402
  PROMPT_POLICY_ID,
  build_infilling_prompts,
  serialize_prompt_jsonl,
)
from evaluation.prompt_provenance import (  # noqa: E402
  PROMPT_ARTIFACT,
  PROMPT_MANIFEST_SCHEMA_VERSION,
)


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    temporary.write_bytes(payload)
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def _clean_git_provenance() -> dict[str, object]:
  """Require a reproducible builder checkout before reading any dataset."""
  try:
    git_sha = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True,
      stderr=subprocess.DEVNULL).strip()
    status = subprocess.check_output(
      ['git', 'status', '--porcelain=v1'], cwd=REPO_ROOT, text=True,
      stderr=subprocess.DEVNULL)
  except (OSError, subprocess.CalledProcessError) as error:
    raise RuntimeError(
      'prompt construction requires a Git checkout with a resolvable HEAD') \
      from error
  if (len(git_sha) != 40
      or any(character not in '0123456789abcdef' for character in git_sha)):
    raise RuntimeError('prompt builder Git SHA is not a full lowercase SHA-1')
  if status:
    raise RuntimeError(
      'prompt construction requires a clean Git checkout so the builder '
      'identity is reproducible')
  return {'git_sha': git_sha, 'clean': True}


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      'Create deterministic contiguous-span prompts from a pinned, '
      'document-local validation dataset without loading model weights.'))
  parser.add_argument('--data-config', required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--manifest', type=Path)
  parser.add_argument('--provenance-dir', type=Path)
  parser.add_argument('--cache-dir', type=Path)
  parser.add_argument('--sequence-length', type=int, default=256)
  parser.add_argument('--span-length', type=int, default=32)
  parser.add_argument('--num-prompts', type=int, default=256)
  parser.add_argument('--selection-seed', type=int, default=31001)
  parser.add_argument('--num-proc', type=int, default=1)
  return parser.parse_args(argv)


def _compose_config(data_config: str, sequence_length: int):
  import hydra
  from omegaconf import OmegaConf, open_dict
  import torch

  resolvers = {
    'cwd': os.getcwd,
    'device_count': torch.cuda.device_count,
    'eval': eval,
    'div_up': lambda x, y: (x + y - 1) // y,
  }
  for name, resolver in resolvers.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)
  with hydra.initialize_config_dir(
      config_dir=str(REPO_ROOT / 'configs'), version_base=None):
    config = hydra.compose(
      config_name='config',
      overrides=[f'data={data_config}', 'model=contextual-forest-small'])
  with open_dict(config):
    config.model.length = int(sequence_length)
  return config


def _value(config, name: str, default=None):
  if hasattr(config, 'get'):
    return config.get(name, default)
  return getattr(config, name, default)


def _disjoint_proof(data_config):
  import data_provenance

  if not bool(_value(
      data_config, 'require_disjoint_train_valid_windows', False)):
    return None
  equality_fields = (
    'dataset_name_or_path', 'dataset_config_name', 'source_split',
    'revision', 'expected_source_num_rows')
  mismatches = [
    field for field in equality_fields
    if _value(data_config, f'train_{field}')
    != _value(data_config, f'valid_{field}')
  ]
  if mismatches:
    raise ValueError(
      f'disjoint source-window proof fields differ: {mismatches}')
  return data_provenance.disjoint_window_proof(
    dataset_name_or_path=_value(
      data_config, 'train_dataset_name_or_path'),
    dataset_config_name=_value(
      data_config, 'train_dataset_config_name'),
    split=_value(data_config, 'train_source_split'),
    revision=_value(data_config, 'train_revision'),
    source_num_rows=int(_value(
      data_config, 'train_expected_source_num_rows')),
    train_window=list(_value(data_config, 'train_source_window')),
    heldout_window=list(_value(data_config, 'valid_source_window')))


def _load_pinned_validation_dataset(config, provenance_dir: Path, num_proc: int):
  import dataloader

  data = config.data
  if not bool(_value(data, 'require_pinned_provenance', False)):
    raise ValueError('data config must require pinned provenance')
  boundary_mode = str(_value(
    data, 'valid_document_boundary_mode', 'concatenate'))
  if boundary_mode not in {'source_document', 'wikitext_articles'}:
    raise ValueError(
      'validation data config must preserve source-document boundaries')
  tokenizer = dataloader.get_tokenizer(config)
  source_window = _value(data, 'valid_source_window', None)
  return dataloader.get_dataset(
    str(data.valid),
    tokenizer,
    wrap=bool(data.wrap),
    mode='validation',
    cache_dir=str(data.cache_dir),
    block_size=int(config.model.length),
    num_proc=num_proc,
    streaming=False,
    revision=str(_value(data, 'valid_revision')),
    dataset_name_or_path=str(_value(
      data, 'valid_dataset_name_or_path')),
    dataset_config_name=_value(data, 'valid_dataset_config_name', None),
    source_split=str(_value(data, 'valid_source_split')),
    source_window=(None if source_window is None else list(source_window)),
    expected_source_num_rows=int(_value(
      data, 'valid_expected_source_num_rows')),
    text_field=str(_value(data, 'valid_text_field')),
    document_boundary_mode=boundary_mode,
    trust_remote_code=bool(_value(
      data, 'valid_trust_remote_code', False)),
    require_pinned_provenance=True,
    tokenizer_name_or_path=str(data.tokenizer_name_or_path),
    tokenizer_revision=str(data.tokenizer_revision),
    provenance_dir=str(provenance_dir),
    provenance_role='valid',
    disjoint_window_proof=_disjoint_proof(data))


def main(argv=None) -> int:
  args = _parse_args(argv)
  if args.sequence_length < 3:
    raise ValueError('--sequence-length must be at least 3')
  if args.span_length <= 0:
    raise ValueError('--span-length must be positive')
  if args.num_prompts <= 0:
    raise ValueError('--num-prompts must be positive')
  if args.selection_seed < 0:
    raise ValueError('--selection-seed must be non-negative')
  if args.num_proc <= 0:
    raise ValueError('--num-proc must be positive')

  repository = _clean_git_provenance()

  output = args.output.expanduser().resolve()
  manifest_path = (
    args.manifest.expanduser().resolve()
    if args.manifest is not None
    else output.with_name(f'{output.name}.manifest.json'))
  provenance_dir = (
    args.provenance_dir.expanduser().resolve()
    if args.provenance_dir is not None
    else output.with_name(f'{output.name}.provenance'))
  if output == manifest_path:
    raise ValueError('prompt JSONL and manifest paths must differ')
  for path in (output, manifest_path):
    if path.exists():
      raise FileExistsError(f'refusing to overwrite {path}')
  if provenance_dir.exists() and any(provenance_dir.iterdir()):
    raise FileExistsError(
      f'provenance directory must be fresh and empty: {provenance_dir}')
  output.parent.mkdir(parents=True, exist_ok=True)
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  provenance_dir.mkdir(parents=True, exist_ok=True)

  config = _compose_config(args.data_config, args.sequence_length)
  if args.cache_dir is not None:
    from omegaconf import open_dict
    with open_dict(config):
      config.data.cache_dir = str(args.cache_dir.expanduser().resolve())
  dataset = _load_pinned_validation_dataset(
    config, provenance_dir, args.num_proc)
  prompts = build_infilling_prompts(
    iter(dataset),
    dataset_id=str(config.data.valid),
    span_length=args.span_length,
    selection_seed=args.selection_seed,
    num_prompts=args.num_prompts)
  payload = serialize_prompt_jsonl(prompts)

  provenance_files = sorted(provenance_dir.glob('valid-*.json'))
  if len(provenance_files) != 1:
    raise RuntimeError(
      f'expected exactly one validation provenance file, found '
      f'{len(provenance_files)}')
  _atomic_write(output, payload)
  data_config_path = (
    REPO_ROOT / 'configs' / 'data' / f'{args.data_config}.yaml')
  manifest = {
    'schema_version': PROMPT_MANIFEST_SCHEMA_VERSION,
    'artifact': PROMPT_ARTIFACT,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'command': sys.argv if argv is None else [sys.argv[0], *argv],
    'repository': repository,
    'data_config': {
      'name': args.data_config,
      'path': str(data_config_path.resolve()),
      'sha256': _sha256_file(data_config_path),
      'logical_validation_dataset': str(config.data.valid),
      'dataset_revision': str(config.data.valid_revision),
      'tokenizer_name_or_path': str(config.data.tokenizer_name_or_path),
      'tokenizer_revision': str(config.data.tokenizer_revision),
    },
    'runtime_provenance': {
      'path': str(provenance_files[0]),
      'sha256': _sha256_file(provenance_files[0]),
    },
    'policy': {
      'policy_id': PROMPT_POLICY_ID,
      'selection_seed': args.selection_seed,
      'span_length': args.span_length,
      'sequence_length': args.sequence_length,
      'record_selection': 'first_n_in_pinned_validation_order',
      'boundary_policy': 'never_mask_first_or_last_token',
    },
    'output': {
      'path': str(output),
      'sha256': hashlib.sha256(payload).hexdigest(),
      'size_bytes': len(payload),
      'num_prompts': len(prompts),
    },
    'model_weights_loaded': False,
  }
  _atomic_write(
    manifest_path,
    (json.dumps(
      manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
  print(json.dumps({
    'output': str(output),
    'manifest': str(manifest_path),
    'num_prompts': len(prompts),
    'sha256': hashlib.sha256(payload).hexdigest(),
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
