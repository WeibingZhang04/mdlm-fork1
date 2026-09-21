#!/usr/bin/env python3
"""Exactly replay an outcome-independently selected generation batch.

The source shards are fully verified before selection.  Selection depends only
on immutable pair identities and declared batch boundaries, never on generated
text or quality metrics.  Every requested sampling mode is then replayed from
the complete original batch, and the captured final raw token IDs must equal
the persisted source records exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from evaluation.generation_harness import (  # noqa: E402
  PairedSampleSpec,
  PromptSpec,
  replay_sampling_group_trajectory,
  stable_sha256,
)
from evaluation.generation_shard_aggregation import (  # noqa: E402
  load_generation_shard,
  sha256_file,
)


ARTIFACT = 'exact_generation_trajectory_replay'
SCHEMA_VERSION = 1
SELECTION_POLICY = 'hash_min_full_source_batch_v1'


def _parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=(
    'Replay one complete source batch and capture exact generation states.'))
  parser.add_argument(
    '--source-shard', action='append', type=Path, required=True,
    help='Verified shard directory or manifest.json; repeat across the pool.')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument(
    '--modes', nargs='+', required=True,
    choices=('factorized', 'factorized_confidence_gated',
             'structured_marginal', 'structured_joint'))
  parser.add_argument('--nfe-budget', type=int, required=True)
  parser.add_argument(
    '--snapshot-call-indices', nargs='+', type=int,
    default=[0, 16, 32, 48, 63, 64])
  parser.add_argument('--device', default='cuda')
  return parser.parse_args(argv)


def _atomic_write(path: Path, content: str) -> None:
  path = path.expanduser().resolve()
  if path.exists():
    raise FileExistsError(f'refusing to overwrite replay artifact {path}')
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def _git_provenance() -> dict[str, Any]:
  def command(*args: str) -> str:
    return subprocess.check_output(
      ['git', *args], cwd=REPO_ROOT, text=True,
      stderr=subprocess.DEVNULL)

  sha = command('rev-parse', 'HEAD').strip()
  status = command('status', '--porcelain=v1').splitlines()
  if status:
    raise RuntimeError(
      'trajectory replay requires a clean Git checkout; found '
      f'{status[:5]}')
  return {'git_sha': sha, 'dirty': False, 'status_porcelain': []}


def _pair_identity(record: Mapping[str, Any]) -> dict[str, Any]:
  return {
    'sample_index': record['sample_index'],
    'pair_key': record['pair_key'],
    'pair_seed': record['pair_seed'],
    'prompt_id': record['prompt_id'],
  }


def _batch_selection_sha256(records: Sequence[Mapping[str, Any]]) -> str:
  return stable_sha256({
    'policy': SELECTION_POLICY,
    'ordered_pair_identities': [_pair_identity(record) for record in records],
  })


def select_outcome_independent_batch(
    loaded_shards: Sequence[Mapping[str, Any]],
    *,
    modes: Sequence[str],
    nfe_budget: int,
) -> dict[str, Any]:
  """Choose the hash-min complete source batch across verified shards."""
  if not loaded_shards:
    raise ValueError('at least one loaded source shard is required')
  if not modes or len(set(modes)) != len(modes):
    raise ValueError('modes must be a non-empty unique sequence')
  candidates = []
  for loaded in loaded_shards:
    manifest = loaded['manifest']
    batch_size = manifest['pairing']['batch_size']
    source_mode = modes[0]
    key = (source_mode, nfe_budget)
    if key not in loaded['groups']:
      raise ValueError(
        f'{loaded["manifest_path"]} lacks source group {key}')
    source_records = loaded['groups'][key]
    for offset in range(0, len(source_records), batch_size):
      batch = source_records[offset:offset + batch_size]
      if len(batch) != batch_size:
        continue
      selection_sha = _batch_selection_sha256(batch)
      candidates.append({
        'selection_sha256': selection_sha,
        'loaded': loaded,
        'batch_number_within_shard': offset // batch_size,
        'sample_indices': [record['sample_index'] for record in batch],
      })
  if not candidates:
    raise ValueError('source pool contains no complete declared-size batch')
  candidates.sort(key=lambda item: (
    item['selection_sha256'],
    str(item['loaded']['manifest_path'])))
  selected = candidates[0]
  loaded = selected['loaded']
  sample_indices = selected['sample_indices']
  records_by_mode = {}
  for mode in modes:
    key = (mode, nfe_budget)
    if key not in loaded['groups']:
      raise ValueError(f'{loaded["manifest_path"]} lacks replay group {key}')
    by_index = {
      record['sample_index']: record for record in loaded['groups'][key]
    }
    try:
      records = [by_index[index] for index in sample_indices]
    except KeyError as error:
      raise ValueError(
        f'replay group {key} lacks selected sample index {error.args[0]}') \
        from error
    if [record['sample_index'] for record in records] != sample_indices:
      raise AssertionError('selected replay batch order changed')
    records_by_mode[mode] = records
  return {
    'selection_policy': SELECTION_POLICY,
    'selection_sha256': selected['selection_sha256'],
    'num_eligible_full_batches': len(candidates),
    'selected_manifest_path': loaded['manifest_path'],
    'selected_manifest_sha256': loaded['manifest_sha256'],
    'batch_number_within_shard': selected['batch_number_within_shard'],
    'sample_indices': sample_indices,
    'loaded': loaded,
    'records_by_mode': records_by_mode,
  }


def paired_samples_from_records(
    records: Sequence[Mapping[str, Any]],
) -> list[PairedSampleSpec]:
  """Reconstruct the harness input identities without decoding any text."""
  samples = []
  for record in records:
    prompt = PromptSpec(
      prompt_id=record['prompt_id'],
      initial_token_ids=tuple(record['initial_token_ids']),
      active_mask=tuple(record['active_mask']),
      reference_token_ids=(
        tuple(record['reference_token_ids'])
        if record['reference_token_ids'] is not None else None),
      metadata=dict(record['prompt_metadata']))
    samples.append(PairedSampleSpec(
      sample_index=record['sample_index'],
      pair_key=record['pair_key'],
      pair_seed=record['pair_seed'],
      prompt=prompt))
  return samples


def _verify_model_artifacts(manifest: Mapping[str, Any]) -> None:
  backbone = manifest['artifacts']['backbone_checkpoint']
  adapter = manifest['artifacts']['structured_adapter']
  checks = (
    (Path(backbone['path']), backbone['sha256'], 'backbone checkpoint'),
    (Path(adapter['path']), adapter['sha256'], 'structured adapter'),
    (Path(adapter['manifest_path']), adapter['manifest_sha256'],
     'structured adapter manifest'),
  )
  for path, expected, label in checks:
    if not path.is_file():
      raise FileNotFoundError(f'{label} is missing: {path}')
    actual = sha256_file(path)
    if actual != expected:
      raise ValueError(
        f'{label} SHA256 mismatch: expected {expected}, found {actual}')


def _source_record_projection(record: Mapping[str, Any]) -> dict[str, Any]:
  fields = (
    'sample_index', 'pair_key', 'pair_seed', 'prompt_id', 'prompt_metadata',
    'sampling_mode', 'requested_nfe_budget', 'measured_nfe', 'batch_seed',
    'initial_token_ids', 'active_mask', 'reference_token_ids',
    'sample_token_ids', 'sample_active_token_ids', 'metrics', 'reference_lm',
  )
  result = {field: record[field] for field in fields if field in record}
  result['source_record_sha256'] = stable_sha256(record)
  return result


def resolve_device(value: str) -> torch.device:
  """Canonicalize an unindexed CUDA alias to the active concrete device."""
  device = torch.device(value)
  if device.type != 'cuda':
    return device
  if not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable')
  if device.index is None:
    return torch.device('cuda', torch.cuda.current_device())
  return device


def _load_model(loaded: Mapping[str, Any], device: torch.device):
  import dataloader
  import diffusion
  from omegaconf import OmegaConf

  _verify_model_artifacts(loaded['manifest'])
  config = OmegaConf.load(loaded['config_path'])
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion(config, tokenizer=tokenizer)
  model = model.to(device=device).eval()
  model.backbone.eval()
  if model.structured_head is not None:
    model.structured_head.eval()
  model.noise.eval()
  if model.ema is not None:
    raise AssertionError('trajectory replay requires EMA disabled')
  return model


def main(argv=None) -> int:
  args = _parse_args(argv)
  if args.nfe_budget < 2:
    raise ValueError('--nfe-budget must be at least 2')
  if len(set(args.modes)) != len(args.modes):
    raise ValueError('--modes contains duplicates')
  repository = _git_provenance()
  loaded_shards = [
    load_generation_shard(path) for path in args.source_shard
  ]
  selected = select_outcome_independent_batch(
    loaded_shards, modes=args.modes, nfe_budget=args.nfe_budget)
  loaded = selected['loaded']
  first_records = selected['records_by_mode'][args.modes[0]]
  samples = paired_samples_from_records(first_records)
  device = resolve_device(args.device)
  model = _load_model(loaded, device)

  trajectories = {}
  source_records = {}
  for mode in args.modes:
    expected_records = selected['records_by_mode'][mode]
    trajectories[mode] = replay_sampling_group_trajectory(
      model,
      samples,
      sampling_mode=mode,
      nfe_budget=args.nfe_budget,
      expected_records=expected_records,
      snapshot_call_indices=args.snapshot_call_indices,
      device=device)
    source_records[mode] = [
      _source_record_projection(record) for record in expected_records
    ]

  eligible_sources = sorted(
    ({
      'manifest_path': str(item['manifest_path']),
      'manifest_sha256': item['manifest_sha256'],
      'samples_sha256': item['manifest']['outputs'][
        'samples_jsonl']['sha256'],
      'shard_index': item['manifest']['pairing']['shard_index'],
    } for item in loaded_shards),
    key=lambda item: item['shard_index'])
  result = {
    'schema_version': SCHEMA_VERSION,
    'artifact': ARTIFACT,
    'repository': repository,
    'selection': {
      'policy': selected['selection_policy'],
      'outcome_independent': True,
      'eligibility': 'complete batches with declared batch size only',
      'selection_sha256': selected['selection_sha256'],
      'num_eligible_full_batches': selected['num_eligible_full_batches'],
      'selected_manifest_path': str(selected['selected_manifest_path']),
      'selected_manifest_sha256': selected['selected_manifest_sha256'],
      'batch_number_within_shard': selected['batch_number_within_shard'],
      'sample_indices': selected['sample_indices'],
      'eligible_sources': eligible_sources,
    },
    'source': {
      'repository': loaded['manifest']['repository'],
      'runtime': loaded['manifest']['host'],
      'artifacts': loaded['manifest']['artifacts'],
      'resolved_config': {
        'path': str(loaded['config_path']),
        'sha256': loaded['manifest']['outputs']['resolved_config']['sha256'],
      },
    },
    'nfe_budget': args.nfe_budget,
    'snapshot_call_indices': sorted(set(args.snapshot_call_indices)),
    'modes': list(args.modes),
    'source_records': source_records,
    'trajectories': trajectories,
  }
  result['artifact_sha256'] = stable_sha256(result)
  output = args.output.expanduser().resolve()
  _atomic_write(
    output,
    json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
  print(json.dumps({
    'event': 'exact_generation_trajectory_replayed',
    'output': str(output),
    'artifact_sha256': result['artifact_sha256'],
    'selection_sha256': selected['selection_sha256'],
    'sample_indices': selected['sample_indices'],
    'modes': args.modes,
    'nfe_budget': args.nfe_budget,
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
