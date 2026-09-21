#!/usr/bin/env python3
"""Run one resumable partition of the frozen CCF--Tensor-Train comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


RUNNER_ROOT = Path('/mnt/contextual-forest/mdlm-causal-a574aca')
PYTHON = Path('/mnt/contextual-forest/venv/bin/python')
OUTPUT_ROOT = Path(
  '/mnt/contextual-forest/experiments/'
  'contextual-forest-tensor-train-matched-v1/ccf-origin-v1')
BACKBONE = Path('/mnt/contextual-forest/checkpoints/mdlm-owt-backbone.pt')
BACKBONE_SHA256 = (
  '1b7c724d0228b1a2c825185c96642ffd706bd828b237f84512f0e1c7b5765573')
ADAPTER = Path(
  '/mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/'
  'runs/export--dynamic_dynamic--s001--k128/attempts/attempt-0001/'
  'adapter.safetensors')
ADAPTER_SHA256 = (
  'c40d9bf0854059a85da88d4f0062ac280e4d7f621338845ae43357a7dfa69c4a')
ADAPTER_MANIFEST = ADAPTER.with_name('adapter-manifest.json')
ADAPTER_MANIFEST_SHA256 = (
  'd527f40926eda894e1ee5c9c1a91317353941463acbdfe059a279884fdbd2da9')
ADAPTER_ORIGIN_EVIDENCE = Path(
  '/mnt/contextual-forest/experiments/'
  'contextual-forest-tensor-train-matched-v1/adapter-pair-origin.json')
ADAPTER_ORIGIN_EVIDENCE_SHA256 = (
  '0c5ef69ee6bd14d1d40bbc8c8e6ea9d412eb7e65a25b42b530cd45e3d456a558')
SOURCE_SHA = 'a574aca873d6de66a3b847c13efd1c7bc4efb66b'
NUM_SHARDS = 32


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _verify_inputs() -> None:
  for path, expected in (
      (BACKBONE, BACKBONE_SHA256),
      (ADAPTER, ADAPTER_SHA256),
      (ADAPTER_MANIFEST, ADAPTER_MANIFEST_SHA256),
      (ADAPTER_ORIGIN_EVIDENCE, ADAPTER_ORIGIN_EVIDENCE_SHA256)):
    if not path.is_file() or _sha256(path) != expected:
      raise RuntimeError(f'artifact identity mismatch: {path}')
  result = subprocess.run(
    ['git', '-C', str(RUNNER_ROOT), 'status', '--porcelain=v1'],
    check=True, capture_output=True, text=True)
  if result.stdout:
    raise RuntimeError('runner checkout is dirty')
  head = subprocess.run(
    ['git', '-C', str(RUNNER_ROOT), 'rev-parse', 'HEAD'],
    check=True, capture_output=True, text=True).stdout.strip()
  if head != SOURCE_SHA:
    raise RuntimeError(f'runner checkout differs: {head}')


def _validate_shard(path: Path, shard_index: int) -> None:
  sys.path.insert(0, str(RUNNER_ROOT))
  from evaluation.generation_shard_aggregation import load_generation_shard

  loaded = load_generation_shard(path)
  manifest = loaded['manifest']
  pairing = manifest['pairing']
  matrix = manifest['matrix']
  expected = {
    'global_num_samples': 256,
    'num_shards': NUM_SHARDS,
    'shard_index': shard_index,
    'sequence_length': 1024,
    'batch_size': 1,
    'base_seed': 260703,
  }
  for key, value in expected.items():
    if pairing.get(key) != value:
      raise RuntimeError(f'{path}: pairing {key} differs')
  if matrix.get('sampling_modes') != ['structured_joint']:
    raise RuntimeError(f'{path}: sampling modes differ')
  if matrix.get('nfe_budgets') != [8, 16, 32]:
    raise RuntimeError(f'{path}: NFE budgets differ')
  if (manifest['artifacts']['backbone_checkpoint']['sha256']
      != BACKBONE_SHA256):
    raise RuntimeError(f'{path}: backbone hash differs')
  if manifest['artifacts']['structured_adapter']['sha256'] != ADAPTER_SHA256:
    raise RuntimeError(f'{path}: adapter hash differs')
  origin = manifest.get('adapter_origin_evidence')
  if (not isinstance(origin, dict)
      or origin.get('evidence_file', {}).get('sha256')
      != ADAPTER_ORIGIN_EVIDENCE_SHA256
      or origin.get('arm') != 'dynamic_dynamic'):
    raise RuntimeError(f'{path}: adapter origin evidence differs')


def _command(shard_index: int, output: Path) -> list[str]:
  return [
    str(PYTHON), 'scripts/run_generation_pilot.py',
    '--backbone-checkpoint', str(BACKBONE),
    '--backbone-sha256', BACKBONE_SHA256,
    '--adapter', str(ADAPTER),
    '--adapter-sha256', ADAPTER_SHA256,
    '--adapter-manifest', str(ADAPTER_MANIFEST),
    '--adapter-manifest-sha256', ADAPTER_MANIFEST_SHA256,
    '--adapter-origin-evidence', str(ADAPTER_ORIGIN_EVIDENCE),
    '--adapter-origin-evidence-sha256', ADAPTER_ORIGIN_EVIDENCE_SHA256,
    '--adapter-origin-arm', 'dynamic_dynamic',
    '--output-dir', str(output),
    '--num-samples', '256',
    '--num-shards', str(NUM_SHARDS),
    '--shard-index', str(shard_index),
    '--sequence-length', '1024',
    '--batch-size', '1',
    '--base-seed', '260703',
    '--modes', 'structured_joint',
    '--nfe-budgets', '8', '16', '32',
    '--device', 'cuda',
    '--model-config', 'contextual-forest-small',
    '--data-config', 'openwebtext-streaming',
    '--reference-lm', 'gpt2-large',
    '--reference-lm-revision',
    '32b71b12589c2f8d625668d2335a01cac3249519',
    '--reference-lm-device', 'cuda',
    '--reference-lm-batch-size', '8',
    '--reference-lm-max-length', '1024',
    '--reference-lm-dtype', 'float32',
    '--override', 'model.structured_decoder.top_k=128',
    '--override', 'trainer.devices=1',
    '--override', 'loader.num_workers=8',
    '--override', f'checkpointing.save_dir={OUTPUT_ROOT}',
    '--override', 'model.structured_decoder.topology_mode=dynamic',
    '--override', 'model.structured_decoder.factor_mode=dynamic',
    '--override', 'model.structured_decoder.training.topology_weight=0.1',
  ]


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--partition-index', type=int, required=True)
  parser.add_argument('--num-partitions', type=int, required=True)
  args = parser.parse_args()
  if not 0 <= args.partition_index < args.num_partitions:
    raise ValueError('invalid partition')
  _verify_inputs()
  OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
  completed = []
  for shard_index in range(args.partition_index, NUM_SHARDS,
                           args.num_partitions):
    output = OUTPUT_ROOT / f'shard-{shard_index:02d}'
    if output.exists():
      _validate_shard(output, shard_index)
      status = 'skipped'
    else:
      subprocess.run(_command(shard_index, output), cwd=RUNNER_ROOT, check=True)
      _validate_shard(output, shard_index)
      status = 'completed'
    completed.append({'shard_index': shard_index, 'status': status})
    print(json.dumps(completed[-1], sort_keys=True), flush=True)
  print(json.dumps({
    'event': 'ccf_tensor_train_matched_partition_complete',
    'partition_index': args.partition_index,
    'num_partitions': args.num_partitions,
    'shards': completed,
  }, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
