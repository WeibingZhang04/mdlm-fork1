#!/usr/bin/env python3
"""Bind verified CCF generation to the released Tensor-Train matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


NFE_BUDGETS = (8, 16, 32)
EVALUATOR_REVISION = '32b71b12589c2f8d625668d2335a01cac3249519'
SEQUENCE_POLICY = (
  'retokenize_decoded_text_score_through_first_nonleading_eos_v1')


def _load(path: Path) -> dict:
  with path.open() as handle:
    value = json.load(handle)
  if not isinstance(value, dict):
    raise TypeError(f'{path} must contain a JSON object')
  return value


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _finite(value: object, context: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f'{context} must be numeric')
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f'{context} must be finite')
  return result


def _validate_ccf(payload: dict) -> dict[int, dict]:
  if (payload.get('artifact') != 'verified_generation_shard_union'
      or payload.get('coverage', {}).get('num_shards') != 32
      or payload.get('coverage', {}).get('global_num_paired_draws') != 256
      or payload.get('coverage', {}).get('verified_output_records') != 768):
    raise ValueError('CCF union coverage is incomplete')
  identity = payload.get('identity', {})
  if (identity.get('global_num_samples') != 256
      or identity.get('sequence_length') != 1024
      or identity.get('batch_size') != 1
      or identity.get('base_seed') != 260703
      or identity.get('sampling_modes') != ['structured_joint']
      or identity.get('nfe_budgets') != list(NFE_BUDGETS)):
    raise ValueError('CCF matched identity differs from the frozen protocol')
  scorer = identity.get('reference_lm', {})
  runtime = scorer.get('runtime_identity', {})
  if (scorer.get('revision') != EVALUATOR_REVISION
      or scorer.get('sequence_policy') != SEQUENCE_POLICY
      or runtime.get('max_length') != 1024
      or runtime.get('batch_size') != 8):
    raise ValueError('CCF evaluator identity differs')
  hardware = payload.get('timing_policy', {}).get('hardware_identity', {})
  if hardware.get('gpu') != 'NVIDIA L4':
    raise ValueError('CCF timing hardware is not NVIDIA L4')
  groups = {}
  for group in payload.get('groups', []):
    if group.get('sampling_mode') != 'structured_joint':
      raise ValueError('CCF union contains an unexpected sampling mode')
    budget = group.get('requested_nfe_budget')
    if budget in groups or budget not in NFE_BUDGETS:
      raise ValueError('CCF union has an invalid NFE grid')
    if group.get('unresolved_mask_tokens') != 0:
      raise ValueError('CCF union contains unresolved masks')
    groups[budget] = group
  if set(groups) != set(NFE_BUDGETS):
    raise ValueError('CCF union is missing an NFE group')
  return groups


def _validate_tt(payload: dict) -> dict[tuple[str, int], dict]:
  if (payload.get('protocol_id') != 'tensor-train-owt-feasibility-v1'
      or payload.get('num_samples_per_job') != 256
      or payload.get('paired_generation_seed') != 260703):
    raise ValueError('Tensor-Train analysis identity differs')
  evaluator = payload.get('evaluator_runtime_identity', {})
  if (evaluator.get('model_revision') != EVALUATOR_REVISION
      or evaluator.get('sequence_policy') != SEQUENCE_POLICY
      or evaluator.get('max_length') != 1024
      or evaluator.get('batch_size') != 8):
    raise ValueError('Tensor-Train evaluator identity differs')
  gpu = payload.get('resource_host_identity', {}).get('gpu', {})
  if gpu.get('name') != 'NVIDIA L4':
    raise ValueError('Tensor-Train timing hardware is not NVIDIA L4')
  cells = {}
  for cell in payload.get('cells', []):
    key = (cell.get('arm'), cell.get('nfe_steps'))
    if key in cells:
      raise ValueError('Tensor-Train analysis repeats a cell')
    cells[key] = cell
  expected = {
    (arm, nfe) for arm in ('marginal', 'tensor_train_rank4')
    for nfe in NFE_BUDGETS
  }
  if set(cells) != expected:
    raise ValueError('Tensor-Train matrix is incomplete')
  return cells


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--ccf-union', type=Path, required=True)
  parser.add_argument('--tensor-train-analysis', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  ccf_path = args.ccf_union.expanduser().resolve()
  tt_path = args.tensor_train_analysis.expanduser().resolve()
  ccf = _validate_ccf(_load(ccf_path))
  tt = _validate_tt(_load(tt_path))
  rows = []
  for nfe in NFE_BUDGETS:
    ccf_group = ccf[nfe]
    marginal = tt[('marginal', nfe)]
    rank4 = tt[('tensor_train_rank4', nfe)]
    ccf_nll = _finite(
      ccf_group['reference_lm']['mean_nll_nats'], f'CCF NFE {nfe} NLL')
    ccf_speed = _finite(
      ccf_group['timing']['aggregate_active_tokens_per_second'],
      f'CCF NFE {nfe} token throughput') / 1024.0
    marginal_nll = _finite(marginal['mean_nll_nats'], 'marginal NLL')
    rank4_nll = _finite(rank4['mean_nll_nats'], 'rank-4 NLL')
    rows.append({
      'nfe': nfe,
      'ccf_mean_nll_nats': ccf_nll,
      'tensor_train_rank4_mean_nll_nats': rank4_nll,
      'released_marginal_mean_nll_nats': marginal_nll,
      'ccf_minus_tensor_train_rank4_nll_nats': ccf_nll - rank4_nll,
      'ccf_minus_released_marginal_nll_nats': ccf_nll - marginal_nll,
      'ccf_samples_per_second': ccf_speed,
      'tensor_train_rank4_samples_per_second': _finite(
        rank4['samples_per_second'], 'rank-4 throughput'),
      'released_marginal_samples_per_second': _finite(
        marginal['samples_per_second'], 'marginal throughput'),
    })
  result = {
    'schema_version': 1,
    'artifact': 'contextual_forest_tensor_train_matched_comparison',
    'protocol_id': 'contextual-forest-tensor-train-matched-v1',
    'claim_scope': (
      'Descriptive alignment on sample count, sequence length, NFE budgets, '
      'generation seed, evaluator model/revision/policy, batch size, and GPU '
      'model. Model-native reverse schedules and checkpoint training histories '
      'are not paired; differences are not causal estimates.'),
    'source': {
      'ccf_union_path': str(ccf_path),
      'ccf_union_sha256': _sha256(ccf_path),
      'tensor_train_analysis_path': str(tt_path),
      'tensor_train_analysis_sha256': _sha256(tt_path),
    },
    'alignment': {
      'num_samples': 256,
      'sequence_length': 1024,
      'nfe_budgets': list(NFE_BUDGETS),
      'generation_seed': 260703,
      'generation_batch_size': 1,
      'evaluator_revision': EVALUATOR_REVISION,
      'evaluator_sequence_policy': SEQUENCE_POLICY,
      'evaluator_batch_size': 8,
      'hardware_model': 'NVIDIA L4',
      'model_native_position_schedule_paired': False,
      'checkpoint_training_history_paired': False,
    },
    'rows': rows,
  }
  output = args.output.expanduser().resolve()
  if output.exists():
    raise FileExistsError(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
  os.replace(temporary, output)
  print(json.dumps({'output': str(output), 'sha256': _sha256(output)},
                   sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
