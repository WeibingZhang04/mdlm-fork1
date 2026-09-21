#!/usr/bin/env python3
"""Explore frozen-checkpoint dependence, with lambda selected on dev only.

Input JSONL rows contain unique ``id``, ``split`` (``dev`` or ``test``), and
either ``input_ids`` or ``text``.  Optional ``document_id`` and ``dataset``
fields are retained.  Each row must supply at least --length clean tokens;
the first window is used.  This is a bounded exploratory diagnostic, not a
new training run or a claim of benchmark significance.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch


def canonical_bytes(value) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_authenticated(path: Path, expected: str) -> bytes:
  if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
    raise ValueError('expected SHA256 must be 64 lowercase hexadecimal digits')
  data = path.read_bytes()
  actual = hashlib.sha256(data).hexdigest()
  if actual != expected:
    raise ValueError(f'SHA256 mismatch for {path.name}: expected {expected}, got {actual}')
  return data


def identity_overrides(identity: dict, length: int) -> list[str]:
  """Restore the exact adapter's head and training semantics in Hydra."""
  if length < 2:
    raise ValueError('length must be at least two')
  if identity['head_semantics'].get('fixed_edge_path') is not None:
    raise ValueError('use an adapter with inline fixed edges, not a local fixed_edge_path')
  values = {
    'top_k': identity['candidate_top_k'],
    'topology_mode': identity['topology_mode'],
    'factor_mode': identity['factor_mode'],
    'independent_mode': identity['independent_mode'],
    **identity['head_semantics'],
    'training.topology_weight': identity['topology_weight'],
    **{f'training.{key}': value for key, value in identity['training_semantics'].items()},
  }
  fixed_edges = values.get('fixed_edges')
  if fixed_edges and max(max(edge) for edge in fixed_edges) >= length:
    raise ValueError('adapter fixed edges exceed requested length; increase --length')
  return [f'model.length={length}'] + [
    f'model.structured_decoder.{key}={json.dumps(value, separators=(",", ":"))}'
    for key, value in values.items()]


def load_examples(payload: bytes, tokenizer, length: int, max_per_split: int, mask_index: int) -> list[dict]:
  """Use explicitly assigned splits; reject overlap before any evaluation."""
  examples, ids, content_splits, document_splits = [], set(), {}, {}
  counts = {'dev': 0, 'test': 0}
  for line_number, line in enumerate(payload.decode().splitlines(), 1):
    if not line.strip():
      continue
    row = json.loads(line)
    row_id, split = row.get('id'), row.get('split')
    if not isinstance(row_id, str) or not row_id or row_id in ids:
      raise ValueError(f'line {line_number}: id must be a unique nonempty string')
    ids.add(row_id)
    if split not in counts:
      raise ValueError(f'{row_id}: split must be explicitly dev or test')
    if ('input_ids' in row) == ('text' in row):
      raise ValueError(f'{row_id}: provide exactly one of input_ids and text')
    tokens = row.get('input_ids')
    if tokens is None:
      if not isinstance(row['text'], str):
        raise ValueError(f'{row_id}: text must be a string')
      tokens = tokenizer.encode(row['text'], add_special_tokens=False)
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
      raise ValueError(f'{row_id}: input_ids must be a list of integers')
    if len(tokens) < length:
      raise ValueError(f'{row_id}: needs {length} tokens, received {len(tokens)}')
    tokens = tokens[:length]
    if any(token < 0 or token >= len(tokenizer) or token == mask_index for token in tokens):
      raise ValueError(f'{row_id}: input contains an invalid or absorbing-mask token')
    token_sha = hashlib.sha256(canonical_bytes(tokens)).hexdigest()
    previous = content_splits.setdefault(token_sha, split)
    if previous != split:
      raise ValueError(f'{row_id}: duplicate token window across dev/test')
    document_id = row.get('document_id')
    if document_id is not None:
      if not isinstance(document_id, str) or not document_id:
        raise ValueError(f'{row_id}: document_id must be a nonempty string')
      document_key = (row.get('dataset'), document_id)
      if document_splits.setdefault(document_key, split) != split:
        raise ValueError(f'{row_id}: document crosses dev/test')
    if counts[split] >= max_per_split:
      continue
    counts[split] += 1
    examples.append({
      'id': row_id, 'split': split, 'input_ids': tokens,
      'input_ids_sha256': token_sha, 'dataset': row.get('dataset'),
      'document_id': document_id,
    })
  if not all(counts.values()):
    raise ValueError('input must include at least one dev and one test example')
  return examples


def deterministic_mask(row_id: str, length: int, mask_rate: float, seed: int) -> torch.Tensor:
  """Nested Bernoulli masks, independent of file order and split label."""
  if not 0 < mask_rate < 1:
    raise ValueError('mask rate must lie strictly between zero and one')
  key = hashlib.sha256(canonical_bytes([seed, row_id])).digest()
  generator = torch.Generator().manual_seed(int.from_bytes(key[:8], 'big') % (2**63))
  return torch.rand(length, generator=generator) < mask_rate


def summarize_records(records: list[dict], lambda_grid: list[float]) -> dict:
  """Fit a single global lambda on token-weighted dev likelihood only."""
  def aggregate(rows):
    count = sum(row['active_token_count'] for row in rows)
    if count == 0:
      raise ValueError('a requested summary contains no active tokens')
    return {
      'observations': len(rows), 'examples': len({row['id'] for row in rows}),
      'active_tokens': count,
      'backbone_nll': -sum(row['backbone_log_probability'] for row in rows) / count,
      'marginal_nll': -sum(row['marginal_log_probability'] for row in rows) / count,
      'joint_nll': -sum(row['joint_log_probability'] for row in rows) / count,
      'singleton_gain_nats_per_token': sum(row['marginal_log_probability'] - row['backbone_log_probability'] for row in rows) / count,
      'dependence_gain_nats_per_token': sum(row['joint_log_probability'] - row['marginal_log_probability'] for row in rows) / count,
      'lambda_nll': {str(value): -sum(row['lambda_log_probabilities'][str(value)] for row in rows) / count for value in lambda_grid},
    }
  splits = {split: [row for row in records if row['split'] == split] for split in ['dev', 'test']}
  results = {split: aggregate(rows) for split, rows in splits.items()}
  # Lowest lambda wins exact ties; test records never participate in this choice.
  selected = min(lambda_grid, key=lambda value: (results['dev']['lambda_nll'][str(value)], value))
  for split, result in results.items():
    result['selected_lambda_nll'] = result['lambda_nll'][str(selected)]
    result['by_mask_rate'] = {
      str(rate): aggregate([row for row in splits[split] if row['mask_rate'] == rate])
      for rate in sorted({row['mask_rate'] for row in splits[split]})}
  return {'selected_lambda': selected, 'selection_split': 'dev',
          'selection_objective': 'total dev NLL / total active dev tokens; lowest lambda breaks ties',
          'splits': results}


def _load_model(args, manifest):
  from scripts.export_contextual_forest_adapter import build_production_model, load_production_expectations
  from scripts.export_structured_adapter import load_adapter_into_head, structured_decoder_identity_from_config
  from scripts.verify_contextual_forest_adapter import verify_contextual_forest_adapter

  expectations = load_production_expectations(args.expectations, expected_sha256=args.expectations_sha256)
  model = build_production_model(
    model_config='contextual-forest-small', data_config='train_openwebtext_pinned',
    backbone_checkpoint=args.checkpoint, expectations=expectations,
    overrides=identity_overrides(manifest['structured_decoder_identity'], args.length),
    runtime_mode='ppl_eval')
  identity, identity_sha = structured_decoder_identity_from_config(model.structured_config)
  if identity != manifest['structured_decoder_identity']:
    raise ValueError('runtime head semantics differ from adapter manifest')
  if manifest['schema_version'] == 4:
    load_adapter_into_head(
      model.structured_head, args.adapter, manifest_path=args.adapter_manifest,
      expected_identity=identity, expected_sha256=args.adapter_sha256,
      expected_manifest_sha256=args.adapter_manifest_sha256)
    verification = {'adapter_schema': 4, 'strict_load': True,
                    'backbone_authenticated': True, 'adapter_production_attested': False}
  elif manifest['schema_version'] == 5:
    verification = verify_contextual_forest_adapter(
      args.adapter, args.adapter_manifest, model=model,
      expected_adapter_sha256=args.adapter_sha256,
      expected_manifest_sha256=args.adapter_manifest_sha256,
      expected_adapter_tensor_count=manifest['adapter_tensor_count'],
      expected_adapter_parameter_count=manifest['adapter_parameter_count'],
      expected_adapter_tensor_bytes=manifest['adapter_tensor_bytes'], expectations=expectations)
  else:
    raise ValueError('only manifest schemas 4 and 5 are supported')
  model.requires_grad_(False)
  model.eval().to(args.device)
  if any(parameter.requires_grad for parameter in model.parameters()):
    raise AssertionError('evaluation model contains trainable parameters')
  return model, verification, identity_sha, expectations


@torch.no_grad()
def evaluate_example(model, example: dict, rate: float, grid: list[float], seed: int, device: str) -> dict:
  from evaluation.dependence_diagnostics import decompose_structured_log_probability
  from structured_objective import infer_structured_distribution, structured_token_log_probability

  clean = torch.tensor([example['input_ids']], dtype=torch.long, device=device)
  active = deterministic_mask(example['id'], clean.shape[1], rate, seed).to(device).unsqueeze(0)
  corrupted = torch.where(active, model.mask_index, clean)
  if model.config.noise.type != 'loglinear':
    raise ValueError('this diagnostic expects the released loglinear noise schedule')
  diffusion_time = rate / (1.0 - float(model.noise.eps))
  if diffusion_time > 1:
    raise ValueError('mask rate exceeds the noise schedule maximum')
  sigma, _ = model.noise(torch.tensor([diffusion_time], device=device))
  output, logits = model._structured_head_output(
    tokens=corrupted, conditioning=sigma[:, None], active_mask=active,
    force_no_grad_backbone=True)
  # The checkpoint runs in its saved precision.  Small forest calculations
  # use FP64 so subtracting likelihoods does not erase weak edge effects.
  output = dataclasses.replace(
    output, unary_log_potentials=output.unary_log_potentials.double(),
    pair_left_factors=output.pair_left_factors.double(),
    pair_right_factors=output.pair_right_factors.double())
  inference = infer_structured_distribution(output, active, backend='low_rank')
  scores = decompose_structured_log_probability(output, logits, clean, active, inference)
  direct_joint = structured_token_log_probability(output, logits, clean, active, inference)
  # The original scorer accumulates residual corrections in FP32; this
  # decomposition accumulates the same per-token values in FP64.
  torch.testing.assert_close(scores.joint_log_probability, direct_joint, atol=2e-4, rtol=2e-7)
  explicit = output.candidate_ids.eq(clean.unsqueeze(-1)).any(-1)
  edge_mask = output.edge_mask[0]
  selected_edges = output.edge_index[0, edge_mask]
  result = {
    key: value for key, value in example.items() if key != 'input_ids'}
  result.update({
    'mask_rate': rate, 'mask_seed': seed, 'sequence_length': clean.shape[1],
    'diffusion_time': diffusion_time, 'conditioning_sigma': float(sigma.item()),
    'active_positions': active[0].nonzero().flatten().tolist(),
    'active_token_count': int(active.sum()),
    'explicit_target_count': int((active & explicit).sum()),
    'corrupted_tokens_sha256': hashlib.sha256(canonical_bytes(corrupted[0].tolist())).hexdigest(),
    'selected_edges': selected_edges.tolist(),
    'edge_log_dependence': scores.edge_log_dependence[0, edge_mask].tolist(),
    'edge_target_is_residual': (~explicit[0][selected_edges]).tolist(),
    'node_log_probability': scores.node_log_probability[0].tolist(),
    'backbone_log_probability': float(scores.backbone_log_probability.item()),
    'marginal_log_probability': float(scores.marginal_log_probability.item()),
    'joint_log_probability': float(scores.joint_log_probability.item()),
    'direct_joint_score_difference': float((scores.joint_log_probability - direct_joint).item()),
    'lambda_log_probabilities': {str(value): float(scores.shrinkage_log_probability(value).item()) for value in grid},
  })
  canonical_bytes(result)  # Reject any non-finite result before writing it.
  return result


def _float_list(text: str) -> list[float]:
  values = sorted(set(float(value) for value in text.split(',')))
  if not values or not all(math.isfinite(value) for value in values):
    raise argparse.ArgumentTypeError('expected a comma-separated list of finite numbers')
  return values


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ['checkpoint', 'expectations', 'adapter', 'adapter-manifest', 'input-jsonl', 'output-jsonl', 'summary']:
    parser.add_argument(f'--{name}', type=Path, required=True)
  for name in ['expectations', 'adapter', 'adapter-manifest', 'input']:
    parser.add_argument(f'--{name}-sha256', required=True)
  parser.add_argument('--length', type=int, default=128)
  parser.add_argument('--max-per-split', type=int, default=32)
  parser.add_argument('--mask-rates', type=_float_list, default=[0.25, 0.5, 0.75, 0.9])
  parser.add_argument('--lambda-grid', type=_float_list, default=[0.0, 0.25, 0.5, 0.75, 1.0])
  parser.add_argument('--seed', type=int, default=20260907)
  parser.add_argument('--device', default='cuda')
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = _args(argv)
  if args.max_per_split < 1 or args.length < 2:
    raise ValueError('max-per-split must be positive and length at least two')
  if not all(0 < value < 1 for value in args.mask_rates):
    raise ValueError('mask rates must lie strictly between zero and one')
  if not all(0 <= value <= 1 for value in args.lambda_grid) or not {0.0, 1.0}.issubset(args.lambda_grid):
    raise ValueError('lambda grid must lie in [0,1] and include both endpoints')
  if args.output_jsonl.exists() or args.summary.exists():
    raise FileExistsError('choose new output paths; existing results are never overwritten')
  if args.output_jsonl.resolve() == args.summary.resolve():
    raise ValueError('output JSONL and summary must have different paths')
  manifest = json.loads(read_authenticated(args.adapter_manifest, args.adapter_manifest_sha256))
  if manifest.get('schema_version') not in {4, 5}:
    raise ValueError('only manifest schemas 4 and 5 are supported')
  input_payload = read_authenticated(args.input_jsonl, args.input_sha256)
  read_authenticated(args.adapter, args.adapter_sha256)
  torch.manual_seed(args.seed)
  model, verification, identity_sha, expectations = _load_model(args, manifest)
  examples = load_examples(input_payload, model.tokenizer, args.length, args.max_per_split, model.mask_index)
  try:
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    git_sha = 'unknown'
  commitments = {
    'checkpoint_sha256': expectations.payload['backbone_wrapper']['sha256'],
    'expectations_sha256': args.expectations_sha256,
    'adapter_sha256': args.adapter_sha256, 'adapter_manifest_sha256': args.adapter_manifest_sha256,
    'structured_identity_sha256': identity_sha, 'input_sha256': args.input_sha256,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'diagnostic_module_sha256': hashlib.sha256((REPO_ROOT / 'evaluation/dependence_diagnostics.py').read_bytes()).hexdigest(),
    'git_sha': git_sha,
  }
  metadata = {
    'artifact': 'staged_frozen_checkpoint_dependence', 'schema_version': 1,
    'scope': 'exploratory frozen-checkpoint diagnostic; no training, no significance claim',
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'commitments': commitments, 'adapter_verification': verification,
    'mask_rates': args.mask_rates, 'lambda_grid': args.lambda_grid, 'mask_seed': args.seed,
    'mask_policy': 'nested Bernoulli uniforms per example id; same masks for all lambda values',
    'length': args.length, 'max_per_split': args.max_per_split,
    'selected_ids': {split: [row['id'] for row in examples if row['split'] == split] for split in ['dev', 'test']},
    'head_semantics': manifest['structured_decoder_identity'],
    'all_parameters_frozen': True, 'forest_inference_dtype': 'float64',
    'device': args.device, 'torch_version': torch.__version__,
  }
  run_id = hashlib.sha256(canonical_bytes(metadata)).hexdigest()
  args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
  args.summary.parent.mkdir(parents=True, exist_ok=True)
  records, started = [], time.monotonic()
  with args.output_jsonl.open('x') as handle:
    for example in examples:
      for rate in args.mask_rates:
        record = evaluate_example(model, example, rate, args.lambda_grid, args.seed, args.device)
        record['run_id'] = run_id
        handle.write(canonical_bytes(record).decode() + '\n')
        handle.flush()
        records.append(record)
      print(json.dumps({'done': example['id'], 'split': example['split'], 'observations': len(records)}), flush=True)
  result = {**metadata, 'run_id': run_id, **summarize_records(records, args.lambda_grid),
            'elapsed_seconds': time.monotonic() - started,
            'records_sha256': hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest()}
  with args.summary.open('x') as handle:
    json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
    handle.write('\n')
  print(json.dumps({'selected_lambda': result['selected_lambda'], 'splits': result['splits']}), flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
