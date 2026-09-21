#!/usr/bin/env python3
"""Train/evaluate the learned frozen-adapter Gate-1 ablation matrix."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path
import platform
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from synthetic.distributions import ContextSwitchingMatching  # noqa: E402
from synthetic.neural_g1 import (  # noqa: E402
  NeuralTrainConfig,
  REPORTED_FACTOR_INIT_SEED,
  REPORTED_FACTOR_INIT_STD,
  REPORTED_FACTOR_WARMUP_STEPS,
  REPORTED_HELDOUT_SEEDS,
  REPORTED_TRAINING_STEPS,
  evaluate_adapter,
  evaluate_neural_gate,
  model_specs,
  train_adapter,
)


REPORTED_MODELS = (
  'parameter_matched_independent',
  'natural_chain',
  'static_forest',
  'fixed_topology_dynamic_factors',
  'dynamic_topology_fixed_factors',
  'contextual_forest',
)


def _validate_resumed_job(
    checkpoint_path: Path,
    existing_rows: list[dict[str, object]],
    history: object,
    *,
    seed: int,
    model_name: str,
    expected_spec: object,
    expected_config: NeuralTrainConfig,
    num_contexts: int) -> None:
  """Reject a nominally complete resume job unless its provenance matches."""
  try:
    checkpoint = torch.load(
      checkpoint_path, map_location='cpu', weights_only=False)
  except Exception as error:
    raise ValueError(
      f'cannot validate resumed checkpoint {checkpoint_path}') from error
  if not isinstance(checkpoint, dict):
    raise ValueError(f'invalid resumed checkpoint payload: {checkpoint_path}')
  if int(checkpoint.get('seed', -1)) != seed:
    raise ValueError(
      f'resumed checkpoint seed mismatch for {model_name}/seed-{seed}')
  if checkpoint.get('model_spec') != expected_spec:
    raise ValueError(
      f'resumed checkpoint model spec mismatch for {model_name}/seed-{seed}')
  if checkpoint.get('train_config') != expected_config:
    raise ValueError(
      f'resumed checkpoint train config mismatch for '
      f'{model_name}/seed-{seed}')
  state_dict = checkpoint.get('model_state_dict')
  if not isinstance(state_dict, dict) or not state_dict:
    raise ValueError(
      f'resumed checkpoint has no model state for {model_name}/seed-{seed}')

  contexts = {int(row['context']) for row in existing_rows}
  if contexts != set(range(num_contexts)):
    raise ValueError(
      f'resumed records have wrong contexts for {model_name}/seed-{seed}')
  if any(int(row['seed']) != seed or row['model'] != model_name
         for row in existing_rows):
    raise ValueError(
      f'resumed records have wrong identity for {model_name}/seed-{seed}')
  if not isinstance(history, list) or not history:
    raise ValueError(
      f'resumed history is empty for {model_name}/seed-{seed}')
  try:
    final_step = int(history[-1]['step'])
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError(
      f'resumed history has no valid final step for '
      f'{model_name}/seed-{seed}') from error
  if final_step != expected_config.steps:
    raise ValueError(
      f'resumed history final step mismatch for {model_name}/seed-{seed}: '
      f'{final_step} != {expected_config.steps}')


def _args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument(
    '--resume', action='store_true',
    help=('resume a partial run in output-dir, skipping only jobs with a '
          'checkpoint, history, and all context-level records'))
  parser.add_argument(
    '--seeds', type=int, nargs='+', default=list(REPORTED_HELDOUT_SEEDS))
  parser.add_argument('--models', nargs='+', default=list(REPORTED_MODELS))
  parser.add_argument('--steps', type=int, default=REPORTED_TRAINING_STEPS)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--learning-rate', type=float, default=1e-2)
  parser.add_argument('--dependency-weight', type=float, default=1.0)
  parser.add_argument(
    '--factor-init-std', type=float, default=REPORTED_FACTOR_INIT_STD)
  parser.add_argument(
    '--factor-init-seed', type=int, default=REPORTED_FACTOR_INIT_SEED)
  parser.add_argument(
    '--factor-warmup-steps', type=int,
    default=REPORTED_FACTOR_WARMUP_STEPS)
  parser.add_argument('--eval-samples', type=int, default=20000)
  parser.add_argument('--log-every', type=int, default=100)
  parser.add_argument(
    '--inference-backend', choices=('auto', 'dense', 'low_rank'),
    default='low_rank')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                      else 'cpu')
  return parser.parse_args(argv)


def _git_sha() -> str:
  try:
    return subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return 'unknown'


def main() -> int:
  args = _args()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  task = ContextSwitchingMatching(vocab_size=4)
  available = model_specs(task)
  unknown = sorted(set(args.models) - set(available))
  if unknown:
    raise ValueError(f'unknown models: {unknown}')
  config = NeuralTrainConfig(
    steps=args.steps,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    dependency_weight=args.dependency_weight,
    factor_init_std=args.factor_init_std,
    factor_init_seed=args.factor_init_seed,
    factor_warmup_steps=args.factor_warmup_steps,
    eval_samples=args.eval_samples,
    log_every=args.log_every,
    inference_backend=args.inference_backend)
  device = torch.device(args.device)
  start = dt.datetime.now(dt.timezone.utc)
  records_path = output_dir / 'records.json'
  history_path = output_dir / 'training_history.json'
  prior_artifacts = (
    list(output_dir.glob('*-seed-*.pt'))
    + [path for path in (records_path, history_path, output_dir / 'gate.json')
       if path.exists()])
  if prior_artifacts and not args.resume:
    names = ', '.join(sorted(path.name for path in prior_artifacts))
    raise FileExistsError(
      f'{output_dir} already contains run artifacts ({names}); '
      'pass --resume or choose a fresh output directory')
  rows = (json.loads(records_path.read_text())
          if args.resume and records_path.exists() else [])
  histories = (json.loads(history_path.read_text())
               if args.resume and history_path.exists() else {})
  skipped_jobs = []
  for seed in args.seeds:
    for model_name in args.models:
      history_key = f'{model_name}/seed-{seed}'
      checkpoint_path = output_dir / f'{model_name}-seed-{seed}.pt'
      existing_rows = [
        row for row in rows
        if int(row['seed']) == seed and row['model'] == model_name]
      is_complete = (
        len(existing_rows) == task.num_contexts
        and len({int(row['context']) for row in existing_rows})
        == task.num_contexts
        and history_key in histories
        and checkpoint_path.exists())
      if is_complete:
        _validate_resumed_job(
          checkpoint_path,
          existing_rows,
          histories[history_key],
          seed=seed,
          model_name=model_name,
          expected_spec=available[model_name],
          expected_config=config,
          num_contexts=task.num_contexts)
        print(f'skipping complete seed={seed} model={model_name}', flush=True)
        skipped_jobs.append({'seed': seed, 'model': model_name})
        continue
      # A preemption can leave only part of a job. Discard that job's partial
      # metadata and rebuild it from the deterministic seed.
      rows = [
        row for row in rows
        if not (int(row['seed']) == seed and row['model'] == model_name)]
      histories.pop(history_key, None)
      spec = available[model_name]
      print(f'training seed={seed} model={model_name}', flush=True)
      model, history = train_adapter(task, spec, seed, config, device)
      rows.extend(evaluate_adapter(model, seed, config.eval_samples))
      histories[history_key] = history
      torch.save({
        'model_state_dict': model.state_dict(),
        'model_spec': spec,
        'train_config': config,
        'seed': seed,
      }, checkpoint_path)
      # Make each completed model durable before moving to the next job so a
      # Spot preemption loses at most one model fit.
      records_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
      history_path.write_text(
        json.dumps(histories, indent=2, sort_keys=True) + '\n')
  end = dt.datetime.now(dt.timezone.utc)

  records_path.write_text(
    json.dumps(rows, indent=2, sort_keys=True) + '\n')
  history_path.write_text(
    json.dumps(histories, indent=2, sort_keys=True) + '\n')
  gate = None
  required = set(model_specs(task)) - {'dynamic_topology_fixed_factors'}
  if required.issubset(args.models):
    gate = evaluate_neural_gate(rows, args.seeds)
    (output_dir / 'gate.json').write_text(
      json.dumps(gate, indent=2, sort_keys=True) + '\n')
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
  manifest = {
    'benchmark': 'g1_learned_frozen_adapter',
    'scientific_scope': (
      'frozen target-independent features; topology trained from exact '
      'conditional-influence targets'),
    'git_sha': _git_sha(),
    'command': sys.argv,
    'resume': args.resume,
    'requested_seeds': args.seeds,
    'requested_models': args.models,
    'resolved_train_config': dataclasses.asdict(config),
    'reported_heldout_protocol': {
      'development_seeds': list(range(1, 9)),
      'heldout_seeds': list(REPORTED_HELDOUT_SEEDS),
      'steps': REPORTED_TRAINING_STEPS,
      'factor_warmup_steps': REPORTED_FACTOR_WARMUP_STEPS,
      'factor_init_std': REPORTED_FACTOR_INIT_STD,
      'factor_init_seed': REPORTED_FACTOR_INIT_SEED,
    },
    'is_exact_reported_heldout_protocol': (
      args.seeds == list(REPORTED_HELDOUT_SEEDS)
      and args.models == list(REPORTED_MODELS)
      and config == NeuralTrainConfig()),
    'skipped_complete_jobs': skipped_jobs,
    'device': str(device),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'gpu': (torch.cuda.get_device_name(device)
            if device.type == 'cuda' else None),
    'python': platform.python_version(),
    'platform': platform.platform(),
    'start_time_utc': start.isoformat(),
    'end_time_utc': end.isoformat(),
  }
  (output_dir / 'manifest.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  run_manifest_path = output_dir / (
    'manifest-' + end.strftime('%Y%m%dT%H%M%SZ') + '.json')
  run_manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  if gate is None:
    return 0
  if len(args.seeds) >= 3:
    return 0 if gate['passed'] else 2
  return 0 if gate['screen_passed'] else 2


if __name__ == '__main__':
  raise SystemExit(main())
