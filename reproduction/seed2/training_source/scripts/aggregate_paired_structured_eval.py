#!/usr/bin/env python3
"""Aggregate paired conditional-denoising NLL runs into a JSON manifest.

The input is the ``metrics.csv`` written by Lightning's ``CSVLogger``.  CSV
rows can be sparse because different logging calls at the same validation
step are written separately; this script coalesces rows with the same
``(epoch, step)`` before selecting the last complete validation event.

This utility only reports conditional-denoising NLL per masked token.  It
does not estimate or infer an ELBO or perplexity.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRING_DIGEST_FILENAME = 'validation_pairing_digest.json'
PAIRING_DIGEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunSpec:
  """One explicitly identified evaluation run."""

  arm: str
  seed: int
  run_dir: Path


def _metric_aliases(prefix: str) -> dict[str, tuple[str, ...]]:
  # Validation-only Lightning logs normally retain the unsuffixed name.
  # ``_epoch`` is included for files produced when on_step and on_epoch were
  # both enabled.  ``retained_unary_mass`` is the name in the current model;
  # the shorter legacy spelling is accepted only as an input alias.
  names = {
    'conditional_nll_per_masked_token': (
      'conditional_nll_per_masked_token',),
    'candidate_recall': ('candidate_recall',),
    'retained_mass': ('retained_unary_mass', 'retained_mass'),
    'active_fraction': ('active_fraction',),
  }
  return {
    key: tuple(
      candidate
      for name in metric_names
      for candidate in (f'{prefix}/{name}', f'{prefix}/{name}_epoch'))
    for key, metric_names in names.items()
  }


def find_metrics_csv(run_path: Path) -> Path:
  """Resolve exactly one Lightning metrics CSV from an explicit run path."""
  run_path = run_path.expanduser().resolve()
  if run_path.is_file():
    if run_path.name != 'metrics.csv':
      raise ValueError(
        f'expected a metrics.csv file, received {run_path}')
    return run_path
  if not run_path.is_dir():
    raise FileNotFoundError(run_path)

  direct = run_path / 'metrics.csv'
  if direct.is_file():
    return direct
  candidates = sorted(run_path.rglob('metrics.csv'))
  if not candidates:
    raise FileNotFoundError(f'no metrics.csv below {run_path}')
  if len(candidates) != 1:
    choices = '\n'.join(f'  - {path}' for path in candidates)
    raise ValueError(
      f'multiple metrics.csv files below {run_path}; pass the exact run '
      f'directory or CSV file:\n{choices}')
  return candidates[0]


def find_pairing_digest(run_path: Path) -> Path:
  """Resolve exactly one validation pairing digest for an explicit run."""
  run_path = run_path.expanduser().resolve()
  if run_path.is_file():
    candidates = []
    for parent in list(run_path.parents)[:4]:
      candidate = parent / PAIRING_DIGEST_FILENAME
      if candidate.is_file():
        candidates.append(candidate)
  elif run_path.is_dir():
    direct = run_path / PAIRING_DIGEST_FILENAME
    if direct.is_file():
      return direct
    candidates = sorted(run_path.rglob(PAIRING_DIGEST_FILENAME))
  else:
    raise FileNotFoundError(run_path)
  candidates = sorted(set(candidates))
  if not candidates:
    raise FileNotFoundError(
      f'no {PAIRING_DIGEST_FILENAME} associated with {run_path}')
  if len(candidates) != 1:
    choices = '\n'.join(f'  - {path}' for path in candidates)
    raise ValueError(
      f'multiple {PAIRING_DIGEST_FILENAME} files associated with '
      f'{run_path}; pass the exact run directory:\n{choices}')
  return candidates[0]


def read_pairing_digest(path: Path) -> dict[str, Any]:
  """Read and validate the sanitized structured-validation commitment."""
  path = path.expanduser().resolve()
  with path.open() as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise TypeError(f'{path} must contain a JSON object')
  expected = {
    'schema_version': PAIRING_DIGEST_SCHEMA_VERSION,
    'artifact': 'structured_validation_pairing_digest',
    'algorithm': 'sha256',
  }
  for field, expected_value in expected.items():
    if payload.get(field) != expected_value:
      raise ValueError(
        f'{path} has invalid {field}: {payload.get(field)!r}; '
        f'expected {expected_value!r}')
  digest = payload.get('sha256')
  if (not isinstance(digest, str) or len(digest) != 64
      or any(character not in '0123456789abcdef' for character in digest)):
    raise ValueError(
      f'{path} sha256 must be 64 lowercase hexadecimal digits')
  for field in ('epoch', 'step'):
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      raise ValueError(f'{path} {field} must be a non-negative integer')
  if not isinstance(payload.get('sanity_checking'), bool):
    raise ValueError(f'{path} sanity_checking must be boolean')
  world_size = payload.get('world_size')
  if (not isinstance(world_size, int) or isinstance(world_size, bool)
      or world_size <= 0):
    raise ValueError(f'{path} world_size must be a positive integer')
  for field in (
      'num_batches', 'num_examples', 'num_token_slots',
      'num_active_tokens'):
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      raise ValueError(
        f'{path} {field} must be a non-negative integer')
  if payload['num_batches'] == 0 or payload['num_examples'] == 0:
    raise ValueError(f'{path} records an empty validation pass')
  rank_streams = payload.get('rank_streams')
  if not isinstance(rank_streams, list) or len(rank_streams) != world_size:
    raise ValueError(
      f'{path} rank_streams must contain one item per rank')
  observed_ranks = [record.get('rank') for record in rank_streams
                    if isinstance(record, dict)]
  if observed_ranks != list(range(world_size)):
    raise ValueError(
      f'{path} rank_streams must be ordered and contiguous by rank')
  if world_size == 1 and rank_streams[0].get('sha256') != digest:
    raise ValueError(
      f'{path} single-rank digest must equal its rank-stream digest')
  return payload


def _manifest_source_path(path: Path, source_path_root: Path | None) -> str:
  resolved = path.expanduser().resolve()
  if source_path_root is None:
    return str(resolved)
  root = source_path_root.expanduser().resolve()
  try:
    return resolved.relative_to(root).as_posix()
  except ValueError as error:
    raise ValueError(
      f'source path {resolved} is outside manifest source root {root}') \
      from error


def _finite_float(value: str, *, source: Path, column: str,
                  row_number: int) -> float | None:
  value = value.strip()
  if not value:
    return None
  try:
    result = float(value)
  except ValueError as error:
    raise ValueError(
      f'{source}:{row_number}: non-numeric {column}={value!r}') from error
  if not math.isfinite(result):
    raise ValueError(
      f'{source}:{row_number}: non-finite {column}={value!r}')
  return result


def _event_key(row: Mapping[str, str], row_index: int) -> tuple[str, str]:
  epoch = (row.get('epoch') or '').strip()
  step = (row.get('step') or '').strip()
  if not epoch and not step:
    # A file without Lightning's event coordinates cannot safely coalesce
    # unrelated rows, so each row remains its own event.
    return ('__row__', str(row_index))
  return (epoch, step)


def _display_coordinate(value: str) -> int | float | str | None:
  if value == '':
    return None
  try:
    number = float(value)
  except ValueError:
    return value
  if number.is_integer():
    return int(number)
  return number


def read_last_complete_metrics(
    metrics_csv: Path,
    *,
    metric_prefix: str = 'val/structured') -> dict[str, Any]:
  """Read the last complete structured validation event from a CSVLogger file.

  Lightning may put metrics from separate ``log_dict`` calls on sparse rows
  that share an epoch and global step.  Values are therefore coalesced by
  event coordinates.  Conflicting values for the same event are rejected.
  """
  metrics_csv = metrics_csv.expanduser().resolve()
  aliases = _metric_aliases(metric_prefix.rstrip('/'))
  with metrics_csv.open(newline='') as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
      raise ValueError(f'{metrics_csv} has no CSV header')
    available = {
      metric: tuple(name for name in names if name in reader.fieldnames)
      for metric, names in aliases.items()
    }
    missing = [metric for metric, names in available.items() if not names]
    if missing:
      raise ValueError(
        f'{metrics_csv} lacks columns for: {", ".join(missing)}')

    events: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, row in enumerate(reader, start=2):
      key = _event_key(row, row_index)
      event = events.setdefault(key, {
        'epoch': key[0],
        'step': key[1],
        'last_row_number': row_index,
        'metrics': {},
      })
      event['last_row_number'] = row_index
      for metric, columns in available.items():
        for column in columns:
          value = _finite_float(
            row.get(column, ''), source=metrics_csv, column=column,
            row_number=row_index)
          if value is None:
            continue
          previous = event['metrics'].get(metric)
          if previous is not None and not math.isclose(
              previous, value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
              f'{metrics_csv}:{row_index}: conflicting {metric} values '
              f'for epoch={key[0]!r}, step={key[1]!r}: '
              f'{previous} versus {value}')
          event['metrics'][metric] = value

  complete = [
    event for event in events.values()
    if all(metric in event['metrics'] for metric in aliases)
  ]
  if not complete:
    raise ValueError(
      f'{metrics_csv} has no validation event containing all required '
      'structured metrics')
  selected = max(complete, key=lambda event: event['last_row_number'])
  return {
    'epoch': _display_coordinate(selected['epoch']),
    'step': _display_coordinate(selected['step']),
    'last_csv_row': selected['last_row_number'],
    **selected['metrics'],
  }


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def git_metadata(repo_root: Path) -> dict[str, Any]:
  repo_root = repo_root.expanduser().resolve()
  try:
    sha = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
      stderr=subprocess.DEVNULL, text=True).strip()
    status = subprocess.check_output(
      ['git', 'status', '--porcelain'], cwd=repo_root,
      stderr=subprocess.DEVNULL, text=True)
  except (OSError, subprocess.CalledProcessError):
    return {'git_sha': None, 'git_worktree_dirty': None}
  return {'git_sha': sha, 'git_worktree_dirty': bool(status.strip())}


def checkpoint_metadata(
    checkpoints: Sequence[tuple[str, Path]],
    *,
    source_path_root: Path | None = None) -> list[dict[str, Any]]:
  seen: set[str] = set()
  result = []
  for label, raw_path in checkpoints:
    if label in seen:
      raise ValueError(f'duplicate checkpoint label: {label}')
    seen.add(label)
    path = raw_path.expanduser().resolve()
    if not path.is_file():
      raise FileNotFoundError(path)
    result.append({
      'label': label,
      'path': _manifest_source_path(path, source_path_root),
      'size_bytes': path.stat().st_size,
      'sha256': sha256_file(path),
    })
  return result


def _linear_percentile(sorted_values: Sequence[float], probability: float) -> float:
  """Linearly interpolate an empirical percentile (R/NumPy type 7)."""
  if not sorted_values:
    raise ValueError('cannot compute a percentile of an empty sample')
  position = (len(sorted_values) - 1) * probability
  lower_index = math.floor(position)
  upper_index = math.ceil(position)
  if lower_index == upper_index:
    return sorted_values[lower_index]
  weight = position - lower_index
  return (
    sorted_values[lower_index] * (1.0 - weight)
    + sorted_values[upper_index] * weight)


def paired_bootstrap_mean_ci(
    improvements: Sequence[float],
    *,
    num_resamples: int = 20_000,
    rng_seed: int = 1701,
    confidence_level: float = 0.95) -> dict[str, Any]:
  """Percentile CI from resampling paired seed-level improvements."""
  if not improvements:
    raise ValueError('paired bootstrap requires at least one improvement')
  if num_resamples <= 0:
    raise ValueError('bootstrap num_resamples must be positive')
  if not 0.0 < confidence_level < 1.0:
    raise ValueError('bootstrap confidence_level must be between 0 and 1')
  if any(not math.isfinite(value) for value in improvements):
    raise ValueError('bootstrap improvements must be finite')

  rng = random.Random(rng_seed)
  sample_size = len(improvements)
  bootstrap_means = []
  for _ in range(num_resamples):
    bootstrap_means.append(math.fsum(
      improvements[rng.randrange(sample_size)]
      for _ in range(sample_size)) / sample_size)
  bootstrap_means.sort()
  tail_probability = (1.0 - confidence_level) / 2.0
  return {
    'method': 'paired_seed_bootstrap_percentile',
    'statistic': 'mean_conditional_nll_improvement_per_masked_token',
    'num_seed_pairs': sample_size,
    'num_resamples': num_resamples,
    'rng': 'Python random.Random (MT19937)',
    'rng_seed': rng_seed,
    'confidence_level': confidence_level,
    'ci_lower': _linear_percentile(
      bootstrap_means, tail_probability),
    'ci_upper': _linear_percentile(
      bootstrap_means, 1.0 - tail_probability),
  }


def aggregate_runs(
    runs: Iterable[RunSpec],
    *,
    baseline_arm: str,
    treatment_arm: str,
    protocol_id: str,
    protocol_metadata: Mapping[str, Any] | None = None,
    checkpoints: Sequence[tuple[str, Path]] = (),
    repo_root: Path = REPO_ROOT,
    source_path_root: Path | None = None,
    metric_prefix: str = 'val/structured',
    require_pairing_digest: bool = False,
    pairing_tolerance: float = 1e-8,
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 1701,
    bootstrap_confidence: float = 0.95,
    timestamp_utc: str | None = None) -> dict[str, Any]:
  """Build a strict paired conditional-NLL manifest."""
  if not baseline_arm or not treatment_arm:
    raise ValueError('baseline and treatment arm names must be non-empty')
  if baseline_arm == treatment_arm:
    raise ValueError('baseline and treatment arm names must differ')
  if not protocol_id:
    raise ValueError('protocol_id must be non-empty')
  if not math.isfinite(pairing_tolerance) or pairing_tolerance < 0:
    raise ValueError('pairing_tolerance must be finite and non-negative')
  if protocol_metadata is not None and not isinstance(
      protocol_metadata, Mapping):
    raise TypeError('protocol_metadata must be a JSON object')

  by_key: dict[tuple[str, int], dict[str, Any]] = {}
  for run in runs:
    if run.arm not in {baseline_arm, treatment_arm}:
      raise ValueError(
        f'unexpected arm {run.arm!r}; expected {baseline_arm!r} or '
        f'{treatment_arm!r}')
    key = (run.arm, run.seed)
    if key in by_key:
      raise ValueError(f'duplicate run for arm={run.arm}, seed={run.seed}')
    metrics_csv = find_metrics_csv(run.run_dir)
    metrics = read_last_complete_metrics(
      metrics_csv, metric_prefix=metric_prefix)
    record = {
      'arm': run.arm,
      'seed': run.seed,
      'run_path': _manifest_source_path(run.run_dir, source_path_root),
      'metrics_csv': _manifest_source_path(metrics_csv, source_path_root),
      'metrics_csv_sha256': sha256_file(metrics_csv),
      **metrics,
    }
    if require_pairing_digest:
      pairing_digest_path = find_pairing_digest(run.run_dir)
      pairing_digest = read_pairing_digest(pairing_digest_path)
      if pairing_digest['sanity_checking']:
        raise ValueError(
          f'{pairing_digest_path} describes a sanity-validation pass')
      for coordinate in ('epoch', 'step'):
        if pairing_digest[coordinate] != metrics[coordinate]:
          raise ValueError(
            f'{pairing_digest_path} {coordinate}='
            f'{pairing_digest[coordinate]!r} does not match selected '
            f'metrics event {coordinate}={metrics[coordinate]!r}')
      record.update({
        'pairing_digest_path': _manifest_source_path(
          pairing_digest_path, source_path_root),
        'pairing_digest_file_sha256': sha256_file(
          pairing_digest_path),
        'pairing_digest': pairing_digest,
      })
    by_key[key] = record

  baseline_seeds = {
    seed for arm, seed in by_key if arm == baseline_arm}
  treatment_seeds = {
    seed for arm, seed in by_key if arm == treatment_arm}
  if not baseline_seeds:
    raise ValueError('no paired runs were supplied')
  if baseline_seeds != treatment_seeds:
    missing_treatment = sorted(baseline_seeds - treatment_seeds)
    missing_baseline = sorted(treatment_seeds - baseline_seeds)
    raise ValueError(
      'unpaired seeds: '
      f'missing {treatment_arm}={missing_treatment}, '
      f'missing {baseline_arm}={missing_baseline}')

  equality_metrics = (
    'candidate_recall', 'retained_mass', 'active_fraction')
  pairs = []
  improvements = []
  for seed in sorted(baseline_seeds):
    baseline = by_key[(baseline_arm, seed)]
    treatment = by_key[(treatment_arm, seed)]
    pairing_digest_sha256 = None
    if require_pairing_digest:
      if baseline['pairing_digest'] != treatment['pairing_digest']:
        raise ValueError(
          f'seed {seed} pairing digests differ between {baseline_arm} '
          f'and {treatment_arm}: '
          f'{baseline["pairing_digest"]["sha256"]} versus '
          f'{treatment["pairing_digest"]["sha256"]}')
      pairing_digest_sha256 = baseline['pairing_digest']['sha256']
    pairing_differences = {}
    for metric in equality_metrics:
      difference = abs(baseline[metric] - treatment[metric])
      pairing_differences[metric] = difference
      if difference > pairing_tolerance:
        raise ValueError(
          f'seed {seed} is not paired for {metric}: '
          f'{baseline_arm}={baseline[metric]}, '
          f'{treatment_arm}={treatment[metric]}, '
          f'absolute difference={difference} exceeds '
          f'tolerance={pairing_tolerance}')
    improvement = (
      baseline['conditional_nll_per_masked_token']
      - treatment['conditional_nll_per_masked_token'])
    improvements.append(improvement)
    pairs.append({
      'seed': seed,
      'baseline': baseline,
      'treatment': treatment,
      'pairing_absolute_differences': pairing_differences,
      'pairing_digest_sha256': pairing_digest_sha256,
      'conditional_nll_improvement_per_masked_token': improvement,
    })

  created = timestamp_utc or dt.datetime.now(
    dt.timezone.utc).isoformat()
  return {
    'schema_version': 1,
    'created_utc': created,
    'objective': 'conditional_denoising_nll_per_masked_token',
    'scope_note': (
      'Reports conditional-denoising NLL per masked token only; '
      'ELBO and perplexity are not computed or inferred.'),
    'protocol': {
      'id': protocol_id,
      'metadata': dict(protocol_metadata or {}),
      'metric_prefix': metric_prefix.rstrip('/'),
      'pairing_tolerance': pairing_tolerance,
      'pairing_metrics': list(equality_metrics),
      'pairing_digest_required': require_pairing_digest,
      'pairing_digest_filename': (
        PAIRING_DIGEST_FILENAME if require_pairing_digest else None),
      'improvement_definition': f'{baseline_arm} minus {treatment_arm}',
    },
    'arms': {
      'baseline': baseline_arm,
      'treatment': treatment_arm,
    },
    'seeds': sorted(baseline_seeds),
    'num_pairs': len(pairs),
    'pairs': pairs,
    'mean_conditional_nll_improvement_per_masked_token': (
      math.fsum(improvements) / len(improvements)),
    'paired_bootstrap': paired_bootstrap_mean_ci(
      improvements,
      num_resamples=bootstrap_resamples,
      rng_seed=bootstrap_seed,
      confidence_level=bootstrap_confidence),
    'repository': git_metadata(repo_root),
    'checkpoints': checkpoint_metadata(
      checkpoints, source_path_root=source_path_root),
  }


def _load_protocol_metadata(path: Path | None) -> Mapping[str, Any]:
  if path is None:
    return {}
  with path.expanduser().resolve().open() as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise TypeError('protocol metadata JSON must contain an object')
  return payload


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--run', action='append', nargs=3, required=True,
    metavar=('ARM', 'SEED', 'RUN_DIR'),
    help='repeat once for every explicit arm/seed run')
  parser.add_argument('--baseline-arm', required=True)
  parser.add_argument('--treatment-arm', required=True)
  parser.add_argument('--protocol-id', required=True)
  parser.add_argument('--protocol-metadata-json', type=Path)
  parser.add_argument(
    '--checkpoint', action='append', nargs=2, default=[],
    metavar=('LABEL', 'PATH'),
    help='optional named checkpoint; repeat for multiple files')
  parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
  parser.add_argument(
    '--source-path-root', type=Path,
    help='store run and CSV paths relative to this provenance root')
  parser.add_argument('--metric-prefix', default='val/structured')
  parser.add_argument(
    '--require-pairing-digest', action='store_true',
    help=(
      'require identical cryptographic validation-input/corruption '
      'commitments for both arms of every seed'))
  parser.add_argument('--pairing-tolerance', type=float, default=1e-8)
  parser.add_argument('--bootstrap-resamples', type=int, default=20_000)
  parser.add_argument('--bootstrap-seed', type=int, default=1701)
  parser.add_argument('--bootstrap-confidence', type=float, default=0.95)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  run_specs = []
  for arm, seed_text, run_dir in args.run:
    try:
      seed = int(seed_text)
    except ValueError as error:
      raise ValueError(f'seed must be an integer: {seed_text!r}') from error
    run_specs.append(RunSpec(arm=arm, seed=seed, run_dir=Path(run_dir)))
  manifest = aggregate_runs(
    run_specs,
    baseline_arm=args.baseline_arm,
    treatment_arm=args.treatment_arm,
    protocol_id=args.protocol_id,
    protocol_metadata=_load_protocol_metadata(
      args.protocol_metadata_json),
    checkpoints=[(label, Path(path)) for label, path in args.checkpoint],
    repo_root=args.repo_root,
    source_path_root=args.source_path_root,
    metric_prefix=args.metric_prefix,
    require_pairing_digest=args.require_pairing_digest,
    pairing_tolerance=args.pairing_tolerance,
    bootstrap_resamples=args.bootstrap_resamples,
    bootstrap_seed=args.bootstrap_seed,
    bootstrap_confidence=args.bootstrap_confidence)
  output = args.output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  print(json.dumps(manifest, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
