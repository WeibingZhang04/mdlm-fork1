#!/usr/bin/env python3
"""Compile the frozen real-text experiment matrix into resumable job specs.

The compiler is intentionally fail closed: unknown manifest fields, missing
data configs, non-pinned inputs, unsafe artifact paths, and matrix values not
declared by the frozen protocol are rejected. It writes inert JSON argv arrays;
``run_compiled_job.py`` is the only supported launcher and never invokes a
shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  # ``python scripts/compile_experiment_matrix.py`` otherwise exposes only the
  # scripts directory, but gated suites use package-qualified lazy imports.
  sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MANIFEST = (
  REPO_ROOT / 'configs/experiment/contextual-forest-expansion-v1.yaml')
DEFAULT_ALLOWED_ARTIFACT_ROOT = Path('/mnt/contextual-forest')
JOB_SCHEMA_VERSION = 2
PLAN_SCHEMA_VERSION = 2
SLUG_PATTERN = re.compile(r'^[a-z0-9][a-z0-9_-]*$')
TRUSTED_PROMOTION_POLICIES = {
  'contextual-forest-expansion-v1': (
    REPO_ROOT / 'configs/experiment'
    / 'contextual-forest-expansion-v1-promotion-policy.yaml'),
}
TRUSTED_CANDIDATE_K_PROMOTION_TEMPLATES = {
  'contextual-forest-expansion-v1': {
    'candidate_k_128_pilot': (
      REPO_ROOT / 'configs/experiment'
      / 'contextual-forest-k128-promotion-policy-template.yaml'),
  },
}
TRUSTED_CAUSAL_PROMOTION_TEMPLATES = {
  'contextual-forest-causal-evidence-v1': {
    'causal_smoke': (
      REPO_ROOT / 'configs/experiment'
      / 'contextual-forest-causal-smoke-promotion-policy-template.yaml'),
  },
}


def _exact_keys(
    value: object,
    expected: Iterable[str],
    *,
    context: str,
) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise TypeError(f'{context} must be a mapping')
  expected_set = set(expected)
  actual = set(value)
  missing = sorted(expected_set - actual)
  unknown = sorted(actual - expected_set)
  if missing or unknown:
    raise ValueError(
      f'{context} schema mismatch: missing={missing}, unknown={unknown}')
  return value


def _slug(value: object, *, context: str) -> str:
  if not isinstance(value, str) or not SLUG_PATTERN.fullmatch(value):
    raise ValueError(
      f'{context} must match {SLUG_PATTERN.pattern!r}, found {value!r}')
  return value


def _strict_bool(value: object, *, context: str) -> bool:
  if not isinstance(value, bool):
    raise TypeError(f'{context} must be boolean')
  return value


def _positive_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f'{context} must be a positive integer')
  return value


def _unique_positive_ints(value: object, *, context: str) -> list[int]:
  if not isinstance(value, list) or not value:
    raise ValueError(f'{context} must be a non-empty list')
  result = [
    _positive_int(item, context=f'{context}[{index}]')
    for index, item in enumerate(value)]
  if len(result) != len(set(result)):
    raise ValueError(f'{context} contains duplicates')
  return result


def _unique_rates(value: object, *, context: str) -> list[float]:
  if not isinstance(value, list) or not value:
    raise ValueError(f'{context} must be a non-empty list')
  result = []
  for index, item in enumerate(value):
    if (not isinstance(item, (int, float)) or isinstance(item, bool)
        or not math.isfinite(float(item)) or not 0.0 < float(item) < 1.0):
      raise ValueError(f'{context}[{index}] must be finite and in (0,1)')
    result.append(float(item))
  if len(result) != len(set(result)):
    raise ValueError(f'{context} contains duplicates')
  return result


def _unique_slugs(value: object, *, context: str) -> list[str]:
  if not isinstance(value, list) or not value:
    raise ValueError(f'{context} must be a non-empty list')
  result = [
    _slug(item, context=f'{context}[{index}]')
    for index, item in enumerate(value)]
  if len(result) != len(set(result)):
    raise ValueError(f'{context} contains duplicates')
  return result


def _sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _canonical_json(value: object) -> str:
  return json.dumps(
    value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _validate_sha256(value: object, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != 64
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(f'{context} must be 64 lowercase hexadecimal digits')
  return value


def _validate_git_revision(value: object, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != 40
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(f'{context} must be 40 lowercase hexadecimal digits')
  return value


def _safe_root(path: Path, allowed_root: Path, *, context: str) -> Path:
  if not path.is_absolute():
    raise ValueError(f'{context} must be absolute: {path}')
  resolved = path.resolve(strict=False)
  allowed = allowed_root.resolve(strict=False)
  if resolved == Path(resolved.anchor) or resolved == allowed:
    raise ValueError(f'{context} is too broad: {resolved}')
  try:
    resolved.relative_to(allowed)
  except ValueError as error:
    raise ValueError(
      f'{context} {resolved} is outside allowed root {allowed}') from error
  return resolved


def _git_metadata(repo_root: Path) -> dict[str, Any]:
  try:
    sha = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
      stderr=subprocess.DEVNULL, text=True).strip()
    dirty = bool(subprocess.check_output(
      ['git', 'status', '--porcelain'], cwd=repo_root,
      stderr=subprocess.DEVNULL, text=True).strip())
  except (OSError, subprocess.CalledProcessError):
    return {'sha': None, 'dirty': None}
  return {'sha': sha, 'dirty': dirty}


def _clean_repository_identity(repo_root: Path) -> dict[str, Any]:
  repository = _git_metadata(repo_root)
  if not isinstance(repository, Mapping) \
      or set(repository) != {'sha', 'dirty'}:
    raise ValueError('repository metadata has an invalid schema')
  _validate_git_revision(repository['sha'], context='repository SHA')
  if repository['dirty'] is not False:
    raise ValueError(
      'experiment plans require a clean committed repository; commit all '
      'scientific code before compiling')
  return dict(repository)


def _read_yaml_or_json(path: Path) -> object:
  if not path.is_file():
    raise FileNotFoundError(path)
  with path.open() as handle:
    return yaml.safe_load(handle)


def load_and_validate_manifest(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  """Load the exact version-1 schema and validate every matrix reference."""
  path = path.expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = _read_yaml_or_json(path)
  manifest = _exact_keys(payload, {
    'schema_version', 'protocol_id', 'protocol_status', 'scientific_scope',
    'artifact_root', 'backbone', 'training', 'controls', 'datasets',
    'evaluation', 'suites', 'analysis',
  }, context='manifest')
  if manifest['schema_version'] != 1:
    raise ValueError('manifest.schema_version must equal 1')
  protocol_id = _slug(manifest['protocol_id'], context='protocol_id')
  if manifest['protocol_status'] != 'frozen_before_primary_results':
    raise ValueError(
      'protocol_status must be frozen_before_primary_results')
  if not isinstance(manifest['scientific_scope'], str) \
      or not manifest['scientific_scope'].strip():
    raise ValueError('scientific_scope must be non-empty')
  if not isinstance(manifest['artifact_root'], str):
    raise TypeError('artifact_root must be a string')

  backbone = _exact_keys(manifest['backbone'], {
    'wrapper_path', 'wrapper_sha256', 'source_repository',
    'source_revision', 'source_sha256', 'ema_available', 'ema_used',
  }, context='backbone')
  if not isinstance(backbone['wrapper_path'], str) \
      or not Path(backbone['wrapper_path']).is_absolute():
    raise ValueError('backbone.wrapper_path must be absolute')
  _validate_sha256(backbone['wrapper_sha256'], context='wrapper_sha256')
  _validate_git_revision(
    backbone['source_revision'], context='source_revision')
  _validate_sha256(backbone['source_sha256'], context='source_sha256')
  if not isinstance(backbone['source_repository'], str) \
      or not backbone['source_repository']:
    raise ValueError('backbone.source_repository must be non-empty')
  if _strict_bool(backbone['ema_available'], context='ema_available'):
    raise ValueError('this protocol requires a raw non-EMA backbone')
  if _strict_bool(backbone['ema_used'], context='ema_used'):
    raise ValueError('this protocol forbids EMA selection')

  training = _exact_keys(manifest['training'], {
    'data_config', 'train_seeds', 'updates', 'batch_size',
    'validation_batches', 'validation_interval', 'checkpoint_interval',
    'head_learning_rate', 'warmup_updates', 'topology_weight',
    'factorized_aux_weight', 'precision', 'preemption_policy',
    'corruption_rng_policy', 'topology_teacher_rng_policy',
    'cross_control_pairing_policy',
  }, context='training')
  training_data_config = _slug(
    training['data_config'], context='training.data_config')
  training_config_path = (
    repo_root / 'configs/data' / f'{training_data_config}.yaml')
  if not training_config_path.is_file():
    raise FileNotFoundError(
      f'pinned training data config missing: {training_config_path}')
  training_seeds = _unique_positive_ints(
    training['train_seeds'], context='training.train_seeds')
  for field in (
      'updates', 'batch_size', 'validation_batches', 'validation_interval',
      'checkpoint_interval', 'warmup_updates'):
    _positive_int(training[field], context=f'training.{field}')
  for field in ('head_learning_rate', 'topology_weight',
                'factorized_aux_weight'):
    value = training[field]
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) or float(value) < 0):
      raise ValueError(f'training.{field} must be finite and non-negative')
  if training['precision'] not in {'bf16', '32', '16-mixed'}:
    raise ValueError('unsupported training.precision')
  if (training['preemption_policy'] !=
      'full_attempt_restart_no_stream_cursor'):
    raise ValueError(
      'streaming training must use full_attempt_restart_no_stream_cursor')
  expected_rng_policies = {
    'corruption_rng_policy':
      'paired_private_torch_generator_v1_seeded_by_train_seed_epoch_rank',
    'topology_teacher_rng_policy':
      'domain_separated_private_torch_generator_v1_offset_4294967291',
    'cross_control_pairing_policy':
      'identical_corruption_stream_for_equal_train_seed_epoch_rank',
  }
  for field, expected in expected_rng_policies.items():
    if training[field] != expected:
      raise ValueError(f'training.{field} must equal {expected!r}')

  controls = manifest['controls']
  required_controls = {
    'dynamic_dynamic': ('dynamic', 'dynamic'),
    'fixed_dynamic': ('fixed', 'dynamic'),
    'dynamic_fixed': ('dynamic', 'fixed'),
    'static_static': ('fixed', 'fixed'),
  }
  _exact_keys(controls, required_controls, context='controls')
  for name, expected_modes in required_controls.items():
    control = _exact_keys(controls[name], {
      'topology_mode', 'factor_mode', 'topology_weight',
    }, context=f'controls.{name}')
    actual_modes = (control['topology_mode'], control['factor_mode'])
    if actual_modes != expected_modes:
      raise ValueError(
        f'controls.{name} modes must be {expected_modes}, got '
        f'{actual_modes}')
    weight = control['topology_weight']
    if (not isinstance(weight, (int, float)) or isinstance(weight, bool)
        or not math.isfinite(float(weight)) or float(weight) < 0):
      raise ValueError(f'controls.{name}.topology_weight is invalid')
    if expected_modes[0] == 'fixed' and float(weight) != 0.0:
      raise ValueError(f'fixed control {name} must disable topology loss')
    if (expected_modes[0] == 'dynamic'
        and float(weight) != float(training['topology_weight'])):
      raise ValueError(
        f'dynamic control {name} must use the frozen training topology '
        f'weight {training["topology_weight"]}')

  datasets = manifest['datasets']
  if not isinstance(datasets, Mapping) or len(datasets) < 2:
    raise ValueError('datasets must contain at least two datasets')
  for dataset_name, raw_dataset in datasets.items():
    _slug(dataset_name, context='dataset name')
    dataset = _exact_keys(raw_dataset, {
      'data_config', 'require_disjoint_training_proof',
    }, context=f'datasets.{dataset_name}')
    data_config = _slug(
      dataset['data_config'], context=f'datasets.{dataset_name}.data_config')
    _strict_bool(
      dataset['require_disjoint_training_proof'],
      context=f'datasets.{dataset_name}.require_disjoint_training_proof')
    config_path = repo_root / 'configs/data' / f'{data_config}.yaml'
    if not config_path.is_file():
      raise FileNotFoundError(
        f'pinned data config missing for {dataset_name}: {config_path}')

  evaluation = _exact_keys(manifest['evaluation'], {
    'corruption_seeds', 'mask_rates', 'candidate_ks',
    'primary_candidate_k', 'batch_size', 'precision',
    'require_pairing_digest', 'comparisons',
  }, context='evaluation')
  eval_seeds = _unique_positive_ints(
    evaluation['corruption_seeds'], context='evaluation.corruption_seeds')
  mask_rates = _unique_rates(
    evaluation['mask_rates'], context='evaluation.mask_rates')
  candidate_ks = _unique_positive_ints(
    evaluation['candidate_ks'], context='evaluation.candidate_ks')
  primary_k = _positive_int(
    evaluation['primary_candidate_k'], context='primary_candidate_k')
  if primary_k not in candidate_ks:
    raise ValueError('primary_candidate_k is not in candidate_ks')
  _positive_int(evaluation['batch_size'], context='evaluation.batch_size')
  if evaluation['precision'] not in {'bf16', '32', '16-mixed'}:
    raise ValueError('unsupported evaluation.precision')
  if not _strict_bool(
      evaluation['require_pairing_digest'],
      context='evaluation.require_pairing_digest'):
    raise ValueError('pairing digests are mandatory')
  comparisons = evaluation['comparisons']
  if not isinstance(comparisons, Mapping) or not comparisons:
    raise ValueError('evaluation.comparisons must be non-empty')
  for name, raw_comparison in comparisons.items():
    _slug(name, context='comparison name')
    comparison = _exact_keys(
      raw_comparison, {'baseline', 'treatment'},
      context=f'comparisons.{name}')
    for role in ('baseline', 'treatment'):
      if comparison[role] not in controls:
        raise ValueError(f'{name}.{role} is not a declared control')
    if comparison['baseline'] == comparison['treatment']:
      raise ValueError(f'{name} compares an arm with itself')

  suites = manifest['suites']
  if not isinstance(suites, Mapping) or not suites:
    raise ValueError('suites must be non-empty')
  for suite_name, raw_suite in suites.items():
    _slug(suite_name, context='suite name')
    suite = _exact_keys(raw_suite, {
      'promotion_from', 'controls', 'train_seeds', 'datasets',
      'corruption_seeds', 'mask_rates', 'candidate_ks',
      'validation_batches',
    }, context=f'suites.{suite_name}')
    promotion_from = suite['promotion_from']
    if promotion_from is not None:
      _slug(
        promotion_from,
        context=f'suites.{suite_name}.promotion_from')
    suite_controls = _unique_slugs(
      suite['controls'], context=f'suites.{suite_name}.controls')
    suite_train_seeds = _unique_positive_ints(
      suite['train_seeds'], context=f'suites.{suite_name}.train_seeds')
    suite_datasets = _unique_slugs(
      suite['datasets'], context=f'suites.{suite_name}.datasets')
    suite_eval_seeds = _unique_positive_ints(
      suite['corruption_seeds'],
      context=f'suites.{suite_name}.corruption_seeds')
    suite_rates = _unique_rates(
      suite['mask_rates'], context=f'suites.{suite_name}.mask_rates')
    suite_ks = _unique_positive_ints(
      suite['candidate_ks'], context=f'suites.{suite_name}.candidate_ks')
    _positive_int(
      suite['validation_batches'],
      context=f'suites.{suite_name}.validation_batches')
    for context, selected, declared in (
        ('controls', suite_controls, controls),
        ('train_seeds', suite_train_seeds, training_seeds),
        ('datasets', suite_datasets, datasets),
        ('corruption_seeds', suite_eval_seeds, eval_seeds),
        ('mask_rates', suite_rates, mask_rates),
        ('candidate_ks', suite_ks, candidate_ks)):
      missing = sorted(set(selected) - set(declared))
      if missing:
        raise ValueError(
          f'suites.{suite_name}.{context} contains undeclared {missing}')
  for suite_name, suite in suites.items():
    promotion_from = suite['promotion_from']
    if promotion_from is not None:
      if promotion_from not in suites:
        raise ValueError(
          f'suites.{suite_name}.promotion_from names unknown suite '
          f'{promotion_from!r}')
      if promotion_from == suite_name:
        raise ValueError(f'suite {suite_name} cannot promote itself')

  analysis_fields = {
    'document_record_schema_version', 'primary_comparison',
    'bootstrap_unit_order', 'average_corruptions_within_document',
    'equal_weight_datasets', 'equal_weight_mask_rates',
    'bootstrap_resamples', 'bootstrap_seed', 'confidence_level',
    'reject_incomplete_factorial_cells',
  }
  if manifest['analysis'].get('document_record_schema_version') == 2:
    analysis_fields.add('permutation_control_gate')
  analysis = _exact_keys(
    manifest['analysis'], analysis_fields, context='analysis')
  if analysis['document_record_schema_version'] not in {1, 2}:
    raise ValueError('document_record_schema_version must equal 1 or 2')
  if analysis['primary_comparison'] not in comparisons:
    raise ValueError('analysis.primary_comparison is not declared')
  if analysis['bootstrap_unit_order'] != ['train_seed', 'document']:
    raise ValueError('bootstrap_unit_order must be [train_seed, document]')
  for field in (
      'average_corruptions_within_document', 'equal_weight_datasets',
      'equal_weight_mask_rates', 'reject_incomplete_factorial_cells'):
    if not _strict_bool(analysis[field], context=f'analysis.{field}'):
      raise ValueError(f'analysis.{field} must remain true')
  _positive_int(
    analysis['bootstrap_resamples'], context='analysis.bootstrap_resamples')
  _positive_int(analysis['bootstrap_seed'], context='analysis.bootstrap_seed')
  confidence = analysis['confidence_level']
  if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
      or not 0.0 < float(confidence) < 1.0):
    raise ValueError('analysis.confidence_level must be in (0,1)')
  if analysis['document_record_schema_version'] == 2:
    permutation_gate = _exact_keys(analysis['permutation_control_gate'], {
      'minimum_pooled_changed_edge_fraction',
      'minimum_condition_changed_edge_fraction',
    }, context='analysis.permutation_control_gate')
    for field, value in permutation_gate.items():
      if (not isinstance(value, (int, float)) or isinstance(value, bool)
          or not math.isfinite(float(value))
          or not 0.0 <= float(value) <= 1.0):
        raise ValueError(
          f'analysis.permutation_control_gate.{field} must be in [0,1]')

  return dict(manifest, protocol_id=protocol_id)


def _rate_tag(rate: float) -> str:
  millionths = round(rate * 1_000_000)
  if not math.isclose(rate, millionths / 1_000_000, abs_tol=1e-12):
    raise ValueError(f'mask rate has more than six decimal places: {rate}')
  return f'{millionths:06d}'


def _hydra_common(
    manifest: Mapping[str, Any], artifact_dir: str) -> list[str]:
  backbone = manifest['backbone']
  return [
    'model=contextual-forest-small',
    f'model.structured_decoder.training.backbone_checkpoint='
    f'{backbone["wrapper_path"]}',
    'model.structured_decoder.training.use_ema_backbone=false',
    'model.structured_decoder.training.strict_backbone_checkpoint=true',
    'training.ema=0',
    'eval.generate_samples=false',
    'trainer.accelerator=cuda',
    'trainer.devices=1',
    'trainer.num_nodes=1',
    f'checkpointing.save_dir={artifact_dir}',
    f'hydra.run.dir={artifact_dir}',
    'wandb=null',
  ]


def _control_overrides(control: Mapping[str, Any]) -> list[str]:
  return [
    f'model.structured_decoder.topology_mode={control["topology_mode"]}',
    f'model.structured_decoder.factor_mode={control["factor_mode"]}',
    'model.structured_decoder.independent_mode=false',
    f'model.structured_decoder.training.topology_weight='
    f'{control["topology_weight"]}',
  ]


def _job(
    *,
    protocol_id: str,
    source_manifest_sha256: str,
    source_repository_sha: str,
    plan_id: str,
    job_id: str,
    kind: str,
    artifact_dir: Path,
    suites: Sequence[str],
    dependencies: Sequence[str],
    identity: Mapping[str, Any],
    argv: Sequence[str],
    execution_mode: str,
    external_inputs: Sequence[Mapping[str, Any]],
    required_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  return {
    'schema_version': JOB_SCHEMA_VERSION,
    'protocol_id': protocol_id,
    'source_manifest_sha256': source_manifest_sha256,
    'source_repository_sha': source_repository_sha,
    'plan_id': plan_id,
    'job_id': job_id,
    'kind': kind,
    'artifact_dir': str(artifact_dir),
    'suites': sorted(set(suites)),
    'dependencies': sorted(set(dependencies)),
    'identity': dict(identity),
    'argv': list(argv),
    'execution_mode': execution_mode,
    'external_inputs': list(external_inputs),
    'required_outputs': list(required_outputs),
  }


def build_jobs(
    manifest: Mapping[str, Any],
    *,
    selected_suites: Sequence[str],
    artifact_root: Path,
    source_manifest_sha256: str,
    source_repository_sha: str,
    plan_id: str,
) -> dict[str, dict[str, Any]]:
  """Expand selected suites, deduplicating canonical jobs across stages."""
  unknown_suites = sorted(set(selected_suites) - set(manifest['suites']))
  if unknown_suites:
    raise ValueError(f'unknown suites: {unknown_suites}')
  if not selected_suites:
    raise ValueError('at least one suite must be selected')

  memberships: dict[tuple[Any, ...], set[str]] = {}
  for suite_name in selected_suites:
    suite = manifest['suites'][suite_name]
    for control_name in suite['controls']:
      for train_seed in suite['train_seeds']:
        for candidate_k in suite['candidate_ks']:
          # Candidate support changes the structured training objective, so a
          # K sweep fits a fresh adapter rather than relabeling a K=64 fit.
          memberships.setdefault(
            ('train', control_name, train_seed, candidate_k), set()).add(
              suite_name)
          memberships.setdefault(
            ('export', control_name, train_seed, candidate_k), set()).add(
              suite_name)
          for dataset in suite['datasets']:
            for eval_seed in suite['corruption_seeds']:
              for mask_rate in suite['mask_rates']:
                key = (
                  'eval', control_name, train_seed, dataset, eval_seed,
                  float(mask_rate), candidate_k,
                  suite['validation_batches'])
                memberships.setdefault(key, set()).add(suite_name)

  jobs: dict[str, dict[str, Any]] = {}
  training = manifest['training']
  evaluation = manifest['evaluation']
  protocol_id = manifest['protocol_id']

  for key, suite_names in memberships.items():
    kind, control_name, train_seed, *rest = key
    control = manifest['controls'][control_name]
    if kind in {'train', 'export'}:
      candidate_k = rest[0]
    else:
      candidate_k = rest[3]
    train_id = (
      f'train--{control_name}--s{train_seed:03d}--k{candidate_k:03d}')
    export_id = (
      f'export--{control_name}--s{train_seed:03d}--k{candidate_k:03d}')

    if kind == 'train':
      artifact_dir = artifact_root / 'runs' / train_id
      argv = [
        '{python}', 'main.py',
        f'data={training["data_config"]}',
        f'seed={train_seed}',
        f'trainer.max_steps={training["updates"]}',
        f'trainer.val_check_interval={training["validation_interval"]}',
        f'trainer.limit_val_batches={training["validation_batches"]}',
        'trainer.num_sanity_val_steps=0',
        f'trainer.precision={training["precision"]}',
        f'loader.global_batch_size={training["batch_size"]}',
        f'loader.eval_global_batch_size={training["batch_size"]}',
        f'loader.batch_size={training["batch_size"]}',
        f'loader.eval_batch_size={training["batch_size"]}',
        'loader.num_workers=0',
        f'model.structured_decoder.training.head_lr='
        f'{training["head_learning_rate"]}',
        f'model.structured_decoder.training.factorized_aux_weight='
        f'{training["factorized_aux_weight"]}',
        f'model.structured_decoder.top_k={candidate_k}',
        f'lr_scheduler.num_warmup_steps={training["warmup_updates"]}',
        f'callbacks.checkpoint_every_n_steps.every_n_train_steps='
        f'{training["checkpoint_interval"]}',
        'checkpointing.resume_from_ckpt=false',
      ]
      argv += _control_overrides(control)
      # Fixed topology/factor controls intentionally bypass trainable head
      # branches. Lightning wraps even a one-GPU run in DDP, whose default
      # reducer otherwise aborts as soon as it sees those unused parameters.
      # Detection changes only gradient-reduction bookkeeping; the loss,
      # optimizer, data order, and active gradients remain unchanged.
      if (control['topology_mode'] != 'dynamic'
          or control['factor_mode'] != 'dynamic'):
        argv.append('strategy.find_unused_parameters=true')
      argv += _hydra_common(manifest, '{artifact_dir}')
      job = _job(
        protocol_id=protocol_id,
        source_manifest_sha256=source_manifest_sha256,
        source_repository_sha=source_repository_sha,
        plan_id=plan_id,
        job_id=train_id,
        kind='train',
        artifact_dir=artifact_dir,
        suites=sorted(suite_names),
        dependencies=[],
        identity={
          'control': control_name, 'train_seed': train_seed,
          'candidate_k': candidate_k, 'updates': training['updates'],
          'corruption_rng_policy': training['corruption_rng_policy'],
          'topology_teacher_rng_policy':
            training['topology_teacher_rng_policy'],
          'cross_control_pairing_policy':
            training['cross_control_pairing_policy'],
        },
        argv=argv,
        execution_mode='fresh_attempt',
        external_inputs=[{
          'role': 'released_backbone_wrapper',
          'path': manifest['backbone']['wrapper_path'],
          'sha256': manifest['backbone']['wrapper_sha256'],
        }],
        required_outputs=[{
          'name': 'checkpoint',
          'pattern': 'checkpoints/last.ckpt',
          'exactly_one': True,
        }, {
          'name': 'training_data_provenance',
          'pattern': 'data_provenance/train-*.json',
          'exactly_one': True,
        }, {
          'name': 'training_validation_data_provenance',
          'pattern': 'data_provenance/valid-*.json',
          'exactly_one': True,
        }])
    elif kind == 'export':
      artifact_dir = artifact_root / 'runs' / export_id
      source_checkpoint = (
        f'${{artifact:{train_id}:checkpoints/last.ckpt}}')
      source_sha = f'${{sha256:{train_id}:checkpoints/last.ckpt}}'
      argv = [
        '{python}', 'scripts/export_structured_adapter.py',
        '--checkpoint', source_checkpoint,
        '--output', '{artifact_dir}/adapter.safetensors',
        '--manifest', '{artifact_dir}/adapter-manifest.json',
        '--expected-checkpoint-sha256', source_sha,
        '--expected-global-step', str(training['updates']),
        '--control-identity', control_name,
        '--topology-mode', control['topology_mode'],
        '--factor-mode', control['factor_mode'],
        '--candidate-k', str(candidate_k),
        '--independent-mode', 'false',
        '--topology-weight', str(control['topology_weight']),
      ]
      job = _job(
        protocol_id=protocol_id,
        source_manifest_sha256=source_manifest_sha256,
        source_repository_sha=source_repository_sha,
        plan_id=plan_id,
        job_id=export_id,
        kind='export',
        artifact_dir=artifact_dir,
        suites=sorted(suite_names),
        dependencies=[train_id],
        identity={
          'control': control_name, 'train_seed': train_seed,
          'candidate_k': candidate_k,
          'topology_mode': control['topology_mode'],
          'factor_mode': control['factor_mode'],
          'independent_mode': False,
          'topology_weight': float(control['topology_weight']),
        },
        argv=argv,
        execution_mode='fresh_attempt',
        external_inputs=[],
        required_outputs=[
          {'name': 'adapter', 'pattern': 'adapter.safetensors',
           'exactly_one': True},
          {'name': 'adapter_manifest', 'pattern': 'adapter-manifest.json',
           'exactly_one': True},
        ])
    else:
      dataset, eval_seed, mask_rate, candidate_k, validation_batches = rest
      rate_tag = _rate_tag(mask_rate)
      eval_id = (
        f'eval--{control_name}--s{train_seed:03d}--{dataset}'
        f'--e{eval_seed:03d}--m{rate_tag}--k{candidate_k:03d}'
        f'--n{validation_batches:04d}')
      artifact_dir = artifact_root / 'runs' / eval_id
      data_config = manifest['datasets'][dataset]['data_config']
      argv = [
        '{python}', 'main.py', 'mode=ppl_eval',
        f'data={data_config}',
        f'seed={train_seed}',
        f'eval.corruption_seed={eval_seed}',
        f'eval.structured_mask_rate={mask_rate:.6f}',
        f'eval.adapter_checkpoint='
        f'${{artifact:{export_id}:adapter.safetensors}}',
        f'eval.adapter_sha256='
        f'${{sha256:{export_id}:adapter.safetensors}}',
        f'eval.adapter_manifest='
        f'${{artifact:{export_id}:adapter-manifest.json}}',
        f'eval.adapter_manifest_sha256='
        f'${{sha256:{export_id}:adapter-manifest.json}}',
        'eval.conditional_records.enabled=true',
        f'eval.conditional_records.schema_version='
        f'{manifest["analysis"]["document_record_schema_version"]}',
        f'eval.conditional_records.protocol_id={protocol_id}',
        f'eval.conditional_records.job_id={eval_id}',
        f'eval.conditional_records.arm={control_name}',
        f'eval.conditional_records.train_seed={train_seed}',
        f'model.structured_decoder.top_k={candidate_k}',
        f'trainer.limit_val_batches={validation_batches}',
        'trainer.num_sanity_val_steps=0',
        f'trainer.precision={evaluation["precision"]}',
        f'loader.eval_global_batch_size={evaluation["batch_size"]}',
        f'loader.eval_batch_size={evaluation["batch_size"]}',
        'loader.num_workers=0',
        'checkpointing.resume_from_ckpt=false',
      ]
      if manifest['analysis']['document_record_schema_version'] == 2:
        support_ks = ','.join(
          str(value) for value in evaluation['candidate_ks'])
        argv.append(
          f'eval.conditional_records.support_candidate_ks=[{support_ks}]')
      argv += _control_overrides(control)
      argv += _hydra_common(manifest, '{artifact_dir}')
      job = _job(
        protocol_id=protocol_id,
        source_manifest_sha256=source_manifest_sha256,
        source_repository_sha=source_repository_sha,
        plan_id=plan_id,
        job_id=eval_id,
        kind='eval',
        artifact_dir=artifact_dir,
        suites=sorted(suite_names),
        dependencies=[export_id],
        identity={
          'control': control_name,
          'train_seed': train_seed,
          'dataset': dataset,
          'corruption_seed': eval_seed,
          'mask_rate': mask_rate,
          'candidate_k': candidate_k,
          'validation_batches': validation_batches,
          'require_disjoint_training_proof': manifest['datasets'][dataset][
            'require_disjoint_training_proof'],
        },
        argv=argv,
        execution_mode='fresh_attempt',
        external_inputs=[{
          'role': 'released_backbone_wrapper',
          'path': manifest['backbone']['wrapper_path'],
          'sha256': manifest['backbone']['wrapper_sha256'],
        }],
        required_outputs=[
          {'name': 'pairing_digest',
           'pattern': 'validation_pairing_digest.json',
           'exactly_one': True},
          {'name': 'conditional_records',
           'pattern': 'conditional_denoising_records.rank0.jsonl',
           'exactly_one': True},
          {'name': 'conditional_record_manifest',
           'pattern': 'conditional_denoising_records.manifest.json',
           'exactly_one': True},
          {'name': 'dataset_provenance',
           'pattern': 'data_provenance/valid-*.json',
           'exactly_one': True},
        ])

    previous = jobs.get(job['job_id'])
    if previous is not None and previous != job:
      raise RuntimeError(f'non-canonical duplicate job: {job["job_id"]}')
    jobs[job['job_id']] = job
  return dict(sorted(jobs.items()))


def _atomic_write(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
  try:
    temporary.write_text(text)
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def write_plan(
    output_dir: Path,
    plan: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    *,
    resume: bool,
) -> None:
  """Atomically write a plan, rejecting drift in a resumed directory."""
  output_dir.mkdir(parents=True, exist_ok=True)
  plan_text = json.dumps(plan, indent=2, sort_keys=True) + '\n'
  expected_files = {
    output_dir / 'compiled-plan.json': plan_text,
    **{
      output_dir / 'jobs' / f'{job_id}.json':
      json.dumps(job, indent=2, sort_keys=True) + '\n'
      for job_id, job in jobs.items()
    },
  }
  existing_entries = list(output_dir.iterdir())
  if existing_entries and not resume:
    raise FileExistsError(
      f'{output_dir} is non-empty; pass --resume only for the identical plan')
  if resume:
    existing_plan = output_dir / 'compiled-plan.json'
    if not existing_plan.is_file():
      raise ValueError(
        f'{output_dir} has no compiled-plan.json; refusing unsafe resume')
    if existing_plan.read_text() != plan_text:
      raise ValueError('compiled plan differs from the existing resumed plan')
    for path, expected in expected_files.items():
      if path.exists() and path.read_text() != expected:
        raise ValueError(f'existing compiled artifact drifted: {path}')
  for path, text in expected_files.items():
    if not path.exists():
      _atomic_write(path, text)


def derive_matrix_plan(
    manifest_path: Path,
    *,
    selected_suites: Sequence[str],
    allowed_artifact_root: Path = DEFAULT_ALLOWED_ARTIFACT_ROOT,
    artifact_root_override: Path | None = None,
    repo_root: Path = REPO_ROOT,
    promotion_evidence: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
  """Derive the canonical matrix plan entirely in memory."""
  manifest_path = manifest_path.expanduser().resolve()
  manifest = load_and_validate_manifest(manifest_path, repo_root=repo_root)
  configured_root = (
    artifact_root_override if artifact_root_override is not None
    else Path(manifest['artifact_root']))
  artifact_root = _safe_root(
    configured_root, allowed_artifact_root, context='artifact root')
  repository = _clean_repository_identity(repo_root)
  source_manifest_sha256 = sha256_file(manifest_path)
  evidence_paths = dict(promotion_evidence or {})
  unknown_evidence = sorted(set(evidence_paths) - set(selected_suites))
  if unknown_evidence:
    raise ValueError(
      f'promotion evidence supplied for unselected suites: '
      f'{unknown_evidence}')
  evidence_records = {}
  for suite_name in sorted(set(selected_suites)):
    source_suite = manifest['suites'][suite_name]['promotion_from']
    evidence_path = evidence_paths.get(suite_name)
    if source_suite is None:
      if evidence_path is not None:
        raise ValueError(
          f'ungated suite {suite_name} does not accept promotion evidence')
      continue
    if evidence_path is None:
      raise ValueError(
        f'suite {suite_name} is gated on {source_suite}; provide '
        f'--promotion-evidence {suite_name} PATH')
    resolved_evidence = evidence_path.expanduser().resolve()
    protocol_id = manifest['protocol_id']
    # Import lazily because both evaluators reuse this module's manifest
    # validator.  Evidence is dispatched by its manifest-declared source
    # suite, never by a schema or policy path supplied inside the evidence.
    causal_template = TRUSTED_CAUSAL_PROMOTION_TEMPLATES.get(
      protocol_id, {}).get(source_suite)
    if causal_template is not None:
      from scripts.evaluate_causal_promotion import (  # pylint: disable=import-outside-toplevel
        verify_causal_compiler_evidence,
      )
      evidence = verify_causal_compiler_evidence(
        _read_yaml_or_json(resolved_evidence),
        evidence_path=resolved_evidence,
        promoted_suite=suite_name,
        manifest_path=manifest_path,
        trusted_template_path=causal_template,
        repo_root=repo_root)
    elif source_suite == 'pilot':
      trusted_policy = TRUSTED_PROMOTION_POLICIES.get(protocol_id)
      if trusted_policy is None:
        raise ValueError(
          f'no trusted pilot promotion policy is registered for '
          f'{protocol_id}')
      from scripts.evaluate_experiment_promotion import (  # pylint: disable=import-outside-toplevel
        verify_compiler_evidence,
      )
      evidence = verify_compiler_evidence(
        _read_yaml_or_json(resolved_evidence),
        evidence_path=resolved_evidence,
        promoted_suite=suite_name,
        manifest_path=manifest_path,
        trusted_policy_path=trusted_policy,
        repo_root=repo_root)
    else:
      trusted_template = TRUSTED_CANDIDATE_K_PROMOTION_TEMPLATES.get(
        protocol_id, {}).get(source_suite)
      if trusted_template is None:
        raise ValueError(
          f'no trusted promotion verifier is registered for protocol '
          f'{protocol_id} and source suite {source_suite}')
      from scripts.evaluate_candidate_k_promotion import (  # pylint: disable=import-outside-toplevel
        verify_candidate_compiler_evidence,
      )
      evidence = verify_candidate_compiler_evidence(
        _read_yaml_or_json(resolved_evidence),
        evidence_path=resolved_evidence,
        promoted_suite=suite_name,
        manifest_path=manifest_path,
        trusted_template_path=trusted_template,
        repo_root=repo_root)
    if evidence['source_suite'] != source_suite:
      raise ValueError(
        f'promotion evidence source suite {evidence["source_suite"]!r}; '
        f'expected {source_suite!r}')
    evidence_records[suite_name] = {
      'path': str(resolved_evidence),
      'sha256': sha256_file(resolved_evidence),
      'source_suite': source_suite,
      'route_name': evidence['route_name'],
      'canonical_decision_sha256': evidence['commitments'][
        'canonical_decision_sha256'],
      'source_compiled_plan_sha256': evidence['commitments'][
        'source_compiled_plan_sha256'],
    }
  plan_identity = {
    'protocol_id': manifest['protocol_id'],
    'source_manifest_sha256': source_manifest_sha256,
    'repository': repository,
    'artifact_root': str(artifact_root),
    'selected_suites': sorted(set(selected_suites)),
    'promotion_evidence': evidence_records,
  }
  plan_id = _sha256_text(_canonical_json(plan_identity))
  jobs = build_jobs(
    manifest,
    selected_suites=plan_identity['selected_suites'],
    artifact_root=artifact_root,
    source_manifest_sha256=source_manifest_sha256,
    source_repository_sha=repository['sha'],
    plan_id=plan_id)
  counts: dict[str, int] = {}
  for job in jobs.values():
    counts[job['kind']] = counts.get(job['kind'], 0) + 1
  plan = {
    'schema_version': PLAN_SCHEMA_VERSION,
    **plan_identity,
    'plan_id': plan_id,
    'manifest_protocol_status': manifest['protocol_status'],
    'scientific_scope': manifest['scientific_scope'],
    'job_counts': dict(sorted(counts.items())),
    'num_jobs': len(jobs),
    'job_ids': list(jobs),
    'job_spec_sha256': {
      job_id: _sha256_text(_canonical_json(job))
      for job_id, job in jobs.items()},
  }
  return plan, jobs, artifact_root


def compile_matrix(
    manifest_path: Path,
    *,
    selected_suites: Sequence[str],
    allowed_artifact_root: Path = DEFAULT_ALLOWED_ARTIFACT_ROOT,
    artifact_root_override: Path | None = None,
    output_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
    promotion_evidence: Mapping[str, Path] | None = None,
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
  plan, jobs, artifact_root = derive_matrix_plan(
    manifest_path,
    selected_suites=selected_suites,
    allowed_artifact_root=allowed_artifact_root,
    artifact_root_override=artifact_root_override,
    repo_root=repo_root,
    promotion_evidence=promotion_evidence)
  resolved_output = (
    output_dir.expanduser().resolve() if output_dir is not None
    else artifact_root / 'plans' / '--'.join(sorted(set(selected_suites))))
  try:
    resolved_output.relative_to(artifact_root)
  except ValueError as error:
    raise ValueError(
      f'output directory {resolved_output} must be within artifact root '
      f'{artifact_root}') from error
  if resolved_output == artifact_root:
    raise ValueError('output directory must not equal artifact root')
  write_plan(resolved_output, plan, jobs, resume=resume)
  return plan, jobs, resolved_output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument(
    '--suite', action='append', dest='suites',
    help='suite to compile; repeat as needed (default: pilot)')
  parser.add_argument(
    '--allowed-artifact-root', type=Path,
    default=DEFAULT_ALLOWED_ARTIFACT_ROOT)
  parser.add_argument('--artifact-root', type=Path)
  parser.add_argument('--output-dir', type=Path)
  parser.add_argument(
    '--promotion-evidence', action='append', nargs=2, default=[],
    metavar=('SUITE', 'PATH'),
    help=(
      'required for a gated suite; PATH must be canonical revision-bound '
      'evidence emitted by the registered promotion evaluator'))
  parser.add_argument('--resume', action='store_true')
  return parser.parse_args(argv)


def main() -> int:
  args = _parse_args()
  evidence = {}
  for suite_name, path_text in args.promotion_evidence:
    if suite_name in evidence:
      raise ValueError(
        f'duplicate promotion evidence for suite {suite_name}')
    evidence[suite_name] = Path(path_text)
  plan, _, output = compile_matrix(
    args.manifest,
    selected_suites=args.suites or ['pilot'],
    allowed_artifact_root=args.allowed_artifact_root,
    artifact_root_override=args.artifact_root,
    output_dir=args.output_dir,
    promotion_evidence=evidence,
    resume=args.resume)
  print(json.dumps({
    'plan_id': plan['plan_id'],
    'output_dir': str(output),
    'job_counts': plan['job_counts'],
    'num_jobs': plan['num_jobs'],
  }, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
