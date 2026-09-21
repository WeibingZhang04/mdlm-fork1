#!/usr/bin/env python3
"""Build the predeclared causal-evidence analysis from schema-v2 plans."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from data_provenance import canonical_sha256  # noqa: E402
from evaluation.hierarchical_causal_statistics import (  # noqa: E402
  ContrastTerm,
  aggregate_candidate_support,
  aggregate_causal_contrast,
  aggregate_technical_diagnostics,
  aggregate_topology_permutation_diagnostic,
)
from scripts.aggregate_hierarchical_document_eval import (  # noqa: E402
  load_plan_records,
)
from scripts.compile_experiment_matrix import DEFAULT_MANIFEST  # noqa: E402


SOURCE_COMPARISONS = {
  'contextual_vs_static': {'dynamic_dynamic', 'static_static'},
  'contextual_topology_gain': {'dynamic_dynamic', 'fixed_dynamic'},
  'contextual_factor_gain': {'dynamic_dynamic', 'dynamic_fixed'},
}


def _contrasts() -> list[tuple[str, tuple[ContrastTerm, ...]]]:
  joint = 'nll_sum'
  return [
    ('factorized_backbone_vs_contextual_joint', (
      ContrastTerm('dynamic_dynamic', 'factorized_backbone_nll_sum', 1.0),
      ContrastTerm('dynamic_dynamic', joint, -1.0))),
    ('singleton_product_vs_contextual_joint', (
      ContrastTerm('dynamic_dynamic', 'structured_marginal_nll_sum', 1.0),
      ContrastTerm('dynamic_dynamic', joint, -1.0))),
    ('parameter_matched_no_edge_vs_contextual_joint', (
      ContrastTerm(
        'dynamic_dynamic', 'parameter_matched_no_edge_nll_sum', 1.0),
      ContrastTerm('dynamic_dynamic', joint, -1.0))),
    ('matched_permuted_topology_vs_contextual_joint', (
      ContrastTerm(
        'dynamic_dynamic', 'matched_permuted_topology_nll_sum', 1.0),
      ContrastTerm('dynamic_dynamic', joint, -1.0))),
    ('contextual_joint_vs_static_joint', (
      ContrastTerm('static_static', joint, 1.0),
      ContrastTerm('dynamic_dynamic', joint, -1.0))),
    ('contextual_topology_gain_dynamic_factors', (
      ContrastTerm('fixed_dynamic', joint, 1.0),
      ContrastTerm('dynamic_dynamic', joint, -1.0))),
    ('contextual_topology_gain_fixed_factors', (
      ContrastTerm('static_static', joint, 1.0),
      ContrastTerm('dynamic_fixed', joint, -1.0))),
    ('contextual_factor_gain_dynamic_topology', (
      ContrastTerm('dynamic_fixed', joint, 1.0),
      ContrastTerm('dynamic_dynamic', joint, -1.0))),
    ('contextual_factor_gain_fixed_topology', (
      ContrastTerm('static_static', joint, 1.0),
      ContrastTerm('fixed_dynamic', joint, -1.0))),
    ('topology_by_factor_interaction', (
      ContrastTerm('fixed_dynamic', joint, 1.0),
      ContrastTerm('dynamic_dynamic', joint, -1.0),
      ContrastTerm('static_static', joint, -1.0),
      ContrastTerm('dynamic_fixed', joint, 1.0))),
  ]


def _deduplicate_records(records):
  keyed = {}
  for row in records:
    key = (
      row['arm'], row['train_seed'], row['corruption_seed'], row['dataset'],
      row['dataset_revision'], row['mask_rate'], row['candidate_k'],
      row['document_id'], row['document_index'], row['document_sha256'],
      row['chunk_index'])
    previous = keyed.get(key)
    if previous is not None and previous != row:
      raise ValueError(f'conflicting duplicate source row {key}')
    keyed[key] = row
  return list(keyed.values())


def _merge_source_integrity(
    contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
  """Commit the union of all jobs used by the three causal source views."""
  first_context = next(iter(contexts.values()))
  plan = first_context['plan']
  fixed_fields = {
    'schema_version': 1,
    'source_compiled_plan_path': str(
      (Path(first_context['source_integrity'][
        'source_compiled_plan_path']).expanduser().resolve())),
    'source_compiled_plan_sha256': first_context['plan_sha256'],
    'source_plan_id': plan['plan_id'],
    'source_manifest_sha256': plan['source_manifest_sha256'],
    'source_repository_sha': plan['repository']['sha'],
    'source_repository_clean': plan['repository']['dirty'] is False,
  }
  jobs = {}
  for comparison_name, context in contexts.items():
    if context['plan'] != plan or context['plan_sha256'] != \
        first_context['plan_sha256']:
      raise ValueError(
        f'causal source view {comparison_name} uses a different plan')
    integrity = context['source_integrity']
    for field, expected in fixed_fields.items():
      observed = integrity[field]
      if field == 'source_compiled_plan_path':
        observed = str(Path(observed).expanduser().resolve())
      if observed != expected:
        raise ValueError(
          f'causal source view {comparison_name} has conflicting {field}')
    if set(integrity['jobs']) != set(integrity['validated_job_ids']):
      raise ValueError(
        f'causal source view {comparison_name} has inconsistent job IDs')
    for job_id in integrity['validated_job_ids']:
      job = dict(integrity['jobs'][job_id])
      previous = jobs.get(job_id)
      if previous is not None and previous != job:
        raise ValueError(
          f'causal source views disagree about job {job_id}')
      jobs[job_id] = job
  expected_job_ids = sorted(plan['job_ids'])
  if sorted(jobs) != expected_job_ids:
    missing = sorted(set(expected_job_ids) - set(jobs))
    extra = sorted(set(jobs) - set(expected_job_ids))
    raise ValueError(
      'causal source views do not cover the complete compiled plan: '
      f'missing={missing}, extra={extra}')
  body = {
    **fixed_fields,
    'validated_job_ids': expected_job_ids,
    'jobs': {job_id: jobs[job_id] for job_id in expected_job_ids},
  }
  return {**body, 'commitment_sha256': canonical_sha256(body)}


def build_analysis(
    *,
    plan_dir: Path,
    manifest_path: Path,
    suite_name: str,
    num_resamples: int | None = None,
    rng_seed: int | None = None,
    confidence_level: float | None = None,
    timestamp_utc: str | None = None,
    technical_gates: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  if technical_gates is None:
    # Import lazily to avoid a module cycle: the finalizer imports the compiler,
    # while the promotion evaluator imports this authoritative aggregator.
    from scripts.finalize_causal_smoke_policy import (
      DEFAULT_TEMPLATE,
      _read_mapping,
      _validate_template,
    )
    template = _validate_template(
      _read_mapping(DEFAULT_TEMPLATE, context='causal policy template'),
      manifest_path=manifest_path,
      repo_root=repo_root)
    technical_gates = template['technical_gates']
  technical_gates = dict(technical_gates)
  records = []
  contexts = {}
  # Load the primary pair first, then only those extra views whose arms are
  # present in the selected suite. Every current causal suite contains the
  # complete 2x2; retaining the subset logic makes incomplete legacy plans
  # explicit rather than silently inventing missing factorial cells.
  comparison_names = ['contextual_vs_static']
  primary_records, primary_context = load_plan_records(
    plan_dir,
    manifest_path=manifest_path,
    suite_name=suite_name,
    comparison_name='contextual_vs_static',
    repo_root=repo_root)
  available_arms = set(primary_context['suite']['controls'])
  if (primary_context['manifest']['analysis'][
      'document_record_schema_version'] != 2):
    raise ValueError('causal analysis requires document record schema v2')
  records.extend(primary_records)
  contexts['contextual_vs_static'] = primary_context
  comparison_names.extend(
    name for name, arms in SOURCE_COMPARISONS.items()
    if name != 'contextual_vs_static' and arms.issubset(available_arms))
  for comparison_name in comparison_names[1:]:
    comparison_records, context = load_plan_records(
      plan_dir,
      manifest_path=manifest_path,
      suite_name=suite_name,
      comparison_name=comparison_name,
      repo_root=repo_root)
    if context['manifest']['analysis']['document_record_schema_version'] != 2:
      raise ValueError('causal analysis requires document record schema v2')
    records.extend(comparison_records)
    contexts[comparison_name] = context
  records = _deduplicate_records(records)
  analysis_cfg = next(iter(contexts.values()))['manifest']['analysis']
  num_resamples = (
    analysis_cfg['bootstrap_resamples']
    if num_resamples is None else num_resamples)
  rng_seed = analysis_cfg['bootstrap_seed'] if rng_seed is None else rng_seed
  confidence_level = (
    analysis_cfg['confidence_level']
    if confidence_level is None else confidence_level)

  contrast_results = {}
  for index, (name, terms) in enumerate(_contrasts()):
    if not {term.arm for term in terms}.issubset(available_arms):
      continue
    contrast_results[name] = aggregate_causal_contrast(
      records,
      name=name,
      terms=terms,
      num_resamples=num_resamples,
      rng_seed=rng_seed + index * 101,
      confidence_level=confidence_level)
  support = aggregate_candidate_support(
    records,
    arm='dynamic_dynamic',
    num_resamples=num_resamples,
    rng_seed=rng_seed + 10_000,
    confidence_level=confidence_level)
  topology_permutation = aggregate_topology_permutation_diagnostic(
    records,
    arm='dynamic_dynamic',
    minimum_pooled_changed_edge_fraction=(
      technical_gates['minimum_pooled_changed_edge_fraction']),
    minimum_condition_changed_edge_fraction=(
      technical_gates['minimum_condition_changed_edge_fraction']))
  technical_diagnostics = aggregate_technical_diagnostics(
    records,
    expected_arms=list(primary_context['suite']['controls']),
    expected_train_seeds=list(primary_context['suite']['train_seeds']),
    expected_support_ks=list(
      primary_context['manifest']['evaluation']['candidate_ks']),
    no_edge_absolute_tolerance=(
      technical_gates['no_edge_absolute_tolerance']),
    support_monotonicity_tolerance=(
      technical_gates['support_monotonicity_absolute_tolerance']))

  source_views = {}
  for comparison_name, context in contexts.items():
    source_views[comparison_name] = {
      'compiled_plan_sha256': context['plan_sha256'],
      'source_manifest_sha256': context['plan']['source_manifest_sha256'],
      'source_repository_sha': context['plan']['repository']['sha'],
      'source_integrity_commitment_sha256': (
        context['source_integrity']['commitment_sha256']),
    }
  source_integrity = _merge_source_integrity(contexts)
  plan = primary_context['plan']
  payload = {
    'schema_version': 2,
    'artifact': 'contextual_forest_causal_denoising_analysis',
    'created_utc': timestamp_utc or dt.datetime.now(
      dt.timezone.utc).isoformat(),
    'protocol_id': next(iter(contexts.values()))['manifest']['protocol_id'],
    'suite': suite_name,
    'objective': 'paired_conditional_denoising_nll_per_masked_token',
    'scope_note': (
      'Conditional denoising only; no diffusion ELBO, likelihood, '
      'perplexity, or generation-quality quantity is inferred.'),
    'source_views': source_views,
    'contrasts': contrast_results,
    'candidate_support': support,
    'topology_permutation_diagnostic': topology_permutation,
    'technical_diagnostics': technical_diagnostics,
    'compiled_plan': {
      'plan_id': plan['plan_id'],
      'source_manifest_sha256': plan['source_manifest_sha256'],
      'source_compiled_plan_sha256': primary_context['plan_sha256'],
      'source_repository_sha': plan['repository']['sha'],
      'source_repository_clean': plan['repository']['dirty'] is False,
      'job_artifact_commitment_sha256': source_integrity[
        'commitment_sha256'],
    },
    'source_integrity': source_integrity,
  }
  payload['analysis_sha256'] = canonical_sha256(payload)
  return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan-dir', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument('--suite', required=True)
  parser.add_argument('--bootstrap-resamples', type=int)
  parser.add_argument('--bootstrap-seed', type=int)
  parser.add_argument('--confidence-level', type=float)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  result = build_analysis(
    plan_dir=args.plan_dir,
    manifest_path=args.manifest,
    suite_name=args.suite,
    num_resamples=args.bootstrap_resamples,
    rng_seed=args.bootstrap_seed,
    confidence_level=args.confidence_level)
  output = args.output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.exists():
    raise FileExistsError(output)
  temporary = output.with_name(f'.{output.name}.tmp')
  temporary.write_text(json.dumps(
    result, indent=2, sort_keys=True, allow_nan=False) + '\n')
  temporary.replace(output)
  print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
