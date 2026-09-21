"""Fail-closed paired dynamic-vs-static generation comparison.

Both adapter arms are independently reconstructed from their raw atomic shard
directories.  The comparison accepts only the frozen paper-scale infilling
matrix and permits cross-arm drift solely in the structured adapter identity
and the exact Hydra fields implied by the dynamic/static controls.
"""

from __future__ import annotations

import copy
import datetime as dt
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from evaluation.adapter_pair_origin import (
  bind_generation_arm_to_adapter_origin_evidence,
)
from evaluation.generation_metrics import REFERENCE_LM_SEQUENCE_POLICY
from evaluation.generation_protocol import validate_generation_protocol
from evaluation.generation_shard_aggregation import (
  _assert_equivalent,
  _comparison,
  canonical_sha256,
  load_generation_shard,
)
from evaluation.prompt_provenance import validate_prompt_bundle


PROTOCOL_ID = 'contextual-forest-generation-paper-v1'
REFERENCE_LM = {
  'model_name_or_path': 'gpt2-large',
  'revision': '32b71b12589c2f8d625668d2335a01cac3249519',
  'sequence_policy': REFERENCE_LM_SEQUENCE_POLICY,
}
DATASET_PROTOCOLS = {
  'wikitext103-pinned': {
    'data_config': 'eval_wikitext103_pinned',
    'dataset_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
    'num_prompts': 197,
    'global_num_samples': 788,
    'base_seed': 91001,
  },
  'scientific-papers-arxiv-pinned': {
    'data_config': 'eval_scientific_papers_arxiv_pinned',
    'dataset_revision': '0c23eb103b9f78874e0ac93d01bbd935fb8f59b1',
    'num_prompts': 256,
    'global_num_samples': 1024,
    'base_seed': 92001,
  },
  'scientific-papers-pubmed-pinned': {
    'data_config': 'eval_scientific_papers_pubmed_pinned',
    'dataset_revision': '0c23eb103b9f78874e0ac93d01bbd935fb8f59b1',
    'num_prompts': 256,
    'global_num_samples': 1024,
    'base_seed': 93001,
  },
}
TOKENIZER_REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
NFE_BUDGETS = [8, 16, 32, 64]
DYNAMIC_MODES = ['factorized', 'structured_marginal', 'structured_joint']
STATIC_MODES = ['structured_joint']
NUM_SHARDS = 16
BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
SPAN_LENGTH = 32
PROMPT_SELECTION_SEED = 31001
PAPER_ENDPOINTS = {
  'repetition_rate_1gram', 'repetition_rate_2gram',
  'repetition_rate_4gram', 'reference_token_accuracy',
  'reference_exact_match', 'reference_lm_mean_nll_nats',
}


def _load_yaml(path: Path, *, context: str) -> dict[str, Any]:
  try:
    payload = yaml.safe_load(path.read_text())
  except yaml.YAMLError as error:
    raise ValueError(f'invalid YAML in {path}: {error}') from error
  if not isinstance(payload, dict):
    raise TypeError(f'{context} must be a YAML mapping')
  return payload


def _path_value(payload: Mapping[str, Any], path: Sequence[str], *, context: str):
  current: Any = payload
  for field in path:
    if not isinstance(current, Mapping) or field not in current:
      raise ValueError(f'{context} is missing {".".join(path)}')
    current = current[field]
  return current


def _remove_path(payload: dict[str, Any], path: Sequence[str], *, context: str):
  current: Any = payload
  for field in path[:-1]:
    if not isinstance(current, dict) or field not in current:
      raise ValueError(f'{context} is missing {".".join(path)}')
    current = current[field]
  if not isinstance(current, dict) or path[-1] not in current:
    raise ValueError(f'{context} is missing {".".join(path)}')
  return current.pop(path[-1])


def _adapter_semantics_without_control(identity: Mapping[str, Any]):
  result = copy.deepcopy(dict(identity))
  for field in (
      'control_identity', 'topology_mode', 'factor_mode', 'topology_weight'):
    result.pop(field)
  return result


def _revalidate_adapter_origin(
    manifest: Mapping[str, Any], *, expected_control: str,
) -> dict[str, Any]:
  binding = manifest.get('adapter_origin_evidence')
  if not isinstance(binding, Mapping):
    raise TypeError('paper-scale generation requires adapter-origin evidence')
  artifact = manifest['artifacts']['structured_adapter']
  observed = bind_generation_arm_to_adapter_origin_evidence(
    Path(binding['evidence_file']['path']),
    expected_evidence_sha256=binding['evidence_file']['sha256'],
    arm=expected_control,
    adapter_path=Path(artifact['path']),
    expected_adapter_sha256=artifact['sha256'],
    adapter_manifest_path=Path(artifact['manifest_path']),
    expected_adapter_manifest_sha256=artifact['manifest_sha256'],
    structured_decoder_identity=artifact['semantic_identity'],
  )
  _assert_equivalent(
    observed, binding,
    context=f'{expected_control}.replayed_adapter_origin_evidence')
  return observed


def _validate_arm_config(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_control: str,
) -> dict[str, Any]:
  adapter = manifest['_artifact_identity']['structured_adapter']
  semantic = adapter['semantic_identity']
  if semantic['control_identity'] != expected_control:
    raise ValueError(
      f'{expected_control} arm loaded adapter control '
      f'{semantic["control_identity"]!r}')
  actual_artifact = manifest['artifacts']['structured_adapter']
  expected_values = {
    ('eval', 'adapter_checkpoint'): actual_artifact['path'],
    ('eval', 'adapter_sha256'): actual_artifact['sha256'],
    ('eval', 'adapter_manifest'): actual_artifact['manifest_path'],
    ('eval', 'adapter_manifest_sha256'): actual_artifact['manifest_sha256'],
    ('model', 'structured_decoder', 'topology_mode'):
      semantic['topology_mode'],
    ('model', 'structured_decoder', 'factor_mode'):
      semantic['factor_mode'],
    ('model', 'structured_decoder', 'training', 'topology_weight'):
      semantic['topology_weight'],
    ('model', 'structured_decoder', 'top_k'):
      semantic['candidate_top_k'],
  }
  for path, expected in expected_values.items():
    observed = _path_value(config, path, context=f'{expected_control} config')
    if observed != expected or type(observed) is not type(expected):
      raise ValueError(
        f'{expected_control} config {".".join(path)} differs from '
        'the validated adapter identity')
  normalized = copy.deepcopy(dict(config))
  for path in (
      ('eval', 'adapter_checkpoint'),
      ('eval', 'adapter_sha256'),
      ('eval', 'adapter_manifest'),
      ('eval', 'adapter_manifest_sha256'),
      ('model', 'structured_decoder', 'topology_mode'),
      ('model', 'structured_decoder', 'factor_mode'),
      ('model', 'structured_decoder', 'training', 'topology_weight'),
  ):
    _remove_path(normalized, path, context=f'{expected_control} config')
  return normalized


def _load_selected_records(
    shard_paths: Sequence[Path],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  shards = [load_generation_shard(Path(path)) for path in shard_paths]
  shards.sort(key=lambda shard: shard['manifest']['pairing']['shard_index'])
  records = []
  for shard in shards:
    budgets = shard['manifest']['matrix']['nfe_budgets']
    for budget in budgets:
      records.extend(shard['groups'][(mode, budget)])
  return records, shards


def _selected_records_from_loaded(
    shards: Sequence[Mapping[str, Any]], *, mode: str,
) -> list[dict[str, Any]]:
  records = []
  for shard in shards:
    budgets = shard['manifest']['matrix']['nfe_budgets']
    for budget in budgets:
      records.extend(shard['groups'][(mode, budget)])
  return records


def _expected_union_input_shards(
    shards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
  """Reconstruct the exact raw-shard commitments stored by aggregation."""
  return [
    {
      'shard_index': shard['manifest']['pairing']['shard_index'],
      'manifest_path': str(shard['manifest_path']),
      'manifest_sha256': shard['manifest_sha256'],
      'samples_sha256': shard['manifest']['outputs']['samples_jsonl']['sha256'],
      'num_records': shard['manifest']['matrix']['num_output_records'],
    }
    for shard in sorted(
      shards, key=lambda item: item['manifest']['pairing']['shard_index'])
  ]


def _records_at_budget(
    records: Sequence[Mapping[str, Any]], budget: int,
) -> list[Mapping[str, Any]]:
  return sorted(
    (record for record in records
     if record['requested_nfe_budget'] == budget),
    key=lambda record: record['sample_index'])


def _timing_group(union: Mapping[str, Any], *, mode: str, budget: int):
  matches = [
    group for group in union['groups']
    if group['sampling_mode'] == mode
    and group['requested_nfe_budget'] == budget
  ]
  if len(matches) != 1:
    raise ValueError(f'expected exactly one timing group for {mode}/{budget}')
  return matches[0]['timing']


def _validate_paper_pairing(
    records: Sequence[Mapping[str, Any]],
    *,
    num_prompts: int,
    base_seed: int,
) -> None:
  if len(records) != num_prompts * 4:
    raise ValueError('paper protocol requires exactly four draws per prompt')
  prompt_cycle = [record['prompt_id'] for record in records[:num_prompts]]
  if len(set(prompt_cycle)) != num_prompts:
    raise ValueError('first paper-protocol prompt cycle is not unique')
  counts = {prompt_id: 0 for prompt_id in prompt_cycle}
  for sample_index, record in enumerate(records):
    prompt_id = prompt_cycle[sample_index % num_prompts]
    replicate = sample_index // num_prompts
    expected_key = f'{prompt_id}/replicate-{replicate:04d}'
    if (record['sample_index'] != sample_index
        or record['prompt_id'] != prompt_id
        or record['pair_key'] != expected_key
        or record['pair_seed'] != base_seed + sample_index):
      raise ValueError(
        'paper prompt/draw order differs from deterministic four-draw cycle')
    counts[prompt_id] += 1
  if set(counts.values()) != {4}:
    raise ValueError('paper protocol requires exactly four draws per prompt')


def _require_paper_endpoints(
    records: Sequence[Mapping[str, Any]], *, context: str,
) -> None:
  for record in records:
    endpoints = _record_endpoint_names(record)
    if endpoints != PAPER_ENDPOINTS:
      raise ValueError(
        f'{context} sample {record["sample_index"]} endpoint availability '
        'differs from the frozen paper protocol')


def _record_endpoint_names(record: Mapping[str, Any]) -> set[str]:
  result = {
    f'repetition_rate_{n}gram' for n in (1, 2, 4)}
  if record.get('reference_token_ids') is not None:
    result.update({'reference_token_accuracy', 'reference_exact_match'})
  reference_lm = record.get('reference_lm')
  if (isinstance(reference_lm, Mapping)
      and reference_lm.get('mean_nll_nats') is not None):
    result.add('reference_lm_mean_nll_nats')
  return result


def compare_generation_adapters(
    baseline_shards: Sequence[Path],
    treatment_shards: Sequence[Path],
    *,
    baseline_union: Mapping[str, Any],
    treatment_union: Mapping[str, Any],
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 94001,
    bootstrap_confidence: float = 0.95,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
  """Verify and compare arms bound to their once-aggregated union payloads."""
  created = timestamp_utc or dt.datetime.now(dt.timezone.utc).isoformat()
  if not isinstance(baseline_union, Mapping):
    raise TypeError('baseline_union must be a verified union payload')
  if not isinstance(treatment_union, Mapping):
    raise TypeError('treatment_union must be a verified union payload')
  for name, union in (
      ('baseline_union', baseline_union),
      ('treatment_union', treatment_union)):
    if (union.get('schema_version') != 1
        or union.get('artifact') != 'verified_generation_shard_union'):
      raise ValueError(f'{name} has an unsupported union identity')

  baseline_records, baseline_loaded = _load_selected_records(
    baseline_shards, mode='structured_joint')
  treatment_records, treatment_loaded = _load_selected_records(
    treatment_shards, mode='structured_joint')
  treatment_factorized_records = _selected_records_from_loaded(
    treatment_loaded, mode='factorized')
  treatment_marginal_records = _selected_records_from_loaded(
    treatment_loaded, mode='structured_marginal')
  _assert_equivalent(
    baseline_union.get('input_shards'),
    _expected_union_input_shards(baseline_loaded),
    context='baseline_union.input_shards')
  _assert_equivalent(
    treatment_union.get('input_shards'),
    _expected_union_input_shards(treatment_loaded),
    context='treatment_union.input_shards')
  baseline_manifest = baseline_loaded[0]['manifest']
  treatment_manifest = treatment_loaded[0]['manifest']

  baseline_identity = baseline_union['identity']
  treatment_identity = treatment_union['identity']
  for field in (
      'repository', 'prompts', 'reference_lm', 'runtime',
      'global_pairing_digest', 'base_seed', 'batch_size',
      'global_num_samples', 'num_shards', 'sequence_length', 'nfe_budgets'):
    _assert_equivalent(
      treatment_identity[field], baseline_identity[field],
      context=f'cross_adapter_identity.{field}')
  _assert_equivalent(
    treatment_identity['artifacts']['backbone_checkpoint'],
    baseline_identity['artifacts']['backbone_checkpoint'],
    context='cross_adapter_identity.backbone_checkpoint')
  _assert_equivalent(
    treatment_union['timing_policy']['hardware_identity'],
    baseline_union['timing_policy']['hardware_identity'],
    context='cross_adapter_identity.timing_hardware')
  if baseline_identity['sampling_modes'] != STATIC_MODES:
    raise ValueError('static arm does not contain the frozen sampling-mode grid')
  if treatment_identity['sampling_modes'] != DYNAMIC_MODES:
    raise ValueError('dynamic arm does not contain the frozen sampling-mode grid')
  if baseline_identity['nfe_budgets'] != NFE_BUDGETS:
    raise ValueError('adapter arms do not contain the frozen NFE grid')

  baseline_adapter = baseline_identity['artifacts']['structured_adapter']
  treatment_adapter = treatment_identity['artifacts']['structured_adapter']
  baseline_semantic = baseline_adapter['semantic_identity']
  treatment_semantic = treatment_adapter['semantic_identity']
  if baseline_semantic['control_identity'] != 'static_static':
    raise ValueError('baseline adapter must be static_static')
  if treatment_semantic['control_identity'] != 'dynamic_dynamic':
    raise ValueError('treatment adapter must be dynamic_dynamic')
  _assert_equivalent(
    _adapter_semantics_without_control(treatment_semantic),
    _adapter_semantics_without_control(baseline_semantic),
    context='cross_adapter_identity.shared_adapter_semantics')
  if baseline_adapter['sha256'] == treatment_adapter['sha256']:
    raise ValueError('dynamic and static controls unexpectedly use one adapter')

  baseline_origin = _revalidate_adapter_origin(
    baseline_manifest, expected_control='static_static')
  treatment_origin = _revalidate_adapter_origin(
    treatment_manifest, expected_control='dynamic_dynamic')
  for field in ('evidence_file', 'source'):
    _assert_equivalent(
      treatment_origin[field], baseline_origin[field],
      context=f'cross_adapter_identity.adapter_origin.{field}')
  if (baseline_origin['arm'] != 'static_static'
      or treatment_origin['arm'] != 'dynamic_dynamic'):
    raise ValueError('adapter-origin evidence names the wrong paired arms')
  if (baseline_origin['source']['candidate_k'] !=
      baseline_semantic['candidate_top_k']):
    raise ValueError('adapter-origin plan K differs from generation adapter K')
  _assert_equivalent(
    treatment_origin['adapter']['released_backbone'],
    baseline_origin['adapter']['released_backbone'],
    context='cross_adapter_identity.adapter_origin.released_backbone')

  baseline_config = _load_yaml(
    baseline_loaded[0]['config_path'], context='static resolved config')
  treatment_config = _load_yaml(
    treatment_loaded[0]['config_path'], context='dynamic resolved config')
  normalized_baseline = _validate_arm_config(
    baseline_config, baseline_manifest, expected_control='static_static')
  normalized_treatment = _validate_arm_config(
    treatment_config, treatment_manifest, expected_control='dynamic_dynamic')
  _assert_equivalent(
    normalized_treatment, normalized_baseline,
    context='cross_adapter_identity.resolved_config_after_allowed_fields')

  baseline_protocol = validate_generation_protocol(
    baseline_loaded[0]['config_path'],
    baseline_manifest,
    candidate_top_k=baseline_semantic['candidate_top_k'],
    expected_control='static_static')
  treatment_protocol = validate_generation_protocol(
    treatment_loaded[0]['config_path'],
    treatment_manifest,
    candidate_top_k=treatment_semantic['candidate_top_k'],
    expected_control='dynamic_dynamic')
  expected_treatment_protocol = dict(baseline_protocol)
  expected_treatment_protocol['control_identity'] = 'dynamic_dynamic'
  _assert_equivalent(
    treatment_protocol,
    expected_treatment_protocol,
    context='cross_adapter_identity.frozen_generation_protocol')

  prompt_identity = baseline_identity['prompts']
  if prompt_identity['source'] != 'jsonl':
    raise ValueError('paper-scale adapter comparison requires infilling prompts')
  bundle = prompt_identity['bundle_identity']
  dataset_id = bundle['data_config']['logical_validation_dataset']
  protocol = DATASET_PROTOCOLS.get(dataset_id)
  if protocol is None:
    raise ValueError(f'dataset {dataset_id!r} is outside the frozen protocol')
  observed_protocol = {
    'data_config': bundle['data_config']['name'],
    'dataset_revision': bundle['data_config']['dataset_revision'],
    'num_prompts': bundle['output']['num_prompts'],
    'global_num_samples': baseline_identity['global_num_samples'],
    'base_seed': baseline_identity['base_seed'],
  }
  _assert_equivalent(
    observed_protocol, protocol, context='paper_protocol.dataset')
  generation_git_sha = baseline_identity['repository']['git_sha']
  if bundle['builder_git_sha'] != generation_git_sha:
    raise ValueError(
      'prompt builder and generation runner must use the same clean commit')
  raw_prompt_identity = baseline_manifest['prompts']
  validated_bundle = validate_prompt_bundle(
    Path(raw_prompt_identity['path']),
    Path(raw_prompt_identity['manifest_path']),
    expected_manifest_sha256=raw_prompt_identity['manifest_sha256'],
    expected_data_config=protocol['data_config'],
    expected_sequence_length=SEQUENCE_LENGTH)
  _assert_equivalent(
    validated_bundle, bundle, context='paper_protocol.prompt_bundle')
  if (bundle['data_config']['tokenizer_revision'] != TOKENIZER_REVISION
      or bundle['policy']['sequence_length'] != SEQUENCE_LENGTH
      or bundle['policy']['span_length'] != SPAN_LENGTH
      or bundle['policy']['selection_seed'] != PROMPT_SELECTION_SEED
      or baseline_identity['batch_size'] != BATCH_SIZE
      or baseline_identity['num_shards'] != NUM_SHARDS
      or any(
        baseline_identity['reference_lm'].get(field) != expected
        for field, expected in REFERENCE_LM.items())):
    raise ValueError('generation inputs differ from the frozen paper protocol')

  baseline_pair_records = _records_at_budget(
    baseline_records, NFE_BUDGETS[0])
  treatment_pair_records = _records_at_budget(
    treatment_records, NFE_BUDGETS[0])
  _validate_paper_pairing(
    baseline_pair_records,
    num_prompts=protocol['num_prompts'], base_seed=protocol['base_seed'])
  _validate_paper_pairing(
    treatment_pair_records,
    num_prompts=protocol['num_prompts'], base_seed=protocol['base_seed'])
  _require_paper_endpoints(
    baseline_records, context='static paper arm')
  _require_paper_endpoints(
    treatment_records, context='dynamic paper arm')
  _require_paper_endpoints(
    treatment_factorized_records, context='factorized backbone arm')
  _require_paper_endpoints(
    treatment_marginal_records,
    context='independent structured-marginal arm')

  comparisons = []
  timing = []
  for offset, budget in enumerate(NFE_BUDGETS):
    static_joint_group = _records_at_budget(baseline_records, budget)
    dynamic_joint_group = _records_at_budget(treatment_records, budget)
    factorized_group = _records_at_budget(
      treatment_factorized_records, budget)
    marginal_group = _records_at_budget(
      treatment_marginal_records, budget)
    comparison_specs = (
      (
        factorized_group, marginal_group,
        'structured_marginals_vs_factorized_backbone_at_fixed_nfe',
        'dynamic_dynamic', 'dynamic_dynamic',
        'same backbone and adapter; structured inference is enabled only for '
        'the treatment'),
      (
        marginal_group, dynamic_joint_group,
        'joint_vs_independent_structured_marginals_at_fixed_nfe',
        'dynamic_dynamic', 'dynamic_dynamic',
        'same checkpoint, forest potentials, and exact conditional node '
        'marginals at each matched denoising state; only independent versus '
        'joint forest draws differ'),
      (
        factorized_group, dynamic_joint_group,
        'dynamic_joint_vs_factorized_backbone_at_fixed_nfe',
        'dynamic_dynamic', 'dynamic_dynamic',
        'same released backbone and adapter artifact; treatment activates '
        'contextual structured inference and joint sampling'),
      (
        static_joint_group, dynamic_joint_group,
        'dynamic_adapter_vs_static_adapter_at_fixed_nfe',
        'static_static', 'dynamic_dynamic',
        'paired training plan and shared backbone; adapter control changes '
        'from static/static to dynamic/dynamic'),
    )
    for comparison_index, (
        comparison_baseline, comparison_treatment, comparison_kind,
        baseline_control, treatment_control, causal_control,
    ) in enumerate(comparison_specs):
      comparison = _comparison(
        comparison_baseline,
        comparison_treatment,
        comparison_kind=comparison_kind,
        num_resamples=bootstrap_resamples,
        rng_seed=(
          bootstrap_seed + 10_000 * offset + 1000 * comparison_index),
        confidence_level=bootstrap_confidence)
      comparison['baseline']['adapter_control'] = baseline_control
      comparison['treatment']['adapter_control'] = treatment_control
      comparison['causal_control'] = causal_control
      comparisons.append(comparison)

    baseline_timing = _timing_group(
      baseline_union, mode='structured_joint', budget=budget)
    treatment_timing = _timing_group(
      treatment_union, mode='structured_joint', budget=budget)
    baseline_seconds = baseline_timing['total_wall_clock_seconds']
    treatment_seconds = treatment_timing['total_wall_clock_seconds']
    baseline_rate = baseline_timing['aggregate_active_tokens_per_second']
    treatment_rate = treatment_timing['aggregate_active_tokens_per_second']
    timing.append({
      'requested_nfe_budget': budget,
      'inferential_status': 'descriptive_only',
      'baseline_total_wall_clock_seconds': baseline_seconds,
      'treatment_total_wall_clock_seconds': treatment_seconds,
      'treatment_over_baseline_wall_clock_ratio': (
        treatment_seconds / baseline_seconds),
      'baseline_active_tokens_per_second': baseline_rate,
      'treatment_active_tokens_per_second': treatment_rate,
      'treatment_over_baseline_throughput_ratio': (
        treatment_rate / baseline_rate),
      'baseline_peak_memory_bytes': baseline_timing[
        'peak_memory_bytes_maximum'],
      'treatment_peak_memory_bytes': treatment_timing[
        'peak_memory_bytes_maximum'],
    })
  if any(
      not math.isfinite(value)
      for row in timing
      for key, value in row.items()
      if key.endswith('_ratio')):
    raise ValueError('non-finite cross-adapter timing ratio')

  return {
    'schema_version': 1,
    'artifact': 'paired_generation_adapter_comparison',
    'protocol_id': PROTOCOL_ID,
    'created_utc': created,
    'dataset_id': dataset_id,
    'scientific_scope': (
      'paired infilling generation quality and descriptive end-to-end '
      'timing; not a diffusion ELBO, likelihood, or independent timing trial'),
    'identity': {
      'repository': baseline_identity['repository'],
      'adapter_training_repository': baseline_origin['source']['repository'],
      'backbone_checkpoint': baseline_identity['artifacts'][
        'backbone_checkpoint'],
      'prompts': prompt_identity,
      'reference_lm': baseline_identity['reference_lm'],
      'runtime': baseline_identity['runtime'],
      'timing_hardware': baseline_union['timing_policy']['hardware_identity'],
      'global_pairing_digest': baseline_identity['global_pairing_digest'],
      'candidate_top_k': baseline_semantic['candidate_top_k'],
      'nfe_budgets': NFE_BUDGETS,
      'generation_protocol': {
        key: value for key, value in baseline_protocol.items()
        if key != 'control_identity'
      },
      'validated_controls': ['static_static', 'dynamic_dynamic'],
      'validated_sampling_conditions': [
        'factorized_backbone', 'independent_exact_structured_marginals',
        'dynamic_joint_forest', 'static_joint_forest',
      ],
    },
    'adapters': {
      'baseline_static_static': baseline_adapter,
      'treatment_dynamic_dynamic': treatment_adapter,
    },
    'adapter_origins': {
      'baseline_static_static': baseline_origin,
      'treatment_dynamic_dynamic': treatment_origin,
    },
    'verified_unions': {
      'baseline_static_static': {
        'canonical_sha256': canonical_sha256(baseline_union),
        'coverage': baseline_union['coverage'],
        'input_shards': baseline_union['input_shards'],
      },
      'treatment_dynamic_dynamic': {
        'canonical_sha256': canonical_sha256(treatment_union),
        'coverage': treatment_union['coverage'],
        'input_shards': treatment_union['input_shards'],
      },
    },
    'comparisons': comparisons,
    'timing': timing,
    'endpoint_direction': {
      'reference_token_accuracy': 'positive_favors_dynamic',
      'reference_exact_match': 'positive_favors_dynamic',
      'reference_lm_mean_nll_nats': 'negative_favors_dynamic',
      'repetition_rate': 'descriptive_difference_only',
    },
    'primary_causal_comparison': (
      'joint_vs_independent_structured_marginals_at_fixed_nfe'),
    'bootstrap': {
      'num_resamples': bootstrap_resamples,
      'base_rng_seed': bootstrap_seed,
      'confidence_level': bootstrap_confidence,
      'paired_draw_and_prompt_cluster_intervals': True,
    },
  }
