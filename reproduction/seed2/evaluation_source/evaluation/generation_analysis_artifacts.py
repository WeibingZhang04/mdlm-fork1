"""Trusted schemas for reviewed generation gates and analysis bundles."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from evaluation.generation_queue_artifacts import (
  atomic_write_new,
  load_strict_json,
  sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
IMMUTABLE_RUNNER_GIT_SHA = '09f89c00bbf8c65f679cd40b92609754608817b8'
IMMUTABLE_PROTOCOL_SHA256 = (
  '8b48305568495434836b39f1770342e1b9366b49746e45f7af85cf4233d0b837')
PROTOCOL_ID = 'contextual-forest-generation-paper-v1'
WIKITEXT_LAUNCH_PLAN_SHA256 = (
  '15b75cca776085608e9aa087d040544c598b793dd661bb17aa8086059d017dce')
CROSS_DOMAIN_LAUNCH_PLAN_SHA256S = {
  'arxiv': (
    '18d40e9883a82906fccafcc068f1e7fec5e1bcd8f037fc06ca610d4f5ceb1601'),
  'pubmed': (
    '9acc56ab7a36ced0ebc3dd3363224567d74c02ad14c9a6212c682d66767f8bd2'),
}
GATE_SCHEMA_VERSION = 1
GATE_ARTIFACT = 'reviewed_wikitext_cross_domain_generation_gate'
POST_BUNDLE_SCHEMA_VERSION = 1
POST_BUNDLE_ARTIFACT = 'verified_cross_domain_generation_analysis_bundle'
TRUSTED_CONTROLLER_SOURCE_PATHS = (
  'scripts/run_wikitext_generation_queue.py',
  'scripts/run_cross_domain_generation_queue.py',
  'scripts/run_cross_domain_generation_post.py',
  'scripts/compile_wikitext_cross_domain_gate.py',
  'evaluation/generation_queue_artifacts.py',
  'evaluation/generation_analysis_artifacts.py',
  'evaluation/generation_shard_aggregation.py',
  'evaluation/generation_adapter_comparison.py',
  'evaluation/generation_protocol.py',
)


@dataclass(frozen=True)
class GenerationDatasetContract:
  slug: str
  logical_dataset: str
  data_config: str
  num_prompts: int
  global_num_samples: int
  base_seed: int


DATASET_CONTRACTS = {
  'wikitext': GenerationDatasetContract(
    slug='wikitext',
    logical_dataset='wikitext103-pinned',
    data_config='eval_wikitext103_pinned',
    num_prompts=197,
    global_num_samples=788,
    base_seed=91001),
  'arxiv': GenerationDatasetContract(
    slug='arxiv',
    logical_dataset='scientific-papers-arxiv-pinned',
    data_config='eval_scientific_papers_arxiv_pinned',
    num_prompts=256,
    global_num_samples=1024,
    base_seed=92001),
  'pubmed': GenerationDatasetContract(
    slug='pubmed',
    logical_dataset='scientific-papers-pubmed-pinned',
    data_config='eval_scientific_papers_pubmed_pinned',
    num_prompts=256,
    global_num_samples=1024,
    base_seed=93001),
}


class GenerationArtifactError(ValueError):
  """A gate, union, comparison, or post bundle failed closed."""


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise GenerationArtifactError(f'{context} must be a JSON object')
  return value


def _strict(
    value: object, fields: set[str], *, context: str,
) -> Mapping[str, Any]:
  payload = _mapping(value, context=context)
  if set(payload) != fields:
    raise GenerationArtifactError(
      f'{context} schema mismatch: missing={sorted(fields - set(payload))}, '
      f'unknown={sorted(set(payload) - fields)}')
  return payload


def _lower_hex(value: object, length: int, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != length
      or any(character not in '0123456789abcdef' for character in value)):
    raise GenerationArtifactError(
      f'{context} must be {length} lowercase hexadecimal digits')
  return value


def _canonical_union_sha256(payload: object) -> str:
  encoded = json.dumps(
    payload, sort_keys=True, separators=(',', ':'),
    ensure_ascii=False).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


def inspect_controller_repository(
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  """Authenticate the clean controller checkout used to compile a gate."""
  root = Path(repo_root).expanduser().resolve()
  try:
    git_sha = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
      stderr=subprocess.STDOUT).strip()
    status = subprocess.check_output(
      ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
      cwd=root, text=True, stderr=subprocess.STDOUT)
  except (OSError, subprocess.CalledProcessError) as error:
    raise GenerationArtifactError(
      f'cannot authenticate controller repository {root}: {error}') from error
  _lower_hex(git_sha, 40, context='controller repository Git SHA')
  if status:
    raise GenerationArtifactError(
      'controller repository must be exactly clean before gate compilation '
      'or replay')
  sources = {}
  for relative_path in TRUSTED_CONTROLLER_SOURCE_PATHS:
    path = root / relative_path
    if not path.is_file():
      raise GenerationArtifactError(
        f'trusted controller source is missing: {path}')
    sources[relative_path] = sha256_file(path)
  return {
    'root': str(root),
    'git_sha': git_sha,
    'clean': True,
    'source_sha256': sources,
  }


def validate_controller_repository_binding(
    payload: object,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  binding = _strict(
    payload, {'root', 'git_sha', 'clean', 'source_sha256'},
    context='controller repository binding')
  if not isinstance(binding['root'], str) or not binding['root']:
    raise GenerationArtifactError('controller repository root is invalid')
  _lower_hex(
    binding['git_sha'], 40, context='controller repository binding Git SHA')
  if binding['clean'] is not True:
    raise GenerationArtifactError('controller repository binding is not clean')
  sources = _strict(
    binding['source_sha256'], set(TRUSTED_CONTROLLER_SOURCE_PATHS),
    context='controller source hashes')
  for relative_path, digest in sources.items():
    _lower_hex(
      digest, 64, context=f'controller source hash {relative_path}')
  observed = inspect_controller_repository(repo_root)
  # The recorded root is descriptive and mount-specific; every trust-bearing
  # field is compared exactly against the caller-supplied runtime checkout.
  expected = {
    'git_sha': binding['git_sha'],
    'clean': True,
    'source_sha256': dict(sources),
  }
  actual = {
    'git_sha': observed['git_sha'],
    'clean': observed['clean'],
    'source_sha256': observed['source_sha256'],
  }
  if actual != expected:
    raise GenerationArtifactError(
      'runtime controller repository HEAD or trusted source bytes differ '
      'from the reviewed gate')
  return {
    'root': binding['root'],
    **expected,
  }


def _artifact_reference(
    payload: object, *, context: str,
) -> tuple[Path, str]:
  reference = _strict(payload, {'path', 'sha256'}, context=context)
  path_value = reference['path']
  if not isinstance(path_value, str) or not path_value:
    raise GenerationArtifactError(f'{context}.path must be a non-empty string')
  path = Path(path_value).expanduser().resolve()
  expected_sha = _lower_hex(
    reference['sha256'], 64, context=f'{context}.sha256')
  if not path.is_file():
    raise GenerationArtifactError(f'{context} is missing: {path}')
  actual_sha = sha256_file(path)
  if actual_sha != expected_sha:
    raise GenerationArtifactError(
      f'{context} SHA256 mismatch: expected {expected_sha}, found {actual_sha}')
  return path, expected_sha


def _validate_input_shards(
    value: object, *, contract: GenerationDatasetContract, arm: str,
    global_pairing_digest: str,
) -> list[dict[str, Any]]:
  if not isinstance(value, list) or len(value) != 16:
    raise GenerationArtifactError(f'{arm} union must bind exactly 16 shards')
  normalized = []
  for index, raw in enumerate(value):
    shard = _strict(
      raw,
      {
        'shard_index', 'manifest_path', 'manifest_sha256', 'samples_sha256',
        'num_records',
      },
      context=f'{arm}.input_shards[{index}]')
    if shard['shard_index'] != index:
      raise GenerationArtifactError(f'{arm} shard ordering/coverage is not 0..15')
    manifest_sha = _lower_hex(
      shard['manifest_sha256'], 64,
      context=f'{arm}.input_shards[{index}].manifest_sha256')
    samples_sha = _lower_hex(
      shard['samples_sha256'], 64,
      context=f'{arm}.input_shards[{index}].samples_sha256')
    if (not isinstance(shard['manifest_path'], str)
        or not shard['manifest_path']):
      raise GenerationArtifactError(f'{arm} shard manifest path is invalid')
    # All cross-domain contracts divide exactly by sixteen.  WikiText has
    # twelve shards with 49 draws and four with 50; derive rather than assume.
    shard_draws = (
      (contract.global_num_samples - 1 - index) // 16 + 1)
    mode_count = 3 if arm == 'dynamic_dynamic' else 1
    expected_records = shard_draws * mode_count * 4
    if shard['num_records'] != expected_records:
      raise GenerationArtifactError(
        f'{arm} shard {index} has the wrong complete record count')
    manifest_path = Path(shard['manifest_path']).expanduser().resolve()
    if manifest_path.name != 'manifest.json' or not manifest_path.is_file():
      raise GenerationArtifactError(
        f'{arm} shard {index} manifest is missing or not manifest.json')
    if sha256_file(manifest_path) != manifest_sha:
      raise GenerationArtifactError(
        f'{arm} shard {index} manifest bytes differ from the union')
    manifest = _mapping(
      load_strict_json(manifest_path),
      context=f'{arm} shard {index} manifest')
    outputs = _mapping(
      manifest.get('outputs'), context=f'{arm} shard {index} outputs')
    sample_entry = _strict(
      outputs.get('samples_jsonl'), {'path', 'sha256', 'num_records'},
      context=f'{arm} shard {index} samples output')
    sample_relative = sample_entry['path']
    if not isinstance(sample_relative, str) or not sample_relative:
      raise GenerationArtifactError(
        f'{arm} shard {index} sample path is invalid')
    samples_path = (manifest_path.parent / sample_relative).resolve()
    try:
      samples_path.relative_to(manifest_path.parent)
    except ValueError as error:
      raise GenerationArtifactError(
        f'{arm} shard {index} sample path escapes its atomic directory') \
        from error
    if not samples_path.is_file():
      raise GenerationArtifactError(
        f'{arm} shard {index} samples JSONL is missing')
    if (sample_entry['sha256'] != samples_sha
        or sha256_file(samples_path) != samples_sha
        or sample_entry['num_records'] != expected_records):
      raise GenerationArtifactError(
        f'{arm} shard {index} samples bytes/count differ from the union')
    with samples_path.open('rb') as handle:
      observed_records = 0
      for line_number, line in enumerate(handle, start=1):
        if not line.strip():
          raise GenerationArtifactError(
            f'{arm} shard {index} samples line {line_number} is blank')
        observed_records += 1
    if observed_records != expected_records:
      raise GenerationArtifactError(
        f'{arm} shard {index} samples JSONL has {observed_records} records; '
        f'expected {expected_records}')
    repository = _mapping(
      manifest.get('repository'), context=f'{arm} shard {index} repository')
    pairing = _mapping(
      manifest.get('pairing'), context=f'{arm} shard {index} pairing')
    matrix = _mapping(
      manifest.get('matrix'), context=f'{arm} shard {index} matrix')
    prompts = _mapping(
      manifest.get('prompts'), context=f'{arm} shard {index} prompts')
    prompt_bundle = _mapping(
      prompts.get('bundle_identity'),
      context=f'{arm} shard {index} prompt bundle')
    prompt_data = _mapping(
      prompt_bundle.get('data_config'),
      context=f'{arm} shard {index} prompt data')
    prompt_output = _mapping(
      prompt_bundle.get('output'),
      context=f'{arm} shard {index} prompt output')
    artifacts = _mapping(
      manifest.get('artifacts'), context=f'{arm} shard {index} artifacts')
    adapter = _mapping(
      artifacts.get('structured_adapter'),
      context=f'{arm} shard {index} adapter')
    semantic = _mapping(
      adapter.get('semantic_identity'),
      context=f'{arm} shard {index} adapter semantics')
    expected_modes = (
      ['factorized', 'structured_marginal', 'structured_joint']
      if arm == 'dynamic_dynamic' else ['structured_joint'])
    expected_raw_identity = {
      'repository_sha': repository.get('git_sha'),
      'repository_dirty': repository.get('dirty'),
      'shard_index': pairing.get('shard_index'),
      'num_shards': pairing.get('num_shards'),
      'global_num_samples': pairing.get('global_num_samples'),
      'base_seed': pairing.get('base_seed'),
      'batch_size': pairing.get('batch_size'),
      'sequence_length': pairing.get('sequence_length'),
      'global_pairing_digest': pairing.get('global_pairing_digest'),
      'modes': matrix.get('sampling_modes'),
      'budgets': matrix.get('nfe_budgets'),
      'matrix_records': matrix.get('num_output_records'),
      'logical_dataset': prompt_data.get('logical_validation_dataset'),
      'data_config': prompt_data.get('name'),
      'num_prompts': prompt_output.get('num_prompts'),
      'control_identity': semantic.get('control_identity'),
    }
    required_raw_identity = {
      'repository_sha': IMMUTABLE_RUNNER_GIT_SHA,
      'repository_dirty': False,
      'shard_index': index,
      'num_shards': 16,
      'global_num_samples': contract.global_num_samples,
      'base_seed': contract.base_seed,
      'batch_size': 8,
      'sequence_length': 256,
      'global_pairing_digest': global_pairing_digest,
      'modes': expected_modes,
      'budgets': [8, 16, 32, 64],
      'matrix_records': expected_records,
      'logical_dataset': contract.logical_dataset,
      'data_config': contract.data_config,
      'num_prompts': contract.num_prompts,
      'control_identity': arm,
    }
    if expected_raw_identity != required_raw_identity:
      raise GenerationArtifactError(
        f'{arm} shard {index} raw identity differs from the frozen condition')
    normalized.append(dict(shard))
  return normalized


def validate_verified_union(
    payload: object,
    *,
    contract: GenerationDatasetContract,
    arm: str,
) -> dict[str, Any]:
  if arm not in {'dynamic_dynamic', 'static_static'}:
    raise GenerationArtifactError(f'unsupported union arm {arm!r}')
  union = _strict(
    payload,
    {
      'schema_version', 'artifact', 'experiment', 'created_utc', 'scope_note',
      'identity', 'coverage', 'input_shards', 'groups', 'comparisons',
      'bootstrap', 'timing_policy',
    },
    context=f'{arm} verified union')
  if (union['schema_version'] != 1
      or union['artifact'] != 'verified_generation_shard_union'):
    raise GenerationArtifactError(f'{arm} union has an unsupported schema')
  identity = _mapping(union['identity'], context=f'{arm}.identity')
  repository = _mapping(
    identity.get('repository'), context=f'{arm}.identity.repository')
  if (repository.get('git_sha') != IMMUTABLE_RUNNER_GIT_SHA
      or repository.get('dirty') is not False):
    raise GenerationArtifactError(f'{arm} union has the wrong runner identity')
  prompts = _mapping(identity.get('prompts'), context=f'{arm}.identity.prompts')
  bundle = _mapping(
    prompts.get('bundle_identity'), context=f'{arm}.prompt_bundle')
  data = _mapping(bundle.get('data_config'), context=f'{arm}.prompt_data')
  output = _mapping(bundle.get('output'), context=f'{arm}.prompt_output')
  if (data.get('logical_validation_dataset') != contract.logical_dataset
      or data.get('name') != contract.data_config
      or output.get('num_prompts') != contract.num_prompts):
    raise GenerationArtifactError(f'{arm} union binds the wrong prompt dataset')
  expected_modes = (
    ['factorized', 'structured_marginal', 'structured_joint']
    if arm == 'dynamic_dynamic' else ['structured_joint'])
  if (identity.get('base_seed') != contract.base_seed
      or identity.get('global_num_samples') != contract.global_num_samples
      or identity.get('num_shards') != 16
      or identity.get('batch_size') != 8
      or identity.get('sequence_length') != 256
      or identity.get('sampling_modes') != expected_modes
      or identity.get('nfe_budgets') != [8, 16, 32, 64]):
    raise GenerationArtifactError(f'{arm} union differs from the frozen grid')
  artifacts = _mapping(
    identity.get('artifacts'), context=f'{arm}.identity.artifacts')
  adapter = _mapping(
    artifacts.get('structured_adapter'), context=f'{arm}.adapter')
  semantic = _mapping(
    adapter.get('semantic_identity'), context=f'{arm}.adapter.semantic_identity')
  if semantic.get('control_identity') != arm:
    raise GenerationArtifactError(f'{arm} union carries the wrong adapter arm')
  global_pairing_digest = _lower_hex(
    identity.get('global_pairing_digest'), 64,
    context=f'{arm}.identity.global_pairing_digest')

  coverage = _mapping(union['coverage'], context=f'{arm}.coverage')
  expected_records = (
    contract.global_num_samples * len(expected_modes) * 4)
  if (coverage.get('num_shards') != 16
      or coverage.get('shard_indices') != list(range(16))
      or coverage.get('global_num_paired_draws') != contract.global_num_samples
      or coverage.get('num_unique_prompts') != contract.num_prompts
      or coverage.get('num_sampling_modes') != len(expected_modes)
      or coverage.get('num_nfe_budgets') != 4
      or coverage.get('num_groups') != len(expected_modes) * 4
      or coverage.get('expected_output_records') != expected_records
      or coverage.get('verified_output_records') != expected_records
      or coverage.get('global_pairing_digest') != global_pairing_digest):
    raise GenerationArtifactError(f'{arm} union coverage is incomplete')
  draws = coverage.get('paired_draws_per_prompt')
  if (not isinstance(draws, Mapping) or len(draws) != contract.num_prompts
      or set(draws.values()) != {4}):
    raise GenerationArtifactError(
      f'{arm} union does not contain four draws for every prompt')
  input_shards = _validate_input_shards(
    union['input_shards'], contract=contract, arm=arm,
    global_pairing_digest=global_pairing_digest)
  return {
    'coverage': dict(coverage),
    'input_shards': input_shards,
    'global_pairing_digest': global_pairing_digest,
  }


def validate_paired_comparison(
    payload: object, *, contract: GenerationDatasetContract,
) -> dict[str, Any]:
  comparison = _strict(
    payload,
    {
      'schema_version', 'artifact', 'protocol_id', 'created_utc', 'dataset_id',
      'scientific_scope', 'identity', 'adapters', 'adapter_origins',
      'verified_unions', 'comparisons', 'timing', 'endpoint_direction',
      'primary_causal_comparison', 'bootstrap',
    },
    context='paired adapter comparison')
  if (comparison['schema_version'] != 1
      or comparison['artifact'] != 'paired_generation_adapter_comparison'
      or comparison['protocol_id'] != PROTOCOL_ID
      or comparison['dataset_id'] != contract.logical_dataset
      or comparison['primary_causal_comparison'] !=
      'joint_vs_independent_structured_marginals_at_fixed_nfe'):
    raise GenerationArtifactError('paired comparison identity is not frozen')
  identity = _mapping(comparison['identity'], context='comparison.identity')
  repository = _mapping(
    identity.get('repository'), context='comparison.identity.repository')
  generation_protocol = _mapping(
    identity.get('generation_protocol'),
    context='comparison.identity.generation_protocol')
  if (repository.get('git_sha') != IMMUTABLE_RUNNER_GIT_SHA
      or repository.get('dirty') is not False
      or generation_protocol.get('protocol_id') != PROTOCOL_ID
      or generation_protocol.get('protocol_sha256') !=
      IMMUTABLE_PROTOCOL_SHA256
      or identity.get('nfe_budgets') != [8, 16, 32, 64]):
    raise GenerationArtifactError('paired comparison protocol identity differs')
  bootstrap = _mapping(comparison['bootstrap'], context='comparison.bootstrap')
  if (bootstrap.get('paired_draw_and_prompt_cluster_intervals') is not True
      or bootstrap.get('num_resamples') != 20_000
      or bootstrap.get('confidence_level') != 0.95):
    raise GenerationArtifactError('paired comparison bootstrap is not frozen')
  rows = comparison['comparisons']
  if not isinstance(rows, list) or len(rows) != 16:
    raise GenerationArtifactError('paired comparison must contain 16 contrasts')
  expected_kinds = {
    'structured_marginals_vs_factorized_backbone_at_fixed_nfe',
    'joint_vs_independent_structured_marginals_at_fixed_nfe',
    'dynamic_joint_vs_factorized_backbone_at_fixed_nfe',
    'dynamic_adapter_vs_static_adapter_at_fixed_nfe',
  }
  observed_cells = set()
  for row_index, raw in enumerate(rows):
    row = _mapping(raw, context=f'comparison.comparisons[{row_index}]')
    baseline = _mapping(row.get('baseline'), context='comparison baseline')
    budget = baseline.get('requested_nfe_budget')
    kind = row.get('comparison_kind')
    observed_cells.add((kind, budget))
    if (row.get('num_paired_draws') != contract.global_num_samples
        or row.get('num_prompt_clusters') != contract.num_prompts):
      raise GenerationArtifactError('comparison paired coverage is incomplete')
    endpoints = _mapping(row.get('endpoints'), context='comparison endpoints')
    if not endpoints:
      raise GenerationArtifactError('comparison has no endpoints')
    for endpoint, interval_value in endpoints.items():
      intervals = _mapping(
        interval_value, context=f'comparison endpoint {endpoint}')
      clusters = _mapping(
        intervals.get('prompt_clusters'),
        context=f'comparison endpoint {endpoint}.prompt_clusters')
      paired = _mapping(
        intervals.get('paired_draws'),
        context=f'comparison endpoint {endpoint}.paired_draws')
      if (clusters.get('num_prompt_clusters') != contract.num_prompts
          or clusters.get('num_paired_draws') != contract.global_num_samples
          or paired.get('num_paired_draws') != contract.global_num_samples):
        raise GenerationArtifactError(
          'comparison interval omits paired draws or prompt clusters')
  expected_cells = {
    (kind, budget) for kind in expected_kinds for budget in (8, 16, 32, 64)}
  if observed_cells != expected_cells:
    raise GenerationArtifactError('paired comparison contrast/NFE grid differs')
  verified = _mapping(
    comparison['verified_unions'], context='comparison.verified_unions')
  if set(verified) != {'baseline_static_static', 'treatment_dynamic_dynamic'}:
    raise GenerationArtifactError('comparison verified-union arms differ')
  return {
    'verified_unions': verified,
    'num_comparisons': len(rows),
  }


def validate_analysis_triplet(
    dynamic_payload: object,
    static_payload: object,
    comparison_payload: object,
    *,
    contract: GenerationDatasetContract,
) -> dict[str, Any]:
  dynamic = validate_verified_union(
    dynamic_payload, contract=contract, arm='dynamic_dynamic')
  static = validate_verified_union(
    static_payload, contract=contract, arm='static_static')
  comparison = validate_paired_comparison(
    comparison_payload, contract=contract)
  if dynamic['global_pairing_digest'] != static['global_pairing_digest']:
    raise GenerationArtifactError('dynamic/static global pairing digests differ')
  embedded = comparison['verified_unions']
  expected_embedded = {
    'baseline_static_static': (static, static_payload),
    'treatment_dynamic_dynamic': (dynamic, dynamic_payload),
  }
  for name, (expected, standalone_payload) in expected_embedded.items():
    observed = _mapping(embedded[name], context=f'comparison.{name}')
    if (observed.get('coverage') != expected['coverage']
        or observed.get('input_shards') != expected['input_shards']):
      raise GenerationArtifactError(
        f'paired comparison does not bind the standalone {name} union')
    canonical_sha = _lower_hex(
      observed.get('canonical_sha256'), 64,
      context=f'comparison.{name}.canonical_sha256')
    if canonical_sha != _canonical_union_sha256(standalone_payload):
      raise GenerationArtifactError(
        f'paired comparison canonical hash differs from {name} union')
  return {
    'dataset_slug': contract.slug,
    'logical_dataset': contract.logical_dataset,
    'global_num_samples': contract.global_num_samples,
    'num_prompts': contract.num_prompts,
    'dynamic_records': contract.global_num_samples * 12,
    'static_records': contract.global_num_samples * 4,
    'num_paired_comparisons': comparison['num_comparisons'],
  }


def compile_reviewed_wikitext_gate(
    dynamic_union_path: Path,
    static_union_path: Path,
    comparison_path: Path,
    *,
    decision: str,
    review_statement: str,
    reviewed_utc: str | None = None,
    controller_repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  if decision not in {'proceed', 'hold'}:
    raise GenerationArtifactError('gate decision must be proceed or hold')
  if not isinstance(review_statement, str) or not review_statement.strip():
    raise GenerationArtifactError('gate review statement must be non-empty')
  paths = {
    'dynamic_union': Path(dynamic_union_path).expanduser().resolve(),
    'static_union': Path(static_union_path).expanduser().resolve(),
    'paired_comparison': Path(comparison_path).expanduser().resolve(),
  }
  payloads = {}
  references = {}
  for name, path in paths.items():
    if not path.is_file():
      raise GenerationArtifactError(f'gate input is missing: {path}')
    payloads[name] = load_strict_json(path)
    references[name] = {'path': str(path), 'sha256': sha256_file(path)}
  summary = validate_analysis_triplet(
    payloads['dynamic_union'], payloads['static_union'],
    payloads['paired_comparison'], contract=DATASET_CONTRACTS['wikitext'])
  controller_repository = inspect_controller_repository(controller_repo_root)
  return {
    'schema_version': GATE_SCHEMA_VERSION,
    'artifact': GATE_ARTIFACT,
    'decision': decision,
    'review': {
      'status': 'reviewed',
      'statement': review_statement.strip(),
      'reviewed_utc': reviewed_utc or dt.datetime.now(dt.timezone.utc).isoformat(),
    },
    'identity': {
      'protocol_id': PROTOCOL_ID,
      'protocol_sha256': IMMUTABLE_PROTOCOL_SHA256,
      'immutable_runner_git_sha': IMMUTABLE_RUNNER_GIT_SHA,
      'wikitext_launch_plan_sha256': WIKITEXT_LAUNCH_PLAN_SHA256,
      'dataset_slug': 'wikitext',
      'logical_dataset': DATASET_CONTRACTS['wikitext'].logical_dataset,
      'controller_repository': controller_repository,
    },
    'artifacts': references,
    'validated_analysis': summary,
  }


def validate_reviewed_wikitext_gate(
    gate_path: Path,
    *,
    expected_sha256: str,
    require_proceed: bool = True,
    controller_repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  gate_path = Path(gate_path).expanduser().resolve()
  expected_sha256 = _lower_hex(
    expected_sha256, 64, context='reviewed gate SHA256')
  if not gate_path.is_file():
    raise GenerationArtifactError(f'reviewed gate is missing: {gate_path}')
  actual_sha = sha256_file(gate_path)
  if actual_sha != expected_sha256:
    raise GenerationArtifactError(
      f'reviewed gate SHA256 mismatch: expected {expected_sha256}, '
      f'found {actual_sha}')
  gate = _strict(
    load_strict_json(gate_path),
    {
      'schema_version', 'artifact', 'decision', 'review', 'identity',
      'artifacts', 'validated_analysis',
    },
    context='reviewed WikiText gate')
  if (gate['schema_version'] != GATE_SCHEMA_VERSION
      or gate['artifact'] != GATE_ARTIFACT):
    raise GenerationArtifactError('reviewed gate schema is unsupported')
  if gate['decision'] not in {'proceed', 'hold'}:
    raise GenerationArtifactError('reviewed gate decision is invalid')
  if require_proceed and gate['decision'] != 'proceed':
    raise GenerationArtifactError('reviewed WikiText gate does not authorize launch')
  review = _strict(
    gate['review'], {'status', 'statement', 'reviewed_utc'},
    context='reviewed gate review')
  if (review['status'] != 'reviewed'
      or not isinstance(review['statement'], str)
      or not review['statement'].strip()
      or not isinstance(review['reviewed_utc'], str)
      or not review['reviewed_utc']):
    raise GenerationArtifactError('reviewed gate lacks a complete review decision')
  identity = _strict(
    gate['identity'],
    {
      'protocol_id', 'protocol_sha256', 'immutable_runner_git_sha',
      'wikitext_launch_plan_sha256', 'dataset_slug', 'logical_dataset',
      'controller_repository',
    },
    context='reviewed gate identity')
  expected_identity = {
    'protocol_id': PROTOCOL_ID,
    'protocol_sha256': IMMUTABLE_PROTOCOL_SHA256,
    'immutable_runner_git_sha': IMMUTABLE_RUNNER_GIT_SHA,
    'wikitext_launch_plan_sha256': WIKITEXT_LAUNCH_PLAN_SHA256,
    'dataset_slug': 'wikitext',
    'logical_dataset': DATASET_CONTRACTS['wikitext'].logical_dataset,
  }
  observed_static_identity = {
    key: identity[key] for key in expected_identity
  }
  if observed_static_identity != expected_identity:
    raise GenerationArtifactError('reviewed gate identity differs from policy')
  controller_repository = validate_controller_repository_binding(
    identity['controller_repository'], repo_root=controller_repo_root)
  artifacts = _strict(
    gate['artifacts'],
    {'dynamic_union', 'static_union', 'paired_comparison'},
    context='reviewed gate artifacts')
  payloads = {}
  for name, reference in artifacts.items():
    path, _ = _artifact_reference(reference, context=f'gate artifact {name}')
    payloads[name] = load_strict_json(path)
  summary = validate_analysis_triplet(
    payloads['dynamic_union'], payloads['static_union'],
    payloads['paired_comparison'], contract=DATASET_CONTRACTS['wikitext'])
  if gate['validated_analysis'] != summary:
    raise GenerationArtifactError('reviewed gate analysis summary was tampered')
  return {
    'path': str(gate_path),
    'sha256': actual_sha,
    'decision': gate['decision'],
    'review': dict(review),
    'identity': {
      **expected_identity,
      'controller_repository': controller_repository,
    },
    'controller_repository': controller_repository,
    'artifacts': {name: dict(value) for name, value in artifacts.items()},
    'validated_analysis': summary,
  }


def write_reviewed_wikitext_gate(path: Path, payload: Mapping[str, Any]) -> str:
  atomic_write_new(
    path, json.dumps(payload, indent=2, sort_keys=True) + '\n')
  return sha256_file(path)


def reviewed_gate_launch_authorization(
    reviewed_gate: Mapping[str, Any],
) -> dict[str, Any]:
  return {
    'artifact': GATE_ARTIFACT,
    'path': reviewed_gate.get('path'),
    'sha256': reviewed_gate.get('sha256'),
    'decision': reviewed_gate.get('decision'),
    'controller_repository': reviewed_gate.get('controller_repository'),
  }


def validate_cross_domain_queue_completion(
    path: Path,
    *,
    contract: GenerationDatasetContract,
    reviewed_gate: Mapping[str, Any],
    expected_tasks: list[Mapping[str, Any]] | None = None,
    union_input_shards: Mapping[str, list[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
  """Replay dataset/task/manifest bindings in queue completion evidence."""
  if contract.slug == 'wikitext':
    raise GenerationArtifactError('cross-domain completion cannot be WikiText')
  path = Path(path).expanduser().resolve()
  if not path.is_file():
    raise GenerationArtifactError(
      f'dataset-bound queue completion evidence is missing: {path}')
  payload = _strict(
    load_strict_json(path),
    {
      'schema_version', 'artifact', 'dataset_slug', 'logical_dataset',
      'immutable_runner_git_sha', 'launch_plan_sha256', 'num_tasks', 'tasks',
      'completed_utc', 'launch_authorization',
    },
    context='queue completion evidence')
  if (payload['schema_version'] != 1
      or payload['artifact'] != 'frozen_generation_queue_completion'
      or payload['dataset_slug'] != contract.slug
      or payload['logical_dataset'] != contract.logical_dataset
      or payload['immutable_runner_git_sha'] != IMMUTABLE_RUNNER_GIT_SHA
      or payload['launch_plan_sha256'] !=
      CROSS_DOMAIN_LAUNCH_PLAN_SHA256S[contract.slug]
      or payload['num_tasks'] != 32
      or not isinstance(payload['completed_utc'], str)
      or not payload['completed_utc']):
    raise GenerationArtifactError(
      'queue completion evidence identity differs from policy')
  authorization = _strict(
    payload['launch_authorization'],
    {'artifact', 'path', 'sha256', 'decision', 'controller_repository'},
    context='queue completion launch authorization')
  expected_authorization = reviewed_gate_launch_authorization(reviewed_gate)
  if dict(authorization) != expected_authorization:
    raise GenerationArtifactError(
      'queue completion evidence binds a different launch gate or controller')
  tasks = payload['tasks']
  if not isinstance(tasks, list) or len(tasks) != 32:
    raise GenerationArtifactError(
      'queue completion evidence must contain exactly 32 ordered tasks')
  expected_signature = [
    (arm, index)
    for arm in ('dynamic_dynamic', 'static_static')
    for index in range(16)
  ]
  normalized_tasks = []
  for position, (raw, (arm, shard_index)) in enumerate(
      zip(tasks, expected_signature)):
    task = _strict(
      raw,
      {
        'task_id', 'dataset_slug', 'arm', 'shard_index', 'output_dir',
        'manifest_sha256',
      },
      context=f'queue completion task {position}')
    expected_task_id = (
      f'{contract.slug}-{arm}-shard-{shard_index:02d}')
    output_value = task['output_dir']
    if (task['task_id'] != expected_task_id
        or task['dataset_slug'] != contract.slug
        or task['arm'] != arm
        or task['shard_index'] != shard_index
        or not isinstance(output_value, str)
        or not output_value
        or not Path(output_value).is_absolute()):
      raise GenerationArtifactError(
        f'queue completion task {position} differs from the frozen grid')
    output_dir = Path(output_value).expanduser().resolve()
    if str(output_dir) != output_value:
      raise GenerationArtifactError(
        f'queue completion task {position} output path is not canonical')
    manifest_path = output_dir / 'manifest.json'
    manifest_sha = _lower_hex(
      task['manifest_sha256'], 64,
      context=f'queue completion task {position} manifest SHA256')
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha:
      raise GenerationArtifactError(
        f'queue completion task {position} differs from current manifest bytes')
    normalized = {
      'task_id': expected_task_id,
      'dataset_slug': contract.slug,
      'arm': arm,
      'shard_index': shard_index,
      'output_dir': str(output_dir),
      'manifest_sha256': manifest_sha,
    }
    if expected_tasks is not None:
      if len(expected_tasks) != 32 or dict(expected_tasks[position]) != normalized:
        raise GenerationArtifactError(
          f'queue completion task {position} differs from the launch plan')
    if union_input_shards is not None:
      if set(union_input_shards) != {'dynamic_dynamic', 'static_static'}:
        raise GenerationArtifactError(
          'queue completion union bindings have the wrong arms')
      arm_bindings = union_input_shards[arm]
      if len(arm_bindings) != 16:
        raise GenerationArtifactError(
          f'queue completion union binding for {arm} is incomplete')
      union_shard = arm_bindings[shard_index]
      if (union_shard.get('shard_index') != shard_index
          or Path(union_shard.get('manifest_path', '')).expanduser().resolve()
          != manifest_path
          or union_shard.get('manifest_sha256') != manifest_sha):
        raise GenerationArtifactError(
          f'queue completion task {position} differs from its verified union')
    normalized_tasks.append(normalized)
  return {
    'path': str(path),
    'sha256': sha256_file(path),
    'dataset_slug': contract.slug,
    'logical_dataset': contract.logical_dataset,
    'launch_plan_sha256': payload['launch_plan_sha256'],
    'launch_authorization': expected_authorization,
    'tasks': normalized_tasks,
  }


def validate_cross_domain_post_bundle(
    bundle_path: Path,
    *,
    expected_sha256: str | None = None,
    controller_repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
  bundle_path = Path(bundle_path).expanduser().resolve()
  if not bundle_path.is_file():
    raise GenerationArtifactError(f'post-analysis bundle is missing: {bundle_path}')
  actual_sha = sha256_file(bundle_path)
  if expected_sha256 is not None and actual_sha != _lower_hex(
      expected_sha256, 64, context='post-analysis bundle SHA256'):
    raise GenerationArtifactError('post-analysis bundle SHA256 mismatch')
  bundle = _strict(
    load_strict_json(bundle_path),
    {
      'schema_version', 'artifact', 'created_utc', 'dataset_slug',
      'logical_dataset', 'immutable_runner_git_sha', 'launch_plan_sha256',
      'queue_completion_evidence', 'reviewed_wikitext_gate', 'artifacts',
      'validated_analysis',
    },
    context='cross-domain post-analysis bundle')
  slug = bundle['dataset_slug']
  contract = DATASET_CONTRACTS.get(slug)
  if contract is None or slug == 'wikitext':
    raise GenerationArtifactError('post-analysis bundle dataset is unsupported')
  if (bundle['schema_version'] != POST_BUNDLE_SCHEMA_VERSION
      or bundle['artifact'] != POST_BUNDLE_ARTIFACT
      or bundle['logical_dataset'] != contract.logical_dataset
      or bundle['immutable_runner_git_sha'] != IMMUTABLE_RUNNER_GIT_SHA
      or bundle['launch_plan_sha256'] !=
      CROSS_DOMAIN_LAUNCH_PLAN_SHA256S[slug]):
    raise GenerationArtifactError('post-analysis bundle identity differs')
  queue_path, queue_sha = _artifact_reference(
    bundle['queue_completion_evidence'], context='queue completion evidence')
  gate_reference = _strict(
    bundle['reviewed_wikitext_gate'],
    {'path', 'sha256', 'decision', 'controller_repository'},
    context='post-analysis reviewed gate')
  if gate_reference['decision'] != 'proceed':
    raise GenerationArtifactError('post-analysis bundle lacks proceed decision')
  gate = validate_reviewed_wikitext_gate(
    Path(gate_reference['path']),
    expected_sha256=gate_reference['sha256'],
    require_proceed=True,
    controller_repo_root=controller_repo_root)
  if gate_reference['controller_repository'] != gate['controller_repository']:
    raise GenerationArtifactError(
      'post-analysis gate reference binds a different controller repository')
  artifacts = _strict(
    bundle['artifacts'],
    {'dynamic_union', 'static_union', 'paired_comparison'},
    context='post-analysis artifacts')
  payloads = {}
  normalized_artifacts = {}
  for name, reference in artifacts.items():
    path, sha = _artifact_reference(
      reference, context=f'post-analysis artifact {name}')
    payloads[name] = load_strict_json(path)
    normalized_artifacts[name] = {'path': str(path), 'sha256': sha}
  summary = validate_analysis_triplet(
    payloads['dynamic_union'], payloads['static_union'],
    payloads['paired_comparison'], contract=contract)
  queue = validate_cross_domain_queue_completion(
    queue_path,
    contract=contract,
    reviewed_gate=gate,
    union_input_shards={
      'dynamic_dynamic': payloads['dynamic_union']['input_shards'],
      'static_static': payloads['static_union']['input_shards'],
    })
  if queue['sha256'] != queue_sha:
    raise GenerationArtifactError(
      'queue completion replay hash differs from its bundle reference')
  if bundle['validated_analysis'] != summary:
    raise GenerationArtifactError('post-analysis bundle summary was tampered')
  return {
    'path': str(bundle_path),
    'sha256': actual_sha,
    'dataset_slug': slug,
    'logical_dataset': contract.logical_dataset,
    'launch_plan_sha256': bundle['launch_plan_sha256'],
    'queue_completion_evidence': queue,
    'reviewed_wikitext_gate': gate,
    'artifacts': normalized_artifacts,
    'validated_analysis': summary,
  }
