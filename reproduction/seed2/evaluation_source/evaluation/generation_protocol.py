"""Fail-closed validation for the frozen paper-scale generation protocol.

The generation runner deliberately remains a general-purpose executable.  A
paper result becomes admissible only when this module verifies its complete
resolved Hydra config and its prompt/runtime manifest against the committed
protocol.  Exactly seven config values are normalized: four adapter artifact
locations/digests and the three structured-decoder fields that define the two
named arms.  Those values are not ignored; they are independently bound to the
validated adapter manifest before removal.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATION_PROTOCOL_PATH = (
  REPO_ROOT / 'configs' / 'experiment'
  / 'contextual-forest-generation-paper-v1.yaml')

PROTOCOL_ID = 'contextual-forest-generation-paper-v1'
PROTOCOL_SCHEMA_VERSION = 1
PROMPT_ARTIFACT = 'pinned_document_local_infilling_prompts'
PROMPT_MANIFEST_SCHEMA_VERSION = 2

ADAPTER_PATHS = (
  ('eval', 'adapter_checkpoint'),
  ('eval', 'adapter_sha256'),
  ('eval', 'adapter_manifest'),
  ('eval', 'adapter_manifest_sha256'),
)
ARM_PATHS = (
  ('model', 'structured_decoder', 'topology_mode'),
  ('model', 'structured_decoder', 'factor_mode'),
  ('model', 'structured_decoder', 'training', 'topology_weight'),
)
PLACEHOLDERS = {
  '__DATA_CONFIG__', '__VALID_DATASET__', '__CANDIDATE_TOP_K__',
}


class _UniqueKeyLoader(yaml.SafeLoader):
  """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_mapping(loader, node, deep=False):
  mapping = {}
  for key_node, value_node in node.value:
    key = loader.construct_object(key_node, deep=deep)
    if key in mapping:
      raise ValueError(f'duplicate YAML mapping key: {key!r}')
    mapping[key] = loader.construct_object(value_node, deep=deep)
  return mapping


_UniqueKeyLoader.add_constructor(
  yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _plain(value: Any, *, context: str = 'value') -> Any:
  """Convert a config tree to strict JSON-compatible Python values."""
  if isinstance(value, Mapping):
    result = {}
    for key, item in value.items():
      if not isinstance(key, str):
        raise TypeError(f'{context} contains a non-string mapping key')
      result[key] = _plain(item, context=f'{context}.{key}')
    return result
  if isinstance(value, (list, tuple)):
    return [
      _plain(item, context=f'{context}[{index}]')
      for index, item in enumerate(value)
    ]
  if value is None or type(value) in {str, bool, int}:
    return value
  if type(value) is float:
    if not math.isfinite(value):
      raise ValueError(f'{context} contains a non-finite float')
    return value
  raise TypeError(
    f'{context} contains unsupported value type {type(value).__name__}')


def canonical_sha256(value: Any) -> str:
  payload = _plain(value)
  encoded = json.dumps(
    payload, sort_keys=True, separators=(',', ':'),
    ensure_ascii=False, allow_nan=False).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _load_yaml(path: Path, *, context: str) -> dict[str, Any]:
  try:
    payload = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
  except yaml.YAMLError as error:
    raise ValueError(f'invalid YAML in {path}: {error}') from error
  if not isinstance(payload, Mapping):
    raise TypeError(f'{context} must be a YAML mapping')
  return _plain(payload, context=context)


def _strict_fields(
    payload: Mapping[str, Any], expected: set[str], *, context: str,
) -> None:
  observed = set(payload)
  if observed != expected:
    raise ValueError(
      f'{context} schema mismatch: missing={sorted(expected - observed)}, '
      f'unknown={sorted(observed - expected)}')


def _lower_hex(value: object, length: int, *, context: str) -> str:
  if (not isinstance(value, str) or len(value) != length
      or any(character not in '0123456789abcdef' for character in value)):
    raise ValueError(
      f'{context} must be {length} lowercase hexadecimal digits')
  return value


def _positive_int(value: object, *, context: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f'{context} must be a positive integer')
  return value


def _path_value(
    payload: Mapping[str, Any], path: Sequence[str], *, context: str,
) -> Any:
  current: Any = payload
  for field in path:
    if not isinstance(current, Mapping) or field not in current:
      raise ValueError(f'{context} is missing {".".join(path)}')
    current = current[field]
  return current


def _remove_path(
    payload: dict[str, Any], path: Sequence[str], *, context: str,
) -> Any:
  current: Any = payload
  for field in path[:-1]:
    if not isinstance(current, dict) or field not in current:
      raise ValueError(f'{context} is missing {".".join(path)}')
    current = current[field]
  if not isinstance(current, dict) or path[-1] not in current:
    raise ValueError(f'{context} is missing {".".join(path)}')
  return current.pop(path[-1])


def _assert_exact(actual: Any, expected: Any, *, context: str) -> None:
  """Type-exact recursive equality with a useful first-difference path."""
  if isinstance(expected, Mapping):
    if not isinstance(actual, Mapping):
      raise ValueError(f'{context} must be an object')
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
      raise ValueError(
        f'{context} keys differ: missing={sorted(expected_keys - actual_keys)}, '
        f'unknown={sorted(actual_keys - expected_keys)}')
    for key in expected:
      _assert_exact(actual[key], expected[key], context=f'{context}.{key}')
    return
  if isinstance(expected, list):
    if not isinstance(actual, list) or len(actual) != len(expected):
      raise ValueError(f'{context} list differs from the frozen protocol')
    for index, (actual_item, expected_item) in enumerate(
        zip(actual, expected)):
      _assert_exact(
        actual_item, expected_item, context=f'{context}[{index}]')
    return
  if actual != expected or type(actual) is not type(expected):
    raise ValueError(
      f'{context} differs from the frozen protocol: '
      f'{actual!r} versus {expected!r}')


def _materialize(value: Any, *, dataset: Mapping[str, Any], candidate_k: int):
  if value == '__DATA_CONFIG__':
    return copy.deepcopy(dataset['resolved_data'])
  if value == '__VALID_DATASET__':
    return dataset['logical_validation_dataset']
  if value == '__CANDIDATE_TOP_K__':
    return candidate_k
  if isinstance(value, Mapping):
    return {
      key: _materialize(item, dataset=dataset, candidate_k=candidate_k)
      for key, item in value.items()
    }
  if isinstance(value, list):
    return [
      _materialize(item, dataset=dataset, candidate_k=candidate_k)
      for item in value
    ]
  return value


def _find_placeholders(value: Any) -> set[str]:
  result = set()
  if isinstance(value, str) and value.startswith('__') and value.endswith('__'):
    result.add(value)
  elif isinstance(value, Mapping):
    for item in value.values():
      result.update(_find_placeholders(item))
  elif isinstance(value, list):
    for item in value:
      result.update(_find_placeholders(item))
  return result


def _validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
  expected_top_level = {
    'schema_version', 'protocol_id', 'protocol_status', 'scientific_scope',
    'parameters', 'normalization', 'artifacts', 'arms', 'generation',
    'datasets', 'resolved_config_projection',
  }
  _strict_fields(protocol, expected_top_level, context='generation protocol')
  if (protocol['schema_version'] != PROTOCOL_SCHEMA_VERSION
      or protocol['protocol_id'] != PROTOCOL_ID
      or protocol['protocol_status']
      != 'frozen_before_paper_scale_generation'):
    raise ValueError('unsupported or non-frozen generation protocol')

  parameters = protocol['parameters']
  _strict_fields(
    parameters, {'candidate_top_k', 'dataset_configs'},
    context='generation protocol parameters')
  _strict_fields(
    parameters['candidate_top_k'], {'allowed_values'},
    context='candidate_top_k parameter')
  allowed_k = parameters['candidate_top_k']['allowed_values']
  if (not isinstance(allowed_k, list) or allowed_k != [64, 128, 256]
      or any(_positive_int(value, context='allowed candidate K') != value
             for value in allowed_k)):
    raise ValueError('candidate_top_k parameter must freeze [64, 128, 256]')
  dataset_names = parameters['dataset_configs']
  datasets = protocol['datasets']
  if (not isinstance(dataset_names, list)
      or dataset_names != list(datasets)
      or len(set(dataset_names)) != 3):
    raise ValueError('dataset parameter must name the three ordered datasets')

  normalization = protocol['normalization']
  _strict_fields(
    normalization,
    {'adapter_manifest_bound_paths', 'arm_manifest_bound_paths'},
    context='generation config normalization')
  _assert_exact(
    normalization['adapter_manifest_bound_paths'],
    [list(path) for path in ADAPTER_PATHS],
    context='generation config adapter normalization')
  _assert_exact(
    normalization['arm_manifest_bound_paths'],
    [list(path) for path in ARM_PATHS],
    context='generation config arm normalization')

  _strict_fields(
    protocol['artifacts'], {'backbone_checkpoint'},
    context='generation protocol artifacts')
  _strict_fields(
    protocol['artifacts']['backbone_checkpoint'], {'path', 'sha256'},
    context='generation protocol backbone')
  _lower_hex(
    protocol['artifacts']['backbone_checkpoint']['sha256'], 64,
    context='generation protocol backbone SHA256')

  arms = protocol['arms']
  if set(arms) != {'dynamic_dynamic', 'static_static'}:
    raise ValueError('generation protocol must contain exactly two arms')
  expected_arm_fields = {
    'topology_mode', 'factor_mode', 'topology_weight', 'sampling_modes',
  }
  for name, arm in arms.items():
    _strict_fields(arm, expected_arm_fields, context=f'arm {name}')
  expected_arm_semantics = {
    'dynamic_dynamic': ('dynamic', 'dynamic', 0.1),
    'static_static': ('fixed', 'fixed', 0.0),
  }
  for name, expected in expected_arm_semantics.items():
    observed = (
      arms[name]['topology_mode'], arms[name]['factor_mode'],
      arms[name]['topology_weight'])
    if observed != expected:
      raise ValueError(f'arm {name} semantics are not frozen')
  if (arms['dynamic_dynamic']['sampling_modes'] != [
      'factorized', 'structured_marginal', 'structured_joint']
      or arms['static_static']['sampling_modes'] != ['structured_joint']):
    raise ValueError('generation sampling-mode grids are not frozen')

  generation = protocol['generation']
  generation_fields = {
    'sequence_length', 'span_length', 'selection_seed', 'nfe_budgets',
    'num_shards', 'batch_size', 'prompt_policy', 'repository', 'host',
    'reference_lm',
  }
  _strict_fields(generation, generation_fields, context='generation settings')
  if (generation['sequence_length'] != 256
      or generation['span_length'] != 32
      or generation['selection_seed'] != 31001
      or generation['nfe_budgets'] != [8, 16, 32, 64]
      or generation['num_shards'] != 16
      or generation['batch_size'] != 8):
    raise ValueError('paper-scale generation grid is not frozen')
  reference_lm = generation['reference_lm']
  _strict_fields(
    reference_lm,
    {'model_name_or_path', 'revision', 'sequence_policy',
     'runtime_configuration'},
    context='generation reference LM')
  runtime_configuration = reference_lm['runtime_configuration']
  runtime_fields = {
    'model_name_or_path', 'model_revision', 'model_class',
    'model_config_class', 'tokenizer_name_or_path', 'tokenizer_revision',
    'tokenizer_class', 'tokenizer_vocab_size', 'tokenizer_bos_token_id',
    'tokenizer_eos_token_id', 'tokenizer_pad_token_id',
    'tokenizer_padding_side', 'tokenizer_truncation_side',
    'tokenization_policy', 'sequence_policy', 'add_special_tokens',
    'batch_size', 'max_length', 'requested_dtype', 'parameter_dtypes',
    'precision_policy', 'device',
  }
  if not isinstance(runtime_configuration, Mapping):
    raise TypeError('generation reference-LM runtime must be an object')
  _strict_fields(
    runtime_configuration, runtime_fields,
    context='generation reference-LM runtime')
  reference_links = {
    'model_name_or_path': reference_lm['model_name_or_path'],
    'model_revision': reference_lm['revision'],
    'tokenizer_revision': reference_lm['revision'],
    'sequence_policy': reference_lm['sequence_policy'],
    'batch_size': generation['batch_size'],
    'max_length': generation['sequence_length'],
    'device': generation['host']['device'],
    'parameter_dtypes': generation['host']['parameter_dtypes'],
  }
  for field, expected in reference_links.items():
    if (runtime_configuration.get(field) != expected
        or type(runtime_configuration.get(field)) is not type(expected)):
      raise ValueError(
        f'generation reference-LM runtime {field} is inconsistent')

  expected_dataset_fields = {
    'logical_validation_dataset', 'data_config_path',
    'data_config_sha256', 'dataset_revision', 'tokenizer_name_or_path',
    'tokenizer_revision', 'document_boundary_mode', 'num_prompts',
    'global_num_samples', 'base_seed', 'runtime_specification_sha256',
    'runtime_specification', 'resolved_data',
  }
  for name, dataset in datasets.items():
    _strict_fields(dataset, expected_dataset_fields, context=f'dataset {name}')
    data_path = (REPO_ROOT / dataset['data_config_path']).resolve()
    try:
      data_path.relative_to((REPO_ROOT / 'configs' / 'data').resolve())
    except ValueError as error:
      raise ValueError(f'dataset {name} config path escapes configs/data') \
        from error
    if not data_path.is_file():
      raise FileNotFoundError(data_path)
    expected_data_sha = _lower_hex(
      dataset['data_config_sha256'], 64,
      context=f'dataset {name} config SHA256')
    if sha256_file(data_path) != expected_data_sha:
      raise ValueError(f'dataset {name} committed config SHA256 mismatch')
    on_disk_data = _load_yaml(data_path, context=f'dataset {name} data config')
    _assert_exact(
      on_disk_data, dataset['resolved_data'],
      context=f'dataset {name} resolved data projection')
    expected_spec_sha = _lower_hex(
      dataset['runtime_specification_sha256'], 64,
      context=f'dataset {name} runtime specification SHA256')
    if canonical_sha256(dataset['runtime_specification']) != expected_spec_sha:
      raise ValueError(
        f'dataset {name} runtime specification self-hash mismatch')
    expected_links = {
      'logical_dataset_name': dataset['logical_validation_dataset'],
      'source_revision': dataset['dataset_revision'],
      'tokenizer_name_or_path': dataset['tokenizer_name_or_path'],
      'tokenizer_revision': dataset['tokenizer_revision'],
      'document_boundary_mode': dataset['document_boundary_mode'],
      'block_size': generation['sequence_length'],
    }
    for field, expected in expected_links.items():
      if dataset['runtime_specification'].get(field) != expected:
        raise ValueError(
          f'dataset {name} runtime specification {field} is inconsistent')
    resolved_data = dataset['resolved_data']
    if (resolved_data.get('valid') != dataset['logical_validation_dataset']
        or resolved_data.get('valid_revision') != dataset['dataset_revision']
        or resolved_data.get('tokenizer_name_or_path')
        != dataset['tokenizer_name_or_path']
        or resolved_data.get('tokenizer_revision')
        != dataset['tokenizer_revision']
        or resolved_data.get('valid_document_boundary_mode')
        != dataset['document_boundary_mode']):
      raise ValueError(f'dataset {name} data semantics are inconsistent')

  projection = protocol['resolved_config_projection']
  placeholders = _find_placeholders(projection)
  if placeholders != PLACEHOLDERS:
    raise ValueError(
      f'resolved config projection placeholders differ: {placeholders}')
  for path in (*ADAPTER_PATHS, *ARM_PATHS):
    current: Any = projection
    for field in path:
      if not isinstance(current, Mapping) or field not in current:
        break
      current = current[field]
    else:
      raise ValueError(
        f'normalized projection unexpectedly retains {".".join(path)}')
  return copy.deepcopy(dict(protocol))


def load_generation_protocol(
    path: Path = DEFAULT_GENERATION_PROTOCOL_PATH,
) -> dict[str, Any]:
  """Load and internally verify the committed generation protocol."""
  path = Path(path).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  return _validate_protocol(_load_yaml(path, context='generation protocol'))


def materialize_resolved_config(
    protocol: Mapping[str, Any],
    *,
    dataset_config: str,
    candidate_top_k: int,
) -> dict[str, Any]:
  """Materialize the exact normalized config for one allowed dataset/K."""
  candidate_top_k = _positive_int(
    candidate_top_k, context='candidate_top_k')
  allowed = protocol['parameters']['candidate_top_k']['allowed_values']
  if candidate_top_k not in allowed:
    raise ValueError(
      f'candidate_top_k {candidate_top_k} is outside {allowed}')
  if dataset_config not in protocol['datasets']:
    raise ValueError(
      f'dataset config {dataset_config!r} is outside the frozen protocol')
  return _materialize(
    protocol['resolved_config_projection'],
    dataset=protocol['datasets'][dataset_config],
    candidate_k=candidate_top_k)


def _semantic_identity_from_projection(
    config: Mapping[str, Any], *, control: str, arm: Mapping[str, Any],
) -> dict[str, Any]:
  structured = config['model']['structured_decoder']
  training = structured['training']
  return {
    'control_identity': control,
    'topology_mode': arm['topology_mode'],
    'factor_mode': arm['factor_mode'],
    'candidate_top_k': structured['top_k'],
    'independent_mode': structured['independent_mode'],
    'topology_weight': arm['topology_weight'],
    'head_semantics': {
      field: structured[field]
      for field in (
        'rank', 'time_embed_dim', 'topology_dim', 'local_window',
        'num_anchor_slots', 'contextual_neighbors', 'component_size_cap',
        'min_edge_score', 'fixed_edges', 'fixed_edge_path')
    },
    'training_semantics': {
      field: training[field]
      for field in (
        'objective_name', 'factorized_aux_weight', 'topology_strategy',
        'topology_temperature', 'topology_minimum_choices',
        'topology_edge_weight', 'topology_anchor_weight',
        'topology_slot_weight', 'topology_on_validation')
    },
  }


def _validate_prompt_identity(
    prompt: Mapping[str, Any],
    *,
    dataset_name: str,
    dataset: Mapping[str, Any],
    generation: Mapping[str, Any],
    repository_sha: str,
) -> None:
  if prompt.get('source') != 'jsonl':
    raise ValueError('paper-scale generation requires JSONL infilling prompts')
  if prompt.get('num_prompt_records') != dataset['num_prompts']:
    raise ValueError('prompt record count differs from the frozen protocol')
  _lower_hex(
    prompt.get('sha256'), 64, context='prompt JSONL SHA256')
  _lower_hex(
    prompt.get('manifest_sha256'), 64,
    context='prompt manifest SHA256')
  bundle = prompt.get('bundle_identity')
  if not isinstance(bundle, Mapping):
    raise TypeError('prompt bundle identity must be an object')
  expected_bundle_fields = {
    'schema_version', 'artifact', 'manifest_sha256', 'builder_git_sha',
    'data_config', 'runtime_provenance', 'policy', 'output',
  }
  _strict_fields(bundle, expected_bundle_fields, context='prompt bundle')
  if (bundle['schema_version'] != PROMPT_MANIFEST_SCHEMA_VERSION
      or bundle['artifact'] != PROMPT_ARTIFACT
      or bundle['manifest_sha256'] != prompt['manifest_sha256']):
    raise ValueError('prompt bundle identity is unsupported or inconsistent')
  if (generation['repository']['prompt_builder_same_git_sha']
      and bundle['builder_git_sha'] != repository_sha):
    raise ValueError(
      'prompt builder Git SHA differs from the generation repository')

  expected_data = {
    'name': dataset_name,
    'sha256': dataset['data_config_sha256'],
    'logical_validation_dataset': dataset['logical_validation_dataset'],
    'dataset_revision': dataset['dataset_revision'],
    'tokenizer_name_or_path': dataset['tokenizer_name_or_path'],
    'tokenizer_revision': dataset['tokenizer_revision'],
  }
  _assert_exact(
    bundle['data_config'], expected_data,
    context='prompt bundle data_config')
  expected_policy = {
    **generation['prompt_policy'],
    'selection_seed': generation['selection_seed'],
    'span_length': generation['span_length'],
    'sequence_length': generation['sequence_length'],
  }
  _assert_exact(
    bundle['policy'], expected_policy, context='prompt bundle policy')
  runtime = bundle['runtime_provenance']
  if not isinstance(runtime, Mapping):
    raise TypeError('prompt runtime provenance identity must be an object')
  _strict_fields(
    runtime, {'sha256', 'specification_sha256', 'manifest_sha256'},
    context='prompt runtime provenance identity')
  for field in ('sha256', 'manifest_sha256'):
    _lower_hex(runtime[field], 64, context=f'prompt runtime {field}')
  if runtime['specification_sha256'] != \
      dataset['runtime_specification_sha256']:
    raise ValueError(
      'prompt runtime specification differs from the exact dataset, '
      'tokenizer, or document-boundary semantics')
  output = bundle['output']
  if not isinstance(output, Mapping):
    raise TypeError('prompt bundle output identity must be an object')
  _strict_fields(
    output, {'sha256', 'size_bytes', 'num_prompts'},
    context='prompt bundle output')
  if (output['sha256'] != prompt['sha256']
      or output['num_prompts'] != dataset['num_prompts']):
    raise ValueError('prompt output identity differs from the frozen bundle')
  _positive_int(output['size_bytes'], context='prompt bundle output size')


def validate_generation_protocol(
    resolved_config: Mapping[str, Any] | Path,
    manifest: Mapping[str, Any],
    *,
    candidate_top_k: int,
    expected_control: str | None = None,
    protocol_path: Path = DEFAULT_GENERATION_PROTOCOL_PATH,
) -> dict[str, Any]:
  """Validate one generation shard against the absolute paper protocol.

  This is the intended comparator integration point.  Call it on the first
  independently validated shard from each adapter arm; shard aggregation
  already requires every other shard to have the same resolved config and
  manifest identity.
  """
  protocol_path = Path(protocol_path).expanduser().resolve()
  protocol = load_generation_protocol(protocol_path)
  if isinstance(resolved_config, (str, Path)):
    config = _load_yaml(
      Path(resolved_config).expanduser().resolve(),
      context='resolved generation config')
  else:
    config = _plain(resolved_config, context='resolved generation config')
  manifest = _plain(manifest, context='generation manifest')

  repository = manifest.get('repository')
  if not isinstance(repository, Mapping):
    raise TypeError('generation repository provenance must be an object')
  repository_sha = _lower_hex(
    repository.get('git_sha'), 40, context='generation repository Git SHA')
  if (protocol['generation']['repository']['require_clean']
      and repository.get('dirty') is not False):
    raise ValueError('generation repository must be clean')

  artifacts = manifest.get('artifacts')
  if not isinstance(artifacts, Mapping):
    raise TypeError('generation artifacts must be an object')
  backbone = artifacts.get('backbone_checkpoint')
  adapter = artifacts.get('structured_adapter')
  if not isinstance(backbone, Mapping) or not isinstance(adapter, Mapping):
    raise TypeError('generation backbone and adapter identities are required')
  for field, expected in protocol['artifacts']['backbone_checkpoint'].items():
    if backbone.get(field) != expected:
      raise ValueError(f'backbone {field} differs from the frozen protocol')
  semantic = adapter.get('semantic_identity')
  if not isinstance(semantic, Mapping):
    raise TypeError('structured adapter semantic identity must be an object')
  for field in ('path', 'manifest_path'):
    value = adapter.get(field)
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
      raise ValueError(f'structured adapter {field} must be an absolute path')
  for field in ('sha256', 'manifest_sha256', 'identity_sha256'):
    _lower_hex(
      adapter.get(field), 64, context=f'structured adapter {field}')
  if adapter['identity_sha256'] != canonical_sha256(semantic):
    raise ValueError('structured adapter semantic identity SHA256 mismatch')
  control = semantic.get('control_identity')
  if expected_control is not None and control != expected_control:
    raise ValueError(
      f'expected adapter control {expected_control!r}, found {control!r}')
  if control not in protocol['arms']:
    raise ValueError(f'adapter control {control!r} is outside the protocol')
  arm = protocol['arms'][control]

  candidate_top_k = _positive_int(
    candidate_top_k, context='candidate_top_k')
  allowed_k = protocol['parameters']['candidate_top_k']['allowed_values']
  if candidate_top_k not in allowed_k:
    raise ValueError(
      f'candidate_top_k {candidate_top_k} is outside {allowed_k}')
  if semantic.get('candidate_top_k') != candidate_top_k:
    raise ValueError('adapter candidate K differs from the requested protocol')

  prompts = manifest.get('prompts')
  if not isinstance(prompts, Mapping):
    raise TypeError('generation prompt identity must be an object')
  bundle = prompts.get('bundle_identity')
  if not isinstance(bundle, Mapping) or not isinstance(
      bundle.get('data_config'), Mapping):
    raise TypeError('generation prompt bundle data identity is required')
  dataset_name = bundle['data_config'].get('name')
  if dataset_name not in protocol['datasets']:
    raise ValueError(
      f'dataset config {dataset_name!r} is outside the frozen protocol')
  dataset = protocol['datasets'][dataset_name]

  expected_config = materialize_resolved_config(
    protocol, dataset_config=dataset_name,
    candidate_top_k=candidate_top_k)
  expected_semantic = _semantic_identity_from_projection(
    expected_config, control=control, arm=arm)
  _assert_exact(
    semantic, expected_semantic,
    context='structured adapter semantic identity')

  adapter_bindings = {
    ('eval', 'adapter_checkpoint'): adapter.get('path'),
    ('eval', 'adapter_sha256'): adapter.get('sha256'),
    ('eval', 'adapter_manifest'): adapter.get('manifest_path'),
    ('eval', 'adapter_manifest_sha256'): adapter.get('manifest_sha256'),
  }
  arm_bindings = {
    ('model', 'structured_decoder', 'topology_mode'):
      arm['topology_mode'],
    ('model', 'structured_decoder', 'factor_mode'):
      arm['factor_mode'],
    ('model', 'structured_decoder', 'training', 'topology_weight'):
      arm['topology_weight'],
  }
  normalized = copy.deepcopy(config)
  for path, expected in {**adapter_bindings, **arm_bindings}.items():
    observed = _path_value(config, path, context='resolved generation config')
    if observed != expected or type(observed) is not type(expected):
      raise ValueError(
        f'resolved generation config {".".join(path)} differs from its '
        'manifest-bound identity')
    _remove_path(normalized, path, context='resolved generation config')
  _assert_exact(
    normalized, expected_config,
    context='normalized resolved generation config')

  _validate_prompt_identity(
    prompts, dataset_name=dataset_name, dataset=dataset,
    generation=protocol['generation'], repository_sha=repository_sha)

  pairing = manifest.get('pairing')
  if not isinstance(pairing, Mapping):
    raise TypeError('generation pairing identity must be an object')
  expected_pairing = {
    'base_seed': dataset['base_seed'],
    'batch_size': protocol['generation']['batch_size'],
    'global_num_samples': dataset['global_num_samples'],
    'num_shards': protocol['generation']['num_shards'],
    'sequence_length': protocol['generation']['sequence_length'],
  }
  for field, expected in expected_pairing.items():
    if pairing.get(field) != expected or type(pairing.get(field)) is not type(
        expected):
      raise ValueError(
        f'generation pairing {field} differs from the frozen protocol')
  shard_index = pairing.get('shard_index')
  if (not isinstance(shard_index, int) or isinstance(shard_index, bool)
      or not 0 <= shard_index < pairing['num_shards']):
    raise ValueError('generation shard_index is invalid')
  expected_shard_samples = (
    (pairing['global_num_samples'] - 1 - shard_index)
    // pairing['num_shards'] + 1)
  if pairing.get('shard_num_samples') != expected_shard_samples:
    raise ValueError('generation shard sample count is incomplete')

  matrix = manifest.get('matrix')
  if not isinstance(matrix, Mapping):
    raise TypeError('generation matrix identity must be an object')
  if matrix.get('sampling_modes') != arm['sampling_modes']:
    raise ValueError('generation sampling modes differ from the frozen arm')
  if matrix.get('nfe_budgets') != protocol['generation']['nfe_budgets']:
    raise ValueError('generation NFE budgets differ from the frozen protocol')
  expected_records = (
    expected_shard_samples * len(arm['sampling_modes'])
    * len(protocol['generation']['nfe_budgets']))
  if matrix.get('num_output_records') != expected_records:
    raise ValueError('generation output record count is incomplete')

  reference_lm = manifest.get('reference_lm')
  if not isinstance(reference_lm, Mapping):
    raise TypeError('frozen reference-LM scoring is required')
  reference_spec = protocol['generation']['reference_lm']
  for field in ('model_name_or_path', 'revision', 'sequence_policy'):
    expected = reference_spec[field]
    if reference_lm.get(field) != expected:
      raise ValueError(
        f'reference LM {field} differs from the frozen protocol')
  runtime_identity = reference_lm.get('runtime_identity')
  if not isinstance(runtime_identity, Mapping):
    raise TypeError('reference LM runtime_identity is required')
  expected_runtime_fields = (
    set(reference_spec['runtime_configuration'])
    | {'schema_version', 'python', 'torch', 'cuda_runtime',
       'transformers', 'tokenizers'})
  _strict_fields(
    runtime_identity, expected_runtime_fields,
    context='reference LM runtime identity')
  if runtime_identity['schema_version'] != 1:
    raise ValueError('unsupported reference-LM runtime identity')
  for field, expected in reference_spec['runtime_configuration'].items():
    if (runtime_identity.get(field) != expected
        or type(runtime_identity.get(field)) is not type(expected)):
      raise ValueError(
        f'reference LM runtime {field} differs from the frozen protocol')
  for field in ('python', 'torch', 'transformers', 'tokenizers'):
    if not isinstance(runtime_identity[field], str) or not \
        runtime_identity[field].strip():
      raise ValueError(f'reference LM runtime {field} must be non-empty')
  cuda_runtime = runtime_identity['cuda_runtime']
  if (runtime_identity['device'].split(':', 1)[0] == 'cuda'
      and (not isinstance(cuda_runtime, str) or not cuda_runtime.strip())):
    raise ValueError('reference LM CUDA runtime identity is required')

  host = manifest.get('host')
  if not isinstance(host, Mapping):
    raise TypeError('generation host identity must be an object')
  expected_host = protocol['generation']['host']
  for field in ('device', 'precision_policy', 'parameter_dtypes'):
    if host.get(field) != expected_host[field]:
      raise ValueError(
        f'generation host {field} differs from the frozen protocol')
  if host.get('packages') != expected_host['critical_packages']:
    raise ValueError(
      'generation critical package versions differ from the frozen protocol')

  return {
    'schema_version': 1,
    'protocol_id': PROTOCOL_ID,
    'protocol_path': str(protocol_path),
    'protocol_sha256': sha256_file(protocol_path),
    'dataset_config': dataset_name,
    'logical_validation_dataset': dataset['logical_validation_dataset'],
    'candidate_top_k': candidate_top_k,
    'control_identity': control,
    'normalized_resolved_config_sha256': canonical_sha256(normalized),
    'prompt_runtime_specification_sha256': dataset[
      'runtime_specification_sha256'],
    'reference_lm_runtime_identity_sha256': canonical_sha256(
      runtime_identity),
  }


__all__ = [
  'DEFAULT_GENERATION_PROTOCOL_PATH',
  'canonical_sha256',
  'load_generation_protocol',
  'materialize_resolved_config',
  'validate_generation_protocol',
]
