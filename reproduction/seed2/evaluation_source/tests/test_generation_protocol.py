import unittest

from evaluation.generation_protocol import (
  canonical_sha256,
  load_generation_protocol,
  materialize_resolved_config,
  validate_generation_protocol,
)


def _set_path(payload, path, value):
  current = payload
  for field in path[:-1]:
    current = current[field]
  current[path[-1]] = value


def _semantic_identity(config, *, control, arm):
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


def _fixture(
    *, dataset_name='eval_wikitext103_pinned', candidate_k=128,
    control='dynamic_dynamic'):
  protocol = load_generation_protocol()
  dataset = protocol['datasets'][dataset_name]
  arm = protocol['arms'][control]
  config = materialize_resolved_config(
    protocol, dataset_config=dataset_name, candidate_top_k=candidate_k)
  semantic = _semantic_identity(config, control=control, arm=arm)
  adapter = {
    'path': f'/mnt/contextual-forest/adapters/{control}-k{candidate_k}.safetensors',
    'sha256': 'a' * 64,
    'manifest_path': (
      f'/mnt/contextual-forest/adapters/{control}-k{candidate_k}.json'),
    'manifest_sha256': 'b' * 64,
    'identity_sha256': canonical_sha256(semantic),
    'semantic_identity': semantic,
  }
  bindings = {
    ('eval', 'adapter_checkpoint'): adapter['path'],
    ('eval', 'adapter_sha256'): adapter['sha256'],
    ('eval', 'adapter_manifest'): adapter['manifest_path'],
    ('eval', 'adapter_manifest_sha256'): adapter['manifest_sha256'],
    ('model', 'structured_decoder', 'topology_mode'):
      arm['topology_mode'],
    ('model', 'structured_decoder', 'factor_mode'):
      arm['factor_mode'],
    ('model', 'structured_decoder', 'training', 'topology_weight'):
      arm['topology_weight'],
  }
  for path, value in bindings.items():
    current = config
    for field in path[:-1]:
      current = current[field]
    current[path[-1]] = value

  repository_sha = 'd' * 40
  prompt_sha = 'e' * 64
  prompt_manifest_sha = 'f' * 64
  generation = protocol['generation']
  bundle = {
    'schema_version': 2,
    'artifact': 'pinned_document_local_infilling_prompts',
    'manifest_sha256': prompt_manifest_sha,
    'builder_git_sha': repository_sha,
    'data_config': {
      'name': dataset_name,
      'sha256': dataset['data_config_sha256'],
      'logical_validation_dataset': dataset[
        'logical_validation_dataset'],
      'dataset_revision': dataset['dataset_revision'],
      'tokenizer_name_or_path': dataset['tokenizer_name_or_path'],
      'tokenizer_revision': dataset['tokenizer_revision'],
    },
    'runtime_provenance': {
      'sha256': '1' * 64,
      'specification_sha256': dataset['runtime_specification_sha256'],
      'manifest_sha256': '2' * 64,
    },
    'policy': {
      **generation['prompt_policy'],
      'selection_seed': generation['selection_seed'],
      'span_length': generation['span_length'],
      'sequence_length': generation['sequence_length'],
    },
    'output': {
      'sha256': prompt_sha,
      'size_bytes': 123456,
      'num_prompts': dataset['num_prompts'],
    },
  }
  shard_index = 0
  shard_samples = (
    (dataset['global_num_samples'] - 1 - shard_index)
    // generation['num_shards'] + 1)
  manifest = {
    'repository': {'git_sha': repository_sha, 'dirty': False},
    'artifacts': {
      'backbone_checkpoint': {
        **protocol['artifacts']['backbone_checkpoint'],
        'size_bytes': 123,
      },
      'structured_adapter': adapter,
    },
    'prompts': {
      'source': 'jsonl',
      'path': '/mnt/contextual-forest/prompts/prompts.jsonl',
      'sha256': prompt_sha,
      'num_prompt_records': dataset['num_prompts'],
      'manifest_path': '/mnt/contextual-forest/prompts/prompts.manifest.json',
      'manifest_sha256': prompt_manifest_sha,
      'bundle_identity': bundle,
    },
    'pairing': {
      'base_seed': dataset['base_seed'],
      'batch_size': generation['batch_size'],
      'global_num_samples': dataset['global_num_samples'],
      'shard_num_samples': shard_samples,
      'num_shards': generation['num_shards'],
      'shard_index': shard_index,
      'sequence_length': generation['sequence_length'],
    },
    'matrix': {
      'sampling_modes': arm['sampling_modes'],
      'nfe_budgets': generation['nfe_budgets'],
      'num_output_records': (
        shard_samples * len(arm['sampling_modes'])
        * len(generation['nfe_budgets'])),
    },
    'reference_lm': {
      'model_name_or_path': generation['reference_lm']['model_name_or_path'],
      'revision': generation['reference_lm']['revision'],
      'sequence_policy': generation['reference_lm']['sequence_policy'],
      'runtime_identity': {
        'schema_version': 1,
        **generation['reference_lm']['runtime_configuration'],
        'python': '3.10.14',
        'torch': '2.5.1+cu121',
        'cuda_runtime': '12.1',
        'transformers': '4.38.2',
        'tokenizers': '0.15.2',
      },
      'num_scored_sequences': 100,
      'num_scored_tokens': 1000,
      'mean_nll_nats': 3.0,
      'perplexity': 20.085536923187668,
    },
    'host': {
      'device': generation['host']['device'],
      'precision_policy': generation['host']['precision_policy'],
      'parameter_dtypes': generation['host']['parameter_dtypes'],
      'packages': generation['host']['critical_packages'],
    },
  }
  return config, manifest


class GenerationProtocolTest(unittest.TestCase):

  def test_accepts_each_dataset_and_both_frozen_arms(self):
    cases = (
      ('eval_wikitext103_pinned', 64, 'dynamic_dynamic'),
      ('eval_scientific_papers_arxiv_pinned', 128, 'static_static'),
      ('eval_scientific_papers_pubmed_pinned', 256, 'dynamic_dynamic'),
    )
    for dataset_name, candidate_k, control in cases:
      with self.subTest(dataset=dataset_name, control=control):
        config, manifest = _fixture(
          dataset_name=dataset_name, candidate_k=candidate_k,
          control=control)
        identity = validate_generation_protocol(
          config, manifest, candidate_top_k=candidate_k,
          expected_control=control)
        self.assertEqual(identity['dataset_config'], dataset_name)
        self.assertEqual(identity['candidate_top_k'], candidate_k)
        self.assertEqual(identity['control_identity'], control)
        self.assertEqual(len(identity['normalized_resolved_config_sha256']), 64)
        self.assertEqual(
          len(identity['reference_lm_runtime_identity_sha256']), 64)

  def test_rejects_drift_anywhere_in_complete_resolved_config(self):
    mutations = {
      'sampler': (('sampling', 'predictor'), 'analytic'),
      'noise': (('noise', 'sigma_max'), 21),
      'model': (('model', 'dropout'), 0.2),
      'eval': (('eval', 'disable_ema'), False),
      'batch': (('loader', 'eval_batch_size'), 4),
      'precision': (('trainer', 'precision'), '32-true'),
    }
    for name, (path, value) in mutations.items():
      with self.subTest(field=name):
        config, manifest = _fixture()
        _set_path(config, path, value)
        with self.assertRaisesRegex(ValueError, 'frozen protocol'):
          validate_generation_protocol(
            config, manifest, candidate_top_k=128)

    config, manifest = _fixture()
    config['unexpected'] = True
    with self.assertRaisesRegex(ValueError, 'keys differ'):
      validate_generation_protocol(config, manifest, candidate_top_k=128)

  def test_adapter_and_arm_fields_are_bound_before_normalization(self):
    config, manifest = _fixture()
    config['eval']['adapter_checkpoint'] = '/tmp/other.safetensors'
    with self.assertRaisesRegex(ValueError, 'manifest-bound identity'):
      validate_generation_protocol(config, manifest, candidate_top_k=128)

    config, manifest = _fixture()
    config['model']['structured_decoder']['topology_mode'] = 'fixed'
    with self.assertRaisesRegex(ValueError, 'manifest-bound identity'):
      validate_generation_protocol(config, manifest, candidate_top_k=128)

  def test_rejects_prompt_data_tokenizer_and_boundary_drift(self):
    mutations = {
      'data config SHA': (
        ('data_config', 'sha256'), '3' * 64),
      'dataset revision': (
        ('data_config', 'dataset_revision'), '4' * 40),
      'tokenizer revision': (
        ('data_config', 'tokenizer_revision'), '5' * 40),
      'runtime boundary specification': (
        ('runtime_provenance', 'specification_sha256'), '6' * 64),
      'prompt boundary policy': (
        ('policy', 'boundary_policy'), 'mask_any_position'),
    }
    for name, (path, value) in mutations.items():
      with self.subTest(field=name):
        config, manifest = _fixture()
        bundle = manifest['prompts']['bundle_identity']
        _set_path(bundle, path, value)
        with self.assertRaises(ValueError):
          validate_generation_protocol(
            config, manifest, candidate_top_k=128)

  def test_candidate_k_is_the_only_non_dataset_config_parameter(self):
    config, manifest = _fixture(candidate_k=128)
    with self.assertRaisesRegex(ValueError, 'candidate K differs'):
      validate_generation_protocol(
        config, manifest, candidate_top_k=64)
    with self.assertRaisesRegex(ValueError, 'outside'):
      validate_generation_protocol(
        config, manifest, candidate_top_k=512)

  def test_rejects_dirty_or_different_prompt_builder_checkout(self):
    config, manifest = _fixture()
    manifest['repository']['dirty'] = True
    with self.assertRaisesRegex(ValueError, 'must be clean'):
      validate_generation_protocol(config, manifest, candidate_top_k=128)

    config, manifest = _fixture()
    manifest['prompts']['bundle_identity']['builder_git_sha'] = '9' * 40
    with self.assertRaisesRegex(ValueError, 'prompt builder Git SHA'):
      validate_generation_protocol(config, manifest, candidate_top_k=128)

  def test_rejects_reference_lm_tokenizer_and_scoring_runtime_drift(self):
    for field, value in (
        ('tokenizer_revision', '9' * 40),
        ('batch_size', 4),
        ('max_length', 128),
        ('requested_dtype', 'float16')):
      with self.subTest(field=field):
        config, manifest = _fixture()
        manifest['reference_lm']['runtime_identity'][field] = value
        with self.assertRaisesRegex(ValueError, f'runtime {field}'):
          validate_generation_protocol(
            config, manifest, candidate_top_k=128)


if __name__ == '__main__':
  unittest.main()
