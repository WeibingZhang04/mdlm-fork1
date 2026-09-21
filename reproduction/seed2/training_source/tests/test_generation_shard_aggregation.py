import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from evaluation import generation_adapter_comparison
from evaluation.generation_harness import summarize_group
from evaluation.infilling_prompts import deterministic_span_start
from evaluation.generation_metrics import (
  REFERENCE_LM_SEQUENCE_POLICY,
  paired_token_metrics,
  repetition_rate,
)
from evaluation.generation_shard_aggregation import (
  REFERENCE_LM_EXP_REL_TOL,
  _reference_lm_identity,
  _summarize_reference_lm,
  _validate_reference_lm_score,
  aggregate_generation_shards,
  canonical_sha256,
  paired_bootstrap_intervals,
  pairing_digest,
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_specs():
  result = []
  per_prompt = {}
  for sample_index in range(8):
    document_index = sample_index % 2
    document_sha = hashlib.sha256(
      f'document-{document_index}'.encode()).hexdigest()
    span_start = deterministic_span_start(
      dataset_id='wiki-pinned', document_sha256=document_sha,
      chunk_index=0, sequence_length=3, span_length=1,
      selection_seed=11)
    prompt_id = (
      f'wiki-pinned/document-{document_index:09d}/'
      'chunk-00000/span-0001')
    per_prompt.setdefault(prompt_id, 0)
    replicate = per_prompt[prompt_id]
    per_prompt[prompt_id] += 1
    observed = 10 + document_index
    result.append({
      'sample_index': sample_index,
      'pair_key': f'{prompt_id}/replicate-{replicate:04d}',
      'pair_seed': 700 + sample_index,
      'prompt_id': prompt_id,
      'prompt_metadata': {
        'prompt_policy_id': 'document-local-contiguous-span-v1',
        'dataset_id': 'wiki-pinned',
        'source_document_index': document_index,
        'source_document_sha256': document_sha,
        'source_chunk_index': 0,
        'sequence_length': 3,
        'span_start': span_start,
        'span_stop': span_start + 1,
        'span_length': 1,
        'selection_seed': 11,
      },
      'initial_token_ids': [observed, 99, 20 + document_index],
      'active_mask': [False, True, False],
      'reference_token_ids': [observed, 2, 20 + document_index],
    })
  return result


def _prompt_bundle_identity():
  return {
    'schema_version': 2,
    'artifact': 'pinned_document_local_infilling_prompts',
    'manifest_sha256': '1' * 64,
    'builder_git_sha': 'a' * 40,
    'data_config': {
      'name': 'eval_wikitext103_pinned',
      'sha256': '3' * 64,
      'logical_validation_dataset': 'wiki-pinned',
      'dataset_revision': '4' * 40,
      'tokenizer_name_or_path': 'openai-community/gpt2',
      'tokenizer_revision': '5' * 40,
    },
    'runtime_provenance': {
      'sha256': '6' * 64,
      'specification_sha256': '7' * 64,
      'manifest_sha256': '8' * 64,
    },
    'policy': {
      'policy_id': 'document-local-contiguous-span-v1',
      'selection_seed': 11,
      'span_length': 1,
      'sequence_length': 3,
      'record_selection': 'first_n_in_pinned_validation_order',
      'boundary_policy': 'never_mask_first_or_last_token',
    },
    'output': {
      'sha256': 'd' * 64,
      'size_bytes': 100,
      'num_prompts': 2,
    },
  }


def _manifest_reference_lm(records):
  summary = _summarize_reference_lm(records)
  assert summary is not None
  summary['runtime_identity'] = {
    'schema_version': 1,
    'model_name_or_path': 'fixture-reference-lm',
    'model_revision': '6' * 40,
    'model_class': 'fixture.ReferenceModel',
    'model_config_class': 'fixture.ReferenceConfig',
    'tokenizer_name_or_path': 'fixture-reference-lm',
    'tokenizer_revision': '6' * 40,
    'tokenizer_class': 'fixture.ReferenceTokenizer',
    'tokenizer_vocab_size': 100,
    'tokenizer_bos_token_id': 0,
    'tokenizer_eos_token_id': 0,
    'tokenizer_pad_token_id': 0,
    'tokenizer_padding_side': 'right',
    'tokenizer_truncation_side': 'right',
    'tokenization_policy': (
      'fast_tokenizer_right_padding_right_truncation_add_special_tokens_v1'),
    'sequence_policy': REFERENCE_LM_SEQUENCE_POLICY,
    'add_special_tokens': True,
    'batch_size': 4,
    'max_length': 3,
    'requested_dtype': 'float32',
    'parameter_dtypes': ['torch.float32'],
    'precision_policy': (
      'explicit_checkpoint_dtype_no_autocast_float32_cross_entropy_v1'),
    'device': 'cuda:0',
    'python': '3.10.14',
    'torch': '2.5.1+cu121',
    'cuda_runtime': '12.1',
    'transformers': '4.38.2',
    'tokenizers': '0.15.2',
  }
  return summary


def _structured_identity(control='dynamic_dynamic'):
  topology_mode, factor_mode = {
    'dynamic_dynamic': ('dynamic', 'dynamic'),
    'static_static': ('fixed', 'fixed'),
  }[control]
  return {
    'control_identity': control,
    'topology_mode': topology_mode,
    'factor_mode': factor_mode,
    'candidate_top_k': 64,
    'independent_mode': False,
    'topology_weight': 0.1 if control == 'dynamic_dynamic' else 0.0,
    'head_semantics': {
      'rank': 16,
      'time_embed_dim': 64,
      'topology_dim': 64,
      'local_window': 64,
      'num_anchor_slots': 8,
      'contextual_neighbors': 8,
      'component_size_cap': 32,
      'min_edge_score': None,
      'fixed_edges': None,
      'fixed_edge_path': None,
    },
    'training_semantics': {
      'objective_name': 'structured_conditional_nll',
      'factorized_aux_weight': 0.1,
      'topology_strategy': 'teacher_influence',
      'topology_temperature': 1.0,
      'topology_minimum_choices': 2,
      'topology_edge_weight': 1.0,
      'topology_anchor_weight': 1.0,
      'topology_slot_weight': 1.0,
      'topology_on_validation': False,
    },
  }


def _adapter_origin_binding(
    *, control, structured_identity, adapter_path, adapter_sha,
    adapter_manifest_path, adapter_manifest_sha):
  source_checkpoint_sha = '4' * 64
  source = {
    'compiled_plan_path': '/experiments/plan/compiled-plan.json',
    'compiled_plan_sha256': '5' * 64,
    'plan_id': '6' * 64,
    'protocol_id': 'contextual-forest-expansion-v1',
    'source_manifest_sha256': '7' * 64,
    # Adapter training and generation intentionally use distinct clean
    # commits; the adapter origin is replayed rather than conflated with the
    # later generation-runner revision.
    'repository': {'sha': 'f' * 40, 'clean': True},
    'suite': 'candidate_k_128_pilot',
    'candidate_k': structured_identity['candidate_top_k'],
    'train_seed': 1,
    'legacy_plan_schema': False,
  }
  binding = {
    'schema_version': 1,
    'artifact': 'contextual_forest_generation_adapter_origin_binding',
    'evidence_file': {
      'path': '/experiments/adapter-pair-origin.json',
      'sha256': '8' * 64,
      'evidence_sha256': '9' * 64,
    },
    'source': source,
    'arm': control,
    'adapter': {
      'path': adapter_path,
      'sha256': adapter_sha,
      'manifest_path': adapter_manifest_path,
      'manifest_sha256': adapter_manifest_sha,
      'structured_decoder_identity': structured_identity,
      'structured_decoder_identity_sha256': canonical_sha256(
        structured_identity),
      'source_checkpoint_sha256': source_checkpoint_sha,
      'source_checkpoint_global_step': 1000,
      'released_backbone': {
        'repository': 'kuleshov-group/mdlm-owt',
        'revision': 'main',
        'source_sha256': 'b' * 64,
        'source_size_bytes': 1000,
        'tensor_count': 131,
      },
    },
    'plan_export': {
      'train_job_id': f'train--{control}--s001--k064',
      'train_job_spec_sha256': 'c' * 64,
      'train_job_execution_sha256': 'd' * 64,
      'train_success_marker_sha256': 'e' * 64,
      'checkpoint_sha256': source_checkpoint_sha,
      'training_data_provenance_sha256': 'f' * 64,
      'training_validation_data_provenance_sha256': '1' * 64,
      'export_job_id': f'export--{control}--s001--k064',
      'export_job_spec_sha256': '2' * 64,
      'export_job_execution_sha256': '3' * 64,
      'export_success_marker_sha256': '4' * 64,
      'adapter_sha256': adapter_sha,
      'adapter_manifest_sha256': adapter_manifest_sha,
    },
  }
  binding['binding_sha256'] = canonical_sha256(binding)
  return binding


def _record(
    pair,
    *,
    mode,
    budget,
    shard_index,
    global_digest,
    shard_digest,
    batch_seed,
    batch_size,
    with_reference_lm=False,
):
  if mode == 'structured_joint':
    sampled = [
      pair['initial_token_ids'][0], 2, pair['initial_token_ids'][2]]
  elif pair['sample_index'] % 2:
    sampled = [
      pair['initial_token_ids'][0], 5, pair['initial_token_ids'][2]]
  else:
    sampled = [
      pair['initial_token_ids'][0], 2, pair['initial_token_ids'][2]]
  active_values = [
    token for token, active in zip(sampled, pair['active_mask']) if active]
  reference_active = [
    token for token, active in zip(
      pair['reference_token_ids'], pair['active_mask']) if active]
  metrics = {
    'repetition_rate': {
      str(n): repetition_rate(active_values, n=n) for n in (1, 2, 4)
    },
    **paired_token_metrics(active_values, reference_active),
  }
  elapsed = 2.0 + 0.1 * budget + (0.2 if mode == 'structured_joint' else 0.0)
  timing = {
    'batch_seed': batch_seed,
    'batch_size': batch_size,
    'requested_nfe_budget': budget,
    'measured_nfe': budget,
    'wall_clock_seconds': elapsed,
    'active_tokens': batch_size,
    'active_tokens_per_second': batch_size / elapsed,
    'sequence_tokens_per_second': (batch_size * 3) / elapsed,
    'peak_memory_bytes': 100 + budget,
    'unresolved_mask_tokens': 0,
  }
  record = {
    'schema_version': 1,
    **pair,
    'sampling_mode': mode,
    'requested_nfe_budget': budget,
    'measured_nfe': budget,
    'batch_seed': timing['batch_seed'],
    'sample_token_ids': sampled,
    'sample_active_token_ids': active_values,
    'text': ' '.join(str(value) for value in sampled),
    'metrics': metrics,
    'timing': timing,
    'global_pairing_digest': global_digest,
    'shard_pairing_digest': shard_digest,
    'num_shards': 2,
    'shard_index': shard_index,
  }
  if with_reference_lm:
    record['reference_lm'] = {
      'model_name_or_path': 'fixture-reference-lm',
      'revision': '6' * 40,
      'sequence_policy': REFERENCE_LM_SEQUENCE_POLICY,
      'token_count': 2,
      'mean_nll_nats': 1.0,
      'perplexity': math.e,
    }
  return record


def _write_shards(
    root: Path,
    *,
    control='dynamic_dynamic',
    modes=('factorized', 'structured_joint'),
    with_reference_lm=False,
):
  root.mkdir(parents=True, exist_ok=True)
  pair_specs = _pair_specs()
  global_digest = pairing_digest(pair_specs)
  shard_dirs = []
  for shard_index in range(2):
    shard_dir = root / f'shard-{shard_index}'
    shard_dir.mkdir()
    shard_pairs = [
      pair for pair in pair_specs
      if pair['sample_index'] % 2 == shard_index
    ]
    batch_seed = int(canonical_sha256([
      {'pair_key': pair['pair_key'], 'pair_seed': pair['pair_seed']}
      for pair in shard_pairs
    ])[:16], 16) % (2 ** 63 - 1)
    shard_digest = pairing_digest(shard_pairs)
    records = []
    groups = []
    for mode in modes:
      for budget in (2, 4):
        group = [
          _record(
            pair, mode=mode, budget=budget, shard_index=shard_index,
            global_digest=global_digest, shard_digest=shard_digest,
            batch_seed=batch_seed, batch_size=len(shard_pairs),
            with_reference_lm=with_reference_lm)
          for pair in shard_pairs
        ]
        summary = summarize_group(group)
        summary['input_pairing_digest'] = shard_digest
        if with_reference_lm:
          summary['reference_lm'] = _summarize_reference_lm(group)
        groups.append(summary)
        records.extend(group)

    samples_path = shard_dir / 'samples.jsonl'
    samples_path.write_text(''.join(
      json.dumps(record, sort_keys=True) + '\n' for record in records))
    summary_payload = {
      'schema_version': 1,
      'experiment': 'paired_contextual_forest_generation_pilot',
      'global_pairing_digest': global_digest,
      'input_pairing_digest': shard_digest,
      'global_num_paired_samples': 8,
      'num_paired_samples': 4,
      'shard_index': shard_index,
      'num_shards': 2,
      'groups': groups,
      'reference_lm': (
        _manifest_reference_lm(records) if with_reference_lm else None),
    }
    summary_path = shard_dir / 'summary.json'
    summary_path.write_text(json.dumps(
      summary_payload, indent=2, sort_keys=True) + '\n')
    structured_identity = _structured_identity(control)
    adapter_sha = 'c' * 64 if control == 'dynamic_dynamic' else '9' * 64
    adapter_manifest_sha = (
      'e' * 64 if control == 'dynamic_dynamic' else 'a' * 64)
    adapter_path = f'/models/{control}.safetensors'
    adapter_manifest_path = f'/models/{control}.manifest.json'
    config_path = shard_dir / 'resolved_config.yaml'
    config_path.write_text(yaml.safe_dump({
      'model': {
        'structured_decoder': {
          'topology_mode': structured_identity['topology_mode'],
          'factor_mode': structured_identity['factor_mode'],
          'top_k': structured_identity['candidate_top_k'],
          'training': {
            'topology_weight': structured_identity['topology_weight'],
          },
        },
      },
      'eval': {
        'adapter_checkpoint': adapter_path,
        'adapter_sha256': adapter_sha,
        'adapter_manifest': adapter_manifest_path,
        'adapter_manifest_sha256': adapter_manifest_sha,
      },
      'data': {'name': 'fixture'},
      'length': 3,
    }, sort_keys=True))
    per_shard_records = len(shard_pairs) * len(modes) * 2
    manifest = {
      'schema_version': 1,
      'experiment': 'paired_contextual_forest_generation_pilot',
      'scientific_scope': 'test fixture',
      'command': ['test', '--shard-index', str(shard_index)],
      'start_time_utc': '2026-08-30T00:00:00+00:00',
      'end_time_utc': '2026-08-30T00:01:00+00:00',
      'duration_seconds': 60.0,
      'host': {
        'hostname': f'worker-{shard_index}',
        'platform': 'Linux-6.8-x86_64',
        'python': '3.10.14',
        'torch': '2.5.1+cu121',
        'cuda_runtime': '12.1',
        'device': 'cuda:0',
        'gpu': 'NVIDIA L4',
        'parameter_dtypes': ['torch.float32'],
        'precision_policy': (
          'checkpoint dtype; DIT-managed bf16 autocast on CUDA'),
        'packages': {
          'numpy': '2.1.3',
          'safetensors': '0.4.5',
          'tokenizers': '0.15.2',
          'transformers': '4.38.2',
        },
      },
      'repository': {
        'git_sha': 'a' * 40,
        'dirty': False,
        'status_porcelain': [],
        'tracked_diff_sha256': hashlib.sha256(b'').hexdigest(),
        'untracked_files': [],
        'dirty_content_sha256': hashlib.sha256(b'').hexdigest(),
      },
      'artifacts': {
        'backbone_checkpoint': {
          'path': '/models/backbone.pt',
          'sha256': 'b' * 64,
          'size_bytes': 1000,
        },
        'structured_adapter': {
          'path': adapter_path,
          'sha256': adapter_sha,
          'size_bytes': 200,
          'manifest_path': adapter_manifest_path,
          'manifest_sha256': adapter_manifest_sha,
          'identity_sha256': canonical_sha256(structured_identity),
          'semantic_identity': structured_identity,
        },
      },
      'adapter_origin_evidence': _adapter_origin_binding(
        control=control,
        structured_identity=structured_identity,
        adapter_path=adapter_path,
        adapter_sha=adapter_sha,
        adapter_manifest_path=adapter_manifest_path,
        adapter_manifest_sha=adapter_manifest_sha),
      'prompts': {
        'source': 'jsonl',
        'path': '/inputs/prompts.jsonl',
        'sha256': 'd' * 64,
        'num_prompt_records': 2,
        'manifest_path': '/inputs/prompts.jsonl.manifest.json',
        'manifest_sha256': '1' * 64,
        'bundle_identity': _prompt_bundle_identity(),
      },
      'pairing': {
        'digest_algorithm': 'sha256-canonical-json-v2-prompt-metadata',
        'global_pairing_digest': global_digest,
        'shard_pairing_digest': shard_digest,
        'base_seed': 700,
        'batch_size': 4,
        'global_num_samples': 8,
        'shard_num_samples': 4,
        'num_shards': 2,
        'shard_index': shard_index,
        'sequence_length': 3,
      },
      'spot_interruption_policy': {'resume_supported': False},
      'matrix': {
        'sampling_modes': list(modes),
        'nfe_budgets': [2, 4],
        'num_output_records': per_shard_records,
      },
      'outputs': {
        'samples_jsonl': {
          'path': 'samples.jsonl',
          'sha256': _sha256(samples_path),
          'num_records': per_shard_records,
        },
        'summary_json': {
          'path': 'summary.json',
          'sha256': _sha256(summary_path),
        },
        'resolved_config': {
          'path': 'resolved_config.yaml',
          'sha256': _sha256(config_path),
        },
      },
      'reference_lm': (
        _manifest_reference_lm(records) if with_reference_lm else None),
    }
    (shard_dir / 'manifest.json').write_text(json.dumps(
      manifest, indent=2, sort_keys=True) + '\n')
    shard_dirs.append(shard_dir)
  return shard_dirs


def _read_manifest(shard_dir: Path):
  return json.loads((shard_dir / 'manifest.json').read_text())


def _write_manifest(shard_dir: Path, manifest):
  (shard_dir / 'manifest.json').write_text(json.dumps(
    manifest, indent=2, sort_keys=True) + '\n')


def _rewrite_samples(shard_dir: Path, records):
  samples_path = shard_dir / 'samples.jsonl'
  samples_path.write_text(''.join(
    json.dumps(record, sort_keys=True) + '\n' for record in records))
  manifest = _read_manifest(shard_dir)
  manifest['outputs']['samples_jsonl']['sha256'] = _sha256(samples_path)
  _write_manifest(shard_dir, manifest)


class GenerationShardAggregationTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.shards = _write_shards(self.root)

  def tearDown(self):
    self.temporary.cleanup()

  def test_complete_union_recomputes_metrics_and_clusters_repeated_prompts(self):
    result = aggregate_generation_shards(
      self.shards,
      bootstrap_resamples=200,
      bootstrap_seed=19,
      timestamp_utc='2026-08-30T02:00:00+00:00')

    coverage = result['coverage']
    self.assertEqual(coverage['global_num_paired_draws'], 8)
    self.assertEqual(coverage['num_unique_prompts'], 2)
    self.assertEqual(
      sorted(coverage['paired_draws_per_prompt'].values()), [4, 4])
    self.assertEqual(coverage['verified_output_records'], 32)
    self.assertEqual(
      result['identity']['runtime']['cuda_runtime'], '12.1')
    self.assertEqual(
      result['identity']['runtime']['platform'], 'Linux-6.8-x86_64')
    self.assertEqual(
      result['identity']['runtime']['packages']['transformers'], '4.38.2')
    self.assertEqual(
      result['timing_policy']['hardware_identity']['gpu'], 'NVIDIA L4')
    self.assertEqual(len(result['groups']), 4)
    structured = next(
      group for group in result['groups']
      if group['sampling_mode'] == 'structured_joint'
      and group['requested_nfe_budget'] == 2)
    self.assertEqual(structured['reference']['token_accuracy'], 1.0)
    self.assertEqual(
      structured['timing']['inferential_status'], 'descriptive_only')
    comparison = next(
      item for item in result['comparisons']
      if item['comparison_kind'] == 'sampling_mode_at_fixed_nfe'
      and item['baseline']['requested_nfe_budget'] == 2)
    accuracy = comparison['endpoints']['reference_token_accuracy']
    self.assertAlmostEqual(
      accuracy['paired_draws']['point_estimate'], 0.5)
    self.assertEqual(accuracy['paired_draws']['num_paired_draws'], 8)
    self.assertEqual(
      accuracy['prompt_clusters']['num_prompt_clusters'], 2)
    self.assertEqual(
      accuracy['prompt_clusters']['draws_per_prompt'],
      coverage['paired_draws_per_prompt'])

  def test_rejects_missing_or_duplicate_shard_coverage(self):
    with self.assertRaisesRegex(ValueError, 'incomplete shard coverage'):
      aggregate_generation_shards(
        self.shards[:1], bootstrap_resamples=5)
    with self.assertRaisesRegex(ValueError, 'duplicate shard path'):
      aggregate_generation_shards(
        [self.shards[0], self.shards[0]], bootstrap_resamples=5)

  def test_rejects_clean_repository_and_cross_shard_identity_drift(self):
    manifest = _read_manifest(self.shards[1])
    manifest['repository']['dirty'] = True
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'clean repository'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'second-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['artifacts']['structured_adapter']['sha256'] = 'e' * 64
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'adapter_origin_evidence differs'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'adapter-manifest-fixture')
    manifest = _read_manifest(self.shards[1])
    structured_identity = _structured_identity('static_static')
    manifest['artifacts']['structured_adapter'][
      'semantic_identity'] = structured_identity
    manifest['artifacts']['structured_adapter'][
      'identity_sha256'] = canonical_sha256(structured_identity)
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'adapter_origin_evidence differs'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_runtime_drift_and_gpu_timing_pooling(self):
    manifest = _read_manifest(self.shards[1])
    manifest['host']['torch'] = '2.6.0+cu124'
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'gpu-drift-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['host']['gpu'] = 'NVIDIA A100-SXM4-80GB'
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'refusing to pool timing'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_platform_or_critical_package_runtime_drift(self):
    manifest = _read_manifest(self.shards[1])
    manifest['host']['platform'] = 'Windows-11-x86_64'
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'package-drift-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['host']['packages']['transformers'] = '4.50.0'
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_reference_lm_scoring_runtime_drift(self):
    self.shards = _write_shards(
      self.root / 'reference-runtime-fixture', with_reference_lm=True)
    shard = self.shards[1]
    manifest = _read_manifest(shard)
    summary_path = shard / 'summary.json'
    summary = json.loads(summary_path.read_text())
    for payload in (manifest['reference_lm'], summary['reference_lm']):
      payload['runtime_identity']['max_length'] = 4
    summary_path.write_text(json.dumps(
      summary, indent=2, sort_keys=True) + '\n')
    manifest['outputs']['summary_json']['sha256'] = _sha256(summary_path)
    _write_manifest(shard, manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_resolved_config_and_prompt_identity_drift(self):
    config_path = self.shards[1] / 'resolved_config.yaml'
    config_path.write_text('model: different\nlength: 3\n')
    manifest = _read_manifest(self.shards[1])
    manifest['outputs']['resolved_config']['sha256'] = _sha256(config_path)
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'cross_shard_identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_adapter_manifest_path_is_not_part_of_cryptographic_identity(self):
    manifest = _read_manifest(self.shards[1])
    manifest['artifacts']['structured_adapter']['manifest_path'] = (
      '/a/different/mount/adapter.manifest.json')
    _write_manifest(self.shards[1], manifest)
    result = aggregate_generation_shards(
      self.shards, bootstrap_resamples=5)
    self.assertNotIn(
      'manifest_path', result['identity']['artifacts']['structured_adapter'])

  def test_rejects_even_tiny_deterministic_metric_mismatches(self):
    summary_path = self.shards[1] / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['groups'][0]['reference']['token_accuracy'] += 5e-7
    summary_path.write_text(json.dumps(
      summary, indent=2, sort_keys=True) + '\n')
    manifest = _read_manifest(self.shards[1])
    manifest['outputs']['summary_json']['sha256'] = _sha256(summary_path)
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'differs from recomputed'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'record-metric-fixture')
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0]['metrics']['reference_token_accuracy'] += 5e-7
    _rewrite_samples(self.shards[0], records)
    with self.assertRaisesRegex(ValueError, 'differs from recomputed'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'prompt-identity-fixture')
    manifest = _read_manifest(self.shards[1])
    manifest['prompts']['sha256'] = 'f' * 64
    _write_manifest(self.shards[1], manifest)
    with self.assertRaisesRegex(ValueError, 'prompt bundle identity'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_hash_count_and_pairing_corruption(self):
    samples_path = self.shards[0] / 'samples.jsonl'
    samples_path.write_text(samples_path.read_text() + '{}\n')
    with self.assertRaisesRegex(ValueError, 'SHA256'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'third-fixture')
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[-1]['pair_key'] = 'wrong-pair'
    samples_path.write_text(''.join(
      json.dumps(record, sort_keys=True) + '\n' for record in records))
    manifest = _read_manifest(self.shards[0])
    manifest['outputs']['samples_jsonl']['sha256'] = _sha256(samples_path)
    _write_manifest(self.shards[0], manifest)
    with self.assertRaisesRegex(ValueError, 'pairing digest mismatch'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_unresolved_masks_even_with_updated_file_hash(self):
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0]['sample_token_ids'][1] = 99
    records[0]['sample_active_token_ids'][0] = 99
    samples_path.write_text(''.join(
      json.dumps(record, sort_keys=True) + '\n' for record in records))
    manifest = _read_manifest(self.shards[0])
    manifest['outputs']['samples_jsonl']['sha256'] = _sha256(samples_path)
    _write_manifest(self.shards[0], manifest)
    with self.assertRaisesRegex(ValueError, 'unresolved mask'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_forged_batch_seed_and_noncanonical_batch_order(self):
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0]['batch_seed'] += 1
    records[0]['timing']['batch_seed'] += 1
    _rewrite_samples(self.shards[0], records)
    with self.assertRaisesRegex(ValueError, 'does not commit'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'batch-order-fixture')
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0], records[1] = records[1], records[0]
    _rewrite_samples(self.shards[0], records)
    with self.assertRaisesRegex(ValueError, 'ascending modulo-assigned'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

    self.shards = _write_shards(self.root / 'partial-batch-fixture')
    samples_path = self.shards[0] / 'samples.jsonl'
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]
    records[0]['timing']['batch_size'] = 3
    _rewrite_samples(self.shards[0], records)
    with self.assertRaisesRegex(ValueError, 'final partial batch'):
      aggregate_generation_shards(self.shards, bootstrap_resamples=5)

  def test_rejects_wrong_reference_lm_sequence_policy_exactly(self):
    score = {
      'model_name_or_path': 'gpt2-large',
      'revision': 'a' * 40,
      'sequence_policy': 'some_other_nonempty_policy',
      'token_count': 1,
      'mean_nll_nats': 1.0,
      'perplexity': 2.718281828459045,
    }
    with self.assertRaisesRegex(ValueError, 'sequence_policy must equal'):
      _validate_reference_lm_score(score, context='score')
    with self.assertRaisesRegex(ValueError, 'sequence_policy must equal'):
      _reference_lm_identity({
        'model_name_or_path': score['model_name_or_path'],
        'revision': score['revision'],
        'sequence_policy': score['sequence_policy'],
        'mean_nll_nats': score['mean_nll_nats'],
        'perplexity': score['perplexity'],
        'runtime_identity': _manifest_reference_lm([
          {'reference_lm': {
            **score,
            'sequence_policy': REFERENCE_LM_SEQUENCE_POLICY,
          }}
        ])['runtime_identity'],
        'num_scored_sequences': 1,
        'num_scored_tokens': 1,
      }, context='manifest.reference_lm')

    score['sequence_policy'] = REFERENCE_LM_SEQUENCE_POLICY
    self.assertEqual(
      _validate_reference_lm_score(score, context='score')['sequence_policy'],
      REFERENCE_LM_SEQUENCE_POLICY)

    # The only numeric tolerance is the explicitly bounded exp(mean NLL)
    # redundancy check for a reference-LM score.
    score['perplexity'] *= 1.0 + REFERENCE_LM_EXP_REL_TOL / 2.0
    _validate_reference_lm_score(score, context='score')
    score['perplexity'] *= 1.0 + REFERENCE_LM_EXP_REL_TOL * 4.0
    with self.assertRaisesRegex(ValueError, 'inconsistent with mean NLL'):
      _validate_reference_lm_score(score, context='score')

  def test_prompt_cluster_bootstrap_reports_draws_separately(self):
    intervals = paired_bootstrap_intervals(
      {0: 0.0, 1: 2.0, 2: 0.0, 3: 2.0},
      {0: 'a', 1: 'b', 2: 'a', 3: 'b'},
      num_resamples=100,
      rng_seed=3)
    self.assertEqual(intervals['paired_draws']['num_paired_draws'], 4)
    self.assertEqual(intervals['prompt_clusters']['num_prompt_clusters'], 2)
    self.assertEqual(
      intervals['prompt_clusters']['draws_per_prompt'], {'a': 2, 'b': 2})
    self.assertAlmostEqual(
      intervals['prompt_clusters']['point_estimate'], 1.0)


class GenerationAdapterComparisonTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.dynamic = _write_shards(
      self.root / 'dynamic', control='dynamic_dynamic',
      modes=('factorized', 'structured_marginal', 'structured_joint'),
      with_reference_lm=True)
    self.static = _write_shards(
      self.root / 'static', control='static_static',
      modes=('structured_joint',), with_reference_lm=True)
    self.protocol_patch = mock.patch.multiple(
      generation_adapter_comparison,
      REFERENCE_LM={
        'model_name_or_path': 'fixture-reference-lm',
        'revision': '6' * 40,
        'sequence_policy': REFERENCE_LM_SEQUENCE_POLICY,
      },
      TOKENIZER_REVISION='5' * 40,
      DATASET_PROTOCOLS={
        'wiki-pinned': {
          'data_config': 'eval_wikitext103_pinned',
          'dataset_revision': '4' * 40,
          'num_prompts': 2,
          'global_num_samples': 8,
          'base_seed': 700,
        },
      },
      NFE_BUDGETS=[2, 4],
      DYNAMIC_MODES=[
        'factorized', 'structured_marginal', 'structured_joint'],
      STATIC_MODES=['structured_joint'],
      NUM_SHARDS=2,
      BATCH_SIZE=4,
      SEQUENCE_LENGTH=3,
      SPAN_LENGTH=1,
      PROMPT_SELECTION_SEED=11)
    self.protocol_patch.start()
    self.prompt_validation = mock.patch(
      'evaluation.generation_adapter_comparison.validate_prompt_bundle',
      return_value=_prompt_bundle_identity())
    self.prompt_validation.start()
    self.generation_protocol_validation = mock.patch(
      'evaluation.generation_adapter_comparison.validate_generation_protocol',
      side_effect=lambda _config, _manifest, *, candidate_top_k,
      expected_control: {
        'schema_version': 1,
        'protocol_id': 'contextual-forest-generation-paper-v1',
        'protocol_path': '/fixture/generation-protocol.yaml',
        'protocol_sha256': '2' * 64,
        'dataset_config': 'eval_wikitext103_pinned',
        'logical_validation_dataset': 'wiki-pinned',
        'candidate_top_k': candidate_top_k,
        'control_identity': expected_control,
        'normalized_resolved_config_sha256': '3' * 64,
        'prompt_runtime_specification_sha256': '7' * 64,
      })
    self.generation_protocol_validation.start()
    self.adapter_origin_patch = mock.patch(
      'evaluation.generation_adapter_comparison.'
      'bind_generation_arm_to_adapter_origin_evidence',
      side_effect=lambda _evidence_path, *, expected_evidence_sha256,
      arm, adapter_path, expected_adapter_sha256, adapter_manifest_path,
      expected_adapter_manifest_sha256, structured_decoder_identity:
      _adapter_origin_binding(
        control=arm,
        structured_identity=structured_decoder_identity,
        adapter_path=str(adapter_path),
        adapter_sha=expected_adapter_sha256,
        adapter_manifest_path=str(adapter_manifest_path),
        adapter_manifest_sha=expected_adapter_manifest_sha256))
    self.adapter_origin_validation = self.adapter_origin_patch.start()

  def tearDown(self):
    self.adapter_origin_patch.stop()
    self.generation_protocol_validation.stop()
    self.prompt_validation.stop()
    self.protocol_patch.stop()
    self.temporary.cleanup()

  def _aggregated_unions(self, *, bootstrap_resamples: int):
    baseline = aggregate_generation_shards(
      self.static,
      baseline_mode='structured_joint',
      bootstrap_resamples=bootstrap_resamples,
      bootstrap_seed=117,
      timestamp_utc='2026-08-30T02:59:00+00:00')
    treatment = aggregate_generation_shards(
      self.dynamic,
      baseline_mode='factorized',
      bootstrap_resamples=bootstrap_resamples,
      bootstrap_seed=217,
      timestamp_utc='2026-08-30T02:59:00+00:00')
    return baseline, treatment

  def test_reloads_both_unions_and_computes_paired_adapter_intervals(self):
    baseline_union, treatment_union = self._aggregated_unions(
      bootstrap_resamples=40)
    result = generation_adapter_comparison.compare_generation_adapters(
      self.static,
      self.dynamic,
      baseline_union=baseline_union,
      treatment_union=treatment_union,
      bootstrap_resamples=40,
      bootstrap_seed=17,
      timestamp_utc='2026-08-30T03:00:00+00:00')
    self.assertEqual(result['dataset_id'], 'wiki-pinned')
    self.assertEqual(result['identity']['candidate_top_k'], 64)
    self.assertEqual(
      result['identity']['generation_protocol']['protocol_sha256'],
      '2' * 64)
    self.assertEqual(self.adapter_origin_validation.call_count, 2)
    self.assertEqual(len(result['comparisons']), 8)
    comparison = next(
      item for item in result['comparisons']
      if item['comparison_kind']
      == 'dynamic_adapter_vs_static_adapter_at_fixed_nfe'
      and item['baseline']['requested_nfe_budget'] == 2)
    self.assertEqual(
      comparison['baseline']['adapter_control'], 'static_static')
    self.assertEqual(
      comparison['treatment']['adapter_control'], 'dynamic_dynamic')
    # Both fixture controls use the same structured-joint samples, so the
    # paired quality difference is exactly zero.
    self.assertEqual(
      comparison['endpoints']['reference_token_accuracy'][
        'paired_draws']['point_estimate'],
      0.0)
    decisive = next(
      item for item in result['comparisons']
      if item['comparison_kind']
      == 'joint_vs_independent_structured_marginals_at_fixed_nfe'
      and item['baseline']['requested_nfe_budget'] == 2)
    self.assertEqual(decisive['baseline']['sampling_mode'],
                     'structured_marginal')
    self.assertEqual(decisive['treatment']['sampling_mode'],
                     'structured_joint')
    self.assertEqual(
      result['primary_causal_comparison'],
      'joint_vs_independent_structured_marginals_at_fixed_nfe')
    self.assertEqual(
      result['timing'][0]['inferential_status'], 'descriptive_only')
    self.assertEqual(
      result['verified_unions']['baseline_static_static']['canonical_sha256'],
      canonical_sha256(baseline_union))
    self.assertEqual(
      result['verified_unions']['treatment_dynamic_dynamic'][
        'canonical_sha256'],
      canonical_sha256(treatment_union))

  def test_rejects_cross_arm_prompt_or_unallowed_config_drift(self):
    for shard in self.static:
      manifest = _read_manifest(shard)
      manifest['prompts']['manifest_sha256'] = 'f' * 64
      manifest['prompts']['bundle_identity']['manifest_sha256'] = 'f' * 64
      _write_manifest(shard, manifest)
    baseline_union, treatment_union = self._aggregated_unions(
      bootstrap_resamples=5)
    with self.assertRaisesRegex(ValueError, 'cross_adapter_identity.prompts'):
      generation_adapter_comparison.compare_generation_adapters(
        self.static,
        self.dynamic,
        baseline_union=baseline_union,
        treatment_union=treatment_union,
        bootstrap_resamples=5)

    self.static = _write_shards(
      self.root / 'static-config-drift', control='static_static',
      modes=('structured_joint',), with_reference_lm=True)
    for shard in self.static:
      config_path = shard / 'resolved_config.yaml'
      config = yaml.safe_load(config_path.read_text())
      config['data']['name'] = 'forged'
      config_path.write_text(yaml.safe_dump(config, sort_keys=True))
      manifest = _read_manifest(shard)
      manifest['outputs']['resolved_config']['sha256'] = _sha256(config_path)
      _write_manifest(shard, manifest)
    baseline_union, treatment_union = self._aggregated_unions(
      bootstrap_resamples=5)
    with self.assertRaisesRegex(
        ValueError, 'resolved_config_after_allowed_fields'):
      generation_adapter_comparison.compare_generation_adapters(
        self.static,
        self.dynamic,
        baseline_union=baseline_union,
        treatment_union=treatment_union,
        bootstrap_resamples=5)


if __name__ == '__main__':
  unittest.main()
