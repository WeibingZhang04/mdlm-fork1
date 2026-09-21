import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import data_provenance

from evaluation.topology_diagnostics import (
  GPU_EXCLUSIVITY_ARTIFACT,
  GPU_EXCLUSIVITY_POLICY,
  GPU_MONITOR_INTERVAL_SECONDS,
  SCHEMA_VERSION,
  SUBMISSION_GPU_LOCK,
  _matched_node_permutation,
  _pair_jaccard,
  _source_descriptor,
  _time_shuffle_mapping,
  active_mask_sha256_for,
  aggregate_plan,
  build_analysis,
  canonical_sha256,
  corruption_context_sha256_for,
  clean_example_sha256_for,
  emit_topology_records,
  evaluate_topology_head_interventions,
  load_record_bundle,
  record_id_for,
  source_selection_sha256,
  source_units_from_ordered_dataset,
  topology_metrics,
  validate_analysis,
  validate_compiled_topology_plan_lineage,
  validate_gpu_exclusivity_evidence,
  write_record_bundle,
  write_source_selection_manifest,
)
from scripts.compile_experiment_matrix import (
  DEFAULT_MANIFEST,
  JOB_SCHEMA_VERSION,
  PLAN_SCHEMA_VERSION,
  _job,
  write_plan,
)
from scripts.compile_topology_diagnostics import (
  compile_topology_plan,
  derive_topology_plan,
)
from scripts.run_compiled_job import (
  SUCCESS_MARKER,
  _job_digest,
  _job_execution_digest,
  _load_plan,
  _output_records,
)
from scripts.run_topology_diagnostics import _run_with_gpu_exclusivity
from scripts.run_tensor_train_feasibility import _exclusive_gpu_lock


def _protocol():
  return {
    'schema_version': 2,
    'artifact': 'contextual_forest_topology_diagnostics_protocol',
    'candidate_top_k': 128,
    'protocol_id': 'topology-test-v1',
    'protocol_status': 'frozen_before_topology_results',
    'scientific_scope': 'fixture topology evidence only',
    'corruption_policy': {
      'forward_process': 'absorbing_mask_diffusion',
      'base_noise': (
        'sha256_uint53_per_position_v1_one_private_vector_per_'
        'source_unit_x_corruption_seed'),
      'time_coupling': (
        'reuse_identical_base_noise_across_time_grid_and_train_seeds'),
      'active_nodes': 'mask_token_and_attention_mask',
      'active_mask_nesting': 'active_sets_nondecreasing_with_time',
      'context_commitment': 'canonical_corruption_hash_bundle_v1',
      'mask_threshold': (
        'uint53_lt_floor_requested_probability_times_2pow53'),
    },
    'time_parameterization': 'absorbing_mask_probability',
    'topology_head_time_transform': (
      'negative_log1p_one_minus_probability_v1'),
    'determinism': 'torch_eval_no_grad_dropout_disabled',
    'evaluator_source_path': 'evaluation/topology_diagnostics.py',
    'intervention_locus': (
      'structured_decoder_topology_time_input_only_'
      'reuse_backbone_hidden_unary_v1'),
    'corruption_seeds': [11, 12],
    'time_points': [0.25, 0.75],
    'interventions': {
      'learned': {'effective_time': 'requested'},
      'matched_permuted': {
        'algorithm': 'sha256_sort_active_nodes_v1',
        'effective_time': 'requested',
        'permutation_seed': 7,
        'minimum_pooled_edge_set_changed_fraction': 0.75,
      },
      'fixed_time': {'effective_time': 0.5},
      'zero_time': {'effective_time': 0.0},
      'timestep_shuffled': {
        'algorithm': 'sha256_sort_rotate_time_grid_v1',
        'effective_time': 'deterministic_permutation_of_time_grid',
        'shuffle_seed': 13,
      },
    },
    'natural_order_chain': 'consecutive_active_positions',
    'nonlocal_edge_threshold': 1,
    'component_depth_root': 'minimum_active_position',
    'component_size_cap': 6,
    'completeness': (
      'exact_source_unit_x_corruption_seed_x_time_point_x_intervention_grid'),
    'require_nonempty_learned_forest': True,
    'source_selection': {
      'arm': 'dynamic_dynamic',
      'bundling': 'one_bundle_per_dataset_x_train_seed',
      'datasets': {'wiki-pinned': {
        'num_source_units': 1,
        'data_config_path': 'configs/data/eval_wikitext103_pinned.yaml',
        'dataset_revision': '9' * 40,
        'tokenizer_revision': '8' * 40,
      }},
      'train_seeds': [1],
      'require_identical_source_units_across_train_seeds': True,
      'source_unit_order': 'first_n_pinned_document_local_eval_order',
    },
  }


def _source_binding(*, train_seed=1, checkpoint_sha256=None):
  return {
    'schema_version': 2,
    'artifact': 'contextual_forest_topology_source_binding',
    'job_id': f'eval--dynamic-dynamic--wiki--s{train_seed:03d}',
    'compiled_plan_sha256': '1' * 64,
    'plan_id': '2' * 64,
    'job_spec_sha256': 'c' * 64,
    'job_execution_sha256': 'd' * 64,
    'repository_sha': '3' * 40,
    'repository_clean': True,
    'adapter_sha256': checkpoint_sha256 or str(train_seed) * 64,
    'adapter_export_manifest_sha256': '4' * 64,
    'data_config_sha256': 'e' * 64,
    'dataset_provenance_sha256': '5' * 64,
    'evaluator_source_sha256': '6' * 64,
    'arm': 'dynamic_dynamic',
    'dataset': 'wiki-pinned',
    'train_seed': train_seed,
    'source_selection_sha256': canonical_sha256([{
      'dataset': 'wiki-pinned',
      'dataset_revision': '9' * 40,
      'selection_index': 0,
      'source_unit_id': 'wiki-pinned:12:chunk-0',
      'document_id': 'wiki-pinned:12',
      'document_sha256': 'a' * 64,
      'chunk_index': 0,
      'clean_example_sha256': 'b' * 64,
      'sequence_length': 6,
    }]),
  }


def _metadata(
    *,
    reference_record_id=None,
    intervention_seed=None,
    time_donor_record_id=None,
    time_donor_index=None,
    node_permutation=None,
):
  return {
    'reference_record_id': reference_record_id,
    'intervention_seed': intervention_seed,
    'time_donor_record_id': time_donor_record_id,
    'time_donor_index': time_donor_index,
    'node_permutation': node_permutation,
  }


def _base_record(
    *,
    protocol,
    source_binding,
    corruption_seed,
    time_index,
    intervention,
    effective_time,
    metadata,
    edges,
):
  protocol_sha = canonical_sha256(protocol)
  binding_sha = canonical_sha256(source_binding)
  row = {
    'schema_version': 2,
    'artifact': 'contextual_forest_topology_diagnostic_record',
    'record_id': '0' * 64,
    'protocol_id': protocol['protocol_id'],
    'protocol_sha256': protocol_sha,
    'source_binding_sha256': binding_sha,
    'job_id': source_binding['job_id'],
    'dataset': source_binding['dataset'],
    'dataset_revision': protocol['source_selection']['datasets'][
      source_binding['dataset']]['dataset_revision'],
    'train_seed': source_binding['train_seed'],
    'source_unit_id': 'wiki-pinned:12:chunk-0',
    'document_id': 'wiki-pinned:12',
    'document_sha256': 'a' * 64,
    'selection_index': 0,
    'chunk_index': 0,
    'clean_example_sha256': 'b' * 64,
    'sequence_length': 6,
    'corruption_seed': corruption_seed,
    'base_noise_sha256': canonical_sha256({
      'corruption_seed': corruption_seed,
      'source': 'wiki-pinned:12:chunk-0',
    }),
    'corrupted_tokens_sha256': canonical_sha256({
      'corruption_seed': corruption_seed, 'time_index': time_index,
      'tokens': [10, 20, 30]}),
    'attention_mask_sha256': canonical_sha256([1, 1, 1, 1, 1, 1]),
    'active_mask_sha256': active_mask_sha256_for(
      sequence_length=6, active_nodes=[0, 1, 2, 3, 4, 5]),
    'corruption_context_sha256': '0' * 64,
    'requested_time_index': time_index,
    'requested_time': protocol['time_points'][time_index],
    'effective_time': effective_time,
    'intervention': intervention,
    'intervention_metadata': metadata,
    'active_nodes': [0, 1, 2, 3, 4, 5],
    'selected_edges': [list(edge) for edge in sorted(edges)],
  }
  row['corruption_context_sha256'] = corruption_context_sha256_for(row)
  row['record_id'] = record_id_for(row)
  return row


def _records(protocol=None, source_binding=None):
  protocol = protocol or _protocol()
  source_binding = source_binding or _source_binding()
  learned = {}
  edge_choices = [
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    [(0, 2), (1, 2), (2, 4), (3, 4), (4, 5)],
    [(0, 1), (1, 3), (2, 3), (3, 5), (4, 5)],
    [(0, 1), (1, 2), (2, 4), (3, 5), (4, 5)],
  ]
  choice_index = 0
  for corruption_seed in protocol['corruption_seeds']:
    for time_index, requested_time in enumerate(protocol['time_points']):
      row = _base_record(
        protocol=protocol,
        source_binding=source_binding,
        corruption_seed=corruption_seed,
        time_index=time_index,
        intervention='learned',
        effective_time=requested_time,
        metadata=_metadata(),
        edges=edge_choices[choice_index])
      choice_index += 1
      learned[(corruption_seed, time_index)] = row

  records = list(learned.values())
  for corruption_seed in protocol['corruption_seeds']:
    sample_learned = learned[(corruption_seed, 0)]
    time_mapping = _time_shuffle_mapping(
      protocol_id=protocol['protocol_id'],
      source_group_key=(
        _source_descriptor(sample_learned), corruption_seed),
      num_times=len(protocol['time_points']),
      seed=protocol['interventions']['timestep_shuffled']['shuffle_seed'])
    for time_index, requested_time in enumerate(protocol['time_points']):
      reference = learned[(corruption_seed, time_index)]
      permutation = _matched_node_permutation(
        reference['active_nodes'],
        seed=protocol['interventions']['matched_permuted'][
          'permutation_seed'])
      mapping = dict(permutation)
      permuted_edges = sorted([
        tuple(sorted((mapping[left], mapping[right])))
        for left, right in reference['selected_edges']
      ])
      records.append(_base_record(
        protocol=protocol,
        source_binding=source_binding,
        corruption_seed=corruption_seed,
        time_index=time_index,
        intervention='matched_permuted',
        effective_time=requested_time,
        metadata=_metadata(
          reference_record_id=reference['record_id'],
          intervention_seed=protocol['interventions'][
            'matched_permuted']['permutation_seed'],
          node_permutation=permutation),
        edges=permuted_edges))
      for intervention, effective_time in (
          ('fixed_time', protocol['interventions']['fixed_time'][
            'effective_time']),
          ('zero_time', protocol['interventions']['zero_time'][
            'effective_time'])):
        records.append(_base_record(
          protocol=protocol,
          source_binding=source_binding,
          corruption_seed=corruption_seed,
          time_index=time_index,
          intervention=intervention,
          effective_time=effective_time,
          metadata=_metadata(reference_record_id=reference['record_id']),
          edges=reference['selected_edges']))
      donor_index = time_mapping[time_index]
      donor = learned[(corruption_seed, donor_index)]
      records.append(_base_record(
        protocol=protocol,
        source_binding=source_binding,
        corruption_seed=corruption_seed,
        time_index=time_index,
        intervention='timestep_shuffled',
        effective_time=donor['requested_time'],
        metadata=_metadata(
          reference_record_id=reference['record_id'],
          intervention_seed=protocol['interventions'][
            'timestep_shuffled']['shuffle_seed'],
          time_donor_record_id=donor['record_id'],
          time_donor_index=donor_index),
        # The donor selects only the effective topology-head time. The model
        # output remains a forward pass of this requested corruption context;
        # it is never copied from the donor record.
        edges=reference['selected_edges']))
  return records


def _source_integrity(protocol):
  body = {
    'schema_version': 2,
    'artifact': 'contextual_forest_topology_source_integrity',
    'protocol_id': protocol['protocol_id'],
    'protocol_sha256': canonical_sha256(protocol),
    'compiled_plan_sha256': '1' * 64,
    'plan_id': '2' * 64,
    'source_manifest_sha256': '3' * 64,
    'repository_sha': '4' * 40,
    'repository_clean': True,
    'validated_job_ids': ['fixture'],
    'jobs': {},
    'dependencies': {},
  }
  return {**body, 'commitment_sha256': canonical_sha256(body)}


def _aggregate(protocol_path, manifest_paths):
  protocol = json.loads(protocol_path.read_text())
  protocol_sha = canonical_sha256(protocol)
  bundles = [
    load_record_bundle(
      path, protocol=protocol, protocol_sha256=protocol_sha)
    for path in manifest_paths]
  return build_analysis(
    protocol=protocol, protocol_sha256=protocol_sha, bundles=bundles,
    source_integrity=_source_integrity(protocol))


def _sha_file(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_success_marker(job, run_dir):
  marker = {
    'schema_version': 2,
    'artifact': 'compiled_experiment_job_success',
    'job_id': job['job_id'],
    'originating_plan_id': job['plan_id'],
    'source_repository_sha': job['source_repository_sha'],
    'job_execution_sha256': _job_execution_digest(job),
    'run_dir': str(run_dir.resolve()),
    'argv': job['argv'],
    'start_time_utc': '2026-08-31T00:00:00+00:00',
    'end_time_utc': '2026-08-31T00:01:00+00:00',
    'outputs': _output_records(run_dir, job['required_outputs']),
  }
  marker_path = Path(job['artifact_dir']) / SUCCESS_MARKER
  marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + '\n')
  return marker_path


def _trusted_plan_fixture(root, protocol):
  protocol_path = root / 'protocol.json'
  protocol_path.write_text(json.dumps(protocol, sort_keys=True))
  plan_dir = root / 'plan'
  jobs_dir = plan_dir / 'jobs'
  jobs_dir.mkdir(parents=True)
  export_artifact = root / 'runs' / 'export'
  eval_artifact = root / 'runs' / 'eval'
  export_run = export_artifact / 'attempts' / 'attempt-0001'
  eval_run = eval_artifact / 'attempts' / 'attempt-0001'
  export_run.mkdir(parents=True)
  eval_run.mkdir(parents=True)
  plan_id = '2' * 64
  repository_sha = '3' * 40
  source_manifest_sha = '4' * 64

  export_job = {
    'schema_version': JOB_SCHEMA_VERSION,
    'protocol_id': 'compiled-topology-fixture',
    'source_manifest_sha256': source_manifest_sha,
    'plan_id': plan_id,
    'source_repository_sha': repository_sha,
    'job_id': 'export--dynamic-dynamic--s001',
    'kind': 'export',
    'artifact_dir': str(export_artifact.resolve()),
    'suites': ['topology'],
    'dependencies': [],
    'identity': {
      'control': 'dynamic_dynamic', 'train_seed': 1,
      'candidate_k': 128},
    'argv': ['python', 'export.py'],
    'execution_mode': 'fresh_attempt',
    'external_inputs': [],
    'required_outputs': [
      {'name': 'adapter', 'pattern': 'adapter.safetensors',
       'exactly_one': True},
      {'name': 'adapter_manifest', 'pattern': 'adapter-manifest.json',
       'exactly_one': True},
    ],
  }
  eval_job = {
    'schema_version': JOB_SCHEMA_VERSION,
    'protocol_id': 'compiled-topology-fixture',
    'source_manifest_sha256': source_manifest_sha,
    'plan_id': plan_id,
    'source_repository_sha': repository_sha,
    'job_id': 'eval--dynamic-dynamic--s001--wiki-pinned--topology',
    'kind': 'eval',
    'artifact_dir': str(eval_artifact.resolve()),
    'suites': ['topology'],
    'dependencies': [export_job['job_id']],
    'identity': {
      'control': 'dynamic_dynamic', 'dataset': 'wiki-pinned',
      'train_seed': 1, 'candidate_k': 128},
    'argv': ['python', 'evaluate.py'],
    'execution_mode': 'fresh_attempt',
    'external_inputs': [],
    'required_outputs': [
      {'name': 'topology_records', 'pattern': 'topology_records.jsonl',
       'exactly_one': True},
      {'name': 'topology_record_manifest',
       'pattern': 'topology_records.manifest.json', 'exactly_one': True},
      {'name': 'topology_source_selection',
       'pattern': 'topology_source_selection.json', 'exactly_one': True},
      {'name': 'dataset_provenance', 'pattern': 'dataset_provenance.json',
       'exactly_one': True},
      {'name': 'gpu_exclusivity', 'pattern': 'gpu_exclusivity.json',
       'exactly_one': True},
    ],
  }
  jobs = {
    export_job['job_id']: export_job,
    eval_job['job_id']: eval_job,
  }
  promotion_path = root / 'candidate-k128-confirmation.json'
  promotion_path.write_text('{}\n')
  promotion_entry = {
    'path': str(promotion_path.resolve()),
    'sha256': _sha_file(promotion_path),
    'source_suite': 'candidate_k_128_pilot',
    'route_name': 'confirmation',
    'canonical_decision_sha256': '7' * 64,
    'source_compiled_plan_sha256': '8' * 64,
  }
  source_plan_path = root / 'source-compiled-plan.json'
  source_plan_payload = {
    'plan_id': '9' * 64,
    'source_manifest_sha256': source_manifest_sha,
    'repository': {'sha': repository_sha, 'dirty': False},
    'promotion_evidence': {
      'candidate_k_128_confirmation': promotion_entry},
  }
  source_plan_path.write_text(
    json.dumps(source_plan_payload, indent=2, sort_keys=True) + '\n')
  plan = {
    'schema_version': PLAN_SCHEMA_VERSION,
    'protocol_id': 'compiled-topology-fixture',
    'source_manifest_sha256': source_manifest_sha,
    'plan_id': plan_id,
    'repository': {'sha': repository_sha, 'dirty': False},
    'selected_suites': ['topology_diagnostics'],
    'compiled_plan_dir': str(plan_dir.resolve()),
    'topology_protocol': {
      'path': str(protocol_path.resolve()),
      'protocol_id': protocol['protocol_id'],
      'canonical_sha256': canonical_sha256(protocol),
      'file_sha256': _sha_file(protocol_path),
      'protocol_status': protocol['protocol_status'],
    },
    'source_compiled_plan': {
      'path': str(source_plan_path.resolve()),
      'sha256': _sha_file(source_plan_path),
      'plan_id': source_plan_payload['plan_id'],
    },
    'promotion_evidence': {
      'candidate_k_128_confirmation': promotion_entry},
    'job_ids': sorted(jobs),
    'job_spec_sha256': {
      job_id: _job_digest(job) for job_id, job in jobs.items()},
  }
  for job_id, job in jobs.items():
    (jobs_dir / f'{job_id}.json').write_text(
      json.dumps(job, indent=2, sort_keys=True) + '\n')
  plan_path = plan_dir / 'compiled-plan.json'
  plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')

  adapter_path = export_run / 'adapter.safetensors'
  adapter_path.write_bytes(b'fixture-adapter')
  adapter_manifest_path = export_run / 'adapter-manifest.json'
  adapter_manifest_path.write_text(json.dumps({
    'structured_decoder_identity': {
      'control_identity': 'dynamic_dynamic',
      'topology_mode': 'dynamic', 'factor_mode': 'dynamic',
      'candidate_top_k': 128, 'independent_mode': False,
      'topology_weight': 0.1,
      'head_semantics': {'component_size_cap': 6},
      'training_semantics': {},
    }}, sort_keys=True))
  _write_success_marker(export_job, export_run)

  dataset_specification = protocol['source_selection']['datasets'][
    'wiki-pinned']
  descriptor = {
    'dataset': 'wiki-pinned',
    'dataset_revision': dataset_specification['dataset_revision'],
    'selection_index': 0,
    'source_unit_id': 'wiki-pinned:12:chunk-0',
    'document_id': 'wiki-pinned:12', 'document_sha256': 'a' * 64,
    'chunk_index': 0, 'clean_example_sha256': 'b' * 64,
    'sequence_length': 6,
  }
  selection_path = write_source_selection_manifest(
    path=eval_run / 'topology_source_selection.json',
    protocol=protocol, dataset='wiki-pinned', entries=[descriptor])
  provenance = data_provenance.build_manifest(
    specification={
      'source_revision': dataset_specification['dataset_revision'],
      'tokenizer_revision': dataset_specification['tokenizer_revision'],
      'document_boundary_mode': 'wikitext_articles'},
    observed={'processed_num_sequences': 1})
  provenance_path = eval_run / 'dataset_provenance.json'
  provenance_path.write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + '\n')
  data_config_path = (
    Path(__file__).resolve().parents[1]
    / protocol['source_selection']['datasets']['wiki-pinned'][
      'data_config_path'])
  evaluator_source_path = (
    Path(__file__).resolve().parents[1] / protocol['evaluator_source_path'])
  binding = _source_binding()
  binding.update({
    'job_id': eval_job['job_id'],
    'compiled_plan_sha256': _sha_file(plan_path),
    'plan_id': plan_id,
    'job_spec_sha256': plan['job_spec_sha256'][eval_job['job_id']],
    'job_execution_sha256': _job_execution_digest(eval_job),
    'repository_sha': repository_sha,
    'adapter_sha256': _sha_file(adapter_path),
    'adapter_export_manifest_sha256': _sha_file(adapter_manifest_path),
    'data_config_sha256': _sha_file(data_config_path),
    'dataset_provenance_sha256': _sha_file(provenance_path),
    'evaluator_source_sha256': _sha_file(evaluator_source_path),
    'source_selection_sha256': canonical_sha256([descriptor]),
  })
  manifest_path = write_record_bundle(
    output_dir=eval_run, protocol=protocol, source_binding=binding,
    records=_records(protocol, binding))
  gpu_exclusivity_path = eval_run / 'gpu_exclusivity.json'
  gpu_exclusivity_path.write_text(json.dumps({
    'schema_version': SCHEMA_VERSION,
    'artifact': GPU_EXCLUSIVITY_ARTIFACT,
    'job_id': eval_job['job_id'],
    'required': True,
    'policy': GPU_EXCLUSIVITY_POLICY,
    'lock_path': str(SUBMISSION_GPU_LOCK),
    'lock_acquired': True,
    'monitor_interval_seconds': GPU_MONITOR_INTERVAL_SECONDS,
    'monitor_samples': 3,
    'preflight_other_compute_pids': [],
    'postflight_other_compute_pids': [],
    'foreign_pid_observations': [],
    'monitor_errors': [],
  }, indent=2, sort_keys=True) + '\n')
  eval_marker_path = _write_success_marker(eval_job, eval_run)
  return {
    'protocol_path': protocol_path,
    'plan_dir': plan_dir,
    'plan_path': plan_path,
    'jobs': jobs,
    'eval_marker_path': eval_marker_path,
    'manifest_path': manifest_path,
    'adapter_path': adapter_path,
    'selection_path': selection_path,
    'source_plan_path': source_plan_path,
    'promotion_path': promotion_path,
    'gpu_exclusivity_path': gpu_exclusivity_path,
  }


def _k128_source_plan_fixture(root):
  artifact_root = root / 'artifacts'
  plan_dir = artifact_root / 'plans' / 'candidate-k128-confirmation'
  promotion_path = root / 'candidate-k128-confirmation.json'
  promotion_path.write_text('{}\n')
  plan_id = '7' * 64
  repository_sha = '8' * 40
  source_manifest_sha = _sha_file(DEFAULT_MANIFEST)
  jobs = {}
  for train_seed in (1, 2, 3):
    train_id = f'train--dynamic_dynamic--s{train_seed:03d}--k128'
    export_id = f'export--dynamic_dynamic--s{train_seed:03d}--k128'
    train = _job(
      protocol_id='contextual-forest-expansion-v1',
      source_manifest_sha256=source_manifest_sha,
      source_repository_sha=repository_sha,
      plan_id=plan_id,
      job_id=train_id,
      kind='train',
      artifact_dir=artifact_root / 'runs' / train_id,
      suites=['candidate_k_128_confirmation'],
      dependencies=[],
      identity={
        'control': 'dynamic_dynamic', 'train_seed': train_seed,
        'candidate_k': 128},
      argv=['{python}', 'main.py'],
      execution_mode='fresh_attempt',
      external_inputs=[{
        'role': 'released_backbone_wrapper',
        'path': str(root / 'backbone.pt'),
        'sha256': '9' * 64,
      }],
      required_outputs=[{
        'name': 'checkpoint', 'pattern': 'checkpoints/last.ckpt',
        'exactly_one': True,
      }])
    export = _job(
      protocol_id='contextual-forest-expansion-v1',
      source_manifest_sha256=source_manifest_sha,
      source_repository_sha=repository_sha,
      plan_id=plan_id,
      job_id=export_id,
      kind='export',
      artifact_dir=artifact_root / 'runs' / export_id,
      suites=['candidate_k_128_confirmation'],
      dependencies=[train_id],
      identity={
        'control': 'dynamic_dynamic', 'train_seed': train_seed,
        'candidate_k': 128, 'topology_mode': 'dynamic',
        'factor_mode': 'dynamic', 'independent_mode': False,
        'topology_weight': 0.1},
      argv=['{python}', 'scripts/export_structured_adapter.py'],
      execution_mode='fresh_attempt',
      external_inputs=[],
      required_outputs=[
        {'name': 'adapter', 'pattern': 'adapter.safetensors',
         'exactly_one': True},
        {'name': 'adapter_manifest', 'pattern': 'adapter-manifest.json',
         'exactly_one': True},
      ])
    jobs[train_id] = train
    jobs[export_id] = export
  jobs = dict(sorted(jobs.items()))
  plan = {
    'schema_version': PLAN_SCHEMA_VERSION,
    'protocol_id': 'contextual-forest-expansion-v1',
    'source_manifest_sha256': source_manifest_sha,
    'repository': {'sha': repository_sha, 'dirty': False},
    'artifact_root': str(artifact_root.resolve()),
    'selected_suites': ['candidate_k_128_confirmation'],
    'promotion_evidence': {
      'candidate_k_128_confirmation': {
        'path': str(promotion_path.resolve()),
        'sha256': _sha_file(promotion_path),
        'source_suite': 'candidate_k_128_pilot',
        'route_name': 'confirmation',
        'canonical_decision_sha256': 'a' * 64,
        'source_compiled_plan_sha256': 'b' * 64,
      }},
    'plan_id': plan_id,
    'manifest_protocol_status': 'frozen_before_primary_results',
    'scientific_scope': 'fixture K=128 confirmation',
    'job_counts': {'export': 3, 'train': 3},
    'num_jobs': len(jobs),
    'job_ids': list(jobs),
    'job_spec_sha256': {
      job_id: _job_digest(job) for job_id, job in jobs.items()},
  }
  write_plan(plan_dir, plan, jobs, resume=False)
  return plan_dir, plan, jobs, artifact_root


class TopologyDiagnosticsTest(unittest.TestCase):

  def _write_protocol(self, directory, protocol):
    path = Path(directory) / 'protocol.json'
    path.write_text(json.dumps(protocol, sort_keys=True), encoding='utf-8')
    return path

  def test_bundle_aggregation_reports_requested_topology_evidence(self):
    protocol = _protocol()
    source_binding = _source_binding()
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol_path = self._write_protocol(root, protocol)
      manifest_path = write_record_bundle(
        output_dir=root / 'bundle',
        protocol=protocol,
        source_binding=source_binding,
        records=_records(protocol, source_binding))
      analysis = _aggregate(protocol_path, [manifest_path])

      self.assertEqual(analysis['grid_validation']['num_source_units'], 1)
      self.assertEqual(analysis['grid_validation']['num_records'], 20)
      self.assertGreaterEqual(
        analysis['grid_validation'][
          'matched_permuted_pooled_edge_set_changed_fraction'], 0.75)
      learned = analysis['learned_topology_summary']
      self.assertGreater(learned['pooled_nonlocal_edge_fraction'], 0.0)
      self.assertGreater(
        learned['pooled_natural_chain_precision'], 0.0)
      self.assertEqual(
        analysis['learned_edge_stability']['across_corruption_seeds'][
          'overall']['all_edge_jaccard']['count'], 2)
      self.assertEqual(
        analysis['learned_edge_stability']['across_requested_times'][
          'overall']['all_edge_jaccard']['count'], 2)
      self.assertEqual(
        analysis['intervention_comparisons']['matched_permuted'][
          'num_pairs'], 4)
      self.assertEqual(
        validate_analysis(analysis)['analysis_sha256'],
        analysis['analysis_sha256'])

  def test_replay_is_exact_and_detects_raw_file_tampering(self):
    protocol = _protocol()
    source_binding = _source_binding()
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol_path = self._write_protocol(root, protocol)
      manifest_path = write_record_bundle(
        output_dir=root / 'bundle',
        protocol=protocol,
        source_binding=source_binding,
        records=_records(protocol, source_binding))
      analysis = _aggregate(protocol_path, [manifest_path])
      analysis_path = root / 'analysis.json'
      analysis_path.write_text(
        json.dumps(analysis, sort_keys=True), encoding='utf-8')
      replayed = validate_analysis(json.loads(analysis_path.read_text()))
      self.assertEqual(replayed, analysis)

      record_path = manifest_path.parent / 'topology_records.jsonl'
      record_path.write_text(
        record_path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        _aggregate(protocol_path, [manifest_path])

  def test_rejects_incomplete_grid_before_writing(self):
    protocol = _protocol()
    source_binding = _source_binding()
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'incomplete topology grid'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle',
          protocol=protocol,
          source_binding=source_binding,
          records=_records(protocol, source_binding)[:-1])

  def test_rejects_cycle_in_raw_selected_edges(self):
    protocol = _protocol()
    source_binding = _source_binding()
    records = _records(protocol, source_binding)
    records[0]['selected_edges'] = [
      [0, 1], [0, 2], [1, 2], [3, 4], [4, 5]]
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'contains a cycle'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle',
          protocol=protocol,
          source_binding=source_binding,
          records=records)

  def test_rejects_noncanonical_matched_permutation(self):
    protocol = _protocol()
    source_binding = _source_binding()
    records = _records(protocol, source_binding)
    candidate = next(
      row for row in records if row['intervention'] == 'matched_permuted')
    candidate['intervention_metadata']['node_permutation'] = [
      [node, node] for node in candidate['active_nodes']]
    candidate['selected_edges'] = copy.deepcopy(next(
      row for row in records
      if row['record_id']
      == candidate['intervention_metadata']['reference_record_id'])[
        'selected_edges'])
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'frozen algorithm'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle',
          protocol=protocol,
          source_binding=source_binding,
          records=records)

  def test_rejects_timestep_shuffle_donor_drift(self):
    protocol = _protocol()
    source_binding = _source_binding()
    records = _records(protocol, source_binding)
    candidate = next(
      row for row in records if row['intervention'] == 'timestep_shuffled')
    candidate['intervention_metadata']['time_donor_index'] = (
      candidate['requested_time_index'])
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'deterministic permutation'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle',
          protocol=protocol,
          source_binding=source_binding,
          records=records)

  def test_rejects_dirty_repository_source_binding(self):
    protocol = _protocol()
    source_binding = _source_binding()
    source_binding['repository_clean'] = False
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'clean repository'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle',
          protocol=protocol,
          source_binding=source_binding,
          records=[])

  def test_requires_complete_dataset_by_training_seed_bundle_grid(self):
    protocol = _protocol()
    protocol['source_selection']['train_seeds'] = [1, 2]
    seed1_binding = _source_binding(train_seed=1)
    seed2_binding = _source_binding(train_seed=2)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol_path = self._write_protocol(root, protocol)
      first = write_record_bundle(
        output_dir=root / 'seed1',
        protocol=protocol,
        source_binding=seed1_binding,
        records=_records(protocol, seed1_binding))
      second = write_record_bundle(
        output_dir=root / 'seed2',
        protocol=protocol,
        source_binding=seed2_binding,
        records=_records(protocol, seed2_binding))
      with self.assertRaisesRegex(ValueError, 'grid is incomplete'):
        _aggregate(protocol_path, [first])
      analysis = _aggregate(protocol_path, [first, second])
      self.assertEqual(analysis['grid_validation']['num_source_units'], 2)
      self.assertEqual(
        analysis['learned_edge_stability']['across_training_seeds'][
          'overall']['all_edge_jaccard']['count'], 4)

  def test_component_cap_is_a_raw_graph_invariant(self):
    protocol = _protocol()
    protocol['component_size_cap'] = 2
    binding = _source_binding()
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'component_size_cap=2'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle', protocol=protocol,
          source_binding=binding, records=_records(protocol, binding))

  def test_fixed_time_anchor_must_reproduce_learned_edges(self):
    protocol = _protocol()
    protocol['interventions']['fixed_time']['effective_time'] = 0.25
    binding = _source_binding()
    records = _records(protocol, binding)
    candidate = next(
      row for row in records
      if row['intervention'] == 'fixed_time'
      and row['requested_time'] == 0.25)
    candidate['selected_edges'] = [
      [0, 2], [1, 2], [2, 3], [3, 4], [4, 5]]
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'exactly reproduce'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle', protocol=protocol,
          source_binding=binding, records=records)

  def test_corruption_context_commitment_is_derived_not_opaque(self):
    protocol = _protocol()
    binding = _source_binding()
    records = _records(protocol, binding)
    records[0]['active_mask_sha256'] = 'f' * 64
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'active_mask_sha256'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle', protocol=protocol,
          source_binding=binding, records=records)

  def test_cross_seed_corruption_drift_is_rejected(self):
    protocol = _protocol()
    protocol['source_selection']['train_seeds'] = [1, 2]
    first_binding = _source_binding(train_seed=1)
    second_binding = _source_binding(train_seed=2)
    second_records = _records(protocol, second_binding)
    for record in second_records:
      if (record['corruption_seed'] == protocol['corruption_seeds'][0]
          and record['requested_time_index'] == 0):
        record['corrupted_tokens_sha256'] = 'f' * 64
        record['corruption_context_sha256'] = (
          corruption_context_sha256_for(record))
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol_path = self._write_protocol(root, protocol)
      first = write_record_bundle(
        output_dir=root / 'first', protocol=protocol,
        source_binding=first_binding,
        records=_records(protocol, first_binding))
      second = write_record_bundle(
        output_dir=root / 'second', protocol=protocol,
        source_binding=second_binding, records=second_records)
      with self.assertRaisesRegex(ValueError, 'corruption context differs'):
        _aggregate(protocol_path, [first, second])

  def test_distinct_corruption_seeds_require_distinct_base_noise(self):
    protocol = _protocol()
    binding = _source_binding()
    records = _records(protocol, binding)
    first_noise = next(
      record['base_noise_sha256'] for record in records
      if record['corruption_seed'] == protocol['corruption_seeds'][0])
    for record in records:
      if record['corruption_seed'] == protocol['corruption_seeds'][1]:
        record['base_noise_sha256'] = first_noise
        record['corruption_context_sha256'] = (
          corruption_context_sha256_for(record))
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol_path = self._write_protocol(root, protocol)
      manifest = write_record_bundle(
        output_dir=root / 'bundle', protocol=protocol,
        source_binding=binding, records=records)
      with self.assertRaisesRegex(ValueError, 'reused one base-noise'):
        _aggregate(protocol_path, [manifest])

  def test_source_selection_order_is_part_of_commitment(self):
    record = _records()[0]
    record['selection_index'] = 1
    with self.assertRaisesRegex(ValueError, 'contiguous from zero'):
      source_selection_sha256([record])

    protocol = _protocol()
    protocol['source_selection']['datasets']['wiki-pinned'][
      'num_source_units'] = 2
    first = _source_descriptor(_records()[0])
    second = dict(first)
    second.update({
      'selection_index': 1,
      'source_unit_id': 'wiki-pinned:13:chunk-0',
      'document_id': 'wiki-pinned:13',
      'document_sha256': 'c' * 64,
      'clean_example_sha256': 'd' * 64,
    })
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'first-N evaluation order'):
        write_source_selection_manifest(
          path=Path(directory) / 'selection.json', protocol=protocol,
          dataset='wiki-pinned', entries=[second, first])

  def test_time_shuffle_is_pre_run_cross_seed_derangement(self):
    protocol = _protocol()
    first = _records(protocol, _source_binding(train_seed=1))[0]
    second = _records(protocol, _source_binding(train_seed=2))[0]
    first_mapping = _time_shuffle_mapping(
      protocol_id=protocol['protocol_id'],
      source_group_key=(
        _source_descriptor(first), first['corruption_seed']),
      num_times=len(protocol['time_points']),
      seed=protocol['interventions']['timestep_shuffled']['shuffle_seed'])
    second_mapping = _time_shuffle_mapping(
      protocol_id=protocol['protocol_id'],
      source_group_key=(
        _source_descriptor(second), second['corruption_seed']),
      num_times=len(protocol['time_points']),
      seed=protocol['interventions']['timestep_shuffled']['shuffle_seed'])
    self.assertEqual(first_mapping, second_mapping)
    self.assertEqual(sorted(first_mapping.values()), [0, 1])
    self.assertTrue(all(index != donor for index, donor in first_mapping.items()))

  def test_exact_topology_and_jaccard_metrics(self):
    records = _records()
    first = next(
      row for row in records
      if row['intervention'] == 'learned'
      and row['corruption_seed'] == 11
      and row['requested_time_index'] == 0)
    second = next(
      row for row in records
      if row['intervention'] == 'learned'
      and row['corruption_seed'] == 11
      and row['requested_time_index'] == 1)
    first_metrics = topology_metrics(first, nonlocal_threshold=1)
    second_metrics = topology_metrics(second, nonlocal_threshold=1)
    self.assertEqual(first_metrics['component_sizes'], [6])
    self.assertEqual(first_metrics['minimum_position_rooted_depths'], [5])
    self.assertEqual(first_metrics['component_diameters'], [5])
    self.assertEqual(second_metrics['natural_chain_overlap_count'], 3)
    self.assertEqual(second_metrics['nonlocal_edge_count'], 2)
    self.assertEqual(second_metrics['edge_distance_histogram'], {'1': 3, '2': 2})
    comparison = _pair_jaccard(first, second)
    self.assertAlmostEqual(comparison['all_edge_jaccard'], 3 / 7)
    self.assertAlmostEqual(comparison['shared_node_edge_jaccard'], 3 / 7)
    self.assertEqual(comparison['shared_active_node_fraction'], 1.0)

  def test_emitter_reuses_backbone_and_changes_only_head_time(self):
    import types

    import torch

    class FakeHead:
      training = False

      def __init__(self):
        self.hidden_pointers = []
        self.times = []

      def __call__(
          self, *, hidden_states, unary_logits, timestep, active_mask,
          topology_mode):
        del unary_logits, active_mask
        self.hidden_pointers.append(hidden_states.data_ptr())
        self.times.append(timestep.detach().cpu().tolist())
        self.assert_dynamic = topology_mode
        edge_index = torch.zeros(2, 2, 2, dtype=torch.long)
        edge_mask = torch.zeros(2, 2, dtype=torch.bool)
        for index, value in enumerate(timestep.tolist()):
          edge_index[index, 0] = torch.tensor(
            [0, 1] if value < 0.5 else [1, 2])
          edge_mask[index, 0] = True
        return types.SimpleNamespace(
          edge_index=edge_index, edge_mask=edge_mask)

    protocol = _protocol()
    head = FakeHead()
    descriptors = [_source_descriptor(record) for record in (
      _records()[0], _records()[1])]
    emitted = evaluate_topology_head_interventions(
      head=head,
      hidden_states=torch.zeros(2, 3, 4),
      unary_logits=torch.zeros(2, 3, 5),
      requested_time=torch.tensor([0.25, 0.75]),
      active_mask=torch.ones(2, 3, dtype=torch.bool),
      requested_time_indices=[0, 1],
      source_descriptors=descriptors,
      corruption_seeds=[11, 11],
      protocol=protocol)
    self.assertEqual(len(head.times), 4)
    self.assertEqual(len(set(head.hidden_pointers)), 1)
    self.assertEqual(head.assert_dynamic, 'dynamic')
    self.assertEqual(emitted[0]['learned']['selected_edges'], [[0, 1]])
    self.assertEqual(emitted[1]['learned']['selected_edges'], [[1, 2]])
    self.assertNotIn(
      'time_donor_edges', emitted[0]['timestep_shuffled'])

  def test_model_emitter_builds_complete_frozen_corruption_grid(self):
    import types

    import torch

    class FakeHead:
      training = False
      top_k = 128
      component_size_cap = 6

      def __init__(self):
        self.calls = 0

      def __call__(
          self, *, hidden_states, unary_logits, timestep, active_mask,
          topology_mode):
        del hidden_states, unary_logits, timestep
        self.calls += 1
        self.assert_dynamic = topology_mode
        batch, length = active_mask.shape
        edge_index = torch.zeros(
          batch, max(length - 1, 1), 2, dtype=torch.long,
          device=active_mask.device)
        edge_mask = torch.zeros(
          batch, max(length - 1, 1), dtype=torch.bool,
          device=active_mask.device)
        for batch_index in range(batch):
          nodes = active_mask[batch_index].nonzero(
            as_tuple=False).flatten()
          if len(nodes) >= 2:
            edge_index[batch_index, 0] = nodes[:2]
            edge_mask[batch_index, 0] = True
        return types.SimpleNamespace(
          edge_index=edge_index, edge_mask=edge_mask)

    class FakeModel:
      training = False
      mask_index = 99

      def __init__(self):
        self.structured_head = FakeHead()
        self.backbone_calls = 0

      def _structured_backbone_output(
          self, *, tokens, conditioning, force_no_grad):
        self.backbone_calls += 1
        self.force_no_grad = force_no_grad
        self.conditioning = conditioning.detach().cpu()
        batch, length = tokens.shape
        return (
          torch.zeros(batch, length, 4, device=tokens.device),
          torch.zeros(batch, length, 100, device=tokens.device))

    protocol = _protocol()
    # Exercise the production grid values that are not exactly representable
    # after a float32 tensor round-trip.
    protocol['time_points'] = [0.1, 0.9]
    tokens = [1] * 64
    attention = [True] * 64
    source = {
      'dataset': 'wiki-pinned',
      'dataset_revision': protocol['source_selection']['datasets'][
        'wiki-pinned']['dataset_revision'],
      'selection_index': 0,
      'source_unit_id': 'wiki-pinned:12:chunk-0',
      'document_id': 'wiki-pinned:12',
      'document_sha256': 'a' * 64,
      'chunk_index': 0,
      'clean_example_sha256': clean_example_sha256_for(tokens, attention),
      'sequence_length': len(tokens),
      'input_ids': tokens,
      'attention_mask': attention,
    }
    descriptor = {
      field: source[field] for field in _source_descriptor({
        **source,
      })
    }
    binding = _source_binding()
    binding['source_selection_sha256'] = canonical_sha256([descriptor])
    model = FakeModel()
    records = emit_topology_records(
      model=model,
      source_units=[source],
      protocol=protocol,
      source_binding=binding,
      device=torch.device('cpu'),
      batch_size=1)
    self.assertEqual(len(records), 20)
    self.assertEqual(model.backbone_calls, 4)
    self.assertEqual(model.structured_head.calls, 16)
    self.assertTrue(model.force_no_grad)
    self.assertEqual(model.structured_head.assert_dynamic, 'dynamic')
    for record in records:
      if record['intervention'] in {'learned', 'matched_permuted'}:
        self.assertEqual(record['effective_time'], record['requested_time'])
    learned = [
      record for record in records if record['intervention'] == 'learned']
    for corruption_seed in protocol['corruption_seeds']:
      trajectory = sorted(
        (record for record in learned
         if record['corruption_seed'] == corruption_seed),
        key=lambda record: record['requested_time_index'])
      self.assertEqual(
        len({record['base_noise_sha256'] for record in trajectory}), 1)
      self.assertTrue(
        set(trajectory[0]['active_nodes']).issubset(
          trajectory[1]['active_nodes']))
    with tempfile.TemporaryDirectory() as directory:
      manifest = write_record_bundle(
        output_dir=Path(directory) / 'bundle',
        protocol=protocol,
        source_binding=binding,
        records=records)
      loaded, _ = load_record_bundle(
        manifest,
        protocol=protocol,
        protocol_sha256=canonical_sha256(protocol))
    self.assertEqual(len(loaded), 20)

  def test_ordered_dataset_selection_uses_literal_first_n_rows(self):
    protocol = _protocol()
    protocol['source_selection']['datasets']['wiki-pinned'][
      'num_source_units'] = 2
    rows = []
    for document_index in (7, 3, 99):
      rows.append({
        'input_ids': [document_index, 1, 2],
        'attention_mask': [1, 1, 1],
        'source_document_index': document_index,
        'source_document_sha256': f'{document_index:064x}',
        'source_chunk_index': 0,
      })
    selected = source_units_from_ordered_dataset(
      rows, protocol=protocol, dataset='wiki-pinned')
    self.assertEqual(
      [source['document_id'] for source in selected],
      ['wiki-pinned:7', 'wiki-pinned:3'])
    self.assertEqual(
      [source['selection_index'] for source in selected], [0, 1])

  def test_topology_gpu_guard_uses_shared_lock_and_rejects_foreign_cuda(self):
    self.assertEqual(
      SUBMISSION_GPU_LOCK,
      Path('/mnt/contextual-forest/experiments/.submission-gpu.lock'))
    with tempfile.TemporaryDirectory() as directory:
      lock_path = Path(directory) / '.submission-gpu.lock'
      operation = mock.Mock(return_value='finished')
      with mock.patch(
          'scripts.run_tensor_train_feasibility._other_compute_pids',
          return_value=[]):
        result, evidence = _run_with_gpu_exclusivity(
          operation, job_id='topology-fixture', lock_path=lock_path)
      self.assertEqual(result, 'finished')
      self.assertEqual(operation.call_count, 1)
      self.assertGreaterEqual(evidence['monitor_samples'], 3)
      self.assertEqual(evidence['foreign_pid_observations'], [])

      operation.reset_mock()
      with _exclusive_gpu_lock(lock_path):
        with self.assertRaisesRegex(RuntimeError, 'holds the GPU lock'):
          _run_with_gpu_exclusivity(
            operation, job_id='topology-fixture', lock_path=lock_path)
      operation.assert_not_called()

      responses = iter(([], [], [991]))
      with mock.patch(
          'scripts.run_tensor_train_feasibility._other_compute_pids',
          side_effect=lambda: next(responses, [991])):
        with self.assertRaisesRegex(RuntimeError, 'exclusivity'):
          _run_with_gpu_exclusivity(
            operation, job_id='topology-fixture', lock_path=lock_path)

  def test_gpu_exclusivity_evidence_rejects_foreign_observation(self):
    evidence = {
      'schema_version': SCHEMA_VERSION,
      'artifact': GPU_EXCLUSIVITY_ARTIFACT,
      'job_id': 'topology-fixture',
      'required': True,
      'policy': GPU_EXCLUSIVITY_POLICY,
      'lock_path': str(SUBMISSION_GPU_LOCK),
      'lock_acquired': True,
      'monitor_interval_seconds': GPU_MONITOR_INTERVAL_SECONDS,
      'monitor_samples': 3,
      'preflight_other_compute_pids': [],
      'postflight_other_compute_pids': [],
      'foreign_pid_observations': [],
      'monitor_errors': [],
    }
    self.assertEqual(
      validate_gpu_exclusivity_evidence(
        evidence, expected_job_id='topology-fixture'),
      evidence)
    evidence['foreign_pid_observations'] = [{
      'time_utc': '2026-08-31T00:00:00+00:00', 'pids': [991]}]
    with self.assertRaisesRegex(ValueError, 'foreign_pid_observations'):
      validate_gpu_exclusivity_evidence(
        evidence, expected_job_id='topology-fixture')

  def test_dedicated_compiler_emits_six_runnable_topology_jobs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_dir, _, source_jobs, artifact_root = \
        _k128_source_plan_fixture(root)
      output_dir = artifact_root / 'plans' / 'topology'
      with mock.patch(
          'scripts.compile_topology_diagnostics.'
          '_validate_repository_checkout'), \
          mock.patch(
            'scripts.compile_topology_diagnostics.'
            '_validate_canonical_source_plan'):
        plan, jobs, observed_output = compile_topology_plan(
          source_plan_dir=source_dir, output_dir=output_dir)
      self.assertEqual(observed_output, output_dir.resolve())
      topology_jobs = [
        job for job in jobs.values()
        if job['identity'].get('diagnostic') == 'topology']
      self.assertEqual(len(topology_jobs), 6)
      self.assertEqual(
        sum(job['identity']['num_records'] for job in topology_jobs), 14_400)
      self.assertEqual(
        {(job['identity']['dataset'], job['identity']['train_seed'])
         for job in topology_jobs},
        {(dataset, seed) for dataset in (
          'scientific_papers_arxiv', 'wikitext103') for seed in (1, 2, 3)})
      for job in topology_jobs:
        self.assertIn('scripts/run_topology_diagnostics.py', job['argv'])
        self.assertEqual(len(job['dependencies']), 1)
        self.assertEqual(
          {output['name'] for output in job['required_outputs']}, {
            'topology_records', 'topology_record_manifest',
            'topology_source_selection', 'dataset_provenance',
            'gpu_exclusivity'})
      loaded_plan, loaded_jobs = _load_plan(output_dir)
      self.assertEqual(loaded_plan['plan_id'], plan['plan_id'])
      self.assertEqual(set(loaded_jobs), set(jobs))
      for train_seed in (1, 2, 3):
        for prefix in ('train', 'export'):
          job_id = (
            f'{prefix}--dynamic_dynamic--s{train_seed:03d}--k128')
          self.assertEqual(
            _job_execution_digest(jobs[job_id]),
            _job_execution_digest(source_jobs[job_id]))

  def test_canonical_plan_reconstruction_rejects_argv_and_dependency_drift(self):
    protocol_path = (
      Path(__file__).resolve().parents[1] / 'configs' / 'evaluation'
      / 'contextual-forest-topology-diagnostics-v1.json')
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = canonical_sha256(protocol)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_dir, _, _, artifact_root = _k128_source_plan_fixture(root)
      output_dir = artifact_root / 'plans' / 'topology'
      with mock.patch(
          'scripts.compile_topology_diagnostics.'
          '_validate_repository_checkout'), \
          mock.patch(
            'scripts.compile_topology_diagnostics.'
            '_validate_canonical_source_plan'):
        plan, jobs, _ = compile_topology_plan(
          source_plan_dir=source_dir, output_dir=output_dir)
        validate_compiled_topology_plan_lineage(
          plan,
          jobs=jobs,
          plan_dir=output_dir,
          protocol_path=protocol_path,
          protocol=protocol,
          protocol_sha256=protocol_sha)

        topology_job_id = next(
          job_id for job_id, job in jobs.items()
          if job['identity'].get('diagnostic') == 'topology'
          and job['identity']['train_seed'] == 1)
        mutations = {}

        argv_jobs = copy.deepcopy(jobs)
        argv_jobs[topology_job_id]['argv'][1] = 'scripts/forged_emitter.py'
        mutations['argv'] = argv_jobs

        dependency_jobs = copy.deepcopy(jobs)
        dependency_jobs[topology_job_id]['dependencies'] = [
          'export--dynamic_dynamic--s002--k128']
        mutations['dependency'] = dependency_jobs

        numeric_jobs = copy.deepcopy(jobs)
        numeric_jobs[topology_job_id]['identity']['num_records'] = float(
          numeric_jobs[topology_job_id]['identity']['num_records'])
        mutations['numeric_type'] = numeric_jobs

        boolean_jobs = copy.deepcopy(jobs)
        boolean_jobs[topology_job_id]['required_outputs'][0][
          'exactly_one'] = 1
        mutations['boolean_type'] = boolean_jobs

        for label, tampered_jobs in mutations.items():
          with self.subTest(label=label):
            tampered_plan = copy.deepcopy(plan)
            tampered_plan['job_spec_sha256'][topology_job_id] = _job_digest(
              tampered_jobs[topology_job_id])
            with self.assertRaisesRegex(
                ValueError, 'canonical compiler reconstruction'):
              validate_compiled_topology_plan_lineage(
                tampered_plan,
                jobs=tampered_jobs,
                plan_dir=output_dir,
                protocol_path=protocol_path,
                protocol=protocol,
                protocol_sha256=protocol_sha)

        numeric_plan = copy.deepcopy(plan)
        numeric_plan['num_jobs'] = float(numeric_plan['num_jobs'])
        with self.assertRaisesRegex(
            ValueError, 'canonical compiler reconstruction'):
          validate_compiled_topology_plan_lineage(
            numeric_plan,
            jobs=jobs,
            plan_dir=output_dir,
            protocol_path=protocol_path,
            protocol=protocol,
            protocol_sha256=protocol_sha)

  def test_parent_and_derived_coordinated_argv_tampering_is_rejected(self):
    protocol_path = (
      Path(__file__).resolve().parents[1] / 'configs' / 'evaluation'
      / 'contextual-forest-topology-diagnostics-v1.json')
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = canonical_sha256(protocol)
    mutations = {
      'train': 'train--dynamic_dynamic--s001--k128',
      'export': 'export--dynamic_dynamic--s001--k128',
    }
    for label, tampered_job_id in mutations.items():
      with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_dir, source_plan, source_jobs, artifact_root = \
          _k128_source_plan_fixture(root)
        output_dir = artifact_root / 'plans' / 'topology'
        promotion_path = Path(
          source_plan['promotion_evidence'][
            'candidate_k_128_confirmation']['path'])

        tampered_source_plan = copy.deepcopy(source_plan)
        tampered_source_jobs = copy.deepcopy(source_jobs)
        tampered_source_jobs[tampered_job_id]['argv'].append(
          '--coordinated-forged-argument')
        tampered_source_plan['job_spec_sha256'][tampered_job_id] = \
          _job_digest(tampered_source_jobs[tampered_job_id])
        (source_dir / 'jobs' / f'{tampered_job_id}.json').write_text(
          json.dumps(
            tampered_source_jobs[tampered_job_id], indent=2,
            sort_keys=True) + '\n')
        (source_dir / 'compiled-plan.json').write_text(
          json.dumps(tampered_source_plan, indent=2, sort_keys=True) + '\n')

        reconstructed_parent = (
          copy.deepcopy(source_plan), copy.deepcopy(source_jobs),
          artifact_root.resolve())
        with mock.patch(
            'scripts.compile_topology_diagnostics.'
            '_validate_repository_checkout'), \
            mock.patch(
              'scripts.compile_topology_diagnostics.derive_matrix_plan',
              return_value=reconstructed_parent) as reconstruct_parent:
          # Model a coordinated attacker: re-run the derived compiler after
          # changing the parent, so the source hash, derived plan ID, copied
          # job, and every derived job commitment are internally consistent.
          with mock.patch(
              'scripts.compile_topology_diagnostics.'
              '_validate_canonical_source_plan'):
            tampered_plan, tampered_jobs, observed_output = \
              derive_topology_plan(
                source_plan_dir=source_dir,
                output_dir=output_dir,
                protocol_path=protocol_path)

          self.assertEqual(observed_output, output_dir.resolve())
          self.assertFalse(output_dir.exists())
          self.assertEqual(
            tampered_plan['source_compiled_plan']['sha256'],
            _sha_file(source_dir / 'compiled-plan.json'))
          self.assertEqual(
            tampered_jobs[tampered_job_id]['argv'],
            tampered_source_jobs[tampered_job_id]['argv'])
          with self.assertRaisesRegex(
              ValueError, 'K=128 topology parent job.*canonical compiler'):
            validate_compiled_topology_plan_lineage(
              tampered_plan,
              jobs=tampered_jobs,
              plan_dir=output_dir,
              protocol_path=protocol_path,
              protocol=protocol,
              protocol_sha256=protocol_sha)
          self.assertFalse(output_dir.exists())

        reconstruct_parent.assert_called_once()
        call = reconstruct_parent.call_args
        self.assertEqual(call.args, (DEFAULT_MANIFEST,))
        self.assertEqual(
          call.kwargs['selected_suites'],
          ['candidate_k_128_confirmation'])
        self.assertEqual(
          call.kwargs['artifact_root_override'], artifact_root.resolve())
        self.assertEqual(
          call.kwargs['promotion_evidence'], {
            'candidate_k_128_confirmation': promotion_path.resolve()})

  def test_trusted_plan_authenticates_outputs_and_post_run_marker(self):
    protocol = _protocol()
    protocol['source_selection']['datasets']['wiki-pinned'].update({
      'dataset_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
      'tokenizer_revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
    })
    with tempfile.TemporaryDirectory() as directory:
      fixture = _trusted_plan_fixture(Path(directory), protocol)
      with mock.patch(
          'scripts.run_compiled_job._validate_repository_checkout'), \
          mock.patch(
            'evaluation.topology_diagnostics._validate_trusted_protocol_path'), \
          mock.patch(
            'evaluation.topology_diagnostics._validate_adapter_export',
            return_value={}), \
          mock.patch(
            'scripts.compile_topology_diagnostics.derive_topology_plan',
            return_value=(
              json.loads(fixture['plan_path'].read_text()),
              fixture['jobs'], fixture['plan_dir'].resolve())):
        analysis = aggregate_plan(
          plan_dir=fixture['plan_dir'],
          protocol_path=fixture['protocol_path'])
      integrity = analysis['source_integrity']
      self.assertEqual(
        integrity['validated_job_ids'],
        ['eval--dynamic-dynamic--s001--wiki-pinned--topology'])
      self.assertEqual(
        integrity['jobs'][integrity['validated_job_ids'][0]][
          'success_marker_sha256'],
        _sha_file(fixture['eval_marker_path']))
      self.assertIn(
        'export--dynamic-dynamic--s001', integrity['dependencies'])

      fixture['adapter_path'].write_bytes(b'tampered-adapter')
      with mock.patch(
          'scripts.run_compiled_job._validate_repository_checkout'), \
          mock.patch(
            'evaluation.topology_diagnostics._validate_trusted_protocol_path'), \
          mock.patch(
            'evaluation.topology_diagnostics._validate_adapter_export',
            return_value={}), \
          mock.patch(
            'scripts.compile_topology_diagnostics.derive_topology_plan',
            return_value=(
              json.loads(fixture['plan_path'].read_text()),
              fixture['jobs'], fixture['plan_dir'].resolve())):
        with self.assertRaisesRegex(ValueError, 'outputs drifted'):
          aggregate_plan(
            plan_dir=fixture['plan_dir'],
            protocol_path=fixture['protocol_path'])

  def test_protocol_status_is_exactly_trusted(self):
    protocol = _protocol()
    protocol['protocol_status'] = 'changed_after_results'
    binding = _source_binding()
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'frozen_before_topology_results'):
        write_record_bundle(
          output_dir=Path(directory) / 'bundle', protocol=protocol,
          source_binding=binding, records=[])

  def test_trusted_plan_rehashes_parent_compiled_plan(self):
    protocol = _protocol()
    protocol['source_selection']['datasets']['wiki-pinned'].update({
      'dataset_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
      'tokenizer_revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
    })
    with tempfile.TemporaryDirectory() as directory:
      fixture = _trusted_plan_fixture(Path(directory), protocol)
      fixture['source_plan_path'].write_text('{}\n')
      with mock.patch(
          'scripts.run_compiled_job._validate_repository_checkout'), \
          mock.patch(
            'evaluation.topology_diagnostics._validate_trusted_protocol_path'):
        with self.assertRaisesRegex(ValueError, 'source compiled plan hash'):
          aggregate_plan(
            plan_dir=fixture['plan_dir'],
            protocol_path=fixture['protocol_path'])

  def test_trusted_plan_rehashes_k128_promotion_evidence(self):
    protocol = _protocol()
    protocol['source_selection']['datasets']['wiki-pinned'].update({
      'dataset_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
      'tokenizer_revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
    })
    with tempfile.TemporaryDirectory() as directory:
      fixture = _trusted_plan_fixture(Path(directory), protocol)
      fixture['promotion_path'].write_text('{"decision":"changed"}\n')
      with mock.patch(
          'scripts.run_compiled_job._validate_repository_checkout'), \
          mock.patch(
            'evaluation.topology_diagnostics._validate_trusted_protocol_path'):
        with self.assertRaisesRegex(ValueError, 'promotion evidence drifted'):
          aggregate_plan(
            plan_dir=fixture['plan_dir'],
            protocol_path=fixture['protocol_path'])

  def test_trusted_plan_hash_drift_is_rejected(self):
    protocol = _protocol()
    protocol['source_selection']['datasets']['wiki-pinned'].update({
      'dataset_revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
      'tokenizer_revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
    })
    with tempfile.TemporaryDirectory() as directory:
      fixture = _trusted_plan_fixture(Path(directory), protocol)
      plan = json.loads(fixture['plan_path'].read_text())
      plan['post_result_note'] = 'changes compiled plan bytes'
      fixture['plan_path'].write_text(
        json.dumps(plan, indent=2, sort_keys=True) + '\n')
      with mock.patch(
          'scripts.run_compiled_job._validate_repository_checkout'), \
          mock.patch(
            'evaluation.topology_diagnostics._validate_trusted_protocol_path'), \
          mock.patch(
            'evaluation.topology_diagnostics._validate_adapter_export',
            return_value={}), \
          mock.patch(
            'scripts.compile_topology_diagnostics.derive_topology_plan',
            return_value=(plan, fixture['jobs'], fixture['plan_dir'].resolve())):
        with self.assertRaisesRegex(ValueError, 'compiled job'):
          aggregate_plan(
            plan_dir=fixture['plan_dir'],
            protocol_path=fixture['protocol_path'])


if __name__ == '__main__':
  unittest.main()
