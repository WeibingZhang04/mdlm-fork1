import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from evaluation.tensor_train_baseline import (
  CACHE_POLICY,
  EXPECTED_CHECKPOINT_CONFIG_SHA256,
  EXPECTED_CHECKPOINT_STATE_KEYS,
  EXPECTED_CHECKPOINT_STEPS,
  GPU_EXCLUSIVITY_POLICY,
  OFFICIAL_CHECKPOINT_REVISION,
  OFFICIAL_SOURCE_REVISION,
  SUBMISSION_GPU_LOCK,
  canonical_sha256,
  compile_plan,
  load_compiled_plan,
  load_protocol,
  sha256_file,
  validate_completed_run,
  verify_complete_matrix,
  write_plan,
)
from scripts.compile_tensor_train_feasibility import DEFAULT_PROTOCOL
from scripts.run_tensor_train_feasibility import (
  _ForeignPidMonitor,
  _PositionScheduleRecorder,
  _exclusive_gpu_lock,
  _prepare_offline_cache,
  _reseed_sampling,
  run_job,
)


class TensorTrainBaselineTest(unittest.TestCase):

  def _identities(self, root: Path):
    source = {
      'path': str(root / 'tensor-train'),
      'revision': OFFICIAL_SOURCE_REVISION,
      'clean': True,
      'origin': 'https://github.com/ssamt/tensor-train.git',
    }
    checkpoints = {
      'marginal': {
        'path': str(root / 'checkpoints/owt/marginal.pt'),
        'sha256': (
          '84fc03cacd818df293602987d4367b8ead7c96539b9b536b748bc86a6cd7079c'),
        'size_bytes': 7_092_428,
        'huggingface_repository': 'ssamt/tensor-train',
        'huggingface_revision': OFFICIAL_CHECKPOINT_REVISION,
        'relative_path': 'owt/marginal.pt',
      },
      'tensor_train_rank4': {
        'path': str(root / 'checkpoints/owt/ttd_4_marg.pt'),
        'sha256': (
          '8ad8d956af127795686489e9f3496e7a634da18ecf79464df1319668a2a3a7a2'),
        'size_bytes': 127_728_181,
        'huggingface_repository': 'ssamt/tensor-train',
        'huggingface_revision': OFFICIAL_CHECKPOINT_REVISION,
        'relative_path': 'owt/ttd_4_marg.pt',
      },
    }
    harness = {
      'path': str(root / 'harness'),
      'revision': 'a' * 40,
      'clean': True,
      'origin': 'git@github.com:yuntian-group/mdlm.git',
    }
    snapshots = {}
    specifications = {
      'backbone': (
        'kuleshov-group/mdlm-owt',
        'd0958fa851335ece6c15260ce0025f030673c0fb'),
      'tokenizer': (
        'openai-community/gpt2',
        '607a30d783dfa663caf39e06633721c8d4cfcd7e'),
      'evaluator': (
        'gpt2-large', '32b71b12589c2f8d625668d2335a01cac3249519'),
    }
    for name, (repository, revision) in specifications.items():
      files = [{
        'path': 'config.json',
        'size_bytes': 1,
        'sha256': hashlib.sha256(name.encode()).hexdigest(),
      }]
      snapshots[name] = {
        'repository': repository,
        'revision': revision,
        'snapshot_path': str(root / f'cache/{name}/snapshots/{revision}'),
        'snapshot_revision': revision,
        'files': files,
        'files_manifest_sha256': canonical_sha256(files),
      }
    cache = {
      'root': str(root / 'cache'),
      'policy': CACHE_POLICY,
      'snapshots': snapshots,
    }
    cache['identity_sha256'] = canonical_sha256(cache)
    return source, checkpoints, harness, cache

  def _compile(self, root: Path):
    source, checkpoints, harness, cache = self._identities(root)
    return compile_plan(
      DEFAULT_PROTOCOL,
      source_root=root / 'tensor-train',
      checkpoint_root=root / 'checkpoints',
      artifact_root=root / 'artifacts',
      harness_repo_root=root / 'harness',
      cache_root=root / 'cache',
      source_identity=source,
      checkpoint_identity=checkpoints,
      harness_identity=harness,
      cache_identity=cache)

  @staticmethod
  def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')

  def _write_completed_run(
      self, plan, job, plan_path: Path, run_dir: Path | None = None) -> None:
    run_dir = Path(job['artifact_dir']) if run_dir is None else Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tokens_per_step = job['generation']['tokens_per_step']
    chunks = [
      list(range(step * tokens_per_step, (step + 1) * tokens_per_step))
      for step in range(job['generation']['nfe_steps'])]
    schedule_hash = canonical_sha256(chunks)
    schedule_records = [{
      'sample_id': index,
      'chunks': chunks,
      'position_schedule_sha256': schedule_hash,
    } for index in range(256)]
    schedule_path = run_dir / 'position-schedules.json'
    self._write_json(schedule_path, {
      'schema_version': 1,
      'artifact': 'tensor_train_owt_position_schedules',
      'job_id': job['job_id'],
      'arm': job['arm'],
      'nfe_steps': job['generation']['nfe_steps'],
      'generation_seed': 260703,
      'schedule_policy': job['generation']['schedule_policy'],
      'num_samples': 256,
      'sequence_length': 1024,
      'records': schedule_records,
    })
    score = {
      'model_name_or_path': 'gpt2-large',
      'revision': '32b71b12589c2f8d625668d2335a01cac3249519',
      'sequence_policy': (
        'retokenize_decoded_text_score_through_first_nonleading_eos_v1'),
      'token_count': 1,
      'mean_nll_nats': 1.0,
      'perplexity': math.e,
    }
    records = []
    for index in range(256):
      tokens = [index + 1] * 1024
      text = f'sample {index}'
      records.append({
        'schema_version': 1,
        'sample_id': index,
        'job_id': job['job_id'],
        'arm': job['arm'],
        'nfe_steps': job['generation']['nfe_steps'],
        'generation_seed': 260703,
        'token_ids': tokens,
        'token_ids_sha256': canonical_sha256(tokens),
        'text': text,
        'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'position_schedule_sha256': schedule_hash,
        'reference_lm': score,
      })
    samples_path = run_dir / 'samples.jsonl'
    samples_path.write_text(
      ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records))
    runtime_identity = {
      'schema_version': 1,
      'model_name_or_path': 'gpt2-large',
      'model_revision': score['revision'],
      'model_class': 'transformers.GPT2LMHeadModel',
      'model_config_class': 'transformers.GPT2Config',
      'tokenizer_name_or_path': 'gpt2-large',
      'tokenizer_revision': score['revision'],
      'tokenizer_class': 'transformers.GPT2TokenizerFast',
      'tokenizer_vocab_size': 50257,
      'tokenizer_bos_token_id': 50256,
      'tokenizer_eos_token_id': 50256,
      'tokenizer_pad_token_id': 50256,
      'tokenizer_padding_side': 'right',
      'tokenizer_truncation_side': 'right',
      'tokenization_policy': (
        'fast_tokenizer_right_padding_right_truncation_add_special_tokens_v1'),
      'sequence_policy': score['sequence_policy'],
      'add_special_tokens': True,
      'batch_size': 8,
      'max_length': 1024,
      'requested_dtype': 'float32',
      'parameter_dtypes': ['torch.float32'],
      'precision_policy': (
        'explicit_checkpoint_dtype_no_autocast_float32_cross_entropy_v1'),
      'device': 'cuda',
      'python': '3.12.11',
      'torch': '2.3.1',
      'cuda_runtime': '12.1',
      'transformers': '4.46.2',
      'tokenizers': '0.20.3',
    }
    score_summary = {
      'model_name_or_path': 'gpt2-large',
      'revision': score['revision'],
      'sequence_policy': score['sequence_policy'],
      'runtime_identity': runtime_identity,
      'num_scored_sequences': 256,
      'num_scored_tokens': 256,
      'mean_nll_nats': 1.0,
      'perplexity': math.e,
    }
    metrics_path = run_dir / 'metrics.json'
    self._write_json(metrics_path, {
      'schema_version': 1,
      'artifact': 'tensor_train_owt_feasibility_metrics',
      'job_id': job['job_id'],
      'arm': job['arm'],
      'nfe_steps': job['generation']['nfe_steps'],
      'num_samples': 256,
      'sequence_length': 1024,
      'generation_seed': 260703,
      'token_entropy_nats': math.log(256),
      'reference_lm': score_summary,
    })
    host = {
      'hostname': 'fixture-host',
      'platform': 'Linux-fixture',
      'python': '3.12.11',
      'torch': '2.3.1',
      'cuda_runtime': '12.1',
      'gpu': {
        'index': 0,
        'name': 'NVIDIA L4',
        'uuid': 'GPU-fixture',
        'driver_version': 'fixture',
        'memory_total_mib': 23034,
      },
      'critical_packages': {
        'torch': '2.3.1',
        'transformers': '4.46.2',
        'tokenizers': '0.20.3',
        'flash-attn': '2.7.4.post1',
        'packaging': '23.2',
        'lightning': '2.2.1',
        'pytorch-lightning': '2.2.1',
        'torchmetrics': '1.3.2',
        'lightning-utilities': '0.15.3',
        'fsspec': '2024.2.0',
        'setuptools': '69.5.1',
        'timm': '0.9.16',
        'torchvision': '0.18.1',
      },
      'precision_policy': 'upstream_float32_no_autocast',
    }
    resource_path = run_dir / 'resource-metrics.json'
    self._write_json(resource_path, {
      'schema_version': 1,
      'artifact': 'tensor_train_owt_resource_metrics',
      'job_id': job['job_id'],
      'measurement_scope': 'single_job_uncontended_end_to_end_v1',
      'host': host,
      'timing_seconds': {
        'model_load': 1.0,
        'generation': 2.0,
        'evaluator_load_and_scoring': 3.0,
        'total': 7.0,
      },
      'throughput': {
        'generation_samples_per_second': 128.0,
        'generation_tokens_per_second': 131072.0,
        'evaluator_samples_per_second': 85.3333333333,
      },
      'cuda_memory_bytes': {
        'model_load_peak_allocated': 1,
        'model_load_peak_reserved': 1,
        'generation_peak_allocated': 1,
        'generation_peak_reserved': 1,
        'evaluator_peak_allocated': 1,
        'evaluator_peak_reserved': 1,
      },
      'process': {'pid': 1, 'max_rss_bytes': 1},
      'generation': {
        'requested_nfe_steps': job['generation']['nfe_steps'],
        'observed_mean_steps': float(job['generation']['nfe_steps']),
        'tokens_per_step': job['generation']['tokens_per_step'],
        'num_samples': 256,
        'sequence_length': 1024,
        'batch_size': 1,
      },
      'gpu_exclusivity': {
        'required': True,
        'policy': GPU_EXCLUSIVITY_POLICY,
        'lock_path': str(SUBMISSION_GPU_LOCK),
        'lock_acquired': True,
        'monitor_interval_seconds': 1.0,
        'monitor_samples': 2,
        'preflight_other_compute_pids': [],
        'postflight_other_compute_pids': [],
        'foreign_pid_observations': [],
        'monitor_errors': [],
      },
    })
    config_path = run_dir / 'resolved_config.yaml'
    decomposition = job['decomposition']
    resolved_config = {
      'name': 'mdlm',
      'model': {'name': 'small'},
      'training': {'weight_noise_coeff': 0.01},
      'generation': {
        'length': 1024,
        'ckpt_path': job['checkpoint']['path'],
        'total_samples': 256,
        'eval_model_name': 'gpt2-large',
        'temperature': 1.0,
        'batch_size': 1,
        'sampling': 'top-k',
        'k': tokens_per_step,
        'gamma': 0.1,
        'ordering': 'random',
      },
      'type': {'model': 'mdlm', 'mdlm': 'owt'},
      'rank_arch': '2-layer',
      'algo': {
        'decomp': decomposition['decomposition'],
        'cp': {
          'rank': decomposition['rank'] if job['arm'] == 'marginal' else 1,
          'rank_weights': decomposition['rank_weights'],
        },
        'tt': {
          'rank': decomposition['rank'] if job['arm'] != 'marginal' else 4,
          'marginal_head': decomposition['marginal_head'],
        },
        'output_dtype': 'float32',
        'causal_attention': False,
      },
      'dataset': {'cache_dir': '/fixture'},
      'is_di4c': False,
      'hydra': {'run': {'dir': '.'}},
    }
    config_path.write_text(yaml.safe_dump(resolved_config, sort_keys=True))

    def descriptor(path):
      return {
        'path': path.name,
        'sha256': sha256_file(path),
        'size_bytes': path.stat().st_size,
      }

    manifest_path = run_dir / 'manifest.json'
    self._write_json(manifest_path, {
      'schema_version': 1,
      'artifact': 'tensor_train_owt_feasibility_run',
      'scientific_scope': plan['scientific_scope'],
      'job_id': job['job_id'],
      'arm': job['arm'],
      'nfe_steps': job['generation']['nfe_steps'],
      'job_spec_sha256': job['job_spec_sha256'],
      'plan_id': plan['plan_id'],
      'plan_file_sha256': sha256_file(plan_path),
      'start_time_utc': '2026-08-31T00:00:00+00:00',
      'end_time_utc': '2026-08-31T00:00:07+00:00',
      'runtime': host,
      'source': job['source'],
      'checkpoint': job['checkpoint'],
      'harness_repository': job['harness_repository'],
      'model_inputs': job['model_inputs'],
      'cache': job['cache'],
      'model_load': {
        'class': 'mdlm.MDLM',
        'backbone_class': 'fixture.Backbone',
        'tokenizer_class': 'fixture.Tokenizer',
        'mask_token_id': 50257,
        'tokenizer_vocab_size': 50257,
        'ema_used': False,
        'checkpoint_step': EXPECTED_CHECKPOINT_STEPS[job['arm']],
        'checkpoint_payload_keys': [
          'config', 'model', 'optimizer', 'scheduler', 'step'],
        'checkpoint_state_keys': list(
          EXPECTED_CHECKPOINT_STATE_KEYS[job['arm']]),
        'checkpoint_config_sha256': EXPECTED_CHECKPOINT_CONFIG_SHA256[
          job['arm']],
        'checkpoint_config_identity': {
          'name': 'mdlm',
          'rank_arch': '2-layer',
          'type_model': 'mdlm',
          'type_mdlm': 'owt',
          'decomposition': decomposition['decomposition'],
          'rank': decomposition['rank'],
          'rank_weights': decomposition['rank_weights'],
          'marginal_head': decomposition['marginal_head'],
          'output_dtype': 'float32',
          'causal_attention': False,
        },
        'missing_keys': ['backbone.fixture'],
        'unexpected_keys': [],
        'parameter_dtypes': ['torch.float32'],
        'decomposition': decomposition,
        'backbone_input': job['model_inputs']['backbone'],
        'tokenizer_input': job['model_inputs']['tokenizer'],
        'cache_identity_sha256': job['cache']['identity_sha256'],
      },
      'evaluator': score_summary,
      'outputs': {
        'samples_jsonl': descriptor(samples_path),
        'metrics_json': descriptor(metrics_path),
        'resource_metrics_json': descriptor(resource_path),
        'resolved_config_yaml': descriptor(config_path),
        'position_schedules_json': descriptor(schedule_path),
      },
      'interruption_policy': job['runtime']['interruption_policy'],
    })
    self._write_json(run_dir / '_SUCCESS.json', {
      'schema_version': 1,
      'artifact': 'tensor_train_owt_feasibility_success',
      'job_id': job['job_id'],
      'job_spec_sha256': job['job_spec_sha256'],
      'manifest_sha256': sha256_file(manifest_path),
    })

  def _rehash_run(self, run_dir: Path) -> None:
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for descriptor in manifest['outputs'].values():
      path = run_dir / descriptor['path']
      descriptor['sha256'] = sha256_file(path)
      descriptor['size_bytes'] = path.stat().st_size
    self._write_json(manifest_path, manifest)
    success_path = run_dir / '_SUCCESS.json'
    success = json.loads(success_path.read_text())
    success['manifest_sha256'] = sha256_file(manifest_path)
    self._write_json(success_path, success)

  def test_frozen_protocol_and_six_job_step_mapping(self):
    protocol = load_protocol(DEFAULT_PROTOCOL)
    self.assertEqual(protocol['source']['revision'], OFFICIAL_SOURCE_REVISION)
    self.assertEqual(
      protocol['checkpoints']['revision'], OFFICIAL_CHECKPOINT_REVISION)
    self.assertEqual(protocol['runtime']['critical_packages'], {
      'torch': '2.3.1',
      'transformers': '4.46.2',
      'tokenizers': '0.20.3',
      'flash-attn': '2.7.4.post1',
      'packaging': '23.2',
      'lightning': '2.2.1',
      'pytorch-lightning': '2.2.1',
      'torchmetrics': '1.3.2',
      'lightning-utilities': '0.15.3',
      'fsspec': '2024.2.0',
      'setuptools': '69.5.1',
      'timm': '0.9.16',
      'torchvision': '0.18.1',
    })
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)

    self.assertEqual(plan['num_jobs'], 6)
    self.assertEqual(list(jobs), [
      'owt--marginal--nfe008--s260703',
      'owt--marginal--nfe016--s260703',
      'owt--marginal--nfe032--s260703',
      'owt--tensor_train_rank4--nfe008--s260703',
      'owt--tensor_train_rank4--nfe016--s260703',
      'owt--tensor_train_rank4--nfe032--s260703',
    ])
    self.assertEqual(
      {job['generation']['nfe_steps']: job['generation']['tokens_per_step']
       for job in jobs.values()},
      {8: 128, 16: 64, 32: 32})
    self.assertTrue(all(
      job['generation']['num_samples'] == 256
      and job['generation']['sequence_length'] == 1024
      and job['generation']['ordering'] == 'random'
      and job['generation']['temperature'] == 1.0
      and job['generation']['batch_size'] == 1
      and job['generation']['generation_seed'] == 260703
      for job in jobs.values()))

  def test_protocol_rejects_mutable_or_unknown_fields(self):
    payload = yaml.safe_load(DEFAULT_PROTOCOL.read_text())
    payload['source']['revision'] = 'b' * 40
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'protocol.yaml'
      path.write_text(yaml.safe_dump(payload))
      with self.assertRaisesRegex(ValueError, 'audited official revision'):
        load_protocol(path)

    payload = yaml.safe_load(DEFAULT_PROTOCOL.read_text())
    payload['generation']['unregistered_option'] = True
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'protocol.yaml'
      path.write_text(yaml.safe_dump(payload))
      with self.assertRaisesRegex(ValueError, 'schema mismatch'):
        load_protocol(path)

    payload = yaml.safe_load(DEFAULT_PROTOCOL.read_text())
    del payload['runtime']['critical_packages']['lightning']
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'protocol.yaml'
      path.write_text(yaml.safe_dump(payload))
      with self.assertRaisesRegex(ValueError, 'schema mismatch'):
        load_protocol(path)

  def test_compiled_plan_replays_job_commitments(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      replayed_plan, replayed_jobs = load_compiled_plan(plan_path)
      self.assertEqual(replayed_plan, plan)
      self.assertEqual(replayed_jobs, jobs)

      job_path = plan_path.parent / f'{plan["job_ids"][0]}.json'
      job_payload = json.loads(job_path.read_text())
      job_payload['generation']['batch_size'] = 16
      job_path.write_text(json.dumps(job_payload))
      with self.assertRaisesRegex(ValueError, 'job-spec commitment mismatch'):
        load_compiled_plan(plan_path)

  def test_complete_run_replay_and_resume_are_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      job = jobs[plan['job_ids'][0]]
      self._write_completed_run(plan, job, plan_path)
      validated = validate_completed_run(
        Path(job['artifact_dir']), plan=plan, job=job)
      self.assertEqual(len(validated['samples']), 256)
      reused = run_job(plan_path, job['job_id'], resume=True)
      self.assertEqual(
        reused['event'], 'tensor_train_feasibility_job_reused')

      samples_path = Path(job['artifact_dir']) / 'samples.jsonl'
      samples_path.write_text(samples_path.read_text() + '{}\n')
      with self.assertRaisesRegex(ValueError, 'output hash mismatch'):
        run_job(plan_path, job['job_id'], resume=True)

  def test_completed_run_requires_submission_wide_gpu_lock(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      job = jobs[plan['job_ids'][0]]
      self._write_completed_run(plan, job, plan_path)
      run_dir = Path(job['artifact_dir'])
      resource_path = run_dir / 'resource-metrics.json'
      resource = json.loads(resource_path.read_text())
      resource['gpu_exclusivity']['lock_path'] = str(
        root / '.private-gpu.lock')
      self._write_json(resource_path, resource)
      self._rehash_run(run_dir)
      with self.assertRaisesRegex(
          ValueError, 'submission-wide lock'):
        validate_completed_run(run_dir, plan=plan, job=job)

  def test_complete_matrix_requires_all_pairs_and_one_evaluator(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      for job in jobs.values():
        self._write_completed_run(plan, job, plan_path)
      result = verify_complete_matrix(plan_path)

    self.assertEqual(result['num_jobs'], 6)
    self.assertEqual(len(result['cells']), 6)
    self.assertEqual(
      {(cell['arm'], cell['nfe_steps']) for cell in result['cells']}, {
        ('marginal', 8), ('marginal', 16), ('marginal', 32),
        ('tensor_train_rank4', 8),
        ('tensor_train_rank4', 16),
        ('tensor_train_rank4', 32),
      })

  def test_new_run_path_is_atomic_and_validated(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      job = jobs[plan['job_ids'][0]]

      def fake_execute(**kwargs):
        self._write_completed_run(
          kwargs['plan'], kwargs['job'], kwargs['plan_path'],
          run_dir=kwargs['temporary_dir'])

      with mock.patch(
          'scripts.run_tensor_train_feasibility._exclusive_gpu_lock',
          return_value=contextlib.nullcontext(SUBMISSION_GPU_LOCK)), mock.patch(
            'scripts.run_tensor_train_feasibility._execute',
            side_effect=fake_execute) as execute:
        result = run_job(plan_path, job['job_id'], resume=False)
      self.assertEqual(result['event'], 'tensor_train_feasibility_job_complete')
      self.assertEqual(execute.call_count, 1)
      self.assertEqual(
        execute.call_args.kwargs['lock_path'], SUBMISSION_GPU_LOCK)
      self.assertTrue(Path(job['artifact_dir'], '_SUCCESS.json').is_file())
      self.assertFalse(any(
        path.name.startswith(f'.{job["job_id"]}.partial-')
        for path in Path(job['artifact_dir']).parent.iterdir()))

  def test_failed_new_run_preserves_partial_without_final_directory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      job = jobs[plan['job_ids'][0]]
      with mock.patch(
          'scripts.run_tensor_train_feasibility._exclusive_gpu_lock',
          return_value=contextlib.nullcontext(SUBMISSION_GPU_LOCK)), mock.patch(
            'scripts.run_tensor_train_feasibility._execute',
            side_effect=RuntimeError('injected new-run failure')):
        with self.assertRaisesRegex(RuntimeError, 'injected new-run failure'):
          run_job(plan_path, job['job_id'], resume=False)
      self.assertFalse(Path(job['artifact_dir']).exists())
      partials = [
        path for path in Path(job['artifact_dir']).parent.iterdir()
        if path.name.startswith(f'.{job["job_id"]}.partial-')]
      self.assertEqual(len(partials), 1)

  def test_offline_cache_bytes_are_replayed_before_execution(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      job = jobs[plan['job_ids'][0]]
      changed = json.loads(json.dumps(job['cache']))
      changed['snapshots']['backbone']['files'][0]['sha256'] = 'f' * 64
      changed['snapshots']['backbone']['files_manifest_sha256'] = canonical_sha256(
        changed['snapshots']['backbone']['files'])
      changed_body = {key: value for key, value in changed.items()
                      if key != 'identity_sha256'}
      changed['identity_sha256'] = canonical_sha256(changed_body)
      with mock.patch.dict(os.environ, {}, clear=False), mock.patch(
          'scripts.run_tensor_train_feasibility.cached_model_identities',
          return_value=changed):
        with self.assertRaisesRegex(RuntimeError, 'cached model bytes'):
          _prepare_offline_cache(plan, job)

  def test_resolved_config_and_model_load_are_semantically_replayed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      job = jobs[plan['job_ids'][0]]
      self._write_completed_run(plan, job, plan_path)
      run_dir = Path(job['artifact_dir'])

      config_path = run_dir / 'resolved_config.yaml'
      config = yaml.safe_load(config_path.read_text())
      config['generation']['ordering'] = 'left-to-right'
      config_path.write_text(yaml.safe_dump(config, sort_keys=True))
      self._rehash_run(run_dir)
      with self.assertRaisesRegex(ValueError, 'generation config'):
        validate_completed_run(run_dir, plan=plan, job=job)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      job = jobs[plan['job_ids'][0]]
      self._write_completed_run(plan, job, plan_path)
      run_dir = Path(job['artifact_dir'])
      manifest_path = run_dir / 'manifest.json'
      manifest = json.loads(manifest_path.read_text())
      manifest['model_load'] = {'fixture': True}
      self._write_json(manifest_path, manifest)
      self._rehash_run(run_dir)
      with self.assertRaisesRegex(ValueError, 'model_load schema mismatch'):
        validate_completed_run(run_dir, plan=plan, job=job)

  def test_evaluator_runtime_requires_complete_exact_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      job = jobs[plan['job_ids'][0]]
      self._write_completed_run(plan, job, plan_path)
      run_dir = Path(job['artifact_dir'])
      metrics_path = run_dir / 'metrics.json'
      metrics = json.loads(metrics_path.read_text())
      del metrics['reference_lm']['runtime_identity']['tokenizer_revision']
      self._write_json(metrics_path, metrics)
      manifest_path = run_dir / 'manifest.json'
      manifest = json.loads(manifest_path.read_text())
      manifest['evaluator'] = metrics['reference_lm']
      self._write_json(manifest_path, manifest)
      self._rehash_run(run_dir)
      with self.assertRaisesRegex(ValueError, 'runtime identity schema mismatch'):
        validate_completed_run(run_dir, plan=plan, job=job)

  def test_matrix_rejects_equal_seed_with_different_position_schedule(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      plan, jobs = self._compile(root)
      plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
      for job in jobs.values():
        self._write_completed_run(plan, job, plan_path)
      target = jobs['owt--tensor_train_rank4--nfe008--s260703']
      run_dir = Path(target['artifact_dir'])
      schedule_path = run_dir / 'position-schedules.json'
      schedule = json.loads(schedule_path.read_text())
      chunks = schedule['records'][0]['chunks']
      chunks[0][-1], chunks[1][0] = chunks[1][0], chunks[0][-1]
      chunks[0].sort()
      chunks[1].sort()
      changed_hash = canonical_sha256(chunks)
      schedule['records'][0]['position_schedule_sha256'] = changed_hash
      self._write_json(schedule_path, schedule)
      samples_path = run_dir / 'samples.jsonl'
      rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
      rows[0]['position_schedule_sha256'] = changed_hash
      samples_path.write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
      self._rehash_run(run_dir)
      with self.assertRaisesRegex(ValueError, 'sample/schedule identity'):
        verify_complete_matrix(plan_path)

  def test_entropy_scope_and_manifest_evaluator_are_bound(self):
    mutations = ('entropy', 'scope', 'evaluator')
    for mutation in mutations:
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan, jobs = self._compile(root)
        plan_path = write_plan(plan, jobs, root / 'artifacts/plan')
        job = jobs[plan['job_ids'][0]]
        self._write_completed_run(plan, job, plan_path)
        run_dir = Path(job['artifact_dir'])
        if mutation == 'entropy':
          metrics_path = run_dir / 'metrics.json'
          metrics = json.loads(metrics_path.read_text())
          metrics['token_entropy_nats'] += 0.1
          self._write_json(metrics_path, metrics)
          expected = 'token entropy'
        else:
          manifest_path = run_dir / 'manifest.json'
          manifest = json.loads(manifest_path.read_text())
          if mutation == 'scope':
            manifest['scientific_scope'] = 'changed after execution'
            expected = 'manifest identity'
          else:
            manifest['evaluator']['perplexity'] += 1.0
            expected = 'manifest evaluator'
          self._write_json(manifest_path, manifest)
        self._rehash_run(run_dir)
        with self.assertRaisesRegex(ValueError, expected):
          validate_completed_run(run_dir, plan=plan, job=job)

  def test_post_load_reseed_cancels_arm_dependent_rng_consumption(self):
    import torch

    torch.manual_seed(1)
    _ = torch.rand(7)
    _reseed_sampling(torch, 260703)
    first = torch.rand(16)
    torch.manual_seed(1)
    _ = torch.rand(1009)
    _reseed_sampling(torch, 260703)
    second = torch.rand(16)
    self.assertTrue(torch.equal(first, second))

  def test_schedule_recorder_binds_complete_selected_position_chunks(self):
    import torch

    generation = {
      'num_samples': 1,
      'batch_size': 1,
      'nfe_steps': 2,
      'tokens_per_step': 2,
      'sequence_length': 4,
      'generation_seed': 260703,
      'schedule_policy': 'recorded_selected_position_chunks_v1',
    }
    recorder = _PositionScheduleRecorder(generation)
    recorder.record(
      torch.tensor([[0, 2]]), ordering='random', requested_k=2,
      logprobs=None)
    recorder.record(
      torch.tensor([[1, 3]]), ordering='random', requested_k=2,
      logprobs=None)
    artifact = recorder.finalize(job={
      'job_id': 'fixture', 'arm': 'marginal', 'generation': generation})
    self.assertEqual(
      artifact['records'][0]['position_schedule_sha256'],
      canonical_sha256([[0, 2], [1, 3]]))

  def test_gpu_lock_and_continuous_monitor_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      lock_path = Path(directory) / 'gpu.lock'
      with _exclusive_gpu_lock(lock_path):
        with self.assertRaisesRegex(RuntimeError, 'holds the GPU lock'):
          with _exclusive_gpu_lock(lock_path):
            self.fail('nested lock unexpectedly succeeded')

    responses = iter(([], [], [999], [999], [999]))

    def fake_pids():
      return next(responses, [999])

    with mock.patch(
        'scripts.run_tensor_train_feasibility._other_compute_pids',
        side_effect=fake_pids):
      with self.assertRaisesRegex(RuntimeError, 'exclusivity monitor'):
        with _ForeignPidMonitor(60.0) as monitor:
          monitor.snapshot(lock_path=Path('/fixture/gpu.lock'))
          monitor.snapshot(lock_path=Path('/fixture/gpu.lock'))

  def test_real_preflight_entrypoint_is_importable_without_gpu(self):
    from scripts.preflight_tensor_train_feasibility import (
      _descriptive_resource_probe,
      run_preflight,
    )

    self.assertTrue(callable(run_preflight))
    probe = _descriptive_resource_probe(
      generation_seconds=1.25,
      generation_peak_allocated_bytes=10,
      generation_peak_reserved_bytes=12)
    self.assertEqual(probe['generation_seconds'], 1.25)
    with self.assertRaisesRegex(RuntimeError, 'finite and positive'):
      _descriptive_resource_probe(
        generation_seconds=float('nan'),
        generation_peak_allocated_bytes=10,
        generation_peak_reserved_bytes=12)
    with self.assertRaisesRegex(RuntimeError, 'reserved memory'):
      _descriptive_resource_probe(
        generation_seconds=1.25,
        generation_peak_allocated_bytes=13,
        generation_peak_reserved_bytes=12)


if __name__ == '__main__':
  unittest.main()
