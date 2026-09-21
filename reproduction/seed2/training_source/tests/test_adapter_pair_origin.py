import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from safetensors.torch import save_file
import torch

from evaluation.adapter_pair_origin import (
  ARMS,
  _load_and_validate_adapter_origin_manifest,
  bind_generation_arm_to_adapter_origin_evidence,
  build_adapter_pair_origin_evidence,
  load_and_validate_adapter_pair_origin_evidence,
  write_adapter_pair_origin_evidence,
)
from scripts.compile_experiment_matrix import (
  JOB_SCHEMA_VERSION,
  PLAN_SCHEMA_VERSION,
  sha256_file,
)
from scripts.export_structured_adapter import RELEASED_BACKBONE_IDENTITY
from scripts.run_compiled_job import (
  SUCCESS_MARKER,
  _job_digest,
  _job_execution_digest,
  _output_records,
)


def _manifest_payload(
    *,
    arm: str,
    candidate_k: int,
    checkpoint_path: Path,
    adapter_path: Path,
    global_step: int,
) -> dict:
  topology_mode = 'dynamic' if arm == 'dynamic_dynamic' else 'fixed'
  topology_weight = 0.1 if arm == 'dynamic_dynamic' else 0.0
  identity = {
    'control_identity': arm,
    'topology_mode': topology_mode,
    'factor_mode': topology_mode,
    'candidate_top_k': candidate_k,
    'independent_mode': False,
    'topology_weight': topology_weight,
    'head_semantics': {'fixture': True},
    'training_semantics': {'fixture': True},
  }
  return {
    'schema_version': 4,
    'structured_decoder_identity': identity,
    'structured_decoder_identity_sha256': hashlib.sha256(
      json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest(),
    'source_checkpoint_sha256': sha256_file(checkpoint_path),
    'source_checkpoint_size_bytes': checkpoint_path.stat().st_size,
    'source_checkpoint_global_step': global_step,
    'adapter_sha256': sha256_file(adapter_path),
    'adapter_size_bytes': adapter_path.stat().st_size,
    'released_backbone': dict(RELEASED_BACKBONE_IDENTITY),
  }


class AdapterPairFixture:

  def __init__(self, root: Path, *, candidate_k: int = 128):
    self.root = root
    self.plan_dir = root / 'plan'
    self.plan_dir.mkdir()
    (self.plan_dir / 'jobs').mkdir()
    self.artifact_root = root / 'artifacts'
    self.plan_id = '1' * 64
    self.protocol_id = 'contextual-forest-expansion-v1'
    self.manifest_sha = '2' * 64
    self.repository_sha = '3' * 40
    self.suite = 'candidate_k_128_pilot'
    self.candidate_k = candidate_k
    self.train_seed = 1
    self.jobs: dict[str, dict] = {}
    self.manifests: dict[str, Path] = {}
    for arm in ARMS:
      self._arm(arm)
    self.plan = {
      'schema_version': PLAN_SCHEMA_VERSION,
      'protocol_id': self.protocol_id,
      'source_manifest_sha256': self.manifest_sha,
      'artifact_root': str(self.artifact_root),
      'selected_suites': [self.suite],
      'plan_id': self.plan_id,
      'repository': {'sha': self.repository_sha, 'dirty': False},
      'num_jobs': len(self.jobs),
      'job_ids': list(self.jobs),
      'job_spec_sha256': {
        job_id: _job_digest(job) for job_id, job in self.jobs.items()},
    }
    (self.plan_dir / 'compiled-plan.json').write_text(json.dumps(self.plan))
    for job_id, job in self.jobs.items():
      (self.plan_dir / 'jobs' / f'{job_id}.json').write_text(json.dumps(job))

  def _base_job(
      self, *, job_id: str, kind: str, artifact_dir: Path,
      identity: dict, dependencies: list[str], argv: list[str],
      required_outputs: list[dict]) -> dict:
    return {
      'schema_version': JOB_SCHEMA_VERSION,
      'protocol_id': self.protocol_id,
      'source_manifest_sha256': self.manifest_sha,
      'plan_id': self.plan_id,
      'source_repository_sha': self.repository_sha,
      'job_id': job_id,
      'kind': kind,
      'artifact_dir': str(artifact_dir),
      'suites': [self.suite],
      'dependencies': dependencies,
      'identity': identity,
      'argv': argv,
      'execution_mode': 'fresh_attempt',
      'external_inputs': [],
      'required_outputs': required_outputs,
    }

  @staticmethod
  def _write_marker(job: dict, run_dir: Path) -> dict:
    outputs = _output_records(run_dir, job['required_outputs'])
    marker = {
      'schema_version': 2,
      'artifact': 'compiled_experiment_job_success',
      'job_id': job['job_id'],
      'originating_plan_id': job['plan_id'],
      'source_repository_sha': job['source_repository_sha'],
      'job_execution_sha256': _job_execution_digest(job),
      'run_dir': str(run_dir),
      'argv': job['argv'],
      'start_time_utc': '2026-08-30T00:00:00+00:00',
      'end_time_utc': '2026-08-30T00:01:00+00:00',
      'outputs': outputs,
    }
    marker_path = Path(job['artifact_dir']) / SUCCESS_MARKER
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker))
    return marker

  def _arm(self, arm: str) -> None:
    suffix = f's{self.train_seed:03d}--k{self.candidate_k:03d}'
    train_id = f'train--{arm}--{suffix}'
    export_id = f'export--{arm}--{suffix}'
    train_artifact = self.artifact_root / 'runs' / train_id
    train_run = train_artifact / 'attempts' / 'attempt-0001'
    (train_run / 'checkpoints').mkdir(parents=True)
    (train_run / 'data_provenance').mkdir()
    checkpoint_path = train_run / 'checkpoints' / 'last.ckpt'
    checkpoint_path.write_bytes(f'{arm}-checkpoint'.encode())
    (train_run / 'data_provenance' / 'train-fixture.json').write_text(
      '{"split":"train"}\n')
    (train_run / 'data_provenance' / 'valid-fixture.json').write_text(
      '{"split":"validation"}\n')
    train_job = self._base_job(
      job_id=train_id,
      kind='train',
      artifact_dir=train_artifact,
      identity={
        'control': arm,
        'train_seed': self.train_seed,
        'candidate_k': self.candidate_k,
        'updates': 1000,
      },
      dependencies=[],
      argv=['python', 'main.py'],
      required_outputs=[
        {'name': 'checkpoint', 'pattern': 'checkpoints/last.ckpt',
         'exactly_one': True},
        {'name': 'training_data_provenance',
         'pattern': 'data_provenance/train-*.json', 'exactly_one': True},
        {'name': 'training_validation_data_provenance',
         'pattern': 'data_provenance/valid-*.json', 'exactly_one': True},
      ],
    )
    self.jobs[train_id] = train_job
    self._write_marker(train_job, train_run)

    export_artifact = self.artifact_root / 'runs' / export_id
    export_run = export_artifact / 'attempts' / 'attempt-0001'
    export_run.mkdir(parents=True)
    adapter_path = export_run / 'adapter.safetensors'
    manifest_path = export_run / 'adapter-manifest.json'
    adapter_path.write_bytes(f'{arm}-adapter'.encode())
    manifest = _manifest_payload(
      arm=arm,
      candidate_k=self.candidate_k,
      checkpoint_path=checkpoint_path,
      adapter_path=adapter_path,
      global_step=1000,
    )
    manifest_path.write_text(json.dumps(manifest))
    topology_mode = 'dynamic' if arm == 'dynamic_dynamic' else 'fixed'
    topology_weight = 0.1 if arm == 'dynamic_dynamic' else 0.0
    export_job = self._base_job(
      job_id=export_id,
      kind='export',
      artifact_dir=export_artifact,
      identity={
        'control': arm,
        'train_seed': self.train_seed,
        'candidate_k': self.candidate_k,
        'topology_mode': topology_mode,
        'factor_mode': topology_mode,
        'independent_mode': False,
        'topology_weight': topology_weight,
      },
      dependencies=[train_id],
      argv=[
        'python', 'scripts/export_structured_adapter.py',
        '--checkpoint', str(checkpoint_path.resolve()),
        '--output', str(adapter_path.resolve()),
        '--manifest', str(manifest_path.resolve()),
        '--expected-checkpoint-sha256', sha256_file(checkpoint_path),
        '--expected-global-step', '1000',
      ],
      required_outputs=[
        {'name': 'adapter', 'pattern': 'adapter.safetensors',
         'exactly_one': True},
        {'name': 'adapter_manifest', 'pattern': 'adapter-manifest.json',
         'exactly_one': True},
      ],
    )
    self.jobs[export_id] = export_job
    self._write_marker(export_job, export_run)
    self.manifests[arm] = manifest_path

  @staticmethod
  def validate_manifest(
      manifest_path: Path, adapter_path: Path, *, expected_identity: dict,
      expected_adapter_sha256: str,
      expected_manifest_sha256: str) -> dict:
    if sha256_file(manifest_path) != expected_manifest_sha256:
      raise ValueError('fixture manifest SHA mismatch')
    if sha256_file(adapter_path) != expected_adapter_sha256:
      raise ValueError('fixture adapter SHA mismatch')
    payload = json.loads(manifest_path.read_text())
    if payload['structured_decoder_identity'] != expected_identity:
      raise ValueError('fixture identity mismatch')
    return payload

  def refresh_export_marker(self, arm: str) -> None:
    suffix = f's{self.train_seed:03d}--k{self.candidate_k:03d}'
    export_id = f'export--{arm}--{suffix}'
    job = self.jobs[export_id]
    run_dir = Path(job['artifact_dir']) / 'attempts' / 'attempt-0001'
    self._write_marker(job, run_dir)


class AdapterPairOriginTest(unittest.TestCase):

  def _patches(self, fixture: AdapterPairFixture):
    return (
      mock.patch(
        'scripts.aggregate_hierarchical_document_eval.'
        '_validate_repository_checkout'),
      mock.patch(
        'evaluation.adapter_pair_origin.load_and_validate_adapter_manifest',
        side_effect=fixture.validate_manifest),
    )

  def test_builds_and_revalidates_complete_pair_origin(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = AdapterPairFixture(Path(directory))
      checkout_patch, manifest_patch = self._patches(fixture)
      with checkout_patch, manifest_patch:
        payload = build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite=fixture.suite,
          candidate_k=128,
          train_seed=1,
          created_utc='2026-08-30T12:00:00+00:00',
        )
        output = Path(directory) / 'adapter-pair-origin.json'
        file_sha = write_adapter_pair_origin_evidence(output, payload)
        validated = load_and_validate_adapter_pair_origin_evidence(
          output,
          expected_evidence_sha256=file_sha,
          expected_plan_sha256=sha256_file(
            fixture.plan_dir / 'compiled-plan.json'),
          expected_suite=fixture.suite,
          expected_candidate_k=128,
          expected_train_seed=1,
        )

      self.assertEqual(set(validated['arms']), set(ARMS))
      self.assertFalse(validated['source']['legacy_plan_schema'])
      for arm in ARMS:
        train = validated['arms'][arm]['train']
        export = validated['arms'][arm]['export']
        self.assertEqual(set(train['outputs']), {
          'checkpoint', 'training_data_provenance',
          'training_validation_data_provenance'})
        self.assertEqual(set(export['outputs']), {
          'adapter', 'adapter_manifest'})
        self.assertEqual(
          validated['arms'][arm]['adapter_origin']
          ['source_checkpoint_sha256'],
          train['outputs']['checkpoint']['sha256'])
        self.assertEqual(
          validated['arms'][arm]['adapter_origin']['released_backbone'],
          RELEASED_BACKBONE_IDENTITY)

  def test_rejects_manifest_checkpoint_not_committed_by_train_marker(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = AdapterPairFixture(Path(directory))
      manifest_path = fixture.manifests['dynamic_dynamic']
      payload = json.loads(manifest_path.read_text())
      payload['source_checkpoint_sha256'] = 'f' * 64
      manifest_path.write_text(json.dumps(payload))
      fixture.refresh_export_marker('dynamic_dynamic')
      checkout_patch, manifest_patch = self._patches(fixture)
      with checkout_patch, manifest_patch, self.assertRaisesRegex(
          ValueError, 'source checkpoint differs from the train marker'):
        build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite=fixture.suite,
          candidate_k=128,
          train_seed=1,
        )

  def test_rejects_source_output_drift_even_with_unchanged_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = AdapterPairFixture(Path(directory))
      checkout_patch, manifest_patch = self._patches(fixture)
      output = Path(directory) / 'adapter-pair-origin.json'
      with checkout_patch, manifest_patch:
        payload = build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite=fixture.suite,
          candidate_k=128,
          train_seed=1,
        )
        write_adapter_pair_origin_evidence(output, payload)
      checkpoint = Path(
        payload['arms']['static_static']['train']['outputs']
        ['checkpoint']['path'])
      checkpoint.write_bytes(b'drifted checkpoint')
      checkout_patch, manifest_patch = self._patches(fixture)
      with checkout_patch, manifest_patch, self.assertRaisesRegex(
          ValueError, 'outputs drifted'):
        load_and_validate_adapter_pair_origin_evidence(output)

  def test_rejects_semantic_evidence_tamper_after_self_hash_rewrite(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = AdapterPairFixture(Path(directory))
      checkout_patch, manifest_patch = self._patches(fixture)
      with checkout_patch, manifest_patch:
        payload = build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite=fixture.suite,
          candidate_k=128,
          train_seed=1,
        )
      payload['arms']['dynamic_dynamic']['train']['job_spec_sha256'] = 'f' * 64
      body = {key: value for key, value in payload.items()
              if key != 'evidence_sha256'}
      payload['evidence_sha256'] = hashlib.sha256(
        json.dumps(
          body, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ).encode()
      ).hexdigest()
      output = Path(directory) / 'tampered-origin.json'
      output.write_text(json.dumps(payload))
      checkout_patch, manifest_patch = self._patches(fixture)
      with checkout_patch, manifest_patch, self.assertRaisesRegex(
          ValueError, 'differs from live sources'):
        load_and_validate_adapter_pair_origin_evidence(output)

  def test_legacy_path_uses_analysis_marker_validator(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = AdapterPairFixture(Path(directory), candidate_k=64)
      fixture.suite = 'pilot'
      fixture.plan['selected_suites'] = ['pilot']
      fixture.plan['repository'] = {
        'sha': fixture.repository_sha, 'dirty': False}
      for job in fixture.jobs.values():
        job['suites'] = ['pilot']
        job.pop('source_repository_sha')
        marker_path = Path(job['artifact_dir']) / SUCCESS_MARKER
        marker = json.loads(marker_path.read_text())
        marker['schema_version'] = 1
        marker.pop('source_repository_sha')
        marker['job_execution_sha256'] = _job_execution_digest(job)
        marker_path.write_text(json.dumps(marker))
      fixture.plan['job_spec_sha256'] = {
        job_id: _job_digest(job) for job_id, job in fixture.jobs.items()}
      plan_path = fixture.plan_dir / 'compiled-plan.json'
      plan_path.write_text(json.dumps(fixture.plan))
      plan_sha = sha256_file(plan_path)
      with mock.patch(
          'evaluation.adapter_pair_origin._load_plan_for_analysis',
          return_value=(fixture.plan, fixture.jobs, plan_sha, True)
      ) as loader, mock.patch(
          'evaluation.adapter_pair_origin.load_and_validate_adapter_manifest',
          side_effect=fixture.validate_manifest):
        payload = build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite='pilot',
          candidate_k=64,
          train_seed=1,
        )
      loader.assert_called_once_with(
        fixture.plan_dir.resolve(), require_current_repository_match=False)
      self.assertTrue(payload['source']['legacy_plan_schema'])

  def test_validates_exact_legacy_adapter_manifest_schema(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      adapter_path = root / 'adapter.safetensors'
      manifest_path = root / 'adapter-manifest.json'
      save_file(
        {'weight': torch.ones(2)},
        str(adapter_path),
        metadata={
          'artifact_role': 'contextual_forest_structured_head',
          'source_namespace': 'structured_head.',
          'file_namespace': 'prefix-stripped',
        },
      )
      manifest = {
        'artifact_role': 'contextual_forest_structured_adapter',
        'schema_version': 1,
        'format': 'safetensors',
        'adapter_file': 'adapter.safetensors',
        'adapter_sha256': sha256_file(adapter_path),
        'adapter_size_bytes': adapter_path.stat().st_size,
        'adapter_tensor_count': 1,
        'adapter_parameter_count': 2,
        'adapter_tensor_bytes': 8,
        'adapter_namespace_in_source': 'structured_head.*',
        'adapter_namespace_in_file': 'prefix-stripped',
        'tensor_schema': {
          'weight': {'shape': [2], 'dtype': 'torch.float32'}},
        'source_checkpoint_sha256': 'a' * 64,
        'source_checkpoint_size_bytes': 100,
        'source_checkpoint_global_step': 1000,
        'source_state_dict_tensor_count': (
          RELEASED_BACKBONE_IDENTITY['tensor_count'] + 1),
        'omitted_frozen_backbone_tensor_count': (
          RELEASED_BACKBONE_IDENTITY['tensor_count']),
        'ema_available': False,
        'ema_used': False,
        'required_loader': (
          'scripts.export_structured_adapter.load_adapter_into_head'),
        'required_loader_strict': True,
        'released_backbone': dict(RELEASED_BACKBONE_IDENTITY),
      }
      manifest_path.write_text(json.dumps(manifest))
      validated = _load_and_validate_adapter_origin_manifest(
        manifest_path=manifest_path,
        adapter_path=adapter_path,
        manifest_payload=manifest,
        legacy_plan=True,
        expected_adapter_sha256=sha256_file(adapter_path),
        expected_manifest_sha256=sha256_file(manifest_path),
      )
      self.assertEqual(validated, manifest)
      with self.assertRaisesRegex(ValueError, 'only for the pinned legacy'):
        _load_and_validate_adapter_origin_manifest(
          manifest_path=manifest_path,
          adapter_path=adapter_path,
          manifest_payload=manifest,
          legacy_plan=False,
          expected_adapter_sha256=sha256_file(adapter_path),
          expected_manifest_sha256=sha256_file(manifest_path),
        )

  def test_binds_exact_schema_v4_plan_export_for_generation(self):
    with tempfile.TemporaryDirectory() as directory:
      fixture = AdapterPairFixture(Path(directory))
      checkout_patch, manifest_patch = self._patches(fixture)
      evidence_path = Path(directory) / 'adapter-pair-origin.json'
      with checkout_patch, manifest_patch:
        evidence = build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite=fixture.suite,
          candidate_k=128,
          train_seed=1,
        )
        evidence_file_sha = write_adapter_pair_origin_evidence(
          evidence_path, evidence)
        arm = 'dynamic_dynamic'
        export_outputs = evidence['arms'][arm]['export']['outputs']
        manifest_path = Path(export_outputs['adapter_manifest']['path'])
        identity = json.loads(manifest_path.read_text())[
          'structured_decoder_identity']
        binding = bind_generation_arm_to_adapter_origin_evidence(
          evidence_path,
          expected_evidence_sha256=evidence_file_sha,
          arm=arm,
          adapter_path=Path(export_outputs['adapter']['path']),
          expected_adapter_sha256=export_outputs['adapter']['sha256'],
          adapter_manifest_path=manifest_path,
          expected_adapter_manifest_sha256=(
            export_outputs['adapter_manifest']['sha256']),
          structured_decoder_identity=identity,
        )

      self.assertEqual(binding['arm'], arm)
      self.assertEqual(binding['source'], evidence['source'])
      self.assertEqual(
        binding['adapter']['structured_decoder_identity'], identity)
      self.assertEqual(
        binding['adapter']['source_checkpoint_sha256'],
        evidence['arms'][arm]['train']['outputs']['checkpoint']['sha256'])
      claimed_binding_sha = binding.pop('binding_sha256')
      self.assertEqual(
        claimed_binding_sha,
        hashlib.sha256(json.dumps(
          binding, sort_keys=True, separators=(',', ':'),
          allow_nan=False).encode()).hexdigest())

  def test_generation_binding_rejects_nonidentical_manifest_reexport(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      fixture = AdapterPairFixture(root)
      checkout_patch, manifest_patch = self._patches(fixture)
      evidence_path = root / 'adapter-pair-origin.json'
      with checkout_patch, manifest_patch:
        evidence = build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite=fixture.suite,
          candidate_k=128,
          train_seed=1,
        )
        evidence_file_sha = write_adapter_pair_origin_evidence(
          evidence_path, evidence)
      arm = 'dynamic_dynamic'
      export_outputs = evidence['arms'][arm]['export']['outputs']
      source_manifest = Path(export_outputs['adapter_manifest']['path'])
      reexport_manifest = root / 'adapter-manifest.json'
      parsed = json.loads(source_manifest.read_text())
      reexport_manifest.write_text(json.dumps(parsed, indent=2) + '\n')
      identity = parsed['structured_decoder_identity']
      checkout_patch, manifest_patch = self._patches(fixture)
      with checkout_patch, manifest_patch, self.assertRaisesRegex(
          ValueError, 'not the exact plan export'):
        bind_generation_arm_to_adapter_origin_evidence(
          evidence_path,
          expected_evidence_sha256=evidence_file_sha,
          arm=arm,
          adapter_path=Path(export_outputs['adapter']['path']),
          expected_adapter_sha256=export_outputs['adapter']['sha256'],
          adapter_manifest_path=reexport_manifest,
          expected_adapter_manifest_sha256=sha256_file(reexport_manifest),
          structured_decoder_identity=identity,
        )

  def test_generation_binding_rejects_wrong_loaded_arm_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      fixture = AdapterPairFixture(root)
      checkout_patch, manifest_patch = self._patches(fixture)
      evidence_path = root / 'adapter-pair-origin.json'
      with checkout_patch, manifest_patch:
        evidence = build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite=fixture.suite,
          candidate_k=128,
          train_seed=1,
        )
        evidence_file_sha = write_adapter_pair_origin_evidence(
          evidence_path, evidence)
        arm = 'dynamic_dynamic'
        export_outputs = evidence['arms'][arm]['export']['outputs']
        manifest_path = Path(export_outputs['adapter_manifest']['path'])
        identity = json.loads(manifest_path.read_text())[
          'structured_decoder_identity']
        identity['control_identity'] = 'static_static'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
          bind_generation_arm_to_adapter_origin_evidence(
            evidence_path,
            expected_evidence_sha256=evidence_file_sha,
            arm=arm,
            adapter_path=Path(export_outputs['adapter']['path']),
            expected_adapter_sha256=export_outputs['adapter']['sha256'],
            adapter_manifest_path=manifest_path,
            expected_adapter_manifest_sha256=(
              export_outputs['adapter_manifest']['sha256']),
            structured_decoder_identity=identity,
          )

  def test_generation_binding_rejects_manifest_source_origin_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      fixture = AdapterPairFixture(root)
      checkout_patch, manifest_patch = self._patches(fixture)
      evidence_path = root / 'adapter-pair-origin.json'
      with checkout_patch, manifest_patch:
        evidence = build_adapter_pair_origin_evidence(
          fixture.plan_dir,
          suite=fixture.suite,
          candidate_k=128,
          train_seed=1,
        )
        evidence_file_sha = write_adapter_pair_origin_evidence(
          evidence_path, evidence)
      arm = 'dynamic_dynamic'
      export_outputs = evidence['arms'][arm]['export']['outputs']
      manifest_path = Path(export_outputs['adapter_manifest']['path'])
      identity = json.loads(manifest_path.read_text())[
        'structured_decoder_identity']
      calls = 0

      def drift_on_runner_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = fixture.validate_manifest(*args, **kwargs)
        if calls == 3:
          result = dict(result)
          result['source_checkpoint_sha256'] = 'f' * 64
        return result

      with mock.patch(
          'scripts.aggregate_hierarchical_document_eval.'
          '_validate_repository_checkout'), mock.patch(
            'evaluation.adapter_pair_origin.load_and_validate_adapter_manifest',
            side_effect=drift_on_runner_validation), self.assertRaisesRegex(
              ValueError, 'source_checkpoint_sha256 differs'):
        bind_generation_arm_to_adapter_origin_evidence(
          evidence_path,
          expected_evidence_sha256=evidence_file_sha,
          arm=arm,
          adapter_path=Path(export_outputs['adapter']['path']),
          expected_adapter_sha256=export_outputs['adapter']['sha256'],
          adapter_manifest_path=manifest_path,
          expected_adapter_manifest_sha256=(
            export_outputs['adapter_manifest']['sha256']),
          structured_decoder_identity=identity,
        )


if __name__ == '__main__':
  unittest.main()
