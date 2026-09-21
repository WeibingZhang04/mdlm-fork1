import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from evaluation.generation_analysis_artifacts import (
  DATASET_CONTRACTS,
  IMMUTABLE_PROTOCOL_SHA256,
  IMMUTABLE_RUNNER_GIT_SHA,
  REPO_ROOT,
  TRUSTED_CONTROLLER_SOURCE_PATHS,
  GenerationArtifactError,
  compile_reviewed_wikitext_gate,
  validate_cross_domain_post_bundle,
  validate_reviewed_wikitext_gate,
  write_reviewed_wikitext_gate,
)
from evaluation.generation_queue_artifacts import sha256_file
from scripts.run_cross_domain_generation_post import (
  POST_BUNDLE_DIRECTORY,
  POST_STAGING_DIRECTORY,
  build_post_bundle,
)
from scripts.run_cross_domain_generation_queue import (
  CROSS_DOMAIN_LAUNCH_PLAN_SHA256S,
  DATASETS,
  CrossDomainQueueController,
)
from scripts.run_wikitext_generation_queue import QueueFailure, QueuePaths


def _write_json(path: Path, payload) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _canonical_sha256(payload) -> str:
  return hashlib.sha256(json.dumps(
    payload, sort_keys=True, separators=(',', ':'),
    ensure_ascii=False).encode()).hexdigest()


def _controller_repository(root: Path) -> Path:
  root.mkdir(parents=True, exist_ok=True)
  for relative_path in TRUSTED_CONTROLLER_SOURCE_PATHS:
    source = REPO_ROOT / relative_path
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
  for command in (
      ['git', 'init', '-q'],
      ['git', 'config', 'user.email', 'fixture@example.invalid'],
      ['git', 'config', 'user.name', 'Fixture'],
      ['git', 'add', '.'],
      ['git', 'commit', '-q', '-m', 'trusted controller fixture']):
    subprocess.run(command, cwd=root, check=True, capture_output=True)
  return root


def _write_raw_shard(
    shard_dir: Path, contract, arm: str, shard_index: int, *, command=None,
):
  shard_dir.mkdir(parents=True, exist_ok=True)
  modes = (
    ['factorized', 'structured_marginal', 'structured_joint']
    if arm == 'dynamic_dynamic' else ['structured_joint'])
  shard_draws = (
    (contract.global_num_samples - 1 - shard_index) // 16 + 1)
  num_records = shard_draws * len(modes) * 4
  samples_path = shard_dir / 'samples.jsonl'
  samples_path.write_text('{}\n' * num_records)
  manifest = {
    'command': command or ['synthetic-fixture'],
    'repository': {
      'git_sha': IMMUTABLE_RUNNER_GIT_SHA,
      'dirty': False,
    },
    'pairing': {
      'shard_index': shard_index,
      'num_shards': 16,
      'global_num_samples': contract.global_num_samples,
      'base_seed': contract.base_seed,
      'batch_size': 8,
      'sequence_length': 256,
      'global_pairing_digest': 'a' * 64,
    },
    'matrix': {
      'sampling_modes': modes,
      'nfe_budgets': [8, 16, 32, 64],
      'num_output_records': num_records,
    },
    'prompts': {
      'bundle_identity': {
        'data_config': {
          'name': contract.data_config,
          'logical_validation_dataset': contract.logical_dataset,
        },
        'output': {'num_prompts': contract.num_prompts},
      },
    },
    'artifacts': {
      'structured_adapter': {
        'semantic_identity': {'control_identity': arm},
      },
    },
    'outputs': {
      'samples_jsonl': {
        'path': 'samples.jsonl',
        'sha256': sha256_file(samples_path),
        'num_records': num_records,
      },
    },
  }
  _write_json(shard_dir / 'manifest.json', manifest)
  if command is not None:
    (shard_dir / 'summary.json').write_text('{}\n')
    (shard_dir / 'resolved_config.yaml').write_text('{}\n')
  return {
    'shard_index': shard_index,
    'manifest_path': str((shard_dir / 'manifest.json').resolve()),
    'manifest_sha256': sha256_file(shard_dir / 'manifest.json'),
    'samples_sha256': sha256_file(samples_path),
    'num_records': num_records,
  }


def _input_shards(contract, arm, shard_dirs, *, commands=None):
  return [
    _write_raw_shard(
      shard_dir, contract, arm, index,
      command=None if commands is None else commands[index])
    for index, shard_dir in enumerate(shard_dirs)
  ]


def _union(contract, arm, root: Path, *, shard_dirs=None, commands=None):
  modes = (
    ['factorized', 'structured_marginal', 'structured_joint']
    if arm == 'dynamic_dynamic' else ['structured_joint'])
  records = contract.global_num_samples * len(modes) * 4
  coverage = {
    'num_shards': 16,
    'shard_indices': list(range(16)),
    'global_num_paired_draws': contract.global_num_samples,
    'num_unique_prompts': contract.num_prompts,
    'paired_draws_per_prompt': {
      f'prompt-{index:04d}': 4 for index in range(contract.num_prompts)},
    'num_sampling_modes': len(modes),
    'num_nfe_budgets': 4,
    'num_groups': len(modes) * 4,
    'expected_output_records': records,
    'verified_output_records': records,
    'global_pairing_digest': 'a' * 64,
    'record_digest_algorithm': 'sha256-canonical-json-array-v1',
    'canonical_union_records_sha256': 'b' * 64,
  }
  return {
    'schema_version': 1,
    'artifact': 'verified_generation_shard_union',
    'experiment': 'paired_contextual_forest_generation_pilot',
    'created_utc': '2026-08-31T00:00:00+00:00',
    'scope_note': 'fixture',
    'identity': {
      'repository': {
        'git_sha': IMMUTABLE_RUNNER_GIT_SHA, 'dirty': False},
      'artifacts': {
        'structured_adapter': {
          'semantic_identity': {'control_identity': arm}}},
      'prompts': {
        'bundle_identity': {
          'data_config': {
            'name': contract.data_config,
            'logical_validation_dataset': contract.logical_dataset,
          },
          'output': {'num_prompts': contract.num_prompts},
        }},
      'base_seed': contract.base_seed,
      'global_num_samples': contract.global_num_samples,
      'num_shards': 16,
      'batch_size': 8,
      'sequence_length': 256,
      'sampling_modes': modes,
      'nfe_budgets': [8, 16, 32, 64],
      'global_pairing_digest': 'a' * 64,
    },
    'coverage': coverage,
    'input_shards': _input_shards(
      contract, arm,
      shard_dirs or [
        root / arm / f'shard-{index:02d}' for index in range(16)],
      commands=commands),
    'groups': [],
    'comparisons': [],
    'bootstrap': {},
    'timing_policy': {},
  }


def _comparison(contract, dynamic, static):
  kinds = (
    'structured_marginals_vs_factorized_backbone_at_fixed_nfe',
    'joint_vs_independent_structured_marginals_at_fixed_nfe',
    'dynamic_joint_vs_factorized_backbone_at_fixed_nfe',
    'dynamic_adapter_vs_static_adapter_at_fixed_nfe',
  )
  rows = []
  for budget in (8, 16, 32, 64):
    for kind in kinds:
      rows.append({
        'comparison_kind': kind,
        'baseline': {'requested_nfe_budget': budget},
        'treatment': {'requested_nfe_budget': budget},
        'num_paired_draws': contract.global_num_samples,
        'num_prompt_clusters': contract.num_prompts,
        'endpoints': {
          'reference_token_accuracy': {
            'paired_draws': {
              'num_paired_draws': contract.global_num_samples},
            'prompt_clusters': {
              'num_paired_draws': contract.global_num_samples,
              'num_prompt_clusters': contract.num_prompts,
            },
          },
        },
      })
  return {
    'schema_version': 1,
    'artifact': 'paired_generation_adapter_comparison',
    'protocol_id': 'contextual-forest-generation-paper-v1',
    'created_utc': '2026-08-31T00:00:00+00:00',
    'dataset_id': contract.logical_dataset,
    'scientific_scope': 'fixture',
    'identity': {
      'repository': {
        'git_sha': IMMUTABLE_RUNNER_GIT_SHA, 'dirty': False},
      'generation_protocol': {
        'protocol_id': 'contextual-forest-generation-paper-v1',
        'protocol_sha256': IMMUTABLE_PROTOCOL_SHA256,
      },
      'nfe_budgets': [8, 16, 32, 64],
    },
    'adapters': {},
    'adapter_origins': {},
    'verified_unions': {
      'baseline_static_static': {
        'canonical_sha256': _canonical_sha256(static),
        'coverage': static['coverage'],
        'input_shards': static['input_shards'],
      },
      'treatment_dynamic_dynamic': {
        'canonical_sha256': _canonical_sha256(dynamic),
        'coverage': dynamic['coverage'],
        'input_shards': dynamic['input_shards'],
      },
    },
    'comparisons': rows,
    'timing': [],
    'endpoint_direction': {},
    'primary_causal_comparison': (
      'joint_vs_independent_structured_marginals_at_fixed_nfe'),
    'bootstrap': {
      'num_resamples': 20_000,
      'base_rng_seed': 94_001,
      'confidence_level': 0.95,
      'paired_draw_and_prompt_cluster_intervals': True,
    },
  }


def _triplet(root: Path, slug: str):
  contract = DATASET_CONTRACTS[slug]
  dynamic = _union(contract, 'dynamic_dynamic', root / 'raw')
  static = _union(contract, 'static_static', root / 'raw')
  comparison = _comparison(contract, dynamic, static)
  paths = {
    'dynamic': root / f'{slug}-dynamic.json',
    'static': root / f'{slug}-static.json',
    'comparison': root / f'{slug}-comparison.json',
  }
  _write_json(paths['dynamic'], dynamic)
  _write_json(paths['static'], static)
  _write_json(paths['comparison'], comparison)
  return paths, dynamic, static, comparison


def _paths(root: Path) -> QueuePaths:
  return QueuePaths(
    runner_repo=root / 'runner',
    python=root / 'venv' / 'bin' / 'python',
    experiment_root=root / 'generation',
    expansion_root=root / 'expansion',
    backbone=root / 'backbone.pt')


class ReviewedGateTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.controller_repo = _controller_repository(
      self.root / 'trusted-controller')
    paths, _, _, _ = _triplet(self.root, 'wikitext')
    self.gate_payload = compile_reviewed_wikitext_gate(
      paths['dynamic'], paths['static'], paths['comparison'],
      decision='proceed', review_statement='Reviewed prompt-cluster results.',
      controller_repo_root=self.controller_repo)
    self.gate = self.root / 'gate.json'
    self.gate_sha = write_reviewed_wikitext_gate(
      self.gate, self.gate_payload)

  def tearDown(self):
    self.temporary.cleanup()

  def test_compiles_and_replays_hash_bound_proceed_gate(self):
    result = validate_reviewed_wikitext_gate(
      self.gate, expected_sha256=self.gate_sha,
      controller_repo_root=self.controller_repo)
    self.assertEqual(result['decision'], 'proceed')
    self.assertEqual(result['validated_analysis']['num_prompts'], 197)

  def test_hold_gate_and_wrong_gate_hash_fail_closed(self):
    hold = dict(self.gate_payload)
    hold['decision'] = 'hold'
    hold_path = self.root / 'hold.json'
    hold_sha = write_reviewed_wikitext_gate(hold_path, hold)
    with self.assertRaisesRegex(GenerationArtifactError, 'does not authorize'):
      validate_reviewed_wikitext_gate(
        hold_path, expected_sha256=hold_sha,
        controller_repo_root=self.controller_repo)
    with self.assertRaisesRegex(GenerationArtifactError, 'SHA256 mismatch'):
      validate_reviewed_wikitext_gate(
        self.gate, expected_sha256='0' * 64,
        controller_repo_root=self.controller_repo)

  def test_referenced_output_drift_and_semantic_tampering_fail(self):
    comparison_path = Path(
      self.gate_payload['artifacts']['paired_comparison']['path'])
    comparison = json.loads(comparison_path.read_text())
    comparison['comparisons'][0]['num_prompt_clusters'] = 196
    _write_json(comparison_path, comparison)
    with self.assertRaisesRegex(GenerationArtifactError, 'SHA256 mismatch'):
      validate_reviewed_wikitext_gate(
        self.gate, expected_sha256=self.gate_sha,
        controller_repo_root=self.controller_repo)

    gate = json.loads(self.gate.read_text())
    gate['artifacts']['paired_comparison']['sha256'] = sha256_file(
      comparison_path)
    semantic_gate = self.root / 'semantic-gate.json'
    semantic_sha = write_reviewed_wikitext_gate(semantic_gate, gate)
    with self.assertRaisesRegex(GenerationArtifactError, 'paired coverage'):
      validate_reviewed_wikitext_gate(
        semantic_gate, expected_sha256=semantic_sha,
        controller_repo_root=self.controller_repo)

  def test_cross_domain_outputs_cannot_authorize_the_wikitext_gate(self):
    paths, _, _, _ = _triplet(self.root / 'wrong-domain', 'arxiv')
    with self.assertRaisesRegex(GenerationArtifactError, 'wrong prompt dataset'):
      compile_reviewed_wikitext_gate(
        paths['dynamic'], paths['static'], paths['comparison'],
        decision='proceed', review_statement='Wrong dataset.',
        controller_repo_root=self.controller_repo)

  def test_dirty_or_wrong_head_controller_checkout_fails_gate_replay(self):
    trusted_file = self.controller_repo / TRUSTED_CONTROLLER_SOURCE_PATHS[0]
    trusted_file.write_text(trusted_file.read_text() + '\n# dirty\n')
    with self.assertRaisesRegex(
        GenerationArtifactError, 'must be exactly clean'):
      validate_reviewed_wikitext_gate(
        self.gate, expected_sha256=self.gate_sha,
        controller_repo_root=self.controller_repo)

    trusted_file.write_bytes(
      (REPO_ROOT / TRUSTED_CONTROLLER_SOURCE_PATHS[0]).read_bytes())
    (self.controller_repo / 'README.fixture').write_text('new clean commit\n')
    subprocess.run(
      ['git', 'add', '.'], cwd=self.controller_repo, check=True,
      capture_output=True)
    subprocess.run(
      ['git', 'commit', '-q', '-m', 'wrong head'],
      cwd=self.controller_repo, check=True, capture_output=True)
    with self.assertRaisesRegex(
        GenerationArtifactError, 'HEAD or trusted source bytes differ'):
      validate_reviewed_wikitext_gate(
        self.gate, expected_sha256=self.gate_sha,
        controller_repo_root=self.controller_repo)

  def test_raw_manifest_or_samples_drift_fails_gate_replay(self):
    dynamic = Path(self.gate_payload['artifacts']['dynamic_union']['path'])
    union = json.loads(dynamic.read_text())
    first_manifest = Path(union['input_shards'][0]['manifest_path'])
    samples = first_manifest.parent / 'samples.jsonl'
    samples.write_text(samples.read_text() + '{}\n')
    with self.assertRaisesRegex(
        GenerationArtifactError, 'samples bytes/count differ'):
      validate_reviewed_wikitext_gate(
        self.gate, expected_sha256=self.gate_sha,
        controller_repo_root=self.controller_repo)


class CrossDomainPostBundleTest(unittest.TestCase):

  def _controller(self, root: Path, slug: str, gate_identity):
    paths = _paths(root)
    dataset = DATASETS[slug]
    controller = CrossDomainQueueController(
      dataset,
      paths,
      reviewed_gate_sha256=gate_identity['sha256'],
      reviewed_gate_path=Path(gate_identity['path']),
      controller_repo_root=self.controller_repo,
      completion_validator=lambda task, queue_paths: None)
    controller.reviewed_gate_identity = gate_identity
    controller.launch_authorization = {
      'artifact': 'reviewed_wikitext_cross_domain_generation_gate',
      'path': gate_identity['path'],
      'sha256': gate_identity['sha256'],
      'decision': 'proceed',
      'controller_repository': gate_identity['controller_repository'],
    }
    controller.verify_environment = lambda: None
    for phase in controller.plan.phases:
      for task in phase:
        _write_raw_shard(
          task.output_dir,
          DATASET_CONTRACTS[slug],
          task.arm.name,
          task.shard_index,
          command=[task.command[1], *task.command[2:]])
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.log_path.write_text('synthetic completed task\n')
    tasks = [task for phase in controller.plan.phases for task in phase]
    completion = {
      'schema_version': 1,
      'artifact': 'frozen_generation_queue_completion',
      'dataset_slug': slug,
      'logical_dataset': dataset.logical_dataset,
      'immutable_runner_git_sha': IMMUTABLE_RUNNER_GIT_SHA,
      'launch_plan_sha256': CROSS_DOMAIN_LAUNCH_PLAN_SHA256S[slug],
      'launch_authorization': {
        'artifact': 'reviewed_wikitext_cross_domain_generation_gate',
        'path': gate_identity['path'],
        'sha256': gate_identity['sha256'],
        'decision': 'proceed',
        'controller_repository': gate_identity['controller_repository'],
      },
      'num_tasks': len(tasks),
      'tasks': [
        {
          'task_id': task.task_id,
          'dataset_slug': slug,
          'arm': task.arm.name,
          'shard_index': task.shard_index,
          'output_dir': str(task.output_dir.resolve()),
          'manifest_sha256': sha256_file(
            task.output_dir / 'manifest.json'),
        }
        for task in tasks
      ],
      'completed_utc': '2026-08-31T00:00:00+00:00',
    }
    _write_json(
      paths.experiment_root / slug / 'queue-complete.json', completion)
    return controller

  def _analysis_payloads(self, controller, slug):
    contract = DATASET_CONTRACTS[slug]
    dynamic_tasks = controller.plan.phases[0]
    static_tasks = controller.plan.phases[1]
    dynamic = _union(
      contract,
      'dynamic_dynamic',
      controller.plan.paths.experiment_root / slug,
      shard_dirs=[task.output_dir for task in dynamic_tasks],
      commands=[[task.command[1], *task.command[2:]] for task in dynamic_tasks])
    static = _union(
      contract,
      'static_static',
      controller.plan.paths.experiment_root / slug,
      shard_dirs=[task.output_dir for task in static_tasks],
      commands=[[task.command[1], *task.command[2:]] for task in static_tasks])
    return dynamic, static, _comparison(contract, dynamic, static)

  def _gate(self, root: Path):
    self.controller_repo = _controller_repository(root / 'controller')
    paths, _, _, _ = _triplet(root, 'wikitext')
    payload = compile_reviewed_wikitext_gate(
      paths['dynamic'], paths['static'], paths['comparison'],
      decision='proceed', review_statement='Reviewed.',
      controller_repo_root=self.controller_repo)
    gate_path = root / 'gate.json'
    gate_sha = write_reviewed_wikitext_gate(gate_path, payload)
    return validate_reviewed_wikitext_gate(
      gate_path, expected_sha256=gate_sha,
      controller_repo_root=self.controller_repo)

  def test_end_to_end_atomic_bundle_for_both_cross_domains(self):
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      gate = self._gate(base / 'gate-inputs')
      for slug in ('arxiv', 'pubmed'):
        root = base / slug
        controller = self._controller(root, slug, gate)
        dynamic, static, _ = self._analysis_payloads(
          controller, slug)

        def aggregate(_shards, *, baseline_mode, **_kwargs):
          return dynamic if baseline_mode == 'factorized' else static

        comparison_calls = []

        def compare(
            _baseline_shards, _treatment_shards, *,
            baseline_union, treatment_union, **_kwargs):
          self.assertIs(baseline_union, static)
          self.assertIs(treatment_union, dynamic)
          comparison_calls.append((baseline_union, treatment_union))
          return _comparison(
            DATASET_CONTRACTS[slug], treatment_union, baseline_union)

        bundle_path, bundle_sha = build_post_bundle(
          controller,
          aggregate_fn=aggregate,
          compare_fn=compare)
        self.assertEqual(len(comparison_calls), 1)
        validated = validate_cross_domain_post_bundle(
          bundle_path, expected_sha256=bundle_sha,
          controller_repo_root=self.controller_repo)
        self.assertEqual(validated['dataset_slug'], slug)
        self.assertEqual(
          validated['validated_analysis']['dynamic_records'], 12_288)
        self.assertFalse(
          (controller.plan.paths.experiment_root / slug
           / POST_STAGING_DIRECTORY).exists())
        self.assertTrue(
          (controller.plan.paths.experiment_root / slug
           / POST_BUNDLE_DIRECTORY).is_dir())
        with self.assertRaisesRegex(QueueFailure, 'refusing to overwrite'):
          build_post_bundle(
            controller, aggregate_fn=aggregate, compare_fn=compare)

  def test_missing_completion_marker_prevents_aggregation_or_staging(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      gate = self._gate(root / 'gate-inputs')
      controller = self._controller(root, 'arxiv', gate)
      marker = (
        controller.plan.paths.experiment_root / 'arxiv'
        / 'queue-complete.json')
      marker.unlink()
      calls = []
      with self.assertRaisesRegex(QueueFailure, 'completion evidence is missing'):
        build_post_bundle(
          controller,
          aggregate_fn=lambda *_args, **_kwargs: calls.append('aggregate'),
          compare_fn=lambda *_args, **_kwargs: calls.append('compare'))
      self.assertEqual(calls, [])
      self.assertFalse(
        (controller.plan.paths.experiment_root / 'arxiv'
         / POST_STAGING_DIRECTORY).exists())

  def test_completion_evidence_or_manifest_tampering_prevents_aggregation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      gate = self._gate(root / 'gate-inputs')
      controller = self._controller(root, 'arxiv', gate)
      first_manifest = controller.plan.phases[0][0].output_dir / 'manifest.json'
      first_manifest.write_text(first_manifest.read_text() + ' ')
      calls = []
      with self.assertRaisesRegex(
          QueueFailure, 'differs from current manifest bytes'):
        build_post_bundle(
          controller,
          aggregate_fn=lambda *_args, **_kwargs: calls.append('aggregate'),
          compare_fn=lambda *_args, **_kwargs: calls.append('compare'))
      self.assertEqual(calls, [])

  def test_output_hash_drift_is_detected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      gate = self._gate(root / 'gate-inputs')
      controller = self._controller(root, 'arxiv', gate)
      dynamic, static, comparison = self._analysis_payloads(
        controller, 'arxiv')
      bundle_path, _ = build_post_bundle(
        controller,
        aggregate_fn=lambda _shards, *, baseline_mode, **_kwargs: (
          dynamic if baseline_mode == 'factorized' else static),
        compare_fn=lambda *_args, **_kwargs: comparison)
      dynamic_path = bundle_path.parent / 'verified-dynamic-union.json'
      dynamic_path.write_text(dynamic_path.read_text() + ' ')
      with self.assertRaisesRegex(GenerationArtifactError, 'SHA256 mismatch'):
        validate_cross_domain_post_bundle(
          bundle_path, controller_repo_root=self.controller_repo)

  def test_rehashed_nonprojected_union_tampering_fails_canonical_binding(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      gate = self._gate(root / 'gate-inputs')
      controller = self._controller(root, 'arxiv', gate)
      dynamic, static, comparison = self._analysis_payloads(
        controller, 'arxiv')
      bundle_path, _ = build_post_bundle(
        controller,
        aggregate_fn=lambda _shards, *, baseline_mode, **_kwargs: (
          dynamic if baseline_mode == 'factorized' else static),
        compare_fn=lambda *_args, **_kwargs: comparison)

      static_path = bundle_path.parent / 'verified-static-union.json'
      tampered_static = json.loads(static_path.read_text())
      tampered_static['scope_note'] = 'rehashed non-projected tampering'
      _write_json(static_path, tampered_static)
      bundle = json.loads(bundle_path.read_text())
      bundle['artifacts']['static_union']['sha256'] = sha256_file(static_path)
      _write_json(bundle_path, bundle)

      with self.assertRaisesRegex(
          GenerationArtifactError,
          'canonical hash differs from baseline_static_static union'):
        validate_cross_domain_post_bundle(
          bundle_path, controller_repo_root=self.controller_repo)

  def test_rehashed_outer_bundle_cannot_hide_completion_evidence_tampering(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      gate = self._gate(root / 'gate-inputs')
      controller = self._controller(root, 'arxiv', gate)
      dynamic, static, comparison = self._analysis_payloads(
        controller, 'arxiv')
      bundle_path, _ = build_post_bundle(
        controller,
        aggregate_fn=lambda _shards, *, baseline_mode, **_kwargs: (
          dynamic if baseline_mode == 'factorized' else static),
        compare_fn=lambda *_args, **_kwargs: comparison)
      marker_path = (
        controller.plan.paths.experiment_root / 'arxiv'
        / 'queue-complete.json')
      marker = json.loads(marker_path.read_text())
      marker['tasks'][0]['arm'] = 'static_static'
      _write_json(marker_path, marker)
      bundle = json.loads(bundle_path.read_text())
      bundle['queue_completion_evidence']['sha256'] = sha256_file(marker_path)
      _write_json(bundle_path, bundle)
      with self.assertRaisesRegex(
          GenerationArtifactError, 'differs from the frozen grid'):
        validate_cross_domain_post_bundle(
          bundle_path, controller_repo_root=self.controller_repo)

  def test_stale_staging_requires_explicit_preserving_recovery(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      gate = self._gate(root / 'gate-inputs')
      controller = self._controller(root, 'arxiv', gate)
      staging = (
        controller.plan.paths.experiment_root / 'arxiv'
        / POST_STAGING_DIRECTORY)
      staging.mkdir(parents=True)
      (staging / 'partial.txt').write_text('preserve')
      dynamic, static, comparison = self._analysis_payloads(
        controller, 'arxiv')

      def aggregate(_shards, *, baseline_mode, **_kwargs):
        return dynamic if baseline_mode == 'factorized' else static

      with self.assertRaisesRegex(QueueFailure, 'staging directory is preserved'):
        build_post_bundle(
          controller, aggregate_fn=aggregate,
          compare_fn=lambda *_args, **_kwargs: comparison)
      bundle_path, _ = build_post_bundle(
        controller,
        recover_stale_staging=True,
        aggregate_fn=aggregate,
        compare_fn=lambda *_args, **_kwargs: comparison)
      self.assertTrue(bundle_path.is_file())
      preserved = list(staging.parent.glob(
        f'{POST_STAGING_DIRECTORY}.preserved-*'))
      self.assertEqual(len(preserved), 1)
      self.assertEqual((preserved[0] / 'partial.txt').read_text(), 'preserve')


if __name__ == '__main__':
  unittest.main()
