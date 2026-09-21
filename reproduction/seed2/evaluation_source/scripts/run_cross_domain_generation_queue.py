#!/usr/bin/env python3
"""Run one frozen two-worker arXiv or PubMed generation shard queue.

This is the cross-domain continuation of the paper-scale WikiText protocol.
It changes only the pinned prompt bundle, dataset configuration, sample count,
and base seed.  The runner revision, adapters, scorer, sampling modes, NFE
budgets, and all other scientific settings remain fixed.  Existing incomplete
artifacts are preserved and cause a fail-closed stop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

_wiki_queue = importlib.import_module(
  'scripts.run_wikitext_generation_queue'
  if __package__ else 'run_wikitext_generation_queue')

from evaluation.generation_analysis_artifacts import (  # noqa: E402
  CROSS_DOMAIN_LAUNCH_PLAN_SHA256S,
  REPO_ROOT,
  reviewed_gate_launch_authorization,
  validate_reviewed_wikitext_gate,
)

ADAPTER_ORIGIN_EVIDENCE_SHA256 = (
  _wiki_queue.ADAPTER_ORIGIN_EVIDENCE_SHA256)
BACKBONE_SHA256 = _wiki_queue.BACKBONE_SHA256
BATCH_SIZE = _wiki_queue.BATCH_SIZE
DEFAULT_PATHS = _wiki_queue.DEFAULT_PATHS
IMMUTABLE_PROTOCOL_SHA256 = _wiki_queue.IMMUTABLE_PROTOCOL_SHA256
IMMUTABLE_RUNNER_GIT_SHA = _wiki_queue.IMMUTABLE_RUNNER_GIT_SHA
IMMUTABLE_RUNNER_SHA256 = _wiki_queue.IMMUTABLE_RUNNER_SHA256
NFE_BUDGETS = _wiki_queue.NFE_BUDGETS
NUM_SHARDS = _wiki_queue.NUM_SHARDS
SEQUENCE_LENGTH = _wiki_queue.SEQUENCE_LENGTH
QueueController = _wiki_queue.QueueController
QueueFailure = _wiki_queue.QueueFailure
QueuePaths = _wiki_queue.QueuePaths
QueuePlan = _wiki_queue.QueuePlan
ShardTask = _wiki_queue.ShardTask
_append_log = _wiki_queue._append_log
_arm_specs = _wiki_queue._arm_specs
_normalize_runner_arguments = _wiki_queue._normalize_runner_arguments
_require_exact_launch_plan = _wiki_queue._require_exact_launch_plan
_sha256_file = _wiki_queue._sha256_file
launch_plan_sha256 = _wiki_queue.launch_plan_sha256

@dataclass(frozen=True)
class DatasetSpec:
  slug: str
  data_config: str
  logical_dataset: str
  data_config_sha256: str
  prompt_name: str
  prompt_sha256: str
  prompt_manifest_sha256: str
  prompt_provenance_relative_path: str
  prompt_provenance_sha256: str
  num_prompts: int
  global_num_samples: int
  base_seed: int


DATASETS = {
  'arxiv': DatasetSpec(
    slug='arxiv',
    data_config='eval_scientific_papers_arxiv_pinned',
    logical_dataset='scientific-papers-arxiv-pinned',
    data_config_sha256=(
      '651bc0b9be186244abdb6a0547b2ccb507fe84b28807fe9914f047de85f69046'),
    prompt_name='arxiv-span32.jsonl',
    prompt_sha256=(
      'ab3076602b0bc1988a8bac76e01f34a9662a579669f46692b18f8b1b760ee262'),
    prompt_manifest_sha256=(
      'a449a7006ba5da286c8b36d0fa539a535d73160ad585c048f025a0f4186182c1'),
    prompt_provenance_relative_path=(
      'arxiv-span32.jsonl.provenance/valid-4c660805cca2b31bff8b.json'),
    prompt_provenance_sha256=(
      'c5f59ff25c32a0ca9e290d9f7d22a385d1c2a06a2542a46dc1594d1c56ef08a6'),
    num_prompts=256,
    global_num_samples=1024,
    base_seed=92001,
  ),
  'pubmed': DatasetSpec(
    slug='pubmed',
    data_config='eval_scientific_papers_pubmed_pinned',
    logical_dataset='scientific-papers-pubmed-pinned',
    data_config_sha256=(
      'e006d73ab8555e8bfebbb83b6bce912110c77adf944b4c1029b219427f4cfbe8'),
    prompt_name='pubmed-span32.jsonl',
    prompt_sha256=(
      '0b34c431b05cd45d7867bf67e4658c6e5fcb3a71575f0e77629af3a3003efa4e'),
    prompt_manifest_sha256=(
      '2d01f012fbc994b2643c1aec4f905fcca3b5805524ae5ffb4e8b675ff2ffc9aa'),
    prompt_provenance_relative_path=(
      'pubmed-span32.jsonl.provenance/valid-d5c64dc21006e0989c86.json'),
    prompt_provenance_sha256=(
      '0315c9b80762732677967877c960dca11fbe6636cd5dfad81a92cc475b70f499'),
    num_prompts=256,
    global_num_samples=1024,
    base_seed=93001,
  ),
}


def _task_command(
    paths: QueuePaths,
    dataset: DatasetSpec,
    arm,
    shard_index: int,
    output_dir: Path,
) -> tuple[str, ...]:
  root = paths.experiment_root
  prompt_path = root / 'prompts' / dataset.prompt_name
  return (
    str(paths.python),
    'scripts/run_generation_pilot.py',
    '--backbone-checkpoint', str(paths.backbone),
    '--backbone-sha256', BACKBONE_SHA256,
    '--adapter', str(arm.adapter),
    '--adapter-sha256', arm.adapter_sha256,
    '--adapter-manifest', str(arm.adapter_manifest),
    '--adapter-manifest-sha256', arm.adapter_manifest_sha256,
    '--adapter-origin-evidence', str(root / 'adapter-pair-origin.json'),
    '--adapter-origin-evidence-sha256', ADAPTER_ORIGIN_EVIDENCE_SHA256,
    '--adapter-origin-arm', arm.name,
    '--output-dir', str(output_dir),
    '--prompt-jsonl', str(prompt_path),
    '--prompt-manifest', str(Path(f'{prompt_path}.manifest.json')),
    '--prompt-manifest-sha256', dataset.prompt_manifest_sha256,
    '--num-samples', str(dataset.global_num_samples),
    '--num-shards', str(NUM_SHARDS),
    '--shard-index', str(shard_index),
    '--sequence-length', str(SEQUENCE_LENGTH),
    '--batch-size', str(BATCH_SIZE),
    '--base-seed', str(dataset.base_seed),
    '--modes', *arm.modes,
    '--nfe-budgets', *(str(value) for value in NFE_BUDGETS),
    '--device', 'cuda',
    '--model-config', 'contextual-forest-small',
    '--data-config', dataset.data_config,
    '--reference-lm', 'gpt2-large',
    '--reference-lm-revision',
    '32b71b12589c2f8d625668d2335a01cac3249519',
    '--reference-lm-device', 'cuda',
    '--reference-lm-batch-size', '8',
    '--reference-lm-max-length', '256',
    '--reference-lm-dtype', 'float32',
    '--override', 'model.structured_decoder.top_k=128',
    '--override', 'trainer.devices=1',
    '--override', 'loader.num_workers=8',
    '--override', (
      'checkpointing.save_dir='
      '/mnt/contextual-forest/mdlm-generation-paper-v1'),
    '--override',
    f'model.structured_decoder.topology_mode={arm.topology_mode}',
    '--override',
    f'model.structured_decoder.factor_mode={arm.factor_mode}',
    '--override', (
      'model.structured_decoder.training.topology_weight='
      f'{arm.topology_weight}'),
  )


def _make_task(
    paths: QueuePaths,
    dataset: DatasetSpec,
    arm,
    shard_index: int,
) -> ShardTask:
  output_dir = (
    paths.experiment_root / dataset.slug / arm.name
    / f'shard-{shard_index:02d}')
  arm_log_name = 'dynamic' if arm.name == 'dynamic_dynamic' else 'static'
  log_path = (
    paths.experiment_root / 'logs'
    / f'{dataset.slug}-{arm_log_name}-shard{shard_index:02d}.log')
  return ShardTask(
    arm=arm,
    shard_index=shard_index,
    output_dir=output_dir,
    log_path=log_path,
    command=_task_command(paths, dataset, arm, shard_index, output_dir),
    dataset_slug=dataset.slug,
  )


def frozen_cross_domain_plan(
    dataset_slug: str,
    paths: QueuePaths = DEFAULT_PATHS,
) -> QueuePlan:
  try:
    dataset = DATASETS[dataset_slug]
  except KeyError as error:
    raise QueueFailure(f'unsupported cross-domain dataset {dataset_slug!r}') \
      from error
  arms = _arm_specs(paths)
  phases = tuple(
    tuple(
      _make_task(paths, dataset, arms[arm_name], shard_index)
      for shard_index in range(NUM_SHARDS))
    for arm_name in ('dynamic_dynamic', 'static_static')
  )
  return QueuePlan(paths=paths, initial_tasks=(), phases=phases)


_VALIDATION_PROGRAM = r'''
import json
from pathlib import Path
import sys

from evaluation.generation_protocol import validate_generation_protocol
from evaluation.generation_shard_aggregation import load_generation_shard

path = Path(sys.argv[1]).resolve()
expected_arm = sys.argv[2]
expected_index = int(sys.argv[3])
expected_repo = sys.argv[4]
expected_adapter = str(Path(sys.argv[5]).resolve())
expected_adapter_sha = sys.argv[6]
expected_manifest = str(Path(sys.argv[7]).resolve())
expected_manifest_sha = sys.argv[8]
expected_origin_sha = sys.argv[9]
expected_dataset_config = sys.argv[10]
expected_logical_dataset = sys.argv[11]
expected_prompt_sha = sys.argv[12]
expected_prompt_manifest_sha = sys.argv[13]
expected_num_prompts = int(sys.argv[14])
expected_num_samples = int(sys.argv[15])
expected_base_seed = int(sys.argv[16])

shard = load_generation_shard(path)
manifest = shard['manifest']
validate_generation_protocol(
    shard['config_path'], manifest, candidate_top_k=128,
    expected_control=expected_arm)
repository = manifest['repository']
if repository['git_sha'] != expected_repo or repository['dirty'] is not False:
  raise ValueError('shard repository is not the immutable clean runner checkout')
pairing = manifest['pairing']
expected_pairing = {
  'shard_index': expected_index,
  'num_shards': 16,
  'global_num_samples': expected_num_samples,
  'base_seed': expected_base_seed,
  'batch_size': 8,
  'sequence_length': 256,
}
for field, expected in expected_pairing.items():
  if pairing.get(field) != expected or type(pairing.get(field)) is not type(expected):
    raise ValueError(f'shard pairing {field} differs from the frozen queue')
adapter = manifest['artifacts']['structured_adapter']
expected_adapter_fields = {
  'path': expected_adapter,
  'sha256': expected_adapter_sha,
  'manifest_path': expected_manifest,
  'manifest_sha256': expected_manifest_sha,
}
for field, expected in expected_adapter_fields.items():
  if adapter.get(field) != expected:
    raise ValueError(f'shard adapter {field} differs from the frozen queue')
origin_file = manifest['adapter_origin_evidence']['evidence_file']
if origin_file.get('sha256') != expected_origin_sha:
  raise ValueError('shard adapter-origin file differs from the frozen queue')
prompts = manifest['prompts']
expected_prompt_fields = {
  'sha256': expected_prompt_sha,
  'manifest_sha256': expected_prompt_manifest_sha,
  'num_prompt_records': expected_num_prompts,
}
for field, expected in expected_prompt_fields.items():
  if prompts.get(field) != expected:
    raise ValueError(f'shard prompt {field} differs from the frozen queue')
bundle = prompts['bundle_identity']
if bundle['data_config']['name'] != expected_dataset_config:
  raise ValueError('prompt data config differs from the frozen queue')
if bundle['data_config']['logical_validation_dataset'] != expected_logical_dataset:
  raise ValueError('prompt logical dataset differs from the frozen queue')
if bundle['output']['sha256'] != expected_prompt_sha:
  raise ValueError('prompt bundle output hash differs from the frozen queue')
expected_modes = (
  ['factorized', 'structured_marginal', 'structured_joint']
  if expected_arm == 'dynamic_dynamic' else ['structured_joint'])
matrix = manifest['matrix']
if matrix.get('sampling_modes') != expected_modes:
  raise ValueError('shard modes differ from the frozen queue')
if matrix.get('nfe_budgets') != [8, 16, 32, 64]:
  raise ValueError('shard NFE budgets differ from the frozen queue')
expected_shard_samples = (expected_num_samples - 1 - expected_index) // 16 + 1
expected_records = expected_shard_samples * len(expected_modes) * 4
if matrix.get('num_output_records') != expected_records:
  raise ValueError('shard record count differs from complete modulo coverage')
print(json.dumps({
  'manifest_sha256': shard['manifest_sha256'],
  'num_output_records': matrix['num_output_records'],
  'shard_index': expected_index,
  'arm': expected_arm,
  'dataset': expected_logical_dataset,
}, sort_keys=True))
'''


def _completion_validator(
    task: ShardTask,
    paths: QueuePaths,
    dataset: DatasetSpec,
) -> None:
  environment = dict(os.environ)
  environment.pop('PYTHONPATH', None)
  result = subprocess.run(
    [
      str(paths.python), '-c', _VALIDATION_PROGRAM,
      str(task.output_dir), task.arm.name, str(task.shard_index),
      IMMUTABLE_RUNNER_GIT_SHA, str(task.arm.adapter),
      task.arm.adapter_sha256, str(task.arm.adapter_manifest),
      task.arm.adapter_manifest_sha256, ADAPTER_ORIGIN_EVIDENCE_SHA256,
      dataset.data_config, dataset.logical_dataset, dataset.prompt_sha256,
      dataset.prompt_manifest_sha256, str(dataset.num_prompts),
      str(dataset.global_num_samples), str(dataset.base_seed),
    ],
    cwd=paths.runner_repo,
    env=environment,
    text=True,
    capture_output=True,
    check=False,
  )
  _append_log(
    task.log_path,
    f'\n[cross-domain-validator] returncode={result.returncode}\n'
    f'{result.stdout}{result.stderr}')
  if result.returncode != 0:
    raise QueueFailure(
      f'{task.task_id} failed cryptographic/protocol validation; '
      f'see {task.log_path}')


class CrossDomainQueueController(QueueController):
  """Queue controller with exact cross-domain input authentication."""

  def __init__(
      self,
      dataset: DatasetSpec,
      paths: QueuePaths = DEFAULT_PATHS,
      *,
      reviewed_gate_sha256: str,
      reviewed_gate_path: Path | None = None,
      controller_repo_root: Path = REPO_ROOT,
      poll_seconds: float = 30.0,
      recover_stale_lock: bool = False,
      **kwargs,
  ) -> None:
    self.dataset = dataset
    self.reviewed_gate_sha256 = reviewed_gate_sha256
    self.reviewed_gate_path = (
      Path(reviewed_gate_path).expanduser().resolve()
      if reviewed_gate_path is not None else
      paths.experiment_root / 'wikitext'
      / 'reviewed-cross-domain-gate-v1.json')
    self.reviewed_gate_identity = None
    self.controller_repo_root = Path(controller_repo_root).expanduser().resolve()
    kwargs.setdefault(
      'completion_validator',
      lambda task, queue_paths: _completion_validator(
        task, queue_paths, dataset))
    super().__init__(
      frozen_cross_domain_plan(dataset.slug, paths),
      poll_seconds=poll_seconds,
      expected_launch_plan_sha256=CROSS_DOMAIN_LAUNCH_PLAN_SHA256S[
        dataset.slug],
      logical_dataset=dataset.logical_dataset,
      recover_stale_lock=recover_stale_lock,
      **kwargs,
    )

  def verify_environment(self) -> None:
    paths = self.plan.paths
    dataset = self.dataset
    prompt = paths.experiment_root / 'prompts' / dataset.prompt_name
    prompt_manifest = Path(f'{prompt}.manifest.json')
    prompt_provenance = (
      paths.experiment_root / 'prompts'
      / dataset.prompt_provenance_relative_path)
    required_files = {
      paths.runner_repo / 'scripts/run_generation_pilot.py':
        IMMUTABLE_RUNNER_SHA256,
      paths.runner_repo / 'configs/experiment'
      / 'contextual-forest-generation-paper-v1.yaml':
        IMMUTABLE_PROTOCOL_SHA256,
      paths.backbone: BACKBONE_SHA256,
      paths.experiment_root / 'adapter-pair-origin.json':
        ADAPTER_ORIGIN_EVIDENCE_SHA256,
      prompt: dataset.prompt_sha256,
      prompt_manifest: dataset.prompt_manifest_sha256,
      prompt_provenance: dataset.prompt_provenance_sha256,
      paths.runner_repo / 'configs' / 'data' / f'{dataset.data_config}.yaml':
        dataset.data_config_sha256,
    }
    for arm in _arm_specs(paths).values():
      required_files[arm.adapter] = arm.adapter_sha256
      required_files[arm.adapter_manifest] = arm.adapter_manifest_sha256
    for path, expected_sha in required_files.items():
      if not path.is_file():
        raise QueueFailure(f'required immutable input is missing: {path}')
      actual_sha = _sha256_file(path)
      if actual_sha != expected_sha:
        raise QueueFailure(
          f'immutable input SHA256 mismatch for {path}: '
          f'expected {expected_sha}, found {actual_sha}')
    try:
      prompt_evidence = json.loads(prompt_manifest.read_text())
    except (OSError, json.JSONDecodeError) as error:
      raise QueueFailure('cannot parse the pinned prompt manifest') from error
    exact_prompt_fields = {
      ('schema_version',): 2,
      ('repository', 'git_sha'): IMMUTABLE_RUNNER_GIT_SHA,
      ('repository', 'clean'): True,
      ('data_config', 'name'): dataset.data_config,
      ('data_config', 'logical_validation_dataset'): dataset.logical_dataset,
      ('data_config', 'sha256'): dataset.data_config_sha256,
      ('output', 'num_prompts'): dataset.num_prompts,
      ('output', 'sha256'): dataset.prompt_sha256,
      ('policy', 'policy_id'): 'document-local-contiguous-span-v1',
      ('policy', 'record_selection'): 'first_n_in_pinned_validation_order',
      ('policy', 'boundary_policy'): 'never_mask_first_or_last_token',
      ('policy', 'selection_seed'): 31001,
      ('policy', 'sequence_length'): SEQUENCE_LENGTH,
      ('policy', 'span_length'): 32,
      ('runtime_provenance', 'sha256'): dataset.prompt_provenance_sha256,
      ('runtime_provenance', 'path'): str(prompt_provenance.resolve()),
    }
    for keys, expected in exact_prompt_fields.items():
      value = prompt_evidence
      for key in keys:
        if not isinstance(value, dict) or key not in value:
          raise QueueFailure(
            f'prompt manifest lacks {".".join(keys)}')
        value = value[key]
      if value != expected or type(value) is not type(expected):
        raise QueueFailure(
          f'prompt manifest {".".join(keys)} differs from the frozen value')
    if not paths.python.is_file() or not os.access(paths.python, os.X_OK):
      raise QueueFailure(f'frozen Python is not executable: {paths.python}')
    try:
      revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=paths.runner_repo,
        text=True, stderr=subprocess.STDOUT).strip()
      status = subprocess.check_output(
        ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
        cwd=paths.runner_repo, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
      raise QueueFailure('cannot verify immutable runner repository') from error
    if revision != IMMUTABLE_RUNNER_GIT_SHA:
      raise QueueFailure(
        f'runner checkout is {revision}, expected {IMMUTABLE_RUNNER_GIT_SHA}')
    if status:
      raise QueueFailure('immutable runner checkout is dirty')
    try:
      self.reviewed_gate_identity = validate_reviewed_wikitext_gate(
        self.reviewed_gate_path,
        expected_sha256=self.reviewed_gate_sha256,
        require_proceed=True,
        controller_repo_root=self.controller_repo_root)
    except (OSError, TypeError, ValueError) as error:
      raise QueueFailure(
        'reviewed WikiText cross-domain launch gate failed validation: '
        f'{error}') \
        from error
    self.launch_authorization = reviewed_gate_launch_authorization(
      self.reviewed_gate_identity)
    _require_exact_launch_plan(
      self.plan, CROSS_DOMAIN_LAUNCH_PLAN_SHA256S[dataset.slug])
    if self.plan.initial_tasks:
      raise QueueFailure('cross-domain queues never adopt existing processes')
    observed_signature = tuple(
      tuple(
        (task.dataset_slug, task.arm.name, task.shard_index, task.adopted_pid)
        for task in phase)
      for phase in self.plan.phases)
    expected_signature = (
      tuple(
        (dataset.slug, 'dynamic_dynamic', index, None)
        for index in range(NUM_SHARDS)),
      tuple(
        (dataset.slug, 'static_static', index, None)
        for index in range(NUM_SHARDS)),
    )
    if observed_signature != expected_signature:
      raise QueueFailure(
        'queue task ordering differs from the exact two-phase 32-shard grid')
    for phase in self.plan.phases:
      for task in phase:
        parsed = _normalize_runner_arguments(task.command[2:])
        if parsed.get('--data-config') != dataset.data_config:
          raise QueueFailure('queue command carries the wrong data config')


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', choices=tuple(DATASETS), required=True)
  parser.add_argument(
    '--wikitext-gate-sha256', required=True,
    help='reviewed SHA256 printed by compile_wikitext_cross_domain_gate.py')
  parser.add_argument(
    '--wikitext-gate', type=Path,
    help=(
      'reviewed gate path; defaults to the frozen experiment WikiText '
      'gate location'))
  parser.add_argument(
    '--poll-seconds', type=float, default=30.0,
    help='seconds between queue state checks')
  parser.add_argument(
    '--recover-stale-lock', action='store_true',
    help='preserve and replace one reviewed stale/invalid shared queue lock')
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  controller = CrossDomainQueueController(
    DATASETS[args.dataset],
    reviewed_gate_sha256=args.wikitext_gate_sha256,
    reviewed_gate_path=args.wikitext_gate,
    poll_seconds=args.poll_seconds,
    recover_stale_lock=args.recover_stale_lock)
  try:
    controller.run()
  except Exception as error:
    print(json.dumps({
      'event': 'cross_domain_generation_queue_failed',
      'dataset': args.dataset,
      'error_type': type(error).__name__,
      'error': str(error),
    }, sort_keys=True), flush=True)
    return 1
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
