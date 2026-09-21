#!/usr/bin/env python3
"""Emit one authenticated contextual-forest topology diagnostic bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.topology_diagnostics import (  # noqa: E402
  GPU_EXCLUSIVITY_ARTIFACT,
  GPU_EXCLUSIVITY_POLICY,
  GPU_MONITOR_INTERVAL_SECONDS,
  REPO_ROOT as TOPOLOGY_REPO_ROOT,
  SCHEMA_VERSION,
  SOURCE_DESCRIPTOR_FIELDS,
  SUBMISSION_GPU_LOCK,
  _validate_trusted_protocol_path,
  canonical_sha256,
  emit_topology_records,
  read_protocol,
  sha256_file,
  source_units_from_ordered_dataset,
  validate_compiled_topology_plan_lineage,
  validate_gpu_exclusivity_evidence,
  write_record_bundle,
  write_source_selection_manifest,
)
from scripts.run_compiled_job import (  # noqa: E402
  _job_execution_digest,
  _load_plan,
  _validate_repository_checkout,
  _validated_marker,
)
from scripts.run_tensor_train_feasibility import (  # noqa: E402
  GPU_EXCLUSIVITY_POLICY as SHARED_GPU_EXCLUSIVITY_POLICY,
  _ForeignPidMonitor,
  _exclusive_gpu_lock,
)


TOPOLOGY_OUTPUTS = {
  'topology_records': 'topology_records.jsonl',
  'topology_record_manifest': 'topology_records.manifest.json',
  'topology_source_selection': 'topology_source_selection.json',
  'dataset_provenance': 'data_provenance/valid-*.json',
  'gpu_exclusivity': 'gpu_exclusivity.json',
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol', type=Path, required=True)
  parser.add_argument('--plan-dir', type=Path, required=True)
  parser.add_argument('--job-id', required=True)
  parser.add_argument('--backbone-checkpoint', type=Path, required=True)
  parser.add_argument('--backbone-sha256', required=True)
  parser.add_argument('--adapter', type=Path, required=True)
  parser.add_argument('--adapter-sha256', required=True)
  parser.add_argument('--adapter-manifest', type=Path, required=True)
  parser.add_argument('--adapter-manifest-sha256', required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--device', default='cuda')
  return parser.parse_args(argv)


def _marker_named_output(
    marker: Mapping[str, Any], name: str,
) -> Mapping[str, Any]:
  matches = [output for output in marker['outputs'] if output['name'] == name]
  if len(matches) != 1:
    raise ValueError(f'adapter dependency has no unique {name!r} output')
  return matches[0]


def _marker_output_path(
    marker: Mapping[str, Any], output: Mapping[str, Any],
) -> Path:
  run_dir = Path(marker['run_dir']).expanduser().resolve()
  path = (run_dir / output['relative_path']).resolve()
  try:
    path.relative_to(run_dir)
  except ValueError as error:
    raise ValueError('dependency output escapes its run directory') from error
  if not path.is_file() or sha256_file(path) != output['sha256']:
    raise ValueError(f'dependency output hash drifted: {path}')
  return path


def _validate_runtime_job(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
  plan_dir = args.plan_dir.expanduser().resolve()
  plan, jobs = _load_plan(plan_dir)
  _validate_repository_checkout(plan, repo_root=REPO_ROOT)
  validate_compiled_topology_plan_lineage(
    plan,
    jobs=jobs,
    plan_dir=plan_dir,
    protocol_path=args.protocol,
    protocol=protocol,
    protocol_sha256=protocol_sha256)
  if args.job_id not in jobs:
    raise ValueError(f'compiled plan has no job {args.job_id!r}')
  job = jobs[args.job_id]
  identity = job['identity']
  required_identity = {
    'diagnostic', 'topology_protocol_id', 'topology_protocol_sha256',
    'control', 'dataset', 'train_seed', 'candidate_k',
  }
  if (job['kind'] != 'eval'
      or job['execution_mode'] != 'fresh_attempt'
      or not required_identity.issubset(identity)
      or identity['diagnostic'] != 'topology'
      or identity['topology_protocol_id'] != protocol['protocol_id']
      or identity['topology_protocol_sha256'] != protocol_sha256
      or identity['control'] != protocol['source_selection']['arm']
      or identity['candidate_k'] != protocol['candidate_top_k']
      or identity['dataset'] not in protocol['source_selection']['datasets']
      or identity['train_seed'] not in protocol['source_selection'][
        'train_seeds']):
    raise ValueError('compiled topology job differs from the frozen protocol')
  observed_outputs = {
    item.get('name'): item.get('pattern')
    for item in job['required_outputs'] if isinstance(item, Mapping)}
  if observed_outputs != TOPOLOGY_OUTPUTS or any(
      item.get('exactly_one') is not True for item in job['required_outputs']):
    raise ValueError('compiled topology job output contract drifted')
  if len(job['dependencies']) != 1:
    raise ValueError('topology job requires exactly one adapter dependency')
  dependency = jobs[job['dependencies'][0]]
  marker = _validated_marker(dependency, required=True)
  assert marker is not None
  adapter_output = _marker_named_output(marker, 'adapter')
  manifest_output = _marker_named_output(marker, 'adapter_manifest')
  adapter_path = _marker_output_path(marker, adapter_output)
  adapter_manifest_path = _marker_output_path(marker, manifest_output)
  if (args.adapter.expanduser().resolve() != adapter_path
      or args.adapter_manifest.expanduser().resolve() != adapter_manifest_path
      or args.adapter_sha256 != adapter_output['sha256']
      or args.adapter_manifest_sha256 != manifest_output['sha256']):
    raise ValueError('runtime adapter arguments differ from dependency outputs')
  external = [
    item for item in job['external_inputs']
    if item.get('role') == 'released_backbone_wrapper']
  if len(external) != 1:
    raise ValueError('topology job requires one released backbone input')
  backbone_path = args.backbone_checkpoint.expanduser().resolve()
  if (backbone_path != Path(external[0]['path']).expanduser().resolve()
      or args.backbone_sha256 != external[0]['sha256']
      or not backbone_path.is_file()
      or sha256_file(backbone_path) != args.backbone_sha256):
    raise ValueError('runtime backbone differs from compiled external input')
  output_dir = args.output_dir.expanduser().resolve()
  artifact_dir = Path(job['artifact_dir']).expanduser().resolve()
  try:
    relative_output = output_dir.relative_to(artifact_dir)
  except ValueError as error:
    raise ValueError('topology output directory escapes job artifact_dir') \
      from error
  if (len(relative_output.parts) != 2
      or relative_output.parts[0] != 'attempts'
      or not relative_output.parts[1].startswith('attempt-')):
    raise ValueError('topology job must run in a fresh attempt directory')
  return plan, jobs, job, sha256_file(plan_dir / 'compiled-plan.json')


def _register_resolvers(torch: Any) -> None:
  from omegaconf import OmegaConf  # noqa: PLC0415

  resolvers = {
    'cwd': os.getcwd,
    'device_count': torch.cuda.device_count,
    'eval': eval,
    'div_up': lambda left, right: (left + right - 1) // right,
  }
  for name, resolver in resolvers.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)


def _compose_config(
    *,
    data_config_name: str,
    train_seed: int,
    output_dir: Path,
    backbone_path: Path,
    adapter_path: Path,
    adapter_sha256: str,
    adapter_manifest_path: Path,
    adapter_manifest_sha256: str,
    candidate_k: int,
    component_size_cap: int,
    torch: Any,
) -> Any:
  import hydra  # noqa: PLC0415
  from omegaconf import open_dict  # noqa: PLC0415

  _register_resolvers(torch)
  with hydra.initialize_config_dir(
      config_dir=str(REPO_ROOT / 'configs'), version_base=None):
    config = hydra.compose(
      config_name='config',
      overrides=[
        'model=contextual-forest-small',
        f'data={data_config_name}',
      ])
  with open_dict(config):
    config.mode = 'ppl_eval'
    config.seed = int(train_seed)
    config.model.structured_decoder.top_k = int(candidate_k)
    config.model.structured_decoder.component_size_cap = int(
      component_size_cap)
    config.model.structured_decoder.topology_mode = 'dynamic'
    config.model.structured_decoder.factor_mode = 'dynamic'
    config.model.structured_decoder.independent_mode = False
    config.model.structured_decoder.training.backbone_checkpoint = str(
      backbone_path)
    config.model.structured_decoder.training.use_ema_backbone = False
    config.model.structured_decoder.training.strict_backbone_checkpoint = True
    config.eval.checkpoint_path = ''
    config.eval.adapter_checkpoint = str(adapter_path)
    config.eval.adapter_sha256 = adapter_sha256
    config.eval.adapter_manifest = str(adapter_manifest_path)
    config.eval.adapter_manifest_sha256 = adapter_manifest_sha256
    config.eval.disable_ema = True
    config.eval.generate_samples = False
    config.eval.compute_generative_perplexity = False
    config.training.ema = 0.0
    config.checkpointing.resume_from_ckpt = False
    config.checkpointing.save_dir = str(output_dir)
    config.data.provenance_dir = str(output_dir / 'data_provenance')
  return config


def _data_option(data_config: Any, name: str, default: Any = None) -> Any:
  if hasattr(data_config, 'get'):
    return data_config.get(name, default)
  return getattr(data_config, name, default)


def _load_ordered_dataset(
    *, config: Any, tokenizer: Any, dataloader_module: Any,
) -> Any:
  data = config.data

  def valid(name: str, default: Any = None) -> Any:
    return _data_option(data, f'valid_{name}', default)

  return dataloader_module.get_dataset(
    data.valid,
    tokenizer,
    wrap=bool(data.wrap),
    mode='validation',
    cache_dir=data.cache_dir,
    block_size=int(config.model.length),
    num_proc=1,
    streaming=False,
    revision=valid('revision'),
    dataset_name_or_path=valid('dataset_name_or_path'),
    dataset_config_name=valid('dataset_config_name'),
    source_split=valid('source_split'),
    source_window=valid('source_window'),
    expected_source_num_rows=valid('expected_source_num_rows'),
    text_field=valid('text_field'),
    document_boundary_mode=valid('document_boundary_mode', 'concatenate'),
    trust_remote_code=bool(valid('trust_remote_code', False)),
    require_pinned_provenance=bool(
      _data_option(data, 'require_pinned_provenance', False)),
    tokenizer_name_or_path=_data_option(data, 'tokenizer_name_or_path'),
    tokenizer_revision=_data_option(data, 'tokenizer_revision'),
    provenance_dir=_data_option(data, 'provenance_dir'),
    provenance_role='valid',
    disjoint_window_proof=None)


def _run_with_gpu_exclusivity(
    operation: Callable[[], Any],
    *,
    job_id: str,
    lock_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
  """Run one operation under the submission-wide CUDA exclusivity gate."""
  if SHARED_GPU_EXCLUSIVITY_POLICY != GPU_EXCLUSIVITY_POLICY:
    raise RuntimeError('shared GPU-exclusivity implementation drifted')
  selected_lock = (
    SUBMISSION_GPU_LOCK if lock_path is None else lock_path
  ).expanduser().resolve(strict=False)
  with _exclusive_gpu_lock(selected_lock) as acquired_lock:
    monitor = _ForeignPidMonitor(GPU_MONITOR_INTERVAL_SECONDS)
    with monitor:
      result = operation()
    evidence = monitor.snapshot(lock_path=acquired_lock)
    if (evidence['foreign_pid_observations']
        or evidence['monitor_errors']):
      raise RuntimeError('GPU exclusivity was lost during topology evaluation')
  return result, {
    'schema_version': SCHEMA_VERSION,
    'artifact': GPU_EXCLUSIVITY_ARTIFACT,
    'job_id': job_id,
    **evidence,
  }


def _write_gpu_exclusivity_evidence(
    *,
    output_dir: Path,
    evidence: Mapping[str, Any],
) -> Path:
  path = output_dir.expanduser().resolve() / 'gpu_exclusivity.json'
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('x', encoding='utf-8') as handle:
    json.dump(evidence, handle, indent=2, sort_keys=True)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())
  return path


def _run_topology_job_unlocked(args: argparse.Namespace) -> dict[str, Any]:
  if TOPOLOGY_REPO_ROOT != REPO_ROOT:
    raise RuntimeError('topology evaluator repository root drifted')
  protocol_path = args.protocol.expanduser().resolve()
  protocol, protocol_sha = read_protocol(protocol_path)
  _validate_trusted_protocol_path(protocol_path, protocol)
  plan, _, job, plan_sha = _validate_runtime_job(
    args=args, protocol=protocol, protocol_sha256=protocol_sha)
  if args.batch_size <= 0:
    raise ValueError('--batch-size must be positive')

  import torch  # noqa: PLC0415
  import dataloader  # noqa: PLC0415
  import diffusion  # noqa: PLC0415

  device = torch.device(args.device)
  if device.type != 'cuda':
    raise ValueError('authoritative topology jobs require a CUDA device')
  if not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable')
  identity = job['identity']
  dataset = identity['dataset']
  dataset_specification = protocol['source_selection']['datasets'][dataset]
  data_config_path = (
    REPO_ROOT / dataset_specification['data_config_path']).resolve()
  if not data_config_path.is_file():
    raise FileNotFoundError(data_config_path)
  config = _compose_config(
    data_config_name=data_config_path.stem,
    train_seed=identity['train_seed'],
    output_dir=args.output_dir.expanduser().resolve(),
    backbone_path=args.backbone_checkpoint.expanduser().resolve(),
    adapter_path=args.adapter.expanduser().resolve(),
    adapter_sha256=args.adapter_sha256,
    adapter_manifest_path=args.adapter_manifest.expanduser().resolve(),
    adapter_manifest_sha256=args.adapter_manifest_sha256,
    candidate_k=protocol['candidate_top_k'],
    component_size_cap=protocol['component_size_cap'],
    torch=torch)
  tokenizer = dataloader.get_tokenizer(config)
  dataset_object = _load_ordered_dataset(
    config=config, tokenizer=tokenizer, dataloader_module=dataloader)
  source_units = source_units_from_ordered_dataset(
    dataset_object, protocol=protocol, dataset=dataset)
  descriptors = [{
    field: source[field] for field in SOURCE_DESCRIPTOR_FIELDS
  } for source in source_units]
  selection_sha = canonical_sha256(descriptors)
  selection_path = (
    args.output_dir.expanduser().resolve()
    / 'topology_source_selection.json')
  write_source_selection_manifest(
    path=selection_path,
    protocol=protocol,
    dataset=dataset,
    entries=descriptors)
  provenance_paths = sorted(
    (args.output_dir.expanduser().resolve() / 'data_provenance').glob(
      'valid-*.json'))
  if len(provenance_paths) != 1:
    raise ValueError(
      f'topology data load produced {len(provenance_paths)} provenance files')
  provenance_path = provenance_paths[0]

  source_binding = {
    'schema_version': 2,
    'artifact': 'contextual_forest_topology_source_binding',
    'job_id': job['job_id'],
    'compiled_plan_sha256': plan_sha,
    'plan_id': plan['plan_id'],
    'job_spec_sha256': plan['job_spec_sha256'][job['job_id']],
    'job_execution_sha256': _job_execution_digest(job),
    'repository_sha': plan['repository']['sha'],
    'repository_clean': True,
    'adapter_sha256': args.adapter_sha256,
    'adapter_export_manifest_sha256': args.adapter_manifest_sha256,
    'data_config_sha256': sha256_file(data_config_path),
    'dataset_provenance_sha256': sha256_file(provenance_path),
    'evaluator_source_sha256': sha256_file(
      REPO_ROOT / protocol['evaluator_source_path']),
    'arm': identity['control'],
    'dataset': dataset,
    'train_seed': identity['train_seed'],
    'source_selection_sha256': selection_sha,
  }
  model = diffusion.Diffusion(config, tokenizer=tokenizer).to(device).eval()
  model.backbone.eval()
  model.structured_head.eval()
  model.noise.eval()
  if model.ema is not None:
    raise AssertionError('authoritative topology evaluation forbids EMA')
  with torch.no_grad():
    records = emit_topology_records(
      model=model,
      source_units=source_units,
      protocol=protocol,
      source_binding=source_binding,
      device=device,
      batch_size=args.batch_size)
  manifest_path = write_record_bundle(
    output_dir=args.output_dir,
    protocol=protocol,
    source_binding=source_binding,
    records=records)
  expected_records = (
    dataset_specification['num_source_units']
    * len(protocol['corruption_seeds'])
    * len(protocol['time_points'])
    * len(protocol['interventions']))
  if len(records) != expected_records:
    raise RuntimeError('topology emitter returned an incomplete record grid')
  return {
    'status': 'written',
    'job_id': job['job_id'],
    'dataset': dataset,
    'train_seed': identity['train_seed'],
    'num_source_units': len(source_units),
    'num_records': len(records),
    'record_manifest': str(manifest_path),
    'record_manifest_sha256': sha256_file(manifest_path),
  }


def run_topology_job(args: argparse.Namespace) -> dict[str, Any]:
  result, evidence = _run_with_gpu_exclusivity(
    lambda: _run_topology_job_unlocked(args), job_id=args.job_id)
  validated_evidence = validate_gpu_exclusivity_evidence(
    evidence, expected_job_id=args.job_id)
  evidence_path = _write_gpu_exclusivity_evidence(
    output_dir=args.output_dir, evidence=validated_evidence)
  return {
    **result,
    'gpu_exclusivity': str(evidence_path),
    'gpu_exclusivity_sha256': sha256_file(evidence_path),
  }


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  print(json.dumps(run_topology_job(args), sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
