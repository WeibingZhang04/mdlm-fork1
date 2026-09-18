#!/usr/bin/env python3
"""CCF checkpoint evaluation matrices, with a same-GPU first-sample gate.

Built from the repository's existing export/pilot/verification workflows.
Sampler algorithm/code credits: docs/ccf-fast-generation.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = Path(os.environ.get('CCF_CACHE_ROOT', str(ROOT / 'local-data')))
BACKBONE_SHA = '7508daae475e7c0aa39dd7014e786fa9788fe1fc37f040c076e2df021e45f605'
V1_SHA = '61892523cb59ad6da842326df3fcf319989d5b9efb50da8d0b039aa99e0a15bf'
ARMS = {'static_static': ('fixed', 'fixed', 0.0),
        'fixed_dynamic': ('fixed', 'dynamic', 0.0),
        'dynamic_fixed': ('dynamic', 'fixed', 0.1),
        'dynamic_dynamic': ('dynamic', 'dynamic', 0.1)}
SEPARATE_RUNS = {
  (8, 'fixed_dynamic'): 'ccf_separate_r8_fixed_dynamic_s001_step7000_run-separate-r8-fd',
  (8, 'dynamic_dynamic'): 'ccf_separate_r8_dynamic_dynamic_s001_step7000_run-separate-r8-dd',
  (16, 'fixed_dynamic'): 'ccf_separate_r16_fixed_dynamic_s001_step7000_run-separate-r16-fd',
  (16, 'dynamic_dynamic'): 'ccf_separate_r16_dynamic_dynamic_s001_step7000_run-separate-r16-dd',
}
HISTORICAL_R8 = {
  2000: 'ccf_separate_checkpoints_run-separate-r8-checkpoints/dynamic_dynamic/step2000/attempt-0.sGG8eC/generation',
  6000: 'ccf_separate_checkpoints_run-separate-r8-checkpoints/dynamic_dynamic/step6000/attempt-0.wT9ZgU/generation',
}


def matrix(suite='selected'):
  if suite == 'full_checkpoint_sweep':
    basic_7k = matrix('basic_7k')
    cells = matrix('low_steps_sweep') + basic_7k[1:]
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    cells.sort(key=lambda cell: (cell['step'], cell['family'], arm_order[cell['arm']]))
    # One released MDLM baseline, then all 56 CCF checkpoints. No result reuse.
    return [basic_7k[0]] + cells
  if suite == 'low_steps_sweep':
    # Basic 7k and both MDLM budgets reuse the completed twenty-sample study.
    cells = []
    for step in range(1000, 7001, 1000):
      if step < 7000:
        run = ('stale/four_arm_s001_k128_run-baseline-1k' if step == 1000 else
               'four_arm_continue_s001_k128_to3k_run-continue-3k' if step <= 3000 else
               'four_arm_continue_s001_k128_3k_to6k_run-continue-6k')
        for arm in ARMS:
          cells.append(dict(family='B', arm=arm, step=step, rank=16,
                            embedding='shared', mode='structured_joint',
                            checkpoint=str(CACHE / 'runs' / run / arm / 'checkpoints' / f'0-{step}.ckpt')))
      for (rank, arm), run in SEPARATE_RUNS.items():
        cells.append(dict(family='C' if rank == 8 else 'D', arm=arm, step=step,
                          rank=rank, embedding='separate', mode='structured_joint',
                          checkpoint=str(CACHE / 'runs' / run / 'training/checkpoints' / f'0-{step}.ckpt')))
    return cells
  if suite == 'all_1k':
    archive = CACHE / 'runs/stale/four_arm_s001_k128_run-baseline-1k'
    cells = [dict(family='B', arm=arm, step=1000, rank=16, embedding='shared',
                  checkpoint=str(archive / arm / 'checkpoints/0-1000.ckpt'),
                  mode='structured_joint') for arm in ARMS]
    for (rank, arm), run in SEPARATE_RUNS.items():
      cells.append(dict(family='C' if rank == 8 else 'D', arm=arm, step=1000,
                        rank=rank, embedding='separate', mode='structured_joint',
                        checkpoint=str(CACHE / 'runs' / run / 'training/checkpoints/0-1000.ckpt')))
    return cells
  if suite == 'original_5k6k':
    archive = CACHE / 'runs/four_arm_continue_s001_k128_3k_to6k_run-continue-6k'
    return [dict(family='B', arm=arm, step=step, rank=16, embedding='shared',
                 checkpoint=str(archive / arm / 'checkpoints' / f'0-{step}.ckpt'),
                 mode='structured_joint') for step in (5000, 6000) for arm in ARMS]
  cells = []
  for arm in ARMS:
    checkpoint = CACHE / 'runs/four_arm_continue_s001_k128_6k_to10k_run-continue-10k' / arm / 'checkpoints/0-7000.ckpt'
    cells.append(dict(family='B', arm=arm, step=7000, rank=16, embedding='shared',
                      checkpoint=str(checkpoint), mode='structured_joint'))
  cells.insert(0, dict(cells[0], family='A', mode='factorized'))
  if suite == 'basic_7k':
    return cells
  for rank, steps, family in ((8, (2000, 6000), 'C'), (16, (2000, 4000), 'D')):
    for arm in ('fixed_dynamic', 'dynamic_dynamic'):
      for step in steps:
        checkpoint = CACHE / 'runs' / SEPARATE_RUNS[rank, arm] / 'training/checkpoints' / f'0-{step}.ckpt'
        cells.append(dict(family=family, arm=arm, step=step, rank=rank,
                          embedding='separate', checkpoint=str(checkpoint), mode='structured_joint'))
  return cells


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def save(path, value):
  with path.open('x') as handle:
    json.dump(value, handle, indent=2)


def generation_args(cell, out, adapter_sha, manifest_sha):
  topology, factor, weight = ARMS[cell['arm']]
  samples = cell.get('num_samples', 5)
  sampling_steps = cell.get('sampling_steps', 1000)
  args = ['--backbone-checkpoint', str(CACHE / 'checkpoints/mdlm-owt-backbone.pt'),
    '--backbone-sha256', BACKBONE_SHA, '--adapter', str(out / 'adapter.safetensors'),
    '--adapter-sha256', adapter_sha, '--adapter-manifest', str(out / 'adapter.manifest.json'),
    '--adapter-manifest-sha256', manifest_sha, '--output-dir', str(out / 'generation'),
    '--num-samples', str(samples), '--sequence-length', '1024', '--batch-size', '1',
    '--base-seed', '91001', '--modes', cell['mode'], '--nfe-budgets', str(sampling_steps + 1),
    '--device', 'cuda', '--model-config', 'contextual-forest-small',
    '--data-config', 'train_openwebtext_pinned', '--allow-dirty',
    '--reference-lm', 'gpt2-large', '--reference-lm-revision', '32b71b12589c2f8d625668d2335a01cac3249519',
    '--reference-lm-device', 'cuda', '--reference-lm-batch-size', '1',
    '--reference-lm-max-length', '1024', '--reference-lm-dtype', 'float32']
  for override in [f'data.cache_dir={CACHE}/huggingface', 'model.structured_decoder.top_k=128',
    f'model.structured_decoder.rank={cell["rank"]}',
    f'++model.structured_decoder.factor_embedding_mode={cell["embedding"]}',
    '++model.structured_decoder.factor_conditioner_hidden_dim=0',
    f'model.structured_decoder.topology_mode={topology}',
    f'model.structured_decoder.factor_mode={factor}',
    'model.structured_decoder.independent_mode=false',
    f'model.structured_decoder.training.topology_weight={weight}', f'checkpointing.save_dir={out}']:
    args += ['--override', override]
  return args


def gated_pilot(pilot, cell, out, args):
  import torch
  import structured_objective as objective
  from scripts.audit_ccf_sampling_v4 import experiment
  original_run = pilot.run_sampling_group
  first = True
  # R8 DD uses the original v1 sampler, to investigate the failed historical replay.
  reference = 'v1' if cell['family'] == 'C' and cell['arm'] == 'dynamic_dynamic' else 'v2'
  v1 = None
  if reference == 'v1':
    # Content-addressed source survives replaying the collaborator patches.
    source = subprocess.check_output(
      ['git', 'cat-file', 'blob', 'd8d05932ede6125cea511f654884d1ee64812131'], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == V1_SHA
    v1 = types.ModuleType('_ccf_selected_v1_reference')
    sys.modules[v1.__name__] = v1
    exec(compile(source, 'pinned_v1_structured_utils.py', 'exec'), v1.__dict__)

  def run(*positional, **keyword):
    nonlocal first
    if first:
      if v1 is None:
        old_records, old_timing = original_run(*positional, **keyword)
      else:
        with mock.patch.object(objective, 'structured_utils', v1):
          old_records, old_timing = original_run(*positional, **keyword)
      old_cpu, old_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with experiment('level_draws'):
      records, timing = original_run(*positional, **keyword)
    if first:
      checks = dict(tokens_equal=old_records[0]['sample_token_ids'] == records[0]['sample_token_ids'],
        nfe_equal=old_timing['measured_nfe'] == timing['measured_nfe'],
        cpu_rng_equal=torch.equal(old_cpu, torch.get_rng_state()),
        cuda_rng_equal=torch.equal(old_cuda, torch.cuda.get_rng_state()))
      report = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
        reference=reference, candidate='level_draws', reference_timing=old_timing, candidate_timing=timing,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        scope=f'One full {cell.get("sampling_steps", 1000)}-transition first sample on the same loaded model/GPU; final tokens/NFE/RNG, not every intermediate state.')
      if (reference == 'v1' and cell['step'] in HISTORICAL_R8
          and cell.get('sampling_steps', 1000) == 1000):
        old_dir = CACHE / 'runs' / HISTORICAL_R8[cell['step']]
        historical = json.loads((old_dir / 'samples.jsonl').read_text().splitlines()[0])
        report['historical_replay'] = dict(
          source=str(old_dir), source_host=json.loads((old_dir / 'manifest.json').read_text())['host'],
          reference_matches_historical=old_records[0]['sample_token_ids'] == historical['sample_token_ids'],
          fast_matches_historical=records[0]['sample_token_ids'] == historical['sample_token_ids'],
          note='A cross-GPU mismatch is reported, not attributed to the speed fix without same-GPU evidence. These are fresh five-sample evaluations, not pooled historical results.')
      save(out / 'verification.json', report)
      save(out / 'reference-first-sample.json', old_records[0])
      assert all(checks.values()), report
      first = False
      print(json.dumps({'event': 'same_gpu_gate_passed', 'reference': reference,
                        'historical_replay': report.get('historical_replay')}), flush=True)
    return records, timing

  with mock.patch.object(pilot, 'run_sampling_group', run):
    return pilot.main(args)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--index', type=int)
  parser.add_argument('--inventory', action='store_true')
  parser.add_argument('--suite', choices=('selected', 'original_5k6k', 'all_1k', 'basic_7k', 'low_steps_sweep', 'full_checkpoint_sweep'), default='selected')
  parser.add_argument('--num-samples', type=int, default=5)
  parser.add_argument('--sampling-steps', type=int, default=1000)
  parser.add_argument('--output-root', type=Path)
  options = parser.parse_args()
  if options.num_samples < 1 or options.sampling_steps < 1:
    parser.error('num-samples and sampling-steps must be positive')
  cells = matrix(options.suite)
  for cell in cells:
    cell.update(num_samples=options.num_samples, sampling_steps=options.sampling_steps)
  if options.inventory:
    missing = [c['checkpoint'] for c in cells if not Path(c['checkpoint']).is_file()]
    print(json.dumps(dict(cells=cells, samples=options.num_samples * len(cells), missing=missing), indent=2))
    assert not missing
    return
  cell = cells[options.index]
  parent = options.output_root / f'{options.index:02d}_{cell["family"]}_{cell["arm"]}_step{cell["step"]}'
  parent.mkdir(parents=True, exist_ok=True)
  out = Path(tempfile.mkdtemp(prefix=f'attempt-{os.environ.get("SLURM_JOB_ID", "local")}-', dir=parent))
  assert sha(CACHE / 'checkpoints/mdlm-owt-backbone.pt') == BACKBONE_SHA
  topology, factor, weight = ARMS[cell['arm']]
  checkpoint_sha = sha(cell['checkpoint'])
  with (out / 'export-report.json').open('x') as log:
    subprocess.run([sys.executable, 'scripts/export_structured_adapter.py',
      '--checkpoint', cell['checkpoint'], '--expected-checkpoint-sha256', checkpoint_sha,
      '--expected-global-step', str(cell['step']), '--output', str(out / 'adapter.safetensors'),
      '--manifest', str(out / 'adapter.manifest.json'), '--control-identity', cell['arm'],
      '--topology-mode', topology, '--factor-mode', factor, '--candidate-k', '128',
      '--independent-mode', 'false', '--topology-weight', str(weight)], check=True, stdout=log)
  args = generation_args(cell, out, sha(out / 'adapter.safetensors'), sha(out / 'adapter.manifest.json'))
  save(out / 'request.json', dict(cell=cell, checkpoint_sha256=checkpoint_sha, pilot_arguments=args,
    suite=options.suite, samples=options.num_samples, sampling_steps=options.sampling_steps,
    base_seed=91001, snapshot=str(ROOT),
    pairing=f'Standard pilot: replicate-0000 through replicate-{options.num_samples - 1:04d}; identical across cells.',
    source_sha256={name: sha(ROOT / name) for name in (
      'scripts/evaluate_ccf_selected_five.py', 'scripts/run_generation_pilot.py',
      'scripts/audit_ccf_sampling_v3.py', 'scripts/audit_ccf_sampling_v4.py', 'structured_utils.py',
      'structured_objective.py', 'diffusion.py', 'models/structured_decoder.py',
      'evaluation/generation_harness.py', 'evaluation/generation_metrics.py')},
    local_commit=os.environ.get('CCF_LOCAL_EVAL_COMMIT'), attribution='docs/ccf-fast-generation.md'))
  from scripts import run_generation_pilot as pilot
  result = pilot.main(args) if cell['mode'] == 'factorized' else gated_pilot(pilot, cell, out, args)
  assert result == 0
  save(out / 'completed.json', dict(status='completed', samples=options.num_samples, cell=cell))
  print(json.dumps(dict(status='completed', output=str(out))), flush=True)


if __name__ == '__main__':
  main()
