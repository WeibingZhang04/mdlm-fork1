"""Isolated candidate tests and same-GPU full-generation comparisons to v2."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_ccf_sampling_v3 import unchecked_rows
from scripts.audit_ccf_sampling_v4 import experiment, BEST_V3, MODES as V4_MODES
from scripts.verify_ccf_sampling_optimization import problem, compare_draws, verify
import structured_objective as objective
import structured_utils as utils

MODES = ('unchecked', 'unchecked_batched', 'unchecked_batched_roots_kruskal') + V4_MODES


@torch.no_grad()
def checks(mode, device, baseline_mode='v2'):
  draw_cases = 0
  for dtype in (torch.float32, torch.float64):
    for rows in (1, 3, 32):
      for width in (5, 33, 129, 257):
        for seed in (0, 1, 91001, 91002):
          logits = torch.randn(rows, width, dtype=dtype, device=device) * 20
          logits[:, 0] = -torch.inf
          compare_draws(lambda g: utils._sample_rows(logits, g),
                        lambda g: unchecked_rows(logits, g), device, seed)
          draw_cases += 1
  with experiment(mode):
    legacy = verify(device)
  benchmarks = []
  for active_kind in ('all', 'mixed', 'none'):
    output, logits, active = problem(device, k=128, length=1024, active_kind=active_kind)
    def draw(generator):
      return objective.sample_structured_tokens(output, logits, active, generator=generator)
    def baseline(generator):
      with experiment(baseline_mode):
        return draw(generator)
    def candidate(generator):
      with experiment(mode):
        return draw(generator)
    for seed in (91001, 91002):
      compare_draws(baseline, candidate, device, seed)
    timings = {baseline_mode: [], mode: []}
    for repeat in range(6):
      variants = [(baseline_mode, baseline), (mode, candidate)]
      if repeat % 2:
        variants.reverse()
      for name, implementation in variants:
        rng = torch.Generator(device=device).manual_seed(91001)
        torch.cuda.synchronize()
        start = time.perf_counter()
        implementation(rng)
        torch.cuda.synchronize()
        if repeat:
          timings[name].append(time.perf_counter() - start)
    medians = {key: statistics.median(values) for key, values in timings.items()}
    benchmarks.append(dict(active_kind=active_kind, median_seconds=medians,
                           speedup=medians[baseline_mode] / medians[mode]))
  return dict(native_draw_cases=draw_cases, pinned_reference=legacy,
              l1024_token_rng_cases=6, isolated_benchmarks=benchmarks)


def load_model(output_dir):
  import dataloader
  import diffusion
  from scripts import run_generation_pilot as pilot
  cache = Path('/u401/n23zhang/mdlm_data/tree_mdlm_cache')
  source = cache / 'runs/ccf_separate_r16_dynamic_dynamic_s001_step7000_job1543861/attempt-1543861-r0.AZ0GWx'
  backbone = cache / 'checkpoints/mdlm-owt-backbone.pt'
  adapter = source / 'dynamic_dynamic.safetensors'
  manifest = source / 'dynamic_dynamic.manifest.json'
  args = pilot._parse_args([
    '--backbone-checkpoint', str(backbone), '--backbone-sha256', pilot.sha256_file(backbone),
    '--adapter', str(adapter), '--adapter-sha256', pilot.sha256_file(adapter),
    '--adapter-manifest', str(manifest), '--adapter-manifest-sha256', pilot.sha256_file(manifest),
    '--output-dir', str(output_dir), '--sequence-length', '1024', '--batch-size', '1',
    '--num-samples', '1', '--base-seed', '91001', '--modes', 'structured_joint', '--nfe-budgets', '1001',
    '--model-config', 'contextual-forest-small', '--data-config', 'train_openwebtext_pinned',
    '--override', 'data.cache_dir=' + str(cache / 'huggingface'),
    '--override', 'model.structured_decoder.top_k=128',
    '--override', 'model.structured_decoder.rank=16',
    '--override', '++model.structured_decoder.factor_embedding_mode=separate',
    '--override', '++model.structured_decoder.factor_conditioner_hidden_dim=0',
    '--override', 'model.structured_decoder.topology_mode=dynamic',
    '--override', 'model.structured_decoder.factor_mode=dynamic',
    '--override', 'model.structured_decoder.independent_mode=false',
    '--override', 'model.structured_decoder.training.topology_weight=0.1'])
  config = pilot._compose_config(args)
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion(config, tokenizer=tokenizer).cuda().eval()
  return model, tokenizer, {'adapter': str(adapter), 'adapter_sha256': pilot.sha256_file(adapter),
                            'backbone_sha256': pilot.sha256_file(backbone)}


@torch.no_grad()
def full_samples(mode, output_dir, report, save, baseline_mode='v2'):
  from evaluation import generation_harness as harness
  start = time.perf_counter()
  model, tokenizer, provenance = load_model(output_dir)
  torch.cuda.synchronize()
  report.update(provenance, loading_seconds=time.perf_counter() - start)
  device = torch.device('cuda')
  prompt = harness.unconditional_prompt(mask_token_id=model.mask_index, sequence_length=1024)
  report['full_samples'] = []
  save()
  for seed in (91001, 91002):
    specs = harness.expand_paired_samples([prompt], num_samples=1, base_seed=seed)
    outputs = {}
    order = (baseline_mode, mode) if seed == 91001 else (mode, baseline_mode)
    for name in order:
      print(json.dumps({'event': 'sample_start', 'variant': name, 'seed': seed}), flush=True)
      # Scoped patches only. No per-node profiling wrappers or extra syncs.
      with experiment(name):
        records, timing = harness.run_sampling_group(model, specs, sampling_mode='structured_joint',
          nfe_budget=1001, tokenizer=tokenizer, device=device)
      rng_cpu = torch.get_rng_state().clone()
      rng_cuda = torch.cuda.get_rng_state().clone()
      outputs[name] = (records, timing, rng_cpu, rng_cuda)
      (output_dir / f'{name}-seed{seed}.json').write_text(json.dumps(records, indent=2))
      print(json.dumps({'event': 'sample_complete', 'variant': name, 'seed': seed, 'timing': timing}), flush=True)
    old, new = outputs[baseline_mode], outputs[mode]
    tokens_equal = old[0][0]['sample_token_ids'] == new[0][0]['sample_token_ids']
    rng_equal = torch.equal(old[2], new[2]) and torch.equal(old[3], new[3])
    nfe_equal = old[1]['measured_nfe'] == new[1]['measured_nfe']
    row = dict(seed=seed, tokens_equal=tokens_equal, rng_equal=rng_equal, nfe_equal=nfe_equal,
               baseline_mode=baseline_mode, baseline_timing=old[1], candidate_timing=new[1],
               speedup=old[1]['wall_clock_seconds'] / new[1]['wall_clock_seconds'])
    report['full_samples'].append(row)
    save()
    assert tokens_equal and rng_equal and nfe_equal, 'Full-generation equivalence failed'
  specs = harness.expand_paired_samples([prompt], num_samples=1, base_seed=91001)
  _, report['same_gpu_mdlm_timing'] = harness.run_sampling_group(model, specs,
    sampling_mode='factorized', nfe_budget=1001, tokenizer=tokenizer, device=device)
  save()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=MODES, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--baseline-mode', choices=('v2', BEST_V3), default='v2')
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=False)
  report = dict(mode=args.mode, baseline_mode=args.baseline_mode, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                production_changed=False, status='testing',
                caveat='Two full generations per implementation, alternating order. Loading/scoring excluded. No full per-update trajectory comparison; final tokens, NFE, CPU/CUDA RNG checked.')
  def save():
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))
  save()
  try:
    report['checks'] = checks(args.mode, torch.device('cuda'), args.baseline_mode)
    report['status'] = 'synthetic_checks_passed'
    save()
    full_samples(args.mode, args.output_dir, report, save, args.baseline_mode)
    report['status'] = 'passed'
  except Exception as error:
    report.update(status='failed', error=repr(error))
    raise
  finally:
    save()
  print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
  main()
