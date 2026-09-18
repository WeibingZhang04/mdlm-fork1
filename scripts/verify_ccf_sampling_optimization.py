#!/usr/bin/env python3
"""Check token/RNG equivalence against pinned pre-fix repository code.

Local repository code only; no downloaded or third-party implementation.
Also usable on a Slurm GPU allocation with --device cuda --benchmark.
Benchmark is an isolated joint sampler, not full-model generation.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import structured_objective as current_objective
import structured_utils as current_utils
from models.structured_decoder import ContextualCouplingForestHead

BASELINE_COMMIT = '99891f9441d9aef7082a088963ae2970bc992563'
BASELINE_SHA256 = {
  'structured_utils.py': '9c26e3ecd764945fcdf171d205e22f5f331f2822baa1e7418358908a1dda9226',
  'structured_objective.py': '57f54eef2c1bbc6d1efb313a4ec3b936c7175c5a93294a1f850ddfd567a787c9',
  'evaluation/generation_harness.py': '4d8735386b85d42982492adeb369ebf89d1ae95e294fa30791b1c8732c899c63',
}


def load_baseline(path):
  source = subprocess.check_output(
    ['git', 'show', BASELINE_COMMIT + ':' + path], cwd=ROOT)
  if hashlib.sha256(source).hexdigest() != BASELINE_SHA256[path]:
    raise RuntimeError('Pre-fix source hash mismatch: ' + path)
  name = '_ccf_legacy_' + path.replace('/', '_').replace('.', '_')
  module = types.ModuleType(name)
  module.__file__ = str(ROOT / path)
  sys.modules[name] = module
  exec(compile(source, BASELINE_COMMIT + ':' + path, 'exec'), module.__dict__)
  return module


def legacy_modules():
  utils = load_baseline('structured_utils.py')
  objective = load_baseline('structured_objective.py')
  objective.structured_utils = utils
  return utils, objective


def problem(device, *, k=32, length=9, active_kind='mixed',
            topology='dynamic', factor='dynamic', embedding='shared', hidden=0):
  torch.manual_seed(413)
  if device.type == 'cuda':
    torch.cuda.manual_seed_all(413)
  # Synthetic head inputs keep diagnostic costs small; no backbone shortcut
  # is used in any production evaluation.
  parameters = inspect.signature(ContextualCouplingForestHead).parameters
  options = {}
  if 'factor_embedding_mode' in parameters:
    options['factor_embedding_mode'] = embedding
  elif embedding != 'shared':
    raise ValueError('Separate embedding option is unavailable in this checkout')
  if 'factor_conditioner_hidden_dim' in parameters:
    options['factor_conditioner_hidden_dim'] = hidden
  elif hidden:
    raise ValueError('FiLM hidden-layer option is unavailable in this checkout')
  head = ContextualCouplingForestHead(
    hidden_size=32, vocab_size=max(k * 2, 65), top_k=k, rank=16,
    time_embed_dim=8, topology_dim=16, num_anchor_slots=4,
    contextual_neighbors=2, component_size_cap=32,
    topology_mode=topology, factor_mode=factor, **options).to(device).eval()
  active = torch.ones(2, length, dtype=torch.bool, device=device)
  if active_kind == 'mixed':
    active[0, ::2] = False
    active[1, 1::3] = False
  elif active_kind == 'none':
    active[:] = False
  logits = torch.randn(2, length, head.vocab_size, device=device)
  with torch.no_grad():
    output = head(torch.randn(2, length, 32, device=device), logits,
                  torch.tensor([0.3, 0.7], device=device), active)
  return output, logits, active


def compare_draws(old, new, device, seed):
  old_generator = torch.Generator(device=device).manual_seed(seed)
  new_generator = torch.Generator(device=device).manual_seed(seed)
  old_tokens = old(old_generator)
  new_tokens = new(new_generator)
  if not torch.equal(old_tokens, new_tokens):
    raise AssertionError('Sampled tokens differ')
  if not torch.equal(old_generator.get_state(), new_generator.get_state()):
    raise AssertionError('Generator state differs')


@torch.no_grad()
def verify(device):
  old_utils, old_objective = legacy_modules()
  cases = 0
  for topology in ('fixed', 'dynamic'):
    for factor in ('fixed', 'dynamic'):
      for k in (4, 32):
        for active_kind in ('all', 'mixed', 'none'):
          output, logits, active = problem(
            device, k=k, active_kind=active_kind, topology=topology, factor=factor)
          for seed in (91001, 91002):
            compare_draws(
              lambda g: old_objective.sample_structured_tokens(
                output, logits, active, num_samples=3, generator=g),
              lambda g: current_objective.sample_structured_tokens(
                output, logits, active, num_samples=3, generator=g), device, seed)
            cases += 1
  for embedding, hidden in (('separate', 0), ('shared', 128)):
    parameters = inspect.signature(ContextualCouplingForestHead).parameters
    if embedding != 'shared' and 'factor_embedding_mode' not in parameters:
      continue
    if hidden and 'factor_conditioner_hidden_dim' not in parameters:
      continue
    output, logits, active = problem(device, embedding=embedding, hidden=hidden)
    for backend in ('dense', 'low_rank'):
      old_inference = old_objective.infer_structured_distribution(output, active, backend)
      new_inference = current_objective.infer_structured_distribution(output, active, backend)
      compare_draws(
        lambda g: old_objective.sample_structured_tokens(
          output, logits, active, num_samples=3, generator=g, inference=old_inference),
        lambda g: current_objective.sample_structured_tokens(
          output, logits, active, num_samples=3, generator=g, inference=new_inference),
        device, 91001)
      cases += 1
  # Noncanonical edge directions, branching, isolated nodes, padded edges,
  # hard state constraints, and explicit/residual clamps.
  node = torch.randn(2, 7, 33, device=device)
  left = torch.rand(2, 5, 32, 16, device=device) + 0.1
  right = torch.rand_like(left) + 0.1
  edges = torch.tensor([[1, 0], [1, 2], [3, 2], [4, 1], [99, -1]], device=device)
  edge_mask = torch.tensor([True, True, True, True, False], device=device)
  clamps = torch.tensor([[-1, -1, 32, -1, -1, 2, -1],
                         [-1, 4, -1, -1, -1, -1, -1]], device=device)
  state_mask = torch.ones_like(node, dtype=torch.bool)
  state_mask[:, 3, 0:10] = False
  compare_draws(
    lambda g: old_utils.sample_forest_low_rank(
      node, left, right, edges, 5, edge_mask=edge_mask,
      state_mask=state_mask, clamped_states=clamps, generator=g),
    lambda g: current_utils.sample_forest_low_rank(
      node, left, right, edges, 5, edge_mask=edge_mask,
      state_mask=state_mask, clamped_states=clamps, generator=g), device, 91001)
  cases += 1
  # Production uses the default generator, so verify that path too.
  output, logits, active = problem(device)
  results = []
  for implementation in (old_objective, current_objective):
    torch.manual_seed(91001)
    if device.type == 'cuda':
      torch.cuda.manual_seed_all(91001)
    tokens = implementation.sample_structured_tokens(output, logits, active)
    results.append((tokens, torch.get_rng_state(),
                    torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
  assert torch.equal(results[0][0], results[1][0])
  assert torch.equal(results[0][1], results[1][1])
  if device.type == 'cuda':
    assert torch.equal(results[0][2], results[1][2])
  return dict(equivalence_cases=cases + 1, tokens_identical=True, rng_identical=True)


@torch.no_grad()
def benchmark(device, repetitions=3):
  _, old = legacy_modules()
  records = []
  for active_kind in ('all', 'mixed', 'none'):
    output, logits, active = problem(device, k=128, length=1024, active_kind=active_kind)
    timings = {}
    for name, module in (('old', old), ('new', current_objective)):
      times = []
      for iteration in range(repetitions + 1):
        g = torch.Generator(device=device).manual_seed(91001)
        if device.type == 'cuda':
          torch.cuda.synchronize(device)
        start = time.perf_counter()
        module.sample_structured_tokens(output, logits, active, generator=g)
        if device.type == 'cuda':
          torch.cuda.synchronize(device)
        if iteration:
          times.append(time.perf_counter() - start)
      timings[name] = statistics.median(times)
    records.append(dict(active_kind=active_kind, batch_size=2, length=1024,
                        k=128, rank=16, synthetic_vocab=256,
                        median_seconds=timings, speedup=timings['old']/timings['new']))
  return records


@torch.no_grad()
def verify_real_model_steps(device):
  """Compare real checkpoint updates at three points of the unchanged schedule.

  This is not a complete generation replay. The inputs deliberately cover
  fully, half, and sparsely masked 1024-token states. Output and default RNG
  must match exactly on the same device. No production output is written.
  """
  import dataloader
  import diffusion
  from scripts import run_generation_pilot as pilot

  cache = Path(os.environ.get('CCF_CACHE_ROOT', str(ROOT / 'local-data')))
  exports = cache / 'runs/four_arm_continue_s001_k128_shared_r16_6k_to10k_lrdecay_run-lr-decay/exported_adapters'
  backbone = cache / 'checkpoints/mdlm-owt-backbone.pt'
  adapter = exports / 'dynamic_dynamic.safetensors'
  manifest = exports / 'dynamic_dynamic.manifest.json'
  args = pilot._parse_args([
    '--backbone-checkpoint', str(backbone), '--backbone-sha256', pilot.sha256_file(backbone),
    '--adapter', str(adapter), '--adapter-sha256', pilot.sha256_file(adapter),
    '--adapter-manifest', str(manifest), '--adapter-manifest-sha256', pilot.sha256_file(manifest),
    '--output-dir', '/tmp/ccf-diagnostic-not-written', '--sequence-length', '1024',
    '--batch-size', '1', '--num-samples', '1', '--base-seed', '91001',
    '--modes', 'structured_joint', '--nfe-budgets', '1001',
    '--model-config', 'contextual-forest-small', '--data-config', 'train_openwebtext_pinned',
    '--override', 'data.cache_dir=' + str(cache / 'huggingface'),
    '--override', 'model.structured_decoder.top_k=128',
    '--override', 'model.structured_decoder.rank=16',
    '--override', 'model.structured_decoder.topology_mode=dynamic',
    '--override', 'model.structured_decoder.factor_mode=dynamic',
    '--override', 'model.structured_decoder.independent_mode=false',
    '--override', 'model.structured_decoder.training.topology_weight=0.1'])
  config = pilot._compose_config(args)
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion(config, tokenizer=tokenizer).to(device).eval()
  model.structured_sampling_mode = 'structured_joint'
  _, old = legacy_modules()
  timesteps = torch.linspace(1, 1e-5, 1001, device=device)
  dt = (1 - 1e-5) / 1000
  records = []
  for index, active_count in ((0, 1024), (499, 512), (899, 102)):
    x = (torch.arange(1024, device=device)[None] % 100) + 10
    x[:, :active_count] = model.mask_index
    t = timesteps[index] * torch.ones(1, 1, device=device)
    values = []
    for module in (old, current_objective):
      torch.manual_seed(91001)
      torch.cuda.manual_seed_all(91001)
      with mock.patch.object(diffusion, 'structured_objective', module):
        value = model._ddpm_update(x.clone(), t, dt)
      values.append((value, torch.get_rng_state(), torch.cuda.get_rng_state(device)))
    if not all(torch.equal(a, b) for a, b in zip(*values)):
      raise AssertionError('Real-model token/RNG mismatch at update index ' + str(index))
    records.append(dict(update_index=index, active_count=active_count,
                        tokens_identical=True, rng_identical=True))
  return dict(checkpoint=str(adapter), adapter_sha256=pilot.sha256_file(adapter),
              sequence_length=1024, schedule_reverse_steps=1000,
              full_trajectory_verified=False, checks=records)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
  parser.add_argument('--benchmark', action='store_true')
  parser.add_argument('--real-step-check', action='store_true')
  args = parser.parse_args()
  torch.set_num_threads(1)
  device = torch.device(args.device)
  result = dict(baseline_commit=BASELINE_COMMIT, baseline_sha256=BASELINE_SHA256,
                device=str(device), torch_version=torch.__version__, **verify(device))
  if args.benchmark:
    result['isolated_sampler_benchmark'] = benchmark(device)
  if args.real_step_check:
    if device.type != 'cuda':
      raise ValueError('--real-step-check requires cuda')
    result['real_model_checks'] = verify_real_model_steps(device)
  print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
  main()
