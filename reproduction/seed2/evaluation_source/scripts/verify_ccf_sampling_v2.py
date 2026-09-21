#!/usr/bin/env python3
"""Exact checks plus v1/v2 isolated timing; no production outputs written."""
import argparse
import hashlib
import json
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
from scripts.verify_ccf_sampling_optimization import (
  current_objective, problem, compare_draws, verify, verify_real_model_steps)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
  parser.add_argument('--v1-utils', type=Path)
  parser.add_argument('--real-step-check', action='store_true')
  args = parser.parse_args()
  source = (args.v1_utils.read_bytes() if args.v1_utils else subprocess.check_output(
    ['git', 'show', 'fc750ad6a59a869e59c459498dae359db864551f:structured_utils.py'], cwd=ROOT))
  assert hashlib.sha256(source).hexdigest() == '61892523cb59ad6da842326df3fcf319989d5b9efb50da8d0b039aa99e0a15bf'
  v1 = types.ModuleType('_ccf_v1_reference')
  sys.modules[v1.__name__] = v1
  exec(compile(source, 'pinned_v1_structured_utils.py', 'exec'), v1.__dict__)
  torch.set_num_threads(1)
  device = torch.device(args.device)
  result = dict(device=str(device), torch_version=torch.__version__,
                equivalence=verify(device), full_trajectory_verified=False)
  records = []
  def sync():
    if device.type == 'cuda':
      torch.cuda.synchronize(device)
  with torch.no_grad():
    for active_kind in ('all', 'mixed', 'none'):
      output, logits, active = problem(device, k=128, length=1024, active_kind=active_kind)
      def new(generator):
        return current_objective.sample_structured_tokens(output, logits, active, generator=generator)
      def old(generator):
        with mock.patch.object(current_objective, 'structured_utils', v1):
          return new(generator)
      for seed in (91001, 91002):
        compare_draws(old, new, device, seed)
      timings = {'v1': [], 'v2': []}
      for iteration in range(6):
        implementations = [('v1', old), ('v2', new)]
        if iteration % 2:
          implementations.reverse()
        for name, implementation in implementations:
          generator = torch.Generator(device=device).manual_seed(91001)
          sync()
          start = time.perf_counter()
          implementation(generator)
          sync()
          if iteration:
            timings[name].append(time.perf_counter() - start)
      medians = {name: statistics.median(values) for name, values in timings.items()}
      records.append(dict(active_kind=active_kind, batch_size=2, length=1024,
                          k=128, rank=16, synthetic_vocab=256,
                          median_seconds=medians, speedup=medians['v1']/medians['v2']))
  result['v1_comparison_cases'] = 6
  result['isolated_sampler_benchmark'] = records
  if args.real_step_check:
    result['real_model_checks'] = verify_real_model_steps(device)
  print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
  main()
