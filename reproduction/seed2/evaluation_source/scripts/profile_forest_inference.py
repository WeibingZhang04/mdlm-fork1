#!/usr/bin/env python3
"""Profile dense versus endpoint-factor exact forest inference."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import structured_utils  # noqa: E402


DEFAULT_WARMUP_REPETITIONS = 3
DEFAULT_MEASURED_REPETITIONS = 10


def bounded_chain_edges(
    length: int,
    component_size: int,
    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  if length < 1 or component_size < 1:
    raise ValueError('length and component_size must be positive')
  edges = []
  for start in range(0, length, component_size):
    stop = min(start + component_size, length)
    edges.extend((position, position + 1)
                 for position in range(start, stop - 1))
  padded = torch.zeros(
    max(length - 1, 0), 2, dtype=torch.long, device=device)
  mask = torch.zeros(
    max(length - 1, 0), dtype=torch.bool, device=device)
  if edges:
    padded[:len(edges)] = torch.tensor(edges, dtype=torch.long, device=device)
    mask[:len(edges)] = True
  return padded, mask


def _elapsed_ms(callable_, device: torch.device, repetitions: int) -> float:
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
  start = time.perf_counter()
  for _ in range(repetitions):
    callable_()
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
  return 1000.0 * (time.perf_counter() - start) / repetitions


def _tensor_bytes(*tensors: torch.Tensor) -> int:
  """Return logical bytes for independently allocated benchmark inputs."""
  return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _benchmark_inputs(
    length: int,
    candidate_count: int,
    rank: int,
    component_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
  """Construct deterministic inputs shared numerically across backends."""
  generator = torch.Generator(device=device).manual_seed(
    1000 + length + candidate_count)
  node = torch.randn(
    1, length, candidate_count + 1,
    device=device, generator=generator, dtype=torch.float32)
  left = torch.rand(
    1, max(length - 1, 0), candidate_count, rank,
    device=device, generator=generator) + 0.25
  right = torch.rand(
    1, max(length - 1, 0), candidate_count, rank,
    device=device, generator=generator) + 0.25
  edges, edge_mask = bounded_chain_edges(
    length, component_size, device)
  state_mask = torch.ones_like(node, dtype=torch.bool)
  return node, left, right, edges, edge_mask, state_mask


def _profile_low_rank_backend(
    length: int,
    candidate_count: int,
    rank: int,
    component_size: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
) -> tuple[float, int | None, int, int]:
  """Measure low-rank inference with only low-rank inputs resident."""
  node, left, right, edges, edge_mask, state_mask = _benchmark_inputs(
    length, candidate_count, rank, component_size, device)
  input_bytes = _tensor_bytes(
    node, left, right, edges, edge_mask, state_mask)
  active_edges = int(edge_mask.sum())

  def low_rank_call():
    return structured_utils.forest_sum_product_low_rank(
      node, left, right, edges,
      edge_mask=edge_mask, state_mask=state_mask,
      max_component_size=component_size)

  for _ in range(warmup):
    low_rank_call()
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
  elapsed_ms = _elapsed_ms(low_rank_call, device, repetitions)
  peak_bytes = (
    int(torch.cuda.max_memory_allocated(device))
    if device.type == 'cuda' else None)
  return elapsed_ms, peak_bytes, input_bytes, active_edges


def _profile_dense_backend(
    length: int,
    candidate_count: int,
    rank: int,
    component_size: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
) -> tuple[float, int | None, int]:
  """Measure dense inference with construction intermediates released."""
  node, left, right, edges, edge_mask, state_mask = _benchmark_inputs(
    length, candidate_count, rank, component_size, device)
  dense = structured_utils.materialize_low_rank_pair_factors(
    left, right, edge_mask=edge_mask)
  log_dense = dense.log()
  del dense, left, right
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
  input_bytes = _tensor_bytes(
    node, log_dense, edges, edge_mask, state_mask)

  def dense_call():
    return structured_utils.forest_sum_product(
      node, log_dense, edges,
      edge_mask=edge_mask, state_mask=state_mask,
      max_component_size=component_size)

  for _ in range(warmup):
    dense_call()
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
  elapsed_ms = _elapsed_ms(dense_call, device, repetitions)
  peak_bytes = (
    int(torch.cuda.max_memory_allocated(device))
    if device.type == 'cuda' else None)
  return elapsed_ms, peak_bytes, input_bytes


def profile_shape(
    length: int,
    candidate_count: int,
    rank: int,
    component_size: int,
    warmup: int,
    repetitions: int,
    device: torch.device) -> dict[str, object]:
  if warmup < 0:
    raise ValueError('warmup repetitions must be nonnegative')
  if repetitions <= 0:
    raise ValueError('measured repetitions must be positive')
  low_rank_ms, low_rank_peak, low_rank_input_bytes, active_edges = (
    _profile_low_rank_backend(
      length, candidate_count, rank, component_size,
      warmup, repetitions, device))
  if device.type == 'cuda':
    torch.cuda.empty_cache()

  result = {
    'batch_size': 1,
    'length': length,
    'k': candidate_count,
    'rank': rank,
    'active_edges': active_edges,
    'component_size': component_size,
    'warmup_repetitions': warmup,
    'measured_repetitions': repetitions,
    'low_rank_ms': low_rank_ms,
    'low_rank_peak_bytes': low_rank_peak,
    'low_rank_steady_state_input_bytes': low_rank_input_bytes,
    'dense_ms': None,
    'dense_peak_bytes': None,
    'dense_steady_state_input_bytes': None,
    'dense_over_low_rank_time': None,
  }
  try:
    dense_ms, dense_peak, dense_input_bytes = _profile_dense_backend(
      length, candidate_count, rank, component_size,
      warmup, repetitions, device)
    result.update({
      'dense_ms': dense_ms,
      'dense_peak_bytes': dense_peak,
      'dense_steady_state_input_bytes': dense_input_bytes,
      'dense_over_low_rank_time': dense_ms / low_rank_ms,
    })
  except RuntimeError as error:
    result['dense_error'] = str(error)
    if device.type == 'cuda':
      torch.cuda.empty_cache()
  return result


def verify_dense_low_rank_agreement() -> dict[str, float]:
  torch.manual_seed(72)
  node = torch.randn(1, 6, 5, dtype=torch.float64)
  left = torch.rand(1, 5, 4, 3, dtype=torch.float64) + 0.2
  right = torch.rand(1, 5, 4, 3, dtype=torch.float64) + 0.2
  edges, mask = bounded_chain_edges(6, 3, torch.device('cpu'))
  dense = structured_utils.materialize_low_rank_pair_factors(
    left, right, edge_mask=mask)
  dense_result = structured_utils.forest_sum_product(
    node, dense.log(), edges, edge_mask=mask)
  low_rank_result = structured_utils.forest_sum_product_low_rank(
    node, left, right, edges, edge_mask=mask)
  return {
    'max_log_partition_error': float((
      dense_result.log_partition
      - low_rank_result.log_partition).abs().max()),
    'max_node_marginal_error': float((
      dense_result.node_marginals
      - low_rank_result.node_marginals).abs().max()),
  }


def cuda_provenance(device: torch.device) -> dict[str, object] | None:
  if device.type != 'cuda':
    return None
  properties = torch.cuda.get_device_properties(device)
  try:
    driver = subprocess.check_output(
      ['nvidia-smi', '--query-gpu=driver_version',
       '--format=csv,noheader'], text=True).splitlines()[0].strip()
  except (OSError, subprocess.CalledProcessError, IndexError):
    driver = None
  return {
    'name': properties.name,
    'compute_capability': f'{properties.major}.{properties.minor}',
    'total_memory_bytes': int(properties.total_memory),
    'driver_version': driver,
    'cudnn_version': torch.backends.cudnn.version(),
  }


def profile_protocol(
    warmup: int,
    repetitions: int,
    device: torch.device) -> dict[str, object]:
  """Describe the timing boundary used by every emitted profile row."""
  if warmup < 0:
    raise ValueError('warmup repetitions must be nonnegative')
  if repetitions <= 0:
    raise ValueError('measured repetitions must be positive')
  return {
    'warmup_repetitions_per_backend': warmup,
    'measured_repetitions_per_backend': repetitions,
    'reported_statistic': 'arithmetic_mean_wall_clock_ms_per_call',
    'timed_scope': (
      'exact forest inference only; input, candidate-factor, and forest '
      'construction are excluded'),
    'cuda_synchronization': (
      'before_and_after_each_backend_timing_block'
      if device.type == 'cuda' else 'not_applicable'),
    'peak_memory_scope': (
      'torch allocator peak after reset; each backend retains only its '
      'required steady-state inputs plus inference outputs and temporaries'),
    'dense_construction_memory': (
      'excluded; raw dense factors and low-rank construction inputs are '
      'released before dense warmup and measurement'),
    'steady_state_input_accounting': (
      'logical tensor bytes are reported per backend in every profile row'),
  }


def _args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--shape', action='append', default=[],
                      help='LENGTH,K; repeat for multiple shapes')
  parser.add_argument('--rank', type=int, default=16)
  parser.add_argument('--component-size', type=int, default=32)
  parser.add_argument(
    '--warmup', type=int, default=DEFAULT_WARMUP_REPETITIONS)
  parser.add_argument(
    '--repetitions', type=int, default=DEFAULT_MEASURED_REPETITIONS)
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                      else 'cpu')
  return parser.parse_args(argv)


def main() -> int:
  args = _args()
  shapes = args.shape or ['256,32', '256,64', '512,64',
                          '1024,64', '1024,128']
  parsed_shapes = [tuple(map(int, shape.split(','))) for shape in shapes]
  device = torch.device(args.device)
  protocol = profile_protocol(args.warmup, args.repetitions, device)
  rows = [profile_shape(
    length, k, args.rank, args.component_size,
    args.warmup, args.repetitions, device)
          for length, k in parsed_shapes]
  try:
    git_sha = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    git_sha = 'unknown'
  try:
    git_dirty = bool(subprocess.check_output(
      ['git', 'status', '--porcelain'], cwd=REPO_ROOT, text=True).strip())
  except (OSError, subprocess.CalledProcessError):
    git_dirty = None
  payload = {
    'benchmark': 'forest_inference_profile',
    'git_sha': git_sha,
    'git_dirty': git_dirty,
    'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'platform': platform.platform(),
    'device': str(device),
    'gpu': (torch.cuda.get_device_name(device)
            if device.type == 'cuda' else None),
    'cuda_device_properties': cuda_provenance(device),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'python': platform.python_version(),
    # Record whether device filtering was active without publishing a GPU or
    # MIG UUID from CUDA_VISIBLE_DEVICES.
    'cuda_visible_devices_is_set': 'CUDA_VISIBLE_DEVICES' in os.environ,
    'profile_protocol': protocol,
    'resolved_shapes': [
      {'length': length, 'k': k, 'batch_size': 1}
      for length, k in parsed_shapes],
    'rank': args.rank,
    'component_size': args.component_size,
    'agreement': verify_dense_low_rank_agreement(),
    'rows': rows,
  }
  args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
  args.output.resolve().write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n')
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
