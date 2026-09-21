#!/usr/bin/env python3
"""LOCAL-ONLY speed experiments; no production modules or jobs are changed.

Unchecked categorical algorithm adapted from PyTorch v2.2.2 Distributions.cpp
multinomial_out single-sample path. Source and license:
https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/native/Distributions.cpp
docs/third-party/pytorch-v2.2.2-LICENSE.txt
All other experiments adapt this repository's existing code.
"""
import ast
import cProfile
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import inspect
import json
from pathlib import Path
import statistics
import sys
import time
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import structured_utils as utils
import structured_objective as objective
from scripts.verify_ccf_sampling_optimization import problem, verify, compare_draws
import models.structured_decoder as decoder

ORIGINAL_INFER = utils._single_low_rank_sum_product


def unchecked_rows(logits, generator):
  # Same softmax, exponential draw shape/order, division, and argmax as native
  # multinomial with num_samples=1. Only valid-input checks are omitted.
  probabilities = torch.softmax(logits, dim=-1)
  noise = torch.empty_like(probabilities).exponential_(1, generator=generator)
  torch.div(probabilities, noise, out=noise)
  return noise.argmax(dim=-1)


def batched_upward(nodes, left, right, edges, topology, *, sampling_only=False,
                   batch_roots=False):
  if not sampling_only:
    return ORIGINAL_INFER(nodes, left, right, edges, topology)
  children = {node: [neighbor for neighbor, _, _ in topology.adjacency[node]
                     if neighbor != topology.parent[node]]
              for node in topology.order}
  depth = {}
  buckets = defaultdict(list)
  for node in topology.order:
    parent = topology.parent[node]
    depth[node] = 0 if parent < 0 else depth[parent] + 1
    if parent >= 0:
      buckets[(depth[node], len(children[node]))].append(node)
  messages = {}
  for level, degree in sorted(buckets, reverse=True):
    group = buckets[(level, degree)]
    local = nodes[group]
    # Preserve adjacency summation order; do not replace with a tree reduction.
    for slot in range(degree):
      local = local + torch.stack([messages[(children[node][slot], node)] for node in group])
    source = []
    target = []
    for node in group:
      edge = topology.parent_edge[node]
      a, b = (left[edge], right[edge]) if topology.edge_left[edge] == node else (right[edge], left[edge])
      source.append(a)
      target.append(b)
    outgoing = utils._batched_low_rank_message(
      local, torch.stack(source), torch.stack(target), safe_logsumexp=False)
    for slot, node in enumerate(group):
      messages[(node, topology.parent[node])] = outgoing[slot]
  beliefs = {}
  if batch_roots:
    root_groups = defaultdict(list)
    for root in topology.roots:
      root_groups[len(topology.adjacency[root])].append(root)
    for degree, roots in root_groups.items():
      value = nodes[roots]
      for slot in range(degree):
        value = value + torch.stack([
          messages[(topology.adjacency[root][slot][0], root)] for root in roots])
      for root, row in zip(roots, value.unbind()):
        beliefs[root] = row
    values = torch.stack([beliefs[root] for root in topology.roots])
    partitions = torch.logsumexp(values, -1)
    utils._require(bool(torch.isfinite(partitions).all().item()),
                   'constraints leave the forest with no finite-probability state')
    normalized = values - partitions[:, None]
    normalized = normalized - torch.logsumexp(normalized, -1, keepdim=True)
    return partitions.sum(), dict(zip(topology.roots, normalized.unbind())), messages
  for root in topology.roots:
    value = nodes[root]
    for neighbor, _, _ in topology.adjacency[root]:
      value = value + messages[(neighbor, root)]
    beliefs[root] = value
  partitions = torch.stack([torch.logsumexp(beliefs[root], -1) for root in topology.roots])
  utils._require(bool(torch.isfinite(partitions).all().item()),
                 'constraints leave the forest with no finite-probability state')
  marginals = {}
  for root in topology.roots:
    value = beliefs[root] - partitions[topology.component[root]]
    marginals[root] = value - torch.logsumexp(value, -1)
  return partitions.sum(), marginals, messages


def without_checks(function):
  """Diagnostic ablation only: omit _require and validation-only reductions."""
  class RemoveChecks(ast.NodeTransformer):
    def visit_If(self, node):
      node = self.generic_visit(node)
      if not node.body:
        node.body = [ast.Pass()]
      return node
    def visit_Expr(self, node):
      if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == '_require':
        return None
      return self.generic_visit(node)
    def visit_Assign(self, node):
      if any(isinstance(target, ast.Name) and target.id in ('invalid_nodes', 'valid_factors') for target in node.targets):
        return None
      return self.generic_visit(node)
  tree = ast.fix_missing_locations(RemoveChecks().visit(ast.parse(inspect.getsource(function))))
  namespace = {}
  exec(compile(tree, '<validation-ablation>', 'exec'), vars(utils), namespace)
  return namespace[function.__name__]


@contextmanager
def experiment(mode):
  with ExitStack() as stack:
    if mode != 'v2':
      stack.enter_context(mock.patch.object(utils, '_sample_rows', unchecked_rows))
    if 'no_validation' in mode:
      for name in ('_validate_low_rank_inputs', '_constrain_nodes', '_build_topology'):
        stack.enter_context(mock.patch.object(utils, name, without_checks(getattr(utils, name))))
    if 'batched' in mode:
      from functools import partial
      stack.enter_context(mock.patch.object(utils, '_single_low_rank_sum_product',
        partial(batched_upward, batch_roots='roots' in mode)))
    if 'kruskal' in mode:
      stack.enter_context(mock.patch.object(decoder, '_bounded_kruskal_indices', list_kruskal()))
    yield


def list_kruskal():
  """Keep stable argsort/selection logic, replace per-edge CPU tensor indexing."""
  source = inspect.getsource(decoder._bounded_kruskal_indices)
  source = source.replace('edges_cpu = proposal_edge_index.detach().cpu()',
                          'edges_cpu = proposal_edge_index.detach().cpu().tolist()')
  source = source.replace('active_cpu = active_mask.detach().cpu()',
                          'active_cpu = active_mask.detach().cpu().tolist()\n  score_values = scores_cpu.tolist()')
  source = source.replace('bool(active_cpu[batch_index, i])', 'active_cpu[batch_index][i]')
  source = source.replace('float(scores_cpu[batch_index, proposal_slot])', 'score_values[batch_index][proposal_slot]')
  source = source.replace('edges_cpu[batch_index, proposal_slot].tolist()', 'edges_cpu[batch_index][proposal_slot]')
  namespace = {}
  exec(compile(source, '<list-kruskal-prototype>', 'exec'), vars(decoder), namespace)
  return namespace['_bounded_kruskal_indices']


@torch.no_grad()
def profile_head():
  """Real tensor sizes but synthetic logits/weights: not trained-model timing."""
  torch.manual_seed(415)
  head = decoder.ContextualCouplingForestHead(
    hidden_size=768, vocab_size=50258, top_k=128, rank=16,
    factor_embedding_mode='separate').eval()
  hidden = torch.randn(1, 1024, 768)
  logits = torch.randn(1, 1024, 50258)
  logits[..., -1] = -torch.inf
  timestep = torch.tensor([0.5])
  fast_kruskal = list_kruskal()
  records = []
  def unused_proposals(context, active):
    batch = context.shape[0]
    return (torch.empty(batch, 0, 2, dtype=torch.long),
            torch.empty(batch, 0, dtype=torch.bool), context.new_empty(batch, 0),
            context.new_empty(batch, 0, 0), torch.empty(batch, 0, dtype=torch.long),
            context.new_empty(batch, 0, 0))
  for topology in ('fixed', 'dynamic'):
    for active_count in (1024, 512, 102):
      active = torch.arange(1024)[None] < active_count
      def forward():
        return head(hidden, logits, timestep, active, topology_mode=topology)
      baseline = forward()
      with ExitStack() as stack:
        if topology == 'fixed':
          stack.enter_context(mock.patch.object(head.edge_proposer, 'forward', unused_proposals))
        else:
          stack.enter_context(mock.patch.object(decoder, '_bounded_kruskal_indices', fast_kruskal))
        candidate = forward()
        timings = []
        for _ in range(3):
          start = time.perf_counter()
          forward()
          timings.append(time.perf_counter() - start)
      for name in ('candidate_ids', 'unary_log_potentials', 'candidate_state_mask',
                   'pair_left_factors', 'pair_right_factors', 'edge_index', 'edge_mask'):
        assert torch.equal(getattr(baseline, name), getattr(candidate, name)), name
      timings_base = []
      for _ in range(3):
        start = time.perf_counter()
        forward()
        timings_base.append(time.perf_counter() - start)
      profiler = cProfile.Profile()
      profiler.runcall(forward)
      wanted = {'_candidate_lattice', '_bounded_kruskal_indices', '_selected_edges',
                '_node_candidate_factors', 'score_edges'}
      import pstats
      breakdown = {name: {'calls': stats[1], 'cumulative_seconds': stats[3]}
                   for (_, _, name), stats in pstats.Stats(profiler).stats.items()
                   if name in wanted}
      records.append(dict(topology=topology, active_count=active_count,
        baseline_head_seconds=statistics.median(timings_base),
        candidate_head_seconds=statistics.median(timings),
        consumed_head_tensors_identical=True, nested_cpu_profile=breakdown))
  return records


def main():
  torch.set_num_threads(1)
  device = torch.device('cpu')
  modes = ('v2', 'unchecked', 'unchecked_no_validation', 'unchecked_batched',
           'unchecked_batched_no_validation')
  result = dict(device='cpu', torch_version=torch.__version__,
                production_changed=False, gpu_verified=False, full_trajectory_verified=False,
                equivalence={}, benchmarks=[])
  for mode in modes[1:]:
    try:
      with experiment(mode):
        result['equivalence'][mode] = verify(device)
    except AssertionError as error:
      result['equivalence'][mode] = {'passed': False, 'error': str(error)}
  with torch.no_grad():
    for active_kind in ('all', 'mixed', 'none'):
      output, logits, active = problem(device, k=128, length=1024, active_kind=active_kind)
      def draw(generator):
        return objective.sample_structured_tokens(output, logits, active, generator=generator)
      medians = {}
      for mode in modes:
        def candidate(generator):
          with experiment(mode):
            return draw(generator)
        match = True
        try:
          for seed in (91001, 91002):
            compare_draws(draw, candidate, device, seed)
        except AssertionError:
          match = False
        times = []
        with experiment(mode):
          for iteration in range(6):
            generator = torch.Generator(device=device).manual_seed(91001)
            start = time.perf_counter()
            draw(generator)
            if iteration:
              times.append(time.perf_counter() - start)
        medians[mode] = {'seconds': statistics.median(times), 'tokens_rng_identical': match}
      for record in medians.values():
        record['speedup_vs_v2'] = medians['v2']['seconds'] / record['seconds']
      result['benchmarks'].append(dict(active_kind=active_kind, batch_size=2,
        length=1024, k=128, rank=16, synthetic_vocab=256, variants=medians))
  result['head_profiles'] = profile_head()
  print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
  main()
