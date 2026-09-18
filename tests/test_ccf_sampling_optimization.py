"""Exact regression tests for conservative CCF sampling speed fixes."""
from unittest import mock
import unittest
import subprocess
import sys

import torch

import structured_objective
import structured_utils
from evaluation import generation_harness
from scripts.verify_ccf_sampling_optimization import load_baseline, problem, verify
from tests.test_generation_harness import FakeDiffusionModel


class SamplingOptimizationTest(unittest.TestCase):
  def test_cpu_tokens_and_rng_match_pinned_prefixed_code(self):
    result = verify(torch.device('cpu'))
    self.assertGreaterEqual(result['equivalence_cases'], 50)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_cuda_tokens_and_rng_match_pinned_prefixed_code(self):
    self.assertTrue(verify(torch.device('cuda'))['rng_identical'])

  def test_no_unused_inference_and_single_topology_build(self):
    output, logits, active = problem(torch.device('cpu'))
    with mock.patch.object(structured_objective, 'infer_structured_distribution',
                           side_effect=AssertionError('Unused inference pass')), \
         mock.patch.object(structured_utils, '_build_topology',
                           wraps=structured_utils._build_topology) as build:
      structured_objective.sample_structured_tokens(output, logits, active)
    self.assertEqual(build.call_count, 1)

  def test_harness_replay_preserves_tokens_nfe_rng_and_evidence(self):
    old = load_baseline('evaluation/generation_harness.py')
    for observed in (False, True):
      tokens = torch.full((2, 8), 99)
      if observed:
        tokens[:, 0] = 3
      results = []
      for module in (old, generation_harness):
        model = FakeDiffusionModel()
        torch.manual_seed(191)
        results.append((*module.sample_from_initial_state(
          model, tokens, nfe_budget=1001), torch.get_rng_state()))
      self.assertTrue(torch.equal(results[0][0], results[1][0]))
      self.assertEqual(results[0][1], results[1][1])
      self.assertTrue(torch.equal(results[0][2], results[1][2]))

  def test_evidence_checks_not_removed_for_conditional_generation(self):
    model = FakeDiffusionModel()
    model._ddpm_update = lambda x, t, dt: torch.zeros_like(x)
    with self.assertRaisesRegex(AssertionError, 'modified observed'):
      generation_harness.sample_from_initial_state(
        model, torch.tensor([[3, 99]]), nfe_budget=3)

  def test_host_orientation_matches_noncanonical_edges(self):
    edges = torch.tensor([[[1, 0], [1, 2], [3, 2], [99, -1]]])
    topology = structured_utils._build_topology(
      edges, torch.tensor([[True, True, True, False]]), 5, None, None)[0]
    self.assertEqual(topology.edge_left, (1, 1, 3, 99))

  def test_joint_sampler_does_not_compute_full_marginals(self):
    output, logits, active = problem(torch.device('cpu'), k=32)
    with mock.patch.object(structured_utils, '_single_low_rank_sum_product',
                           wraps=structured_utils._single_low_rank_sum_product) as infer:
      structured_objective.sample_structured_tokens(output, logits, active)
    self.assertTrue(infer.call_args_list)
    self.assertTrue(all(call.kwargs['sampling_only'] for call in infer.call_args_list))

  def test_root_marginals_and_upward_messages_are_bitwise_equal(self):
    for active_kind in ('all', 'mixed', 'none'):
      output, _, active = problem(torch.device('cpu'), k=32, active_kind=active_kind)
      nodes, left, right, edges, _, topologies = structured_utils._validate_low_rank_inputs(
        output.unary_log_potentials, output.pair_left_factors,
        output.pair_right_factors, output.edge_index, output.edge_mask,
        output.candidate_state_mask,
        structured_objective._structured_clamped_states(active), None, None)
      for batch_index, topology in enumerate(topologies):
        _, full, messages = structured_utils._single_low_rank_sum_product(
          nodes[batch_index], left[batch_index], right[batch_index],
          edges[batch_index], topology)
        _, roots, upward = structured_utils._single_low_rank_sum_product(
          nodes[batch_index], left[batch_index], right[batch_index],
          edges[batch_index], topology, sampling_only=True)
        self.assertEqual(set(roots), set(topology.roots))
        self.assertEqual(len(upward), len(topology.active_edges))
        for root in roots:
          self.assertTrue(torch.equal(roots[root], full[root]))
        for key in upward:
          self.assertTrue(torch.equal(upward[key], messages[key]))

  def _check_native_draws(self, device):
    old_utils = load_baseline('structured_utils.py')
    for rows in (1, 3):
      for dtype in (torch.float32, torch.float64):
        logits = torch.randn(rows, 129, dtype=dtype, device=device)
        logits[:, :4] = -torch.inf
        old_g = torch.Generator(device=device).manual_seed(91001)
        new_g = torch.Generator(device=device).manual_seed(91001)
        for _ in range(5):
          old = old_utils._sample_rows(logits, old_g)
          new = structured_utils._sample_rows(logits, new_g)
          self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(old_g.get_state(), new_g.get_state()))

  def test_native_multinomial_cpu_equivalence(self):
    self._check_native_draws(torch.device('cpu'))

  def test_native_multinomial_rejects_invalid_weights_cpu(self):
    for bad in (float('nan'), float('inf'), -float('inf')):
      with self.assertRaises(RuntimeError):
        structured_utils._sample_rows(torch.tensor([[bad, bad], [0., 1.]]), None)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_native_multinomial_cuda_equivalence(self):
    self._check_native_draws(torch.device('cuda'))

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_native_multinomial_rejects_invalid_weights_cuda(self):
    # CUDA may signal a device assert, so isolate intentionally bad inputs.
    for bad in ('nan', 'inf', '-inf'):
      code = '''
import torch
import structured_utils
try:
  structured_utils._sample_rows(torch.full((1, 3), float(BAD), device='cuda'), None)
  torch.cuda.synchronize()
except RuntimeError:
  pass
else:
  raise AssertionError('Invalid probabilities were accepted')
'''.replace('BAD', repr(bad))
      child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
      self.assertEqual(child.returncode, 0, child.stderr)

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
  def test_cuda_native_draw_removes_two_duplicate_scalar_reads(self):
    logits = torch.randn(1, 129, device='cuda')
    counts = []
    for implementation in (load_baseline('structured_utils.py'), structured_utils):
      implementation._sample_rows(logits, None)
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        implementation._sample_rows(logits, None)
      counts.append(sum(event.count for event in profile.key_averages()
                        if event.key == 'aten::_local_scalar_dense'))
    # PyTorch itself may synchronize; only our two duplicate reads are removed.
    self.assertEqual(counts[0] - counts[1], 2)

  def test_sampler_still_rejects_impossible_constraints(self):
    nodes = torch.full((1, 2, 3), -torch.inf)
    factors = torch.ones(1, 1, 2, 2)
    with self.assertRaisesRegex(ValueError, 'finite-probability'):
      structured_utils.sample_forest_low_rank(
        nodes, factors, factors, torch.tensor([[0, 1]]), 1)


if __name__ == '__main__':
  unittest.main()
