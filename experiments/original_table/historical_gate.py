"""Historical first-sample token/NFE/RNG check against the reference sampler."""
import json
from pathlib import Path
from unittest import mock
def save(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2)

def gated_pilot(pilot, cell, out, args):
  import torch
  import structured_objective as objective
  from historical_sampler import experiment
  original_run = pilot.run_sampling_group
  first = True
  # R8 DD uses the original v1 sampler, to investigate the failed historical replay.
  reference = 'v1' if cell['family'] == 'C' and cell['arm'] == 'dynamic_dynamic' else 'v2'
  v1 = None
  if reference == 'v1':
    import historical_reference as v1

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
      save(out / 'verification.json', report)
      save(out / 'reference-first-sample.json', old_records[0])
      assert all(checks.values()), report
      first = False
      print(json.dumps({'event': 'same_gpu_gate_passed', 'reference': reference,
                        'historical_replay': report.get('historical_replay')}), flush=True)
    return records, timing

  with mock.patch.object(pilot, 'run_sampling_group', run):
    return pilot.main(args)


