#!/usr/bin/env python3
"""Opt-in verified level-batched CCF generation; same arguments as the pilot.

GPU verification: 1544523_1, two complete samples, exact tokens/NFE/RNG.
Algorithm and implementation attribution: docs/ccf-fast-generation.md.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_ccf_sampling_v4 import experiment


def run(pilot, argv=None):
  args = pilot._parse_args(argv)
  provenance = {
    'sampler_implementation': 'level_draws',
    'verification_job': '1544523_1',
    'verification_scope': 'Two full rank16 DD7000 samples: final tokens, NFE, CPU/CUDA RNG; not every intermediate state.',
    'attribution': 'docs/ccf-fast-generation.md',
    'source_sha256': {
      name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
      for name in ('scripts/run_generation_fast.py', 'scripts/run_generation_pilot.py',
                   'scripts/audit_ccf_sampling_v3.py', 'scripts/audit_ccf_sampling_v4.py',
                   'structured_utils.py', 'structured_objective.py',
                   'models/structured_decoder.py', 'diffusion.py',
                   'evaluation/generation_harness.py')},
  }
  with experiment('level_draws'):
    result = pilot.main(argv)
  if result == 0:
    with (args.output_dir / 'sampler-provenance.json').open('x') as handle:
      json.dump(provenance, handle, indent=2)
  return result


def main(argv=None):
  from scripts import run_generation_pilot as pilot
  return run(pilot, argv)


if __name__ == '__main__':
  raise SystemExit(main())
