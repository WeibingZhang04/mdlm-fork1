"""Historical evaluator/sampler unchanged, with a read-only cache assertion."""
import json
import sys
from common import STATE, save
sys.path.insert(0, STATE['eval_code'])
import torch
from models.dit import Rotary
original = Rotary.forward
seen = []


def checked_forward(self, x, *args, **kwargs):
    result = original(self, x, *args, **kwargs)
    assert self.cos_cached.dtype == torch.float32, 'Historical generation must use FP32 positional cache'
    if not seen:
        seen.append(str(self.cos_cached.dtype))
    return result


Rotary.forward = checked_forward
from scripts import run_generation_pilot as pilot
from scripts.audit_ccf_sampling_v4 import experiment
args = json.loads(open(sys.argv[1]).read())
with experiment('level_draws'):
    result = pilot.main(args)
assert result in (None, 0)
assert seen
save(sys.argv[2], dict(cache_dtype=seen[0], sampler='historical level_draws', architecture_changed=False))
