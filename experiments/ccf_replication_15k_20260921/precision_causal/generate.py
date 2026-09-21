"""Same generation precision and scorer for every checkpoint; assert FP32 cache."""
import sys,json
from pathlib import Path
from campaign_common import save
code=Path(sys.argv[1]);sys.path.insert(0,str(code))
import torch
from models.dit import Rotary
from scripts import run_generation_pilot as pilot
from scripts.audit_ccf_sampling_v4 import experiment
args=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
original=Rotary.forward;seen=[]
def checked(self,*a,**kw):
    result=original(self,*a,**kw)
    assert self.cos_cached.dtype==torch.float32,'Generation cache precision drift'
    if not seen:seen.append(dict(cos_dtype=str(self.cos_cached.dtype),sequence_length=self.seq_len_cached,gpu=torch.cuda.get_device_name()))
    return result
Rotary.forward=checked
try:
    with experiment('level_draws'):result=pilot.main(args)
finally:Rotary.forward=original
assert result in (None,0)
assert seen
save(out/'generation-precision.json',seen[0])
