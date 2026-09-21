"""Explicit process-local generation ablations; no saved model is modified."""
import json,sys
from pathlib import Path
from campaign_common import save,sha
code=Path(sys.argv[1]);sys.path.insert(0,str(code))
import torch,diffusion
from models.dit import Rotary
from models.structured_decoder import ContextualCouplingForestHead as Head
from scripts import run_generation_pilot as pilot
from scripts.audit_ccf_sampling_v4 import experiment
argsfile=Path(sys.argv[2]);control=sys.argv[3];out=Path(sys.argv[4])
assert control in ('native','native_fp32_normalization','neutral_factors')
args=json.loads(argsfile.read_text())
save(out/'control.json',dict(control=control,changes={'native':'No intervention','native_fp32_normalization':'Cast native MDLM output logits to FP32 immediately before SUBS normalization; native sampler and reveal kernel unchanged','neutral_factors':'Set every coupling potential to one using the existing independent_mode ablation; retain candidates, graph and structured sampler'}[control],training_or_weights_changed=False,primary_scoring_unchanged=True,runner_sha256=sha(Path(__file__))))
orig_rotary=Rotary.forward;orig_head=Head.forward;orig_subs=diffusion.Diffusion._subs_parameterization
calls={'rotary':0,'neutral_head':0,'fp32_normalization':0}
def rotary(self,*a,**kw):
    value=orig_rotary(self,*a,**kw);assert self.cos_cached.dtype==torch.float32
    calls['rotary']+=1;return value
def head(self,*a,**kw):
    kw['independent_mode']=True;value=orig_head(self,*a,**kw)
    assert value.independent_mode;calls['neutral_head']+=1;return value
def subs(self,logits,xt):
    calls['fp32_normalization']+=1;return orig_subs(self,logits.float(),xt)
Rotary.forward=rotary
if control=='neutral_factors':Head.forward=head
if control=='native_fp32_normalization':diffusion.Diffusion._subs_parameterization=subs
try:
    with experiment('level_draws'):result=pilot.main(args)
finally:
    Rotary.forward=orig_rotary;Head.forward=orig_head;diffusion.Diffusion._subs_parameterization=orig_subs
assert result in (None,0) and calls['rotary']>0
assert control!='neutral_factors' or calls['neutral_head']>0
assert control!='native_fp32_normalization' or calls['fp32_normalization']>0
save(out/'control-passed.json',dict(calls=calls,gpu=torch.cuda.get_device_name(),generation_rotary_cache='torch.float32'))
