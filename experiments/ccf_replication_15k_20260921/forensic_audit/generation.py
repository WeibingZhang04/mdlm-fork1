import sys,subprocess,json
from pathlib import Path
from campaign_common import *
from common import STUDY,cases
gate_folder=json.loads((STUDY/'deployment.json').read_text()).get('conditional_output','conditional')
assert (STUDY/gate_folder/'passed.json').is_file(),'Identity/numerical checks must pass first'
folder=STUDY/'generation';folder.mkdir(exist_ok=False)
matrix=[]
for label,v,exp in cases():
    modes=('factorized','structured_marginal','structured_joint') if label=='FD_old6k' else ('structured_marginal','structured_joint') if label=='FD_fresh6k' else ('structured_joint',)
    matrix.append((label,v,exp,'native',modes))
_,v,exp=cases()[0]
matrix += [('MDLM_fp32_normalization',v,exp,'native_fp32_normalization',('factorized',)),('neutral_factors',v,exp,'neutral_factors',('structured_marginal','structured_joint'))]
for label,v,exp,control,modes in matrix:
    dest=folder/label;dest.mkdir()
    args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(8,32),samples=100,seed=250001,modes=modes)
    save(dest/'args.json',args)
    with (dest/'run.log').open('w') as f:subprocess.run([sys.executable,str(STUDY/'run_control.py'),STATE['code'],str(dest/'args.json'),control,str(dest)],stdout=f,stderr=subprocess.STDOUT,check=True)
    save(dest/'completed.json',dict(case=label,control=control,samples_per_cell=100,seed=250001))
save(folder/'completed.json',dict(status='completed',cases=len(matrix),samples_per_cell=100))
