import os,sys,subprocess
from pathlib import Path
from campaign_common import *
study=Path(os.environ['CCF_PRECISION_STUDY']);v=VARIANTS[3]
assert (study/'train-pair-passed.json').is_file()
folder=study/'evaluation';folder.mkdir(exist_ok=False)
cases=[('before6k',ROOT/'gate/generation'/v['name'])]
for precision in ('bf16','fp32'):
    namespace='precision_causal/'+study.name+'/train_'+precision
    cases.append(('after8k_'+precision,ROOT/(namespace+'_exports')/v['name']/'step008000'))
for label,exp in cases:
    dest=folder/label;dest.mkdir()
    modes=('factorized','structured_marginal','structured_joint') if label=='before6k' else ('structured_marginal','structured_joint')
    args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(8,32),samples=100,seed=240001,modes=modes)
    save(dest/'args.json',args)
    with (dest/'run.log').open('w') as f:subprocess.run([sys.executable,str(study/'generate.py'),STATE['code'],str(dest/'args.json'),str(dest)],stdout=f,stderr=subprocess.STDOUT,check=True)
    save(dest/'completed.json',dict(label=label,samples_per_cell=100,base_seed=240001,denoising_steps=[8,32]))
save(folder/'completed.json',dict(status='completed',all_cases=[x[0] for x in cases]))
