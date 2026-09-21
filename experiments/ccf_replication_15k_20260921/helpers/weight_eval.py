import sys,subprocess
from campaign_common import *
idx=int(sys.argv[1]);weight=[.03,.1,.3][idx];namespace=['weight003','weight010','weight030'][idx];v=VARIANTS[3]
assert (ROOT/namespace/v['name']/'completed.json').is_file()
for label,exp,wgt,modes in [('before6k',ROOT/'gate/generation'/v['name'],.1,('structured_joint',)),('after8k',ROOT/(namespace+'_exports')/v['name']/'step008000',weight,('factorized','structured_marginal','structured_joint'))]:
 dest=ROOT/'weight_evaluation'/namespace/label;dest.mkdir(parents=True,exist_ok=False)
 args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(8,32),samples=100,seed=100001,modes=modes)
 args=[f'model.structured_decoder.training.topology_weight={wgt}' if a.startswith('model.structured_decoder.training.topology_weight=') else a for a in args]
 save(dest/'args.json',args)
 with (dest/'run.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'helpers/run_pilot.py'),STATE['code'],str(dest/'args.json')],check=True,stdout=log,stderr=subprocess.STDOUT)
 save(dest/'completed.json',dict(status='completed'))
