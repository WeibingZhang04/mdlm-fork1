import sys,subprocess,json,datetime
from pathlib import Path
from campaign_common import *
assert (ROOT/'gate/passed.json').is_file()
folder=ROOT/'legacy_replay';folder.mkdir(exist_ok=True)
# Exact old checkpoint/seeds, separately replayed on both code versions.
v=VARIANTS[1];exp=ROOT/'gate/generation'/v['name']
checks=[]
for label,code in [('old',STATE['old_code']),('new',STATE['code'])]:
    dest=folder/label
    if dest.exists():raise RuntimeError('Replay already attempted: '+str(dest))
    dest.mkdir()
    args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(8,16,32),samples=20,modes=('factorized','structured_joint'))
    save(dest/'args.json',args)
    with (dest/'run.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'helpers/run_pilot.py'),code,str(dest/'args.json')],check=True,stdout=log,stderr=subprocess.STDOUT)
a=[json.loads(s) for s in (folder/'old/generation/samples.jsonl').read_text().splitlines()]
b=[json.loads(s) for s in (folder/'new/generation/samples.jsonl').read_text().splitlines()]
assert len(a)==len(b)==120
for x,y in zip(a,b):
    assert x['sample_token_ids']==y['sample_token_ids'] and x['measured_nfe']==y['measured_nfe']
    assert x['reference_lm']['token_count']==y['reference_lm']['token_count']
    assert abs(x['reference_lm']['mean_nll_nats']-y['reference_lm']['mean_nll_nats'])<1e-6
save(folder/'passed.json',dict(status='passed',old_new_tokens_and_scores_match=True,matched_pairs=120,scope='20 samples × 3 denoising budgets × MDLM/Basic FD 6k; same GPU, same seeds, no checkpoint selection'))
# Fresh 100-sample MDLM baseline for the whole new confirmation matrix.
dest=ROOT/'baseline100';dest.mkdir()
args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(4,8,16,32),samples=100,seed=100001,modes=('factorized',))
save(dest/'args.json',args)
with (dest/'run.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'helpers/run_pilot.py'),STATE['code'],str(dest/'args.json')],check=True,stdout=log,stderr=subprocess.STDOUT)
save(dest/'completed.json',dict(status='completed'))
