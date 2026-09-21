import sys,subprocess,json
from pathlib import Path
from campaign_common import *
assert (ROOT/'gate/passed.json').is_file()
index=int(sys.argv[1]);v=VARIANTS[[3,7][index//2]];length=[128,1024][index%2]
folder=ROOT/'diagnostics/graph_generation'/v['name']/f'length{length}'
folder.mkdir(parents=True,exist_ok=False)
exp=ROOT/'gate/generation'/v['name'];helper=ROOT/'helpers'
def run(label,kind,samples,score):
    dest=folder/label;dest.mkdir()
    args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(8,32),samples=samples,seed=100001,score=score,length=length)
    save(dest/'args.json',args)
    cmd=[sys.executable,str(helper/'run_pilot.py'),STATE['code'],str(dest/'args.json')] if kind is None else [sys.executable,str(helper/'run_graph_intervention.py'),STATE['code'],str(dest/'args.json'),kind,str(dest/'graph-trace.jsonl')]
    with (dest/'run.log').open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
    save(dest/'completed.json',dict(status='completed',variant=v,sequence_length=length,intervention=kind or 'untraced_native',samples=samples,source_checkpoint_step=6000))
    return [json.loads(x) for x in (dest/'generation/samples.jsonl').read_text().splitlines()]
# A paired smoke check proves the diagnostic tracing leaves sampling unchanged.
a=run('trace_gate_reference',None,1,False);b=run('trace_gate_native','native',1,False)
assert len(a)==len(b)==2
for x,y in zip(a,b):assert x['sample_token_ids']==y['sample_token_ids'] and x['measured_nfe']==y['measured_nfe']
save(folder/'trace-gate-passed.json',dict(tokens_and_nfe_match=True))
for kind in ['native','fixed_graph','active_chain_proposals']:run(kind,kind,100,True)
save(folder/'completed.json',dict(status='completed',variant=v,sequence_length=length,samples_per_cell=100,denoising_steps=[8,32],interventions=['native','fixed_graph','active_chain_proposals']))
