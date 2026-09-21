"""Run actual sampling parity and all training arms before releasing arrays."""
import sys,subprocess,os,json
from pathlib import Path
from campaign_common import *
from topology_variants import VARIANTS as TOPOLOGY_VARIANTS,INFERENCE_VARIANTS,TRAIN_VARIANTS
EXPERIMENT=Path(os.environ['CCF_TOPOLOGY_ROOT']);helpers=Path(__file__).parent
folder=EXPERIMENT/'gate_retry2';folder.mkdir(parents=True,exist_ok=False)
with (folder/'tests.log').open('w') as log:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(helpers),'-p','test_topology_variants.py','-v'],check=True,stdout=log,stderr=subprocess.STDOUT)
v=VARIANTS[3];exp=ROOT/'gate/generation'/v['name']
def run(name,variant):
    dest=folder/name;dest.mkdir()
    args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(4,),samples=1,seed=240001,score=False,length=128)
    save(dest/'args.json',args)
    if variant is None:cmd=[sys.executable,str(ROOT/'helpers/run_pilot.py'),STATE['code'],str(dest/'args.json')]
    else:cmd=[sys.executable,str(helpers/'run_variant_generation.py'),STATE['code'],str(dest/'args.json'),variant,str(dest/'graph-trace.jsonl')]
    with (dest/'run.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    return [json.loads(s) for s in (dest/'generation/samples.jsonl').read_text().splitlines()]
reference=run('reference',None)
for name in INFERENCE_VARIANTS:
    result=run(name,name)
    if name=='native':
        assert len(reference)==len(result)==1
        assert reference[0]['sample_token_ids']==result[0]['sample_token_ids']
        assert reference[0]['measured_nfe']==result[0]['measured_nfe']
initial_states=[];input_streams=[]
for idx,variant in enumerate(TRAIN_VARIANTS):
    with (folder/(variant+'-train.log')).open('w') as log:subprocess.run([sys.executable,str(helpers/'train_variant.py'),str(idx),'smoke'],stdout=log,stderr=subprocess.STDOUT,check=True)
    ns='topology_improvements/'+EXPERIMENT.name+'/smoke_'+variant
    result=json.loads((ROOT/ns/v['name']/'completed.json').read_text())
    assert result['global_step']==6003 and result['backbone_frozen']
    initial_states.append(json.loads((ROOT/ns/v['name']/'initial-state.json').read_text()))
    input_streams.append([json.loads(line) for line in (ROOT/ns/v['name']/'input-identity.jsonl').read_text().splitlines()])
assert all(s==initial_states[0] for s in initial_states)
assert all(s==input_streams[0] for s in input_streams)
assert len(input_streams[0])==3
save(folder/'passed.json',dict(native_tokens_and_nfe_unchanged=True,all_inference_variants_sampled=INFERENCE_VARIANTS,all_train_variants_three_steps=TRAIN_VARIANTS,backbone_frozen=True))
