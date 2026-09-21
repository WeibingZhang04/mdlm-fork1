import os,sys,subprocess,json,datetime
from pathlib import Path
from campaign_common import *
code=Path(STATE['code']);old=Path(STATE['old_code']);out=ROOT/'gate';out.mkdir(exist_ok=True)
sys.path.insert(0,str(code))
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0]>=8,'Backbone BF16/FlashAttention requires a compatible GPU'
assert sha(BACKBONE)==BACKBONE_SHA
save(out/'runtime.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,device_capability=torch.cuda.get_device_capability(),backbone_sha256=BACKBONE_SHA))
def run(cmd,label,debug='1'):
    env=os.environ.copy();env['MDLM_DEBUG_VALIDATION']=debug
    with (out/(label+'.log')).open('w') as f:subprocess.run(cmd,cwd=code,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
run([sys.executable,'-m','unittest','tests.test_structured_training','tests.test_structured_objective','tests.test_factor_embedding_modes','tests.test_dataloader_streaming','tests.test_generation_harness','tests.test_generation_metrics','tests.test_prepare_released_mdlm_owt','tests.test_ccf_sampling_optimization'],'unit-tests')
helper=ROOT/'helpers'
for label,source,debug in [('old',old,'1'),('new',code,'0'),('new_debug',code,'1')]:
    run([sys.executable,str(helper/'runtime_probe.py'),str(source),str(out/label)],'probe-'+label,debug)
old_probe=json.loads((out/'old/probe.json').read_text())
checks=[]
for label in ['new','new_debug']:
    probe=json.loads((out/label/'probe.json').read_text())
    for v in VARIANTS:
        n=v['name'];a=old_probe[n];b=probe[n]
        assert a['train_tokens_sha256']==b['train_tokens_sha256'] and a['valid_tokens_sha256']==b['valid_tokens_sha256']
        assert abs(a['loss']-b['loss'])<=1e-6,(label,n,a['loss'],b['loss'])
        ga=torch.load(out/'old'/n/'grads.pt',map_location='cpu');gb=torch.load(out/label/n/'grads.pt',map_location='cpu')
        assert ga.keys()==gb.keys()
        err=max((ga[k]-gb[k]).abs().max().item() for k in ga)
        assert all(torch.allclose(ga[k],gb[k],atol=1e-6,rtol=1e-5) for k in ga),(n,err)
        checks.append(dict(variant=n,comparison=label,loss_abs_error=abs(a['loss']-b['loss']),max_gradient_abs_error=err,tokens_identical=True))
save(out/'loss-gradient-comparison.json',checks)
from scripts.export_structured_adapter import export_adapter
for v in VARIANTS:
    n=v['name'];folder=out/'generation'/n;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(STATE['old_checkpoints'][n]);topology,factor,weight=ARMS[v['arm']]
    adapter=folder/'adapter.safetensors';manifest=folder/'adapter.manifest.json'
    export_adapter(checkpoint,adapter,manifest,expected_checkpoint_sha256=sha(checkpoint),control_identity=v['arm'],topology_mode=topology,factor_mode=factor,candidate_k=128,independent_mode=False,topology_weight=weight,expected_global_step=6000)
    modes=('factorized','structured_joint') if n=='basic_static_static' else ('structured_joint',)
    for label,source in [('old',old),('new',code)]:
        args=generation_args(v,adapter,manifest,folder/label,samples=1,modes=modes,score=False)
        argsfile=folder/(label+'-args.json');save(argsfile,args)
        run([sys.executable,str(helper/'run_pilot.py'),str(source),str(argsfile)],'generation-'+n+'-'+label,'0')
    a=[json.loads(x) for x in (folder/'old/samples.jsonl').read_text().splitlines()]
    b=[json.loads(x) for x in (folder/'new/samples.jsonl').read_text().splitlines()]
    assert len(a)==len(b)
    for x,y in zip(a,b):
        assert x['sample_token_ids']==y['sample_token_ids'] and x['measured_nfe']==y['measured_nfe'],n
save(out/'passed.json',dict(status='passed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_head=STATE['head'],variants=8,loss_gradient_checks=checks,generation_tokens_nfe_identical=True,scope='Same-GPU real-data forward/backward and 8/16/32-step trajectories at eight 6k checkpoints; not a proof over all possible inputs'))
print('AUDIT GATE PASSED',flush=True)
