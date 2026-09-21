import sys,runpy
from pathlib import Path
from campaign_common import *
assert (ROOT/'gate/passed.json').is_file() and (ROOT/'smoke/passed').is_file()
idx=int(sys.argv[1]);weight=[.03,.1,.3][idx];namespace=['weight003','weight010','weight030'][idx]
smoke=len(sys.argv)>2 and sys.argv[2]=='smoke'
target=6003 if smoke else 8000
if smoke:namespace+='_smoke_retry2'
v=VARIANTS[3];folder=ROOT/namespace/v['name']
if folder.exists():raise RuntimeError('Existing weight trial requires an audited resume')
folder.mkdir(parents=True)
source=STATE['old_checkpoints'][v['name']]
args=train_overrides(v,folder,steps=target)
replacements={'model.structured_decoder.training.topology_weight':weight,'checkpointing.resume_from_ckpt':'true'}
args=[f'{a.split("=",1)[0]}={replacements[a.split("=",1)[0]]}' if a.split('=',1)[0] in replacements else a for a in args]
args += [f'checkpointing.resume_ckpt_path={source}','callbacks.checkpoint_every_n_steps.save_top_k=1','++callbacks.campaign._target_=weight_observer.WeightObserver','++callbacks.campaign.variant_index=3',f'++callbacks.campaign.expected_steps={target}',f'++callbacks.campaign.namespace={namespace}']
save(folder/'request.json',dict(variant=v,weight=weight,overrides=args,source_checkpoint=source,source_checkpoint_sha256=sha(source),source_step=6000,target_step=target,scope='Controlled 2000-step continuation from identical old Basic DD checkpoint; optimizer restored, stream restart shared across weights. Separate from fresh primary 15k runs.'))
sys.path.insert(0,STATE['code']);sys.argv=['main.py',*args]
runpy.run_path(str(Path(STATE['code'])/'main.py'),run_name='__main__')
