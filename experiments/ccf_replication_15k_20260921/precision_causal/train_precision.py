"""Resume the exact same old DD weights and optimizer under each cache policy."""
import sys,runpy,os
from pathlib import Path
from campaign_common import *
study=Path(os.environ['CCF_PRECISION_STUDY']);precision=sys.argv[1];stage=sys.argv[2]
assert precision in ('bf16','fp32') and stage in ('smoke','train')
target=6003 if stage=='smoke' else 8000
namespace='precision_causal/'+study.name+'/'+stage+'_'+precision
v=VARIANTS[3];folder=ROOT/namespace/v['name'];folder.mkdir(parents=True,exist_ok=False)
source=STATE['old_checkpoints'][v['name']]
args=train_overrides(v,folder,steps=target)
args=[a.replace('checkpointing.resume_from_ckpt=false','checkpointing.resume_from_ckpt=true') for a in args]
args += [f'checkpointing.resume_ckpt_path={source}','callbacks.checkpoint_every_n_steps.save_top_k=1','++callbacks.campaign._target_=precision_observer.PrecisionObserver','++callbacks.campaign.variant_index=3',f'++callbacks.campaign.expected_steps={target}',f'++callbacks.campaign.namespace={namespace}',f'++callbacks.campaign.cache_precision={precision}']
save(folder/'request.json',dict(cache_precision=precision,source_checkpoint=source,source_checkpoint_sha256=sha(source),source_step=6000,target_step=target,seed=1,overrides=args,scope='One predeclared numerical intervention; same checkpoint, optimizer and restarted data stream for both policies. No architecture or scoring changes.'))
sys.path.insert(0,STATE['code']);sys.argv=['main.py',*args]
runpy.run_path(str(Path(STATE['code'])/'main.py'),run_name='__main__')
