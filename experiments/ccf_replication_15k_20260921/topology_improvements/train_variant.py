"""Matched 6k -> 8k continuations; original optimizer restored for all arms."""
import sys,runpy,os
from pathlib import Path
from campaign_common import *
from topology_variants import TRAIN_VARIANTS,use_variant
EXPERIMENT=Path(os.environ['CCF_TOPOLOGY_ROOT']);idx=int(sys.argv[1]);variant=TRAIN_VARIANTS[idx]
smoke=len(sys.argv)>2 and sys.argv[2]=='smoke';target=6003 if smoke else 8000
namespace='topology_improvements/'+EXPERIMENT.name+('/smoke_' if smoke else '/train_')+variant
v=VARIANTS[3];folder=ROOT/namespace/v['name'];folder.mkdir(parents=True,exist_ok=False)
source=STATE['old_checkpoints'][v['name']]
args=train_overrides(v,folder,steps=target)
args=[a.replace('checkpointing.resume_from_ckpt=false','checkpointing.resume_from_ckpt=true') for a in args]
args += [f'checkpointing.resume_ckpt_path={source}','callbacks.checkpoint_every_n_steps.save_top_k=1','++callbacks.campaign._target_=variant_observer.VariantObserver','++callbacks.campaign.variant_index=3',f'++callbacks.campaign.expected_steps={target}',f'++callbacks.campaign.namespace={namespace}',f'++callbacks.campaign.topology_variant={variant}']
save(folder/'request.json',dict(topology_variant=variant,source_checkpoint=source,source_checkpoint_sha256=sha(source),source_step=6000,target_step=target,seed=1,overrides=args,scope='Matched continuation screen, not a fresh 15k replication; optimizer restored, restarted data stream shared across variants.',variant_module_sha256=sha(Path(__file__).with_name('topology_variants.py'))))
sys.path.insert(0,STATE['code']);sys.argv=['main.py',*args]
with use_variant(variant):runpy.run_path(str(Path(STATE['code'])/'main.py'),run_name='__main__')
