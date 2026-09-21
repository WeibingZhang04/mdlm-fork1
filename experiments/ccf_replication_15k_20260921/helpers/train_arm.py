import os,sys,json
from pathlib import Path
from campaign_common import *
assert (ROOT/'gate/passed.json').is_file(),'Code audit must pass before training'
assert (ROOT/'smoke/passed').is_file(),'Training/export smoke checks must pass first'
index=int(sys.argv[1]);v=VARIANTS[index];folder=ROOT/'training'/v['name']
if folder.exists():raise RuntimeError('Training output exists; explicit audited resume is required')
folder.mkdir(parents=True)
overrides=train_overrides(v,folder)+['callbacks.checkpoint_every_n_steps.save_top_k=1','++callbacks.campaign._target_=campaign_observer.CampaignObserver',f'++callbacks.campaign.variant_index={index}']
save(folder/'request.json',dict(variant=v,overrides=overrides,source_head=STATE['head'],fixed_probe_is_diagnostic_only=True))
sys.path.insert(0,STATE['code']);os.chdir(STATE['code']);sys.argv=['main.py',*overrides]
import runpy
runpy.run_path(str(Path(STATE["code"])/"main.py"),run_name="__main__")
