import sys,os
from campaign_common import *
index=int(sys.argv[1]);v=VARIANTS[index];folder=ROOT/'smoke'/v['name']
if folder.exists():raise RuntimeError('Smoke attempt exists')
folder.mkdir(parents=True)
args=train_overrides(v,folder,steps=3)+['callbacks.checkpoint_every_n_steps.save_top_k=1','++callbacks.campaign._target_=campaign_observer.CampaignObserver',f'++callbacks.campaign.variant_index={index}','++callbacks.campaign.expected_steps=3','++callbacks.campaign.namespace=smoke']
sys.path.insert(0,STATE['code']);os.chdir(STATE['code']);sys.argv=['main.py',*args]
import runpy
runpy.run_path(str(Path(STATE["code"])/"main.py"),run_name="__main__")
