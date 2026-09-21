import os
from pathlib import Path
from campaign_common import ROOT,VARIANTS
STUDY=Path(os.environ['CCF_FORENSIC_ROOT'])

def cases():
    result=[]
    for index,label in [(1,'FD'),(3,'DD')]:
        v=VARIANTS[index]
        result.append((label+'_old6k',v,ROOT/'gate/generation'/v['name']))
        for step in (6000,15000):
            result.append((label+'_fresh'+str(step//1000)+'k',v,ROOT/'exports'/v['name']/f'step{step:06d}'))
    return result
