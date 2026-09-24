import argparse
from pathlib import Path
import sys
import torch
repo=Path('/u401/n23zhang/clean_tree_mdlm/mdlm-fork1')
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'experiments/original_table'))
from models.dit import Rotary
import run

def forward(self,x,seq_dim=1):
    seq_len=x.shape[seq_dim]
    if self.seq_len_cached != seq_len or self.cos_cached is None or self.cos_cached.device != x.device or self.cos_cached.dtype != torch.bfloat16:
        self.seq_len_cached=seq_len
        with torch.autocast(device_type=x.device.type,enabled=False):
            positions=torch.arange(seq_len,device=x.device,dtype=torch.float32)
            frequencies=torch.outer(positions,self.inv_freq.to(device=x.device,dtype=torch.float32))
            embedding=torch.cat((frequencies,frequencies),dim=-1)
            cosine=embedding.cos().to(torch.bfloat16)
            sine=embedding.sin().to(torch.bfloat16)
        self.cos_cached=cosine[None,:,None,None,:].repeat(1,1,3,1,1)
        self.sin_cached=sine[None,:,None,None,:].repeat(1,1,3,1,1)
        self.cos_cached[:,:,2,:,:].fill_(1.)
        self.sin_cached[:,:,2,:,:].fill_(0.)
    return self.cos_cached,self.sin_cached
Rotary.forward=forward
p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
run.evaluate(argparse.Namespace(study=a.study,suite='confirmation',index=a.index))
