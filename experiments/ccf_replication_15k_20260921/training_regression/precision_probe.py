"""Read-only model replay: same input/seed, first-forward cache precision only."""
import os,sys,json,hashlib
from pathlib import Path
code=Path(sys.argv[1]);out=Path(sys.argv[2]);case=sys.argv[3]
sys.path.insert(0,str(code));out.mkdir(parents=True,exist_ok=False)
import torch,hydra,lightning as L
from omegaconf import OmegaConf
from campaign_common import train_overrides,VARIANTS,save
import main,dataloader,diffusion
from campaign_observer import probe_batch

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def cache(m):
 r=m.backbone.rotary_emb
 return {'length':r.seq_len_cached,'cos_dtype':str(r.cos_cached.dtype) if r.cos_cached is not None else None,'cos_sha256':digest(r.cos_cached.float()) if r.cos_cached is not None else None}
L.seed_everything(1)
v=VARIANTS[4]
with hydra.initialize_config_dir(config_dir=str(code/'configs'),version_base=None):cfg=hydra.compose(config_name='config',overrides=train_overrides(v,out))
tok=dataloader.get_tokenizer(cfg);train,val=dataloader.get_dataloaders(cfg,tok)
preview=next(iter(train));vb=next(iter(val))
m=diffusion.Diffusion(cfg,tokenizer=tok).cuda().train()
initial_head={n:digest(p) for n,p in m.structured_head.named_parameters()}
if case!='cold':
 with torch.random.fork_rng(devices=[m.device.index]):
  _,probe_loader=dataloader.get_dataloaders(cfg,tok,skip_train=True)
  probe=next(iter(probe_loader))
 with torch.random.fork_rng(devices=[m.device.index]):
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=case=='warm_bf16'):
   probe_batch(m,probe,step=0)
warm_cache=cache(m)
m.on_train_epoch_start()
rows=[]
with torch.no_grad():
 for i,batch in enumerate(train):
  x=batch['input_ids'].cuda();att=batch['attention_mask'].cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=m._loss(x,att,phase='train')
  row={'step':i+1,'tokens_sha256':digest(x),'loss':float(loss.loss),'metrics':{k:float(t) for k,t in m._last_structured_metrics.items()},'cache':cache(m)}
  rows.append(row)
  print(case,i+1,row['metrics']['factorized_aux_nll'],row['cache']['cos_dtype'],flush=True)
  if i==9:break
save(out/'result.json',dict(case=case,code=str(code),gpu=torch.cuda.get_device_name(),torch=torch.__version__,initial_head=initial_head,preview_sha256=digest(preview['input_ids']),warm_cache=warm_cache,rows=rows))
