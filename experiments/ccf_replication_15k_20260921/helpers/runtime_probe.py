import os,sys,json,hashlib,gc
from pathlib import Path
code=Path(sys.argv[1]); out=Path(sys.argv[2]);sys.path.insert(0,str(code));out.mkdir(parents=True,exist_ok=True)
from campaign_common import *
import torch,hydra,lightning as L
from omegaconf import OmegaConf
import main,dataloader,diffusion

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
results={}
for v in VARIANTS:
    L.seed_everything(1)
    folder=out/v['name'];folder.mkdir(exist_ok=True)
    with hydra.initialize_config_dir(config_dir=str(code/'configs'),version_base=None):cfg=hydra.compose(config_name='config',overrides=train_overrides(v,folder))
    tok=dataloader.get_tokenizer(cfg)
    train,val=dataloader.get_dataloaders(cfg,tok)
    batch=next(iter(train)); vb=next(iter(val))
    m=diffusion.Diffusion(cfg,tokenizer=tok).cuda().train();m.on_train_epoch_start()
    assert not any(p.requires_grad for p in m.backbone.parameters())
    x=batch['input_ids'].cuda();att=batch['attention_mask'].cuda()
    result=m._loss(x,att,phase='train');assert torch.isfinite(result.loss)
    result.loss.backward()
    grads={n:p.grad.detach().cpu() for n,p in m.structured_head.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    assert sum(g.abs().sum().item() for g in grads.values())>0
    assert all(p.grad is None for p in m.backbone.parameters())
    torch.save(grads,folder/'grads.pt')
    results[v['name']]={'train_tokens_sha256':digest(x),'valid_tokens_sha256':digest(vb['input_ids']),'loss':result.loss.item(),'gradient_keys':list(grads),'grad_sha256':{k:digest(t) for k,t in grads.items()},'metrics':{k:float(t) for k,t in m._last_structured_metrics.items()},'head_parameter_count':sum(p.numel() for p in m.structured_head.parameters()),'backbone_frozen':True,'shape':list(x.shape),'gpu':torch.cuda.get_device_name()}
    print(v['name'],results[v['name']]['loss'],flush=True)
    del grads,result,m,train,val,batch,vb,x,att;gc.collect();torch.cuda.empty_cache()
save(out/'probe.json',results)
