import hashlib, json, os
from pathlib import Path

ROOT=Path(os.environ['CCF_CAMPAIGN_ROOT'])
STATE=json.loads((ROOT/'campaign.json').read_text())
CACHE=Path(STATE['cache'])
BACKBONE=CACHE/'checkpoints/mdlm-owt-backbone.pt'
BACKBONE_SHA='7508daae475e7c0aa39dd7014e786fa9788fe1fc37f040c076e2df021e45f605'
ARMS={'static_static':('fixed','fixed',0.),'fixed_dynamic':('fixed','dynamic',0.),'dynamic_fixed':('dynamic','fixed',.1),'dynamic_dynamic':('dynamic','dynamic',.1)}
VARIANTS=[dict(name='basic_'+a,arm=a,rank=16,embedding='shared') for a in ARMS]
VARIANTS += [dict(name=f'separate_r{r}_{a}',arm=a,rank=r,embedding='separate') for r in (8,16) for a in ('fixed_dynamic','dynamic_dynamic')]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def train_overrides(v,out,steps=15000):
    topology,factor,weight=ARMS[v['arm']]
    values={'mode':'train','data':'train_openwebtext_pinned','data.cache_dir':str(CACHE/'huggingface'),'seed':1,'backbone':'dit','parameterization':'subs','model':'contextual-forest-small','model.length':1024,'model.structured_decoder.top_k':128,'model.structured_decoder.rank':v['rank'],'++model.structured_decoder.factor_embedding_mode':v['embedding'],'++model.structured_decoder.factor_conditioner_hidden_dim':0,'model.structured_decoder.topology_mode':topology,'model.structured_decoder.factor_mode':factor,'model.structured_decoder.independent_mode':'false','model.structured_decoder.training.backbone_mode':'frozen','model.structured_decoder.training.require_pretrained_backbone':'true','model.structured_decoder.training.strict_backbone_checkpoint':'true','model.structured_decoder.training.backbone_checkpoint':str(BACKBONE),'model.structured_decoder.training.use_ema_backbone':'false','model.structured_decoder.training.deterministic_backbone':'true','model.structured_decoder.training.backbone_lr_multiplier':0.,'model.structured_decoder.training.head_lr':.0003,'model.structured_decoder.training.structured_nll_weight':1.,'model.structured_decoder.training.factorized_aux_weight':0.,'model.structured_decoder.training.topology_weight':weight,'training.antithetic_sampling':'true','training.importance_sampling':'false','training.sampling_eps':.001,'training.change_of_variables':'false','training.ema':0.,'optim.lr':.0003,'optim.weight_decay':0.,'lr_scheduler.num_warmup_steps':50,'trainer.max_steps':steps,'trainer.val_check_interval':500,'trainer.limit_val_batches':32,'trainer.num_sanity_val_steps':0,'trainer.devices':1,'trainer.precision':'bf16','loader.global_batch_size':4,'loader.eval_global_batch_size':4,'loader.batch_size':4,'loader.eval_batch_size':4,'loader.num_workers':0,'strategy.find_unused_parameters':'true','eval.generate_samples':'false','eval.compute_generative_perplexity':'false','callbacks.checkpoint_every_n_steps.every_n_train_steps':500,'checkpointing.resume_from_ckpt':'false','checkpointing.save_dir':str(out),'hydra.run.dir':str(out/'hydra'),'wandb':'null'}
    return [f'{k}={val}' for k,val in values.items()]

def generation_args(v,adapter,manifest,out,steps=(8,16,32),samples=20,seed=91001,modes=('structured_joint',),score=True,length=1024):
    topology,factor,weight=ARMS[v['arm']]
    args=['--backbone-checkpoint',str(BACKBONE),'--backbone-sha256',BACKBONE_SHA,'--adapter',str(adapter),'--adapter-sha256',sha(adapter),'--adapter-manifest',str(manifest),'--adapter-manifest-sha256',sha(manifest),'--output-dir',str(out),'--num-samples',str(samples),'--sequence-length',str(length),'--batch-size','1','--base-seed',str(seed),'--modes',*modes,'--nfe-budgets',*[str(s+1) for s in steps],'--device','cuda','--model-config','contextual-forest-small','--data-config','train_openwebtext_pinned','--allow-dirty']
    if score:args+=['--reference-lm','gpt2-large','--reference-lm-revision','32b71b12589c2f8d625668d2335a01cac3249519','--reference-lm-device','cuda','--reference-lm-batch-size','1','--reference-lm-max-length','1024','--reference-lm-dtype','float32']
    for s in [f'data.cache_dir={CACHE}/huggingface','model.structured_decoder.top_k=128',f'model.structured_decoder.rank={v["rank"]}',f'++model.structured_decoder.factor_embedding_mode={v["embedding"]}','++model.structured_decoder.factor_conditioner_hidden_dim=0',f'model.structured_decoder.topology_mode={topology}',f'model.structured_decoder.factor_mode={factor}','model.structured_decoder.independent_mode=false',f'model.structured_decoder.training.topology_weight={weight}',f'checkpointing.save_dir={out}']:
        args+=['--override',s]
    return args
