import json,hashlib,math
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback
from campaign_common import ROOT,VARIANTS,ARMS,sha,save

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def forest_stats(output,active):
    results=[]
    for b in range(active.shape[0]):
        ids=active[b].nonzero().flatten().tolist();parent={i:i for i in ids};degree={i:0 for i in ids}
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        edges=output.edge_index[b][output.edge_mask[b]].detach().cpu().tolist()
        for i,j in edges:
            parent[find(i)]=find(j);degree[i]+=1;degree[j]+=1
        sizes={}
        for i in ids:sizes[find(i)]=sizes.get(find(i),0)+1
        results.append(dict(active_tokens=len(ids),edges=len(edges),edge_fraction_of_tree=len(edges)/max(1,len(ids)-1),components=len(sizes),largest_component=max(sizes.values(),default=0),isolated_fraction=sum(d==0 for d in degree.values())/max(1,len(ids)),mean_edge_span=sum(abs(i-j) for i,j in edges)/max(1,len(edges)),max_degree=max(degree.values(),default=0)))
    return results

@torch.no_grad()
def probe_batch(model,batch,*,step,source='fixed_wikitext_validation',rates=(.25,.5,.75,.9)):
    import structured_objective as objective
    import structured_training
    records=[]; x=batch['input_ids'].to(model.device);attention=batch['attention_mask'].to(model.device).bool()
    flags=[(m,m.training) for m in model.modules()];model.eval()
    try:
        for rate in rates:
            g=torch.Generator().manual_seed(20260921+int(rate*100))
            active=(torch.rand(x.shape,generator=g)<rate).to(x.device)&attention
            xt=torch.where(active,model.mask_index,x)
            t=torch.full((x.shape[0],1),rate/(1-model.noise.eps),device=x.device)
            conditioning=model.noise(t)[0]
            output,unary=model._structured_head_output(xt,conditioning,active,force_no_grad_backbone=True)
            inf=objective.infer_structured_distribution(output,active)
            joint=-objective.structured_token_log_probability(output,unary,x,active,inference=inf)
            marginal=-objective.structured_marginal_token_log_probability(output,unary,x,active,inference=inf)
            base=-objective.factorized_token_log_probability(unary,x,active)
            structure=forest_stats(output,active)
            for b in range(x.shape[0]):
                count=int(active[b].sum());den=max(1,count)
                records.append(dict(step=step,source=source,example=b,mask_rate=rate,clean_tokens_sha256=tensor_sha(x[b]),active_mask_sha256=tensor_sha(active[b]),joint_nll_sum=float(joint[b]),marginal_nll_sum=float(marginal[b]),backbone_nll_sum=float(base[b]),joint_nll=float(joint[b])/den,backbone_gain=float(base[b]-joint[b])/den,dependence_gain=float(marginal[b]-joint[b])/den,**structure[b]))
    finally:
        for m,flag in flags:m.training=flag
    return records

class CampaignObserver(Callback):
    def __init__(self,variant_index,expected_steps=15000,namespace='training'):
        self.v=VARIANTS[int(variant_index)];self.expected_steps=int(expected_steps);self.namespace=namespace
        self.folder=ROOT/namespace/self.v['name'];self.folder.mkdir(parents=True,exist_ok=True)
        self.last_export=-1;self.last_probe=-1;self.probe=None
    def append(self,name,record):
        with (self.folder/name).open('a') as f:f.write(json.dumps(record)+'\n')
    def on_train_start(self,trainer,model):
        import dataloader
        with torch.random.fork_rng(devices=[model.device.index]):
            _,val=dataloader.get_dataloaders(model.config,model.tokenizer,skip_train=True)
            self.probe={k:v.detach().cpu() for k,v in next(iter(val)).items() if torch.is_tensor(v)}
        save(self.folder/'probe-identity.json',dict(source='First four examples from unchanged pinned WikiText validation loader',clean_tokens_sha256=tensor_sha(self.probe['input_ids']),rates=[.25,.5,.75,.9],fixed_mask_seed=20260921,scope='Small fixed diagnostic panel, not an independent test set'))
        self.observe(trainer,model)
    def on_before_optimizer_step(self,trainer,model,optimizer):
        if (trainer.global_step+1)%100:return
        sums={'topology':0.,'factors':0.,'shared_context':0.};counts={k:0 for k in sums}
        for n,p in model.structured_head.named_parameters():
            key='topology' if n.startswith(('edge_proposer','topology_')) else 'shared_context' if n.startswith(('hidden_norm','time_embedding')) else 'factors'
            if p.grad is not None:sums[key]+=float(p.grad.detach().float().square().sum());counts[key]+=1
        self.append('gradient-norms.jsonl',dict(step=trainer.global_step+1,l2_norm={k:math.sqrt(v) for k,v in sums.items()},parameters_with_grad=counts,learning_rates=[g['lr'] for g in optimizer.param_groups]))
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        step=trainer.global_step
        if step%50==0:
            m={k:float(v) for k,v in model._last_structured_metrics.items()}
            joint=m['loss']-float(model.structured_training_config.topology_weight)*m['topology_loss']
            self.append('loss-components.jsonl',dict(step=step,joint_nll=joint,backbone_gain=m['factorized_aux_nll']-joint,**m))
        if step%1000==0:self.observe(trainer,model)
    def observe(self,trainer,model):
        step=trainer.global_step
        if self.probe is not None and step!=self.last_probe:
            with torch.random.fork_rng(devices=[model.device.index]):
                for row in probe_batch(model,self.probe,step=step):self.append('fixed-probe.jsonl',row)
            self.last_probe=step
        if step==0 or step==self.last_export:return
        from scripts.export_structured_adapter import export_adapter
        export_root='exports' if self.namespace=='training' else self.namespace+'_exports'
        folder=ROOT/export_root/self.v['name']/f'step{step:06d}'
        if folder.exists():raise RuntimeError('Refusing to overwrite exported checkpoint '+str(folder))
        folder.mkdir(parents=True)
        full=folder/'export-source.ckpt';trainer.save_checkpoint(str(full))
        topology,factor,weight=ARMS[self.v['arm']]
        report=export_adapter(full,folder/'adapter.safetensors',folder/'adapter.manifest.json',expected_checkpoint_sha256=sha(full),expected_global_step=step,control_identity=self.v['arm'],topology_mode=topology,factor_mode=factor,candidate_k=128,independent_mode=False,topology_weight=weight,expected_frozen_backbone=model.backbone)
        save(folder/'export-report.json',report)
        # This temporary full checkpoint was created above in this campaign.
        # Keep the exported head; Lightning separately keeps current resume checkpoints.
        full.unlink();self.last_export=step
    def on_train_end(self,trainer,model):
        self.observe(trainer,model)
        assert trainer.global_step==self.expected_steps
        save(self.folder/'completed.json',dict(status='completed',global_step=trainer.global_step,backbone_frozen=not any(p.requires_grad for p in model.backbone.parameters())))
