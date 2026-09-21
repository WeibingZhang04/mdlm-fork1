import sys,json,dataclasses,gc,math
from pathlib import Path
from campaign_common import *
assert (ROOT/'gate/passed.json').is_file()
sys.path.insert(0,STATE['code'])
import torch
import dataloader,diffusion,structured_objective as objective,structured_training as training
from scripts import run_generation_pilot as pilot
from evaluation.causal_denoising import matched_permuted_forest_edges
from campaign_observer import forest_stats,tensor_sha
torch.set_grad_enabled(False)
folder=ROOT/'diagnostics/topology6k';folder.mkdir(parents=True,exist_ok=True)
output_path=folder/'records.jsonl'
if output_path.exists():raise RuntimeError('Diagnosis output already exists')

def distillation_components(student,teacher,mask):
    if int(mask.sum())<2:return dict(valid=False)
    s=student[mask].float().log_softmax(-1);t=teacher[mask].float().log_softmax(-1);p=t.exp()
    ce=-(p*s).sum();entropy=-(p*t).sum()
    return dict(valid=True,choices=int(mask.sum()),cross_entropy=float(ce),teacher_entropy=float(entropy),kl_to_teacher=float(ce-entropy),uniform_cross_entropy=math.log(int(mask.sum())),improvement_over_uniform=float(math.log(int(mask.sum()))-ce))

def teacher_diagnostic(model,output,unary,x,xt,active,conditioning,seed):
    ids=active[0].nonzero().flatten();g=torch.Generator().manual_seed(seed)
    if len(ids)<3:return {}
    source=int(ids[torch.randint(len(ids),(1,),generator=g)])
    revealed=xt.clone();revealed[0,source]=x[0,source]
    _,revealed_unary=model._structured_backbone_output(revealed,conditioning,True)
    influence=(training._gold_token_log_probability(revealed_unary,x)-training._gold_token_log_probability(unary,x)).abs()[0]/.25
    valid=active[0].clone();valid[source]=False
    edges=output.proposal_edge_index[0];left,right=edges[:,0],edges[:,1]
    other=torch.where(left==source,right,left)
    edge_mask=output.proposal_edge_mask[0]&((left==source)|(right==source))&valid[other]
    anchors=output.anchor_indices[0]
    return {'edge':distillation_components(output.proposal_scores[0],influence[other],edge_mask),'anchor':distillation_components(output.anchor_logits[0].logsumexp(-1),influence,valid),'slot':distillation_components(output.slot_logits[0,source],influence[anchors],valid[anchors]),'source_position':source,'scope':'Gold used only to score teacher targets; never supplied to the native head or generation'}

for v in VARIANTS:
    if v['arm'] not in ('fixed_dynamic','dynamic_dynamic'):continue
    n=v['name'];exp=ROOT/'gate/generation'/n
    args=pilot._parse_args(generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',folder/n,score=False))
    cfg=pilot._compose_config(args);tok=dataloader.get_tokenizer(cfg);m=diffusion.Diffusion(cfg,tokenizer=tok).cuda().eval()
    _,valid=dataloader.get_dataloaders(cfg,tok,skip_train=True)
    batches=[]
    for i,b in enumerate(valid):
        batches.append(b)
        if i==15:break
    for length in (128,512,1024):
        for rate in (.25,.5,.75,.9):
            for sample,batch in enumerate(batches):
                x=batch['input_ids'][:1,:length].cuda();att=batch['attention_mask'][:1,:length].cuda().bool()
                g=torch.Generator().manual_seed(410000+sample*1000+int(rate*100))
                active=(torch.rand(x.shape,generator=g)<rate).cuda()&att;xt=torch.where(active,m.mask_index,x)
                t=torch.full((1,1),rate/(1-m.noise.eps),device=m.device);condition=m.noise(t)[0]
                hidden,unary=m._structured_backbone_output(xt,condition,True)
                def head(**kw):return m.structured_head(hidden,unary,condition[:,0],active,**kw)
                native=head();N=max(1,int(active.sum()));base=-objective.factorized_token_log_probability(unary,x,active)
                teacher=teacher_diagnostic(m,native,unary,x,xt,active,condition,510000+sample) if v['arm']=='dynamic_dynamic' else None
                for intervention in ('native','fixed_graph','half_edges','cap8','cap128','permuted_edges'):
                    if intervention=='native':o=native
                    elif intervention=='fixed_graph':o=head(topology_mode='fixed')
                    elif intervention=='half_edges':
                        mask=native.edge_mask.clone();slots=mask[0].nonzero().flatten();perm=torch.randperm(len(slots),generator=torch.Generator().manual_seed(610000+sample))
                        mask[0,slots[perm[len(slots)//2:]].to(slots.device)]=False
                        o=head(fixed_edge_index=native.edge_index,fixed_edge_mask=mask)
                    elif intervention=='permuted_edges':
                        edges,mask,_=matched_permuted_forest_edges(native.edge_index,native.edge_mask,active,seed=710000+sample)
                        o=head(fixed_edge_index=edges,fixed_edge_mask=mask)
                    else:
                        cap=m.structured_head.component_size_cap
                        try:m.structured_head.component_size_cap=int(intervention[3:]);o=head()
                        finally:m.structured_head.component_size_cap=cap
                    inf=objective.infer_structured_distribution(o,active)
                    joint=-objective.structured_token_log_probability(o,unary,x,active,inference=inf)
                    marginal=-objective.structured_marginal_token_log_probability(o,unary,x,active,inference=inf)
                    record=dict(variant=n,checkpoint=6000,sequence_length=length,mask_rate=rate,sample=sample,intervention=intervention,clean_tokens_sha256=tensor_sha(x),active_mask_sha256=tensor_sha(active),joint_nll=float(joint[0])/N,marginal_nll=float(marginal[0])/N,backbone_nll=float(base[0])/N,dependence_gain=float(marginal[0]-joint[0])/N,backbone_gain=float(base[0]-joint[0])/N,teacher=teacher if intervention=='native' else None,**forest_stats(o,active)[0])
                    with output_path.open('a') as f:f.write(json.dumps(record)+'\n')
            print(n,length,rate,'done',flush=True)
    del m,hidden,unary,native,o,inf;gc.collect();torch.cuda.empty_cache()
save(folder/'completed.json',dict(status='completed',variants=6,samples_per_length_mask_cell=16,lengths=[128,512,1024],mask_rates=[.25,.5,.75,.9],scope='Paired conditional held-out diagnostics on unchanged WikiText validation source. Topology interventions are diagnostics, not retrained models or generation PPL results.'))
