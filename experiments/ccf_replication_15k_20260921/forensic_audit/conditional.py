"""Paired validation diagnostics: marginal calibration versus dependence gain."""
import sys,gc,json,math,os
from pathlib import Path
from campaign_common import *
from common import STUDY,cases
sys.path.insert(0,STATE['code'])
import torch,dataloader,diffusion
import structured_objective as objective
from scripts import run_generation_pilot as pilot
from campaign_observer import tensor_sha,forest_stats
folder=STUDY/os.environ.get('CCF_CONDITIONAL_OUTPUT','conditional');folder.mkdir(exist_ok=False)
record_path=folder/'records.jsonl';checks=[]
torch.set_grad_enabled(False)
for label,v,exp in cases():
    args=pilot._parse_args(generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',folder/label,score=False))
    cfg=pilot._compose_config(args);tok=dataloader.get_tokenizer(cfg);m=diffusion.Diffusion(cfg,tokenizer=tok).cuda().eval()
    _,valid=dataloader.get_dataloaders(cfg,tok,skip_train=True)
    texts=[]
    for batch in valid:
        for i in range(batch['input_ids'].shape[0]):
            texts.append({k:t[i:i+1].cuda() for k,t in batch.items() if torch.is_tensor(t)})
            if len(texts)==16:break
        if len(texts)==16:break
    for sample,batch in enumerate(texts):
        x=batch['input_ids'];attention=batch['attention_mask'].bool()
        for rate in (.25,.5,.75,.95):
            g=torch.Generator().manual_seed(260001+sample*1000+int(rate*100))
            active=(torch.rand(x.shape,generator=g)<rate).cuda()&attention;xt=torch.where(active,m.mask_index,x)
            t=torch.full((1,1),rate/(1-m.noise.eps),device=m.device);conditioning=m.noise(t)[0]
            h,u=m._structured_backbone_output(xt,conditioning,True)
            assert m.backbone.rotary_emb.cos_cached.dtype==torch.float32
            o=m.structured_head(h,u,conditioning[:,0],active)
            inf=objective.infer_structured_distribution(o,active)
            qlog=inf.marginals.node_log_marginals[active];q=qlog.exp()
            blog=torch.log_softmax(o.unary_log_potentials[active].float(),-1);b=blog.exp()
            kl=torch.where(q>0,q*(qlog-blog),torch.zeros_like(q)).sum(-1)
            base=-objective.factorized_token_log_probability(u,x,active)
            joint=-objective.structured_token_log_probability(o,u,x,active,inference=inf)
            marginal=-objective.structured_marginal_token_log_probability(o,u,x,active,inference=inf)
            full_log=torch.log_softmax(u[active].float(),-1)
            def entropy(logp):return -torch.where(torch.isfinite(logp),logp.exp()*logp,torch.zeros_like(logp)).sum(-1)
            full_H=entropy(full_log);base_compressed_H=entropy(blog)
            tail_H=((full_H-base_compressed_H)/b[:,-1].clamp_min(1e-10)).clamp_min(0)
            head_H=entropy(qlog)+q[:,-1]*tail_H
            row=dict(case=label,sample=sample,mask_rate=rate,clean_tokens_sha256=tensor_sha(x),mask_sha256=tensor_sha(active),active_tokens=int(active.sum()),backbone_nll_sum=float(base),joint_nll_sum=float(joint),marginal_nll_sum=float(marginal),marginal_kl_from_backbone_sum=float(kl.sum()),backbone_entropy_sum=float(full_H.sum()),head_marginal_entropy_sum=float(head_H.sum()),backbone_explicit_mass_sum=float((1-b[:,-1]).sum()),head_explicit_mass_sum=float((1-q[:,-1]).sum()),graph=forest_stats(o,active)[0])
            with record_path.open('a') as f:f.write(json.dumps(row)+'\n')
            if sample==0:
                neutral=m.structured_head(h,u,conditioning[:,0],active,independent_mode=True)
                ni=objective.infer_structured_distribution(neutral,active)
                maxerr=float((ni.marginals.node_log_marginals[active].exp()-b).abs().max())
                nll=-objective.structured_token_log_probability(neutral,u,x,active,inference=ni)
                nllerr=float((nll-base).abs()/active.sum())
                assert maxerr<1e-4 and nllerr<1e-4,(maxerr,nllerr)
                original=m._subs_parameterization
                try:
                    m._subs_parameterization=lambda logits,xt: original(logits.float(),xt)
                    native=m.forward(xt,conditioning)
                finally:m._subs_parameterization=original
                # Compare identical subtraction/logsumexp arithmetic. F.log_softmax
                # differs by a few FP32 ULPs and is not the native SUBS implementation.
                target_log=u-torch.logsumexp(u,dim=-1,keepdim=True)
                unaryerr=float((native[active].exp()-target_log[active].exp()).abs().max())
                assert unaryerr<1e-6,unaryerr
                checks.append(dict(case=label,mask_rate=rate,neutral_marginal_max_abs_error=maxerr,neutral_joint_nll_per_token_abs_error=nllerr,native_fp32_vs_structured_unary_max_abs_error=unaryerr))
    print(label,'64 contexts completed',flush=True)
    del m,texts,valid,batch,x,h,u,o,inf,qlog,q,blog,b,full_log;gc.collect();torch.cuda.empty_cache()
save(folder/'passed.json',dict(status='passed',cases=6,examples_per_case=16,mask_rates=[.25,.5,.75,.95],scope='Fixed validation diagnostic panel, not an untouched test set',neutral_and_fp32_baseline_checks=checks))
