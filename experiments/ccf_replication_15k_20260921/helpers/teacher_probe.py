import torch,math
import structured_training as training
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

