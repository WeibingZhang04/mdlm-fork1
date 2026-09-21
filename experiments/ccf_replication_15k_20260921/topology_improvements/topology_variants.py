"""Explicit experimental topology overrides; baseline source and defaults stay intact.

No new learned parameters. Graph rules receive corrupted-context features and an
active mask only. Dense supervision consumes clean labels only in the training
loss, using the same single teacher reveal as the original objective.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import torch
from models import structured_decoder as decoder
import structured_training as training

@dataclass(frozen=True)
class Variant:
    active_chain: bool = False
    coverage_first: bool = False
    unique_anchors: bool = False
    local_window: object = None
    dense_teacher: bool = False

VARIANTS = {
    'native': Variant(),
    'active_chain': Variant(active_chain=True),
    'chain_coverage': Variant(active_chain=True, coverage_first=True),
    'chain_unique_anchors': Variant(active_chain=True, unique_anchors=True),
    'chain_no_absolute_local': Variant(active_chain=True, local_window=0),
    'chain_absolute_window4': Variant(active_chain=True, local_window=4),
    'chain_dense_teacher': Variant(active_chain=True, dense_teacher=True),
}
INFERENCE_VARIANTS = [k for k in VARIANTS if k != 'chain_dense_teacher']
TRAIN_VARIANTS = ['native', 'active_chain', 'chain_dense_teacher']

def distinct_anchors(logits, active):
    """Fixed slot order, best unused active position; recycle only after exhaustion.

    Deterministic and label-independent. With N<A nodes, duplicates are unavoidable;
    each round covers all N before reuse. This deliberately changes the selector.
    """
    batch, length, slots = logits.shape
    available = active.clone()
    selected = torch.zeros(batch, slots, dtype=torch.long, device=logits.device)
    for slot in range(slots):
        exhausted = ~available.any(1)
        available = torch.where(exhausted[:, None], active, available)
        chosen = logits[:, :, slot].masked_fill(~available, -torch.inf).argmax(1)
        selected[:, slot] = chosen
        available.scatter_(1, chosen[:, None], False)
    return selected.detach()

def append_missing_chain(proposer, context, active, result):
    edges, mask = decoder._fixed_chain_edges(active, 0)
    old_edges, old_mask = result[:2]
    length = active.shape[1]
    # Retain the original proposals exactly; avoid reweighting their teacher loss
    # by appending duplicate chain edges. Existing duplicates remain unchanged.
    for b in range(active.shape[0]):
        keys = old_edges[b, old_mask[b], 0] * length + old_edges[b, old_mask[b], 1]
        chain_keys = edges[b, :, 0] * length + edges[b, :, 1]
        mask[b] &= ~torch.isin(chain_keys, keys)
    scores = proposer.score_edges(context, edges, mask)
    return (torch.cat((result[0], edges), 1), torch.cat((result[1], mask), 1),
            torch.cat((result[2], scores), 1), *result[3:])

def replace_anchor_edges(proposer, context, active, result):
    anchors = distinct_anchors(result[3], active)
    batch, length = active.shape
    k = proposer.contextual_neighbors
    local_count = sum(length-offset for offset in range(1, min(proposer.local_window, length-1)+1))
    if not k:
        return (*result[:4], anchors, result[5])
    slots = result[5].topk(k, dim=-1).indices
    right = anchors.gather(1, slots.reshape(batch, -1)).reshape(batch, length, k)
    left = torch.arange(length, device=active.device)[None, :, None].expand_as(right)
    valid = active[:, :, None] & active.gather(1, right.reshape(batch, -1)).reshape_as(right) & left.ne(right)
    edges = torch.stack((torch.minimum(left,right), torch.maximum(left,right)), -1).reshape(batch,-1,2)
    edges = torch.cat((result[0][:,:local_count],edges),1)
    mask = torch.cat((result[1][:,:local_count],valid.reshape(batch,-1)),1)
    return (edges, mask, proposer.score_edges(context,edges,mask),result[3],anchors,result[5])

def coverage_kruskal(proposal_edge_index, proposal_scores, proposal_edge_mask,
                     active_mask, component_size_cap, min_edge_score):
    """Score-ordered matching, then isolated-node attachment, then bounded Kruskal.

    Every pass enforces endpoint activity, acyclicity, threshold and component cap.
    Coverage is encouraged, not guaranteed. It is NOT a maximum-spanning forest.
    """
    batch,length=active_mask.shape; width=max(0,length-1)
    out=torch.zeros(batch,width,dtype=torch.long,device=active_mask.device)
    valid=torch.zeros_like(out,dtype=torch.bool)
    edges=proposal_edge_index.detach().cpu(); scores=proposal_scores.detach().float().cpu()
    masks=proposal_edge_mask.detach().cpu(); active=active_mask.detach().cpu()
    for b in range(batch):
        parent=list(range(length)); size=[int(a) for a in active[b]]; degree=[0]*length; chosen=[]
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        slots=masks[b].nonzero().flatten()
        order=slots[torch.argsort(scores[b,slots],descending=True,stable=True)].tolist()
        for phase in range(3):
            for s in order:
                if min_edge_score is not None and float(scores[b,s])<min_edge_score:break
                i,j=edges[b,s].tolist()
                if not active[b,i] or not active[b,j]:continue
                if phase==0 and (degree[i] or degree[j]):continue
                if phase==1 and degree[i] and degree[j]:continue
                a,c=find(i),find(j)
                if a==c:continue
                if component_size_cap>0 and size[a]+size[c]>component_size_cap:continue
                if size[a]<size[c] or (size[a]==size[c] and a>c):a,c=c,a
                parent[c]=a;size[a]+=size[c];degree[i]+=1;degree[j]+=1;chosen.append(s)
        if chosen:
            out[b,:len(chosen)]=torch.tensor(chosen,device=out.device);valid[b,:len(chosen)]=True
    return out.detach(),valid.detach()

def dense_teacher_output(output, active_mask, source_positions):
    """Score one source-to-all-active star from original student context.

    Does not insert these training-only edges into the sampled forest. The source
    is mask-sampled by the original trainer before the teacher reveal. O(B L D).
    """
    proposer,context=output._experimental_student_context
    batch,length=active_mask.shape
    nodes=torch.arange(length,device=active_mask.device)[None,:].expand(batch,-1)
    source=source_positions.clamp_min(0)[:,None].expand_as(nodes)
    edges=torch.stack((torch.minimum(source,nodes),torch.maximum(source,nodes)),-1)
    mask=active_mask & source_positions[:,None].ge(0) & nodes.ne(source)
    scores=proposer.score_edges(context,edges,mask)
    return replace(output,proposal_edge_index=edges,proposal_edge_mask=mask,proposal_scores=scores)

@contextmanager
def use_variant(name):
    """Process-local override, restored even after exceptions; no weight mutation."""
    variant=VARIANTS[name]
    if name=='native':
        yield variant
        return
    original_prop=decoder.SparseEdgeProposer.forward
    original_head=decoder.ContextualCouplingForestHead.forward
    original_select=decoder._bounded_kruskal_indices
    original_loss=training.gold_reveal_influence_topology_loss
    pending={}
    def proposer_forward(self,node_context,active_mask):
        window=self.local_window
        try:
            if variant.local_window is not None:self.local_window=variant.local_window
            result=original_prop(self,node_context,active_mask)
            if variant.unique_anchors:result=replace_anchor_edges(self,node_context,active_mask,result)
            if variant.active_chain:result=append_missing_chain(self,node_context,active_mask,result)
        finally:self.local_window=window
        if variant.dense_teacher:pending[id(self)]=node_context
        return result
    def head_forward(self,hidden_states,unary_logits,timestep,active_mask=None,**kwargs):
        output=original_head(self,hidden_states,unary_logits,timestep,active_mask,**kwargs)
        if variant.dense_teacher:
            output._experimental_student_context=(self.edge_proposer,pending.pop(id(self.edge_proposer)))
        return output
    def topology_loss(output,base_unary_logits,revealed_unary_logits,clean_tokens,active_mask,source_positions,**kwargs):
        student=dense_teacher_output(output,active_mask,source_positions)
        return original_loss(student,base_unary_logits,revealed_unary_logits,clean_tokens,active_mask,source_positions,**kwargs)
    decoder.SparseEdgeProposer.forward=proposer_forward
    decoder.ContextualCouplingForestHead.forward=head_forward
    if variant.coverage_first:decoder._bounded_kruskal_indices=coverage_kruskal
    if variant.dense_teacher:training.gold_reveal_influence_topology_loss=topology_loss
    try:yield variant
    finally:
        decoder.SparseEdgeProposer.forward=original_prop
        decoder.ContextualCouplingForestHead.forward=original_head
        decoder._bounded_kruskal_indices=original_select
        training.gold_reveal_influence_topology_loss=original_loss
        pending.clear()


@contextmanager
def generation_variant(name):
    """Install audited sampling optimizations before the graph-rule override.

    The optimization constructs a faster native Kruskal implementation by source
    inspection. It must see the native function; coverage-first selection then
    explicitly replaces it for this variant and restores it on exit.
    """
    from scripts.audit_ccf_sampling_v4 import experiment
    with experiment('level_draws'), use_variant(name) as variant:
        yield variant
