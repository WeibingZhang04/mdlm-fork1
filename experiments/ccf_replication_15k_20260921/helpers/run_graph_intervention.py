"""Audited inference-only graph interventions. Never edits model weights or source."""
import sys,json,contextlib
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code))
from campaign_common import save,sha
from campaign_observer import forest_stats
from scripts import run_generation_pilot as pilot
from scripts.audit_ccf_sampling_v4 import experiment
from models.structured_decoder import ContextualCouplingForestHead,SparseEdgeProposer,_fixed_chain_edges
import torch
args_file=Path(sys.argv[2]);kind=sys.argv[3];trace=Path(sys.argv[4]);args=json.loads(args_file.read_text())
assert kind in ('native','fixed_graph','active_chain_proposals')
trace.parent.mkdir(parents=True,exist_ok=True)
if trace.exists():raise RuntimeError('Refusing to overwrite graph trace')
orig_head=ContextualCouplingForestHead.forward;orig_prop=SparseEdgeProposer.forward
counter=0

def proposal_forward(self,context,active):
    result=orig_prop(self,context,active)
    edges,mask=_fixed_chain_edges(active,0)
    scores=self.score_edges(context,edges,mask)
    return (torch.cat((result[0],edges),1),torch.cat((result[1],mask),1),torch.cat((result[2],scores),1),*result[3:])

def head_forward(self,hidden,unary,timestep,active_mask=None,**kwargs):
    global counter
    if kind=='fixed_graph':kwargs['topology_mode']='fixed'
    output=orig_head(self,hidden,unary,timestep,active_mask,**kwargs)
    if active_mask is None:active_mask=torch.ones(hidden.shape[:2],device=hidden.device,dtype=torch.bool)
    stats=forest_stats(output,active_mask)
    for b,row in enumerate(stats):
        active=active_mask[b].nonzero().flatten().tolist();chain=set(zip(active[:-1],active[1:]))
        proposals=output.proposal_edge_index[b][output.proposal_edge_mask[b]].detach().cpu().tolist()
        edges=set(tuple(e) for e in output.edge_index[b][output.edge_mask[b]].detach().cpu().tolist())
        proposed=set(tuple(e) for e in proposals)
        anchors=output.anchor_indices[b].detach().cpu().tolist()
        row.update(call_index=counter,batch_index=b,sequence_length=hidden.shape[1],active_fraction=len(active)/hidden.shape[1],intervention=kind,unique_anchor_positions=len(set(anchors)),anchor_slots=len(anchors),proposal_count=len(proposals),unique_proposal_count=len(proposed),proposal_duplicate_fraction=1-len(proposed)/max(1,len(proposals)),active_chain_proposal_coverage=len(chain&proposed)/max(1,len(chain)),active_chain_selected_coverage=len(chain&edges)/max(1,len(chain)))
        with trace.open('a') as f:f.write(json.dumps(row)+'\n')
    counter+=1
    return output

save(trace.parent/'intervention.json',dict(intervention=kind,scope='Inference-only change to graph selection with checkpoint weights, candidates, factors and scorer unchanged. Raw pilot manifests identify the source checkpoint; this sidecar identifies the experimental graph intervention.',helper_sha256=sha(Path(__file__)),arguments_sha256=sha(args_file),trace_affects_timing=True))
ContextualCouplingForestHead.forward=head_forward
if kind=='active_chain_proposals':SparseEdgeProposer.forward=proposal_forward
try:
    with experiment('level_draws'):result=pilot.main(args)
finally:
    ContextualCouplingForestHead.forward=orig_head;SparseEdgeProposer.forward=orig_prop
raise SystemExit(result)
