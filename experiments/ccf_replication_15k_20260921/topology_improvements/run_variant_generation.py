"""Generate with an explicitly recorded process-local topology variant."""
import sys,json
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code))
from campaign_common import save,sha
from campaign_observer import forest_stats
from topology_variants import generation_variant,VARIANTS
from dataclasses import asdict
from models import structured_decoder as decoder
from scripts import run_generation_pilot as pilot
import torch
args_file=Path(sys.argv[2]);variant=sys.argv[3];trace=Path(sys.argv[4]);args=json.loads(args_file.read_text())
assert variant in VARIANTS
trace.parent.mkdir(parents=True,exist_ok=True)
if trace.exists():raise RuntimeError('Refusing to overwrite trace')
save(trace.parent/'variant.json',{'variant':variant,'settings':asdict(VARIANTS[variant]),'architecture_change':'Graph construction rules only; no new parameters. Dense-teacher variant also changes training supervision, never inference inputs.','weights_modified_by_this_process':False,'uses_clean_targets_in_generation':False,'base_model_code':str(code),'variant_module_sha256':sha(Path(__file__).with_name('topology_variants.py')),'runner_sha256':sha(Path(__file__)),'arguments_sha256':sha(args_file),'trace_affects_timing':True,'baseline_precision_policy':'Unchanged campaign policy; CCF fp32-normalized logits versus native MDLM path remains disclosed.'})
with generation_variant(variant):
    original=decoder.ContextualCouplingForestHead.forward;counter=[0]
    def traced(self,hidden_states,unary_logits,timestep,active_mask=None,**kwargs):
        output=original(self,hidden_states,unary_logits,timestep,active_mask,**kwargs)
        if active_mask is None:active_mask=torch.ones(hidden_states.shape[:2],device=hidden_states.device,dtype=torch.bool)
        rows=forest_stats(output,active_mask)
        with trace.open('a') as f:
            for b,row in enumerate(rows):
                nodes=active_mask[b].nonzero().flatten().tolist();chain=set(zip(nodes[:-1],nodes[1:]))
                proposals=output.proposal_edge_index[b,output.proposal_edge_mask[b]].detach().cpu().tolist()
                selected={tuple(e) for e in output.edge_index[b,output.edge_mask[b]].detach().cpu().tolist()}
                proposed={tuple(e) for e in proposals};anchors=output.anchor_indices[b].detach().cpu().tolist()
                row.update(call_index=counter[0],batch_index=b,variant=variant,length=hidden_states.shape[1],active_fraction=len(nodes)/hidden_states.shape[1],unique_anchors=len(set(anchors)) if nodes else 0,anchor_slots=len(anchors),proposal_count=len(proposals),unique_proposals=len(proposed),chain_proposed_fraction=len(chain&proposed)/max(1,len(chain)),chain_selected_fraction=len(chain&selected)/max(1,len(chain)))
                f.write(json.dumps(row)+'\n')
        counter[0]+=1
        return output
    decoder.ContextualCouplingForestHead.forward=traced
    try:result=pilot.main(args)
    finally:decoder.ContextualCouplingForestHead.forward=original
raise SystemExit(result)
