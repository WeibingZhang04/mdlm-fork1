"""DO NOT touch other people's files. DO NOT touch other people's jobs.
Do not interfere with other people's processes. Inference-only local hooks.
Uses existing structured_decoder forest selection and validation; see README.md.
"""
from contextlib import contextmanager
import hashlib
import json
import random
from unittest.mock import patch


def geometry(active, edges, cap):
    parent = {n:n for n in active}; size = dict.fromkeys(active, 1); degree = dict.fromkeys(active, 0)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    seen = set()
    for u,v in edges:
        assert u in parent and v in parent and u < v and (u,v) not in seen
        seen.add((u,v)); a,b = find(u),find(v); assert a != b
        assert cap <= 0 or size[a]+size[b] <= cap
        parent[b] = a; size[a] += size[b]; degree[u] += 1; degree[v] += 1
    sizes = sorted(size[n] for n in active if find(n)==n)
    return dict(active=len(active), edges=len(edges), components=len(sizes),
                component_sizes=sizes, degree_multiset=sorted(degree.values()),
                isolated=sum(d==0 for d in degree.values()),
                mean_edge_span=sum(v-u for u,v in edges)/len(edges) if edges else 0)


def relabel(active, edges, seed):
    perm = list(active); random.Random(seed).shuffle(perm)
    mapping = dict(zip(active,perm))
    return [tuple(sorted((mapping[u],mapping[v]))) for u,v in edges]


@contextmanager
def apply(variant, sample_seed, trace):
    import torch
    import models.structured_decoder as decoder
    cls = decoder.ContextualCouplingForestHead; original = cls._selected_edges
    count = 0
    def selected(self, **kw):
        nonlocal count
        assert self.topology_mode == self.factor_mode == 'dynamic'
        assert not self.independent_mode and self.component_size_cap == 32
        native_edges, native_mask, native_scores = original(self, **kw)
        edges, mask, scores = native_edges, native_mask, native_scores
        if variant == 'chain':
            edges, mask = decoder._fixed_chain_edges(kw['active_mask'], self.component_size_cap)
        elif variant == 'random':
            edges = native_edges.clone()
            for b in range(len(edges)):
                active = kw['active_mask'][b].nonzero().flatten().tolist()
                old = [tuple(e) for e in native_edges[b,native_mask[b]].tolist()]
                seed = int.from_bytes(hashlib.sha256(f'{sample_seed}:{count}:{b}:graph-v1'.encode()).digest()[:8], 'big')
                new = relabel(active, old, seed)
                if new: edges[b,native_mask[b]] = torch.tensor(new,device=edges.device)
        elif variant not in ('native','marginal'):
            raise ValueError(variant)
        if variant in ('chain','random'):
            scores = self.edge_proposer.score_edges(kw['topology_context'],edges,mask).masked_fill(~mask,0)
        for b in range(len(edges)):
            active = kw['active_mask'][b].nonzero().flatten().tolist()
            old = geometry(active,[tuple(e) for e in native_edges[b,native_mask[b]].tolist()],self.component_size_cap)
            new = geometry(active,[tuple(e) for e in edges[b,mask[b]].tolist()],self.component_size_cap)
            if variant == 'random':
                assert all(new[k]==old[k] for k in ('edges','component_sizes','degree_multiset','isolated'))
            # Compact statistics for each sample/denoising call, not model inputs.
            trace.write(json.dumps(dict(sample_seed=sample_seed,forward=count,batch=b,variant=variant,
                **{k:v for k,v in new.items() if k not in ('component_sizes','degree_multiset')},
                native_edges=old['edges'],native_isolated=old['isolated']))+'\n')
        count += 1
        return edges,mask,scores
    with patch.object(cls,'_selected_edges',selected): yield


def self_test():
    import torch
    import io
    from models.structured_decoder import ContextualCouplingForestHead
    torch.manual_seed(19)
    h = ContextualCouplingForestHead(16,31,top_k=8,rank=4,component_size_cap=32).eval()
    hidden=torch.randn(1,9,16); logits=torch.randn(1,9,31); t=torch.tensor([.5])
    active=torch.tensor([[True,False,True,True,False,True,False,True,True]])
    state={k:v.clone() for k,v in h.state_dict().items()}
    reference=h(hidden,logits,t,active); rng=torch.get_rng_state().clone()
    for variant in ('native','chain','random','marginal'):
        with apply(variant,12,io.StringIO()): result=h(hidden,logits,t,active)
        assert torch.equal(rng,torch.get_rng_state())
        assert torch.equal(reference.candidate_ids,result.candidate_ids)
        assert torch.equal(reference.unary_log_potentials,result.unary_log_potentials)
        if variant in ('native','marginal'):
            assert torch.equal(reference.edge_index,result.edge_index)
            assert torch.equal(reference.pair_left_factors,result.pair_left_factors)
        if variant=='chain':
            assert result.edge_index[0,result.edge_mask[0]].tolist()==[[0,2],[2,3],[3,5],[5,7],[7,8]]
        assert all(torch.equal(v,h.state_dict()[k]) for k,v in state.items())
    active=list(range(90)); chain=[(i,i+1) for i in range(89) if (i+1)%30]
    assert geometry(active,chain,32)['component_sizes']==[30,30,30]
    for seed in range(20):
        new=relabel(active,chain,seed)
        a,b=geometry(active,chain,32),geometry(active,new,32)
        assert all(a[k]==b[k] for k in ('edges','component_sizes','degree_multiset','isolated'))
    print('Topology invariants, unchanged weights/unaries, chain gaps, RNG preservation: passed')
