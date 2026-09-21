import copy, unittest
import torch
from models import structured_decoder as d
import structured_training as st
from topology_variants import *

class TopologyVariantsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.head=d.ContextualCouplingForestHead(hidden_size=12,vocab_size=17,top_k=5,rank=3,time_embed_dim=8,topology_dim=10,local_window=2,num_anchor_slots=4,contextual_neighbors=2,component_size_cap=4)
        self.hidden=torch.randn(3,12,12);self.logits=torch.randn(3,12,17)
        self.active=torch.tensor([[1,0,0,0,1,0,0,0,0,1,0,1],[0]*12,[0,1,0,0,0,0,0,0,0,0,0,0]],dtype=torch.bool)
    def forward(self):
        return self.head(hidden_states=self.hidden,unary_logits=self.logits,timestep=torch.ones(3)*.5,active_mask=self.active)
    def test_native_bitwise_and_rng_and_restore(self):
        a=self.forward();rng=torch.get_rng_state().clone();weights=copy.deepcopy(self.head.state_dict());fn=d.SparseEdgeProposer.forward
        with use_variant('native'):b=self.forward()
        for name in ('candidate_ids','edge_index','edge_mask','pair_left_factors','proposal_scores'):self.assertTrue(torch.equal(getattr(a,name),getattr(b,name)))
        for variant in VARIANTS:
            with use_variant(variant):self.forward()
            self.assertIs(d.SparseEdgeProposer.forward,fn)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,v in weights.items():self.assertTrue(torch.equal(v,self.head.state_dict()[n]))
    def test_active_chain_covers_gaps_without_appended_duplicates(self):
        with use_variant('active_chain'):out=self.forward()
        for b in range(3):
            nodes=self.active[b].nonzero().flatten().tolist()
            edges=out.proposal_edge_index[b,out.proposal_edge_mask[b]].tolist()
            for pair in zip(nodes[:-1],nodes[1:]):self.assertIn(list(pair),edges)
            native=self.forward(); p=native.proposal_edge_mask.shape[1]
            added=out.proposal_edge_index[b,p:][out.proposal_edge_mask[b,p:]].tolist()
            old=native.proposal_edge_index[b,native.proposal_edge_mask[b]].tolist()
            self.assertTrue(all(e not in old for e in added))
    def test_all_graphs_forest_active_cap_and_vocabulary_unchanged(self):
        for variant in VARIANTS:
            with use_variant(variant):out=self.forward()
            self.assertTrue(torch.equal(out.candidate_ids,self.logits.topk(5,-1).indices))
            for b in range(3):
                parent=list(range(12));size=[1]*12
                def find(i):
                    while parent[i]!=i:i=parent[i]
                    return i
                for i,j in out.edge_index[b,out.edge_mask[b]].tolist():
                    self.assertTrue(self.active[b,i] and self.active[b,j]);a,c=find(i),find(j);self.assertNotEqual(a,c)
                    parent[c]=a;size[a]+=size[c];self.assertLessEqual(size[a],4)
    def test_unique_anchors_and_empty_rows(self):
        logits=torch.zeros(3,12,4)
        anchors=distinct_anchors(logits,self.active)
        self.assertEqual(anchors[0].tolist(),[0,4,9,11]);self.assertEqual(anchors[1].tolist(),[0]*4);self.assertEqual(anchors[2].tolist(),[1]*4)
    def test_coverage_matching_beats_hub_under_cap(self):
        edges=torch.tensor([[[0,1],[0,2],[0,3],[2,3]]]);scores=torch.tensor([[4.,3.,2.,1.]])
        mask=torch.ones(1,4,dtype=torch.bool);active=torch.ones(1,4,dtype=torch.bool)
        ids,valid=coverage_kruskal(edges,scores,mask,active,2,None)
        self.assertEqual(edges[0,ids[0,valid[0]]].tolist(),[[0,1],[2,3]])
        ids,valid=coverage_kruskal(edges,scores,mask,active,2,2.)
        self.assertEqual(edges[0,ids[0,valid[0]]].tolist(),[[0,1]])
    def test_dense_teacher_student_has_no_target_dependency_and_gradients(self):
        clean=torch.randint(17,(3,12));source=torch.tensor([0,-1,1])
        revealed=self.logits.clone();revealed[0,4,clean[0,4]]+=3
        with use_variant('chain_dense_teacher'):
            out=self.forward();dense=dense_teacher_output(out,self.active,source)
            self.assertEqual(int(dense.proposal_edge_mask.sum()),3)
            before=out.edge_index.clone()
            loss=st.gold_reveal_influence_topology_loss(out,self.logits, revealed,clean,self.active,source)
            loss.loss.backward()
            grads=[p.grad for p in self.head.edge_proposer.edge_scorer.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads));self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
            # Changing clean targets changes supervision, but never student scores
            # or graph at fixed corrupted context, mask, time and sampled source.
            permuted=(clean+1)%17
            out2=self.forward();dense2=dense_teacher_output(out2,self.active,source)
            st.gold_reveal_influence_topology_loss(out2,self.logits,revealed,permuted,self.active,source)
            self.assertTrue(torch.equal(before,out2.edge_index));self.assertTrue(torch.equal(dense.proposal_scores,dense2.proposal_scores))
        with use_variant('active_chain'):chain=self.forward()
        self.assertTrue(torch.equal(out.edge_index,chain.edge_index));self.assertTrue(torch.equal(out.edge_mask,chain.edge_mask))
    def test_generation_optimization_composes_with_every_variant(self):
        from scripts.audit_ccf_sampling_v4 import experiment
        original=d._bounded_kruskal_indices
        with experiment('level_draws'):baseline=self.forward()
        for name in VARIANTS:
            with generation_variant(name):
                output=self.forward()
                if name=='chain_coverage':self.assertIs(d._bounded_kruskal_indices,coverage_kruskal)
                if name=='native':
                    self.assertTrue(torch.equal(baseline.edge_index,output.edge_index))
                    self.assertTrue(torch.equal(baseline.edge_mask,output.edge_mask))
            self.assertIs(d._bounded_kruskal_indices,original)
    def test_restores_on_exception(self):
        fn=d.SparseEdgeProposer.forward
        with self.assertRaises(RuntimeError):
            with use_variant('chain_dense_teacher'):raise RuntimeError('intentional')
        self.assertIs(d.SparseEdgeProposer.forward,fn)

if __name__=='__main__':unittest.main()
