from campaign_observer import *
from campaign_common import *
from teacher_probe import teacher_diagnostic

class WeightObserver(CampaignObserver):
    def observe(self,trainer,model):
        step=trainer.global_step
        if self.probe is not None and step!=self.last_probe:
            flags=[(m,m.training) for m in model.modules()];model.eval()
            try:
                with torch.no_grad(),torch.random.fork_rng(devices=[model.device.index]):
                    for row in probe_batch(model,self.probe,step=step):self.append('fixed-probe.jsonl',row)
                    for rate in (.25,.5,.75,.9):
                        xall=self.probe['input_ids'];attall=self.probe['attention_mask'].bool()
                        g=torch.Generator().manual_seed(20260921+int(rate*100))
                        activeall=(torch.rand(xall.shape,generator=g)<rate)&attall
                        for b in range(len(xall)):
                            x=xall[b:b+1].to(model.device);active=activeall[b:b+1].to(model.device)
                            xt=torch.where(active,model.mask_index,x)
                            t=torch.full((1,1),rate/(1-model.noise.eps),device=model.device);condition=model.noise(t)[0]
                            output,unary=model._structured_head_output(xt,condition,active,force_no_grad_backbone=True)
                            teacher=teacher_diagnostic(model,output,unary,x,xt,active,condition,510000+b)
                            self.append('teacher-probe.jsonl',dict(step=step,mask_rate=rate,example=b,teacher=teacher))
            finally:
                for m,flag in flags:m.training=flag
            self.last_probe=step
        if step!=self.expected_steps or step==self.last_export:return
        from scripts.export_structured_adapter import export_adapter
        folder=ROOT/(self.namespace+'_exports')/self.v['name']/f'step{step:06d}'
        folder.mkdir(parents=True,exist_ok=False)
        full=folder/'export-source.ckpt';trainer.save_checkpoint(str(full))
        topology,factor,_=ARMS[self.v['arm']];weight=float(model.structured_training_config.topology_weight)
        report=export_adapter(full,folder/'adapter.safetensors',folder/'adapter.manifest.json',expected_checkpoint_sha256=sha(full),expected_global_step=step,control_identity=self.v['arm'],topology_mode=topology,factor_mode=factor,candidate_k=128,independent_mode=False,topology_weight=weight,expected_frozen_backbone=model.backbone)
        save(folder/'export-report.json',report);full.unlink();self.last_export=step
