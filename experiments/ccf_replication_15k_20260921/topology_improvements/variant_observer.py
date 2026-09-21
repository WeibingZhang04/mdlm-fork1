from weight_observer import WeightObserver
from campaign_common import *
from topology_variants import VARIANTS as TOPOLOGY_VARIANTS,dense_teacher_output
from dataclasses import asdict
from pathlib import Path
from campaign_observer import tensor_sha
import torch,hashlib

class VariantObserver(WeightObserver):
    def __init__(self,topology_variant,**kwargs):
        super().__init__(**kwargs);self.topology_variant=topology_variant
    def on_train_start(self,trainer,model):
        h=hashlib.sha256()
        for name,tensor in sorted(model.structured_head.state_dict().items()):
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        save(self.folder/"initial-state.json",dict(global_step=trainer.global_step,head_sha256=h.hexdigest(),backbone_frozen=not any(p.requires_grad for p in model.backbone.parameters())))
        super().on_train_start(trainer,model)
    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        if trainer.global_step<6003:
            self.append("input-identity.jsonl",dict(step=trainer.global_step,clean_tokens_sha256=tensor_sha(batch["input_ids"]),cpu_rng_sha256=tensor_sha(torch.get_rng_state()),cuda_rng_sha256=tensor_sha(torch.cuda.get_rng_state(model.device)),corruption_rng_sha256=tensor_sha(model._structured_training_corruption_generator.get_state()),teacher_rng_sha256=tensor_sha(model._structured_training_topology_generator.get_state())))
    def observe(self,trainer,model):
        super().observe(trainer,model)
        save(self.folder/'topology-variant.json',dict(variant=self.topology_variant,settings=asdict(TOPOLOGY_VARIANTS[self.topology_variant]),module_sha256=sha(Path(__file__).with_name('topology_variants.py')),source_step=6000,target_step=self.expected_steps,scope='Explicit graph/objective override required for reproduction; raw adapter metadata describes base parameter architecture only.'))
        if self.last_export==trainer.global_step:
            folder=ROOT/(self.namespace+'_exports')/self.v['name']/f'step{trainer.global_step:06d}'
            save(folder/'topology-variant.json',dict(variant=self.topology_variant,settings=asdict(TOPOLOGY_VARIANTS[self.topology_variant]),module_sha256=sha(Path(__file__).with_name('topology_variants.py'))))
