"""Explicit cache-initialization intervention; model architecture is unchanged."""
import torch,hashlib
from campaign_observer import CampaignObserver,tensor_sha
from campaign_common import save

def head_digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.structured_head.state_dict().items()):
        h.update(name.encode());h.update(tensor_sha(t).encode())
    return h.hexdigest()

class PrecisionObserver(CampaignObserver):
    def __init__(self,cache_precision,**kwargs):
        super().__init__(**kwargs)
        if cache_precision not in ('bf16','fp32'):raise ValueError(cache_precision)
        self.cache_precision=cache_precision
        self.expected_dtype=torch.bfloat16 if cache_precision=='bf16' else torch.float32
    def on_train_start(self,trainer,model):
        self.source_step=int(trainer.global_step)
        rotary=model.backbone.rotary_emb
        assert rotary.seq_len_cached is None,'An earlier forward changed the startup state'
        before=head_digest(model)
        cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(model.device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.cache_precision=='bf16'):
            rotary(torch.empty(1,model.config.model.length,1,device=model.device))
        assert rotary.cos_cached.dtype==self.expected_dtype
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state(model.device))
        assert before==head_digest(model)
        save(self.folder/'initial-state.json',dict(source_step=self.source_step,cache_precision=self.cache_precision,cos_dtype=str(rotary.cos_cached.dtype),cos_sha256=tensor_sha(rotary.cos_cached.float()),initial_head_sha256=before,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cache_initialization_preserves_rng_and_parameters=True))
        super().on_train_start(trainer,model)
        assert rotary.cos_cached.dtype==self.expected_dtype
    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        assert model.backbone.rotary_emb.cos_cached.dtype==self.expected_dtype
        if trainer.global_step<self.source_step+3:
            self.append('input-identity.jsonl',dict(step=trainer.global_step,tokens_sha256=tensor_sha(batch['input_ids']),attention_sha256=tensor_sha(batch['attention_mask']),cpu_rng_sha256=tensor_sha(torch.get_rng_state()),cuda_rng_sha256=tensor_sha(torch.cuda.get_rng_state(model.device)),corruption_rng_sha256=tensor_sha(model._structured_training_corruption_generator.get_state()),topology_rng_sha256=tensor_sha(model._structured_training_topology_generator.get_state())))
