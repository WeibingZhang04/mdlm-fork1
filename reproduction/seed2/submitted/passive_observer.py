"""Read existing training state only: no model forwards, probes or RNG draws."""
import json
import torch
from lightning.pytorch.callbacks import Callback
from common import save, tensor_sha


class PassiveObserver(Callback):
    def __init__(self, folder, expected_steps):
        from pathlib import Path
        self.folder = Path(folder)
        self.expected_steps = int(expected_steps)

    def append(self, name, row):
        with (self.folder / name).open('a') as f:
            f.write(json.dumps(row) + '\n')

    def on_train_start(self, trainer, model):
        rotary = model.backbone.rotary_emb
        assert rotary.seq_len_cached is None, 'Unexpected startup forward/cache initialization'
        assert not any(p.requires_grad for p in model.backbone.parameters())
        self.start_step = int(trainer.global_step)
        save(self.folder / 'startup.json', dict(
            source_step=self.start_step, expected_steps=self.expected_steps,
            seed=int(model.config.seed), cache_empty=True,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            initial_head={k: tensor_sha(v) for k, v in model.structured_head.state_dict().items()}))

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        rotary = model.backbone.rotary_emb
        if int(trainer.global_step) == self.start_step:
            assert rotary.seq_len_cached is None
        else:
            assert rotary.cos_cached.dtype == torch.bfloat16
        if int(trainer.global_step) < self.start_step + 3:
            self.append('input-identity.jsonl', dict(
                step=int(trainer.global_step), tokens=tensor_sha(batch['input_ids']),
                attention=tensor_sha(batch['attention_mask']),
                cpu_rng=tensor_sha(torch.get_rng_state()),
                cuda_rng=tensor_sha(torch.cuda.get_rng_state(model.device)),
                corruption_rng=tensor_sha(model._structured_training_corruption_generator.get_state()),
                topology_rng=tensor_sha(model._structured_training_topology_generator.get_state())))

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        rotary = model.backbone.rotary_emb
        assert rotary.cos_cached.dtype == torch.bfloat16, 'Training cache precision changed'
        step = int(trainer.global_step)
        if step == self.start_step + 1:
            save(self.folder / 'first-cache.json', dict(step=step, dtype=str(rotary.cos_cached.dtype),
                 cos_sha256=tensor_sha(rotary.cos_cached), sin_sha256=tensor_sha(rotary.sin_cached)))
        if step % 10 == 0 or step <= self.start_step + 3:
            metrics = {k: float(v) for k, v in model._last_structured_metrics.items()}
            joint = metrics['loss'] - float(model.structured_training_config.topology_weight) * metrics['topology_loss']
            self.append('loss-components.jsonl', dict(step=step, joint_nll=joint,
                learning_rates=[g['lr'] for g in trainer.optimizers[0].param_groups], **metrics))

    def on_train_end(self, trainer, model):
        assert int(trainer.global_step) == self.expected_steps
        assert model.backbone.rotary_emb.cos_cached.dtype == torch.bfloat16
        save(self.folder / 'completed.json', dict(global_step=int(trainer.global_step),
             cache_dtype='torch.bfloat16', backbone_frozen=True))
