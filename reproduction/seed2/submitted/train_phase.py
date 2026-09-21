import os
import runpy
import sys
from pathlib import Path
from common import ARMS, STATE, STUDY, save, sha

index, phase, target = map(int, sys.argv[1:4])
folder = Path(sys.argv[4])
resume = sys.argv[5]
observed = sys.argv[6] == 'observed'
gate = sys.argv[7] == 'gate'
folder.mkdir(parents=True, exist_ok=False)
config_name = str(phase) + '_' + ARMS[index]
args = ['--config-path', str(STUDY / 'configs'), '--config-name', config_name,
        'seed=2', 'trainer.max_steps=' + str(target),
        'checkpointing.save_dir=' + str(folder), 'hydra.run.dir=' + str(folder / 'hydra'),
        'checkpointing.resume_from_ckpt=' + ('false' if resume == '-' else 'true')]
if resume != '-':
    assert Path(resume).is_file()
    args += ['checkpointing.resume_ckpt_path=' + resume]
if gate:
    args += ['callbacks.checkpoint_every_n_steps.every_n_train_steps=3']
if observed:
    args += ['++callbacks.passive._target_=passive_observer.PassiveObserver',
             '++callbacks.passive.folder=' + str(folder),
             '++callbacks.passive.expected_steps=' + str(target)]
save(folder / 'request.json', dict(arm=ARMS[index], historical_phase=phase,
    target_step=target, train_seed=2, args=args, source=STATE['train_code'],
    resume=resume, resume_sha256=None if resume == '-' else sha(resume), gate=gate))
sys.path.insert(0, STATE['train_code'])
os.chdir(folder)
if gate:
    # Test-only read assertions around the complete original fit call.
    import lightning as L
    import torch
    original_fit = L.Trainer.fit
    def checked_fit(trainer, model, *fit_args, **fit_kwargs):
        assert model.backbone.rotary_emb.seq_len_cached is None
        result = original_fit(trainer, model, *fit_args, **fit_kwargs)
        assert model.backbone.rotary_emb.cos_cached.dtype == torch.bfloat16
        assert int(trainer.global_step) == target
        assert not any(p.requires_grad for p in model.backbone.parameters())
        save(folder / 'cache-check.json', dict(cold_start=True, final_cache='torch.bfloat16',
             global_step=int(trainer.global_step), loss_metrics={k:float(v) for k,v in model._last_structured_metrics.items()}))
        return result
    L.Trainer.fit = checked_fit
sys.argv = ['main.py', *args]
runpy.run_path(str(Path(STATE['train_code']) / 'main.py'), run_name='__main__')
