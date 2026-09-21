"""Smoke the original training entry point with no added training callback."""
import json
import subprocess
import sys
import torch
from pathlib import Path
from common import ARMS, STATE, STUDY, save


def difference(a, b):
    values = []
    for name, x in a.items():
        y = b[name]
        assert x.shape == y.shape and x.dtype == y.dtype
        if not torch.equal(x, y):
            values.append(dict(name=name, max_abs=float((x.double()-y.double()).abs().max()),
                               different_elements=int((x!=y).sum())))
    return values


old = Path(STATE['first_attempt_root'])
a, b = [torch.load(old/'gate/dynamic_fixed'/name/'checkpoints/last.ckpt', map_location='cpu')
        for name in ['plain', 'observed']]
save(STUDY/'gate/first-attempt-differences.json', dict(
    source=str(old), differences=difference(a['state_dict'], b['state_dict']),
    conclusion='Strict equality failed. The added observer is excluded from all production runs.'))
del a, b
results=[]
for index, arm in enumerate(ARMS):
    out=STUDY/'gate'/arm/'plain'
    subprocess.run([sys.executable,str(STUDY/'train_phase.py'),str(index),'1000','3',str(out),'-','plain','gate'],check=True)
    cache=json.loads((out/'cache-check.json').read_text())
    subprocess.run([sys.executable,str(STUDY/'export.py'),str(index),'3',str(out/'checkpoints/last.ckpt'),
                    str(STUDY/'gate/exports'/arm/'step000003')],check=True)
    subprocess.run([sys.executable,str(STUDY/'evaluate.py'),str(index),'gate'],check=True)
    results.append(dict(arm=arm,training_cache=cache['final_cache'],cold_start=cache['cold_start'],
                        original_training_callbacks_only=True,export_and_generation_smoke=True))
    save(STUDY/'gate/progress.json',results)

# A second untouched DF run measures repeatability without the rejected observer.
arm='dynamic_fixed';repeat=STUDY/'gate'/arm/'plain_repeat'
subprocess.run([sys.executable,str(STUDY/'train_phase.py'),'2','1000','3',str(repeat),'-','plain','gate'],check=True)
a,b=[torch.load(STUDY/'gate'/arm/name/'checkpoints/last.ckpt',map_location='cpu') for name in ['plain','plain_repeat']]
save(STUDY/'gate/plain-repeat-differences.json',dict(differences=difference(a['state_dict'],b['state_dict']),
    conclusion='Measured repeatability of two original training calls on this allocation; no deterministic-algorithm setting changed.'))
del a,b

# Check original checkpoint/optimizer resume and cold BF16 cache in a new process.
out=STUDY/'gate/dynamic_dynamic/resume_plain'
subprocess.run([sys.executable,str(STUDY/'train_phase.py'),'3','3000','6',str(out),
    str(STUDY/'gate/dynamic_dynamic/plain/checkpoints/last.ckpt'),'plain','gate'],check=True)
assert json.loads((out/'cache-check.json').read_text())['global_step']==6
save(STUDY/'gate/passed.json',dict(status='passed',arms=results,original_resume_smoke=True,
    added_production_training_callbacks=False,training_cache='BF16',generation_cache='FP32',
    scope='Original-path runtime smoke, not proof of bitwise determinism or replication efficacy',
    prior_failed_gate_preserved=str(old)))
