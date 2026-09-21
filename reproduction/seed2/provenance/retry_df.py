"""Replay an interrupted phase from its original boundary; preserve old files."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from common import ARMS, STUDY, save, sha

index = 2
arm = ARMS[index]
attempt = Path(os.environ['CCF_DF_RETRY'])
original = STUDY / 'training' / arm
assert (STUDY / 'gate/passed.json').is_file()
assert not (original / 'completed.json').exists()
assert json.loads((original / 'to1000/completed.json').read_text())['global_step'] == 1000
previous = str(original / 'to1000/checkpoints/0-1000.ckpt')
source = torch.load(previous, map_location='cpu')
assert int(source['global_step']) == 1000
del source
save(attempt / 'request.json', dict(
    arm=arm, source_checkpoint=previous, source_sha256=sha(previous), train_seed=2,
    policy='Replay interrupted 1k-to-3k phase from its original 1k boundary, then 3k-to-6k and 6k-to-8k. Preserve original unfinished phase. No additional mid-phase resume or scientific setting change.',
    prior_training_job='1557403_2'))
phases = []
for phase in [3000, 6000, 8000]:
    folder = attempt / ('to' + str(phase))
    subprocess.run([sys.executable, str(STUDY / 'train_phase.py'), str(index), str(phase),
                   str(phase), str(folder), previous, 'plain', 'production'], check=True)
    previous = str(folder / 'checkpoints/last.ckpt')
    checkpoint = torch.load(previous, map_location='cpu')
    assert int(checkpoint['global_step']) == phase
    del checkpoint
    rows = []
    for path in sorted(folder.glob('lightning_logs/version_*/metrics.csv')):
        with path.open() as f:
            rows.extend({k:float(v) for k,v in row.items() if v} for row in csv.DictReader(f))
    assert rows
    save(folder / 'original-metrics.json', rows)
    save(folder / 'completed.json', dict(global_step=phase, added_training_callbacks=False))
    phases.append(dict(target=phase, folder=str(folder)))
    save(original / 'retry-state.json', dict(attempt=str(attempt), completed_phases=phases))
    for step in {6000:[5000,6000],8000:[7000,8000]}.get(phase, []):
        ckpt = folder / 'checkpoints' / ('0-' + str(step) + '.ckpt')
        assert ckpt.is_file()
        subprocess.run([sys.executable, str(STUDY / 'export.py'), str(index), str(step),
                       str(ckpt), str(STUDY / 'exports' / arm / ('step%06d' % step))], check=True)
record = dict(global_step=8000, training_seed=2, restart_boundaries=[1000,3000,6000],
              evaluated_checkpoints=[5000,6000,7000], retry_attempt=str(attempt),
              original_completed_phase=str(original / 'to1000'), replacement_phases=phases)
save(attempt / 'completed.json', record)
save(original / 'completed.json', record)
