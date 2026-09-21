import csv
import subprocess
import sys
import torch
from common import ARMS, PHASES, STUDY, save

index = int(sys.argv[1])
assert (STUDY / 'gate/passed.json').is_file()
base = STUDY / 'training' / ARMS[index]
assert not base.exists(), 'Refusing to overwrite a previous training attempt'
previous = '-'
for phase in PHASES:
    folder = base / ('to' + str(phase))
    subprocess.run([sys.executable, str(STUDY / 'train_phase.py'), str(index), str(phase),
                   str(phase), str(folder), previous, 'plain', 'production'], check=True)
    previous = str(folder / 'checkpoints/last.ckpt')
    checkpoint = torch.load(previous, map_location='cpu')
    assert int(checkpoint['global_step']) == phase
    del checkpoint
    # Extract existing CSV logs after the original training process exits.
    rows = []
    for path in sorted(folder.glob('lightning_logs/version_*/metrics.csv')):
        with path.open() as f:
            for row in csv.DictReader(f):
                rows.append({k:float(v) for k,v in row.items() if v})
    assert rows, 'Original CSV loss/validation logs missing'
    save(folder / 'original-metrics.json', rows)
    save(folder / 'completed.json', dict(global_step=phase, added_training_callbacks=False,
         metrics_source='Original Lightning CSV logger; extracted after training process exited'))
    for step in ({6000: [5000, 6000], 8000: [7000, 8000]}.get(phase, [])):
        checkpoint = folder / 'checkpoints' / ('0-' + str(step) + '.ckpt')
        assert checkpoint.is_file(), str(checkpoint)
        subprocess.run([sys.executable, str(STUDY / 'export.py'), str(index), str(step),
            str(checkpoint), str(STUDY / 'exports' / ARMS[index] / ('step%06d' % step))], check=True)
save(base / 'completed.json', dict(global_step=8000, training_seed=2,
    restart_boundaries=PHASES[:-1], evaluated_checkpoints=[5000, 6000, 7000]))
