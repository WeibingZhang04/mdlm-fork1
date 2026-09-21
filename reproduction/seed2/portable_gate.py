"""Fresh relocation smoke, separate from the preserved original gate history."""
import subprocess
import sys
from common import ARMS, STUDY, save

for index, arm in enumerate(ARMS):
    folder = STUDY / 'gate' / arm / 'plain'
    subprocess.run([sys.executable, str(STUDY / 'train_phase.py'), str(index), '1000', '3',
                    str(folder), '-', 'plain', 'gate'], check=True)
    subprocess.run([sys.executable, str(STUDY / 'export.py'), str(index), '3',
                    str(folder / 'checkpoints/last.ckpt'),
                    str(STUDY / 'gate/exports' / arm / 'step000003')], check=True)
    subprocess.run([sys.executable, str(STUDY / 'evaluate.py'), str(index), 'gate'], check=True)
subprocess.run([sys.executable, str(STUDY / 'train_phase.py'), '3', '3000', '6',
                str(STUDY / 'gate/dynamic_dynamic/resume'),
                str(STUDY / 'gate/dynamic_dynamic/plain/checkpoints/last.ckpt'),
                'plain', 'gate'], check=True)
save(STUDY / 'gate/passed.json', {
    'status': 'passed', 'arms': ARMS, 'resume_checked': True,
    'scope': 'Fresh portability smoke; not a claim of identical long-run weights or PPL.',
    'original_gate_history': 'See package provenance; failed observer attempt was not used in production.'})
