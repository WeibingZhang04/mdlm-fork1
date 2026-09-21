import subprocess
import sys
from common import ARMS, STATE, STUDY, save
# Argument construction only; runtime model/evaluator modules come from historical eval_code.
from campaign_common import VARIANTS, generation_args

index = int(sys.argv[1])
gate = len(sys.argv) > 2 and sys.argv[2] == 'gate'
if not gate:
    assert (STUDY / 'training' / ARMS[index] / 'completed.json').is_file()
steps = [3] if gate else [5000, 6000, 7000]
for step in steps:
    exp = STUDY / ('gate/exports' if gate else 'exports') / ARMS[index] / ('step%06d' % step)
    out = STUDY / ('gate/evaluation' if gate else 'evaluation') / ARMS[index] / ('step%06d' % step)
    out.mkdir(parents=True, exist_ok=False)
    modes = ('factorized', 'structured_joint') if step == steps[0] else ('structured_joint',)
    args = generation_args(VARIANTS[index], exp / 'adapter.safetensors', exp / 'adapter.manifest.json',
        out / 'generation', steps=(8,) if gate else (8, 16, 32),
        samples=1 if gate else 100, seed=100001, modes=modes)
    save(out / 'args.json', args)
    save(out / 'request.json', dict(checkpoint=step, train_seed=2, sample_seed=100001,
        samples_per_cell=1 if gate else 100, denoising_steps=[8] if gate else [8, 16, 32],
        generation_source=STATE['eval_code'], gate=gate))
    with (out / 'run.log').open('w') as log:
        subprocess.run([sys.executable, str(STUDY / 'run_generation.py'), str(out / 'args.json'),
                       str(out / 'cache-check.json')], stdout=log, stderr=subprocess.STDOUT, check=True)
    save(out / 'completed.json', dict(status='completed'))
save(STUDY / ('gate/evaluation' if gate else 'evaluation') / ARMS[index] / 'completed.json', dict(status='completed'))
