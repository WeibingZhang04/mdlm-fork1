"""Run untouched original seed-1 FD training phases, then paired generation."""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from common import STUDY, STATE, save
from campaign_common import CACHE, VARIANTS, generation_args


def execute(args, logfile):
    with Path(logfile).open('x') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)


def original_checkpoint(step):
    folder = ('stale/four_arm_s001_k128_job1541029' if step <= 1000 else
              'four_arm_continue_s001_k128_to3k_job1542157' if step <= 3000 else
              'four_arm_continue_s001_k128_3k_to6k_job1542808')
    return CACHE / 'runs' / folder / 'fixed_dynamic/checkpoints' / ('0-' + str(step) + '.ckpt')


def generate(label, checkpoint, step, count, seed, modes, denoising=(8, 16, 32)):
    folder = STUDY / 'evaluation' / label
    folder.mkdir(parents=True, exist_ok=False)
    export = folder / 'export'
    execute([sys.executable, str(STUDY / 'export.py'), '1', str(step), str(checkpoint), str(export)],
            folder / 'export.log')
    args = generation_args(VARIANTS[1], export / 'adapter.safetensors', export / 'adapter.manifest.json',
                           folder / 'generation', steps=denoising, samples=count, seed=seed, modes=modes)
    save(folder / 'args.json', args)
    save(folder / 'request.json', dict(checkpoint=str(checkpoint), step=step, count=count,
                                     seed=seed, modes=modes, denoising_steps=denoising))
    execute([sys.executable, str(STUDY / 'run_generation.py'), str(folder / 'args.json'),
             str(folder / 'cache-check.json')], folder / 'run.log')
    summary = json.loads((folder / 'generation/summary.json').read_text())
    assert all(g['num_sequences'] == count and g['reference_lm']['num_scored_sequences'] == count
               for g in summary['groups'])
    save(folder / 'completed.json', {'status': 'completed'})
    return folder


execute([sys.executable, '-m', 'pip', 'freeze'], STUDY / 'environment-pip-freeze.txt')
execute(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv'], STUDY / 'gpu.csv')
# A fresh-process evaluator smoke before spending compute on training. It cannot prewarm training caches.
generate('smoke-original', original_checkpoint(6000), 6000, 1, 100001,
         ('factorized', 'structured_joint'), denoising=(8,))
for phase in [1000, 3000, 6000]:
    if phase == 1000 and (STUDY / 'reuse-1k.json').is_file():
        reuse = json.loads((STUDY / 'reuse-1k.json').read_text())
        assert reuse['completed_step'] == 1000
        from common import sha
        assert sha(reuse['checkpoint']) == reuse['checkpoint_sha256']
        assert (STUDY / 'training/to1000/fixed_dynamic/checkpoints/0-1000.ckpt').resolve() == Path(reuse['checkpoint']).resolve()
        shutil.copy2(reuse['weight_comparison'], STUDY / 'weights-1000.json')
        save(STUDY / 'progress.json', {'phase_completed': 1000, 'reused_completed_phase': reuse})
        continue
    execute(['bash', str(STUDY / ('phase-' + str(phase) + '.sh'))], STUDY / ('phase-' + str(phase) + '.log'))
    folder = STUDY / ('training/to' + str(phase)) / 'fixed_dynamic'
    rows = []
    for path in sorted(folder.glob('lightning_logs/version_*/metrics.csv')):
        with path.open() as handle: rows.extend(dict(row) for row in csv.DictReader(handle))
    assert rows, 'Original CSV logs missing'
    save(folder / 'original-metrics.json', rows)
    execute([sys.executable, str(STUDY / 'compare_checkpoints.py'), str(original_checkpoint(phase)),
             str(folder / 'checkpoints' / ('0-' + str(phase) + '.ckpt')),
             str(STUDY / ('weights-' + str(phase) + '.json'))],
            STUDY / ('compare-' + str(phase) + '.log'))
    save(STUDY / 'progress.json', {'phase_completed': phase})

comparisons = []
for step, count, seed in [(2000, 20, 91001), (6000, 20, 91001), (6000, 100, 100001)]:
    suffix = str(step) + '-' + str(count)
    old = generate('original-' + suffix, original_checkpoint(step), step, count, seed,
                   ('factorized', 'structured_joint'))
    phase = 3000 if step == 2000 else 6000
    fresh = STUDY / ('training/to' + str(phase)) / 'fixed_dynamic/checkpoints' / ('0-' + str(step) + '.ckpt')
    new = generate('retrained-' + suffix, fresh, step, count, seed, ('structured_joint',))
    def records(folder):
        return {(r['sampling_mode'], r['requested_nfe_budget'], r['sample_index']): r
                for r in [json.loads(line) for line in (folder / 'generation/samples.jsonl').read_text().splitlines()]
                if r['sampling_mode'] == 'structured_joint'}
    a, b = records(old), records(new)
    assert set(a) == set(b)
    old_groups = {g['requested_nfe_budget']: g for g in json.loads((old / 'generation/summary.json').read_text())['groups']
                  if g['sampling_mode'] == 'structured_joint'}
    for g in json.loads((new / 'generation/summary.json').read_text())['groups']:
        budget = g['requested_nfe_budget']; keys = [k for k in a if k[1] == budget]
        comparisons.append(dict(checkpoint=step, samples=count, seed=seed, denoising_steps=budget-1,
            original_ppl=old_groups[budget]['reference_lm']['perplexity'],
            retrained_ppl=g['reference_lm']['perplexity'],
            identical_sequences=sum(a[k]['sample_token_ids'] == b[k]['sample_token_ids'] for k in keys),
            total_sequences=len(keys)))
    save(STUDY / 'generation-comparisons.json', comparisons)
save(STUDY / 'completed.json', {'status': 'completed', 'arm': 'fixed_dynamic', 'training_seed': 1,
                              'training_steps': 6000, 'comparisons': comparisons})
