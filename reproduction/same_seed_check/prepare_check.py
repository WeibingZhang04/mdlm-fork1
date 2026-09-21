"""Prepare an FD seed-1 replay of the original 12a4579 launchers in fresh paths."""
import argparse
import hashlib
import json
import shutil
import shlex
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SEED2 = PACKAGE.parent / 'seed2'


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(SEED2 / 'training_source', output / 'training_code')
    shutil.copytree(PACKAGE / 'original_additions', output / 'training_code', dirs_exist_ok=True)
    shutil.copytree(SEED2 / 'evaluation_source', output / 'evaluation_code')
    source_hashes = json.loads((PACKAGE / '12a4579-source-sha256.json').read_text())
    for name, expected in source_hashes.items():
        assert hashlib.sha256((output / 'training_code' / name).read_bytes()).hexdigest() == expected, name
    state = json.loads((SEED2 / 'submitted/study.json').read_text())
    state.update(root=str(output), train_code=str(output / 'training_code'),
                 eval_code=str(output / 'evaluation_code'), training_seed=1,
                 final_step=6000, checkpoints=[2000, 6000], restart_boundaries=[1000, 3000])
    (output / 'study.json').write_text(json.dumps(state, indent=2) + '\n')
    cache = json.loads((SEED2 / 'manifest.json').read_text())['cache']
    (output / 'campaign.json').write_text(json.dumps({'cache': cache}) + '\n')
    for name in ['common.py', 'run_generation.py', 'export.py']:
        shutil.copy2(SEED2 / 'submitted' / name, output / name)
    shutil.copy2(SEED2 / 'helpers/campaign_common.py', output / 'campaign_common.py')
    for name in ['run_check.py', 'compare_checkpoints.py']:
        shutil.copy2(PACKAGE / name, output / name)
    edits = {}
    scripts = ['train_four_adaptors.sh', 'continue_four_ccf_matched_to_3k.sh',
               'continue_four_ccf_matched_3k_to_6k.sh']
    for phase, script in zip([1000, 3000, 6000], scripts):
        original = (output / 'training_code/scripts' / script).read_text()
        body = original.replace('cd /u401/n23zhang/tree_mdlm/mdlm',
                                'cd ' + str(output / 'training_code'))
        if phase == 1000:
            body = body.replace('export CCF_RUN_ROOT=' + cache + '/runs/four_arm_s001_k128',
                                'export CCF_RUN_ROOT=' + str(output / 'training/to1000'))
        else:
            old_source = {'3000': 'stale/four_arm_s001_k128_job1541029',
                          '6000': 'four_arm_continue_s001_k128_to3k_job1542157'}[str(phase)]
            prior = 1000 if phase == 3000 else 3000
            body = body.replace('SOURCE_ROOT="$CACHE_ROOT/runs/' + old_source + '"',
                                'SOURCE_ROOT="' + str(output / ('training/to' + str(prior))) + '"')
            old_output = 'four_arm_continue_s001_k128_' + ('to3k' if phase == 3000 else '3k_to6k')
            body = body.replace('RUN_ROOT="$CACHE_ROOT/runs/' + old_output + '_${JOB_TAG}"',
                                'RUN_ROOT="' + str(output / ('training/to' + str(phase))) + '"')
        # Only directory lines change. Seed, precision, optimizer, data, callbacks and steps stay exact.
        before, after = original.splitlines(), body.splitlines()
        changes = [{'before': a, 'after': b} for a, b in zip(before, after) if a != b]
        assert len(before) == len(after)
        assert len(changes) == (2 if phase == 1000 else 3), (script, changes)
        assert all(c['before'].startswith(('cd ', 'export CCF_RUN_ROOT=', 'SOURCE_ROOT=', 'RUN_ROOT=')) for c in changes)
        path = output / ('phase-' + str(phase) + '.sh')
        path.write_text(body)
        subprocess.run(['bash', '-n', str(path)], check=True)
        edits[script] = changes
    (output / 'path-only-launcher-edits.json').write_text(json.dumps(edits, indent=2) + '\n')
    (output / 'protocol.json').write_text(json.dumps({
        'arm': 'fixed_dynamic', 'training_seed': 1, 'source_commit': '12a4579956339c48c80b074ef6b6d5d4e9691a44',
        'training_steps': 6000, 'restart_boundaries': [1000, 3000],
        'no_added_training_callbacks_or_startup_forwards': True,
        'comparisons': [
            {'checkpoint': 2000, 'samples': 20, 'generation_seed': 91001},
            {'checkpoint': 6000, 'samples': 20, 'generation_seed': 91001},
            {'checkpoint': 6000, 'samples': 100, 'generation_seed': 100001}],
        'denoising_steps': [8, 16, 32],
        'controls': 'Archived original checkpoints, regenerated on the same allocation; matched MDLM baseline.',
        'interpretation': 'Report exact weight/token identity and PPL differences separately. Same seed is not a guarantee.'
    }, indent=2) + '\n')
    shell = '''#!/bin/bash
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
export CCF_LEGACY_STUDY=STUDY_PATH
export CCF_CAMPAIGN_ROOT="$CCF_LEGACY_STUDY"
export CCF_CACHE_ROOT="CACHE_PATH"
export HF_HUB_CACHE="$CCF_CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CCF_LEGACY_STUDY"
export SLURM_ARRAY_TASK_ID=1
cd "$CCF_LEGACY_STUDY"
python -u run_check.py
'''.replace('CACHE_PATH', cache).replace('STUDY_PATH', shlex.quote(str(output)))
    (output / 'run.sh').write_text(shell)
    subprocess.run(['bash', '-n', str(output / 'run.sh')], check=True)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    out = prepare(args.output)
    result = {'root': str(out), 'submitted': False}
    if args.submit:
        command = ['sbatch', '--parsable', '--partition=ALL', '--gres=gpu:1', '--mem=40G',
            '--cpus-per-task=4', '--ntasks=1', '--time=12:00:00', '--no-requeue',
            '--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908',
            '--mail-user=n23zhang@uwaterloo.ca',
            '--mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS',
            '--job-name=ccf-original-fd-s1-check', '--output=' + str(out / 'job-%j.out'),
            '--error=' + str(out / 'job-%j.err'), str(out / 'run.sh')]
        result.update(job=subprocess.check_output(command, text=True).strip().split(';')[0],
                      command=command, submitted=True)
    (out / 'deployment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
