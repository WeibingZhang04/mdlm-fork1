"""Relocate the exact seed-2 production files; never overwrite an existing run."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / 'manifest.json').read_text())
    for relative, record in manifest['files'].items():
        assert sha(package / relative) == record['sha256'], relative
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ['train_arm.py', 'train_phase.py', 'evaluate.py', 'run_generation.py',
                 'export.py', 'common.py', 'passive_observer.py']:
        shutil.copy2(package / 'submitted' / name, output / name)
    shutil.copytree(package / 'submitted/configs', output / 'configs')
    shutil.copy2(package / 'helpers/campaign_common.py', output / 'campaign_common.py')
    state = json.loads((package / 'submitted/study.json').read_text())
    original = dict(state)
    state.update(root=str(output), train_code=str(package / 'training_source'),
                 eval_code=str(package / 'evaluation_source'))
    (output / 'study.json').write_text(json.dumps(state, indent=2) + '\n')
    cache = manifest['cache']
    (output / 'campaign.json').write_text(json.dumps({'cache': cache}) + '\n')
    (output / 'relocation.json').write_text(json.dumps({
        'original_root': original['root'], 'package': str(package),
        'changes': {key: [original[key], state[key]] for key in ['root', 'train_code', 'eval_code']},
        'production_python_files_byte_identical': True,
        'new_gate': 'A fresh portability smoke; historical failed/passed gate evidence is retained separately.'
    }, indent=2) + '\n')
    shutil.copy2(package / 'portable_gate.py', output / 'portable_gate.py')
    shell = '''#!/bin/bash
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
export CCF_LEGACY_STUDY="$(cd "$(dirname "$0")" && pwd)"
export CCF_CAMPAIGN_ROOT="$CCF_LEGACY_STUDY"
export CCF_CACHE_ROOT="CACHE_PATH"
export HF_HUB_CACHE="$CCF_CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CCF_LEGACY_STUDY"
cd "$CCF_LEGACY_STUDY"
case "$1" in
  gate) python -u portable_gate.py ;;
  train) python -u train_arm.py "${SLURM_ARRAY_TASK_ID:?}" ;;
  evaluate) python -u evaluate.py "${SLURM_ARRAY_TASK_ID:?}" ;;
  *) echo 'Expected gate, train or evaluate' >&2; exit 2 ;;
esac
'''.replace('CACHE_PATH', cache)
    (output / 'run.sh').write_text(shell)
    subprocess.run(['bash', '-n', str(output / 'run.sh')], check=True)
    return output


def submit(output):
    common = ['sbatch', '--parsable', '--partition=ALL', '--gres=gpu:1', '--mem=40G',
              '--cpus-per-task=4', '--no-requeue',
              '--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908',
              '--mail-user=n23zhang@uwaterloo.ca',
              '--mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS']
    jobs = {}
    for action, options in [
        ('gate', ['--time=01:00:00']),
        ('train', ['--time=24:00:00', '--array=0-3%2']),
        ('evaluate', ['--time=12:00:00', '--array=0-3%2'])
    ]:
        if action == 'train': options += ['--dependency=afterok:' + jobs['gate']['job']]
        if action == 'evaluate': options += ['--dependency=aftercorr:' + jobs['train']['job']]
        command = common + options + ['--job-name=ccf-preserved-s2-' + action,
            '--output=' + str(output / (action + '-%A_%a.out')),
            '--error=' + str(output / (action + '-%A_%a.err')), str(output / 'run.sh'), action]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        jobs[action] = {'job': job, 'command': command}
        (output / 'submission.json').write_text(json.dumps(jobs, indent=2) + '\n')
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--submit', action='store_true', help='Actually submit a new four-arm study.')
    args = parser.parse_args()
    output = prepare(args.output)
    print(json.dumps({'output': str(output), 'jobs': submit(output) if args.submit else None}))
