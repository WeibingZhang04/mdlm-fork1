#!/usr/bin/env python3
"""Prepare a fresh reproduction; submit jobs only with explicit --submit.

Added while archiving the campaign. The original jobs were launched by the
historical orchestration scripts, not this convenience entry point.
"""
import argparse
import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, default=PACKAGE.parents[1])
    parser.add_argument('--output-parent', type=Path)
    parser.add_argument('--submit', action='store_true', help='Explicitly submit the new campaign to Slurm after preparing it')
    parser.add_argument('--mail-user', default='n23zhang@uwaterloo.ca')
    args = parser.parse_args()
    original = json.loads((PACKAGE / 'campaign.json').read_text())
    source = args.source_repo.expanduser().resolve()
    cache = Path(original['cache'])
    parent = args.output_parent or cache / 'runs'
    required = [cache / 'checkpoints/mdlm-owt-backbone.pt', Path(original['old_code'])]
    required += [Path(p) for p in original['old_checkpoints'].values()]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        parser.error('External artifacts are required; see campaign.json: ' + ', '.join(missing))
    if not parent.is_dir():
        parser.error('--output-parent must be an existing directory')
    subprocess.run(['git', 'cat-file', '-e', original['head'] + '^{commit}'], cwd=source, check=True)
    root = Path(tempfile.mkdtemp(prefix='ccf_replication15k_', dir=parent))
    code = root / 'code'
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(source), str(code)], check=True)
    subprocess.run(['git', 'checkout', '--detach', original['head']], cwd=code, check=True)
    shutil.copytree(PACKAGE / 'helpers', root / 'helpers', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (root / 'logs').mkdir()
    state = {key: original[key] for key in ['cache', 'head', 'old_code', 'old_checkpoints']}
    state.update(root=str(root), code=str(code), source_repo=str(source), jobs={}, status='prepared',
                 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 reproduced_from=original['root'],
                 helper_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'helpers').glob('*.py')})
    for script in (PACKAGE / 'slurm').glob('*.sh'):
        text = script.read_text().replace(original['root'], str(root))
        (root / script.name).write_text(text)
    state_path = root / 'campaign.json'
    def save():
        state_path.write_text(json.dumps(state, indent=2) + '\n')
    save()
    print('Prepared:', root, flush=True)
    print('Model code is pinned to:', state['head'], flush=True)
    if not args.submit:
        print('No jobs submitted. Review the files; use --submit on a new invocation to prepare and submit a separate fresh campaign.')
        return
    common = ['sbatch', '--parsable', '--partition=ALL', '--mem=36G', '--cpus-per-task=4', '--gres=gpu:1',
              '--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908', '--mail-user=' + args.mail_user,
              '--mail-type=ALL,ARRAY_TASKS,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50']
    plan = [
        ('audit', 'gate.sh', '01:30:00', None, None),
        ('smoke', 'smoke.sh', '00:40:00', None, ('afterok', 'audit')),
        ('legacy_replay', 'legacy_replay.sh', '03:00:00', None, ('afterok', 'audit')),
        ('topology6k', 'topology6k.sh', '03:00:00', None, ('afterok', 'audit')),
        ('training', 'training.sh', '24:00:00', '0-7%2', ('afterok', 'smoke')),
        ('evaluation', 'evaluation.sh', '12:00:00', '0-7%2', ('aftercorr', 'training')),
        ('weights', 'weights.sh', '08:00:00', '0-2%1', ('afterok', 'smoke')),
        ('graph_generation', 'graph_generation.sh', '04:00:00', '0-3%1', ('afterok', 'audit')),
    ]
    for name, script, time_limit, array, dependency in plan:
        command = common + ['--job-name=ccf15k-' + name, '--time=' + time_limit]
        if array:
            command.append('--array=' + array)
        if dependency:
            command.append('--dependency=' + dependency[0] + ':' + state['jobs'][dependency[1]])
        suffix = '%A_%a' if array else '%j'
        command += ['--output=' + str(root / 'logs' / (name + '-' + suffix + '.out')),
                    '--error=' + str(root / 'logs' / (name + '-' + suffix + '.err')), str(root / script)]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected sbatch result: ' + job)
        state['jobs'][name] = job
        state.setdefault('submission_commands', {})[name] = command
        state['status'] = 'submitted'
        save()
        print(name, job, flush=True)


if __name__ == '__main__':
    main()
