#!/usr/bin/env python3
"""Run a frozen checkpoint selection with new paired confirmation seeds.

Orchestration only: reuse the pinned pilot export, sampler gate, and scorer.
Algorithm and implementation attribution: docs/ccf-fast-generation.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_ccf_selected_five as base

SOURCE_FILES = (
    'scripts/evaluate_ccf_confirmation.py',
    'scripts/evaluate_ccf_selected_five.py', 'scripts/run_generation_pilot.py',
    'scripts/audit_ccf_sampling_v3.py', 'scripts/audit_ccf_sampling_v4.py',
    'structured_utils.py', 'structured_objective.py', 'diffusion.py',
    'models/structured_decoder.py', 'evaluation/generation_harness.py',
    'evaluation/generation_metrics.py',
)


def load_selection(path, expected_sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Selection hash mismatch')
    selection = json.loads(raw)
    if selection['num_samples'] != 100 or selection['base_seed'] != 100001:
        raise ValueError('Confirmation requires 100 fresh seeds starting at 100001')
    known = {(c['family'], c['arm'], c['step']): c
             for c in base.matrix('full_checkpoint_sweep')}
    budgets = selection['sampling_budgets']
    if not budgets or len(set(budgets)) != len(budgets) or any(b not in (4, 8, 16, 32) for b in budgets):
        raise ValueError('Unsupported or duplicated sampling budgets')
    cells, seen = [], set()
    for requested in selection['cells']:
        key = tuple(requested[k] for k in ('family', 'arm', 'step'))
        steps = requested['sampling_steps']
        if steps not in budgets or key + (steps,) in seen:
            raise ValueError('Invalid or duplicated configuration')
        seen.add(key + (steps,))
        cell = dict(known[key], sampling_steps=steps, num_samples=100,
                    base_seed=100001)
        cells.append(cell)
    if not cells:
        raise ValueError('Empty selection')
    for steps in budgets:
        if sum(c['family'] == 'A' and c['sampling_steps'] == steps for c in cells) != 1:
            raise ValueError('Exactly one fresh MDLM baseline per sampling budget required')
    return selection, cells


def confirmation_args(cell, out, adapter_sha, manifest_sha):
    args = base.generation_args(cell, out, adapter_sha, manifest_sha)
    args[args.index('--base-seed') + 1] = str(cell['base_seed'])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    options = parser.parse_args()
    selection, cells = load_selection(options.selection, options.selection_sha256)
    if options.inventory:
        missing = sorted({c['checkpoint'] for c in cells if not Path(c['checkpoint']).is_file()})
        print(json.dumps(dict(cells=cells, samples=100 * len(cells), missing=missing), indent=2))
        if missing:
            raise FileNotFoundError('Missing selected checkpoints')
        return 0
    if options.index is None or not 0 <= options.index < len(cells) or options.output_root is None:
        parser.error('A valid index and output-root are required')
    cell = dict(cells[options.index])
    if options.verify_only:
        if cell['mode'] != 'structured_joint':
            parser.error('Verification tasks must select a CCF cell')
        cell['num_samples'] = 1
    parent = (options.output_root / f'steps_{cell["sampling_steps"]}' /
              f'{options.index:02d}_{cell["family"]}_{cell["arm"]}_step{cell["step"]}')
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f'attempt-{os.environ.get("SLURM_JOB_ID", "local")}-', dir=parent))
    assert base.sha(base.CACHE / 'checkpoints/mdlm-owt-backbone.pt') == base.BACKBONE_SHA
    topology, factor, weight = base.ARMS[cell['arm']]
    checkpoint_sha = base.sha(cell['checkpoint'])
    with (out / 'export-report.json').open('x') as log:
        subprocess.run([sys.executable, 'scripts/export_structured_adapter.py',
            '--checkpoint', cell['checkpoint'], '--expected-checkpoint-sha256', checkpoint_sha,
            '--expected-global-step', str(cell['step']), '--output', str(out / 'adapter.safetensors'),
            '--manifest', str(out / 'adapter.manifest.json'), '--control-identity', cell['arm'],
            '--topology-mode', topology, '--factor-mode', factor, '--candidate-k', '128',
            '--independent-mode', 'false', '--topology-weight', str(weight)], check=True, stdout=log)
    args = confirmation_args(cell, out, base.sha(out / 'adapter.safetensors'),
                             base.sha(out / 'adapter.manifest.json'))
    base.save(out / 'request.json', dict(cell=cell, checkpoint_sha256=checkpoint_sha,
        pilot_arguments=args, suite='confirmation_100', samples=cell['num_samples'],
        sampling_steps=cell['sampling_steps'], base_seed=cell['base_seed'],
        verification_only=options.verify_only, snapshot=str(ROOT),
        selection_sha256=options.selection_sha256, selection_rule=selection['selection_rule'],
        selected_pilot=selection['cells'][options.index],
        pilot_snapshot_sha256=selection['pilot_snapshot_sha256'],
        pairing='Fresh base_seed=100001; replicate IDs paired across all cells and budgets. No pilot samples pooled.',
        source_sha256={name: base.sha(ROOT / name) for name in SOURCE_FILES},
        local_commit=os.environ.get('CCF_LOCAL_EVAL_COMMIT'),
        attribution='docs/ccf-fast-generation.md'))
    from scripts import run_generation_pilot as pilot
    result = pilot.main(args) if cell['mode'] == 'factorized' else base.gated_pilot(pilot, cell, out, args)
    assert result == 0
    summary = json.loads((out / 'generation/summary.json').read_text())
    group = summary['groups'][0]
    assert group['num_sequences'] == cell['num_samples']
    assert group['unresolved_mask_tokens'] == 0
    if cell['mode'] == 'structured_joint':
        assert json.loads((out / 'verification.json').read_text())['status'] == 'passed'
    base.save(out / 'completed.json', dict(status='completed', samples=cell['num_samples'],
        cell=cell, verification_only=options.verify_only))
    print(json.dumps(dict(status='completed', output=str(out))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
