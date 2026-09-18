"""Isolated 500-transition evaluation and fail-closed GPU equivalence gate."""
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
CACHE = Path(os.environ['CCF_CACHE_ROOT'])
BACKBONE = CACHE / 'checkpoints/mdlm-owt-backbone.pt'
BACKBONE_SHA = '7508daae475e7c0aa39dd7014e786fa9788fe1fc37f040c076e2df021e45f605'
ARMS = {
    'static_static': ('fixed', 'fixed', 0.0),
    'fixed_dynamic': ('fixed', 'dynamic', 0.0),
    'dynamic_fixed': ('dynamic', 'fixed', 0.1),
    'dynamic_dynamic': ('dynamic', 'dynamic', 0.1),
}
SEPARATE = {
    (8, 'fixed_dynamic'): 'ccf_separate_r8_fixed_dynamic_s001_step7000_run-separate-r8-fd',
    (8, 'dynamic_dynamic'): 'ccf_separate_r8_dynamic_dynamic_s001_step7000_run-separate-r8-dd',
    (16, 'fixed_dynamic'): 'ccf_separate_r16_fixed_dynamic_s001_step7000_run-separate-r16-fd',
    (16, 'dynamic_dynamic'): 'ccf_separate_r16_dynamic_dynamic_s001_step7000_run-separate-r16-dd',
}

def cell(family, arm, step):
    if family == 'basic':
        rank, embedding = 16, 'shared'
        archive = ('four_arm_continue_s001_k128_to3k_run-continue-3k' if step == 2000 else
                   'four_arm_continue_s001_k128_3k_to6k_run-continue-6k' if step <= 6000 else
                   'four_arm_continue_s001_k128_6k_to10k_run-continue-10k')
        checkpoint = CACHE / 'runs' / archive / arm / 'checkpoints' / f'0-{step}.ckpt'
    else:
        rank, embedding = int(family[1:]), 'separate'
        checkpoint = CACHE / 'runs' / SEPARATE[rank, arm] / 'training/checkpoints' / f'0-{step}.ckpt'
    return dict(family=family, arm=arm, step=step, rank=rank, embedding=embedding,
                checkpoint=str(checkpoint), mode='structured_joint')

def plans():
    # R8 first; within that group, prioritize the lower prior single-sample PPLs.
    r8 = [cell('r8', arm, step) for arm, step in [
        ('dynamic_dynamic', 2000), ('dynamic_dynamic', 6000),
        ('fixed_dynamic', 6000), ('fixed_dynamic', 4000),
        ('dynamic_dynamic', 4000), ('fixed_dynamic', 7000),
        ('dynamic_dynamic', 7000), ('fixed_dynamic', 2000)]]
    r16 = [cell('r16', arm, step) for arm, step in [
        ('dynamic_dynamic', 2000), ('fixed_dynamic', 4000),
        ('dynamic_dynamic', 4000), ('dynamic_dynamic', 7000),
        ('fixed_dynamic', 7000), ('fixed_dynamic', 6000),
        ('fixed_dynamic', 2000), ('dynamic_dynamic', 6000)]]
    basic = [cell('basic', arm, step)
             for arm in ('dynamic_dynamic', 'fixed_dynamic', 'dynamic_fixed', 'static_static')
             for step in (2000, 4000, 6000, 10000)]
    # Baseline uses the same authenticated backbone and harness; head predictions bypassed.
    baseline = dict(cell('basic', 'static_static', 2000), mode='factorized', label='mdlm_baseline')
    verify = [r8[0], r8[2], r16[0], r16[1]] + [cell('basic', arm, 2000) for arm in ARMS]
    return dict(r8=r8, r16=r16, basic=basic, baseline=[baseline], verify=verify)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)

def prepare(c, out):
    assert sha(BACKBONE) == BACKBONE_SHA, 'Backbone identity changed'
    checkpoint = Path(c['checkpoint'])
    checkpoint_sha = sha(checkpoint)
    topology, factor, weight = ARMS[c['arm']]
    adapter, manifest = out / 'adapter.safetensors', out / 'adapter.manifest.json'
    cmd = [sys.executable, 'scripts/export_structured_adapter.py',
           '--checkpoint', str(checkpoint), '--expected-checkpoint-sha256', checkpoint_sha,
           '--expected-global-step', str(c['step']), '--output', str(adapter), '--manifest', str(manifest),
           '--control-identity', c['arm'], '--topology-mode', topology, '--factor-mode', factor,
           '--candidate-k', '128', '--independent-mode', 'false', '--topology-weight', str(weight)]
    with (out / 'export-report.json').open('x') as log:
        subprocess.run(cmd, check=True, stdout=log)
    args = ['--backbone-checkpoint', str(BACKBONE), '--backbone-sha256', BACKBONE_SHA,
            '--adapter', str(adapter), '--adapter-sha256', sha(adapter),
            '--adapter-manifest', str(manifest), '--adapter-manifest-sha256', sha(manifest),
            '--output-dir', str(out / 'generation'), '--num-samples', '2', '--sequence-length', '1024',
            '--batch-size', '1', '--base-seed', '91001', '--modes', c['mode'], '--nfe-budgets', '501',
            '--device', 'cuda', '--model-config', 'contextual-forest-small',
            '--data-config', 'train_openwebtext_pinned', '--allow-dirty']
    for override in [f'data.cache_dir={CACHE}/huggingface', 'model.structured_decoder.top_k=128',
                     f'model.structured_decoder.rank={c["rank"]}',
                     f'++model.structured_decoder.factor_embedding_mode={c["embedding"]}',
                     '++model.structured_decoder.factor_conditioner_hidden_dim=0',
                     f'model.structured_decoder.topology_mode={topology}',
                     f'model.structured_decoder.factor_mode={factor}',
                     'model.structured_decoder.independent_mode=false',
                     f'model.structured_decoder.training.topology_weight={weight}',
                     f'checkpointing.save_dir={out}']:
        args += ['--override', override]
    save(out / 'request.json', dict(cell=c, checkpoint_sha256=checkpoint_sha,
        reverse_iterations=500, reserved_cleanup_calls=1, pilot_arguments=args,
        pair_seeds=[91001, 91002], snapshot=str(ROOT), job_id=os.environ.get('SLURM_JOB_ID')))
    return args

def verify(c, out, args):
    import torch
    import dataloader
    import diffusion
    from scripts import run_generation_pilot as pilot
    from scripts.audit_ccf_sampling_v4 import experiment
    from evaluation import generation_harness as harness
    assert torch.cuda.is_available(), 'GPU gate requires CUDA'
    torch.set_num_threads(1)
    config = pilot._compose_config(pilot._parse_args(args))
    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion.Diffusion(config, tokenizer=tokenizer).cuda().eval()
    assert model.ema is None
    prompt = harness.unconditional_prompt(mask_token_id=model.mask_index, sequence_length=1024)
    specs = harness.expand_paired_samples([prompt], num_samples=2, base_seed=91001)[:1]
    outputs = {}
    with torch.no_grad():
        for variant in ('v2', 'level_draws'):
            with experiment(variant):
                records, timing = harness.run_sampling_group(model, specs, sampling_mode='structured_joint',
                    nfe_budget=501, tokenizer=tokenizer, device=torch.device('cuda'))
            outputs[variant] = (records, timing, torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            save(out / f'{variant}-verification-sample.json', records)
    old, new = outputs['v2'], outputs['level_draws']
    checks = dict(tokens_equal=old[0][0]['sample_token_ids'] == new[0][0]['sample_token_ids'],
                  nfe_equal=old[1]['measured_nfe'] == new[1]['measured_nfe'],
                  cpu_rng_equal=torch.equal(old[2], new[2]), cuda_rng_equal=torch.equal(old[3], new[3]))
    save(out / 'verification.json', dict(status='passed' if all(checks.values()) else 'failed',
        cell=c, checks=checks, seed=91001, reverse_iterations=500, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), reference='v2', candidate='level_draws',
        reference_timing=old[1], candidate_timing=new[1],
        scope='One full 500-step sample per configuration; final tokens/NFE/CPU and CUDA RNG, not every intermediate state'))
    assert all(checks.values()), checks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['inventory', 'verify', 'generate'])
    parser.add_argument('--group', choices=['r8', 'r16', 'basic', 'baseline'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output-root', type=Path)
    a = parser.parse_args()
    matrix = plans()
    if a.phase == 'inventory':
        cells = matrix['r8'] + matrix['r16'] + matrix['basic']
        missing = [c['checkpoint'] for c in cells if not Path(c['checkpoint']).is_file()]
        print(json.dumps(dict(plans=matrix, missing=missing, generation_cells=33, requested_samples=66), indent=2))
        assert not missing, missing
        return
    c = matrix['verify' if a.phase == 'verify' else a.group][a.index]
    parent = a.output_root / ('verification' if a.phase == 'verify' else a.group) / f'{a.index:02d}_{c["arm"]}_step{c["step"]}'
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f'attempt-{os.environ.get("SLURM_JOB_ID", "local")}-', dir=parent))
    args = prepare(c, out)
    if a.phase == 'verify':
        verify(c, out, args)
    else:
        args += ['--reference-lm', 'gpt2-large', '--reference-lm-revision', '32b71b12589c2f8d625668d2335a01cac3249519',
                 '--reference-lm-device', 'cuda', '--reference-lm-batch-size', '1',
                 '--reference-lm-max-length', '1024', '--reference-lm-dtype', 'float32']
        entry = 'scripts/run_generation_fast.py' if c['mode'] == 'structured_joint' else 'scripts/run_generation_pilot.py'
        subprocess.run([sys.executable, entry, *args], check=True)
    print(json.dumps(dict(status='completed', phase=a.phase, cell=c, output=str(out))), flush=True)

if __name__ == '__main__':
    main()
