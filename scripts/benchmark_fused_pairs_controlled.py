"""Paired old/new CRF generation benchmark; run the same script on both nodes.

Old: original generation.py from a pinned Git snapshot, inference=segments.
New: installed generation.py and fused_pairs.py, inference=fused-pairs.
All shared inference/model sources must be byte-identical between snapshots.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, text=True, capture_output=True)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/u401/n23zhang/rework-data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--modes', nargs='+', default=['contextual', 'global', 'count'])
    args = parser.parse_args()
    assert args.draws > 0 and args.rounds > 0
    assert not args.output.exists(), 'Do not overwrite a previous run'
    old, new = args.old_root.resolve(), args.new_root.resolve()
    shared = {}
    for folder in ('chain_crf', 'models'):
        for path in (old/folder).rglob('*.py'):
            relative = path.relative_to(old)
            if str(relative) == 'chain_crf/generation.py':
                continue
            assert path.read_bytes() == (new/relative).read_bytes(), str(relative)
            shared[str(relative)] = digest(path)
    assert (old/'configs/model/small.yaml').read_bytes() == (new/'configs/model/small.yaml').read_bytes()
    sys.path.insert(0, str(new))
    import torch
    from chain_crf.backbone import FrozenMDLM
    from chain_crf.generation import generate as new_generate
    from chain_crf.core import build_candidates
    from chain_crf.fused_pairs import infer_fused_pairs, _COUNT_CACHE, _count_table
    from chain_crf.segments import segmented_log_marginals
    from scripts.evaluate_chain_crf import load_head
    import triton
    old_generate = load_module('pre_fusion_generation', old/'chain_crf/generation.py').generate
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(73921)
    torch.cuda.manual_seed_all(73921)
    assert torch.cuda.device_count() == 1, 'Exactly one scheduler-allocated GPU required'
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    apps = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'])
    uuid = next((row[1].strip() for row in csv.reader(apps['stdout'].splitlines())
                 if len(row) == 2 and row[0].strip() == str(os.getpid())), None)
    if uuid is None:
        uuid = str(getattr(props, 'uuid', os.environ.get('CUDA_VISIBLE_DEVICES', '')))
    assert uuid, 'Cannot identify allocated GPU'
    if not uuid.startswith(('GPU-', 'MIG-')):
        uuid = 'GPU-' + uuid
    fields = ['uuid', 'name', 'driver_version', 'memory.total', 'memory.used', 'utilization.gpu',
              'temperature.gpu', 'clocks.current.sm', 'clocks.current.memory', 'power.draw',
              'power.limit', 'pstate']

    def gpu_state():
        q = command(['nvidia-smi', '-i', uuid, '--query-gpu='+','.join(fields),
                     '--format=csv,noheader,nounits'])
        if q['returncode']:
            return q
        rows = list(csv.reader(q['stdout'].splitlines()))
        return dict(zip(fields, (s.strip() for s in rows[0])))

    def other_gpu_processes():
        q = command(['nvidia-smi', '-i', uuid, '--query-compute-apps=pid',
                     '--format=csv,noheader,nounits'])
        # Log only whether another process shares this GPU, not its identity.
        pids = [int(v.strip()) for v in q['stdout'].splitlines() if v.strip().isdigit()]
        return len([pid for pid in pids if pid != os.getpid()]) if q['returncode'] == 0 else None

    initial_gpu_state = gpu_state()
    assert initial_gpu_state.get('uuid') == uuid, initial_gpu_state
    assert other_gpu_processes() is not None, 'GPU process monitoring unavailable'

    report = {'status': 'running', 'host': socket.gethostname(), 'job_id': os.environ.get('SLURM_JOB_ID'),
              'pid': os.getpid(), 'gpu': props.name, 'gpu_uuid': uuid,
              'gpu_total_bytes': props.total_memory, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'lscpu': command(['lscpu']), 'torch': torch.__version__, 'triton': triton.__version__,
              'cuda': torch.version.cuda, 'python': platform.python_version(),
              'threads': torch.get_num_threads(), 'initial_gpu_state': initial_gpu_state,
              'torch_flags': {'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                              'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                              'cudnn_benchmark': torch.backends.cudnn.benchmark},
              'source': {'driver': digest(__file__), 'old_generation': digest(old/'chain_crf/generation.py'),
                         'new_generation': digest(new/'chain_crf/generation.py'),
                         'fused_pairs': digest(new/'chain_crf/fused_pairs.py'),
                         'shared_sources': shared},
              'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS',
                  'SLURM_CPUS_PER_TASK', 'OMP_NUM_THREADS', 'TRITON_CACHE_DIR')},
              'settings': {'length': 1024, 'steps': 32, 'batch_size': 1, 'k': 500, 'vocab_cap': 500,
                           'sampling': 'joint', 'sampler': 'ddpm_cache', 'noise_removal': True,
                           'sampling_eps': 1e-5, 'segment_batch_size': 32,
                           'draws': args.draws, 'rounds': args.rounds, 'sample_offset_start': 93001000},
              'comparison': {'old': 'pre-change generation.py, inference=segments',
                             'new': 'installed generation.py, inference=fused-pairs'},
              'scope': 'Synchronized generation wall time including backbone and final denoise; '
                       'excludes loading, correctness checks, warmups, initial count-table preparation, '
                       'telemetry and GC. Every draw/path is prewarmed. Old runs do not retain the new '
                       'count-score cache; new runs prepare it outside the timer. GPU clocks/power '
                       'limits are observed, never changed. Shared-node CPU contention is possible.',
              'correctness': [], 'warmup': [], 'measurements': [], 'summaries': []}

    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(args.output)

    def emit(key, row):
        report[key].append(row)
        save()
        print(key.upper(), json.dumps(row), flush=True)

    save()
    base = args.data_root
    model = FrozenMDLM(base/'checkpoints/mdlm-owt.pt', device='cuda', cache_dir=base/'hf-cache/hub')
    report['backbone'] = model.provenance
    head_root = base/'runs/draft-full-scale-training/preliminary-checkpoints_2026_10_01_14_29_45_UTC'

    @torch.no_grad()
    def check(head, mode):
        tokens = torch.full((1, 96), model.mask_id, device='cuda', dtype=torch.long)
        known = model.tokenizer.encode('The model predicts the next word in the sentence.')
        for index in range(0, 96, 7):
            tokens[0, index] = known[(index // 7) % len(known)]
        t = torch.tensor([.5], device='cuda')
        pred = model(tokens, t)
        packet = build_candidates(pred['log_probs'], tokens, model.mask_id, 500, vocab_cap=500)
        edge = head(packet.candidate_ids, pred['hidden'], t)
        expected = segmented_log_marginals(packet.unary, edge, packet.masked, segment_batch_size=32)
        actual = infer_fused_pairs(packet, head, pred['hidden'], t,
                                  sampling='log_marginals', segment_batch_size=32)
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
        error = (actual.exp()-expected.exp()).abs().max().item()
        assert error < .0005, error
        emit('correctness', {'mode': mode, 'max_marginal_probability_error': error})

    def run(head, mode, version, draw, phase, round_index):
        _COUNT_CACHE.clear()
        gc.collect()
        torch.cuda.synchronize()
        prep_start = time.perf_counter()
        if version == 'new' and mode == 'count':
            with torch.no_grad():
                _count_table(head)
        torch.cuda.synchronize()
        prep_seconds = time.perf_counter() - prep_start
        torch.cuda.empty_cache()
        before = gpu_state()
        others_before = other_gpu_processes()
        assert before.get('uuid') == uuid and others_before is not None, before
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()/2**30
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        tokens, timing = (old_generate if version == 'old' else new_generate)(
            model, head, mode, length=1024, steps=32, batch_size=1, k=500, vocab_cap=500,
            sampler='ddpm_cache', sampling='joint', noise_removal=True, sampling_eps=1e-5,
            device='cuda', inference='segments' if version == 'old' else 'fused-pairs',
            segment_batch_size=32, stage_timing=False, sample_offset=draw)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()/2**30
        reserved = torch.cuda.max_memory_reserved()/2**30
        after = gpu_state()
        others_after = other_gpu_processes()
        assert after.get('uuid') == uuid and others_after is not None, after
        assert tokens.shape == (1, 1024) and not tokens.eq(model.mask_id).any()
        row = {'mode': mode, 'version': version, 'draw': draw, 'round': round_index,
               'wall_seconds': seconds, 'generation_seconds': timing['elapsed_seconds'],
               'backbone_calls': timing['backbone_calls'], 'cache_hits': timing['cache_hits'],
               'resident_before_gib': resident, 'peak_allocated_gib': peak, 'peak_reserved_gib': reserved,
               'count_prepare_seconds': prep_seconds, 'gpu_before': before, 'gpu_after': after,
               'other_gpu_processes_before': others_before, 'other_gpu_processes_after': others_after,
               'token_sha256': hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest()}
        emit(phase, row)
        return row

    for mode in args.modes:
        head, identity = load_head(SimpleNamespace(mode=mode, device='cuda',
            head=head_root/(mode+'-step-30000.pt'), count_mode='pmi', strength=.1,
            counts=base/'runs/draft-full-scale-training/train_2026_10_01_05_46_13_UTC/counts/owt-counts.pt'), model)
        report.setdefault('heads', {})[mode] = identity
        check(head, mode)
        draws = list(range(93001000, 93001000 + args.draws))
        warm_hashes = {}
        for index, draw in enumerate(draws):
            for version in (('old', 'new') if index % 2 == 0 else ('new', 'old')):
                row = run(head, mode, version, draw, 'warmup', -1)
                warm_hashes[version, draw] = row['token_sha256']
        for round_index in range(args.rounds):
            order = list(draws)
            random.Random(719 + round_index).shuffle(order)
            for draw in order:
                index = draw - draws[0]
                versions = ('old', 'new') if (index+round_index) % 2 == 0 else ('new', 'old')
                for version in versions:
                    row = run(head, mode, version, draw, 'measurements', round_index)
                    assert row['token_sha256'] == warm_hashes[version, draw], 'Same-path replay mismatch'
        rows = [r for r in report['measurements'] if r['mode'] == mode]
        stats = {}
        for version in ('old', 'new'):
            selected = [r for r in rows if r['version'] == version]
            times = [r['wall_seconds'] for r in selected]
            quartiles = statistics.quantiles(times, n=4, method='inclusive')
            stats[version] = {'n': len(times), 'median_seconds': statistics.median(times),
                'min_seconds': min(times), 'max_seconds': max(times),
                'q1_seconds': quartiles[0], 'q3_seconds': quartiles[2],
                'max_peak_allocated_gib': max(r['peak_allocated_gib'] for r in selected)}
        paired = []
        for draw in draws:
            for ri in range(args.rounds):
                pair = {r['version']: r for r in rows if r['draw'] == draw and r['round'] == ri}
                paired.append(pair['old']['wall_seconds']/pair['new']['wall_seconds'])
        emit('summaries', {'mode': mode, 'timings': stats,
            'ratio_of_medians': stats['old']['median_seconds']/stats['new']['median_seconds'],
            'median_paired_speedup': statistics.median(paired),
            'paired_speedup_range': [min(paired), max(paired)],
            'same_path_replay': True,
            'same_tokens_across_paths': all(warm_hashes['old', d] == warm_hashes['new', d] for d in draws),
            'other_gpu_processes_observed': any(r['other_gpu_processes_before'] or
                                              r['other_gpu_processes_after'] for r in rows)})
        _COUNT_CACHE.clear()
        del head
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
