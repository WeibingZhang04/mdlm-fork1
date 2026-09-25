#!/usr/bin/env python3
"""Evaluate exact full-vocabulary count chains; no candidate/tail approximation.

Static sparse-potential construction is timed separately. Generation timing
includes the backbone, full-vocabulary unaries, DP, sampling and commitment.
This is a correctness-first FP64 baseline, not a claim of fast inference.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from chain_crf.backbone import FrozenMDLM, SyntheticBackbone, file_sha256
from chain_crf.counts import CountBigramHead
from chain_crf.data import atomic_json, canonical_hash
from chain_crf.generation import synchronize, token_statistics
from chain_crf.sparse_count import (
    SparseCountPotential, sample_sparse_chain, sparse_chain_marginals,
)
from scripts.evaluate_chain_crf import score_gpt2


def inference_components(backend):
    if backend == 'reference':
        return SparseCountPotential, sample_sparse_chain, sparse_chain_marginals
    if backend == 'gpu':
        from chain_crf.sparse_count_gpu import (
            GPUCountPotential, sample_sparse_chain as gpu_sample,
            sparse_chain_marginals as gpu_marginals,
        )
        return GPUCountPotential, gpu_sample, gpu_marginals
    raise ValueError('Backend must be reference or gpu')


def full_vocabulary_unary(log_probs, tokens, mask_id, temperature=1.):
    """Normalized full-V log unaries, with exact observed-token point masses."""
    if log_probs.ndim != 3 or log_probs.shape[:2] != tokens.shape:
        raise ValueError('Need log_probs[B,L,V] and tokens[B,L]')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    vocab = log_probs.shape[-1]
    if not 0 <= mask_id < vocab or ((tokens < 0) | (tokens >= vocab)).any():
        raise ValueError('Invalid mask/token IDs')
    unary = log_probs.double().clone()/temperature
    unary[..., mask_id] = -torch.inf
    visible = torch.full_like(unary, -torch.inf)
    visible.scatter_(-1, tokens[..., None], 0.)
    unary = torch.where(tokens.eq(mask_id)[..., None], unary, visible)
    norm = unary.logsumexp(-1, keepdim=True)
    if not torch.isfinite(norm).all():
        raise ValueError('Each position needs a finite supported clean token')
    return unary-norm


@torch.no_grad()
def generate_sparse_count(backbone, potential, *, length=256, steps=16,
                          batch_size=1, sampling='joint', temperature=1.,
                          device='cuda', sample_offset=0, prefix=None,
                          inference_backend='reference'):
    """Same reveal schedule/offset semantics as chain_crf.generation.generate.

Draws a full original-position chain, including visible/masked boundaries,
then commits only the scheduled positions. Joint and own-marginal modes use
the very same full-vocabulary potential; neither aggregates a residual tail.
"""
    if min(length, steps, batch_size) < 1 or sample_offset < 0:
        raise ValueError('Positive sizes and a nonnegative sample offset required')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if sampling not in ('joint', 'marginal'):
        raise ValueError('sampling must be joint or marginal')
    _, joint_sampler, marginal_inference = inference_components(inference_backend)
    if potential.vocab_size != backbone.vocab_size:
        raise ValueError('Count/backbone vocabulary mismatch')
    if potential.left.device != torch.device(device):
        # torch.device("cuda") has no index, but tensors do; compare resolved
        # device below through a tiny allocation rather than rejecting cuda:0.
        if potential.left.device != torch.empty(0, device=device).device:
            raise ValueError('Potential and generation must use the same device')
    prefix = [] if prefix is None else list(prefix)
    if any(type(v) is not int or v < 0 or v >= backbone.vocab_size or v == backbone.mask_id for v in prefix):
        raise ValueError('Prefix must contain valid observed clean tokens')
    tokens = torch.full((batch_size, len(prefix)+length), backbone.mask_id,
                        dtype=torch.long, device=device)
    if prefix:
        tokens[:, :len(prefix)] = torch.tensor(prefix, device=device)
    orders = []
    for draw_id in range(sample_offset, sample_offset+batch_size):
        rng = torch.Generator().manual_seed(1729+draw_id)
        orders.append(torch.randperm(length, generator=rng)+len(prefix))
    order = torch.stack(orders).to(device)
    generator = torch.Generator(device=device).manual_seed(2718+sample_offset)
    calls, committed = 0, 0
    backbone_seconds, sampling_seconds = 0., 0.
    synchronize(device)
    start = time.perf_counter()
    for step in range(steps):
        next_count = math.ceil((step+1)*length/steps)
        if next_count == committed:
            continue
        t = torch.full((batch_size,), 1.-committed/length, device=device)
        synchronize(device)
        before = time.perf_counter()
        prediction = backbone(tokens, t)
        synchronize(device)
        backbone_seconds += time.perf_counter()-before
        calls += 1
        before = time.perf_counter()
        unary = full_vocabulary_unary(prediction['log_probs'], tokens,
                                      backbone.mask_id, temperature)
        if sampling == 'joint':
            drawn = joint_sampler(unary, potential, generator=generator)
        else:
            probabilities = marginal_inference(unary, potential)
            drawn = torch.multinomial(probabilities.reshape(-1, backbone.vocab_size),
                                      1, generator=generator).reshape(tokens.shape)
        visible = tokens.ne(backbone.mask_id)
        if drawn.eq(backbone.mask_id).any() or not torch.equal(drawn[visible], tokens[visible]):
            raise RuntimeError('Sampler violated mask exclusion or observed-token clamping')
        positions = order[:, committed:next_count]
        tokens.scatter_(1, positions, drawn.gather(1, positions))
        committed = next_count
        synchronize(device)
        sampling_seconds += time.perf_counter()-before
    synchronize(device)
    elapsed = time.perf_counter()-start
    if tokens.eq(backbone.mask_id).any():
        raise RuntimeError('Generation left absorbing masks in the output')
    return tokens, {'elapsed_seconds': elapsed, 'backbone_seconds': backbone_seconds,
                    'sampling_seconds': sampling_seconds, 'backbone_calls': calls,
                    'samples': batch_size, 'generated_tokens': batch_size*length,
                    'prefix_length': len(prefix), 'generated_length': length,
                    'full_vocabulary': True, 'mean_retained_mass': 1.}


def _read_records(path):
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise ValueError('Malformed sample record; do not silently discard interrupted/corrupt data') from error


def _validate_records(records, config, manifest_hash, prefix, vocab_size, mask_id):
    if [r.get('sample_id') for r in records] != list(range(len(records))):
        raise ValueError('Incomplete or duplicated sample IDs')
    offset = config['sample_offset']
    if [r.get('draw_id') for r in records] != list(range(offset, offset+len(records))):
        raise ValueError('Stored draw IDs do not match the requested sample offset')
    if len(records) > config['samples']:
        raise ValueError('Existing sample count exceeds requested target')
    for row in records:
        ids = row.get('token_ids')
        batch_start = row['sample_id']//config['batch_size']*config['batch_size']
        expected_size = min(config['batch_size'], config['samples']-batch_start)
        if row.get('manifest_sha256') != manifest_hash:
            raise ValueError('Stored sample belongs to another manifest')
        if row.get('batch_id') != batch_start or row.get('batch_size') != expected_size:
            raise ValueError('Stored batch identity does not match the fixed batch schedule')
        if not isinstance(ids, list) or len(ids) != len(prefix)+config['length']:
            raise ValueError('Stored sample has invalid length')
        if any(type(v) is not int or v < 0 or v >= vocab_size or v == mask_id for v in ids):
            raise ValueError('Stored sample has invalid token IDs')
        if ids[:len(prefix)] != prefix or row.get('prefix_length') != len(prefix):
            raise ValueError('Stored sample has a changed prefix')
        for key in ('elapsed_seconds', 'backbone_seconds', 'sampling_seconds'):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Stored sample has invalid timing')


def _append_jsonl(path, rows):
    with path.open('a') as stream:
        stream.write(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
        stream.flush()
        os.fsync(stream.fileno())


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=Path)
    parser.add_argument('--mode', choices=['pmi', 'conditional'], default='pmi')
    parser.add_argument('--strength', type=float, default=.1)
    parser.add_argument('--sampling', choices=['joint', 'marginal'], default='joint')
    parser.add_argument('--backend', choices=['reference', 'gpu'], default='reference',
                        help='Exact FP64 inference backend; included in immutable identity')
    parser.add_argument('--backbone-checkpoint', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--sample-offset', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--device', default='cuda')
    prefix = parser.add_mutually_exclusive_group()
    prefix.add_argument('--prefix', default='')
    prefix.add_argument('--prefix-token-ids', type=int, nargs='+')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--score-gpt2', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--warmup', type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.samples, args.batch_size, args.length, args.steps) < 1:
        raise ValueError('Sample/batch counts, length and steps must be positive')
    if args.sample_offset < 0 or args.warmup < 0:
        raise ValueError('Sample offset and warmup must be nonnegative')
    if not math.isfinite(args.strength) or args.strength < 0:
        raise ValueError('Strength must be finite and nonnegative')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if not args.score_only and (args.counts is None or not args.counts.is_file()):
        raise ValueError('An existing --counts file is required')
    return args


def _score_existing(args):
    manifest = json.loads((args.output/'manifest.json').read_text())
    if manifest['backbone'].get('synthetic_only'):
        raise ValueError('Synthetic fixtures must not be externally scored as text results')
    records = _read_records(args.output/'samples.jsonl')
    config = manifest['config']
    _validate_records(records, config, canonical_hash(manifest), manifest['prefix_token_ids'],
                      manifest['backbone']['vocab_size'], manifest['backbone']['mask_id'])
    if len(records) != config['samples']:
        raise ValueError('Scoring requires the complete immutable sample set')
    result = score_gpt2(records, args.device)
    result['manifest_sha256'] = canonical_hash(manifest)
    atomic_json(result, args.output/'gpt2-large.json')


def _run(args):
    before = time.perf_counter()
    model = SyntheticBackbone(device=args.device) if args.synthetic else FrozenMDLM(
        args.backbone_checkpoint, device=args.device, cache_dir=args.cache_dir)
    synchronize(args.device)
    backbone_setup = time.perf_counter()-before
    if args.prefix and model.tokenizer is None:
        raise ValueError('Synthetic fixture does not tokenize text; use --prefix-token-ids')
    prefix = args.prefix_token_ids or (model.tokenizer.encode(args.prefix, add_special_tokens=False)
                                       if args.prefix else [])
    if any(v < 0 or v >= model.vocab_size or v == model.mask_id for v in prefix):
        raise ValueError('Prefix must contain valid observed clean tokens')
    before = time.perf_counter()
    head = CountBigramHead.load(args.counts, mode=args.mode, strength=args.strength).to(args.device)
    if head.vocab_size != model.vocab_size:
        raise ValueError('Count/backbone vocabulary mismatch')
    synchronize(args.device)
    count_loading = time.perf_counter()-before
    before = time.perf_counter()
    potential_class, _, _ = inference_components(args.backend)
    potential = potential_class.from_head(head)
    synchronize(args.device)
    potential_setup = time.perf_counter()-before
    count_sha = file_sha256(args.counts)
    sidecar_path = args.counts.with_suffix('.json')
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None
    if sidecar is not None and sidecar.get('counts_sha256') != count_sha:
        raise ValueError('Counts sidecar does not match the count checkpoint')
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()
                     if key not in ('output', 'resume', 'score_gpt2', 'score_only', 'warmup')}
    source_files = [
        'scripts/evaluate_chain_sparse_count.py', 'chain_crf/sparse_count.py',
        'chain_crf/counts.py', 'chain_crf/backbone.py', 'chain_crf/data.py',
        'chain_crf/generation.py', 'scripts/evaluate_chain_crf.py',
        'scripts/train_chain_crf.py', 'chain_crf/core.py', 'chain_crf/heads.py',
        'models/dit.py', 'configs/model/small.yaml', 'scripts/prepare_released_mdlm_owt.py',
    ]
    if args.backend == 'gpu':
        source_files.append('chain_crf/sparse_count_gpu.py')
    manifest = {
        'format': 'chain_sparse_count_eval_v1', 'config': configuration,
        'backbone': model.provenance, 'backbone_identity_sha256': canonical_hash(model.provenance),
        'counts_sha256': count_sha, 'counts_metadata': sidecar,
        'counts_metadata_sha256': file_sha256(sidecar_path) if sidecar else None,
        'source_sha256': {name: file_sha256(ROOT/name) for name in source_files},
        'prefix_token_ids': prefix,
        'method': {'support': 'full vocabulary excluding absorbing mask',
                   'adjacency': 'all original adjacent positions including visible boundaries',
                   'potential': args.mode, 'strength': args.strength,
                   'smoothing': head.smoothing, 'inference_dtype': 'float64',
                   'inference_backend': args.backend,
                   'static_potential_timing': 'excluded from generation; reported in setup_runs.jsonl',
                   'reveal_seed': '1729 + draw_id',
                   'token_seed': '2718 + first draw_id of original batch',
                   'reveal_counts': 'ceil((step+1)*generated_length/steps)'},
    }
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume manifest does not match this code/model/configuration')
    else:
        if (args.output/'samples.jsonl').exists():
            raise ValueError('Refusing to adopt samples without their original manifest')
        atomic_json(manifest, manifest_path)
    manifest_hash = canonical_hash(manifest)
    records_path = args.output/'samples.jsonl'
    records = _read_records(records_path)
    _validate_records(records, configuration, manifest_hash, prefix, model.vocab_size, model.mask_id)
    kwargs = dict(length=args.length, steps=args.steps, sampling=args.sampling,
                  temperature=args.temperature, device=args.device, prefix=prefix,
                  inference_backend=args.backend)
    warmup_seconds = 0.
    if len(records) < args.samples:
        before = time.perf_counter()
        for index in range(args.warmup):
            generate_sparse_count(model, potential, batch_size=args.batch_size,
                                  sample_offset=args.sample_offset+args.samples+index*args.batch_size,
                                  **kwargs)
        synchronize(args.device)
        warmup_seconds = time.perf_counter()-before
    setup = {'manifest_sha256': manifest_hash, 'resumed': args.resume,
             'backbone_loading_seconds': backbone_setup, 'count_loading_seconds': count_loading,
             'static_potential_seconds': potential_setup, 'warmup_seconds': warmup_seconds,
             'warmup_batches': args.warmup if len(records) < args.samples else 0,
             'observed_count_pairs': head.pair_keys.numel(),
             'sparse_corrections': potential.sparse._nnz(), 'vocab_size': model.vocab_size,
             'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
             'device': str(potential.left.device),
             'device_name': torch.cuda.get_device_name(potential.left.device)
             if potential.left.device.type == 'cuda' else 'CPU'}
    _append_jsonl(args.output/'setup_runs.jsonl', [setup])
    del head  # The static potential now owns everything needed for inference.
    # Replay an interrupted partial batch from its ORIGINAL offset. Otherwise
    # the per-batch RNG seed would change even though draw IDs stayed the same.
    first_batch = len(records)//args.batch_size*args.batch_size
    for offset in range(first_batch, args.samples, args.batch_size):
        size = min(args.batch_size, args.samples-offset)
        if offset+size <= len(records):
            continue
        tokens, timing = generate_sparse_count(model, potential, batch_size=size,
                                               sample_offset=args.sample_offset+offset, **kwargs)
        token_rows = tokens.cpu().tolist()
        existing = max(0, len(records)-offset)
        for index in range(existing):
            if records[offset+index]['token_ids'] != token_rows[index]:
                raise ValueError('Partial-batch replay changed previously persisted tokens')
        batch = []
        for index in range(existing, size):
            row = token_rows[index]
            record = {'sample_id': offset+index, 'draw_id': args.sample_offset+offset+index,
                      'token_ids': row, 'prefix_length': len(prefix),
                      'text': model.tokenizer.decode(row) if model.tokenizer else ' '.join(map(str, row)),
                      'batch_id': offset, 'batch_size': size, 'manifest_sha256': manifest_hash,
                      **timing}
            for key in ('elapsed_seconds', 'backbone_seconds', 'sampling_seconds'):
                record[key] = timing[key]/size
            batch.append(record)
        _append_jsonl(records_path, batch)
        records.extend(batch)
        print(json.dumps({'completed': len(records), 'target': args.samples, 'batch': timing}), flush=True)
    elapsed = sum(row['elapsed_seconds'] for row in records)
    setup_runs = _read_records(args.output/'setup_runs.jsonl')
    result = {'manifest_sha256': manifest_hash, 'samples': len(records),
              'elapsed_seconds': elapsed, 'seconds_per_sample': elapsed/len(records),
              'backbone_seconds': sum(row['backbone_seconds'] for row in records),
              'sampling_seconds': sum(row['sampling_seconds'] for row in records),
              'backbone_calls_per_sample': sum(row['backbone_calls'] for row in records)/len(records),
              'generated_tokens_per_second': len(records)*args.length/elapsed,
              'setup_invocations': len(setup_runs),
              'static_potential_setup_seconds_first_invocation': setup_runs[0]['static_potential_seconds'],
              'static_potential_setup_seconds_all_invocations': sum(row['static_potential_seconds'] for row in setup_runs),
              'timing_scope': 'generation includes backbone/unaries/DP/sampling/commit; setup and warmup separate',
              **token_statistics([row['token_ids'][len(prefix):] for row in records])}
    atomic_json(result, args.output/'metrics.json')
    print(json.dumps(result, indent=2), flush=True)
    if args.score_gpt2:
        if args.synthetic:
            raise ValueError('Synthetic fixtures must not be externally scored as text results')
        del model, potential
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
        _score_existing(args)


def main(argv=None):
    args = _arguments(argv)
    if args.score_only:
        _score_existing(args)
        return
    if args.synthetic and args.score_gpt2:
        raise ValueError('Synthetic fixtures must not be externally scored as text results')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Output exists; choose a new run or explicitly --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    # An advisory lock is released by the OS even after a crash/SIGKILL. The
    # persistent lock file is not a stale-lock marker and must not be unlinked
    # (unlinking it can allow two writers to lock different inodes).
    with (args.output/'.evaluation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another evaluator is writing this run') from error
        try:
            _run(args)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == '__main__':
    main()
