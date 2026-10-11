"""Matched, prewarmed A/B benchmark of materialized versus on-demand edges."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chain_crf.backbone import FrozenMDLM
from chain_crf.generation import generate
from chain_crf.core import build_candidates
from chain_crf.fused_pairs import infer_fused_pairs
from chain_crf.segments import segmented_log_marginals
from scripts.evaluate_chain_crf import load_head


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', default=['contextual', 'global', 'count'])
    p.add_argument('--ks', nargs='+', type=int, default=[500])
    p.add_argument('--draws', type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(4)
    base = Path('/u401/n23zhang/rework-data')
    heads = base/'runs/draft-full-scale-training/preliminary-checkpoints_2026_10_01_14_29_45_UTC'
    model = FrozenMDLM(base/'checkpoints/mdlm-owt.pt', device='cuda', cache_dir=base/'hf-cache/hub')
    report = dict(gpu=torch.cuda.get_device_name(), host=socket.gethostname(),
                  torch=torch.__version__, job_id=os.environ.get('SLURM_JOB_ID'),
                  backbone=model.provenance, warmup=[], measurements=[], correctness=[], summaries=[],
                  scope='Generation only, including final noise removal. All measured draw/backend combinations prewarmed; alternating order. Compilation and initial count-table setup are excluded. Count-table setup occurs during the correctness check and is not timed separately. No scoring/loading time. Shared GPU activity may affect timings.')
    report['source_sha256'] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in (ROOT/'chain_crf').glob('*.py')}

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    def emit(kind, row):
        report[kind].append(row)
        save()
        print(kind.upper(), json.dumps(row), flush=True)

    @torch.no_grad()
    def check(head, mode, k):
        # Same real-backbone packet for both inference algorithms; no sampling divergence.
        tokens = torch.full((1, 96), model.mask_id, device='cuda', dtype=torch.long)
        known = model.tokenizer.encode('The model predicts the next word in the sentence.')
        for index in range(0, 96, 7):
            tokens[0, index] = known[(index // 7) % len(known)]
        t = torch.tensor([.5], device='cuda')
        pred = model(tokens, t)
        packet = build_candidates(pred['log_probs'], tokens, model.mask_id, k, vocab_cap=k)
        edge = head(packet.candidate_ids, pred['hidden'], t)
        expected = segmented_log_marginals(packet.unary, edge, packet.masked, segment_batch_size=32)
        actual = infer_fused_pairs(packet, head, pred['hidden'], t,
                                  sampling='log_marginals', segment_batch_size=32)
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
        maximum = (actual.exp() - expected.exp()).abs().max().item()
        assert maximum < .0005, maximum
        emit('correctness', dict(mode=mode, k=k, max_marginal_probability_error=maximum))

    def run(head, mode, k, backend, draw):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        tokens, timing = generate(model, head, mode, length=1024, steps=32,
            batch_size=1, k=k, vocab_cap=k, sampler='ddpm_cache', sampling='joint',
            noise_removal=True, sampling_eps=1e-5, device='cuda', inference=backend,
            segment_batch_size=32, stage_timing=False, sample_offset=draw)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert tokens.shape == (1, 1024) and not tokens.eq(model.mask_id).any()
        return dict(mode=mode, k=k, backend=backend, draw=draw,
                    seconds=timing['elapsed_seconds'], wall_seconds=elapsed,
                    backbone_calls=timing['backbone_calls'], cache_hits=timing['cache_hits'],
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    token_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest())

    for mode in args.modes:
        head, identity = load_head(SimpleNamespace(mode=mode, device='cuda',
            head=heads/(mode+'-step-30000.pt'), count_mode='pmi', strength=.1,
            counts=base/'runs/draft-full-scale-training/train_2026_10_01_05_46_13_UTC/counts/owt-counts.pt'), model)
        report.setdefault('heads', {})[mode] = identity
        for k in args.ks:
            check(head, mode, k)
            warm_hashes = {}
            for draw in range(93001000, 93001000 + args.draws):
                for backend in ('segments', 'fused-pairs'):
                    row = run(head, mode, k, backend, draw)
                    warm_hashes[backend, draw] = row['token_sha256']
                    emit('warmup', row)
            for index, draw in enumerate(range(93001000, 93001000 + args.draws)):
                for backend in (('segments', 'fused-pairs') if index % 2 == 0 else ('fused-pairs', 'segments')):
                    row = run(head, mode, k, backend, draw)
                    assert row['token_sha256'] == warm_hashes[backend, draw], 'Same-backend replay failed'
                    emit('measurements', row)
            selected = [r for r in report['measurements'] if r['mode'] == mode and r['k'] == k]
            seconds = {b: statistics.median(r['seconds'] for r in selected if r['backend'] == b)
                       for b in ('segments', 'fused-pairs')}
            memory = {b: max(r['peak_allocated_gib'] for r in selected if r['backend'] == b)
                      for b in ('segments', 'fused-pairs')}
            equal = all(warm_hashes['segments', d] == warm_hashes['fused-pairs', d]
                        for d in range(93001000, 93001000 + args.draws))
            emit('summaries', dict(mode=mode, k=k, median_seconds=seconds, peak_allocated_gib=memory,
                                  speedup=seconds['segments']/seconds['fused-pairs'],
                                  same_backend_replay=True, same_tokens_across_backends=equal))
        del head
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
