#!/usr/bin/env python3
"""Save exact generation histories for examples, never for latency benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chain_crf.backbone import FrozenMDLM, SyntheticBackbone, file_sha256
from chain_crf.data import atomic_json
from chain_crf.trace import generate_with_trace, masked_state_text
from scripts.evaluate_chain_crf import load_head


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['backbone','count','global','contextual','independent'], default='backbone')
    parser.add_argument('--backbone-checkpoint', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--head', type=Path)
    parser.add_argument('--counts', type=Path)
    parser.add_argument('--count-mode', choices=['pmi','conditional'], default='pmi')
    parser.add_argument('--strength', type=float, default=.1)
    parser.add_argument('--sampling', choices=['joint','marginal'], default='joint')
    parser.add_argument('--inference', choices=['dense','segments'], default='segments')
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--sample-offset', type=int, default=20000)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args(argv)
    if min(args.length, args.steps, args.samples, args.batch_size, args.k) < 1 or args.sample_offset < 0:
        raise ValueError('Positive counts and a nonnegative draw offset are required')
    if not args.temperature > 0:
        raise ValueError('Temperature must be positive')
    if args.synthetic and args.prefix:
        raise ValueError('Synthetic fixture cannot tokenize a text prefix')
    if args.output.exists():
        raise FileExistsError('Choose a new trace output; existing histories are never overwritten')
    model = SyntheticBackbone(device=args.device) if args.synthetic else FrozenMDLM(
        args.backbone_checkpoint, device=args.device, cache_dir=args.cache_dir)
    head, head_info = load_head(args, model)
    prefix = model.tokenizer.encode(args.prefix, add_special_tokens=False) if args.prefix else []
    root = Path(__file__).resolve().parents[1]
    sources = ['scripts/trace_chain_generation.py','scripts/evaluate_chain_crf.py',
               'scripts/train_chain_crf.py','chain_crf/trace.py','chain_crf/generation.py',
               'chain_crf/core.py','chain_crf/heads.py','chain_crf/counts.py',
               'chain_crf/segments.py','chain_crf/backbone.py','chain_crf/data.py','models/dit.py']
    manifest = {
        'schema': 'chain_generation_trace_manifest_v1',
        'config': {key: str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        'backbone': model.provenance, 'head': head_info,
        'source_sha256': {name:file_sha256(root/name) for name in sources},
        'benchmark_eligible': False,
        'selection': 'all_requested_consecutive_draw_ids_no_quality_filter',
        'display_note': 'Mask runs represent exact positions; visible spans are decoded separately. Token IDs are authoritative.',
    }
    args.output.mkdir(parents=True)
    atomic_json(manifest, args.output/'manifest.json')
    traces = []
    for offset in range(0, args.samples, args.batch_size):
        size = min(args.batch_size, args.samples-offset)
        _, _, trace = generate_with_trace(
            model, head, args.mode, length=args.length, steps=args.steps, batch_size=size,
            k=args.k, sampling=args.sampling, temperature=args.temperature, device=args.device,
            sample_offset=args.sample_offset+offset, prefix=prefix, inference=args.inference)
        for example in trace['examples']:
            example['sample_id'] = offset+example['batch_row']
            example['final_text'] = masked_state_text(example['final_token_ids'], model.mask_id, model.tokenizer)
            for event in example['events']:
                for when in ('before','after'):
                    event[f'{when}_display_text'] = masked_state_text(
                        event[f'{when}_token_ids'], model.mask_id, model.tokenizer)
        traces.append(trace)
        atomic_json({'manifest_schema':manifest['schema'], 'complete':offset+size == args.samples,
                     'samples':offset+size, 'batches':traces}, args.output/'traces.json')
    print(f'Saved {args.samples} exact generation histories. Trace runtimes are not benchmark measurements.')


if __name__ == '__main__':
    main()
