#!/usr/bin/env python3
"""Count within-row adjacent pairs from the declared training split only."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from chain_crf.counts import CountBigramHead
from chain_crf.backbone import file_sha256
from chain_crf.data import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vocab-size', type=int, default=50258)
    p.add_argument('--mask-id', type=int, default=50257)
    p.add_argument('--max-tokens', type=int, default=10000000)
    p.add_argument('--smoothing', type=float, default=.1)
    args = p.parse_args()
    if args.max_tokens<2 or args.vocab_size<2 or not 0<=args.mask_id<args.vocab_size:
        raise ValueError('Need valid vocabulary/mask ID and at least two training tokens')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite counts: {args.output}')
    if args.data.suffix == '.jsonl':
        def rows():
            with args.data.open() as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if row.get('split','train')!='train':
                            raise ValueError('Count fitting requires training rows')
                        if not isinstance(row.get('input_ids'),list) or any(type(t) is not int for t in row['input_ids']):
                            raise ValueError('Count input_ids must be lists of integer token IDs')
                        yield row['input_ids']
        source = {'format': 'jsonl'}
    else:
        payload = torch.load(args.data, map_location='cpu', weights_only=True)
        source = payload.get('provenance', {})
        if source.get('split') in ('dev', 'test', 'validation'):
            raise ValueError('Count fitting requires training data')
        if payload['tokens'].dtype!=torch.long or payload['tokens'].ndim!=2:
            raise ValueError('Prepared count tokens must be a 2-D int64 tensor')
        def rows():
            yield from payload['tokens']
    seen = {'tokens': 0, 'rows': 0, 'edges': 0}
    def limited():
        for row in rows():
            remaining = args.max_tokens - seen['tokens']
            if remaining <= 0:
                break
            ids = torch.as_tensor(row, dtype=torch.long)
            if ids.ndim!=1:
                raise ValueError('Each count row must be one token sequence')
            ids=ids[:remaining]
            if ids.eq(args.mask_id).any():
                raise ValueError('Count training data contains masks')
            seen['tokens'] += len(ids)
            seen['rows'] += 1
            seen['edges'] += max(0, len(ids)-1)
            yield ids
    model = CountBigramHead(args.vocab_size, smoothing=args.smoothing).fit(limited())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    result = dict(seen, data_sha256=file_sha256(args.data), counts_sha256=file_sha256(args.output),
                  source=source, distinct_pairs=model.pair_keys.numel(),
                  boundary_rule='No edges between input rows; document/chunk boundaries are never joined.')
    atomic_json(result, args.output.with_suffix('.json'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
