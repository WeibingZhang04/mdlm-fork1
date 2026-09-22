#!/usr/bin/env python3
"""DO NOT touch other people's files. DO NOT cancel other people's jobs.

Read-only inputs; new, user-owned outputs. Metric definitions follow this
repository's evaluation/generation_metrics.py; see README.md for citations.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import random
import statistics

HOME = Path('/u401/n23zhang')
REPO = Path(__file__).resolve().parents[2]
EOS = 50256
REVISION = '32b71b12589c2f8d625668d2335a01cac3249519'
POLICY = 'retokenize_decoded_text_score_through_first_nonleading_eos_v1'


def account():
    if pwd.getpwuid(os.getuid()).pw_name != 'n23zhang':
        raise PermissionError('Only run server commands as n23zhang')


def owned(path):
    """Reject escapes, symlinks outside HOME, and other owners before I/O."""
    p = Path(path).absolute()
    root = HOME.resolve()
    resolved = p.resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError('Path outside n23zhang home: ' + str(p))
    if p.is_symlink() and p.lstat().st_uid != os.getuid():
        raise PermissionError('Symlink is owned by another account: ' + str(p))
    for q in (resolved, *resolved.parents):
        if q == root.parent:
            break
        if q.exists() and q.stat().st_uid != os.getuid():
            raise PermissionError('Not owned by n23zhang: ' + str(q))
    return resolved


def read(path):
    return json.loads(owned(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with owned(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fresh(path):
    p = owned(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


def save(path, value):
    with owned(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def write_text(path, text):
    with owned(path).open('x') as f:
        f.write(text)


def csv_file(path, rows):
    if not rows:
        raise ValueError('No rows')
    with owned(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def verify(manifest):
    m = read(manifest)
    if Path(m['repo']).resolve() != REPO:
        raise ValueError('Use the prepared checkout')
    for name, digest in m['source_sha256'].items():
        if sha(REPO / name) != digest:
            raise ValueError('Source changed after preparation: ' + name)
    return m


def records(root, cell):
    d = owned(Path(root) / 'cells' / cell['id'])
    done = read(d / 'completed.json')
    p = d / 'generation/samples.jsonl'
    if sha(p) != done['samples_sha256']:
        raise ValueError('Sample hash mismatch: ' + cell['id'])
    with owned(p).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != 500 or len({r['pair_key'] for r in rows}) != 500:
        raise ValueError('Expected 500 unique samples per cell')
    for r in rows:
        score = r['reference_lm']
        if score['revision'] != REVISION or score['sequence_policy'] != POLICY:
            raise ValueError('Scoring protocol mismatch')
        if r['sampling_mode'] != cell['mode'] or r['requested_nfe_budget'] != cell['steps'] + 1:
            raise ValueError('Generation protocol mismatch')
    return rows


def content_tokens(tokens):
    """Remove one leading BOS and truncate before the first non-leading EOS."""
    ids = list(tokens)
    start = int(bool(ids) and ids[0] == EOS)
    end = next((i for i in range(start, len(ids)) if ids[i] == EOS), None)
    return ids[start:end], end


def grams(tokens, n=4):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def repetition(tokens):
    g = grams(tokens)
    return 1 - len(set(g)) / len(g) if g else None


def distinct(sequences):
    g = [v for s in sequences for v in grams(s)]
    return len(set(g)) / len(g) if g else None


def pooled_ppl(scores):
    n = sum(s['token_count'] for s in scores if s['mean_nll_nats'] is not None)
    total = math.fsum(s['token_count'] * s['mean_nll_nats'] for s in scores
                      if s['mean_nll_nats'] is not None)
    return math.exp(total / n) if n else None


def interval(values):
    values = sorted(x for x in values if x is not None)
    return [values[int((len(values)-1)*p)] for p in (.025, .975)] if values else [None, None]


def bootstrap_ppl(rows, repetitions=1000):
    rng = random.Random(20260922)
    return interval([pooled_ppl([rng.choice(rows)['reference_lm'] for _ in rows])
                     for _ in range(repetitions)])


def prefix_ids(ids, length=256):
    _, end = content_tokens(ids)
    if len(ids) < length + 1 or (end is not None and end <= length):
        return None
    return ids[:length+1]


def prefix_score(scorer, text, length=256):
    """Exactly 256 predicted GPT-2 tokens, all strictly before terminal EOS."""
    import torch
    ids = scorer.tokenizer(text, add_special_tokens=True, truncation=True,
                           max_length=1024)['input_ids']
    ids = prefix_ids(ids, length)
    if ids is None:
        return None
    x = torch.tensor([ids], device=scorer.device)
    with torch.no_grad(), torch.autocast(device_type=scorer.device.type, enabled=False):
        logits = scorer.model(input_ids=x, attention_mask=torch.ones_like(x)).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2), x[:, 1:])
    return {'token_count': length, 'mean_nll_nats': float(loss), 'perplexity': math.exp(float(loss))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args(); account(); m = verify(a.manifest)
    os.environ['HF_HUB_CACHE'] = str(owned(m['cache']) / 'huggingface')
    import sys
    sys.path.insert(0, str(REPO))
    from evaluation.generation_metrics import TransformersReferenceLMScorer
    scorer = TransformersReferenceLMScorer('gpt2-large', revision=REVISION,
        device='cuda', batch_size=1, max_length=1024, dtype='float32')
    out = fresh(Path(m['output']) / 'analysis')
    all_rows = []; summary = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cell in m['cells']:
        rows = records(m['output'], cell); seqs = []; prefixes = []; measured = []
        for r in rows:
            tokens, eos = content_tokens(r['sample_token_ids']); seqs.append(tokens)
            prefix = prefix_score(scorer, r['text']); prefixes.append(prefix)
            item = dict(cell=cell['id'], model=cell['model'], training_cache=cell['cache_precision'],
                steps=cell['steps'], pair_key=r['pair_key'], pair_seed=r['pair_seed'],
                ppl=r['reference_lm']['perplexity'], mean_nll=r['reference_lm']['mean_nll_nats'],
                scored_length=r['reference_lm']['token_count'], repetition4=repetition(tokens),
                content_length=len(tokens), eos_position=eos,
                prefix256_ppl=prefix['perplexity'] if prefix else None,
                prefix256_nll=prefix['mean_nll_nats'] if prefix else None)
            measured.append(item)
        all_rows.extend(measured)
        lo, hi = bootstrap_ppl(rows)
        valid = [x for x in prefixes if x is not None]
        summary.append(dict(cell=cell['id'], model=cell['model'], training_cache=cell['cache_precision'],
            steps=cell['steps'], samples=len(rows), ppl=pooled_ppl([r['reference_lm'] for r in rows]),
            ppl_ci_low=lo, ppl_ci_high=hi,
            repetition4=statistics.mean(x['repetition4'] for x in measured if x['repetition4'] is not None)
                        if any(x['repetition4'] is not None for x in measured) else None,
            distinct4=distinct(seqs), median_scored_length=statistics.median(x['scored_length'] for x in measured),
            early_eos128=sum(x['eos_position'] is not None and x['content_length']<128 for x in measured)/len(rows),
            early_eos256=sum(x['eos_position'] is not None and x['content_length']<256 for x in measured)/len(rows),
            no_eos=sum(x['eos_position'] is None for x in measured)/len(rows),
            prefix256_eligible=len(valid), prefix256_ppl=pooled_ppl(valid),
            prefix256_distinct4=distinct([s[:256] for s in seqs if len(s)>=256]),
            blind_preference='pending independent review'))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, key in zip(axes[:2], ('repetition4', 'scored_length')):
            usable = [x for x in measured if x[key] is not None and x['ppl'] is not None]
            ax.scatter([x[key] for x in usable], [x['ppl'] for x in usable], alpha=.3, s=9)
            ax.set(xlabel=key, ylabel='Per-sample PPL', yscale='log')
        axes[2].hist([x['scored_length'] for x in measured], bins=25)
        axes[2].set(xlabel='Scored tokens', ylabel='Samples')
        fig.suptitle(cell['id']); fig.tight_layout()
        fig.savefig(owned(out/(cell['id']+'.png')), dpi=160); plt.close(fig)
    csv_file(out/'samples.csv', all_rows); csv_file(out/'summary.csv', summary)
    # Pair by verified seeds, not row position; no claim that topics match.
    comparisons = []
    for model in ('FD', 'DD'):
        for steps in (8, 16, 32):
            groups = [{x['pair_key']: x for x in all_rows if x['model']==model and
                       x['steps']==steps and x['training_cache']==c} for c in ('bf16', 'fp32')]
            if groups[0].keys() != groups[1].keys(): raise ValueError('Pair keys differ')
            keys = sorted(groups[0]); rng = random.Random(20260922)
            if any(groups[0][k]['pair_seed'] != groups[1][k]['pair_seed'] for k in keys):
                raise ValueError('Pair seeds differ')
            diffs = []
            for _ in range(1000):
                ks = rng.choices(keys, k=len(keys))
                scores = [[{'token_count':g[k]['scored_length'], 'mean_nll_nats':g[k]['mean_nll']}
                           for k in ks] for g in groups]
                diffs.append(pooled_ppl(scores[1])-pooled_ppl(scores[0]))
            lo, hi = interval(diffs)
            comparisons.append(dict(model=model, steps=steps, statistic='FP32 minus BF16 corpus PPL',
                ci_low=lo, ci_high=hi))
    csv_file(out/'paired_differences.csv', comparisons)
    lines = ['# Generated-text diagnostics', '', '500 samples/cell. PPL is token-weighted GPT-2-large.',
             'Prefix256 scores are conditional on 256 predicted tokens before EOS; coverage is reported.',
             'Bootstrap intervals describe sample uncertainty, not multiple training seeds.', '',
             '| Cell | PPL | Rep-4 | Distinct-4 | Median scored length | Early EOS <256 | Prefix n | Prefix PPL |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        fmt = lambda x: 'NA' if x is None else f'{x:.3f}'
        lines.append('| '+s['cell']+' | '+' | '.join(fmt(s[k]) for k in
            ('ppl','repetition4','distinct4','median_scored_length','early_eos256','prefix256_eligible','prefix256_ppl'))+' |')
    lines += ['', 'MAUVE is optional and deferred; if run later, its results are stored separately in ../mauve/results.csv.',
              'Blind review is in ../blind/; preferences are pending until independent ratings arrive.']
    write_text(out/'report.md', '\n'.join(lines)+'\n')
    save(out/'completed.json', {'cells':len(summary), 'samples':len(all_rows), 'scorer':scorer.runtime_identity()})


if __name__ == '__main__':
    main()
