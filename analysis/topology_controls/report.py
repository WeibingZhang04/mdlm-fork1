#!/usr/bin/env python3
"""DO NOT touch other people's files. DO NOT touch other people's jobs.
Do not interfere with other people's processes. See README.md for definitions.
"""
import argparse
import json
import os
from pathlib import Path
import random
import statistics as st
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generated_text'))
from analyze import (REPO,account,verify,owned,read,sha,records,fresh,save,write_text,csv_file,
    content_tokens,repetition,distinct,pooled_ppl,bootstrap_ppl,interval,prefix_score,REVISION)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();account();m=verify(a.manifest);root=owned(m['output'])
    os.environ['HF_HUB_CACHE']=str(owned(m['cache'])/'huggingface');sys.path.insert(0,str(REPO))
    from evaluation.generation_metrics import TransformersReferenceLMScorer
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scorer=TransformersReferenceLMScorer('gpt2-large',revision=REVISION,device='cuda',batch_size=1,max_length=1024,dtype='float32')
    out=fresh(root/'analysis');summary=[];all_samples=[];by_cell={};geometry=[]
    for c in m['cells']:
        rows=records(root,c);by_cell[c['id']]=rows;seqs=[];prefixes=[];samples=[]
        for r in rows:
            tokens,eos=content_tokens(r['sample_token_ids']);seqs.append(tokens)
            prefix=prefix_score(scorer,r['text']);prefixes.append(prefix)
            samples.append(dict(cell=c['id'],pair_key=r['pair_key'],pair_seed=r['pair_seed'],
                ppl=r['reference_lm']['perplexity'],scored_length=r['reference_lm']['token_count'],
                repetition4=repetition(tokens),content_length=len(tokens),terminal_eos=eos is not None,
                prefix256_ppl=prefix['perplexity'] if prefix else None))
        all_samples.extend(samples);valid=[v for v in prefixes if v is not None];lo,hi=bootstrap_ppl(rows)
        gen=read(root/'cells'/c['id']/'generation/summary.json')['groups'][0]
        done=read(root/'cells'/c['id']/'completed.json')
        summary.append(dict(cell=c['id'],training_cache=c['cache_precision'],steps=c['steps'],variant=c['variant'],
            samples=len(rows),ppl=pooled_ppl([r['reference_lm'] for r in rows]),ppl_ci_low=lo,ppl_ci_high=hi,
            repetition4=st.mean(r['repetition4'] for r in samples if r['repetition4'] is not None),
            distinct4=distinct(seqs),median_scored_length=st.median(r['scored_length'] for r in samples),
            early_eos256=sum(r['terminal_eos'] and r['content_length']<256 for r in samples)/len(rows),
            no_eos=sum(not r['terminal_eos'] for r in samples)/len(rows),prefix256_eligible=len(valid),
            prefix256_ppl=pooled_ppl(valid),prefix256_distinct4=distinct([s[:256] for s in seqs if len(s)>=256]),
            mean_measured_nfe=st.mean(r['measured_nfe'] for r in rows),
            generation_seconds=gen['wall_clock_seconds'],gpu=done['gpu']))
        trace=root/'cells'/c['id']/'graph-trace.jsonl';assert sha(trace)==done['graph_sha256']
        with owned(trace).open() as f: graphs=[json.loads(line) for line in f]
        for label,selected in [('all',graphs),('mask_fraction_le_0.3',[g for g in graphs if g['active']<=.3*1024])]:
            if selected:
                geometry.append(dict(cell=c['id'],mask_bin=label,forward_count=len(selected),
                    isolated_fraction=sum(g['isolated'] for g in selected)/max(1,sum(g['active'] for g in selected)),
                    mean_edges=st.mean(g['edges'] for g in selected),mean_components=st.mean(g['components'] for g in selected),
                    mean_edge_span=st.mean(g['mean_edge_span'] for g in selected)))
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for ax,k in zip(axes,('repetition4','scored_length')):
            usable=[r for r in samples if r[k] is not None and r['ppl'] is not None]
            ax.scatter([r[k] for r in usable],[r['ppl'] for r in usable],s=8,alpha=.3)
            ax.set(xlabel=k,ylabel='Per-sample PPL',yscale='log')
        fig.suptitle(c['id']);fig.tight_layout();fig.savefig(owned(out/(c['id']+'.png')),dpi=140);plt.close(fig)
        print('Analyzed',c['id'],flush=True)
    comparisons=[]
    for ids in m['groups']:
        base={r['pair_key']:r for r in by_cell[ids[0]]}
        assert len({s['gpu'] for s in summary if s['cell'] in ids})==1
        for cid in ids[1:]:
            other={r['pair_key']:r for r in by_cell[cid]};assert base.keys()==other.keys()
            assert all(base[k]['pair_seed']==other[k]['pair_seed'] for k in base)
            keys=sorted(base);rng=random.Random(20260922);diffs=[]
            for _ in range(1000):
                draw=rng.choices(keys,k=len(keys))
                diffs.append(pooled_ppl([other[k]['reference_lm'] for k in draw])-pooled_ppl([base[k]['reference_lm'] for k in draw]))
            lo,hi=interval(diffs)
            comparisons.append(dict(native=ids[0],intervention=cid,
                ppl_difference=pooled_ppl([r['reference_lm'] for r in other.values()])-pooled_ppl([r['reference_lm'] for r in base.values()]),
                ci_low=lo,ci_high=hi))
    for name,rows in [('summary',summary),('samples',all_samples),('paired_differences',comparisons),('graph_geometry',geometry)]:csv_file(out/(name+'.csv'),rows)
    lines=['# DD topology interventions','',
        '500 samples/cell; both 6k DD checkpoints; no retraining. Lower PPL is better.',
        'Each group of four variants shares a GPU. Precision labels describe the training cache; generation cache is always FP32.',
        'Random relabeling preserves native graph size and shape at each current state, but changes token identity and distance.',
        'Prefix256 is conditional on eligibility. Bootstrap intervals are descriptive sampling intervals, not multi-seed evidence.',
        'No MAUVE or LLM judgments are included. Timing includes graph tracing; different actual NFE is not equal compute.','',
        '| Cell | PPL | Rep-4 | Distinct-4 | Median length | Early EOS <256 | Prefix n | Prefix PPL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        fmt=lambda v:'NA' if v is None else f'{v:.3f}'
        lines.append('| '+s['cell']+' | '+' | '.join(fmt(s[k]) for k in ('ppl','repetition4','distinct4','median_scored_length','early_eos256','prefix256_eligible','prefix256_ppl'))+' |')
    write_text(out/'report.md','\n'.join(lines)+'\n')
    save(out/'completed.json',dict(cells=len(summary),samples=len(all_samples),scorer=scorer.runtime_identity()))
    with zipfile.ZipFile(owned(root/'topology-results.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in [root/'manifest.json',*sorted(out.iterdir())]:z.write(owned(f),arcname=str(f.relative_to(root)))


if __name__=='__main__':main()
