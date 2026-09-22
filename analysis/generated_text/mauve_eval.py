#!/usr/bin/env python3
"""DO NOT touch other people's files. DO NOT cancel other people's jobs.

MAUVE via the authors' package (Pillutla et al., NeurIPS 2021/JMLR 2023).
Pinned GPT-2-large terminal hidden states; see README.md for exact protocol.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import zipfile
from analyze import (REPO, REVISION, account, owned, verify, read, fresh, save,
                     records, content_tokens, csv_file, write_text, sha)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();account();m=verify(a.manifest)
    root=Path(m['output']);out=fresh(root/'mauve')
    cache=owned(Path(m['cache'])/'huggingface')
    os.environ['HF_HUB_CACHE']=str(cache)
    import datasets
    import numpy as np
    import torch
    import mauve
    from transformers import AutoTokenizer
    sys.path.insert(0,str(REPO))
    from evaluation.generation_metrics import TransformersReferenceLMScorer
    # Reuse only already cached pinned data; no silent revision/network fallback.
    ds=datasets.load_dataset('Skylion007/openwebtext','plain_text',
        revision='79d93d786212f7344586290adb811d4ae6a1762c',split='train',
        cache_dir=str(cache),trust_remote_code=False,
        download_config=datasets.DownloadConfig(local_files_only=True))
    if len(ds)!=8013769:raise ValueError('Unexpected OpenWebText source length')
    ids=list(range(7913769,8013769));random.Random(20260922).shuffle(ids)
    reference=[]
    for i in ids:
        text=ds[i]['text']
        if text and text.strip():reference.append(dict(source_row=i,text=text))
        if len(reference)==500:break
    if len(reference)!=500:raise ValueError('Not enough nonempty held-out documents')
    write_text(out/'reference.jsonl',''.join(json.dumps(x)+'\n' for x in reference))
    scorer=TransformersReferenceLMScorer('gpt2-large',revision=REVISION,device='cuda',batch_size=1,max_length=1024,dtype='float32')
    decoder=AutoTokenizer.from_pretrained('openai-community/gpt2',
        revision='607a30d783dfa663caf39e06633721c8d4cfcd7e',
        trust_remote_code=False)
    def features(texts,name):
        values=[]
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
            for text in texts:
                enc=scorer.tokenizer(text,return_tensors='pt',truncation=True,max_length=1024,add_special_tokens=True)
                if enc['input_ids'].shape[1]==0:
                    # An empty document is a real failure case, retained as EOS.
                    enc={'input_ids':torch.tensor([[scorer.tokenizer.eos_token_id]]),'attention_mask':torch.ones((1,1),dtype=torch.long)}
                enc={k:v.to(scorer.device) for k,v in enc.items()}
                values.append(scorer.model.transformer(**enc).last_hidden_state[0,-1].float().cpu().numpy())
        array=np.stack(values)
        with owned(out/(name+'.npy')).open('xb') as f:np.save(f,array,allow_pickle=False)
        return array
    real=features([r['text'] for r in reference],'human_features')
    results=[]
    for c in m['cells']:
        rs=records(m['output'],c)
        texts=[decoder.decode(content_tokens(r['sample_token_ids'])[0],skip_special_tokens=False) for r in rs]
        generated=features(texts,c['id']+'_features')
        for seed in (0,1,2):
            v=mauve.compute_mauve(p_features=real,q_features=generated,num_buckets='auto',seed=seed,verbose=False)
            value=float(v.mauve)
            if not math.isfinite(value):raise ValueError('MAUVE returned non-finite score')
            results.append(dict(cell=c['id'],clustering_seed=seed,mauve=value,
                frontier_integral=float(v.frontier_integral),human_samples=500,generated_samples=500,
                max_feature_tokens=1024,status='exploratory; three seeds are algorithm sensitivity, not independent data'))
    csv_file(out/'results.csv',results)
    import importlib.metadata
    save(out/'provenance.json',dict(reference_sha256=sha(out/'reference.jsonl'),
        dataset_revision='79d93d786212f7344586290adb811d4ae6a1762c',heldout_interval=[7913769,8013769],
        heldout_scope='Excluded from these adapters\' training; not a claim about backbone pretraining.',
        dataset_fingerprint=ds._fingerprint,scorer=scorer.runtime_identity(),
        feature_policy='GPT-2-large FP32 last-layer terminal hidden state, max 1024 tokens, no padding, batch 1',
        text_policy='Generated text before first non-leading EOS, remove leading BOS; empty text encoded as EOS.',
        packages={n:importlib.metadata.version(n) for n in ('mauve-text','faiss-cpu','numpy','scikit-learn','datasets')},
        citations=['https://arxiv.org/abs/2102.01454','https://www.jmlr.org/papers/v24/23-0023.html','https://github.com/krishnap25/mauve']))
    import csv
    with owned(root/'analysis/summary.csv').open() as f:combined=list(csv.DictReader(f))
    for row in combined:
        vals=[x['mauve'] for x in results if x['cell']==row['cell']]
        row.update(mauve_median=statistics.median(vals),mauve_min=min(vals),mauve_max=max(vals))
    csv_file(root/'comparison.csv',combined)
    write_text(root/'README-results.md','# 500-sample generation comparison\n\n'
        'comparison.csv combines the automatic metrics. analysis/report.md explains PPL and prefix coverage.\n'
        'MAUVE at 500 samples is exploratory; seed ranges measure clustering sensitivity only.\n'
        'blind/blind-review.zip is safe to share with independent judges. It contains no answer key.\n'
        'Judge each pair before seeing model identities or automatic scores. Blind preference is pending.\n'
        'Training used one seed and different allocations; these are not multi-seed causal estimates.\n')
    # This broader archive contains condition labels; do not send it to blind judges.
    with zipfile.ZipFile(owned(root/'analysis-results.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for rel in ('comparison.csv','README-results.md','mauve/results.csv','mauve/provenance.json'):
            z.write(owned(root/rel),arcname=rel)
        for f in sorted((root/'analysis').iterdir()):
            if f.suffix in ('.csv','.png','.md','.json'):z.write(owned(f),arcname=str(f.relative_to(root)))
    save(out/'completed.json',dict(cells=15,samples_per_cell=500))


if __name__=='__main__':main()
