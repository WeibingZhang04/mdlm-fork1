#!/usr/bin/env python3
"""Generate once per configuration; report quality and elapsed time, no CIs."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from chain_crf.backbone import FrozenMDLM,SyntheticBackbone,file_sha256
from chain_crf.counts import CountBigramHead
from chain_crf.data import atomic_json,load_token_data,canonical_hash
from chain_crf.generation import generate,denoising,token_statistics
from scripts.train_chain_crf import make_head


def load_head(args,backbone):
    if args.mode=='backbone':
        return None,{}
    if args.mode=='count':
        if args.counts is None:
            raise ValueError('--counts required for count baseline')
        head=CountBigramHead.load(args.counts,mode=args.count_mode,strength=args.strength).to(args.device)
        if head.vocab_size!=backbone.vocab_size:
            raise ValueError('Count/tokenizer vocabulary mismatch')
        return head,{'count_sha256':file_sha256(args.counts)}
    if args.head is None:
        raise ValueError('--head required for trained model')
    payload=torch.load(args.head,map_location='cpu',weights_only=True)
    config=payload['config']
    if config['mode']!=args.mode or config['vocab_size']!=backbone.vocab_size:
        raise ValueError('Head architecture or vocabulary mismatch')
    if config['hidden_size']!=backbone.hidden_size:
        raise ValueError('Hidden size mismatch')
    trained_backbone=payload['identity']['backbone']
    if payload.get('identity_sha256')!=canonical_hash(payload['identity']):
        raise ValueError('Head training identity checksum mismatch')
    if payload['identity'].get('config')!=config:
        raise ValueError('Head configuration differs from its training identity')
    for key in ('source_safetensors_sha256','tokenizer_revision','synthetic_only','seed'):
        if trained_backbone.get(key)!=backbone.provenance.get(key):
            raise ValueError(f'Head was trained with a different backbone: {key}')
    head=make_head(args.mode,config['vocab_size'],config['hidden_size'],config['rank'],config.get('mlp_size',128))
    head.load_state_dict(payload['head'],strict=True)
    head.to(args.device).eval()
    return head,{'head_sha256':file_sha256(args.head),'training_step':payload['step'],
                 'training_identity':payload['identity'],'training_config':config}


@torch.no_grad()
def score_gpt2(records,device,model_name='gpt2-large',
               revision='32b71b12589c2f8d625668d2335a01cac3249519'):
    """Teacher-forced external-LM score of generated tokens, excluding prefix."""
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(model_name,revision=revision).to(device).eval()
    total_nll=0.
    total_tokens=0
    individual=[]
    for row in records:
        ids=torch.tensor([row['token_ids']],device=device)
        if ids.shape[1]<2:
            raise ValueError('External scoring needs at least two tokens per sample')
        if ids.shape[1]>model.config.n_positions:
            raise ValueError('External scorer window exceeded; use shorter samples or a windowed scorer')
        logits=model(ids).logits[0,:-1].float()
        targets=ids[0,1:]
        losses=torch.nn.functional.cross_entropy(logits,targets,reduction='none')
        start=max(0,int(row.get('prefix_length',0))-1)
        value=float(losses[start:].sum())
        count=len(losses)-start
        if count<1:
            raise ValueError('No generated tokens can be scored in this sample')
        total_nll+=value
        total_tokens+=count
        individual.append({'sample_id':row['sample_id'],'nll':value,'scored_tokens':count})
    return {'model':model_name,'requested_revision':revision,
            'model_commit':getattr(model.config,'_commit_hash',None),
            'total_nll':total_nll,'scored_tokens':total_tokens,
            'mean_nll':total_nll/max(1,total_tokens),'perplexity':math.exp(total_nll/max(1,total_tokens)),
            'samples':individual}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['backbone','count','global','contextual','independent'],default='backbone')
    p.add_argument('--backbone-checkpoint',type=Path)
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--head',type=Path)
    p.add_argument('--counts',type=Path)
    p.add_argument('--count-mode',choices=['pmi','conditional'],default='pmi')
    p.add_argument('--strength',type=float,default=.1)
    p.add_argument('--sampling',choices=['joint','marginal'],default='joint')
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--k',type=int,default=64)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--device',default='cuda')
    p.add_argument('--prefix',default='')
    p.add_argument('--dev-data',type=Path)
    p.add_argument('--dev-examples',type=int,default=32)
    p.add_argument('--denoise-only',action='store_true')
    p.add_argument('--score-gpt2',action='store_true')
    p.add_argument('--score-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--warmup',type=int,default=1)
    args=p.parse_args(argv)
    if args.samples<1 or args.batch_size<1 or args.dev_examples<1:
        raise ValueError('Sample, batch and development counts must be positive')
    records_path=args.output/'samples.jsonl'
    if args.score_only:
        records=[json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        result=score_gpt2(records,args.device)
        atomic_json(result,args.output/'gpt2-large.json')
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError('Output exists; choose a new run or explicitly --resume')
    args.output.mkdir(parents=True,exist_ok=True)
    model=SyntheticBackbone(device=args.device) if args.synthetic else FrozenMDLM(
        args.backbone_checkpoint,device=args.device,cache_dir=args.cache_dir)
    head,head_info=load_head(args,model)
    if args.prefix and model.tokenizer is None:
        raise ValueError('Synthetic fixture does not tokenize text')
    prefix=model.tokenizer.encode(args.prefix,add_special_tokens=False) if args.prefix else []
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                   if k not in ('resume','score_gpt2','warmup')}
    source_root=Path(__file__).resolve().parents[1]
    source_files=['scripts/evaluate_chain_crf.py','scripts/train_chain_crf.py',
                  'chain_crf/generation.py','chain_crf/core.py','chain_crf/heads.py',
                  'chain_crf/counts.py','chain_crf/backbone.py','chain_crf/data.py','models/dit.py']
    manifest={'config':configuration,'backbone':model.provenance,'head':head_info,
              'dev_data_sha256':file_sha256(args.dev_data) if args.dev_data else None,
              'source_sha256':{name:file_sha256(source_root/name) for name in source_files}}
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old!=manifest:
            raise ValueError('Resume manifest does not match this code/model/configuration')
    else:
        atomic_json(manifest,manifest_path)
    if args.dev_data:
        data,_,_=load_token_data(args.dev_data,length=args.length,vocab_size=model.vocab_size,
                                 mask_id=model.mask_id,max_examples=args.dev_examples)
        result=denoising(model,data,head,args.mode,k=args.k,device=args.device,batch_size=args.batch_size)
        atomic_json({'rows':result,'data_sha256':file_sha256(args.dev_data)},args.output/'denoising.json')
        print(json.dumps({'denoising':result}),flush=True)
    if args.denoise_only:
        if not args.dev_data:
            raise ValueError('--denoise-only needs --dev-data')
        return
    kwargs=dict(length=args.length,steps=args.steps,k=args.k,sampling=args.sampling,
                temperature=args.temperature,device=args.device,prefix=prefix)
    for _ in range(args.warmup):
        # Same shape and path; these samples are discarded and not timed in totals.
        generate(model,head,args.mode,batch_size=args.batch_size,sample_offset=1_000_000,**kwargs)
    records=[json.loads(line) for line in records_path.read_text().splitlines() if line.strip()] if records_path.exists() else []
    if [r['sample_id'] for r in records]!=list(range(len(records))):
        raise ValueError('Incomplete or duplicated sample IDs')
    if len(records)>args.samples:
        raise ValueError('Existing sample count exceeds requested target')
    for offset in range(len(records),args.samples,args.batch_size):
        size=min(args.batch_size,args.samples-offset)
        tokens,timing=generate(model,head,args.mode,batch_size=size,sample_offset=offset,**kwargs)
        batch=[]
        for index,row in enumerate(tokens.cpu().tolist()):
            record={'sample_id':offset+index,'token_ids':row,'prefix_length':len(prefix),
                    'text':model.tokenizer.decode(row) if model.tokenizer else ' '.join(map(str,row)),
                    'batch_id':offset,'batch_size':size,**timing}
            # Per-sample normalized times permit summation without double counting.
            for key in ('elapsed_seconds','backbone_seconds','sampling_seconds'):
                record[key]=timing[key]/size
            batch.append(record)
        with records_path.open('a') as f:
            for row in batch:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush()
        records.extend(batch)
        print(json.dumps({'completed':len(records),'target':args.samples,'batch':timing}),flush=True)
    elapsed=sum(r['elapsed_seconds'] for r in records)
    result={'samples':len(records),'elapsed_seconds':elapsed,'seconds_per_sample':elapsed/len(records),
            'backbone_seconds':sum(r['backbone_seconds'] for r in records),
            'sampling_seconds':sum(r['sampling_seconds'] for r in records),
            'backbone_calls_per_sample':sum(r['backbone_calls'] for r in records)/len(records),
            'generated_tokens_per_second':len(records)*args.length/elapsed,
            **token_statistics([r['token_ids'][r['prefix_length']:] for r in records])}
    atomic_json(result,args.output/'metrics.json')
    print(json.dumps(result,indent=2),flush=True)
    if args.score_gpt2:
        del head,model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        atomic_json(score_gpt2(records,args.device),args.output/'gpt2-large.json')


if __name__=='__main__':
    main()
