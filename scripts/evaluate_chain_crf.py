#!/usr/bin/env python3
"""Generate once per configuration; report quality and elapsed time, no CIs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from chain_crf.backbone import FrozenMDLM,SyntheticBackbone,file_sha256
from chain_crf.counts import CountBigramHead
from chain_crf.data import atomic_json,load_token_data,canonical_hash
from chain_crf.generation import generate,denoising,token_statistics,clean_token_ids
from scripts.train_chain_crf import make_head


def load_continuations(path, *, vocab_size, mask_id):
    """Read identified exact-token prefixes and reference lengths, never text."""
    raw=Path(path).read_bytes()
    rows=[]
    seen=set()
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        if not isinstance(row,dict):
            raise ValueError('Continuation rows must be JSON objects')
        prefix=clean_token_ids(row.get('prefix_input_ids'),vocab_size=vocab_size,
                              mask_id=mask_id,label='Prefix')
        reference=clean_token_ids(row.get('reference_continuation_ids'),vocab_size=vocab_size,
                                 mask_id=mask_id,label='Reference continuation')
        if not prefix or not reference:
            raise ValueError('Continuation rows need nonempty prefix and reference token IDs')
        for key in ('document_id','chunk_id','source_split'):
            if not isinstance(row.get(key),str) or not row[key]:
                raise ValueError(f'Continuation row requires a nonempty {key}')
        if row['chunk_id'] in seen:
            raise ValueError('Duplicate continuation chunk_id')
        seen.add(row['chunk_id'])
        if row.get('prefix_length',len(prefix))!=len(prefix):
            raise ValueError('Declared prefix_length differs from exact token IDs')
        if ('token_start' in row) != ('token_stop' in row):
            raise ValueError('Continuation token offsets must occur together')
        if 'token_start' in row:
            if (type(row['token_start']) is not int or type(row['token_stop']) is not int
                    or row['token_start']<0
                    or row['token_stop']-row['token_start']!=len(prefix)+len(reference)):
                raise ValueError('Continuation token offsets disagree with prefix and suffix lengths')
        rows.append({**row,'prefix_input_ids':prefix,'reference_continuation_ids':reference,
                     'input_index':len(rows)})
    if not rows:
        raise ValueError('Continuation file is empty')
    if len({row['source_split'] for row in rows})!=1:
        raise ValueError('Do not mix validation and test/source splits in one continuation file')
    return rows,hashlib.sha256(raw).hexdigest()


def select_continuations(rows, *, offset, samples, one_per_document=False):
    if offset<0 or samples<1:
        raise ValueError('Continuation offset must be nonnegative and sample count positive')
    if one_per_document:
        by_document={}
        for row in rows:
            by_document.setdefault(row['document_id'],row)
        rows=list(by_document.values())
    if offset+samples>len(rows):
        raise ValueError(f'Requested continuation range exceeds {len(rows)} available examples')
    return rows[offset:offset+samples]


def continuation_batch_plan(rows,batch_size):
    """Batch adjacent identical shapes; no padding and no order changes."""
    if batch_size<1:
        raise ValueError('Continuation batch size must be positive')
    batches=[]
    offset=0
    while offset<len(rows):
        shape=(len(rows[offset]['prefix_input_ids']),len(rows[offset]['reference_continuation_ids']))
        stop=offset+1
        while stop<min(offset+batch_size,len(rows)):
            other=(len(rows[stop]['prefix_input_ids']),len(rows[stop]['reference_continuation_ids']))
            if other!=shape:
                break
            stop+=1
        batches.append((offset,stop-offset))
        offset=stop
    return batches


def continuation_identity(row,file_sha256):
    result={key:row[key] for key in ('document_id','chunk_id','source_split','input_index')}
    result.update({key:row[key] for key in ('source_start_row','source_stop_row','token_start','token_stop')
                   if key in row})
    result.update({'continuation_file_sha256':file_sha256,
                   'prefix_input_ids':row['prefix_input_ids'],
                   'reference_continuation_ids':row['reference_continuation_ids']})
    return result


def validate_continuation_resume(records,rows,batches,file_sha256,*,vocab_size,mask_id):
    for offset,size in batches:
        for index in range(offset,min(offset+size,len(records))):
            record,source=records[index],rows[index]
            expected={**continuation_identity(source,file_sha256),
                      'prefix_length':len(source['prefix_input_ids']),
                      'generated_length':len(source['reference_continuation_ids']),
                      'batch_id':offset,'batch_size':size}
            if any(record.get(key)!=value for key,value in expected.items()):
                raise ValueError('Stored continuation identity/shape does not match its source')
            tokens=clean_token_ids(record.get('token_ids'),vocab_size=vocab_size,
                                   mask_id=mask_id,label='Stored sample')
            prefix=source['prefix_input_ids']
            if (len(tokens)!=len(prefix)+len(source['reference_continuation_ids'])
                    or tokens[:len(prefix)]!=prefix):
                raise ValueError('Stored continuation tokens do not preserve the exact source prefix/length')


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
    p.add_argument('--inference',choices=['dense','segments'],default='dense')
    p.add_argument('--length',type=int,default=256,
                   help='Generated suffix length; continuation-file mode instead uses each reference suffix length')
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--sample-offset',type=int,default=0,
                   help='First draw ID; use disjoint IDs for screening and final evaluation')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--k',type=int,default=64)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--device',default='cuda')
    p.add_argument('--prefix',default='')
    p.add_argument('--continuation-file',type=Path,
                   help='JSONL with exact prefix_input_ids, reference_continuation_ids and source identities')
    p.add_argument('--continuation-offset',type=int,default=0,
                   help='First input row after optional document selection; independent of global draw IDs')
    p.add_argument('--one-per-document',action='store_true',
                   help='Use only the first chunk per source article in continuation-file mode')
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
    if args.sample_offset<0:
        raise ValueError('Sample offset must be nonnegative')
    if args.continuation_offset<0:
        raise ValueError('Continuation offset must be nonnegative')
    if args.continuation_file and args.prefix:
        raise ValueError('Choose --prefix or --continuation-file, not both')
    if not args.continuation_file and (args.continuation_offset or args.one_per_document):
        raise ValueError('Continuation selection options require --continuation-file')
    if args.continuation_file and args.denoise_only:
        raise ValueError('--continuation-file requires generation, not --denoise-only')
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
    continuations=None
    continuation_sha256=None
    if args.continuation_file:
        source_rows,continuation_sha256=load_continuations(args.continuation_file,
            vocab_size=model.vocab_size,mask_id=model.mask_id)
        continuations=select_continuations(source_rows,offset=args.continuation_offset,
            samples=args.samples,one_per_document=args.one_per_document)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                   if k not in ('resume','score_gpt2','warmup')}
    source_root=Path(__file__).resolve().parents[1]
    source_files=['scripts/evaluate_chain_crf.py','scripts/train_chain_crf.py',
                  'chain_crf/generation.py','chain_crf/core.py','chain_crf/heads.py',
                  'chain_crf/counts.py','chain_crf/backbone.py','chain_crf/data.py','models/dit.py']
    if args.inference == 'segments':
        source_files.append('chain_crf/segments.py')
    manifest={'config':configuration,'backbone':model.provenance,'head':head_info,
              'dev_data_sha256':file_sha256(args.dev_data) if args.dev_data else None,
              'source_sha256':{name:file_sha256(source_root/name) for name in source_files}}
    if continuations is not None:
        manifest['continuations']={'file_sha256':continuation_sha256,
            'source_rows':len(source_rows),'selected_rows':len(continuations),
            'selected_chunk_ids':[row['chunk_id'] for row in continuations],
            'prefix_policy':'exact IDs; no tokenization or added special tokens',
            'length_policy':'generate exactly len(reference_continuation_ids) for each row',
            'reference_policy':'reference token values never enter generation; only suffix lengths are used'}
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
                temperature=args.temperature,device=args.device,prefix=prefix,inference=args.inference)
    batches=(continuation_batch_plan(continuations,args.batch_size) if continuations is not None
             else [(offset,min(args.batch_size,args.samples-offset))
                   for offset in range(0,args.samples,args.batch_size)])

    def draw_batch(offset,size,draw_offset):
        batch_kwargs=kwargs
        if continuations is not None:
            batch_kwargs={key:value for key,value in kwargs.items() if key not in ('prefix','length')}
            batch_kwargs.update(prefixes=[row['prefix_input_ids'] for row in continuations[offset:offset+size]],
                                length=len(continuations[offset]['reference_continuation_ids']))
        return generate(model,head,args.mode,batch_size=size,sample_offset=draw_offset,**batch_kwargs)

    records=[json.loads(line) for line in records_path.read_text().splitlines() if line.strip()] if records_path.exists() else []
    if [r['sample_id'] for r in records]!=list(range(len(records))):
        raise ValueError('Incomplete or duplicated sample IDs')
    if [r['draw_id'] for r in records]!=list(range(args.sample_offset,args.sample_offset+len(records))):
        raise ValueError('Stored draw IDs do not match the requested sample offset')
    if len(records)>args.samples:
        raise ValueError('Existing sample count exceeds requested target')
    if continuations is not None:
        validate_continuation_resume(records,continuations,batches,continuation_sha256,
                                     vocab_size=model.vocab_size,mask_id=model.mask_id)
    for warmup_index in range(args.warmup):
        # Same shape/path, discarded; draw IDs lie outside the target range.
        warmup_offset=args.sample_offset+args.samples+warmup_index*args.batch_size
        warmup_size=batches[0][1] if continuations is not None else args.batch_size
        draw_batch(0,warmup_size,warmup_offset)
    # Token RNG is seeded once per original batch. If interruption left only
    # some rows on disk, replay that entire batch at its original draw offset;
    # starting a new batch at len(records) would change the remaining draws.
    for offset,size in batches:
        if offset+size<=len(records):
            continue
        tokens,timing=draw_batch(offset,size,args.sample_offset+offset)
        batch=[]
        for index,row in enumerate(tokens.cpu().tolist()):
            local_id=offset+index
            source=continuations[local_id] if continuations is not None else None
            record={'sample_id':local_id,'draw_id':args.sample_offset+local_id,
                    'token_ids':row,'prefix_length':len(source['prefix_input_ids']) if source else len(prefix),
                    'text':model.tokenizer.decode(row) if model.tokenizer else ' '.join(map(str,row)),
                    'batch_id':offset,'batch_size':size,**timing}
            if source is not None:
                record.update(continuation_identity(source,continuation_sha256))
            # Per-sample normalized times permit summation without double counting.
            for key in ('elapsed_seconds','backbone_seconds','sampling_seconds'):
                record[key]=timing[key]/size
            batch.append(record)
        overlap=min(len(records)-offset,len(batch))
        for previous,replayed in zip(records[offset:offset+overlap],batch[:overlap]):
            keys=('sample_id','draw_id','token_ids','prefix_length','batch_id','batch_size')
            if continuations is not None:
                keys+=tuple(continuation_identity(continuations[previous['sample_id']],continuation_sha256))
            if any(previous[key]!=replayed[key] for key in keys):
                raise ValueError('Replayed partial batch differs from its saved prefix')
        pending=batch[overlap:]
        with records_path.open('a') as f:
            for row in pending:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush()
        records.extend(pending)
        print(json.dumps({'completed':len(records),'target':args.samples,'batch':timing,
                          'replayed_existing_rows':overlap}),flush=True)
    elapsed=sum(r['elapsed_seconds'] for r in records)
    result={'samples':len(records),'elapsed_seconds':elapsed,'seconds_per_sample':elapsed/len(records),
            'backbone_seconds':sum(r['backbone_seconds'] for r in records),
            'sampling_seconds':sum(r['sampling_seconds'] for r in records),
            'backbone_calls_per_sample':sum(r['backbone_calls'] for r in records)/len(records),
            'generated_tokens_per_second':sum(len(r['token_ids'])-r['prefix_length'] for r in records)/elapsed,
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
