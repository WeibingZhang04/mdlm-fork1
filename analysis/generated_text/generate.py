#!/usr/bin/env python3
"""DO NOT touch other people's files. DO NOT cancel other people's jobs.

Fresh 500-sample cells using existing exports and the existing sampler/gate.
No training, source copies, sbatch calls, or job cancellation. See README.md.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from analyze import REPO, account, owned, read, sha, fresh, save, verify


def replace_arg(args, flag, value):
    args = list(args)
    if args.count(flag) != 1:
        raise ValueError('Expected one ' + flag)
    args[args.index(flag)+1] = str(value)
    return args


def generation_args(request, out, cell, seed):
    args=request['args']
    for flag,val in (('--output-dir',out/'generation'),('--num-samples',500),
                     ('--base-seed',seed),('--modes',cell['mode'])):
        args=replace_arg(args,flag,val)
    return [('checkpointing.save_dir='+str(out)) if s.startswith('checkpointing.save_dir=') else s for s in args]


def prepare(a):
    if subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip() != 'original_table_base_crf-recovery':
        raise ValueError('Wrong branch')
    states = [read(owned(s)/'study.json') for s in (a.bf16,a.fp32)]
    if any(Path(s['repo']).resolve()!=REPO for s in states): raise ValueError('Wrong study checkout')
    if any(s['source_only'] for s in states): raise ValueError('Source-only study')
    if states[0]['backbone'] != states[1]['backbone']: raise ValueError('Backbones differ')
    different = sorted(k for k in states[0]['source_identities']
        if states[0]['source_identities'][k] != states[1]['source_identities'].get(k))
    if any(k not in ('models/dit.py','experiments/original_table/provenance.json') and not k.endswith('.md') for k in different):
        raise ValueError('Additional source differences need review: '+str(different))
    sources = dict(states[1]['source_identities'])
    for k,v in sources.items():
        if sha(REPO/k)!=v: raise ValueError('Active study source changed: '+k)
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.txt','.sbatch'):
            sources[str(p.relative_to(REPO))]=sha(p)
    cells=[]
    for precision,study in (('bf16',a.bf16),('fp32',a.fp32)):
        for i,c in enumerate(read(owned(study)/'confirmation-cells.json')):
            if c['step']!=6000 or c['arm_id'] not in ('basic_fd','basic_dd'):
                raise ValueError('Expected FD/DD 6k cells')
            model=c['arm_id'][6:].upper(); steps=c['sampling_steps']
            cells.append(dict(id=f'{model.lower()}_{precision}_s{steps}',model=model,
                cache_precision=precision,steps=steps,study=str(owned(study)),source_index=i,
                mode='structured_joint'))
    if len(cells)!=12: raise ValueError('Expected 12 structured cells')
    for steps in (8,16,32):
        source=next(c for c in cells if c['model']=='FD' and c['cache_precision']=='bf16' and c['steps']==steps)
        cells.append(dict(source,id=f'mdlm_s{steps}',model='MDLM',cache_precision='released',mode='factorized'))
    root=fresh(a.out)
    fresh(root/'logs')
    save(root/'manifest.json',dict(repo=str(REPO),output=str(root),cache=states[1]['cache'],
        source_sha256=sources,source_differences=different,cells=cells,num_samples=500,base_seed=300001,
        generation_cache='FP32 for every condition; asserted at runtime',
        ownership='DO NOT touch other people\'s files. DO NOT cancel other people\'s jobs.'))
    print(root/'manifest.json')


def generate(a):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('GPU generation needs a Slurm allocation')
    m=verify(a.manifest); cell=m['cells'][a.index]
    study=owned(cell['study']); old=study/'evaluation/confirmation'/f"{cell['source_index']:03d}"
    if read(old/'completed.json')['status']!='completed': raise ValueError('Source evaluation incomplete')
    request=read(old/'request.json'); out=fresh(Path(m['output'])/'cells'/cell['id'])
    args=generation_args(request,out,cell,m['base_seed'])
    for flag in ('--backbone-checkpoint','--adapter','--adapter-manifest'):
        owned(args[args.index(flag)+1])
    save(out/'request.json',dict(cell=cell,args=args,source_request_sha256=sha(old/'request.json'),
        checkpoint_sha256=request['checkpoint_sha256']))
    os.environ['HF_HUB_CACHE']=str(owned(m['cache'])/'huggingface')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'experiments/original_table'))
    os.chdir(REPO)
    from scripts import run_generation_pilot as pilot
    import historical_gate
    import torch
    from models.dit import Rotary
    from unittest.mock import patch
    original=Rotary.forward
    def checked(self,*pos,**kw):
        result=original(self,*pos,**kw)
        if any(t.dtype!=torch.float32 for t in result):
            raise RuntimeError('Generation rotary cache is not FP32')
        return result
    with patch.object(Rotary,'forward',checked):
        rc=(pilot.main(args) if cell['mode']=='factorized' else
            historical_gate.gated_pilot(pilot,request['cell'],out,args))
    if rc != 0: raise RuntimeError('Generation failed')
    g=read(out/'generation/summary.json')['groups'][0]
    if g['num_sequences']!=500 or g['reference_lm']['num_scored_sequences']!=500 or g['unresolved_mask_tokens']!=0:
        raise ValueError('Incomplete generation')
    save(out/'completed.json',dict(samples_sha256=sha(out/'generation/samples.jsonl'),
        num_samples=500,generation_rotary_dtype='torch.float32',status='completed'))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('bf16','fp32','out'): q.add_argument('--'+name,type=Path,required=True)
    q.set_defaults(fn=prepare)
    q=sub.add_parser('generate');q.add_argument('--manifest',type=Path,required=True)
    q.add_argument('--index',type=int,required=True);q.set_defaults(fn=generate)
    a=p.parse_args();account();a.fn(a)


if __name__=='__main__': main()
