#!/usr/bin/env python3
"""DO NOT touch other people's files. DO NOT touch other people's jobs.
Do not interfere with other people's processes. No training or old-output writes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generated_text'))
from analyze import REPO,account,owned,read,sha,fresh,save,verify
from generate import replace_arg

VARIANTS=('native','chain','random','marginal')


def prepare(a):
    assert subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip()=='original_table_base_crf-recovery'
    parent=verify(a.parent); sources=dict(parent['source_sha256'])
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.sbatch'): sources[str(p.relative_to(REPO))]=sha(p)
    cells=[]; groups=[]
    for precision in ('bf16','fp32'):
        for steps in (8,16,32):
            source=next(c for c in parent['cells'] if c['model']=='DD' and c['cache_precision']==precision and c['steps']==steps)
            old=owned(source['study'])/'evaluation/confirmation'/f"{source['source_index']:03d}"
            assert read(old/'completed.json')['status']=='completed'
            request=read(old/'request.json'); ids=[]
            for variant in VARIANTS:
                cell=dict(source,id=f'dd_{precision}_s{steps}_{variant}',variant=variant,
                    mode='structured_marginal' if variant=='marginal' else 'structured_joint',
                    source_request=request,source_request_sha256=sha(old/'request.json'))
                cells.append(cell);ids.append(cell['id'])
            groups.append(ids)
    out=fresh(a.out);fresh(out/'logs')
    save(out/'manifest.json',dict(repo=str(REPO),output=str(out),cache=parent['cache'],
        parent_manifest=str(owned(a.parent)),parent_sha256=sha(a.parent),
        source_sha256=sources,cells=cells,groups=groups,num_samples=500,base_seed=400001,
        source_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        generation_rotary='FP32 asserted; existing BF16 transformer autocast unchanged',
        safety="DO NOT touch other people's files. DO NOT touch other people's jobs. Do not interfere with other people's processes."))
    print(out/'manifest.json')


def cell_run(m,c,root,count,seed,parity=False):
    import torch
    from unittest.mock import patch
    from scripts import run_generation_pilot as pilot
    from models.dit import Rotary
    from historical_sampler import experiment
    from interventions import apply
    request=c['source_request'];out=fresh(root/'cells'/c['id']);args=list(request['args'])
    for flag,value in [('--output-dir',out/'generation'),('--num-samples',count),('--base-seed',seed),('--modes',c['mode'])]:
        args=replace_arg(args,flag,value)
    args=[('checkpointing.save_dir='+str(out)) if v.startswith('checkpointing.save_dir=') else v for v in args]
    for flag in ('--backbone-checkpoint','--adapter','--adapter-manifest'): owned(args[args.index(flag)+1])
    save(out/'request.json',dict(cell=c,args=args))
    forward=Rotary.forward;original=pilot.run_sampling_group;first=True
    def rotary(self,*args,**kwargs):
        result=forward(self,*args,**kwargs)
        assert all(t.dtype==torch.float32 for t in result)
        return result
    with owned(out/'graph-trace.jsonl').open('x') as trace:
        def run(model,samples,**kw):
            nonlocal first
            assert len(samples)==1
            sample_seed=samples[0].pair_seed
            if parity and first:
                baseline,timing=original(model,samples,**kw)
                cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
            with apply(c['variant'],sample_seed,trace): result,meta=original(model,samples,**kw)
            if parity and first:
                assert baseline[0]['sample_token_ids']==result[0]['sample_token_ids']
                assert timing['measured_nfe']==meta['measured_nfe']
                assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
                save(out/'native-parity.json',dict(tokens=True,nfe=True,cpu_rng=True,cuda_rng=True))
            first=False
            return result,meta
        with experiment('level_draws'),patch.object(Rotary,'forward',rotary),patch.object(pilot,'run_sampling_group',run):
            assert pilot.main(args)==0
    summary=read(out/'generation/summary.json');g=summary['groups'][0]
    assert g['num_sequences']==g['reference_lm']['num_scored_sequences']==count and g['unresolved_mask_tokens']==0
    save(out/'completed.json',dict(status='completed',num_samples=count,
        samples_sha256=sha(out/'generation/samples.jsonl'),graph_sha256=sha(out/'graph-trace.jsonl'),
        variant=c['variant'],gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'],
        node=os.environ.get('SLURMD_NODENAME'),generation_rotary_dtype='torch.float32'))
    torch.cuda.empty_cache()


def execute(a):
    assert os.environ.get('SLURM_JOB_ID'),'Use a Slurm GPU allocation'
    m=verify(a.manifest)
    os.environ['HF_HUB_CACHE']=str(owned(m['cache'])/'huggingface')
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'experiments/original_table'))
    os.chdir(REPO)
    if a.command=='gate':
        from interventions import self_test
        self_test();root=fresh(Path(m['output'])/'gate')
        for c in m['cells']:
            if c['steps']==8:cell_run(m,c,root,2,490001,parity=c['variant']=='native')
        save(root/'completed.json',dict(status='passed',cells=8,samples_per_cell=2))
    else:
        assert read(Path(m['output'])/'gate/completed.json')['status']=='passed'
        for cid in m['groups'][a.index]:
            c=next(c for c in m['cells'] if c['id']==cid)
            cell_run(m,c,Path(m['output']),m['num_samples'],m['base_seed'],parity=c['variant']=='native')


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare');q.add_argument('--parent',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.set_defaults(fn=prepare)
    for name in ('gate','generate'):
        q=s.add_parser(name);q.add_argument('--manifest',type=Path,required=True);q.add_argument('--index',type=int,default=0);q.set_defaults(fn=execute)
    a=p.parse_args();account();a.fn(a)


if __name__=='__main__':main()
