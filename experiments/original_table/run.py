#!/usr/bin/env python3
"""Prepare and run the historical protocol. This program never calls sbatch."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ARMS = {'static_static': ('fixed', 'fixed', 0.0), 'fixed_dynamic': ('fixed', 'dynamic', 0.0),
        'dynamic_fixed': ('dynamic', 'fixed', 0.1), 'dynamic_dynamic': ('dynamic', 'dynamic', 0.1)}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def call(args, cwd=None, log=None):
    if log is None: subprocess.run(args, cwd=cwd, check=True)
    else:
        with Path(log).open('x') as f:
            subprocess.run(args, cwd=cwd, check=True, stdout=f, stderr=subprocess.STDOUT)

def protocol(study): return read(study/'suite/protocol.json')
def require_allocation():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training and generation require a Slurm allocation')

def prepare(options):
    repo = options.repo.resolve(); dest = options.study.resolve(); cache = options.cache.resolve()
    if dest.exists(): raise FileExistsError('Choose a new output directory: '+str(dest))
    # Never put checkpoints, runtime Git repositories, or logs inside the source repository.
    if repo == dest or repo in dest.parents: raise ValueError('Study must be outside the repository')
    cfg = read(HERE/'protocol.json'); provenance = read(HERE/'provenance.json')
    backbone = cache/'checkpoints/mdlm-owt-backbone.pt'
    if not options.source_only and sha(backbone) != cfg['backbone_sha256']:
        raise ValueError('Pinned backbone hash mismatch')
    commit = provenance['base_commit']
    files = list(provenance['base_runtime_sha256'])
    blob = subprocess.check_output(['git','-C',str(repo),'archive',commit,'--',*files])
    dest.mkdir(parents=True)
    shutil.copytree(HERE,dest/'suite',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    identities = {}
    for variant, patches in [('basic',[]),('r8',['r8']),('r16',['r8','r16']),('eval',['r8','r16','eval'])]:
        runtime=dest/'runtime'/variant; runtime.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            for item in tar.getmembers():
                if item.name.startswith('/') or '..' in Path(item.name).parts or item.issym() or item.islnk():
                    raise ValueError('Unsafe archive member')
            tar.extractall(runtime)
        expected=dict(provenance['base_runtime_sha256'])
        for name in patches:
            patch=HERE/'patches'/f'{name}.patch'
            assert sha(patch)==provenance['variants'][name]['patch_sha256']
            call(['git','apply','--check',str(patch)],cwd=runtime)
            call(['git','apply',str(patch)],cwd=runtime)
            expected.update(provenance['variants'][name]['changed_sha256'])
        for name,digest in expected.items():
            if sha(runtime/name)!=digest: raise ValueError('Runtime source mismatch: '+variant+'/'+name)
        if variant=='eval':
            for name in ['historical_sampler.py','historical_gate.py']:
                shutil.copy2(HERE/name,runtime/'scripts'/name)
                expected['scripts/'+name]=sha(runtime/'scripts'/name)
        # Runtime provenance only, outside the source repository; no remote or push.
        call(['git','init','-q'],cwd=runtime)
        call(['git','add','.'],cwd=runtime)
        env=dict(os.environ,GIT_AUTHOR_NAME='Anonymous Researcher',GIT_AUTHOR_EMAIL='anonymous@example.invalid',
                 GIT_COMMITTER_NAME='Anonymous Researcher',GIT_COMMITTER_EMAIL='anonymous@example.invalid')
        subprocess.run(['git','commit','-q','-m','Reconstructed historical runtime'],cwd=runtime,env=env,check=True)
        identities[variant]=expected
    (dest/'logs').mkdir()
    state={'cache':str(cache),'backbone':str(backbone),'base_commit':commit,'source_only':options.source_only,
           'source_identities':identities,'protocol_sha256':sha(dest/'suite/protocol.json')}
    write(dest/'study.json',state)
    cells=make_cells(cfg)
    write(dest/'pilot-cells.json',cells['pilot']);write(dest/'confirmation-cells.json',cells['confirmation'])
    print(json.dumps({'study':str(dest),'train_tasks':len(cfg['arms']), 'pilot_cells':len(cells['pilot']),
                      'confirmation_cells':len(cells['confirmation']),'source_only':options.source_only},indent=2))

def make_cells(cfg):
    pilot=[];confirmation=[]
    ff=next(a for a in cfg['arms'] if a['id']=='basic_ff')
    def cell(arm,step,steps,samples,seed,baseline=False):
        return {'arm_id':arm['id'],'family':'A' if baseline else arm['family'],'arm':arm['arm'],
                'rank':arm['rank'],'embedding':arm['embedding'],'step':step,'sampling_steps':steps,
                'num_samples':samples,'base_seed':seed,'mode':'factorized' if baseline else 'structured_joint'}
    for steps in cfg['sampling']['pilot_steps']:
        pilot.append(cell(ff,7000,steps,20,91001,True))
        for arm in cfg['arms']:
            for step in range(1000,7001,1000):pilot.append(cell(arm,step,steps,20,91001))
    for steps in cfg['sampling']['confirmation_steps']:
        confirmation.append(cell(ff,7000,steps,100,100001,True))
        for arm in cfg['arms']:
            for step in cfg['confirmation_selection'].get(arm['id'],{}).get(str(steps),[]):
                confirmation.append(cell(arm,step,steps,100,100001))
    assert len(pilot)==171 and len(confirmation)==76
    return {'pilot':pilot,'confirmation':confirmation}

def check_state(study,variant):
    state=read(study/'study.json')
    if state['source_only']:raise RuntimeError('Source-only validation study cannot run models')
    assert sha(study/'suite/protocol.json')==state['protocol_sha256']
    for name,digest in state['source_identities'][variant].items():
        if sha(study/'runtime'/variant/name)!=digest:raise ValueError('Runtime was modified: '+name)
    os.environ['HF_HUB_CACHE']=str(Path(state['cache'])/'huggingface')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    return state

def overrides(cfg,arm,phase,study,state):
    p=cfg['phases'][phase]; values=cfg['common_overrides']+p['extra']
    by_key={v.split('=',1)[0]:v for v in values}
    topo,factor,weight=ARMS[arm['arm']]
    run=study/'training'/arm['id']/phase
    prior={'basic_3000':'basic_1000','basic_6000':'basic_3000','basic_10000':'basic_6000'}
    resume=study/'training'/arm['id']/prior.get(phase,'unused')/'checkpoints/last.ckpt'
    substitutions={'backbone':state['backbone'],'hf_cache':str(Path(state['cache'])/'huggingface'),
        'run_dir':str(run),'attempt_dir':str(run),'resume':str(resume),
        'topology':topo,'factor':factor,'weight':str(weight)}
    args=[by_key[k].format(**substitutions) for k in p['key_order']]
    if phase=='basic_1000' and arm['arm']!='dynamic_dynamic':args+=['strategy.find_unused_parameters=true']
    return args,run,resume if phase in prior else None

def train(options):
    require_allocation();study=options.study.resolve();cfg=protocol(study)
    arm=cfg['arms'][options.index];state=check_state(study,arm['runtime'])
    runtime=study/'runtime'/arm['runtime']
    for phase in arm['phases']:
        args,run,resume=overrides(cfg,arm,phase,study,state)
        if run.exists():raise FileExistsError('Existing phase is preserved; inspect it before retrying: '+str(run))
        if resume is not None and not resume.is_file():raise FileNotFoundError(resume)
        run.mkdir(parents=True)
        command=[sys.executable,'-u','main.py',*args]
        write(run/'command.json',{'argv':command,'runtime':arm['runtime'],'resume_sha256':sha(resume) if resume else None})
        # Fresh Python process per original phase; no added startup forward/probe/callback.
        call(command,cwd=runtime,log=run/'training.log')
        ckpt=run/'checkpoints/last.ckpt'
        if not ckpt.is_file():raise RuntimeError('Training did not produce last.ckpt')
        write(run/'completed.json',{'checkpoint_sha256':sha(ckpt),'phase':phase})

def checkpoint(study,cell):
    step=cell['step'];arm=cell['arm_id']
    phase=('basic_1000' if step<=1000 else 'basic_3000' if step<=3000 else 'basic_6000' if step<=6000 else 'basic_10000') if arm.startswith('basic_') else 'r16' if arm.startswith('r16_') else arm
    return study/'training'/arm/phase/'checkpoints'/f'0-{step}.ckpt'

def evaluate(options):
    require_allocation();study=options.study.resolve();cfg=protocol(study)
    state=check_state(study,'eval');cell=read(study/f'{options.suite}-cells.json')[options.index]
    ckpt=checkpoint(study,cell)
    if not ckpt.is_file():raise FileNotFoundError(ckpt)
    out=study/'evaluation'/options.suite/f'{options.index:03d}'
    out.mkdir(parents=True,exist_ok=False)
    runtime=study/'runtime/eval';topo,factor,weight=ARMS[cell['arm']]
    export=[sys.executable,'scripts/export_structured_adapter.py','--checkpoint',str(ckpt),
      '--expected-checkpoint-sha256',sha(ckpt),'--expected-global-step',str(cell['step']),
      '--output',str(out/'adapter.safetensors'),'--manifest',str(out/'adapter.manifest.json'),
      '--control-identity',cell['arm'],'--topology-mode',topo,'--factor-mode',factor,
      '--candidate-k','128','--independent-mode','false','--topology-weight',str(weight)]
    call(export,cwd=runtime,log=out/'export-report.json')
    args=['--backbone-checkpoint',state['backbone'],'--backbone-sha256',cfg['backbone_sha256'],
      '--adapter',str(out/'adapter.safetensors'),'--adapter-sha256',sha(out/'adapter.safetensors'),
      '--adapter-manifest',str(out/'adapter.manifest.json'),'--adapter-manifest-sha256',sha(out/'adapter.manifest.json'),
      '--output-dir',str(out/'generation'),'--num-samples',str(cell['num_samples']),
      '--sequence-length','1024','--batch-size','1','--base-seed',str(cell['base_seed']),
      '--modes',cell['mode'],'--nfe-budgets',str(cell['sampling_steps']+1),'--device','cuda',
      '--model-config','contextual-forest-small','--data-config','train_openwebtext_pinned','--allow-dirty',
      '--reference-lm','gpt2-large','--reference-lm-revision',cfg['reference_lm_revision'],
      '--reference-lm-device','cuda','--reference-lm-batch-size','1','--reference-lm-max-length','1024','--reference-lm-dtype','float32']
    for item in [f'data.cache_dir={state["cache"]}/huggingface','model.structured_decoder.top_k=128',
      f'model.structured_decoder.rank={cell["rank"]}',f'++model.structured_decoder.factor_embedding_mode={cell["embedding"]}',
      '++model.structured_decoder.factor_conditioner_hidden_dim=0',f'model.structured_decoder.topology_mode={topo}',
      f'model.structured_decoder.factor_mode={factor}','model.structured_decoder.independent_mode=false',
      f'model.structured_decoder.training.topology_weight={weight}',f'checkpointing.save_dir={out}']:
        args+=['--override',item]
    write(out/'request.json',{'cell':cell,'args':args,'checkpoint_sha256':sha(ckpt)})
    # Import the frozen evaluator only here, after changing cwd/sys.path to its reconstructed source.
    os.chdir(runtime);sys.path.insert(0,str(runtime))
    from scripts import run_generation_pilot as pilot
    from scripts import historical_gate as gate
    gate.REFERENCE_ROOT=study/'runtime/r16'
    result=pilot.main(args) if cell['mode']=='factorized' else gate.gated_pilot(pilot,cell,out,args)
    if result!=0:raise RuntimeError('Generation failed')
    groups=read(out/'generation/summary.json')['groups']
    assert len(groups)==1 and groups[0]['num_sequences']==cell['num_samples']
    assert groups[0]['reference_lm']['num_scored_sequences']==cell['num_samples']
    assert groups[0]['unresolved_mask_tokens']==0
    write(out/'completed.json',{'cell':cell,'status':'completed'})

def collect(options):
    study=options.study.resolve();rows=[];missing=[]
    for suite in ['pilot','confirmation']:
        for i,cell in enumerate(read(study/f'{suite}-cells.json')):
            out=study/'evaluation'/suite/f'{i:03d}'
            if not (out/'completed.json').is_file():missing.append((suite,i));continue
            g=read(out/'generation/summary.json')['groups'][0]
            rows.append({'suite':suite,**cell,'ppl':g['reference_lm']['perplexity'],
                         'scored_samples':g['reference_lm']['num_scored_sequences'],
                         'actual_nfe':g['measured_nfe_values']})
    write(study/'collected-results.json',{'rows':rows,'missing':missing})
    lines=['# Reproduction results','','PPL is GPT-2-large, first non-leading EOS; lower is better.',
           'Best of the historical three selected checkpoints; this remains an exploratory minimum.','',
           '| Model | 4 steps | 8 steps | 16 steps | 32 steps |','|---|---:|---:|---:|---:|']
    keys=[('A','static_static')]+[(a['family'],a['arm']) for a in protocol(study)['arms'] if a['id'] in protocol(study)['confirmation_selection']]
    for family,arm in keys:
        cells=[]
        for steps in [4,8,16,32]:
            found=[r for r in rows if r['suite']=='confirmation' and r['family']==family and r['arm']==arm and r['sampling_steps']==steps]
            if not found:cells.append('Pending');continue
            best=min(found,key=lambda r:r['ppl']);cells.append(f'{best["ppl"]:.2f} @ {best["step"]//1000}k' if family!='A' else f'{best["ppl"]:.2f}')
        label = 'MDLM' if family == 'A' else {'B':'Basic','C':'Separate R8','D':'Separate R16'}[family]+' '+{'fixed_dynamic':'FD','dynamic_dynamic':'DD','static_static':'FF','dynamic_fixed':'DF'}[arm]
        lines.append('| '+' | '.join([label,*cells])+' |')
    (study/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete_cells':len(rows),'missing_cells':len(missing),'report':str(study/'results.md')}))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--study',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=HERE.parents[1]);p.add_argument('--source-only',action='store_true');p.set_defaults(action=prepare)
    for name,fn in [('train',train),('evaluate',evaluate),('collect',collect)]:
        p=sub.add_parser(name);p.add_argument('--study',type=Path,required=True);p.set_defaults(action=fn)
        if name!='collect':p.add_argument('--index',type=int,required=True)
        if name=='evaluate':p.add_argument('--suite',choices=['pilot','confirmation'],required=True)
    opts=parser.parse_args();opts.action(opts)

if __name__=='__main__':main()
