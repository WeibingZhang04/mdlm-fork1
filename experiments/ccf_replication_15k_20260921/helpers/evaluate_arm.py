import sys,subprocess,datetime
from pathlib import Path
from campaign_common import *
index=int(sys.argv[1]);v=VARIANTS[index]
assert (ROOT/'training'/v['name']/'completed.json').is_file()
base=ROOT/'evaluation'/v['name'];base.mkdir(parents=True,exist_ok=True)
for stage,samples,seed,checkpoints in [('curve20',20,91001,[1000,2000,3000,4000,5000,6000,7000,10000,12000,15000]),('confirmation100',100,100001,[7000,10000,15000])]:
    for step in checkpoints:
        dest=base/stage/f'step{step:06d}'
        if (dest/'completed.json').is_file():continue
        if dest.exists():raise RuntimeError('Incomplete attempt requires a fresh output directory: '+str(dest))
        dest.mkdir(parents=True)
        exp=ROOT/'exports'/v['name']/f'step{step:06d}'
        modes=('factorized','structured_joint') if stage=='confirmation100' and step==7000 else ('structured_joint',)
        args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(4,8,16,32),samples=samples,seed=seed,modes=modes)
        save(dest/'request.json',dict(args=args,variant=v,training_step=step,samples_per_budget=samples,seed=seed,source_head=STATE['head']))
        save(dest/'args.json',args)
        with (dest/'run.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'helpers/run_pilot.py'),STATE['code'],str(dest/'args.json')],check=True,stdout=log,stderr=subprocess.STDOUT)
        save(dest/'completed.json',dict(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
save(base/'completed.json',dict(status='completed',variant=v))
