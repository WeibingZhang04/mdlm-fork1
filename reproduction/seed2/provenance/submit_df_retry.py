import json
from pathlib import Path
import subprocess

w = Path(__file__).resolve().parent
state = json.loads((w / 'deployment.json').read_text())
script = 'root=' + repr(state['root']) + '\nbody=' + repr((w/'retry_df.py').read_text()) + '\n' + r'''
from pathlib import Path
import datetime,hashlib,json,subprocess,tempfile
root=Path(root)
parent=root/'retries';parent.mkdir(exist_ok=True)
attempt=Path(tempfile.mkdtemp(prefix='df_',dir=str(parent)))
(attempt/'retry_df.py').write_text(body)
prefix=(root/'train.sh').read_text().rsplit('python -u',1)[0]
prefix+='export CCF_DF_RETRY='+str(attempt)+'\n'
run=attempt/'run.sh';run.write_text(prefix+'python -u '+str(attempt/'retry_df.py')+'\n')
subprocess.run(['bash','-n',str(run)],check=True)
assert not (root/'exports/dynamic_fixed').exists()
q=subprocess.check_output(['squeue','-h','-j','1557404_2','-o','%i|%u|%T|%j'],text=True).strip()
assert q=='1557404_2|n23zhang|PENDING|ccf-oldbf16-eval',q
cmd=['sbatch','--parsable','--partition=ALL','--job-name=ccf-oldbf16-df-retry','--time=24:00:00',
     '--mem=40G','--cpus-per-task=4','--gres=gpu:1','--no-requeue',
     '--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca',
     '--mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50',
     '--output='+str(attempt/'job-%j.out'),'--error='+str(attempt/'job-%j.err'),str(run)]
job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
receipt=dict(root=str(attempt),study_root=str(root),job=job,command=cmd,
             prior_failed_job='1557403_2',evaluation_job='1557404_2',
             source_step=1000,source_checkpoint=str(root/'training/dynamic_fixed/to1000/checkpoints/0-1000.ckpt'),
             created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             helper_sha256=hashlib.sha256(body.encode()).hexdigest(),dependency_repaired=False)
(attempt/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run(['scontrol','update','JobId=1557404_2','Dependency=afterok:'+job],check=True)
receipt['dependency_repaired']=True
(attempt/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
(root/'training/dynamic_fixed/retry-deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,timeout=55)
assert r.returncode==0,r.stderr
receipt=json.loads(r.stdout)
(w/'df-retry-deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
