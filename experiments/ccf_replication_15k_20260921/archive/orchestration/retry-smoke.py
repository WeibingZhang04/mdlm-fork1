import subprocess,json
from pathlib import Path
w=Path(__file__).parent;s=json.loads((w/'deployment.json').read_text())
helpers={n:(w/n).read_text() for n in ['smoke.py','train_arm.py']}
remote='''
import json,subprocess,hashlib
from pathlib import Path
root=Path(ROOT);state=json.loads((root/'campaign.json').read_text())
old=state['jobs']['smoke'];status=subprocess.check_output(['sacct','-X','-n','-P','-j',old,'--format=State'],text=True).strip();assert status=='FAILED',status
assert not list((root/'training').glob('*'))
(root/'smoke').rename(root/('smoke-attempt-'+old))
for n,t in HELPERS.items():
 p=root/'helpers'/n;p.write_text(t);state['helper_sha256'][n]=hashlib.sha256(p.read_bytes()).hexdigest()
cmd=['sbatch','--parsable','--job-name=ccf15k-smoke','--partition=ALL','--time=00:40:00','--mem=36G','--cpus-per-task=4','--gres=gpu:1','--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca','--mail-type=ALL,TIME_LIMIT','--output='+str(root/'logs/smoke-%j.out'),'--error='+str(root/'logs/smoke-%j.err'),str(root/'smoke.sh')]
job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit();state.setdefault('smoke_attempts',[]).append({'job':old,'failure':'Hydra entry point imported as a module; fixed launcher to execute main.py as __main__.'});state['jobs']['smoke']=job
subprocess.run(['scontrol','update','JobId='+state['jobs']['training'],'Dependency=afterok:'+job],check=True)
(root/'campaign.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))
'''.replace('ROOT',repr(s['root']),1).replace('HELPERS',repr(helpers),1)
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=remote,text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
s=json.loads(r.stdout);(w/'deployment.json').write_text(json.dumps(s,indent=2));print(s['jobs'])
