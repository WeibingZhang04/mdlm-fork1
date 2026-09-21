from pathlib import Path
import json,subprocess
w=Path(__file__).parent;s=json.loads((w/'deployment.json').read_text())
code='''
from pathlib import Path
import json,subprocess,os,datetime
root=Path(ROOT);assert root.stat().st_uid==os.getuid()
s=json.loads((root/'campaign.json').read_text());old=s['jobs']['audit']
status=subprocess.check_output(['sacct','-j',old,'-X','--noheader','--parsable2','-o','State'],text=True).strip()
assert status.startswith('FAILED'),status
if (root/'gate').exists():(root/'gate').rename(root/('gate-attempt-'+old))
p=root/'gate.sh';t=p.read_text().replace('export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled','export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled')
p.write_text(t)
cmd=['sbatch','--parsable','--job-name=ccf15k-audit','--partition=ALL','--time=01:30:00','--mem=36G','--cpus-per-task=4','--gres=gpu:1','--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca','--mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50','--output='+str(root/'logs/audit-%j.out'),'--error='+str(root/'logs/audit-%j.err'),str(p)]
job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
s.setdefault('audit_attempts',[]).append({'job':old,'failure':'Audit launcher forced HF offline; pinned streaming loader requires dataset metadata lookup.'})
s['jobs']['audit']=job;(root/'campaign.json').write_text(json.dumps(s,indent=2));print(json.dumps(s))
'''.replace('ROOT',repr(s['root']))
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','n23zhangWatGPU','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=50)
s=json.loads(r.stdout);(w/'deployment.json').write_text(json.dumps(s,indent=2));print('audit retry',s['jobs']['audit'])
