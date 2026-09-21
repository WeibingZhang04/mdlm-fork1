from pathlib import Path
import subprocess,json,base64
w=Path(__file__).parent
existing=w/'deployment.json'
if existing.exists():raise RuntimeError('Campaign already deployed; inspect its existing gate rather than duplicating jobs')
rows=json.loads(Path('work/live-sweep-data.json').read_text())['rows']
checkpoints={}
for r in rows:
 c=r['cell']
 if c['family']=='A' or c['step']!=6000:continue
 name=('basic' if c['embedding']=='shared' else 'separate_r'+str(c['rank']))+'_'+c['arm']
 checkpoints[name]=c['checkpoint']
assert len(checkpoints)==8
helpers={n:(w/n).read_text() for n in ['campaign_common.py','runtime_probe.py','run_pilot.py','gate.py']}
remote=r"""
from pathlib import Path
import subprocess,json,tempfile,os,datetime,hashlib
cache=Path.home()/'mdlm_data/tree_mdlm_cache'
source=Path.home()/'clean_tree_mdlm/mdlm-fork1'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
assert head=='2051502329429a252d3b806e0ed195ff379c42b6'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
root=Path(tempfile.mkdtemp(prefix='ccf_replication15k_20260921.',dir=cache/'runs'))
code=root/'code'
subprocess.run(['git','clone','--shared','--no-checkout',str(source),str(code)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
subprocess.run(['git','checkout','--detach',head],cwd=code,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
state={'root':str(root),'code':str(code),'cache':str(cache),'head':head,'source_repo':str(source),'old_code':str(cache/'code/ccf_confirmation_100.daiam2ji'),'old_checkpoints':CHECKPOINTS,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'audit_pending','jobs':{}}
(root/'helpers').mkdir();(root/'logs').mkdir()
for n,t in HELPERS.items():(root/'helpers'/n).write_text(t)
state['helper_sha256']={n:hashlib.sha256(t.encode()).hexdigest() for n,t in HELPERS.items()}
(root/'campaign.json').write_text(json.dumps(state,indent=2))
script='''#!/bin/bash
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
export CCF_CAMPAIGN_ROOT="ROOT_VALUE"
export CCF_CACHE_ROOT="CACHE_VALUE"
export HF_HUB_CACHE="$CCF_CACHE_ROOT/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONPATH="$CCF_CAMPAIGN_ROOT/helpers:$CCF_CAMPAIGN_ROOT/code:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
cd "$CCF_CAMPAIGN_ROOT/code"
python -u "$CCF_CAMPAIGN_ROOT/helpers/gate.py"
'''.replace('ROOT_VALUE',str(root)).replace('CACHE_VALUE',str(cache))
(root/'gate.sh').write_text(script)
cmd=['sbatch','--parsable','--job-name=ccf15k-audit','--partition=ALL','--time=01:30:00','--mem=36G','--cpus-per-task=4','--gres=gpu:1','--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca','--mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50','--output='+str(root/'logs/audit-%j.out'),'--error='+str(root/'logs/audit-%j.err'),str(root/'gate.sh')]
job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
state['jobs']['audit']=job
(root/'campaign.json').write_text(json.dumps(state,indent=2))
print(json.dumps(state))
""".replace('CHECKPOINTS',repr(checkpoints)).replace('HELPERS',repr(helpers))
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','n23zhangWatGPU','python3','-'],input=remote,text=True,capture_output=True,check=True,timeout=100)
state=json.loads(r.stdout);existing.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
