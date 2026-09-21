"""Submit a predeclared, gated study under the user's unique campaign root."""
from pathlib import Path
import json,subprocess
here=Path(__file__).resolve().parent;w=here.parent
state=json.loads((w/'deployment.json').read_text())
files={p.name:p.read_text() for p in here.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md') and p.name not in ('deployment-receipt.json',)}
script='''
from pathlib import Path
import json,subprocess,hashlib,tempfile,os,datetime
root=Path(ROOT_VALUE);state=json.loads((root/'campaign.json').read_text())
assert (root/'gate/passed.json').is_file() and (root/'smoke/passed').is_file()
assert subprocess.check_output(['git','-C',state['code'],'rev-parse','HEAD'],text=True).strip()=='2051502329429a252d3b806e0ed195ff379c42b6'
parent=root/'topology_improvements';parent.mkdir(exist_ok=True)
exp=Path(tempfile.mkdtemp(prefix='study_',dir=parent));helpers=exp/'helpers';helpers.mkdir()
for name,text in FILES_VALUE.items():(helpers/name).write_text(text)
base=(root/'gate.sh').read_text().split('python -u ')[0]
base+='export CCF_TOPOLOGY_ROOT='+str(exp)+'\\nexport PYTHONPATH="'+str(helpers)+':$PYTHONPATH"\\n'
receipt={'root':str(exp),'source_head':state['head'],'source_code':state['code'],'helper_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers.iterdir()},'jobs':{},'commands':{},'declared_protocol':json.loads((helpers/'protocol.json').read_text()),'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def submit(name,command,hours,array=None,dependency=None):
 wrapper=exp/(name+'.sh');wrapper.write_text(base+command+'\\n')
 subprocess.run(['bash','-n',str(wrapper)],check=True)
 cmd=['sbatch','--parsable','--job-name=ccf-topo-'+name,'--partition=ALL','--time='+hours,'--mem=40G','--cpus-per-task=4','--gres=gpu:1','--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca','--mail-type=ALL,ARRAY_TASKS,TIME_LIMIT','--output='+str(exp/(name+'-%A_%a.out')),'--error='+str(exp/(name+'-%A_%a.err'))]
 if array:cmd+=['--array='+array]
 if dependency:cmd+=['--dependency=afterok:'+dependency]
 cmd+=[str(wrapper)];job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
 receipt['jobs'][name]=job;receipt['commands'][name]=cmd
 (exp/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\\n')
 return job
prefix='python -u '+str(helpers)+'/'
gate=submit('gate',prefix+'smoke_gate.py','01:00:00')
dev=submit('dev',prefix+'generation_worker.py dev "$SLURM_ARRAY_TASK_ID"','04:00:00','0-3%1',gate)
confirm=submit('confirm',prefix+'generation_worker.py confirm "$SLURM_ARRAY_TASK_ID"','08:00:00','0-1%1',dev)
train=submit('train',prefix+'train_variant.py "$SLURM_ARRAY_TASK_ID"','08:00:00','0-2%1',gate)
trained=submit('trained_confirm',prefix+'evaluate_trained.py','08:00:00',None,train)
# Add only this study; preserve concurrent updates to the parent campaign state.
state=json.loads((root/'campaign.json').read_text())
state.setdefault('topology_improvement_studies',[]).append(receipt)
for name,job in receipt['jobs'].items():state['jobs']['topology_improvement_'+name]=job
(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\\n')
print(json.dumps({'receipt':receipt,'state':state}))
'''.replace('ROOT_VALUE',repr(state['root'])).replace('FILES_VALUE',repr(files))
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,timeout=55)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
d=json.loads(r.stdout);(here/'deployment-receipt.json').write_text(json.dumps(d['receipt'],indent=2)+'\n');(w/'deployment.json').write_text(json.dumps(d['state'],indent=2)+'\n');print(json.dumps({'root':d['receipt']['root'],'jobs':d['receipt']['jobs']},indent=2))
