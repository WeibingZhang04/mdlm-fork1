"""Repair only this study's inactive launcher, preserving the failed attempt."""
from pathlib import Path
import json,subprocess
here=Path(__file__).resolve().parent;receipt=json.loads((here/'deployment-receipt.json').read_text())
changed=['topology_variants.py','run_variant_generation.py','test_topology_variants.py','smoke_gate.py']
files={name:(here/name).read_text() for name in changed}
script='''
from pathlib import Path
import subprocess,json,shutil,hashlib,datetime
exp=Path(EXP_VALUE);root=exp.parents[1];receipt=json.loads((exp/'deployment.json').read_text());jobs=receipt['jobs']
assert jobs['gate']=='1557038'
assert subprocess.check_output(['sacct','-X','-n','-j','1557038','--format=State','-P'],text=True).strip()=='FAILED'
queue=subprocess.check_output(['squeue','-h','-j',','.join(jobs.values()),'-o','%T'],text=True)
assert all(s=='PENDING' for s in queue.splitlines()),'Do not edit a helper used by a running job'
assert not (exp/'gate_retry2').exists()
assert not list(exp.glob('smoke_*')),'Do not reuse completed smoke-training folders'
shutil.copytree(exp/'helpers',exp/'helpers_before_retry2')
shutil.copy2(exp/'deployment.json',exp/'deployment-before-retry2.json')
for name,text in FILES_VALUE.items():(exp/'helpers'/name).write_text(text)
cmd=receipt['commands']['gate']
new=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert new.isdigit()
receipt.setdefault('attempt_history',[]).append({'job':'1557038','status':'FAILED','results':'gate','reason':'Graph override entered before native sampling optimizer; source-inspected Kruskal lookup failed for coverage variant','archived_helpers':'helpers_before_retry2'})
receipt['jobs']['gate']=new;receipt['gate_results']='gate_retry2'
receipt['helper_sha256'].update({n:hashlib.sha256(t.encode()).hexdigest() for n,t in FILES_VALUE.items()})
receipt['gate_retry_submitted_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('dev','train'):
 subprocess.run(['scontrol','update','JobId='+jobs[name],'Dependency=afterok:'+new],check=True)
receipt['dependency_repairs']={name:{'job':jobs[name],'dependency':'afterok:'+new} for name in ('dev','train')}
(exp/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\\n')
state=json.loads((root/'campaign.json').read_text());state['jobs']['topology_improvement_gate']=new
for i,study in enumerate(state['topology_improvement_studies']):
 if study['root']==str(exp):state['topology_improvement_studies'][i]={**receipt,'versioned_commit':study.get('versioned_commit')}
(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\\n')
print(json.dumps({'receipt':receipt,'state':state,'queue':subprocess.check_output(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%T|%R|%E'],text=True)}))
'''.replace('EXP_VALUE',repr(receipt['root'])).replace('FILES_VALUE',repr(files))
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,timeout=55)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
d=json.loads(r.stdout);(here/'deployment-receipt-before-retry2.json').write_text(json.dumps(receipt,indent=2)+'\n');(here/'deployment-receipt.json').write_text(json.dumps(d['receipt'],indent=2)+'\n');(here.parent/'deployment.json').write_text(json.dumps(d['state'],indent=2)+'\n');print(d['queue'])
