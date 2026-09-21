"""Retry only the two failed diagnostic arrays, preserving every old artifact."""
import json,subprocess
from pathlib import Path
w=Path(__file__).resolve().parents[1]; patch=Path(__file__).parent/'repairs'
state=json.loads((w/'deployment.json').read_text())
files={p.name:p.read_text() for p in patch.glob('*.py')}
script='''
from pathlib import Path
import json,subprocess,shutil,hashlib,os
root=Path(ROOT_VALUE); state=json.loads((root/'campaign.json').read_text())
assert state['jobs']['weights']=='1556977' and state['jobs']['graph_generation']=='1556994'
assert not subprocess.check_output(['squeue','-h','-j','1556977,1556994'],text=True).strip()
helpers=root/'diagnostic-repairs-v2';shutil.copytree(root/'helpers',helpers)
for n,t in FILES_VALUE.items():(helpers/n).write_text(t)
commands={}; new_jobs={}
for name,oldwrapper,array in [('weights','weights.sh','0-2%1'),('graph_generation','graph_generation.sh','0-3%1')]:
 text=(root/oldwrapper).read_text().replace('$CCF_CAMPAIGN_ROOT/helpers','$CCF_CAMPAIGN_ROOT/diagnostic-repairs-v2')
 wrapper=root/(name+'-retry2.sh');assert not wrapper.exists();wrapper.write_text(text)
 cmd=['sbatch','--parsable','--array='+array,'--job-name=ccf15k-'+name+'-retry2','--partition=ALL','--time='+('08:00:00' if name=='weights' else '04:00:00'),'--mem=36G','--cpus-per-task=4','--gres=gpu:1','--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca','--mail-type=ALL,ARRAY_TASKS,TIME_LIMIT','--output='+str(root/'logs'/f'{name}-retry2-%A_%a.out'),'--error='+str(root/'logs'/f'{name}-retry2-%A_%a.err'),str(wrapper)]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
 commands[name]=cmd;new_jobs[name]=job
 state.setdefault('diagnostic_attempt_history',[]).append({'kind':name,'job':state['jobs'][name],'status':'failed_before_usable_results','reason':{'weights':'Literal target instead of integer callback expected_steps','graph_generation':'Tracing wrapper keyword argument names mismatched model API'}[name]})
 state['jobs'][name]=job
 (root/'campaign.json').write_text(json.dumps(state,indent=2)+'\\n')
receipt={'new_jobs':new_jobs,'helper_dir':str(helpers),'helper_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers.glob('*.py')},'commands':commands,'original_artifacts_preserved':True}
(root/'diagnostic-repairs-v2.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps({'state':state,'receipt':receipt}))
'''.replace('ROOT_VALUE',repr(state['root'])).replace('FILES_VALUE',repr(files))
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,timeout=55)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
d=json.loads(r.stdout);(w/'deployment.json').write_text(json.dumps(d['state'],indent=2)+'\n');(Path(__file__).parent/'repair-receipt.json').write_text(json.dumps(d['receipt'],indent=2)+'\n');print(json.dumps(d['receipt']['new_jobs']))
