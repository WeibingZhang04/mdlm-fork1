from pathlib import Path
import subprocess,json,hashlib
w=Path(__file__).parent;s=json.loads((w/'deployment.json').read_text())
names=['campaign_observer.py','train_arm.py','smoke.py','evaluate_arm.py','topology_diagnosis.py','legacy_replay.py']
helpers={n:(w/n).read_text() for n in names}
remote='''
from pathlib import Path
import subprocess,json,hashlib
root=Path(ROOT);state=json.loads((root/'campaign.json').read_text())
for n,t in HELPERS.items():
 p=root/'helpers'/n
 assert not p.exists(),str(p)
 p.write_text(t);state['helper_sha256'][n]=hashlib.sha256(t.encode()).hexdigest()
base=(root/'gate.sh').read_text().split('python -u ')[0]
script=base+'test -f "$CCF_CAMPAIGN_ROOT/gate/passed.json"\\n'
for idx in [0,3,7]:script+='python -u "$CCF_CAMPAIGN_ROOT/helpers/smoke.py" '+str(idx)+'\\n'
script+='touch "$CCF_CAMPAIGN_ROOT/smoke/passed"\\n'
(root/'smoke.sh').write_text(script)
cmd=['sbatch','--parsable','--job-name=ccf15k-smoke','--dependency=afterok:'+state['jobs']['audit'],'--partition=ALL','--time=00:40:00','--mem=36G','--cpus-per-task=4','--gres=gpu:1','--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca','--mail-type=ALL,TIME_LIMIT','--output='+str(root/'logs/smoke-%j.out'),'--error='+str(root/'logs/smoke-%j.err'),str(root/'smoke.sh')]
job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit();state['jobs']['smoke']=job
(root/'campaign.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))
'''.replace('ROOT',repr(s['root']),1).replace('HELPERS',repr(helpers),1)
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=remote,text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
s=json.loads(r.stdout);(w/'deployment.json').write_text(json.dumps(s,indent=2));print(json.dumps(s['jobs']))
