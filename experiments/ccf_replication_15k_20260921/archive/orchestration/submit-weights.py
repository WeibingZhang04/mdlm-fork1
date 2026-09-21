from pathlib import Path
import subprocess,json
w=Path(__file__).parent;s=json.loads((w/'deployment.json').read_text())
helpers={n:(w/n).read_text() for n in ['weight_trial.py','weight_observer.py','weight_eval.py','teacher_probe.py']}
remote='''
from pathlib import Path
import subprocess,json,hashlib
root=Path(ROOT);state=json.loads((root/'campaign.json').read_text())
assert 'weights' not in state['jobs']
for n,t in HELPERS.items():
 p=root/'helpers'/n;assert not p.exists();p.write_text(t);state['helper_sha256'][n]=hashlib.sha256(t.encode()).hexdigest()
base=(root/'gate.sh').read_text().split('python -u ')[0]
script=base+'python -u "$CCF_CAMPAIGN_ROOT/helpers/weight_trial.py" "$SLURM_ARRAY_TASK_ID" smoke\\npython -u "$CCF_CAMPAIGN_ROOT/helpers/weight_trial.py" "$SLURM_ARRAY_TASK_ID"\\npython -u "$CCF_CAMPAIGN_ROOT/helpers/weight_eval.py" "$SLURM_ARRAY_TASK_ID"\\n'
(root/'weights.sh').write_text(script)
cmd=['sbatch','--parsable','--array=0-2%1','--job-name=ccf15k-weights','--dependency=afterok:'+state['jobs']['smoke'],'--partition=ALL','--time=08:00:00','--mem=36G','--cpus-per-task=4','--gres=gpu:1','--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca','--mail-type=ALL,ARRAY_TASKS,TIME_LIMIT','--output='+str(root/'logs/weights-%A_%a.out'),'--error='+str(root/'logs/weights-%A_%a.err'),str(root/'weights.sh')]
job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit();state['jobs']['weights']=job
(root/'campaign.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))
'''.replace('ROOT',repr(s['root']),1).replace('HELPERS',repr(helpers),1)
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=remote,text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
s=json.loads(r.stdout);(w/'deployment.json').write_text(json.dumps(s,indent=2));print(s['jobs'])
