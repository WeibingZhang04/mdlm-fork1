from pathlib import Path
import subprocess,json
w=Path(__file__).parent;s=json.loads((w/'deployment.json').read_text())
remote='''
from pathlib import Path
import subprocess,json
root=Path(ROOT);state=json.loads((root/'campaign.json').read_text())
base=(root/'gate.sh').read_text().split('python -u ')[0]
for name,helper,hours in [('legacy_replay','legacy_replay.py','03:00:00'),('topology6k','topology_diagnosis.py','03:00:00')]:
 assert name not in state['jobs']
 script=root/(name+'.sh');script.write_text(base+'python -u "$CCF_CAMPAIGN_ROOT/helpers/'+helper+'"\\n')
 cmd=['sbatch','--parsable','--job-name=ccf15k-'+name,'--dependency=afterok:'+state['jobs']['audit'],'--partition=ALL','--time='+hours,'--mem=36G','--cpus-per-task=4','--gres=gpu:1','--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908','--mail-user=n23zhang@uwaterloo.ca','--mail-type=ALL,TIME_LIMIT','--output='+str(root/('logs/'+name+'-%j.out')),'--error='+str(root/('logs/'+name+'-%j.err')),str(script)]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit();state['jobs'][name]=job
 (root/'campaign.json').write_text(json.dumps(state,indent=2))
print(json.dumps(state))
'''.replace('ROOT',repr(s['root']),1)
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=remote,text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
s=json.loads(r.stdout);(w/'deployment.json').write_text(json.dumps(s,indent=2));print(s['jobs'])
