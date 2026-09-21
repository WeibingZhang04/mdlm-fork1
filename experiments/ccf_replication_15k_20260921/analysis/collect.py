from pathlib import Path
import subprocess,json,gzip
w=Path(__file__).parent;s=json.loads((w/'deployment.json').read_text())
script=r'''
from pathlib import Path
import subprocess,json,gzip,base64,datetime
root=Path(ROOT);state=json.loads((root/'campaign.json').read_text())
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
data={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state,'jobs':{},'files':{},'generation':[]}
ids=','.join(state['jobs'].values())
for tool,args in [('squeue',['-h','-j',ids,'-o','%i|%j|%T|%M|%R']),('sacct',['-X','-n','-P','-j',ids,'--format=JobID,JobName,State,ExitCode,Elapsed,NodeList'])]:
 data['jobs'][tool]=subprocess.check_output([tool,*args],text=True)
patterns=['gate/passed.json','gate/runtime.json','legacy_replay/passed.json','baseline100/completed.json','smoke/*/completed.json','training/*/completed.json','training/*/loss-components.jsonl','training/*/gradient-norms.jsonl','training/*/fixed-probe.jsonl','diagnostics/*/completed.json','diagnostics/*/records.jsonl','evaluation/*/completed.json','weight0*/*/completed.json','weight0*/*/loss-components.jsonl','weight0*/*/gradient-norms.jsonl','weight0*/*/fixed-probe.jsonl','weight0*/*/teacher-probe.jsonl','diagnostics/graph_generation/*/*/completed.json','diagnostics/graph_generation/*/*/*/intervention.json','diagnostics/graph_generation/*/*/*/graph-trace.jsonl']
for pattern in patterns:
 for p in root.glob(pattern):data['files'][str(p.relative_to(root))]=rows(p) if p.suffix=='.jsonl' else read(p)
for pattern in ['legacy_replay/*/generation/summary.json','baseline100/generation/summary.json','evaluation/*/*/*/generation/summary.json','weight_evaluation/*/*/generation/summary.json','diagnostics/graph_generation/*/*/*/generation/summary.json']:
 for p in root.glob(pattern):
  item={'path':str(p.parent.relative_to(root)),'summary':read(p),'records':[]}
  for r in rows(p.parent/'samples.jsonl'):
   tokens=r['sample_token_ids'];eos=[i for i,t in enumerate(tokens) if t==50256]
   compact={k:r.get(k) for k in ['seed','pair_key','sampling_mode','requested_nfe_budget','measured_nfe','reference_lm','repetition_rate','wall_clock_seconds','unresolved_mask_tokens']}
   compact.update(output_token_count=len(tokens),first_eos_position=eos[0] if eos else None,first_nonleading_eos_position=next((i for i in eos if i>0),None),eos_count=len(eos))
   item['records'].append(compact)
  manifest=p.parent/'run_manifest.json'
  if manifest.exists():
   m=read(manifest);item['manifest']={k:m.get(k) for k in ['runtime','source','started_utc','ended_utc','reference_lm']}
  data['generation'].append(item)
print(base64.b64encode(gzip.compress(json.dumps(data).encode())).decode())
'''.replace('ROOT',repr(s['root']),1)
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,timeout=55)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
import base64
raw=gzip.decompress(base64.b64decode(r.stdout));(w/'snapshot.json.gz').write_bytes(gzip.compress(raw));data=json.loads(raw)
print(data['jobs']['squeue']);print('files',len(data['files']),'generation groups',len(data['generation']))
print('audit',data['files'].get('gate/passed.json',{}).get('status'))
