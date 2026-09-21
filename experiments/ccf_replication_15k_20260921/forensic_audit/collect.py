from pathlib import Path
import json,subprocess,gzip,base64
here=Path(__file__).resolve().parent
receipts={k:json.loads(p.read_text()) for k,p in [('forensic',here/'deployment.json'),('causal',here.parent/'precision-causal/deployment.json')]}
script=r'''
from pathlib import Path
import json,subprocess,datetime,gzip,base64
receipts=RECEIPTS

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
data={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'studies':{}}
for name,receipt in receipts.items():
 root=Path(receipt['root']);jobs=list(receipt['jobs'].values()) if 'jobs' in receipt else [receipt['job']]
 d={'root':str(root),'receipt':receipt,'jobs':{},'files':{},'generation':[]}
 for tool,args in [('squeue',['-h','-j',','.join(jobs),'-o','%i|%T|%M|%R']),('sacct',['-X','-n','-P','-j',','.join(jobs),'--format=JobID,State,Elapsed,ExitCode,NodeList'])]:d['jobs'][tool]=subprocess.check_output([tool,*args],text=True)
 patterns=['*pair-passed.json','conditional/passed.json','conditional/records.jsonl','scoring/completed.json','scoring/records.jsonl','scoring/runtime.json','train_*/*/initial-state.json','train_*/*/input-identity.jsonl','train_*/*/loss-components.jsonl','train_*/*/fixed-probe.jsonl','train_*/*/gradient-norms.jsonl','train_*/*/completed.json']
 for pattern in patterns:
  actual=pattern.replace('conditional/',receipt.get('conditional_output','conditional')+'/',1) if pattern.startswith('conditional/') else pattern
  for p in root.glob(actual):
   key=str(p.relative_to(root))
   if pattern.startswith('conditional/'):key='conditional/'+p.name
   d['files'][key]=rows(p) if p.suffix=='.jsonl' else read(p)
 for parent in ['generation','evaluation']:
  for p in (root/parent).glob('*/generation/summary.json'):
   case=p.parent.parent;item={'case':case.name,'summary':read(p),'records':[],'completed':(case/'completed.json').is_file()}
   for r in rows(p.parent/'samples.jsonl'):
    tokens=r['sample_token_ids'];eos=[i for i,t in enumerate(tokens) if t==50256 and i>0]
    item['records'].append({**{k:r[k] for k in ['pair_key','pair_seed','batch_seed','sample_index','sampling_mode','requested_nfe_budget','measured_nfe','reference_lm','metrics']},'output_length':len(tokens),'first_nonleading_eos_position':eos[0] if eos else None,'eos_count':len(eos)})
   for f in ['control.json','control-passed.json','generation-precision.json']:
    if (case/f).exists():item[f]=read(case/f)
   d['generation'].append(item)
 d['errors']={};d['historical_errors']={}
 for p in root.glob('*.err'):
  text=p.read_text(errors='replace')
  if 'Traceback' in text:
   target=d['errors'] if any(j in p.name for j in jobs) else d['historical_errors']
   target[p.name]=text[text.rfind('Traceback'):][-5000:]
 data['studies'][name]=d
print(base64.b64encode(gzip.compress(json.dumps(data).encode())).decode())
'''.replace('RECEIPTS',repr(receipts))
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,check=True,timeout=55)
raw=base64.b64decode(r.stdout);(here/'snapshot.json.gz').write_bytes(raw);d=json.loads(gzip.decompress(raw))
print(d['collected_utc'])
for name,s in d['studies'].items():
 print(name,s['jobs']['squeue'].strip(),len(s['generation']),'generation cases',len(s['errors']),'errors')
 for key in ['conditional/passed.json','train-pair-passed.json','scoring/completed.json']:
  if key in s['files']:print(key,'complete')
