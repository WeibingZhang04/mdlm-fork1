"""Read only this study's artifacts; retain all completed cells and failures."""
from pathlib import Path
import json,subprocess,gzip,base64
here=Path(__file__).resolve().parent;receipt=json.loads((here/'deployment-receipt.json').read_text())
script=r'''
from pathlib import Path
import json,subprocess,datetime,gzip,base64
root=Path(ROOT_VALUE)
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
data={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipt':read(root/'deployment.json'),'generation':[],'training':{},'jobs':{}}
ids=','.join(data['receipt']['jobs'].values())
for tool,args in [('squeue',['-h','-j',ids,'-o','%i|%T|%M|%R']),('sacct',['-X','-n','-P','-j',ids,'--format=JobID,State,Elapsed,ExitCode,NodeList'])]:data['jobs'][tool]=subprocess.check_output([tool,*args],text=True)
# rglob is confined to this user's unique experiment directory.
for p in sorted(root.rglob('generation/summary.json')):
 if any(part.startswith('gate') for part in p.relative_to(root).parts):continue
 folder=p.parent.parent
 item={'path':str(folder.relative_to(root)),'summary':read(p),'records':[],'completed':(folder/'completed.json').exists()}
 for r in rows(p.parent/'samples.jsonl'):
  tokens=r['sample_token_ids'];eos=[i for i,t in enumerate(tokens) if t==50256]
  item['records'].append({**{k:r[k] for k in ['pair_key','pair_seed','batch_seed','sample_index','sampling_mode','requested_nfe_budget','measured_nfe','reference_lm','metrics']},'output_length':len(tokens),'first_eos_position':eos[0] if eos else None,'first_nonleading_eos_position':next((i for i in eos if i>0),None),'eos_count':len(eos)})
 for name in ('variant.json','args.json'):
  if (folder/name).exists():item[name]=read(folder/name)
 manifest=p.parent/'run_manifest.json'
 if manifest.exists():item['manifest']=read(manifest)
 trace=folder/'graph-trace.jsonl'
 if trace.exists():
  bins={}
  for r in rows(trace):
   rate=r['active_fraction'];label='0-.3' if rate<=.3 else '.3-.6' if rate<=.6 else '.6-1'
   b=bins.setdefault(label,{'calls':0,'totals':{}});b['calls']+=1
   for k in ['active_tokens','edges','isolated_fraction','largest_component','unique_anchors','chain_proposed_fraction','chain_selected_fraction','proposal_count','unique_proposals']:
    b['totals'][k]=b['totals'].get(k,0)+r[k]
  item['graph_bins']={k:{'calls':v['calls'],**{n:t/v['calls'] for n,t in v['totals'].items()}} for k,v in bins.items()}
 data['generation'].append(item)
for p in root.glob('train_*/*/*.json*'):
 if p.name in ('loss-components.jsonl','gradient-norms.jsonl','fixed-probe.jsonl','teacher-probe.jsonl','completed.json','initial-state.json','input-identity.jsonl','topology-variant.json','request.json'):
  data['training'][str(p.relative_to(root))]=rows(p) if p.suffix=='.jsonl' else read(p)
gate=root/data['receipt'].get('gate_results','gate')/'passed.json'
if not gate.exists() and (root/'gate_retry2/passed.json').exists():gate=root/'gate_retry2/passed.json'
data['gate_passed']=read(gate) if gate.exists() else None
print(base64.b64encode(gzip.compress(json.dumps(data).encode())).decode())
'''.replace('ROOT_VALUE',repr(receipt['root']))
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,timeout=55)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
raw=base64.b64decode(r.stdout);(here/'results-snapshot.json.gz').write_bytes(raw);d=json.loads(gzip.decompress(raw));print(d['jobs']['squeue']);print('Completed generation folders:',len(d['generation']),'GPU gate passed:',bool(d['gate_passed']))
