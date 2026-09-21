from pathlib import Path
import gzip,json,statistics,collections,random
w=Path(__file__).parent;s=json.loads(gzip.decompress((w/'snapshot.json.gz').read_bytes()))
rows=s['files'].get('diagnostics/topology6k/records.jsonl',[])
idx={(r['variant'],r['sequence_length'],r['mask_rate'],r['sample'],r['intervention']):r for r in rows}
by=collections.defaultdict(list)
for r in rows:by[(r['variant'],r['sequence_length'],r['intervention'])].append(r)
def interval(values):
 rng=random.Random(92026);n=len(values)
 draws=sorted(statistics.mean(rng.choices(values,k=n)) for _ in range(2000))
 return [draws[49],draws[1949]]
summary=[]
for (variant,length,intervention),group in sorted(by.items()):
 if len(group)!=64:continue
 row=dict(variant=variant,length=length,intervention=intervention,examples=16,mask_rates=4)
 for key in ['joint_nll','dependence_gain','backbone_gain','edges','edge_fraction_of_tree','isolated_fraction','mean_edge_span']:
  row[key]=statistics.mean(r[key] for r in group)
 per_text=collections.defaultdict(list)
 for r in group:
  native=idx[(variant,length,r['mask_rate'],r['sample'],'native')]
  per_text[r['sample']].append(r['joint_nll']-native['joint_nll'])
 values=[statistics.mean(v) for v in per_text.values()]
 row['nll_delta_vs_native']=statistics.mean(values);row['delta_bootstrap95_by_text']=interval(values)
 teacher={}
 if intervention=='native' and group[0].get('teacher'):
  for part in ['edge','anchor','slot']:
   valid=[r['teacher'][part] for r in group if r['teacher'][part]['valid']]
   teacher[part]={'valid_fraction':len(valid)/len(group),**{k:statistics.mean(x[k] for x in valid) for k in ['cross_entropy','teacher_entropy','kl_to_teacher','uniform_cross_entropy','improvement_over_uniform','choices']}}
 row['teacher']=teacher;summary.append(row)
(w/'topology-analysis.json').write_text(json.dumps({'collected_utc':s['collected_utc'],'completed':s['files'].get('diagnostics/topology6k/completed.json'),'summary':summary},indent=2))
for row in summary:
 if row['length']==1024:print(row['variant'],row['intervention'],'nll',round(row['joint_nll'],5),'delta',round(row['nll_delta_vs_native'],5),'CI',[round(x,5) for x in row['delta_bootstrap95_by_text']],'density',round(row['edge_fraction_of_tree'],3))
for row in summary:
 if row['length']==1024 and row['teacher']:print(row['variant'],'teacher',json.dumps(row['teacher']))
