"""All cells and prespecified contrasts, with paired descriptive uncertainty."""
from pathlib import Path
import json,gzip,math,collections,importlib.util
import numpy as np
here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired_stats',here.parent/'topology-improvements/analyze_results.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
data=json.loads(gzip.decompress((here/'snapshot.json.gz').read_bytes()))
result={'collected_utc':data['collected_utc'],'studies':{}}
for name,s in data['studies'].items():
 cells=[];index={}
 for item in s['generation']:
  for g in item['summary']['groups']:
   if not g.get('reference_lm'):continue
   mode=g['sampling_mode'];budget=g['requested_nfe_budget'];rows=[r for r in item['records'] if r['sampling_mode']==mode and r['requested_nfe_budget']==budget]
   assert len(rows)==100
   score=g['reference_lm'];eos=[r['first_nonleading_eos_position'] for r in rows if r['first_nonleading_eos_position'] is not None]
   c=dict(case=item['case'],mode=mode,steps=budget-1,ppl=score['perplexity'],samples=len(rows),nfe=g['measured_nfe_values'],repetition=g['mean_repetition_rate'],distinct=g['distinct_n'],mean_scored_tokens=score['num_scored_tokens']/len(rows),fraction_with_eos=len(eos)/len(rows),mean_eos_position_if_present=sum(eos)/len(eos) if eos else None,completed=item['completed'])
   cells.append(c);index[(c['case'],mode,budget)]=(c,rows)
 contrasts=[]
 if name=='causal':
  pairs=[('after8k_bf16','structured_joint','after8k_fp32','structured_joint'),('after8k_bf16','structured_joint','before6k','structured_joint'),('after8k_fp32','structured_joint','before6k','structured_joint'),('after8k_bf16','structured_marginal','after8k_fp32','structured_marginal')]
 else:
  pairs=[]
  for family in ['FD','DD']:
   for label in ['old6k','fresh6k','fresh15k']:pairs.append((family+'_'+label,'structured_joint','FD_old6k','factorized'))
   pairs += [(family+'_fresh6k','structured_joint',family+'_old6k','structured_joint'),(family+'_fresh15k','structured_joint',family+'_fresh6k','structured_joint')]
  pairs += [('MDLM_fp32_normalization','factorized','FD_old6k','factorized'),('neutral_factors','structured_joint','MDLM_fp32_normalization','factorized'),('neutral_factors','structured_marginal','MDLM_fp32_normalization','factorized'),('FD_old6k','structured_joint','neutral_factors','structured_joint'),('FD_fresh6k','structured_joint','neutral_factors','structured_joint')]
 for case,mode,control,controlmode in pairs:
  for budget in [9,33]:
   a=index.get((case,mode,budget));b=index.get((control,controlmode,budget))
   if a and b and a[0]['completed'] and b[0]['completed']:
    contrasts.append(dict(case=case,mode=mode,control=control,control_mode=controlmode,steps=budget-1,**stats.paired_comparison(a[1],b[1])))
 d=dict(cells=cells,contrasts=contrasts,jobs=s['jobs'],errors=s['errors'])
 conditional=s['files'].get('conditional/records.jsonl',[])
 if conditional:
  groups=collections.defaultdict(list)
  for r in conditional:groups[r['case']].append(r)
  summaries=[]
  for case,rows in groups.items():
   n=sum(r['active_tokens'] for r in rows);sums={k:sum(r[k] for r in rows) for k in rows[0] if k.endswith('_sum')}
   summaries.append(dict(case=case,contexts=len(rows),active_tokens=n,**{k[:-4]:v/n for k,v in sums.items()},backbone_to_joint_gain=(sums['backbone_nll_sum']-sums['joint_nll_sum'])/n,dependence_gain=(sums['marginal_nll_sum']-sums['joint_nll_sum'])/n))
  for summary in summaries:
   rows=groups[summary['case']];documents=collections.defaultdict(lambda:[0.,0.,0.,0.])
   for r in rows:
    doc=documents[r['sample']];doc[0]+=r['backbone_nll_sum']-r['joint_nll_sum'];doc[1]+=r['marginal_nll_sum']-r['joint_nll_sum'];doc[2]+=r['head_marginal_entropy_sum']-r['backbone_entropy_sum'];doc[3]+=r['active_tokens']
   vals=np.array([documents[k] for k in sorted(documents)]);idx=np.random.default_rng(20260921).integers(len(vals),size=(10000,len(vals)));sums=vals[idx].sum(1)
   summary['document_bootstrap_ci95']={metric:np.quantile(sums[:,j]/sums[:,3],[.025,.975]).tolist() for j,metric in enumerate(['backbone_to_joint_gain','dependence_gain','entropy_change'])}
   summary['uncertainty_scope']='Descriptive paired bootstrap of 16 validation examples, keeping all four masks together; no multiple-comparison correction'
  d['conditional']=summaries
  if 'conditional/passed.json' in s['files']:
   identities={case:sorted((r['sample'],r['mask_rate'],r['clean_tokens_sha256'],r['mask_sha256']) for r in rows) for case,rows in groups.items()}
   assert len(identities)==6 and all(i==next(iter(identities.values())) for i in identities.values())
   d['conditional_pairing_verified']=True
 rescored=s['files'].get('scoring/records.jsonl',[])
 if rescored:
  groups=collections.defaultdict(lambda:collections.defaultdict(lambda:[0.,0,0]))
  for r in rescored:
   for policy,score in r['policies'].items():
    a=groups[(r['case'],r['mode'],r['budget']-1)][policy];a[0]+=score['nll_sum'];a[1]+=score['token_count'];a[2]+=1
  d['scoring_sensitivity']=[dict(case=k[0],mode=k[1],steps=k[2],policy=p,ppl=math.exp(v[0]/v[1]),samples=v[2],tokens=v[1]) for k,policies in groups.items() for p,v in policies.items()]
 result['studies'][name]=d
(here/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# CCF regression diagnostics','',result['collected_utc'],'','All generation cells use100 samples. PPL is GPT2-large under the original EOS scoring policy. Secondary scores remain separately labeled. One training seed; confidence intervals are descriptive.','']
for name,d in result['studies'].items():
 lines += ['## '+name,'','| Case | Sampling | Denoising steps | PPL | Rep-2 | Distinct-2 | Scored tokens/sample |','|---|---|---:|---:|---:|---:|---:|']
 for c in d['cells']:lines.append(f"| {c['case']} | {c['mode']} | {c['steps']} | {c['ppl']:.2f} | {c['repetition']['2']:.3f} | {c['distinct']['2']:.3f} | {c['mean_scored_tokens']:.1f} |")
 if not d['cells']:lines += ['','No completed scored generation case yet.']
 lines += ['']
(here/'RESULTS.md').write_text('\n'.join(lines).rstrip()+'\n')
print(json.dumps({name:{'cells':len(d['cells']),'contrasts':len(d['contrasts']),'conditional':d.get('conditional',[]),'errors':d['errors']} for name,d in result['studies'].items()},indent=2))
