"""Report every available cell; paired, token-weighted bootstrap, no ranking filter."""
from pathlib import Path
import json,gzip,math
import numpy as np

def paired_comparison(a,b,draws=10000):
    """a = intervention; b = control. Same sample IDs and seeds are required."""
    def keyed(rows):
        result={r['pair_key']:r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate paired sample IDs')
        return result
    aa,bb=keyed(a),keyed(b)
    if set(aa)!=set(bb):raise ValueError('Mismatched sample sets; do not drop samples')
    keys=sorted(aa);av=[];bv=[]
    for k in keys:
        if aa[k]['pair_seed']!=bb[k]['pair_seed']:raise ValueError('Unmatched generation seed')
        for row,dest in [(aa[k],av),(bb[k],bv)]:
            score=row['reference_lm']
            if not score or score['token_count']<=0:raise ValueError('Missing/empty score; incomplete comparison')
            dest.append([score['mean_nll_nats']*score['token_count'],score['token_count']])
        for field in ('model_name_or_path','revision','sequence_policy'):
            if aa[k]['reference_lm'][field]!=bb[k]['reference_lm'][field]:raise ValueError('Scoring policies differ')
    av=np.array(av);bv=np.array(bv)
    point=av[:,0].sum()/av[:,1].sum()-bv[:,0].sum()/bv[:,1].sum()
    idx=np.random.default_rng(20260921).integers(len(keys),size=(draws,len(keys)))
    sa=av[idx].sum(1);sb=bv[idx].sum(1);delta=sa[:,0]/sa[:,1]-sb[:,0]/sb[:,1]
    lo,hi=np.quantile(delta,[.025,.975])
    return dict(samples=len(keys),delta_mean_nll=float(point),ppl_ratio=float(np.exp(point)),ppl_ratio_ci95=[float(np.exp(lo)),float(np.exp(hi))],ci_scope='Paired-sample descriptive percentile interval, not simultaneous across variants; one training seed',bootstrap_draws=draws)

def analyze(data):
    cells=[];index={}
    for item in data['generation']:
        for group in item['summary']['groups']:
            mode=group['sampling_mode'];budget=group['requested_nfe_budget']
            rows=[r for r in item['records'] if r['sampling_mode']==mode and r['requested_nfe_budget']==budget]
            score=group.get('reference_lm')
            if not score:continue
            eos=[r['first_nonleading_eos_position'] for r in rows if r['first_nonleading_eos_position'] is not None]
            cell=dict(path=item['path'],mode=mode,denoising_steps=budget-1,measured_nfe=group['measured_nfe_values'],samples=len(rows),ppl=score['perplexity'],repetition=group['mean_repetition_rate'],distinct=group['distinct_n'],mean_output_length=float(np.mean([r['output_length'] for r in rows])),fraction_with_nonleading_eos=len(eos)/len(rows),mean_nonleading_eos_given_present=float(np.mean(eos)) if eos else None,mean_scored_tokens=score['num_scored_tokens']/len(rows),completed=item['completed'])
            cells.append(cell);index[(item['path'],mode,budget)]=(cell,rows)
    comparisons=[]
    for (path,mode,budget),(cell,rows) in index.items():
        if path.startswith(('dev/','confirm/')) and not path.endswith('/native'):
            baseline=path.rsplit('/',1)[0]+'/native'
        elif path.startswith('trained_confirmation/after8k_') and not path.endswith('after8k_native'):
            baseline='trained_confirmation/after8k_native'
        else:continue
        control=index.get((baseline,mode,budget))
        if control:
            if not (cell['completed'] and control[0]['completed']):continue
            comparison=dict(intervention=path,control=baseline,mode=mode,denoising_steps=budget-1)
            try:comparison.update(paired_comparison(rows,control[1]))
            except ValueError as e:comparison['not_comparable_reason']=str(e)
            comparisons.append(comparison)
    return dict(collected_utc=data['collected_utc'],gate_passed=bool(data['gate_passed']),cells=cells,comparisons=comparisons,jobs=data['jobs'],interpretation='All outcomes retained. New random seeds assess sampling stability, not an untouched corpus test. Evaluate repetition/EOS tradeoffs alongside PPL. No completed cell implies no quality conclusion.')

if __name__=='__main__':
    here=Path(__file__).resolve().parent;result=analyze(json.loads(gzip.decompress((here/'results-snapshot.json.gz').read_bytes())))
    (here/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# DD topology study — current results','',f"Collected: {result['collected_utc']}",'','PPL is token-weighted GPT2-large perplexity; lower is better. Steps are denoising updates. Each row retains its actual NFE.','', '| Case | Mode | Steps | NFE | Samples | PPL | Rep-2 | Distinct-2 | Mean scored tokens |','|---|---|---:|---|---:|---:|---:|---:|---:|']
    for c in result['cells']:lines.append(f"| {c['path']} | {c['mode']} | {c['denoising_steps']} | {c['measured_nfe']} | {c['samples']} | {c['ppl']:.2f} | {c['repetition']['2']:.3f} | {c['distinct']['2']:.3f} | {c['mean_scored_tokens']:.1f} |")
    if not result['cells']:lines+=['','No completed, scored generation cells yet. No improvement claim is supported.']
    lines+=['','All paired confidence intervals and full quality metrics are in analysis.json. Secondary comparisons are exploratory; intervals do not adjust for the full set of comparisons.','',result['interpretation']]
    (here/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('Generation cells:',len(result['cells']),'paired comparisons:',len(result['comparisons']))
