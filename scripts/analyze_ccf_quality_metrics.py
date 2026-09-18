"""Post-process saved unconditional samples; no model or sampler is run.

Repetition and corpus distinct-n definitions follow this project's
evaluation/generation_metrics.py. Hugging Face tokenizers loads the exact
tokenizer artifact used by the saved GPT-2-large scorer.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import gzip
import hashlib
import json
import math
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'work/quality-metrics'
OUT = ROOT / 'outputs'
N_VALUES = (1, 2, 4)
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260917
LABELS = {
    ('A', 'static_static'): 'MDLM',
    ('B', 'fixed_dynamic'): 'Basic FD',
    ('B', 'dynamic_dynamic'): 'Basic DD',
    ('C', 'fixed_dynamic'): 'Separate R8 FD',
    ('C', 'dynamic_dynamic'): 'Separate R8 DD',
    ('D', 'fixed_dynamic'): 'Separate R16 FD',
    ('D', 'dynamic_dynamic'): 'Separate R16 DD',
}


def ngrams(ids, n):
    return [tuple(ids[i:i+n]) for i in range(max(0, len(ids)-n+1))]


def repetition(ids, n):
    grams = ngrams(ids, n)
    return 1-len(set(grams))/len(grams) if grams else 0.0


def content_scope(ids, bos, eos):
    """Ignore a leading BOS only; stop at the first subsequent EOS.

    Positions are zero-based internally. The final EOS is in the scoring
    prefix, but neither BOS nor EOS contributes to content diversity metrics.
    """
    start = int(bool(ids) and ids[0] == bos)
    terminal = next((i for i in range(start, len(ids)) if ids[i] == eos), None)
    content_end = terminal if terminal is not None else len(ids)
    prefix_end = terminal+1 if terminal is not None else len(ids)
    return ids[start:content_end], terminal, prefix_end


def distribution(values):
    a = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(a):
        return {'count': 0, 'mean': None, 'median': None, 'p10': None,
                'p90': None, 'min': None, 'max': None}
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p10': float(np.quantile(a, .1)), 'p90': float(np.quantile(a, .9)),
            'min': float(a.min()), 'max': float(a.max())}


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def aggregate(sequences, draws):
    by_sample = {str(n): np.array([repetition(ids, n) for ids in sequences])
                 for n in N_VALUES}
    mean_rep = {n: float(v.mean()) for n,v in by_sample.items()}
    distinct = {}
    for n in N_VALUES:
        grams = Counter()
        for ids in sequences:
            grams.update(ngrams(ids, n))
        total = sum(grams.values())
        distinct[str(n)] = {'ratio': len(grams)/total if total else None,
                            'unique_ngrams': len(grams), 'total_ngrams': total}
    return {'mean_repetition': mean_rep,
            'repetition_ci95': {n: interval(v[draws].mean(axis=1)) for n,v in by_sample.items()},
            'distinct_n': distinct,
            'token_length': distribution([len(ids) for ids in sequences]),
            'fewer_than_n_tokens': {str(n): sum(len(ids)<n for ids in sequences) for n in N_VALUES}}


def check_definitions():
    assert math.isclose(repetition([1,2,1],1), 1/3)
    assert repetition([],4) == 0
    assert repetition([1,2,1,2],2) == 1/3 or math.isclose(repetition([1,2,1,2],2),1/3)
    assert content_scope([99,1,2,99,3],99,99) == ([1,2],3,4)
    assert content_scope([1,2,99,3],99,99) == ([1,2],2,3)
    assert content_scope([99,99,3],99,99) == ([],1,2)
    assert content_scope([99,1,2],99,99) == ([1,2],None,3)
    assert content_scope([1,2,3],99,99) == ([1,2,3],None,3)


check_definitions()
bundle = json.loads(gzip.decompress((WORK/'records.json.gz').read_bytes()))
tokenizer = Tokenizer.from_file(str(WORK/'tokenizer.json'))
tokenizer.no_truncation()
tokenizer.no_padding()
assert hashlib.sha256((WORK/'tokenizer.json').read_bytes()).hexdigest() == bundle['tokenizer_sha256']
rows = []
per_sample = []
failures = []
draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0,100,size=(BOOTSTRAPS,100))
run_status = {}
for run_name, run in bundle['runs'].items():
    run_status[run_name] = {'completed_configurations': len(run['rows']),
                          'planned_configurations': len(run['selection']['cells']),
                          'scheduler_states': [s['state_counts'] for s in run['statuses']]}
    first_cell = {}
    for index,c in enumerate(run['selection']['cells']):
        first_cell.setdefault((c['family'],c['arm'],c['sampling_steps']),index)
    for row in run['rows']:
        c = row['cell']
        label = LABELS[c['family'],c['arm']]
        records = sorted(row['token_records'],key=lambda r:r['pair_seed'])
        assert len(records)==row['samples']==100
        assert [r['pair_seed'] for r in records] == list(range(100001,100101))
        scorer = row['scorer']
        assert scorer['sequence_policy']=='retokenize_decoded_text_score_through_first_nonleading_eos_v1'
        assert scorer['tokenizer_revision']=='32b71b12589c2f8d625668d2335a01cac3249519'
        eos,bos = scorer['tokenizer_eos_token_id'],scorer['tokenizer_bos_token_id']
        raw_sequences=[]; raw_content=[]; evaluator_content=[]; scored_target_sequences=[]; stats=[]
        encoded = tokenizer.encode_batch([r['text'] for r in records],add_special_tokens=True)
        for r,enc in zip(records,encoded):
            raw=r['sample_token_ids']
            assert len(raw)==1024
            full=enc.ids
            ids=full[:scorer['max_length']]
            raw_c,raw_eos,_=content_scope(raw,bos,eos)
            eval_c,eval_eos,prefix_end=content_scope(ids,bos,eos)
            expected_scored=max(0,prefix_end-1)
            assert expected_scored==r['reference_lm']['token_count'], (run_name,row['index'],r['pair_seed'],expected_scored,r['reference_lm']['token_count'])
            for n in N_VALUES:
                assert math.isclose(repetition(raw,n),r['metrics']['repetition_rate'][str(n)],abs_tol=1e-12)
            raw_sequences.append(raw);raw_content.append(raw_c);evaluator_content.append(eval_c)
            scored_target_sequences.append(ids[1:prefix_end])
            stat={
                'run':run_name,'configuration_index':row['index'],'model':label,
                'checkpoint':None if c['family']=='A' else c['step'],
                'reverse_steps':c['sampling_steps'],'seed':r['pair_seed'],'pair_key':r['pair_key'],
                'raw_token_length':len(raw),'raw_content_tokens':len(raw_c),
                'retokenized_full_tokens':len(full),'evaluator_content_tokens':len(eval_c),
                'scored_tokens':expected_scored,'evaluator_content_characters':len(tokenizer.decode(eval_c,skip_special_tokens=False)),
                'raw_first_nonleading_eos_position_1based':raw_eos+1 if raw_eos is not None else None,
                'evaluator_first_nonleading_eos_position_1based':eval_eos+1 if eval_eos is not None else None,
                'raw_has_leading_bos':bool(raw and raw[0]==bos),'evaluator_has_leading_bos':bool(ids and ids[0]==bos),
                'raw_nonleading_eos_count':sum(x==eos for x in raw)-int(bool(raw and raw[0]==bos)),
                'retokenization_changed_ids':raw!=full,'evaluator_context_truncated':len(full)>scorer['max_length'],
                'raw_repetition':{str(n):repetition(raw,n) for n in N_VALUES},
                'raw_content_repetition':{str(n):repetition(raw_c,n) for n in N_VALUES},
                'evaluator_content_repetition':{str(n):repetition(eval_c,n) for n in N_VALUES},
                'reference_nll':r['reference_lm']['mean_nll_nats'],
                'reference_ppl':r['reference_lm']['perplexity'],'measured_nfe':r['measured_nfe'],
            }
            stats.append(stat)
        scopes={name:aggregate(seqs,draws) for name,seqs in [
            ('raw_full',raw_sequences),('raw_content_to_eos',raw_content),
            ('evaluator_content_to_eos',evaluator_content),('evaluator_scored_targets',scored_target_sequences)]}
        for n in N_VALUES:
            assert math.isclose(scopes['raw_full']['mean_repetition'][str(n)],row['repetition'][str(n)],abs_tol=1e-12)
            assert math.isclose(scopes['raw_full']['distinct_n'][str(n)]['ratio'],row['distinct_n'][str(n)],abs_tol=1e-12)
        total_tokens=sum(s['scored_tokens'] for s in stats)
        nll=sum(s['scored_tokens']*s['reference_nll'] for s in stats)/total_tokens
        assert total_tokens==row['scored_tokens']
        assert math.isclose(math.exp(nll),row['ppl'],rel_tol=1e-6)
        out={
            'run':run_name,'configuration_index':row['index'],'model':label,
            'checkpoint':None if c['family']=='A' else c['step'],
            'reverse_steps':c['sampling_steps'],'samples':len(records),
            'pilot_selected_primary':row['index']==first_cell[c['family'],c['arm'],c['sampling_steps']],
            'ppl':row['ppl'],'mean_nll':nll,'scopes':scopes,
            'evaluator_content_characters':distribution([s['evaluator_content_characters'] for s in stats]),
            'scored_token_length':distribution([s['scored_tokens'] for s in stats]),
            'raw_eos_position_1based':distribution([s['raw_first_nonleading_eos_position_1based'] for s in stats]),
            'evaluator_eos_position_1based':distribution([s['evaluator_first_nonleading_eos_position_1based'] for s in stats]),
            'raw_no_eos_count':sum(s['raw_first_nonleading_eos_position_1based'] is None for s in stats),
            'evaluator_no_eos_count':sum(s['evaluator_first_nonleading_eos_position_1based'] is None for s in stats),
            'early_eos_content_at_most_128_tokens_count':sum(s['evaluator_first_nonleading_eos_position_1based'] is not None and s['evaluator_content_tokens']<=128 for s in stats),
            'empty_content_count':sum(s['evaluator_content_tokens']==0 for s in stats),
            'raw_leading_bos_count':sum(s['raw_has_leading_bos'] for s in stats),
            'retokenization_changed_ids_count':sum(s['retokenization_changed_ids'] for s in stats),
            'evaluator_context_truncated_count':sum(s['evaluator_context_truncated'] for s in stats),
            'actual_nfe_counts':dict(Counter(s['measured_nfe'] for s in stats)),
            'source_samples_sha256':row['samples_sha256'],
            'validation':{'raw_metrics_match_saved':True,'scored_token_counts_match_all_samples':True,'ppl_matches_saved':True},
        }
        rows.append(out);per_sample.extend(stats)
        print(f"Processed {run_name} {row['index']}: {label}, {c['sampling_steps']} steps",flush=True)

by_key={(r['run'],r['configuration_index']):[s for s in per_sample if s['run']==r['run'] and s['configuration_index']==r['configuration_index']] for r in rows}
baselines={(r['run'],r['reverse_steps']):r for r in rows if r['model']=='MDLM'}
for r in rows:
    base=baselines[r['run'],r['reverse_steps']]
    a=by_key[r['run'],r['configuration_index']];b=by_key[base['run'],base['configuration_index']]
    assert [(x['seed'],x['pair_key']) for x in a]==[(x['seed'],x['pair_key']) for x in b]
    comparisons={}
    for n in N_VALUES:
        v=np.array([x['evaluator_content_repetition'][str(n)]-y['evaluator_content_repetition'][str(n)] for x,y in zip(a,b)])
        comparisons[f'repetition_{n}_difference']={'estimate':float(v.mean()),'ci95':interval(v[draws].mean(axis=1))}
    v=np.array([x['evaluator_content_tokens']-y['evaluator_content_tokens'] for x,y in zip(a,b)])
    comparisons['content_token_length_difference']={'estimate':float(v.mean()),'ci95':interval(v[draws].mean(axis=1))}
    r['paired_comparison_to_mdlm']=comparisons

protocol={
    'primary_scope':'evaluator_content_to_eos',
    'tokenizer_revision':'32b71b12589c2f8d625668d2335a01cac3249519',
    'tokenizer_sha256':bundle['tokenizer_sha256'],'tokenizers_version':'0.15.2',
    'repetition':'Per-sample 1 - unique n-grams / total n-grams, then mean across samples; zero when sample length < n.',
    'distinct':'Unique corpus n-grams / total corpus n-grams; no n-grams cross sample boundaries; undefined if denominator zero.',
    'primary_scope_definition':'Decode/re-tokenize as recorded for GPT-2-large; truncate at 1024 input tokens; ignore leading BOS when finding terminal EOS; retain content before first nonleading EOS, excluding BOS and terminal EOS.',
    'eos_position':'1-based input-token position, including any leading BOS; null if no EOS in that view. EOS position summaries condition on EOS observed.',
    'evaluator_scored_targets':'Exactly input positions 1 through first nonleading EOS inclusive (or context end); excludes first input token regardless of its identity. Auxiliary scope in JSON.',
    'raw_full':'All 1024 original token IDs including special tokens; reproduces historical metrics.',
    'bootstrap':{'resamples':BOOTSTRAPS,'seed':BOOTSTRAP_SEED,'unit':'paired generation seed','interval':'95% pointwise percentile','training_seed_variability':False,'distinct_n_interval':None},
    'attribution':'Metric definitions adapted from existing evaluation/generation_metrics.py. Tokenization uses Hugging Face tokenizers and the pinned GPT-2-large tokenizer artifact; no generator or scorer model rerun.',
}
result={'schema_version':1,'collected_utc':bundle['collected_utc'],'protocol':protocol,'runs':run_status,'rows':rows}
(OUT/'ccf-quality-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(OUT/'ccf-quality-metrics-per-sample.jsonl').write_text(''.join(json.dumps(s,allow_nan=False)+'\n' for s in per_sample))

def checkpoint(r):
    return 'Released' if r['checkpoint'] is None else f"{r['checkpoint']//1000}k"


def table(group,scope):
    lines=['| Model | Checkpoint | Steps | PPL | Rep-1 % | Rep-2 % | Rep-4 % | Distinct-2 % | Distinct-4 % | Mean tokens |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in group:
        s=r['scopes'][scope]
        vals=[r['model'],checkpoint(r),str(r['reverse_steps']),f"{r['ppl']:.2f}"]
        vals += [f"{100*s['mean_repetition'][str(n)]:.3f}" for n in N_VALUES]
        vals += [f"{100*s['distinct_n'][str(n)]['ratio']:.3f}" for n in (2,4)]
        vals += [f"{s['token_length']['mean']:.1f}"]
        lines.append('| '+' | '.join(vals)+' |')
    return lines


stamp=datetime.fromisoformat(bundle['collected_utc']).astimezone(ZoneInfo('America/Toronto')).strftime('%Y-%m-%d %H:%M:%S %Z')
primary=[r for r in rows if r['pilot_selected_primary']]
lines=['**CCF repetition, diversity, length, and EOS results**','',f'Saved-record snapshot: {stamp}.','',
       f"Computed from {len(rows)} completed configurations and {len(per_sample):,} saved samples, 100 per configuration. All 57 confirmation configurations at 8/16/32 reverse steps are complete. The four-step verification/evaluation arrays remain pending in this snapshot. No new generations or GPT-2 scoring runs were needed.",'',
       'The PPL/repetition tradeoff persists after EOS alignment. At 16 steps, Basic FD 6k improves PPL from 313.78 to 244.05, while repeated 4-grams increase from 0.278% to 0.776% and distinct bigrams decrease from 76.45% to 71.97%. The paired repetition increase is +0.498 percentage points (95% pointwise interval +0.280 to +0.738). Mean content lengths are 506.8 versus 531.3 tokens; both have 4% of samples without EOS in the evaluator window. These observations do not establish overall human preference.','',
       '**Definitions and validation**','',
       'The primary view uses the saved decoded text and the exact pinned GPT-2-large tokenizer, applies the evaluator\'s 1,024-input-token limit, and stops before the first nonleading EOS. Repetition and diversity exclude the leading BOS, when present, and terminal EOS. Text after EOS is excluded. Output lengths count these content tokens; character counts are available in the JSON. The auxiliary `evaluator_scored_targets` view includes exactly the token positions whose losses contribute to PPL, including terminal EOS and excluding the first input token.','',
       'Rep-n is the mean within-sample fraction of n-gram occurrences beyond their first occurrence. Distinct-n is the corpus-wide fraction of unique n-grams. N-grams never cross sample boundaries. These use different aggregation levels and are not complements. Values below are percentages, not counts. Higher distinct-n alone does not establish better language.','',
       'Validation passed for all samples: re-tokenized scored-token counts exactly match the saved GPT-2 score records; original raw repetition and distinct-n reproduce the stored summaries; recomputed token-weighted PPL matches the stored aggregate. Raw sequences are all 1,024 tokens.','',
       'Different content lengths affect repetition and distinct-n. EOS-aligned metrics describe the same text extent as the evaluator, but this does not remove length effects. Empty/short sequences are counted explicitly, and repetition follows the existing convention of zero when fewer than n tokens are available. No naive bootstrap intervals are assigned to corpus distinct-n.','',
       '**Pilot-selected checkpoints: primary text view**','',
       'These checkpoints were selected using the original 20-sample pilot before the 100 fresh confirmation samples were examined. MDLM is included at every budget. The full set of three selected checkpoints per CCF variant follows below.','']
lines += table(primary,'evaluator_content_to_eos')
lines += ['', '**Output length and EOS: pilot-selected checkpoints**','',
          'EOS positions are 1-based positions in the re-tokenized evaluator input, including any leading BOS. EOS median is computed only over samples with an EOS. “No EOS” means none within the evaluator\'s context; these lengths are capped, not observed stopping times. “Early EOS” means EOS after at most 128 content tokens.','',
          '| Model | Ckpt | Steps | Content tokens, median [p10, p90] | EOS position, median | No EOS % | Early EOS % | Empty % | Leading BOS % |',
          '|---|---|---:|---|---:|---:|---:|---:|---:|']
for r in primary:
    s=r['scopes']['evaluator_content_to_eos']['token_length'];e=r['evaluator_eos_position_1based']['median']
    lines.append(f"| {r['model']} | {checkpoint(r)} | {r['reverse_steps']} | {s['median']:.1f} [{s['p10']:.1f}, {s['p90']:.1f}] | {e:.1f} | {r['evaluator_no_eos_count']:.0f} | {r['early_eos_content_at_most_128_tokens_count']:.0f} | {r['empty_content_count']:.0f} | {r['raw_leading_bos_count']:.0f} |")
lines += ['', '**Paired repetition differences versus MDLM**','',
          'Differences are percentage points on the primary text view, with 95% pointwise intervals from 10,000 paired generation-seed bootstrap draws. Positive values mean more repetition. These intervals do not adjust for multiple comparisons or measure training-seed variation. Distinct-n remains descriptive.','',
          '| Model | Ckpt | Steps | Rep-1 delta [95% CI] | Rep-2 delta [95% CI] | Rep-4 delta [95% CI] |',
          '|---|---|---:|---|---|---|']
for r in primary:
    if r['model']=='MDLM':continue
    vals=[]
    for n in N_VALUES:
        v=r['paired_comparison_to_mdlm'][f'repetition_{n}_difference']
        vals.append(f"{100*v['estimate']:+.3f} [{100*v['ci95'][0]:+.3f}, {100*v['ci95'][1]:+.3f}]")
    lines.append('| '+' | '.join([r['model'],checkpoint(r),str(r['reverse_steps'])]+vals)+' |')
lines += ['', '**All completed configurations: primary text view**','']
lines += table(rows,'evaluator_content_to_eos')
lines += ['', '**All completed configurations: historical raw-token view**','',
          'This retains every original token position, including special tokens and everything after the first EOS, for continuity with the earlier results. PPL is still the original EOS-limited PPL; the metric scopes differ in this table.','']
lines += table(rows,'raw_full')
lines += ['', '**Machine-readable records and provenance**','',
          '[All aggregate metrics](ccf-quality-metrics.json) contain all four scopes, length/EOS quantiles, character lengths, boundary checks, source hashes, and intervals. [Per-sample metrics](ccf-quality-metrics-per-sample.jsonl) contain lengths, EOS positions, repetition, seed pairing, NFE, and saved evaluator scores without sample text or private cluster paths.','',
          'Repetition/distinct-n definitions follow the existing `evaluation/generation_metrics.py`; tokenization uses Hugging Face `tokenizers==0.15.2` and the saved GPT-2-large tokenizer revision `32b71b12589c2f8d625668d2335a01cac3249519`. The tokenizer artifact SHA-256 and source-sample hashes are recorded in the JSON. Distinct-n is an established generation-diversity measure; see [Li et al., 2016](https://aclanthology.org/N16-1014/).','',
          'These metrics describe the 8/16/32-step, 100-sample confirmation. Four-step results are tracked separately rather than mixed into these tables.','']
(OUT/'ccf-quality-metrics.md').write_text('\n'.join(lines))
print(json.dumps({'rows':len(rows),'samples':len(per_sample),'primary_rows':len(primary),'validation':'all_passed'}))
