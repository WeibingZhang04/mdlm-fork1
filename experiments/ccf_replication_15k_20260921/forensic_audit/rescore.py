"""Audit primary score independently; retain fixed-length policies as secondary."""
import sys,json,math
from pathlib import Path
from campaign_common import *
from common import STUDY
sys.path.insert(0,STATE['code'])
import torch
import torch.nn.functional as F
from evaluation.generation_metrics import TransformersReferenceLMScorer
folder=STUDY/'scoring';folder.mkdir(exist_ok=False)
scorer=TransformersReferenceLMScorer('gpt2-large',revision='32b71b12589c2f8d625668d2335a01cac3249519',device='cuda',batch_size=1,max_length=1024,dtype='float32')
save(folder/'runtime.json',scorer.runtime_identity());largest=0.;count=0
with torch.no_grad(),(folder/'records.jsonl').open('w') as out:
    for samplefile in sorted((STUDY/'generation').glob('*/generation/samples.jsonl')):
        case=samplefile.parent.parent.name
        for line in samplefile.read_text().splitlines():
            r=json.loads(line);encoded=scorer.tokenizer([r['text']],return_tensors='pt',padding=True,truncation=True,max_length=1024,add_special_tokens=True)
            ids=encoded['input_ids'].to('cuda');att=encoded['attention_mask'].to('cuda');n=int(att.sum())
            logits=scorer.model(input_ids=ids,attention_mask=att).logits
            losses=F.cross_entropy(logits[:,:-1].float().transpose(1,2),ids[:,1:],reduction='none')[0,:n-1]
            assert len(losses)>0
            eos=next((i for i,t in enumerate(ids[0,:n].tolist()) if i>0 and t==scorer.tokenizer.eos_token_id),n-1)
            primary=losses[:eos];stored=r['reference_lm'];error=abs(float(primary.mean())-stored['mean_nll_nats']);largest=max(largest,error)
            assert len(primary)==stored['token_count'] and error<2e-5,(case,r['pair_key'],error)
            scores={}
            for policy,values in [('primary_through_first_nonleading_eos',primary),('secondary_prefix256_ignore_eos',losses[:255]),('secondary_full1024_ignore_eos',losses)]:
                scores[policy]=dict(nll_sum=float(values.double().sum()),token_count=len(values))
            out.write(json.dumps(dict(case=case,pair_key=r['pair_key'],pair_seed=r['pair_seed'],mode=r['sampling_mode'],budget=r['requested_nfe_budget'],retokenized_length=n,first_nonleading_eos_position=eos if eos<n-1 or int(ids[0,eos])==scorer.tokenizer.eos_token_id else None,policies=scores,primary_absolute_error=error))+'\n');count+=1
        print(case,'rescored',flush=True)
save(folder/'completed.json',dict(status='completed',samples=count,max_primary_absolute_error=largest,secondary_policies_do_not_replace_primary=True))
