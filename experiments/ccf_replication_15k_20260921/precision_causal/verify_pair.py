import os,json
from pathlib import Path
from campaign_common import ROOT,save
study=Path(os.environ['CCF_PRECISION_STUDY'])
for stage in ('smoke','train'):
    folders=[study/(stage+'_'+p)/'basic_dynamic_dynamic' for p in ('bf16','fp32')]
    if not all((p/'completed.json').is_file() for p in folders):continue
    initial=[json.loads((p/'initial-state.json').read_text()) for p in folders]
    assert initial[0]['initial_head_sha256']==initial[1]['initial_head_sha256']
    assert initial[0]['gpu']==initial[1]['gpu']
    inputs=[[json.loads(l) for l in (p/'input-identity.jsonl').read_text().splitlines()] for p in folders]
    assert len(inputs[0])==len(inputs[1])==3
    assert inputs[0]==inputs[1],'First three inputs or RNG states differ'
    assert initial[0]['cos_dtype']=='torch.bfloat16' and initial[1]['cos_dtype']=='torch.float32'
    save(study/(stage+'-pair-passed.json'),dict(status='passed',same_initial_head=True,first_three_batches_and_all_recorded_rng_states_exact=True,same_gpu=initial[0]['gpu'],only_declared_intervention='positional cache initialized under BF16 versus FP32'))
assert (study/'smoke-pair-passed.json').is_file()
