"""Evaluate all continued adapters in one allocation, with matched baselines."""
import sys,subprocess,os
from pathlib import Path
from campaign_common import *
from topology_variants import TRAIN_VARIANTS
EXPERIMENT=Path(os.environ['CCF_TOPOLOGY_ROOT']);v=VARIANTS[3];folder=EXPERIMENT/'trained_confirmation';folder.mkdir(parents=True,exist_ok=False)
cases=[('before6k','native',ROOT/'gate/generation'/v['name'])]
for variant in TRAIN_VARIANTS:
    ns='topology_improvements/'+EXPERIMENT.name+'/train_'+variant
    assert (ROOT/ns/v['name']/'completed.json').is_file()
    cases.append(('after8k_'+variant,variant,ROOT/(ns+'_exports')/v['name']/'step008000'))
for label,variant,exp in cases:
    dest=folder/label;dest.mkdir()
    modes=('factorized','structured_marginal','structured_joint') if label=='before6k' else ('structured_marginal','structured_joint')
    args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(8,32),samples=100,seed=230001,modes=modes)
    save(dest/'args.json',args)
    with (dest/'run.log').open('w') as f:subprocess.run([sys.executable,str(Path(__file__).with_name('run_variant_generation.py')),STATE['code'],str(dest/'args.json'),variant,str(dest/'graph-trace.jsonl')],stdout=f,stderr=subprocess.STDOUT,check=True)
    save(dest/'completed.json',dict(label=label,variant=variant,samples_per_cell=100,base_seed=230001))
save(folder/'completed.json',dict(all_variants=TRAIN_VARIANTS,samples_per_cell=100,base_seed=230001))
