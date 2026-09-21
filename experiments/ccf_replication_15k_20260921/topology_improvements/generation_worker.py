"""Predeclared complete generation matrix, including negative results."""
import sys,subprocess,json,os
from pathlib import Path
from campaign_common import ROOT,STATE,VARIANTS,generation_args,save
from topology_variants import INFERENCE_VARIANTS
EXPERIMENT=Path(os.environ['CCF_TOPOLOGY_ROOT']);HELPERS=Path(__file__).parent
stage=sys.argv[1];index=int(sys.argv[2])
if stage=='dev':v=VARIANTS[[3,7][index//2]];length=[128,1024][index%2];samples=20;seed=210001
elif stage=='confirm':v=VARIANTS[[3,7][index]];length=1024;samples=100;seed=220001
else:raise ValueError(stage)
folder=EXPERIMENT/stage/v['name']/f'length{length}';folder.mkdir(parents=True,exist_ok=False)
exp=ROOT/'gate/generation'/v['name']
for variant in INFERENCE_VARIANTS:
    dest=folder/variant;dest.mkdir()
    modes=('factorized','structured_joint') if variant=='native' else ('structured_joint',)
    args=generation_args(v,exp/'adapter.safetensors',exp/'adapter.manifest.json',dest/'generation',steps=(8,32),samples=samples,seed=seed,modes=modes,length=length)
    save(dest/'args.json',args)
    with (dest/'run.log').open('w') as f:subprocess.run([sys.executable,str(HELPERS/'run_variant_generation.py'),STATE['code'],str(dest/'args.json'),variant,str(dest/'graph-trace.jsonl')],check=True,stdout=f,stderr=subprocess.STDOUT)
    save(dest/'completed.json',dict(stage=stage,variant=variant,samples_per_cell=samples,seed=seed))
save(folder/'completed.json',dict(stage=stage,all_variants=INFERENCE_VARIANTS,samples_per_cell=samples,sequence_length=length,base_seed=seed,denoising_steps=[8,32]))
