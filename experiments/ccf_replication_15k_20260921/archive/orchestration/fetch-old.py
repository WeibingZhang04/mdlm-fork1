from pathlib import Path
import subprocess,json,gzip,difflib,hashlib
w=Path(__file__).parent
new=json.loads(gzip.decompress((w/'repo-source.json.gz').read_bytes()))
names=['dataloader.py','data_provenance.py','main.py','noise_schedule.py','structured_utils.py','structured_pairing.py','evaluation/generation_reference_lm.py','configs/data/train_openwebtext_pinned.yaml','scripts/train_four_ccf_matched_1k.sh','scripts/run_ccf_separate_7k.sh']
code='''
from pathlib import Path
import json,gzip,sys
root=Path('/u401/n23zhang/mdlm_data/tree_mdlm_cache/code/ccf_confirmation_100.daiam2ji')
names=NAMES
r={n:(root/n).read_text() for n in names if (root/n).exists()}
sys.stdout.buffer.write(gzip.compress(json.dumps(r).encode()))
'''.replace('NAMES',repr(names))
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','n23zhangWatGPU','python3','-'],input=code.encode(),capture_output=True,check=True,timeout=50)
old=json.loads(gzip.decompress(r.stdout));(w/'old-extra.json').write_text(json.dumps(old))
for n,a in old.items():
 b=new['files'].get(n,'');diff=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='old/'+n,tofile='new/'+n))
 p=w/'diffs'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(diff)
 print(n,'SAME' if not diff else str(len(diff))+' diff chars')
