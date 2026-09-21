from pathlib import Path
import subprocess,json,gzip
w=Path(__file__).parent
code=r'''
from pathlib import Path
import subprocess,json,gzip,sys
p=Path.home()/'clean_tree_mdlm/mdlm-fork1'
def git(*args):return subprocess.run(['git',*args],cwd=p,capture_output=True,check=True).stdout
names=git('ls-files').decode().splitlines()
files={n:(p/n).read_text() for n in names if Path(n).suffix in ['.py','.yaml','.yml','.sh','.md','.toml'] and not n.startswith(('artifacts/','outputs/')) and (p/n).is_file()}
r={'repo':str(p),'head':git('rev-parse','HEAD').decode().strip(),'status':git('status','--short','--branch').decode(),'log':git('log','-20','--oneline').decode(),'files':files,'tracked_files':names}
for name,args in [('sinfo',['sinfo','-o','%P %a %l %D %G']),('jobs',['squeue','--me','-o','%.22i %.30j %.10T %.10M %.30R'])]:
 r[name]=subprocess.run(args,capture_output=True,text=True).stdout
sys.stdout.buffer.write(gzip.compress(json.dumps(r).encode()))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','n23zhangWatGPU','python3','-'],input=code.encode(),capture_output=True,check=True,timeout=100)
(w/'repo-source.json.gz').write_bytes(r.stdout); d=json.loads(gzip.decompress(r.stdout))
for name,text in d['files'].items():
 p=w/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
print(d['status'],d['log'],d['sinfo'],d['jobs'])
print('\n'.join(n for n in d['files'] if n.startswith(('scripts/','configs/','tests/','docs/'))))
