from pathlib import Path
import json,subprocess
w=Path(__file__).resolve().parent;state=json.loads((w.parent/'deployment.json').read_text())
files={p.name:p.read_text() for p in w.iterdir() if p.suffix in ('.py','.md') and p.name!='deploy.py'}
script='from pathlib import Path\nimport json,subprocess,tempfile,hashlib\nroot=Path('+repr(state['root'])+')\n'
script+='assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=root/"code",text=True).strip()=='+repr(state['head'])+'\n'
script+='parent=root/"precision_causal";parent.mkdir(exist_ok=True)\nstudy=Path(tempfile.mkdtemp(prefix="study_",dir=str(parent)))\n'
script+='files='+repr(files)+'\nfor name,body in files.items():(study/name).write_text(body)\n'
script+='prefix=(root/"training.sh").read_text().split("python -u")[0]\n'
script+='prefix+="export CCF_PRECISION_STUDY="+str(study)+"\\nexport PYTHONPATH="+str(study)+":$PYTHONPATH\\n"\n'
script+='commands=["python -u "+str(study/"train_precision.py")+" "+p+" smoke" for p in ("bf16","fp32")]\ncommands.append("python -u "+str(study/"verify_pair.py"))\ncommands.extend("python -u "+str(study/"train_precision.py")+" "+p+" train" for p in ("bf16","fp32"))\ncommands.extend(["python -u "+str(study/"verify_pair.py"),"python -u "+str(study/"evaluate.py")])\n'
script+='(study/"run.sh").write_text(prefix+"\\n".join(commands)+"\\n")\n'
script+='subprocess.run(["bash","-n",str(study/"run.sh")],check=True)\n'
script+='cmd=["sbatch","--parsable","--partition=ALL","--job-name=ccf-cache-causal","--time=04:00:00","--mem=40G","--cpus-per-task=4","--gres=gpu:1","--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908","--mail-user=n23zhang@uwaterloo.ca","--mail-type=ALL,TIME_LIMIT","--output="+str(study/"job-%j.out"),"--error="+str(study/"job-%j.err"),str(study/"run.sh")]\n'
script+='job=subprocess.check_output(cmd,text=True).strip();receipt=dict(root=str(study),job=job,command=cmd,source_head='+repr(state['head'])+',file_sha256={name:hashlib.sha256(body.encode()).hexdigest() for name,body in files.items()});(study/"deployment.json").write_text(json.dumps(receipt,indent=2)+"\\n");print(json.dumps(receipt,indent=2))\n'
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,check=True,timeout=55)
print(r.stdout);(w/'deployment.json').write_text(r.stdout)
