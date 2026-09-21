from pathlib import Path
import json,subprocess
w=Path(__file__).resolve().parent;state=json.loads((w.parent/'deployment.json').read_text())
files={p.name:p.read_text() for p in w.iterdir() if p.suffix in ('.py','.md') and p.name!='deploy.py'}
script='from pathlib import Path\nimport json,subprocess,tempfile,hashlib\nroot=Path('+repr(state['root'])+')\n'
script+='assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=root/"code",text=True).strip()=='+repr(state['head'])+'\n'
script+='parent=root/"forensic_audit";parent.mkdir(exist_ok=True)\nstudy=Path(tempfile.mkdtemp(prefix="study_",dir=str(parent)))\n'
script+='files='+repr(files)+'\nfor name,body in files.items():(study/name).write_text(body)\n'
script+='prefix=(root/"training.sh").read_text().split("python -u")[0]+"export CCF_FORENSIC_ROOT="+str(study)+"\\nexport PYTHONPATH="+str(study)+":$PYTHONPATH\\n"\n'
script+='for label,names in [("conditional",["conditional.py"]),("generation",["generation.py","rescore.py"])]:\n (study/(label+".sh")).write_text(prefix+"\\n".join("python -u "+str(study/name) for name in names)+"\\n");subprocess.run(["bash","-n",str(study/(label+".sh"))],check=True)\n'
script+='jobs={};commands={}\nfor label,wall in [("conditional","01:00:00"),("generation","04:00:00")]:\n cmd=["sbatch","--parsable","--partition=ALL","--job-name=ccf-forensic-"+label,"--time="+wall,"--mem=40G","--cpus-per-task=4","--gres=gpu:1","--exclude=watgpu1008,watgpu1109,watgpu608,watgpu908","--mail-user=n23zhang@uwaterloo.ca","--mail-type=ALL,TIME_LIMIT","--output="+str(study/(label+"-%j.out")),"--error="+str(study/(label+"-%j.err"))]\n if label=="generation":cmd.append("--dependency=afterok:"+jobs["conditional"])\n cmd.append(str(study/(label+".sh")));jobs[label]=subprocess.check_output(cmd,text=True).strip();commands[label]=cmd\n'
script+='receipt=dict(root=str(study),jobs=jobs,commands=commands,source_head='+repr(state['head'])+',file_sha256={name:hashlib.sha256(body.encode()).hexdigest() for name,body in files.items()});(study/"deployment.json").write_text(json.dumps(receipt,indent=2)+"\\n");print(json.dumps(receipt,indent=2))\n'
r=subprocess.run(['ssh','-o','BatchMode=yes','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,check=True,timeout=55)
print(r.stdout);(w/'deployment.json').write_text(r.stdout)
