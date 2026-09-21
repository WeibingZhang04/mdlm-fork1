import subprocess,json,sys
from pathlib import Path
w=Path(__file__).parent
state=json.loads((w/'deployment.json').read_text())
script='from pathlib import Path\nimport json,subprocess,os\nroot=Path('+repr(state['root'])+')\nstate=json.loads((root/"campaign.json").read_text())\n'+sys.stdin.read()
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','n23zhangWatGPU','python3','-'],input=script,text=True,capture_output=True,timeout=55)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr);sys.exit(r.returncode)
