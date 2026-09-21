import sys,json
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code))
from scripts import run_generation_pilot as pilot
from scripts.audit_ccf_sampling_v4 import experiment
args=json.loads(Path(sys.argv[2]).read_text())
with experiment('level_draws'):
    raise SystemExit(pilot.main(args))
