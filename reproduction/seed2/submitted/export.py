import sys
from pathlib import Path
from common import ARMS, STATE, save, sha

index, step = map(int, sys.argv[1:3])
checkpoint, folder = map(Path, sys.argv[3:5])
folder.mkdir(parents=True, exist_ok=False)
# Reuse the historical export helper; export runs after the training process exits.
sys.path.insert(0, STATE['eval_code'])
from scripts.export_structured_adapter import export_adapter
topology = 'fixed' if index < 2 else 'dynamic'
factor = 'fixed' if index in (0, 2) else 'dynamic'
report = export_adapter(checkpoint, folder / 'adapter.safetensors', folder / 'adapter.manifest.json',
    expected_checkpoint_sha256=sha(checkpoint), expected_global_step=step,
    control_identity=ARMS[index], topology_mode=topology, factor_mode=factor,
    candidate_k=128, independent_mode=False, topology_weight=0.0 if index < 2 else 0.1)
save(folder / 'export-report.json', report)
