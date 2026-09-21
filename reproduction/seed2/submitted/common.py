import hashlib
import json
import os
from pathlib import Path

STUDY = Path(os.environ['CCF_LEGACY_STUDY'])
STATE = json.loads((STUDY / 'study.json').read_text())
ARMS = ['static_static', 'fixed_dynamic', 'dynamic_fixed', 'dynamic_dynamic']
PHASES = [1000, 3000, 6000, 8000]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
