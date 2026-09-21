"""Rescore fresh samples from the exact preserved seed-2 FD5k adapter on a GPU."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from prepare import prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path, help='A new directory; run inside a Slurm GPU allocation.')
options = parser.parse_args()
package = Path(__file__).resolve().parent
output = prepare(options.output)
shutil.copytree(package / 'fd5k', output / 'preserved-fd5k')
args = json.loads((package / 'fd5k/args.json').read_text())
for flag, value in {
    '--adapter': str(output / 'preserved-fd5k/adapter.safetensors'),
    '--adapter-manifest': str(output / 'preserved-fd5k/adapter.manifest.json'),
    '--output-dir': str(output / 'generation')
}.items(): args[args.index(flag) + 1] = value
args = [x if not x.startswith('checkpointing.save_dir=')
        else 'checkpointing.save_dir=' + str(output / 'generation') for x in args]
(output / 'args.json').write_text(json.dumps(args, indent=2) + '\n')
env = dict(os.environ, CCF_LEGACY_STUDY=str(output), CCF_CAMPAIGN_ROOT=str(output),
           PYTHONPATH=str(output), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
           HF_HUB_CACHE=str(Path(json.loads((package / 'manifest.json').read_text())['cache']) / 'huggingface'),
           TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')
subprocess.run([sys.executable, str(output / 'run_generation.py'), str(output / 'args.json'),
                str(output / 'cache-check.json')], env=env, cwd=output, check=True)
