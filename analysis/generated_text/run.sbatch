#!/bin/bash
# DO NOT touch other people's files. DO NOT cancel other people's jobs.
#SBATCH --job-name=ccf-text-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ "$(id -un)" == n23zhang ]] || { echo 'Wrong account' >&2; exit 2; }
REPO=/u401/n23zhang/clean_tree_mdlm/mdlm-fork1
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false SLURM_EXPORT_ENV=ALL
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
MODE="${1:?Pass preflight, generate, analyze-basic, or analyze}"
MANIFEST="${2:?Pass prepared manifest path}"
case "$MODE" in
  preflight)
    srun --ntasks=1 python - "$MANIFEST" <<'CHECK'
# DO NOT touch other people's files. DO NOT cancel other people's jobs.
import os, sys
sys.path.insert(0, 'analysis/generated_text')
from analyze import account, verify, owned, prefix_score, REVISION
account(); m = verify(sys.argv[1])
os.environ['HF_HUB_CACHE'] = str(owned(m['cache']) / 'huggingface')
import torch, matplotlib, transformers
from transformers import AutoTokenizer
from evaluation.generation_metrics import TransformersReferenceLMScorer
assert torch.cuda.is_available()
AutoTokenizer.from_pretrained('openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', trust_remote_code=False)
scorer = TransformersReferenceLMScorer('gpt2-large', revision=REVISION, device='cuda', batch_size=1, max_length=1024, dtype='float32')
score = prefix_score(scorer, 'A short sentence for checking the scoring pipeline. ' * 80)
assert score is not None and score['token_count'] == 256
print({'preflight': 'passed', 'torch': torch.__version__, 'transformers': transformers.__version__, 'matplotlib': matplotlib.__version__, 'prefix_score': score}, flush=True)
CHECK
    ;;
  analyze-basic)
    # Existing mdlm environment only; no MAUVE, downloads, or package installs.
    srun --ntasks=1 python -u analysis/generated_text/analyze.py --manifest "$MANIFEST"
    srun --ntasks=1 python -u analysis/generated_text/blind_review.py create --manifest "$MANIFEST"
    srun --ntasks=1 python - "$MANIFEST" <<'PACK'
# DO NOT touch other people's files. DO NOT cancel other people's jobs.
import sys, zipfile
from pathlib import Path
sys.path.insert(0, 'analysis/generated_text')
from analyze import account, verify, owned
account(); m = verify(sys.argv[1]); root = owned(m['output'])
with zipfile.ZipFile(owned(root/'analysis-results.zip'), 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for p in [root/'manifest.json', *sorted((root/'analysis').iterdir())]:
        if p.is_file(): z.write(owned(p), arcname=str(p.relative_to(root)))
print('Non-MAUVE analysis and blind-review pack complete.', flush=True)
PACK
    ;;
  generate)
    srun --ntasks=1 python -u analysis/generated_text/generate.py generate \
      --manifest "$MANIFEST" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
  analyze)
    # Additional packages live in a separate environment, never installed in mdlm.
    PY="${CCF_ANALYSIS_PYTHON:?Set the separate analysis environment interpreter}"
    python - "$PY" <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, 'analysis/generated_text')
from analyze import owned
p = owned(sys.argv[1])
if not p.is_file() or not os.access(p, os.X_OK): raise SystemExit('Analysis Python is not executable')
PY
    srun --ntasks=1 "$PY" -u analysis/generated_text/analyze.py --manifest "$MANIFEST"
    srun --ntasks=1 "$PY" -u analysis/generated_text/blind_review.py create --manifest "$MANIFEST"
    srun --ntasks=1 "$PY" -u analysis/generated_text/mauve_eval.py --manifest "$MANIFEST" ;;
  *) echo 'Mode must be preflight, generate, analyze-basic, or analyze' >&2; exit 2 ;;
esac
