#!/usr/bin/env bash
set -eo pipefail

cd /u401/n23zhang/crf-rework/mdlm-fork1

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm-crf

base=/u401/n23zhang/rework-data
heads="$base/runs/draft-full-scale-training/preliminary-checkpoints_2026_10_01_14_29_45_UTC"
counts="$base/runs/draft-full-scale-training/train_2026_10_01_05_46_13_UTC/counts/owt-counts.pt"

export HF_HOME="$base/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1

out="$base/runs/sampling-20-$(date -u +%Y-%m-%d_%H-%M-%S_UTC)"
mkdir "$out"
echo "Results: $out"

common=(
  --backbone-checkpoint "$base/checkpoints/mdlm-owt.pt"
  --cache-dir "$HF_HOME/hub"
  --device cuda
  --length 1024
  --steps 16
  --samples 20
  --batch-size 1
  --k 64
  --sampling joint
  --inference segments
  --warmup 1
  --sample-offset 93000000
  --score-gpt2
)

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode backbone --output "$out/vanilla"

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode count --counts "$counts" --count-mode pmi --strength 0.1 \
  --output "$out/bigram"

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode global --head "$heads/global-step-30000.pt" \
  --output "$out/global"

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode contextual --head "$heads/contextual-step-30000.pt" \
  --output "$out/contextual"

python - "$out" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\nSampling results (warmup and PPL scoring excluded from runtime):")
print(f'{"Method":<14} {"Seconds/sample":>15} {"GPT-2-large PPL":>17}')
for name in ("vanilla", "bigram", "global", "contextual"):
    metrics = json.loads((root/name/"metrics.json").read_text())
    score = json.loads((root/name/"gpt2-large.json").read_text())
    print(f'{name:<14} {metrics["seconds_per_sample"]:>15.4f} '
          f'{score["perplexity"]:>17.2f}')
print(f"\nResults: {root}")
PY_SUMMARY
