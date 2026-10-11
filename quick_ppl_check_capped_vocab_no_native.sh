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

vocab_cap="${VOCAB_CAP:-500}"
if [[ "$vocab_cap" != none && ! "$vocab_cap" =~ ^[1-9][0-9]*$ ]]; then
  echo 'VOCAB_CAP must be none or a positive integer' >&2
  exit 2
fi
length=1024
steps=32
samples=5
batch_size=1
warmup=1

out="$base/runs/sampling-capped-no-native-$(date -u +%Y-%m-%d_%H-%M-%S_UTC)_$$"
mkdir "$out"
echo "Results: $out"

common=(
  --backbone-checkpoint "$base/checkpoints/mdlm-owt.pt"
  --cache-dir "$HF_HOME/hub"
  --device cuda
  --length "$length"
  --steps "$steps"
  --sampler ddpm_cache
  --noise-removal
  --sampling-eps 1e-5
  --temperature 1
  --samples "$samples"
  --batch-size "$batch_size"
  --k 500
  --vocab-cap "$vocab_cap"
  --sampling joint
  --inference dense
  --warmup "$warmup"
  --sample-offset 200
  --score-gpt2
)
SECONDS=0
time0=$SECONDS
echo "Timing each method: model loading, warmup, generation, and GPT-2-large scoring included."

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode backbone --output "$out/vanilla"

time_baseline=$SECONDS
echo "Custom-evaluator baseline elapsed: $((time_baseline - time0)) seconds"

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode count --counts "$counts" --count-mode pmi --strength 0.1 \
  --output "$out/bigram"

time_count=$SECONDS
echo "Bigram count elapsed: $((time_count - time_baseline)) seconds"

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode global --head "$heads/global-step-30000.pt" \
  --output "$out/global"

time_global=$SECONDS
echo "Global CRF elapsed: $((time_global - time_count)) seconds"

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode contextual --head "$heads/contextual-step-30000.pt" \
  --output "$out/contextual"

time_contextual=$SECONDS
echo "Contextual CRF elapsed: $((time_contextual - time_global)) seconds"
echo "Total for all four evaluations: $((time_contextual - time0)) seconds"

python - "$out" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\nSampling results (after models load through last sample written; includes synchronization, decoding and sample I/O; excludes loading, warmup and PPL scoring):")
print(f'{"Method":<14} {"Post-load s/sample":>18} {"GPT-2-large PPL":>17} {"Calls/batch":>12}')
for name in ("vanilla", "bigram", "global", "contextual"):
    metrics = json.loads((root/name/"metrics.json").read_text())
    score = json.loads((root/name/"gpt2-large.json").read_text())
    calls = metrics.get("backbone_calls_per_batch", metrics.get("backbone_calls_per_sample"))
    print(f'{name:<14} {metrics["post_load_seconds_per_sample"]:>18.4f} '
          f'{score["perplexity"]:>17.2f} {calls:>12.1f}')
print("All four methods use DDPM-cache with final noise removal; vanilla is the custom evaluator baseline.")
print("Generation-loop-only timing is also saved in metrics.json as seconds_per_sample.")
print(f"\nResults: {root}")
PY_SUMMARY
