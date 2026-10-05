#!/usr/bin/env bash
# Run from an allocated GPU session: bash run_inference_profile.sh
# One measured sample per method, one diffusion step, one excluded warmup.
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm-crf

base=/u401/n23zhang/rework-data
heads="$base/runs/draft-full-scale-training/preliminary-checkpoints_2026_10_01_14_29_45_UTC"
counts="$base/runs/draft-full-scale-training/train_2026_10_01_05_46_13_UTC/counts/owt-counts.pt"

export HF_HOME="$base/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1

python -c 'import torch; assert torch.cuda.is_available(), "Run inside an allocated GPU session"; print("GPU:", torch.cuda.get_device_name(0))'

run_timestamp=$(date -u +%Y-%m-%d_%H-%M-%S_UTC)
out="$base/runs/inference-profile-$run_timestamp"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
echo "Results: $out"

# sample-offset is a reproducible draw ID, not a token or dataset offset.
# This uses the evaluator's matched reveal sampler, not native ddpm_cache.
common=(
  --backbone-checkpoint "$base/checkpoints/mdlm-owt.pt"
  --cache-dir "$HF_HOME/hub"
  --device cuda
  --length 1024
  --steps 32
  --samples 1
  --batch-size 1
  --k 64
  --sampling joint
  --inference segments
  --warmup 1
  --sample-offset 92000000
)

echo "Running vanilla MDLM"
python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode backbone --output "$out/vanilla"

echo "Running count-based bigram"
python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode count --counts "$counts" --count-mode pmi --strength 0.1 \
  --output "$out/bigram"

echo "Running global CRF"
python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode global --head "$heads/global-step-30000.pt" \
  --output "$out/global"

echo "Running contextual CRF"
python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode contextual --head "$heads/contextual-step-30000.pt" \
  --output "$out/contextual"

echo "Profiling contextual CRF (timings include profiler overhead)"
python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode contextual --head "$heads/contextual-step-30000.pt" \
  --profile --output "$out/profile-contextual"

python - "$out" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
print("\nNormal inference timings (warmup excluded; single-sample smoke test):")
print(f'{"Method":<14} {"Total (s)":>12} {"Backbone (s)":>14} {"Sampling (s)":>14}')
for name in ("vanilla", "bigram", "global", "contextual"):
    metrics = json.loads((out / name / "metrics.json").read_text())
    print(f'{name:<14} {metrics["seconds_per_sample"]:>12.6f} '
          f'{metrics["backbone_seconds"]:>14.6f} {metrics["sampling_seconds"]:>14.6f}')
PY

echo "Finished. Results: $out"
echo "Terminal log: $out/run.log"
echo "Profile stages: $out/profile-contextual/profile-stages.json"
echo "Operation table: $out/profile-contextual/profile.txt"
echo "Full trace: $out/profile-contextual/profile.json"
