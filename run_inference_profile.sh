#!/usr/bin/env bash
# Run from an allocated GPU session: bash run_inference_profile.sh
# K=500, cap=500, 32 DDPM-cache steps; timings and separate one-batch traces.
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mdlm-crf}"

base=/u401/n23zhang/rework-data
heads="$base/runs/draft-full-scale-training/preliminary-checkpoints_2026_10_01_14_29_45_UTC"
counts="$base/runs/draft-full-scale-training/train_2026_10_01_05_46_13_UTC/counts/owt-counts.pt"

export HF_HOME="$base/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1

inference="${INFERENCE:-segments}"
samples="${SAMPLES:-10}"
inference_args=(--inference "$inference")
case "$inference" in
  segments) inference_args+=(--segment-batch-size 32) ;;
  dense) ;;
  *) echo 'INFERENCE must be dense or segments' >&2; exit 2 ;;
esac
if [[ ! "$samples" =~ ^[1-9][0-9]*$ ]]; then
  echo 'SAMPLES must be a positive integer' >&2
  exit 2
fi

python - <<'PY_CHECK'
import torch
from models.dit import FLASH_ATTN_AVAILABLE
assert torch.cuda.is_available(), "Run inside an allocated GPU session"
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("External FlashAttention:", FLASH_ATTN_AVAILABLE)
# Availability alone does not establish support for this GPU architecture.
assert (torch.ones(16, device="cuda") + 1).sum().item() == 32
torch.cuda.synchronize()
PY_CHECK

run_timestamp=$(date -u +%Y-%m-%d_%H-%M-%S_UTC)
out="$base/runs/inference-profile-k500-$inference-${run_timestamp}_$$"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
echo "Results: $out"

# sample-offset is a reproducible draw ID, not a token or dataset offset.
# Keep profiling and synchronized stage timers out of benchmark measurements.
common=(
  --backbone-checkpoint "$base/checkpoints/mdlm-owt.pt"
  --cache-dir "$HF_HOME/hub"
  --device cuda
  --length 1024
  --steps 32
  --batch-size 1
  --k 500
  --vocab-cap 500
  --sampler ddpm_cache
  --noise-removal
  --sampling-eps 1e-5
  --temperature 1
  --sampling joint
  "${inference_args[@]}"
  --warmup 1
  --sample-offset 92000000
)

for phase in bench profile; do
  for name in vanilla bigram global contextual; do
    extra=()
    case "$name" in
      vanilla) mode=backbone ;;
      bigram) mode=count; extra=(--counts "$counts" --count-mode pmi --strength 0.1) ;;
      *) mode="$name"; extra=(--head "$heads/$name-step-30000.pt") ;;
    esac
    run_args=(--samples "$samples" --output "$out/$name")
    if [[ "$phase" == profile ]]; then
      run_args=(--samples 1 --profile --output "$out/profile-$name")
    fi
    echo "Running $phase: $name ($inference)"
    python -u scripts/evaluate_chain_crf.py "${common[@]}" \
      --mode "$mode" "${extra[@]}" "${run_args[@]}"
  done
done

python - "$out" <<'PY'
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
print("\nUnprofiled generation timings (loading, warmup and scoring excluded):")
print(f'{"Method":<14} {"Mean s/sample":>14} {"Median s/sample":>16} {"Post-load s/sample":>19}')
for name in ("vanilla", "bigram", "global", "contextual"):
    metrics = json.loads((out / name / "metrics.json").read_text())
    records = [json.loads(line) for line in (out / name / "samples.jsonl").read_text().splitlines() if line.strip()]
    median = statistics.median(row["elapsed_seconds"] for row in records)
    print(f'{name:<14} {metrics["seconds_per_sample"]:>14.6f} '
          f'{median:>16.6f} {metrics["post_load_seconds_per_sample"]:>19.6f}')
print("Profiled runs include profiler overhead; use them for attribution, not latency.")
print("PPL scoring is not run by this profiling script.")
PY

echo "Finished. Results: $out"
echo "Terminal log: $out/run.log"
echo "Profile stages: $out/profile-*/profile-stages.json"
echo "Operation tables: $out/profile-*/profile.txt"
echo "Full traces: $out/profile-*/profile.json"
