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
native_cap="$vocab_cap"
if [[ "$vocab_cap" == none ]]; then
  native_cap=null
elif [[ ! "$vocab_cap" =~ ^[1-9][0-9]*$ ]]; then
  echo 'VOCAB_CAP must be none or a positive integer' >&2
  exit 2
fi
length=1024
steps=32
samples=60
batch_size=1
warmup=1
if (( samples % batch_size != 0 )); then
  echo 'Native sample batches require samples divisible by batch_size' >&2
  exit 2
fi

out="$base/runs/sampling-capped-$(date -u +%Y-%m-%d_%H-%M-%S_UTC)_$$"
mkdir "$out"
echo "Results: $out"

common=(
  --backbone-checkpoint "$base/checkpoints/mdlm-owt.pt"
  --cache-dir "$HF_HOME/hub"
  --device cuda
  --length "$length"
  --steps "$steps"
  --samples "$samples"
  --batch-size "$batch_size"
  --k 64
  --vocab-cap "$vocab_cap"
  --sampling joint
  --inference segments
  --warmup "$warmup"
  --sample-offset 200
  --score-gpt2
)
SECONDS=0
time0=$SECONDS
echo "Timing each method: model loading, warmup, generation, and GPT-2-large scoring included."

# Use the native entry point and its existing offline HF cache/environment.
# The subshell keeps these settings out of the custom evaluator/scorer.
mkdir "$out/native_mdlm"
(
  conda activate "${NATIVE_ENV:-mdlm}"
  export HF_HOME="$base/original_reproduction/cache/mdlm/huggingface"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" TRANSFORMERS_CACHE="$HF_HUB_CACHE"
  export HF_MODULES_CACHE="$HF_HOME/modules" HF_DATASETS_CACHE="$HF_HOME/datasets"
  python -u main.py \
    mode=sample_eval backbone=hf_dit \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    data=openwebtext-split parameterization=subs seed=1 \
    model.length="$length" loader.eval_batch_size="$batch_size" \
    sampling.predictor=ddpm_cache sampling.steps="$steps" \
    sampling.num_sample_batches="$((samples / batch_size))" sampling.semi_ar=False \
    sampling.vocab_cap="$native_cap" sampling.warmup_batches="$warmup" \
    sampling.noise_removal=True \
    eval.sample_output_dir="$out/native_mdlm" \
    hydra.run.dir="$out/native_mdlm" checkpointing.save_dir="$out/native_mdlm" \
    > "$out/native_mdlm/run.log" 2>&1
)
# Score only: generation above never goes through evaluate_chain_crf.py.
python -u scripts/evaluate_chain_crf.py --score-only --device cuda --output "$out/native_mdlm"
time_native=$SECONDS
echo "Native MDLM elapsed: $((time_native - time0)) seconds"

python -u scripts/evaluate_chain_crf.py "${common[@]}" \
  --mode backbone --output "$out/vanilla"

time_baseline=$SECONDS
echo "Custom-evaluator baseline elapsed: $((time_baseline - time_native)) seconds"

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
echo "Total for all five evaluations: $((time_contextual - time0)) seconds"

python - "$out" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\nSampling results (all methods exclude model loading, warmup, decoding, output I/O, and PPL scoring):")
print(f'{"Method":<14} {"Seconds/sample":>15} {"GPT-2-large PPL":>17} {"Calls/batch":>12}')
for name in ("native_mdlm", "vanilla", "bigram", "global", "contextual"):
    metrics = json.loads((root/name/"metrics.json").read_text())
    score = json.loads((root/name/"gpt2-large.json").read_text())
    calls = metrics.get("backbone_calls_per_batch", metrics.get("backbone_calls_per_sample"))
    print(f'{name:<14} {metrics["seconds_per_sample"]:>15.4f} '
          f'{score["perplexity"]:>17.2f} {calls:>12.1f}')
print("Native MDLM uses DDPM-cache and final noise removal; vanilla is the custom evaluator baseline.")
print("Native timing includes final noise removal; custom timing includes its per-step synchronization/diagnostics.")
print(f"\nResults: {root}")
PY_SUMMARY
