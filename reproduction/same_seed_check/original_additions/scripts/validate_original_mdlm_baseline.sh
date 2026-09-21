#!/bin/bash
#SBATCH --job-name=mdlm-base-original
#SBATCH --time=02:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist="watgpu308"

#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm-base-original-%j.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm-base-original-%j.err
#SBATCH --mail-user="n23zhang@uwaterloo.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

source activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

BACKBONE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/checkpoints/mdlm-owt-backbone.pt
HF_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
ADAPTER_DIR=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/stale/four_arm_s001_k128_job1541029/exported_adapters
ADAPTER="$ADAPTER_DIR/static_static.safetensors"
MANIFEST="$ADAPTER_DIR/static_static.manifest.json"
ADAPTER_SHA256=26ea46f17db3873b3607dc1f026506f08ef148c3e9f61531e2b4e9eb7b58b4e1
MANIFEST_SHA256=39253ccbba30cd7cf7c456b4977920016cc22c07c9241396c6fcf50a54a62450

JOB_TAG="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/mdlm_original_protocol_baseline_${JOB_TAG}"
export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

test -f "$BACKBONE"
test -f "$ADAPTER"
test -f "$MANIFEST"
printf '%s  %s\n' "$ADAPTER_SHA256" "$ADAPTER" | sha256sum --check --status
printf '%s  %s\n' "$MANIFEST_SHA256" "$MANIFEST" | sha256sum --check --status
BACKBONE_SHA256=$(sha256sum "$BACKBONE" | awk '{print $1}')

# MDLM's released recipe uses 1,000 reverse transitions at length 1,024.
# The pilot counts the final noise-removal prediction as one NFE, hence 1,001.
# It uses ordinary DDPM because that same kernel is supported by every CCF arm;
# ddpm_cache is a speed optimization for the time-independent base model.
python -u scripts/run_generation_pilot.py \
  --backbone-checkpoint "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA256" \
  --adapter "$ADAPTER" \
  --adapter-sha256 "$ADAPTER_SHA256" \
  --adapter-manifest "$MANIFEST" \
  --adapter-manifest-sha256 "$MANIFEST_SHA256" \
  --output-dir "$RUN_ROOT" \
  --num-samples 8 \
  --sequence-length 1024 \
  --batch-size 1 \
  --base-seed 91001 \
  --modes factorized \
  --nfe-budgets 1001 \
  --device cuda \
  --model-config contextual-forest-small \
  --data-config train_openwebtext_pinned \
  --reference-lm gpt2-large \
  --reference-lm-revision 32b71b12589c2f8d625668d2335a01cac3249519 \
  --reference-lm-device cuda \
  --reference-lm-batch-size 1 \
  --reference-lm-max-length 1024 \
  --reference-lm-dtype float32 \
  --allow-dirty \
  --override "data.cache_dir=$HF_CACHE" \
  --override model.structured_decoder.top_k=128 \
  --override model.structured_decoder.topology_mode=fixed \
  --override model.structured_decoder.factor_mode=fixed \
  --override model.structured_decoder.training.topology_weight=0.0 \
  --override "checkpointing.save_dir=$RUN_ROOT"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['RUN_ROOT'])
summary = json.loads((root / 'summary.json').read_text())['groups'][0]
score = summary['reference_lm']
records = [
  json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()
]

print('\n=== ORIGINAL-PROTOCOL BASELINE SUMMARY ===')
print(f"GPT-2 mean NLL: {score['mean_nll_nats']:.6f}")
print(f"GPT-2 perplexity: {score['perplexity']:.3f}")
print(f"Measured NFE: {summary['measured_nfe_values']}")
print(f"Distinct-2: {summary['distinct_n']['2']:.6f}")
print(f"Repetition-4: {summary['mean_repetition_rate']['4']:.6f}")

preview = []
for record in records:
  preview.append(f"=== base_mdlm sample {record['sample_index']} ===")
  preview.append(record['text'])
  preview.append('')
preview_text = '\n'.join(preview)
(root / 'sample-preview.txt').write_text(preview_text)
print('\n' + preview_text)
print(f"Saved run: {root}")
PY