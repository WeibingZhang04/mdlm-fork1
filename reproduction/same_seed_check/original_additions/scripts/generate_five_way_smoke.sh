#!/bin/bash
#SBATCH --job-name=ccf-gen5-smoke
#SBATCH --time=01:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist="watgpu308"

#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-gen5-smoke-%j.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-gen5-smoke-%j.err
#SBATCH --mail-user="n23zhang@uwaterloo.ca"
#SBATCH --mail-type=ALL


set -euo pipefail

source activate mdlm
cd /u401/n23zhang/tree_mdlm/mdlm

BACKBONE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/checkpoints/mdlm-owt-backbone.pt
HF_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
ADAPTER_DIR=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/stale/four_arm_s001_k128_job1541029/exported_adapters
export RUN_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/generation_five_way_s001_smoke_v1

export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

if [[ -e "$RUN_ROOT" ]]; then
  echo "Output already exists: $RUN_ROOT"
  exit 2
fi
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "Repository is dirty; continuing because this smoke test records dirty-tree provenance."
  git status --short
fi
if [[ ! -f "$BACKBONE" ]]; then
  echo "Backbone not found: $BACKBONE"
  exit 2
fi

ARMS=(static_static fixed_dynamic dynamic_fixed dynamic_dynamic)
TOPOLOGIES=(fixed fixed dynamic dynamic)
FACTORS=(fixed dynamic fixed dynamic)
WEIGHTS=(0.0 0.0 0.1 0.1)
ADAPTER_SHA256=(
  26ea46f17db3873b3607dc1f026506f08ef148c3e9f61531e2b4e9eb7b58b4e1
  8ec2b58ba12df60252b0a18a7d4ce4e473820c63d12d71f494d5408b5615b973
  0af102446019e229bed6251ca0ae4cbe19b30d84b5468183d5576945e3cc7ed6
  0f27587f822ce144aaf86f9d2a919a84b904097744f047189dc21a397c54b83a
)
MANIFEST_SHA256=(
  39253ccbba30cd7cf7c456b4977920016cc22c07c9241396c6fcf50a54a62450
  01fb5a61f76cfd93e617cc6e4d3248c5e292c9b50187abe71391ccf0cc2e3430
  5d438d66cb69c290a54a1d211075a38c6c1c7c029cfc2ffd51e9ab269f8a5eaf
  24a7c1981271e16e62fff8563a37528325cfb10922def72f596cd816a9e75f54
)

BACKBONE_SHA256=$(sha256sum "$BACKBONE" | awk '{print $1}')

for IDX in 0 1 2 3; do
  ARM="${ARMS[$IDX]}"
  ADAPTER="$ADAPTER_DIR/$ARM.safetensors"
  MANIFEST="$ADAPTER_DIR/$ARM.manifest.json"
  printf '%s  %s\n' "${ADAPTER_SHA256[$IDX]}" "$ADAPTER" | sha256sum --check --status
  printf '%s  %s\n' "${MANIFEST_SHA256[$IDX]}" "$MANIFEST" | sha256sum --check --status

  MODES=(structured_joint)
  if [[ "$ARM" == static_static ]]; then
    # The factorized path ignores the adapter and is the released MDLM control.
    MODES=(factorized structured_joint)
  fi

  srun python -u scripts/run_generation_pilot.py \
    --backbone-checkpoint "$BACKBONE" \
    --backbone-sha256 "$BACKBONE_SHA256" \
    --adapter "$ADAPTER" \
    --adapter-sha256 "${ADAPTER_SHA256[$IDX]}" \
    --adapter-manifest "$MANIFEST" \
    --adapter-manifest-sha256 "${MANIFEST_SHA256[$IDX]}" \
    --output-dir "$RUN_ROOT/$ARM" \
    --num-samples 8 \
    --sequence-length 256 \
    --batch-size 4 \
    --base-seed 91001 \
    --modes "${MODES[@]}" \
    --nfe-budgets 32 \
    --device cuda \
    --model-config contextual-forest-small \
    --data-config train_openwebtext_pinned \
    --reference-lm gpt2-large \
    --reference-lm-revision 32b71b12589c2f8d625668d2335a01cac3249519 \
    --reference-lm-device cuda \
    --reference-lm-batch-size 4 \
    --reference-lm-max-length 256 \
    --reference-lm-dtype float32 \
    --allow-dirty \
    --override "data.cache_dir=$HF_CACHE" \
    --override model.structured_decoder.top_k=128 \
    --override "model.structured_decoder.topology_mode=${TOPOLOGIES[$IDX]}" \
    --override "model.structured_decoder.factor_mode=${FACTORS[$IDX]}" \
    --override "model.structured_decoder.training.topology_weight=${WEIGHTS[$IDX]}" \
    --override "checkpointing.save_dir=$RUN_ROOT"
done

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['RUN_ROOT'])
table = [
  'condition\tmode\tGPT2_NLL\tGPT2_PPL\trepetition4\tdistinct2\tactive_tok_s'
]
preview = []
for arm in ('static_static', 'fixed_dynamic', 'dynamic_fixed', 'dynamic_dynamic'):
  payload = json.loads((root / arm / 'summary.json').read_text())
  for group in payload['groups']:
    mode = group['sampling_mode']
    condition = 'base_mdlm' if arm == 'static_static' and mode == 'factorized' else arm
    reference = group['reference_lm']
    table.append(
      f"{condition}\t{mode}\t{reference['mean_nll_nats']:.6f}\t"
      f"{reference['perplexity']:.3f}\t"
      f"{group['mean_repetition_rate']['4']:.6f}\t"
      f"{group['distinct_n']['2']:.6f}\t"
      f"{group['active_tokens_per_second']:.3f}")
  records = [
    json.loads(line) for line in (root / arm / 'samples.jsonl').read_text().splitlines()
  ]
  for mode in ('factorized', 'structured_joint'):
    condition = 'base_mdlm' if mode == 'factorized' else arm
    selected = [record for record in records if record['sampling_mode'] == mode][:2]
    if selected:
      preview.append(f'=== {condition} ({mode}) ===')
      preview.extend(record['text'].replace('\n', ' ') for record in selected)
      preview.append('')

table_text = '\n'.join(table) + '\n'
(root / 'averages.tsv').write_text(table_text)
(root / 'sample-preview.txt').write_text('\n'.join(preview))
print(table_text, end='')
print(f"Full samples: {root / 'static_static' / 'samples.jsonl'} and sibling arm directories")
print(f"Preview: {root / 'sample-preview.txt'}")
print(f"Averages: {root / 'averages.tsv'}")
PY
