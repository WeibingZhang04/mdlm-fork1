#!/bin/bash
# Run after confirmation: bash scripts/train_four_models.sh
# Four calls: CPU count fitting plus three separate head-training jobs.
# All tasks wait for data prep and step 3; MDLM remains frozen inside the trainer.
# Decisions and rationale: notes/mdlm-crf-1024-decisions.md, Step 4.
#SBATCH --job-name=chain-train-1024
#SBATCH --partition=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=watgpu108
#SBATCH --signal=B:TERM@300

set -euo pipefail
if (( $# != 0 )); then
  printf 'Usage: bash scripts/train_four_models.sh (submits jobs)\n' >&2
  exit 2
fi
repo=/u401/n23zhang/crf-rework/mdlm-fork1
base=/u401/n23zhang/rework-data
data_dir="$base/data/draft-full-scale-training/owt-1024-2026-10-01_050424_UTC"
validation="$base/runs/draft-full-scale-training/step3-owt-1024-2026-10-01_051416_UTC/results/validation.json"
checkpoint="$base/checkpoints/mdlm-owt.pt"
dependency=afterok:1579295:1579324

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_dir="${CHAIN_TRAIN_RUN_DIR:-$base/runs/draft-full-scale-training/train_$(date -u +%Y_%m_%d_%H_%M_%S_UTC)}"
  mkdir -p "$(dirname "$run_dir")"
  mkdir "$run_dir" # Refuse to reuse an existing run directory.
  cp "$0" "$run_dir/train_four_models.sh"
  submit_args=(--parsable --dependency="$dependency"
    --output="$run_dir/slurm_%A_%a.out" --error="$run_dir/slurm_%A_%a.err"
    --export=ALL,CHAIN_TRAIN_RUN_DIR="$run_dir")
  # Counts do not need a GPU. Learned heads may run two at a time.
  count_job="$(sbatch "${submit_args[@]}" --array=0 --time=12:00:00 "$run_dir/train_four_models.sh")"
  printf 'counts=%s\n' "$count_job" > "$run_dir/submitted-jobs.txt"
  head_job="$(sbatch "${submit_args[@]}" --array=1-3%2 --gres=gpu:1 "$run_dir/train_four_models.sh")"
  printf 'heads=%s\n' "$head_job" >> "$run_dir/submitted-jobs.txt"
  printf 'Counts: %s; heads: %s\nAll training outputs: %s\n' "$count_job" "$head_job" "$run_dir"
  exit 0
fi

: "${CHAIN_TRAIN_RUN_DIR:?Submit with bash scripts/train_four_models.sh}"
run_dir="$CHAIN_TRAIN_RUN_DIR"
case "$SLURM_ARRAY_TASK_ID" in 0|1|2|3) ;; *) exit 2 ;; esac
set +u
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm-crf
set -u
export REWORK_DATA="$base"
export HF_HOME="$base/hf-cache"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$repo"

# Require the validated data, frozen backbone, code and training shapes.
# Read the final manifest so the budget includes every prepared training row.
settings="$(python - "$data_dir" "$validation" "$repo" "$checkpoint" <<'PY'
import hashlib, json, sys
from pathlib import Path
data, report_path, repo, checkpoint = map(Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
report = json.loads(report_path.read_text())
assert report['status'] == 'passed', 'Step 3 has not passed'
assert sha(data / 'manifest.json') == report['data_manifest_sha256']
manifest = json.loads((data / 'manifest.json').read_text())
assert manifest['length'] == 1024
assert manifest['document_quotas'] == dict(train=200000, dev=2000, heldout_eval=100000)
for role in ('train', 'dev'):
    assert sha(data / f'{role}.pt') == manifest['files'][f'{role}.pt'], role
assert report['source_sha256'], 'Missing validated source identities'
for name, expected in report['source_sha256'].items():
    assert sha(repo / name) == expected, f'Code changed after step 3: {name}'
assert sha(checkpoint) == report['backbone']['loaded_file_sha256']
for mode, rank in (('global', 32), ('contextual', 32), ('independent', 64)):
    profile = report['training_profiles'][mode]
    assert profile['backbone_unchanged']
    for key, expected in dict(length=1024, k=64, batch_size=4, backbone_batch_size=1, rank=rank).items():
        assert profile['config'][key] == expected, (mode, key)
train, dev = manifest['splits']['train'], manifest['splits']['dev']
assert train['rows'] > 0 and dev['rows'] > 0
print((train['rows'] + 3) // 4, dev['rows'], train['packed_tokens'])
PY
)"
read -r train_steps dev_rows train_tokens <<< "$settings"
printf 'Task %s: target_steps=%s, dev_rows=%s, training_tokens=%s\n' \
  "$SLURM_ARRAY_TASK_ID" "$train_steps" "$dev_rows" "$train_tokens"

# One approximate pass through all training rows; all dev rows select best.pt.
common=(--data "$data_dir" --checkpoint "$checkpoint" --cache-dir "$HF_HOME/hub"
  --length 1024 --k 64 --batch-size 4 --backbone-batch-size 1
  --steps "$train_steps" --max-dev-examples "$dev_rows" --seed 1 --device cuda
  --learning-rate 3e-4 --weight-decay 0 --gradient-clip 1 --warmup-steps 1000
  --eval-every 5000 --save-every 500 --mlp-size 128 --threads 4 --max-seconds 82800)
if [[ "$SLURM_ARRAY_TASK_ID" != 0 ]]; then
  gpu_name="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
  [[ "$gpu_name" == 'NVIDIA RTX 6000 Ada Generation' ]] || { printf 'Unexpected GPU: %s\n' "$gpu_name" >&2; exit 1; }
fi

# exec passes Slurm's advance TERM warning to the checkpoint-aware trainer.
case "$SLURM_ARRAY_TASK_ID" in
  0) exec python -u scripts/build_chain_counts.py --data "$data_dir/train.pt" \
       --output "$run_dir/counts/owt-counts.pt" --max-tokens "$train_tokens" \
       --vocab-size 50258 --mask-id 50257 --smoothing .1 > "$run_dir/counts.log" 2>&1 ;;
  1) exec python -u scripts/train_chain_crf.py "${common[@]}" \
       --mode global --rank 32 --output "$run_dir/global" > "$run_dir/global.log" 2>&1 ;;
  2) exec python -u scripts/train_chain_crf.py "${common[@]}" \
       --mode contextual --rank 32 --output "$run_dir/contextual" > "$run_dir/contextual.log" 2>&1 ;;
  3) exec python -u scripts/train_chain_crf.py "${common[@]}" \
       --mode independent --rank 64 --output "$run_dir/independent" > "$run_dir/independent.log" 2>&1 ;;
esac
