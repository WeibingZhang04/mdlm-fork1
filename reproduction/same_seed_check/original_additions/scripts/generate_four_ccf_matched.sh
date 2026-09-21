#!/bin/bash
#SBATCH --job-name=ccf-gen4
#SBATCH --time=24:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist="watgpu108"
#SBATCH --array=0-3%2
#SBATCH --output=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-gen4-%A_%a.out
#SBATCH --error=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf-gen4-%A_%a.err
#SBATCH --mail-user=n23zhang@uwaterloo.ca
#SBATCH --mail-type=ALL

# Generate matched unconditional samples for the four trained CCF adapters.
# The array runs at most two arms simultaneously.  Each arm gets one GPU.
#
# Default: ten 1024-token samples and 1,000 reverse transitions per arm.
# Override the sample count or output label at submission time, for example:
#   sbatch --export=ALL,NUM_SAMPLES=64,RUN_TAG=paper64 \
#     scripts/generate_four_ccf_matched.sh
# To match an existing 10,000-step baseline instead:
#   sbatch --export=ALL,SAMPLING_STEPS=10000,RUN_TAG=matched10k \
#     scripts/generate_four_ccf_matched.sh
#
# For a manual allocated-GPU run, provide an arm index from 0 through 3:
#   RUN_TAG=manual10 bash scripts/generate_four_ccf_matched.sh 0

set -euo pipefail

cd /u401/n23zhang/tree_mdlm/mdlm
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm

BACKBONE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/checkpoints/mdlm-owt-backbone.pt
HF_CACHE=/u401/n23zhang/mdlm_data/tree_mdlm_cache/huggingface
DEFAULT_ADAPTER_DIR=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/stale/four_arm_s001_k128_job1541029/exported_adapters
ADAPTER_DIR="${ADAPTER_DIR:-$DEFAULT_ADAPTER_DIR}"

NUM_SAMPLES="${NUM_SAMPLES:-10}"
SAMPLING_STEPS="${SAMPLING_STEPS:-1000}"
SEQUENCE_LENGTH=1024
BASE_SEED=91001
BATCH_SIZE=1
# run_generation_pilot counts the mandatory final cleanup call in its NFE
# budget.  Thus N reverse transitions correspond to an N+1 NFE budget.

if [[ -n "${RUN_TAG:-}" ]]; then
  JOB_TAG="$RUN_TAG"
elif [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_TAG="job${SLURM_ARRAY_JOB_ID}"
else
  JOB_TAG="manual-$(date -u +%Y%m%dT%H%M%SZ)"
fi
RUN_ROOT="/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/generation_four_ccf_l1024_t${SAMPLING_STEPS}_${JOB_TAG}"

IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ ! "$IDX" =~ ^[0-3]$ ]]; then
  echo "Arm index must be 0, 1, 2, or 3; found: ${IDX:-<empty>}"
  exit 2
fi
if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SAMPLES must be a positive integer; found: $NUM_SAMPLES"
  exit 2
fi
if [[ ! "$SAMPLING_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLING_STEPS must be a positive integer; found: $SAMPLING_STEPS"
  exit 2
fi
NFE_BUDGET=$((SAMPLING_STEPS + 1))

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

ARM="${ARMS[$IDX]}"
ADAPTER="$ADAPTER_DIR/$ARM.safetensors"
MANIFEST="$ADAPTER_DIR/$ARM.manifest.json"
OUTPUT_DIR="$RUN_ROOT/$ARM"

# Fresh training runs write adjacent hash commitments.  Keep compatibility
# with the original archived export, whose commitments predate those files.
if [[ -f "$ADAPTER.sha256" && -f "$MANIFEST.sha256" ]]; then
  ADAPTER_SHA256_VALUE=$(awk '{print $1}' "$ADAPTER.sha256")
  MANIFEST_SHA256_VALUE=$(awk '{print $1}' "$MANIFEST.sha256")
elif [[ "$ADAPTER_DIR" == "$DEFAULT_ADAPTER_DIR" ]]; then
  ADAPTER_SHA256_VALUE="${ADAPTER_SHA256[$IDX]}"
  MANIFEST_SHA256_VALUE="${MANIFEST_SHA256[$IDX]}"
else
  echo "Fresh adapter directory is missing hash files for $ARM: $ADAPTER_DIR"
  exit 2
fi

export HF_HUB_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$BACKBONE" ]]; then
  echo "Backbone not found: $BACKBONE"
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output already exists: $OUTPUT_DIR"
  exit 2
fi

printf '%s  %s\n' "$ADAPTER_SHA256_VALUE" "$ADAPTER" \
  | sha256sum --check --status
printf '%s  %s\n' "$MANIFEST_SHA256_VALUE" "$MANIFEST" \
  | sha256sum --check --status
BACKBONE_SHA256=$(sha256sum "$BACKBONE" | awk '{print $1}')

mkdir -p "$RUN_ROOT"

echo "Arm: $ARM"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES; length: $SEQUENCE_LENGTH; transitions: $SAMPLING_STEPS"

srun --ntasks=1 python -u scripts/run_generation_pilot.py \
  --backbone-checkpoint "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA256" \
  --adapter "$ADAPTER" \
  --adapter-sha256 "$ADAPTER_SHA256_VALUE" \
  --adapter-manifest "$MANIFEST" \
  --adapter-manifest-sha256 "$MANIFEST_SHA256_VALUE" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --base-seed "$BASE_SEED" \
  --modes structured_joint \
  --nfe-budgets "$NFE_BUDGET" \
  --device cuda \
  --model-config contextual-forest-small \
  --data-config train_openwebtext_pinned \
  --allow-dirty \
  --override "data.cache_dir=$HF_CACHE" \
  --override model.structured_decoder.top_k=128 \
  --override "model.structured_decoder.topology_mode=${TOPOLOGIES[$IDX]}" \
  --override "model.structured_decoder.factor_mode=${FACTORS[$IDX]}" \
  --override model.structured_decoder.independent_mode=false \
  --override "model.structured_decoder.training.topology_weight=${WEIGHTS[$IDX]}" \
  --override "checkpointing.save_dir=$RUN_ROOT"

echo "Completed $ARM"
echo "Samples: $OUTPUT_DIR/samples.jsonl"
echo "Summary: $OUTPUT_DIR/summary.json"
