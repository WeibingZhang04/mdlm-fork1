# Tensor-Train OpenWebText feasibility baseline

This harness evaluates the official Tensor-Train release without modifying or
vendoring its source. It freezes the upstream Git revision, Hugging Face
checkpoint revision and bytes, MDLM/tokenizer revisions, GPT-2-large evaluator,
environment versions, random seed, and six matrix cells before execution.
All Hugging Face inputs are prefetched at exact revisions, hashed into the plan,
and reopened under offline/local-only mode during execution.

The feasibility matrix is unconditional OpenWebText-style generation with 256
samples of length 1024 for the released marginal and rank-4 Tensor-Train arms.
Each arm is evaluated at 8, 16, and 32 network evaluations (NFEs), with random
position ordering and temperature 1. The upstream `generation.k` means tokens
unmasked per network evaluation, so the wrapper maps NFE 8/16/32 to
`generation.k` 128/64/32 respectively.

Generation uses batch size 1. Besides matching the release's reported timing
scope, this avoids materializing the rank-4 `B x K x V x R x R` core tensor at
batch 32 on a 24 GiB L4. The wrapper resets every RNG after arm-dependent model
construction, records every selected-position chunk, and requires the exact
schedule hash to match between the marginal and rank-4 arms at each NFE.

## Prepare the immutable offline cache

Cache preparation is the only network-enabled phase. It downloads the pinned
backbone, tokenizer, and GPT-2-large evaluator revisions, then hashes every
cached file. Use a new cache-attestation output path if preparation fails.

```bash
python scripts/prepare_tensor_train_feasibility_cache.py \
  --cache-root /mnt/contextual-forest/third_party/tensor-train-hf-cache \
  --output /mnt/contextual-forest/third_party/tensor-train-hf-cache/identity-001.json
```

## Compile the plan

Use a dedicated Python 3.12 environment containing exactly the versions in
`configs/experiment/tensor-train-owt-feasibility-v1.yaml`. Keep the official
checkout and downloaded checkpoints on the persistent experiment disk. Both
the official checkout and this harness repository must be clean commits.

The audited upstream source imports `lightning` from its dataset utilities,
although the upstream environment file does not declare that package. The
frozen harness therefore pins Lightning 2.2.1 and its directly relevant
dependency closure (`pytorch-lightning`, `torchmetrics`,
`lightning-utilities`, and `fsspec`) as runtime-critical packages. A preflight
also pins `setuptools`, because this Lightning release imports
`pkg_resources`. It must reject an environment that omits or changes any of
those versions. The same upstream utility imports `timm`; the harness pins the
repository's established `timm` 0.9.16 dependency and its PyTorch-2.3.1-matched
`torchvision` 0.18.1 runtime rather than allowing a resolver to select a newer
incompatible Torch stack.

```bash
python scripts/compile_tensor_train_feasibility.py \
  --source-root /mnt/contextual-forest/third_party/tensor-train-9d0087a \
  --checkpoint-root /mnt/contextual-forest/third_party/tensor-train-9d0087a/checkpoints \
  --cache-root /mnt/contextual-forest/third_party/tensor-train-hf-cache \
  --artifact-root /mnt/contextual-forest/experiments/tensor-train-owt-feasibility-v1 \
  --output-dir /mnt/contextual-forest/experiments/tensor-train-owt-feasibility-v1/plan-001
```

Compilation verifies both checkpoint hashes before writing anything. It emits
one immutable JSON job specification per cell and a `compiled-plan.json` that
commits all six specifications, the exact model-input revisions, and the
byte-level offline-cache identity.

## Exact-environment compatibility preflight

Once the L4 is idle, run one non-paper sample before any paper job. This loads
the exact source/checkpoint/cache/runtime, exercises the official sampler, and
writes a fresh attestation. It acquires the submission-wide nonblocking GPU
lock at `/mnt/contextual-forest/experiments/.submission-gpu.lock` and the same
continuous foreign-PID monitor as paper jobs. The attestation also records
CUDA-synchronized generation time and peak allocated/reserved memory for this
single sample. Those resource fields are an operational projection aid, not a
paper result or a substitute for the full-cell measurements.

```bash
python scripts/preflight_tensor_train_feasibility.py \
  --plan /mnt/contextual-forest/experiments/tensor-train-owt-feasibility-v1/plan-001/compiled-plan.json \
  --job-id owt--tensor_train_rank4--nfe008--s260703 \
  --output /mnt/contextual-forest/experiments/tensor-train-owt-feasibility-v1/preflight-r4-nfe8-001.json
```

Do not run this preflight while another generation queue owns the GPU.

## Execute and verify

Run jobs sequentially: the runtime gate requires an otherwise idle visible GPU
so its timing and memory measurements are interpretable. A nonblocking lock
shared with the other submission workloads prevents two harnesses from racing,
while a one-second process monitor makes any foreign CUDA process fail the
attempt and preserve its partial directory.
The example below runs one cell.

```bash
python scripts/run_tensor_train_feasibility.py \
  --plan /mnt/contextual-forest/experiments/tensor-train-owt-feasibility-v1/plan-001/compiled-plan.json \
  --job-id owt--tensor_train_rank4--nfe008--s260703
```

`--resume` never resumes a partial computation. It only makes a completed run
idempotent after replaying every schema, sample, hash, metric, resource field,
and success commitment. A partial or corrupt directory fails closed and stays
untouched; retry using a fresh compiled plan/artifact root.

After all six jobs finish:

```bash
python scripts/verify_tensor_train_feasibility.py \
  --plan /mnt/contextual-forest/experiments/tensor-train-owt-feasibility-v1/plan-001/compiled-plan.json \
  --output /mnt/contextual-forest/experiments/tensor-train-owt-feasibility-v1/verified-matrix.json
```

The verifier requires every arm/NFE cell, identical evaluator runtime identity,
paired sample IDs, seeds, and selected-position schedules, exact output hashes,
semantic replay of the official Hydra configuration and learned state keys,
finite replayable GPT-2-large scores, and continuously monitored uncontended
resource records. Timing remains descriptive for the recorded hardware; it is
not a cross-hardware efficiency claim.

This six-cell artifact is a matched feasibility comparison between the two
released Tensor-Train baseline arms. It is not itself a matched comparison to
Contextual Coupling Forests; that requires a CCF run under the same OWT-1024,
NFE, schedule, evaluator, and hardware protocol.
