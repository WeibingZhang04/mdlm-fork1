# Contextual-forest real-data bridge

This path is opt-in: ordinary `dit`, `crf_dit`, AR, and DiMamba configurations
do not construct or optimize the structured head.

## What it trains

For each real text batch, the existing absorbing forward process produces
`x_t`. The DIT produces hidden states and ordinary unary logits once, then the
contextual-forest head receives only `(hidden_states, unary_logits, sigma,
masked_positions)`. Candidate sets are therefore target-independent. Clean
tokens enter only the training likelihood after candidates and topology have
already been constructed.

The primary metric and objective are explicitly named **conditional denoising
NLL per masked token**. They are not presented as the original MDLM diffusion
ELBO. Generic NLL/BPD/perplexity accumulators are disabled for structured runs
so this conditional objective cannot be mistaken for a likelihood estimate.
Tail tokens retain full support through the exact residual state.
The configured `objective_name` is validated at startup and cannot be relabeled
as an ELBO accidentally.

Dynamic topology uses `gold_reveal_influence` distillation. During training
only, one masked source token per example is revealed to a detached backbone
teacher. The change in clean-token log probability at other masked positions
supervises anchor occupancy, source-to-anchor slot routing, and sparse edge
scores. At inference the teacher and clean sequence are absent; the hard forest
depends only on the corrupted context and time.

## Reproduced CUDA environment

The DIT contextual-forest path has deliberately small, tested CUDA 12.1 direct
dependency pins. It is
separate from `requirements.yaml`: the latter retains the upstream legacy
environment, including FlashAttention/Mamba/Triton pins that are not used by
this path and are incompatible with the tested PyTorch stack.

```bash
python -m venv /mnt/experiment-data/venv
/mnt/experiment-data/venv/bin/python -m pip install --upgrade pip
/mnt/experiment-data/venv/bin/python -m pip install \
  -r requirements-cloud-cu121.txt
/mnt/experiment-data/venv/bin/python -m pip check
```

## First executable checks

Run the focused exact-inference, bridge, streaming, release, and DIT fallback
tests:

```bash
python -m pytest -q \
  test_dit_sdpa_fallback.py \
  tests/test_prepare_released_mdlm_owt.py \
  tests/test_profile_forest.py \
  tests/test_neural_g1.py \
  tests/test_structured_training.py \
  tests/test_structured_objective.py \
  tests/test_structured_forest.py \
  tests/test_dataloader_streaming.py
```

Run a two-step random-backbone text8 plumbing smoke after installing the full
repository environment. The command assumes exactly one visible GPU; setting
`loader.num_workers=0` is supported (persistent workers are enabled only when
the worker count is positive):

```bash
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  model=contextual-forest-tiny \
  data=text8 \
  data.cache_dir=/mnt/experiment-data/data \
  trainer.accelerator=cuda \
  trainer.devices=1 \
  trainer.max_steps=2 \
  trainer.val_check_interval=1 \
  trainer.limit_val_batches=1 \
  trainer.num_sanity_val_steps=1 \
  loader.global_batch_size=2 \
  loader.eval_global_batch_size=2 \
  loader.batch_size=2 \
  loader.eval_batch_size=2 \
  loader.num_workers=0 \
  eval.generate_samples=false \
  training.ema=0 \
  checkpointing.resume_from_ckpt=false \
  wandb=null
```

This smoke only verifies wiring. It is not evidence for the paper.

## Immutable released backbone

The public `kuleshov-group/mdlm-owt` export is raw safetensors, not a Lightning
checkpoint and not an EMA export. Prepare it without executing remote code:

```bash
python scripts/prepare_released_mdlm_owt.py \
  --cache-dir /mnt/experiment-data/huggingface \
  --output /mnt/experiment-data/checkpoints/mdlm-owt-backbone.pt
```

The script pins revision
`d0958fa851335ece6c15260ce0025f030673c0fb`, requires
`model.safetensors` SHA256
`47149e73f7552f39ea9776dbe74d925d25237bcf2ed2e2ec03cdff9d51c82aa4`,
checks its 678,522,728-byte size and 131-key `backbone.*` schema, and writes a
local `state_dict` wrapper. The wrapper has explicit `ema_available=false` and
`ema_used=false` metadata and intentionally has no `ema` payload.

Run the strict full-length structured preflight before training:

```bash
python scripts/preflight_structured_backbone.py \
  --checkpoint /mnt/experiment-data/checkpoints/mdlm-owt-backbone.pt \
  --output /mnt/experiment-data/artifacts/released-backbone-preflight.json \
  --device cuda
```

This checks a strict load, frozen backbone, finite released logits, normalized
structured marginals, and records device, latency, and allocator peak. It is a
plumbing/correctness check, not a quality result.

## Frozen OpenWebText adapter screen

Use streaming OpenWebText for training and the finite WikiText-103 validation
split. For the released raw wrapper, disabling EMA selection is mandatory:

```bash
python main.py \
  model=contextual-forest-small \
  data=openwebtext-streaming \
  data.cache_dir=/mnt/experiment-data/data \
  model.structured_decoder.training.backbone_checkpoint=/mnt/experiment-data/checkpoints/mdlm-owt-backbone.pt \
  model.structured_decoder.training.use_ema_backbone=false \
  model.structured_decoder.training.strict_backbone_checkpoint=true \
  trainer.accelerator=cuda \
  trainer.devices=1 \
  trainer.max_steps=50 \
  trainer.val_check_interval=25 \
  trainer.limit_val_batches=4 \
  trainer.num_sanity_val_steps=2 \
  loader.global_batch_size=1 \
  loader.eval_global_batch_size=1 \
  loader.batch_size=1 \
  loader.eval_batch_size=1 \
  loader.num_workers=0 \
  training.ema=0 \
  eval.generate_samples=false \
  checkpointing.resume_from_ckpt=false \
  wandb=null
```

Fifty steps are only a bounded adapter screen. A paper result requires a
frozen step budget, fixed validation examples/corruptions, multiple
seeds, the factorized frozen-backbone baseline, and paired per-example output.

The small configuration refuses to start a fresh frozen-backbone run without
a checkpoint. To lightly tune the backbone rather than freeze it, override:

```bash
model.structured_decoder.training.backbone_mode=joint \
model.structured_decoder.training.backbone_lr_multiplier=0.05
```

If a different trusted Lightning checkpoint has no compatible EMA shadows,
startup fails with a shape/count diagnostic. Deliberately use raw checkpoint
weights only with
`model.structured_decoder.training.use_ema_backbone=false`; full-checkpoint
evaluation without EMA additionally requires `eval.disable_ema=true`.
EMA shadows in the upstream Lightning format are positional rather than named,
so EMA loading is restricted to trusted checkpoints produced by the same code
and parameter ordering. Shape/count checks cannot detect a permutation among
equal-shaped tensors.

The checkpoint callback monitors
`val/structured/conditional_nll_per_masked_token`; scheduler metadata uses the
same metric name even though the default constant-warmup scheduler does not
consume a monitored value. Its epoch value, candidate recall, and retained
unary mass are accumulated by active-token numerators and denominators rather
than averaging scalar batch means.

## Synthetic held-out result and inference profile

The learned G1 command defaults are the frozen reported protocol: development
on seeds 1--8, held-out seeds 9--13, 900 total steps, a 300-step shared-factor
warmup, factor initialization standard deviation 0.25 and initialization seed
1729. A fresh run is therefore:

```bash
python scripts/train_contextual_forest_g1.py \
  --output-dir /mnt/experiment-data/runs/g1-heldout
```

Its manifest records both the resolved training config and whether the run
exactly matches the reported held-out protocol.

Reproduce the optimized inference timing boundary with the named config's
three warmups and ten measured calls per backend:

```bash
python scripts/profile_forest_inference.py \
  --output /mnt/experiment-data/artifacts/forest-profile.json \
  --warmup 3 \
  --repetitions 10 \
  --device cuda
```

The timer synchronizes CUDA before and after each backend's repeated timing
block. The JSON records the exact warmup/repetition counts, timing and memory
scope, resolved shapes, rank, component cap, Git SHA/dirty state, GPU class,
driver, compute capability, device memory, PyTorch, CUDA, and cuDNN versions.
It deliberately omits the hostname, output path, and raw device identifiers so
the artifact can be shared without exposing private infrastructure. Factor
construction and forest construction are outside the timed inference call.
Peak memory is a steady-state backend comparison: the low-rank measurement
retains node inputs and endpoint factors, while the dense measurement retains
node inputs and one pre-materialized log-factor tensor. Raw dense construction
intermediates and inputs used only to construct that tensor are released before
dense warmup and measurement. Each row records the logical bytes of the inputs
that remain live for each backend.

## Joint sampling and ablations

Sampling is an explicit three-way ablation:

- `factorized`: use independent ordinary-backbone predictions;
- `structured_marginal`: run exact forest inference, then independently draw
  tokens from its exact node marginals;
- `structured_joint`: draw one exact joint forest assignment.

The latter two modes use the same trained head and checkpoint; switching modes
does not alter architecture or state-dict shape. They require the non-caching
DDPM sampler. For example:

```bash
sampling.predictor=ddpm \
model.structured_decoder.sampling.mode=structured_joint
```

Each step draws one joint clean assignment and then applies the ordinary
absorbing reveal probability, preserving correlations among tokens committed
in that step. `ddpm_cache`, analytic sampling, and semi-AR strides are rejected
for this opt-in path rather than silently falling back to independent tokens.

Controlled ablations use full Hydra override paths:

- `model.structured_decoder.independent_mode=true`: neutral pair factors with
  identical architecture parameter count. Pair parameters are inactive, so
  this is an architecture-count control, not an active-parameter-matched unary
  adapter;
- `model.structured_decoder.topology_mode=fixed`: natural-order chain unless
  fixed edges are supplied;
- `model.structured_decoder.fixed_edge_path=/path/edges.pt`: corpus-level
  static forest, induced on the currently masked positions;
- `model.structured_decoder.factor_mode=fixed`: context-independent factors;
- `model.structured_decoder.training.topology_weight=0`: no influence
  distillation.

The run logs structured loss, conditional NLL, candidate recall, retained unary
mass, active fraction, selected edges, factorized auxiliary NLL, all three
topology-loss components, valid teacher examples, and teacher influence. Report
candidate recall and topology-teacher coverage with every result. Edge, anchor,
and slot coverage each log an independently accumulated rate and eligible
example denominator; do not substitute their union coverage.

## Current boundary

The bridge supports continuous-time SUBS with a DIT backbone. It does not yet
derive a structured diffusion ELBO, optimize discrete topology end-to-end, use
cached joint DDPM updates, or support semi-AR sampling. The topology teacher is
a supervised training signal and should be labeled as such in experiments.
