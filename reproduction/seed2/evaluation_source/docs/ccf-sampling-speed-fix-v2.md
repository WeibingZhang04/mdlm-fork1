# CCF sampling speed fix v2 — review record

September 16, 2026. User authorized both changes, then explicitly requested a
minimal implementation rather than additional defensive machinery. The initial
uncommitted deferred-validation prototype was discarded, not deployed.

## Production changes (one file, 15 lines added / 14 removed)

- `structured_utils.py::_single_low_rank_sum_product` has a sampling-only
  flag. Joint sampling skips downward messages and non-root marginals.
  The default still returns full marginals with the same arithmetic/order.
  Root beliefs and upward messages match the original bitwise in tests.
- `_sample_rows` removes duplicate probability validation with two `.item()`
  synchronizations per draw. `torch.multinomial` already rejects invalid
  weights. Softmax, multinomial arguments, draw order, number of draws,
  clamped draws, residual draws, precision, and all hyperparameters remain
  unchanged. This helper change also applies to dense/separable callers;
  their inference algorithms are otherwise unchanged.
- No extra validator, sanitizer, asynchronous assertion wrapper, new sampler,
  or duplicate inference routine. The production diff is net +1 line.

Invalid categorical inputs now raise PyTorch's native RuntimeError rather
than the removed custom ValueError. CUDA may report a device-side assertion;
invalid-input GPU tests therefore use isolated subprocesses. No guarantee is
made about recovering a CUDA context or RNG state after invalid input. Existing
upfront forest/factor/constraint validation remains unchanged.

## Review and reproducibility

- Base local commit: `fc750ad6a59a869e59c459498dae359db864551f`.
- Production patch: `docs/patches/ccf-sampling-speed-v2.patch`, relative to
  that base (v1 already applied). Tests/docs are separate from this patch.
- Exact reference implementations: pinned pre-v1 code from commit
  `99891f9441d9aef7082a088963ae2970bc992563`, plus v1 structured_utils SHA256
  `61892523cb59ad6da842326df3fcf319989d5b9efb50da8d0b039aa99e0a15bf`.
- `scripts/verify_ccf_sampling_v2.py` checks 54 pre-v1 cases and six L1024
  comparisons against v1, then benchmarks v1/v2 on the same device with
  alternating run order. Optional real-model check tests updates 0, 499, 899.
- `tests/test_ccf_sampling_optimization.py` adds root/upward equality,
  sampling-only routing, native categorical validity checks, CUDA scalar-read
  profiling, and impossible-constraint rejection.
- GPU job entry: `scripts/verify_ccf_sampling_v2_gpu.sh`; ordinary mdlm env,
  compatible Slurm allocation only, all notifications, new snapshot and unique
  job/attempt output root. No old experiment snapshot or other user's work is
  modified. Production generation must wait for successful v2 GPU verification.
- The older audit script is a historical prototype for commit fc750ad; use the
  new verifier for current code, not its old v1/prototype timing labels.

## Local results

Relevant suite before the final CUDA-only invalid-input test addition:
81 passed, 4 CUDA-only skips, 30 subtests passed. The final test adds another
CUDA-only skip locally. Exact pre-v1 checks: 54 passed; L1024 v1 comparisons:
six passed. All successful draws have identical tokens and RNG states.

CPU PyTorch 2.14.0, one thread, B2/L1024/K128/R16/synthetic vocab256, five
timed repetitions after warmup:

| Active state | v1 seconds | v2 seconds | Isolated speedup |
| --- | ---: | ---: | ---: |
| All | 0.257792 | 0.156406 | 1.648x |
| Mixed | 0.189100 | 0.125958 | 1.501x |
| None | 0.092667 | 0.071895 | 1.289x |

Fully clamped is diagnostic, not a usual production update. These timings
exclude backbone/head computation and do not establish GPU/end-to-end speed.
A full real-model trajectory comparison has not been performed.

## Server checks and requested follow-up sweeps

- GPU check 1544301 on watgpu108 failed CUDA initialization before any GPU
  equivalence test; four CUDA tests skipped, then the required CUDA benchmark
  correctly failed. No node configuration/process changes were made.
- Retry 1544302 on watgpu1208 ran all 14 tests. Token/RNG checks and native
  invalid-input rejection passed. One profiler test was too strict: it assumed
  zero scalar reads, but PyTorch 2.2.2's native multinomial retains two. The
  corrected test compares old/new and requires removal of our two duplicate
  reads, not removal of PyTorch's own checks. No production change was needed.
- The GPU retry uses a new snapshot; earlier snapshots remain immutable.
- User requested rank8 checkpoints1500/2500/5500/6500 and rank16 checkpoints
  2000/4000/6000, both dynamic-factor arms, one sample each. All 14 checkpoint
  files exist. The existing evaluation runner now accepts CCF_SWEEP_SET values
  rank8_followup/rank16_followup; generation seed91001, length1024, 1000 steps,
  K128 and GPT-2 scoring are unchanged. Outputs include rank/job/arm/step plus
  unique attempt directories. Jobs must depend on passing v2 verification.

## Provenance

Derived from this repository's code and local tests. No online implementation
was searched, copied, or adapted; no new external implementation citation.
This is not a guarantee that no similar code exists elsewhere. Commits remain
local only; deployment copies files, not Git pushes.
