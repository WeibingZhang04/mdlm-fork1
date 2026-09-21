# CCF sampling speed fix v1 — local audit record

## Scope and provenance

Implemented locally after the September 15/16, 2026 performance audit.
Original reference commit: `99891f9441d9aef7082a088963ae2970bc992563`.
The three pre-fix modules match the copies in the running experiment snapshot
`ccf_lrdecay_eval.Xuvdot`; their SHA256 values are pinned in
`scripts/verify_ccf_sampling_optimization.py` and checked before reference code
is executed. All Git commits for this work remain local; no remote Git push.
Deployment means copying files into a new isolated cluster directory, not
changing the remote working repository or old submitted snapshots.

These edits were authored from inspection of this repository's existing code
and the measured call paths. No online implementation was searched, copied,
or adapted for this patch; no new external code citation is required for the
edits themselves. This is a provenance statement, not a claim that no similar
code exists online. Existing repository/dependency licenses and attributions
continue to apply.

## Changes

1. `structured_objective.py`: separate backend/clamp preparation from marginal
   inference. Joint samplers already compute their own marginals/messages, so
   omit the earlier unused inference call when no inference object is supplied.
   Caller-supplied inference objects and explicit backend choices remain valid.
   Likelihood and marginal-sampling APIs still compute their required marginals.
2. `structured_utils.py`: record each edge's left endpoint in CPU topology
   metadata already transferred during validation. Serial low-rank inference
   and joint sampling use those integers instead of per-edge GPU `.item()`
   transfers. Arithmetic/traversal order and validation remain unchanged.
3. `evaluation/generation_harness.py`: determine once whether any original
   prompt evidence exists. Avoid repeated empty boolean gathers/equality checks
   for unconditional prompts; still check real evidence after every update.

Patch: `docs/patches/ccf-sampling-speed-v1.patch` (three production modules).
Tests, diagnostic scripts, this record, and `AGENTS.md` are committed alongside
it. Pre-existing architecture/experiment work is not swept into this commit.
The verification utility supports both the base head and the existing optional
separate-embedding/FiLM head in the current working tree.

## Unchanged contract

1000 reverse steps plus the existing cleanup call; existing early all-masks-
resolved behavior; model/checkpoint/adapter; length; K; rank; component cap;
sampling mode; reveal schedule; number of samples; per-sample seeds; batch size;
precision/autocast; all categorical draws, including clamped nodes and unused
residual draws; GPT-2-large model/revision/precision and first-nonleading-EOS
scoring policy. No backward passes added. No backbone caching across times.

Not attempted: batching categorical draws, skipping fixed-node draws,
active-only residual sampling, reordered/reassociated message arithmetic,
lower precision, approximate inference, or a different sampler.

## Validation

- Local relevant suite: 116 passed, 2 CUDA-only skips, 33 subtests passed.
- Pinned old/new comparisons: 54 exact token + final RNG equivalence cases in
  the current working tree (50 with the original head only). Includes all four
  arms, dense/low-rank, partial/all/no masks, multiple samples, residual support,
  separate embeddings/FiLM where available, reversed/branching/padded edges,
  hard constraints, caller-supplied inference, and default/explicit generators.
- A 1001-call harness replay checks tokens, NFE, and RNG; another test confirms
  conditional-evidence corruption is still rejected.
- A call-count test asserts one topology build and no redundant inference.
- GPU validation script: `scripts/verify_ccf_sampling_speed_gpu.sh`. It runs
  equivalence cases and an isolated sampler benchmark, plus old/new real-model
  DD update checks at indices 0, 499, and 899 of the unchanged 1000-step schedule
  on 1024-token synthetic mask states, using the existing LR-decay DD adapter.
- GPU checks are not a complete 1000-step trajectory replay. Report their
  outcome and measured timing honestly; do not extrapolate isolated timings to
  full generation speed without measuring an actual run.

Local reproduction:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_ccf_sampling_optimization.py
python scripts/verify_ccf_sampling_optimization.py --device cpu --benchmark
```

Inside an allocated compatible GPU node with `mdlm`:

```bash
python scripts/verify_ccf_sampling_optimization.py --device cuda --benchmark --real-step-check
```

Any future production evaluation using these edits must use a new job-specific
output root and source snapshot, and wait for GPU verification to succeed.
Existing LR-decay evaluation `1543754` and FiLM sweep `1543564` are not modified.
