# Fast CCF evaluation and attribution record

September 16, 2026. Objective: reduce implementation overhead without changing
the model, sampling hyperparameters, precision, seeds, reveal schedule or scoring.
Speed parity or quality parity with MDLM has NOT been established.

## Verified entry point

Use `scripts/run_generation_fast.py` in place of `scripts/run_generation_pilot.py`,
with exactly the same arguments and a fresh output directory. The opt-in wrapper
enables `level_draws` for this process only; original defaults are unchanged.
It adds `sampler-provenance.json` with nine source hashes and verification scope.
Existing submitted snapshots must not be changed. New evaluations must use a
new snapshot and successful verification dependency. Unknown architectures and
different software/hardware still need suitable equivalence checks; the two
full samples are not proof for every possible input.

GPU array1544523_1 COMPLETED exit0, watgpu408. Paired full samples:

| Seed | Best v3 baseline (s) | Level batching (s) | Speedup |
| --- | ---: | ---: | ---: |
| 91001 | 135.8672 | 70.0462 | 1.940x |
| 91002 | 136.5980 | 69.6078 | 1.962x |

Both final token arrays, measured backbone-call counts, CPU and CUDA RNG states
matched exactly. Earlier 96 native draw checks, 54 pinned reference cases and
six L1024 checks passed. This does not compare every intermediate trajectory.
Loading/scoring excluded. Root-only batching1544523_0 also passed but took
124.58/125.43s, so it is not the chosen path. Additional tensor/buffer tests
1544526 remain independent and are not promoted before their final results.

`scripts/profile_ccf_level_gpu.sh` profiles the verified level path, replays it
uninstrumented with exact checks, then measures same-GPU factorized MDLM.
New timers separate grouped child conditionals and grouped deterministic draw
arithmetic. Noise generation/indexing remains in other ancestral traversal;
do not label that whole remainder as random-draw time. Host timings include
GPU waits; nested CUDA intervals must not be summed as disjoint kernel time.

## Intellectual and implementation provenance

- **Categorical/exponential race:** the unchecked helper intentionally adapts
  PyTorch 2.2.2's single-sample multinomial fast path (softmax, independent
  exponential noise, division, argmax). Rewriting it in Python does not make
  this algorithm our invention. Source:
  https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/native/Distributions.cpp#L599-L642
  License retained in `docs/third-party/pytorch-v2.2.2-LICENSE.txt`.
- **Tree message passing:** existing low-rank forest equations are evaluated
  using the standard sum-product principle. Foundational algorithm reference:
  Kschischang, Frey and Loeliger (2001), *Factor Graphs and the Sum-Product
  Algorithm*, https://doi.org/10.1109/18.910572 . This is conceptual attribution,
  not a claim that the low-rank implementation was copied from that paper.
- **Forest selection:** stable score ordering and union-find implement a
  constrained maximum-weight variant of Kruskal's greedy forest algorithm.
  Kruskal (1956), *On the shortest spanning subtree of a graph and the traveling
  salesman problem*, https://doi.org/10.1090/S0002-9939-1956-0078686-7 . Component
  caps are repository-specific; do not infer optimality under extra constraints
  from the ordinary minimum-spanning-tree guarantee.
- **Level batching, tensor gathers and noise buffers:** engineering changes
  were derived from this repository's equations and measured bottlenecks.
  No external implementation was used for those edits. This is a provenance
  statement, not a claim that batching or preallocation is novel.
- **DA-DLM:** consulted for timing comparison, not incorporated into this patch.
  Ji et al. (2026), https://arxiv.org/abs/2609.15070 . Do not describe our existing
  tree sampler as invented by DA-DLM or claim its small-block timing applies
  to our full-sequence implementation.

Retain inherited repository/dependency licenses. Any further borrowed algorithm
or implementation must be recorded even if rewritten with different syntax.
All changes are committed locally only, with a review patch; no Git push.

## Submission record

Implementation commit `8c069df5494cc37cbc3393630fdd03ef89a2c512`.
Review patch: `docs/patches/ccf-fast-generation-entry.patch`.
New immutable snapshot:
`/u401/n23zhang/mdlm_data/tree_mdlm_cache/code/ccf_level_profile.JHMjA4`.
Eight profiler/entry/sampler/harness hashes matched local copies before submission.
Profile job **1544546**, ordinary mdlm, 1 GPU, 4 CPUs, 30GB, 15min,
all notifications enabled. Unique output root `ccf_level_profile_job1544546`.
Slurm rejected an explicit afterok1544523_1 dependency with "Job dependency
problem" and created no job on that attempt. Before resubmission without that
unavailable dependency, accounting confirmed COMPLETED exit0 and a machine
check required report.status=passed plus two complete token/RNG/NFE matches.
Existing jobs and snapshots were not changed or cancelled.

Additional result: tensor_traversal1544526_0 passed, 69.2282/69.2404s versus
same-GPU factorized MDLM15.5489s. Simpler level_draws70.0462/69.6078s versus
same-GPU MDLM15.5785s. The small extra gain is not enough evidence for a robust
improvement, so the simpler level path remains the opt-in entry. The remaining
buffer tests were still running at submission. Exact outputs in these tests
preserve their quality; this is not evidence of quality parity with MDLM.
