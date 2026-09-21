# CCF v4: batch deterministic ancestral work, retain the RNG stream

September 16, 2026. User asked to continue inference optimization. Isolated
experiments only; no production model/sampler source or submitted snapshot
changed. New entry: scripts/verify_ccf_sampling_v4_gpu.sh. Candidate code:
scripts/audit_ccf_sampling_v4.py. Reuses the v3 GPU runner with an explicit
baseline option; historical v3 jobs use their immutable old runner.

## Reference and objective

Previous combined v3 array1544464_2 has now passed both full-generation
comparisons against v2. Seed91001:342.416→136.843s (2.502x); seed91002:
candidate138.465s (2.452x speedup). Identical final tokens, CPU/CUDA RNG and
NFE for both seeds. This is the new comparison baseline, not a production
deployment. GPU H200NVL. V4 targets serial conditional rows/draw arithmetic
that remains after v3 upward/root inference batching.

## Candidates

- `root_draws`: batch root softmax/division/argmax only; children retain serial
  conditional arithmetic. Diagnostic decomposition of the larger change.
- `level_draws`: also batch child conditional rows and softmax/division/argmax
  by tree depth and child count. Parents have been sampled before their children.
  This does not replace joint sampling with independent marginals or drop edges.

Each batch element still issues one exponential-noise draw per node, in the
original roots-first/nonroot-traversal order, with the identical shape/dtype,
including clamps. Noise is pre-generated separately, not drawn in one big
tensor. Only deterministic operations are batched. RNG calls elsewhere
(residual tokens, reveal masks) are untouched. Child message-addition order
and low-rank logsumexp formulas are preserved. Same topK, rank, precision,
1000 reverse steps plus existing cleanup/termination, B1 L1024, adapter/weights.
Remaining upfront validation remains; inherited v3 native categorical checking
omission stays experimental for known-valid distributions. No validation
ablation, cached backbone, lower precision, or new sampling hyperparameters.

## Local verification and preliminary timings

17tests passed,4GPU-only skips. Both variants pass54 pinned-reference
token/RNG cases, plus six L1024 comparisons against v2, including all/mixed/no
active masks. Grouped conditional rows match serial values bitwise on the
CPU fixture. Tests include B2, multiple samples, reversed/padded edges,
residual/explicit clamps and supplied state masks via the inherited suite.

CPU PyTorch2.14, one thread, synthetic B2L1024K128R16V256, five timed repeats
after warmup, alternating order; not a full-model or GPU speed claim:

| Active state | Best v3 (s) | Root draws (s) | Level draws (s) |
| --- | ---: | ---: | ---: |
| All | .110137 | .116050 | .081569 |
| Mixed | .082578 | .084062 | .067119 |
| None (diagnostic) | .038268 | .033295 | .033501 |

Full level batching1.35x/all,1.23x/mixed faster than best v3 on CPU. Root-only
slightly slower on active CPU fixtures; GPU launch costs differ, so it remains
a diagnostic comparison rather than an assumed win.

## GPU test protocol and isolation

New snapshot: /u401/n23zhang/mdlm_data/tree_mdlm_cache/code/ccf_v4_gpu_tests.XOPcdo.
Baseline `unchecked_batched_roots_kruskal`; each variant runs96 native-draw
checks,54 pinned pre-v1 checks, sixL1024 comparisons and alternating isolated
timings before loading the real model. Two full paired seeds91001/91002, same
rank16 DD7000 adapter as v3; alternate implementation order; compare every
final token, NFE and CPU/CUDA RNG states. Then same-GPU factorized MDLM control.
No full per-update trajectory claim; loading/reference scoring excluded.

Two array tasks, ordinary mdlm, one GPU4CPU30GB30min each, all mail/time/array
notifications and usual incompatible/problem-node exclusions. Outputs isolated
by job/variant/unique attempt. No current or other user's job modified/cancelled.
No production rollout. Results saved incrementally, mismatch fails that task.

## Provenance

New deterministic batching derives from existing repository sampling equations
and v3 prototype; no new external implementation consulted. Exponential-race
sampling inherits the documented PyTorch2.2.2 implementation attribution:
https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/native/Distributions.cpp#L599-L642
License preserved in docs/third-party/pytorch-v2.2.2-LICENSE.txt.
Changes and test records committed locally only; no Git push.
