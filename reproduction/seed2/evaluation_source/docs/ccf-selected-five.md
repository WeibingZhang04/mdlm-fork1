# Approved five-sample A/B/C/D comparison

User approved this exact checkpoint selection, including both dynamic-factor
arms for separate rank8/rank16. No retraining or sampling-step reduction.

| Array index | Family | Arm | Training updates |
| --- | --- | --- | --- |
| 0 | A: plain MDLM | factorized, head bypassed | released backbone |
| 1–4 | B: shared rank16, constant LR | SS, FD, DF, DD | 7000 |
| 5–6 | C: separate rank8 | FD | 2000, 6000 |
| 7–8 | C: separate rank8 | DD | 2000, 6000 |
| 9–10 | D: separate rank16 | FD | 2000, 4000 |
| 11–12 | D: separate rank16 | DD | 2000, 4000 |

Thirteen configurations, five samples each, 65 scored samples. The original
released backbone SHA remains7508daae475e7c0aa39dd7014e786fa9788fe1fc37f040c076e2df021e45f605.
Baseline A loads B's authenticated SS adapter only because the pilot requires
one; factorized mode bypasses CCF predictions. Shared rank16 and separate
rank8 are parameter-matched; separate rank16 is a larger model.

## Unchanged scientific settings

Length1024, batch1, K128, original1000 transitions plus reserved cleanup call
(budget1001), existing early termination, checkpoint precision and internal
bf16 autocast, reveal logic, original topology weights and affine FiLM.
Standard pilot num_samples5/base_seed91001: replicate-0000 through0004.
The exact same pair keys, seeds and batch seeds are used across all13 cells.
This is not the earlier repeated single-sample/replicate-0000 convention;
do not pool those historical samples or assume later seeds match them.
GPT2-large pinned revision32b71b12589c2f8d625668d2335a01cac3249519, float32,
batch1/maxlength1024, existing first-nonleading-EOS scoring. Report individual
scores/token counts and the pilot's token-weighted aggregate NLL/PPL.
Five generation seeds do not establish robustness across training seeds.

## First-sample verification gate and historical mismatch

Prior job1544727 failed to reproduce old R8DD sample91001 at2k/6k. The2k old
run was on watgpu108 RTX6000Ada; its new replay was on watgpu1208 H200NVL.
Torch2.2.2+cu121/Python3.9.25 and key backbone/head/harness source hashes match.
Hardware differs, but this alone is NOT proof of the mismatch's cause.

For each CCF cell, the new runner evaluates first sample91001 with the reference
and candidate on the SAME loaded model/GPU, full1000-step schedule. It compares
all final token IDs, NFE and CPU/CUDA RNG states. On success it retains that
candidate as sample1 and proceeds with four more; mismatch aborts the cell.
This is an in-process successful-GPU-verification dependency, before samples
are accepted, not a claim of bitwise equality across GPU architectures.
No extra synchronization/checking is inserted into the per-node hot path.

Most cells compare v2 against level_draws. The two R8DD cells instead load the
pinned originalv1 sampler (SHA61892523cb59ad6da842326df3fcf319989d5b9efb50da8d0b039aa99e0a15bf)
and also record whether each implementation reproduces the historical output.
An old cross-hardware mismatch does not silently discard a sample or substitute
a seed. If reference and fast match each other but not history, the fresh
five-sample result is kept with an explicit historical-replay warning.
Only final tokens/NFE/RNG are checked, not all intermediate states.

## Isolation, runtime and source provenance

Entry scripts/evaluate_ccf_selected_five.py and.sh. New immutable snapshot:
`/u401/n23zhang/mdlm_data/tree_mdlm_cache/code/ccf_selected_five.4aXFEQ`.
New output root:
`/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_selected_five.U0Y4t6`.
Per-cell, per-job unique attempt directories; all checkpoints/old outputs are
read-only. mdlm, Slurm1GPU/4CPU/30GB/45min; H200-feature nodes only to narrow
hardware variation. All mail/time-limit/array notifications enabled.
A/B queued first, C/D after that array ends; each cell has its own correctness
gate, so a failure in one model does not authorize skipping another's gate.

New orchestration derives from repository export/pilot/verification patterns.
Existing PyTorch exponential-race attribution/license and standard algorithm
citations are in docs/ccf-fast-generation.md; no new external implementation
was incorporated. Local commits only, no Git push, no changes/cancellations to
existing jobs or anyone else's processes.
