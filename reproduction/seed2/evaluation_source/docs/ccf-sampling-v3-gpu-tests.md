# CCF inference v3: isolated GPU experiments

September 16, 2026. User authorized further inference efficiency tests/queueing.
No production modules changed. Current generation snapshots remain immutable.

## Measured target

Full profile1544331 completed on NVIDIA H200 NVL (watgpu1208). Instrumented
CCF357.206s; same loaded backbone/GPU factorized MDLM12.525s. Loading9.591s
excluded from both. CCF host-exclusive phase costs (include CUDA waits):
categorical draws110.178s, upward arithmetic67.568s, other upward/root work
57.482s, conditional pair rows56.441s, other ancestral traversal30.856s,
forest selection14.790s, backbone11.760s. These disjoint times identify launch/
synchronization and serial inference as the main target, not the backbone.
Timing wrappers add overhead; this is not a zero-overhead ratio measurement.

## Three independently queued variants

0. `unchecked`: same native single-sample exponential-noise draw algorithm,
   omitting native per-row probability validation/synchronization.
1. `unchecked_batched`: also batch independent upward messages by depth and
   child count. Preserve child addition order and serial random draw order.
2. `unchecked_batched_roots_kruskal`: also batch root belief normalization and
   root child-message additions; use bulk CPU lists for stable Kruskal selection.

All modifications are scoped monkey patches in the experiment process. No
sampling steps, length, K, rank, precision, clamped-node draws, residual draws,
reveal logic or model checkpoint changes. Invalid categorical inputs would not
be rejected by the unchecked helper; it is not a general-purpose replacement.
Other validation remains enabled. The broader validation ablation is not in
this round: measured validation/metadata categories were small compared with
draws/messages. No claim of MDLM-equivalent speed or GPU parity yet.

## Tests and measurement

Entry scripts: `scripts/verify_ccf_sampling_v3_gpu.py` and `.sh`.
- 96 native categorical token/RNG cases, float32/float64, including -inf logits.
- 54 pinned pre-v1 token/RNG equivalence cases per candidate.
- Six L1024 token/RNG checks against production v2 (all/mixed/no active sites).
- Isolated v2/candidate timing: warmup plus five repeats, alternating order.
- Only after these pass: two full generations per implementation, seeds91001
  and91002, alternating v2/candidate order. Check all final token IDs, NFE, and
  CPU/CUDA RNG states, then one same-GPU factorized MDLM control. No per-node
  timing wrappers. Loading and GPT2 scoring excluded; no GPT2 evaluation needed
  to establish runtime/equality. First full sample may include warmup costs.
- Full per-update trajectories are not compared, only final tokens/NFE/RNG.
- Uses same separate-rank16 DD7000 adapter as the profile, L1024 K128 B1,
  1000 reverse transitions plus existing cleanup/early termination behavior.
- Report saved incrementally as report.json; failures are explicit and stop
  that candidate. Existing production jobs are not modified or cancelled.

Local checks include exact CPU candidate tests and a mock full-generation
runner test for order, output field, equality and patch restoration. These
do not substitute for server GPU results. Shell syntax/compile checked.

## Provenance and queue isolation

Experimental script extends local v3 audit a389080, rooted in current v2
production4fc49fb. Unchecked draw adapts the pinned PyTorch2.2.2 fast path:
https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/native/Distributions.cpp#L599-L642
License retained in docs/third-party/pytorch-v2.2.2-LICENSE.txt. Other changes
derive from this repository. No new external implementation consulted.

New snapshot /u401/n23zhang/mdlm_data/tree_mdlm_cache/code/ccf_v3_gpu_tests.fUA8aM
starts from verified immutable v2; only experiment/test files added or updated.
Each array task requests one Slurm GPU, 4CPUs, 30GB, 45minutes using mdlm,
all mail/time/array notifications, usual problematic/incompatible node exclusions.
Job/mode/unique-attempt output subdirectories prevent overwriting other runs.
Git commits local only; no push and no changes to other users' processes/files.
