# 1k–7k CCF checkpoint sweep at 16/32 sampling steps

The user requested every 1,000-training-update checkpoint from 1,000 through
7,000, the four basic shared-rank-16 arms, and the separate-rank-8/rank-16
runs. Carry forward both 16/32 sampling settings and 20 samples per configuration.

| Family | Arms | Checkpoints | New evaluations |
|---|---|---|---:|
| Basic shared rank 16 | SS, FD, DF, DD | 1k–6k; reuse 7k | 48 |
| Separate rank 8 | FD, DD | 1k–7k | 28 |
| Separate rank 16 | FD, DD | 1k–7k | 28 |
| MDLM | Released baseline | Reuse both completed budgets | 0 |

104 new evaluations × 20 = 2,080 new scored samples. Reuse the 200 scored
samples from array 1545882 (MDLM and basic 7k at each budget), for 2,280 scored
samples across 114 configurations. MDLM is a single released baseline; it
does not have different CCF training checkpoints.

All 52 new checkpoint paths were found on WatGPU. Basic 1k comes from
`stale/four_arm_s001_k128_job1541029`; 2k/3k from continuation job 1542157;
4k–6k from continuation job 1542808. Separate models use their completed
fresh-training runs already identified in `SEPARATE_RUNS`. No retraining.

Unchanged settings: length 1,024, batch 1, K=128, original factor architecture,
seeds 91001–91020 with paired replicate keys, checkpoint precision and internal
bf16 autocast, existing reveal logic and optional cleanup call, pinned GPT-2-large
float32 scoring through the first nonleading EOS. Budgets 17/33 represent 16/32
reverse transitions plus optional cleanup; actual NFE is recorded.

Every new CCF cell must pass the existing first-sample same-GPU token/NFE/
CPU+CUDA RNG gate before accepting its 20 samples. Separate R8 DD retains the
pinned v1 reference; others use v2. The candidate remains `level_draws`.
Historical 1,000-step output replay is skipped for 16/32-step schedules because
it is not a comparable trajectory. Same-schedule correctness checks still run.
104 additional reference samples are excluded from quality aggregates.

Array 0–103 evaluates adjacent 16/32 pairs for each model/checkpoint. Checkpoint
order is ascending 1k–7k; within each, basic SS/FD/DF/DD (through 6k), then
R8 FD/DD and R16 FD/DD. Reuse basic 7k from the original completed array.
All jobs request watgpu508/H200, 1 GPU, 4 CPUs, 30 GB, and 30 minutes;
at most 2 run concurrently. All email/time-limit/array notifications are enabled.
Compute is obtained through Slurm only.

Use a new immutable snapshot copied from successful low-step array 1545882,
with only evaluation orchestration/tests/docs overlaid. Per-step, per-cell,
per-attempt output directories do not overlap. Save source hashes and exact
reused-output identities in the launch manifest. Keep a local Git record and
patch; do not push. No sampling algorithm edits or new external implementation;
established attribution remains in `docs/ccf-fast-generation.md`.
