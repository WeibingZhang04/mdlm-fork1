# Eight-step generation checkpoint sweep

The user requested an 8-step follow-up to the ongoing 16/32-step sweep.
Carry forward the complete checkpoint matrix and 20 samples per configuration.

| Family | Arms | Training checkpoints | Evaluations |
|---|---|---|---:|
| Released MDLM | Factorized baseline | Released backbone | 1 |
| Basic shared rank 16 | SS, FD, DF, DD | 1k, 2k, 3k, 4k, 5k, 6k, 7k | 28 |
| Separate rank 8 | FD, DD | 1k, 2k, 3k, 4k, 5k, 6k, 7k | 14 |
| Separate rank 16 | FD, DD | 1k, 2k, 3k, 4k, 5k, 6k, 7k | 14 |

All 57 evaluations are fresh: 1,140 scored samples. The 16/32-step baseline
results cannot be reused at 8 steps. MDLM retains the same frozen backbone;
the runner's SS/7k adapter input is authenticated but bypassed in factorized
mode. It is not a 7k-trained MDLM model.

Task 0 is MDLM; tasks 1–56 cover all CCF cells in ascending checkpoint order,
then basic SS/FD/DF/DD, separate R8 FD/DD, and separate R16 FD/DD.
The existing suite `low_steps_sweep` remains unchanged; the new
`full_checkpoint_sweep` includes the previously reused basic 7k cells.

Protocol: 8 reverse transitions plus the existing optional cleanup call
(requested NFE budget 9, actual model calls recorded), length 1,024, batch 1,
K=128, seeds 91001–91020, the same checkpoint precision/internal bf16 autocast,
reveal logic, and pinned GPT-2-large float32 first-nonleading-EOS scoring.
Generation runtime and quality are both recorded.

Each CCF cell must pass the existing same-GPU first-sample reference comparison
for exact final tokens, NFE, and CPU/CUDA RNG before accepting results.
Separate R8 DD uses pinned reference v1; other CCF cells use reference v2.
The candidate remains `level_draws`. Historical 1,000-step output replay is
inapplicable to this schedule. The 56 additional reference samples are not
included in quality aggregates. No sampler algorithm or scoring changes.

Copy the immutable snapshot used by array 1546050 to a fresh directory and
overlay only the orchestration, tests, documentation, and review patch.
Verify source hashes and all checkpoint paths before submission. Keep a
local Git commit and patch without pushing. Attribution remains in
`docs/ccf-fast-generation.md`; no new external implementation was consulted.

Submit array `0-56%2` with `afterany:1546050` so the eight-step follow-up starts
after the current sweep ends. This dependency controls scheduling only;
the eight-step experiment does not consume its results. Request watgpu508/H200,
1 GPU, 4 CPUs, 30 GB, and 30 minutes per task. Enable all configured email,
time-limit, and array-task notifications. Existing submitted jobs and snapshots
are not modified.
