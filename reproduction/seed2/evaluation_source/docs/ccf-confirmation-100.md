# CCF 100-sample confirmation

The 20-sample pilot was used only to select checkpoints. This confirmation generates 100 new sequences per configuration, using pair seeds 100001–100100, paired across models and sampling budgets. Do not pool the original 20 samples into the primary confirmation estimate.

Selection is frozen before confirmation results: take the three lowest token-weighted GPT-2-large PPL checkpoints for each of six FD/DD variants, separately at 8, 16, and 32 reverse transitions; ties favor earlier checkpoints. Include one fresh released MDLM factorized baseline per budget. Basic SS and DF are omitted because their pilot quality was generally weaker. Total: 57 configurations and 5,700 scored sequences.

The MDLM runner authenticates a bookkeeping SS/7k adapter but bypasses it in factorized mode; the baseline is the released backbone, not a fine-tuned 7k model.

| Model | 8-step checkpoints | 16-step checkpoints | 32-step checkpoints |
|---|---|---|---|
| Basic FD | 6k / 3k / 4k | 6k / 5k / 4k | 2k / 3k / 6k |
| Basic DD | 3k / 5k / 7k | 7k / 3k / 6k | 5k / 3k / 6k |
| Separate R8 FD | 6k / 3k / 1k | 5k / 1k / 6k | 6k / 3k / 2k |
| Separate R8 DD | 2k / 4k / 5k | 6k / 5k / 4k | 2k / 6k / 4k |
| Separate R16 FD | 6k / 5k / 2k | 5k / 6k / 1k | 3k / 6k / 5k |
| Separate R16 DD | 6k / 3k / 2k | 2k / 5k / 6k | 3k / 2k / 6k |

Preserve the pilot's length 1,024, batch size 1, K=128, trained parameters, model precision/internal autocast, reveal logic, optional final cleanup, and pinned GPT-2-large float32 first-nonleading-EOS scoring. Only sample count and base seed differ for each matched configuration. Optimized CCF uses the existing `level_draws` implementation, with no sampler changes. Attribution remains in `docs/ccf-fast-generation.md`. New code only selects and schedules existing evaluation functions; no external implementation was consulted or adapted for it.

A separate six-task GPU verification array checks one 32-step sample for each selected model variant. The 57-task production array depends on all verification tasks succeeding. Each production CCF configuration additionally retains the existing same-GPU first-sample gate for final tokens, model-call count, and CPU/CUDA RNG against pinned v1 (separate R8 DD) or v2 (others). Verification reference samples and prerequisite verification outputs are excluded from the 5,700 confirmation sequences. These gates compare final tokens/RNG/NFE, not every intermediate state.

Use a fresh copy of the successful eight-step snapshot; preserve all old snapshots, checkpoints, and outputs. Submit on the same H200 node with at most two simultaneous jobs, one GPU, four CPUs, 30 GB RAM, and 30 minutes per task. Use all existing email, array-task, and time-limit notifications.

The local selection tests pass: 57 configurations, exactly three checkpoints per variant/budget and one baseline, fresh seed arguments, unchanged non-seed/count arguments, duplicate/baseline/checkpoint rejection, and selection-hash tamper detection. Independently checked every selected row against the pilot ranking. Shell syntax and Python compilation passed.

Analysis after completion: recompute token-weighted PPL from all 100 sample records; report all selected configurations with paired seed bootstrap intervals versus the matched MDLM baseline. Compare confirmation scores/rankings with the 20-sample pilot. Report a checkpoint chosen on confirmation results as an exploratory minimum, not a separately confirmed winner. This tests generation-seed stability for existing trained checkpoints, not independent-training-seed stability.
