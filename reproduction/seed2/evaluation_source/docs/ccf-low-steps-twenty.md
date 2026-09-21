# Basic CCF and MDLM at 16/32 sampling steps

User requested both sampling budgets, all four basic CCF arms, the MDLM
baseline, and 20 samples per configuration. Use shared-rank16 constant-LR
training-update7000 checkpoints to match the completed five-sample comparison.
No retraining. The matrix has 5 models x 2 budgets x 20 samples = 200 scored
samples, plus 8 duplicate first-sample reference runs for CCF verification.

The existing pilot receives budgets17/33: 16/32 reverse transitions plus its
existing optional cleanup call. Actual measured NFE is retained. All models
use length1024, batch1, K128, seeds91001-91020 and identical pair keys/batch
seeds. Existing reveal logic, checkpoint precision, bf16 backbone autocast,
GPT2-large pinned revision and first-nonleading-EOS scoring are retained.
Multi-sample PPL is exp(token-weighted mean NLL), not mean per-sample PPL.

MDLM uses the factorized mode, bypassing the structured head; the authenticated
SS adapter is loaded only to satisfy the existing pilot's loading interface.
Each CCF cell first compares reference v2 with level_draws on the same GPU and
model for the requested full trajectory, checking final token IDs, NFE and
CPU/CUDA RNG states. Failure aborts that cell. The accepted fast first sample
is reused as sample1; it is not replaced or cherry-picked.

Entry scripts/evaluate_ccf_low_steps_twenty.sh wraps the existing evaluator.
Array0-4 evaluates16 steps; array5-9 evaluates32 steps. Model order is MDLM,
SS, FD, DF, DD. All use watgpu508/H200 for hardware consistency. The two
budgets and every attempt have distinct output directories. Source is copied
to a fresh immutable snapshot based on the successful selected-five snapshot;
existing jobs, source snapshots, checkpoints and results are not modified.
All email/time-limit/array notifications remain enabled as requested earlier.

Only orchestration and its metadata are generalized; no sampling algorithm
changes. Existing algorithm/code attribution: docs/ccf-fast-generation.md.
No new external implementation was consulted. Local Git record and review
patch only; no Git push.
