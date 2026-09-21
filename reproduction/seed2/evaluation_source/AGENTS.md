# Experiment submission preferences

- For all future Slurm experiment submissions, enable all email notifications
  to `n23zhang@uwaterloo.ca` unless the user explicitly requests otherwise.
- Use `--mail-type=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS`.
  Include the same options for ad hoc `sbatch` submissions and dependent scoring
  or summary jobs, not just stored scripts. The explicit extras cover time-limit
  alerts and individual array tasks, which Slurm's `ALL` alone does not include.
  Reference: https://slurm.schedmd.com/sbatch.html#OPT_mail-type
- This is a future-submission preference. Do not alter already submitted jobs
  or their isolated code snapshots merely to apply it retroactively.
- Never stop, modify, or interfere with other users' jobs, processes, or files.
  Obtain compute through Slurm allocations only.

# CCF sampling performance/reproducibility memory

- Read `docs/ccf-sampling-speed-fix.md` before further generation optimizations
  or launching evaluations from a new implementation snapshot.
- Preserve sampling steps, all hyperparameters, precision, per-seed RNG draw
  ordering, reveal logic, and GPT-2/EOS scoring policy. Equal distributions
  alone are not sufficient for claiming identical seeded results.
- Speed fix v1 removes unused joint-sampling pre-inference, reuses host edge
  orientation metadata, and hoists empty-evidence checks. It does not batch
  draws, omit clamped-node draws, or change trained parameters.
- Keep a local Git record and patch, exact token/RNG tests against the pinned
  pre-fix code, and isolated runtime/source provenance. Do not push commits.
- Existing submitted snapshots are immutable. Use a new snapshot and a
  successful GPU verification dependency for new optimized evaluation jobs.
- Record provenance of new code; identify/cite any external implementation
  actually consulted or adapted. Do not assert global originality merely
  because no external implementation was consulted.
- Speed fix v2: see `docs/ccf-sampling-speed-fix-v2.md`. Keep the implementation
  minimal: sampling-only inference skips unused downward messages/marginals;
  native multinomial validation replaces duplicate synchronous checks. Do not
  reintroduce deferred validators/sanitizers. Verify exact tokens/RNG and keep
  tests outside the production hot path. V1 GPU success does not verify v2.
