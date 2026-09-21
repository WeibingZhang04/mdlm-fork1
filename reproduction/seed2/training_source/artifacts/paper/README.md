# Paper result artifacts

This directory contains the machine-readable records and compact checkpoints
behind the reported experiments.  It is evidence for the public code release;
an anonymous submission archive should omit Git metadata and author-specific
operational notes.

- `g1-heldout/` contains all five post-tuning confirmation seeds (9--13), all
  six ablations, exact enumeration records, training histories, the frozen run
  manifest, and the frozen gate result.
- `structural-oracle/` contains the raw per-condition JSON/CSV records, gate,
  hashes, and privacy-sanitized manifest for the finite-data table-fit oracle.
- `kernel-profile/` contains the CUDA low-rank versus pre-materialized-dense
  inference profile used in the paper.  Each backend is measured in a separate
  steady-state scope that retains only the inputs it requires.
- `owt-confirm/` contains the frozen-backbone OpenWebText/WikiText-103
  confirmation protocol, five cryptographically paired validation-corruption
  seeds for each comparison, paired-seed bootstrap intervals, and the separate
  topology-coverage diagnostic.  These records support conditional denoising
  NLL claims only.
- `topology-audit/` contains the complete 14,400-record descriptive topology
  replay summary and provenance. It does not identify a quality effect.
- `k256-pilot/` contains the clean-revision compiled plan and fully replayed
  one-seed, two-dataset K=256 conditional-denoising aggregate.
- `full-factorial/` contains the complete 600-job, four-arm K=128 factorial
  analysis, its fail-closed consolidation record, and the deterministic source
  selection audit used to resolve independently rerun partition copies.
- `tensor-train-feasibility/` contains the complete six-cell released-baseline
  feasibility matrix, its plan, and successful exact-stack preflight.
- `tensor-train-matched/` contains the complete 32-shard, 768-record CCF union
  and its fail-closed descriptive comparison with both released Tensor-Train
  arms at matched length, sample count, NFE budgets, evaluator, and GPU model.
- `released-backbone-preflight/` contains the strict raw-weight loading and
  length-1024 structured-forward compatibility check.  It is not a language
  quality or latency result.
- `owt-paired-pilot/` is an exploratory three-seed diagnostic.  Its historical
  runs predate cryptographic input/corruption digests, so it is deliberately
  excluded from submission evidence and from the manuscript's measured
  claims.

The real-text confirmation uses one final step-1000 checkpoint from one
training seed and five validation-corruption seeds.  It is not an independent
training-seed study, diffusion-ELBO estimate, perplexity result, or generation
evaluation.
