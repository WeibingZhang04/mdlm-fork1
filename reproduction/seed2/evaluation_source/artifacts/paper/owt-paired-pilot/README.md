# Frozen-backbone paired pilot

This is an exploratory three-seed diagnostic, not the frozen paper result.
It compares the independent and structured evaluation modes on the same
WikiText-103 validation corruptions (seeds 101--103), using 32 configured
validation batches of size 4 and length 1024.  EMA was disabled.

These historical runs did not record a cryptographic digest of the ordered
clean inputs, sampled times, and corruptions.  Equality of aggregate pairing
diagnostics is therefore only a consistency check, not proof of example-level
pairing.  The pilot is excluded from manuscript evidence; the frozen
confirmation protocol requires exact digest equality.

The independent-minus-structured conditional-denoising NLL improvement per
masked token is **0.003266** on average (per-seed values: 0.003309, 0.003060,
0.003429).  Candidate recall, retained unary mass, and active fraction match
exactly within every pair.  The deterministic 20,000-resample paired-seed
bootstrap (RNG seed 1701) gives a percentile 95% interval of
[0.003060, 0.003429].  With only three seed pairs, this interval describes
resampling of this pilot and should not be treated as submission-grade
uncertainty.

[`pilot.json`](pilot.json) is the machine-readable source of truth.  The
evaluation and released-backbone checkpoints existed only on the experiment
disk when this artifact was assembled, so their cloud paths are recorded as
protocol metadata and no checkpoint hash is claimed.  This artifact reports
conditional-denoising NLL only; it does not compute or infer an ELBO or
perplexity.
