# Frozen pilot promotion policy

`configs/experiment/contextual-forest-expansion-v1-promotion-policy.yaml`
freezes the decision rule separately from the scientific experiment manifest.
It is bound to the pilot manifest SHA, compiled plan ID, pinned dataset
revisions, exact 12-cell condition grid, bootstrap configuration, and K=64.
Version 2 additionally pins the exact legacy `compiled-plan.json` SHA-256,
clean source commit `eaffd28a6667307d64f78fdd05c3ea7574f218d0`, and the
plan's complete job-spec commitment map. The analysis-v2 artifact additionally
commits every validated success marker and every output hash used by the
aggregate. Changing any input makes the evaluator reject the aggregate rather
than silently applying a different rule.

The policy separates two scientific questions.

1. **Does the K=64 model justify confirmation?** Confirmation requires a
   pooled improvement of at least 0.01 nats per masked token, a strictly
   positive 95% interval lower endpoint, positive mean improvement on at least
   three of four corpora, and the corresponding breadth and no-material-
   regression checks at mask rates 0.75 and 0.90.
2. **Is K=64 support-limited?** K=64 is sufficient only if all four frozen
   candidate-recall and retained-unary-mass thresholds pass. Missing any one
   threshold is a positive diagnostic indication to run the K=128 pilot, not a
   failed promotion.

The routes are independent. A quality-positive, support-limited pilot promotes
both the K=64 confirmation and K=128 diagnostic suites. A quality-negative,
support-limited pilot promotes only K=128. A quality-positive pilot with
sufficient K=64 support promotes confirmation only. Only a quality-negative
pilot with sufficient K=64 support stops.

## Evaluation

First produce the authoritative paired document-level aggregate:

```bash
python scripts/aggregate_hierarchical_document_eval.py \
  --plan-dir /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/plan-pilot-eaffd28 \
  --manifest configs/experiment/contextual-forest-expansion-v1.yaml \
  --suite pilot \
  --comparison contextual_vs_static \
  --output /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/pilot-analysis.json
```

Run that aggregation from the hardened checkout containing this policy. Its
legacy compatibility path accepts only the exact policy-pinned v1 plan file and
the clean `eaffd28` source identity recorded inside it; no other v1 plan is
grandfathered. It validates the legacy v1 success markers in place, rehashes all
of their outputs, applies the exact WikiText short-split rule, and writes an
analysis-v2 marker/output commitment. It never reruns or mutates pilot jobs.

Then apply the frozen policy and ask it to emit compiler-compatible evidence
only for true routes:

```bash
python scripts/evaluate_experiment_promotion.py \
  --analysis /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/pilot-analysis.json \
  --source-plan /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/plan-pilot-eaffd28/compiled-plan.json \
  --output /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/pilot-routing-decision.json \
  --compiler-evidence-dir /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/promotion-evidence
```

The evaluator refuses incomplete grids, extra conditions, wrong hashes or plan
IDs, drifted bootstrap settings, missing diagnostics, NaNs, infinities, and
support metrics outside `[0, 1]`. It never overwrites a decision or evidence
file. Invalid input emits no evidence. A valid decision returns exit status 0
when at least one route is true and 2 when neither route is true.

## Compiler handoff

Each emitted file uses the revision-bound version-2
`experiment_suite_promotion_decision` schema. For example:

```bash
python scripts/compile_experiment_matrix.py \
  --suite confirmation \
  --promotion-evidence confirmation \
    /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/promotion-evidence/confirmation-promotion.json

python scripts/compile_experiment_matrix.py \
  --suite candidate_k_128_pilot \
  --promotion-evidence candidate_k_128_pilot \
    /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/promotion-evidence/candidate_k_128_pilot-promotion.json
```

The compiler does not trust the evidence booleans. It selects this policy from
a code-owned registry, reloads the committed analysis, source plan, and routing
decision, revalidates every success marker and output hash, deterministically
recomputes the complete raw-record aggregate and bootstrap, and then
reevaluates the decision.
The evidence must then equal the canonical route artifact byte-for-meaning,
including the exact route name and criterion set. Handwritten all-true JSON,
extra criteria, invalid timestamps, a different plan, a dirty or different
source revision, and a changed decision are rejected.

The completed pilot remains usable even though its legacy plan ID omitted the
repository revision: only the exact plan file SHA
`67a11c102084f5987043e0cb1ddfcf03e3a0f53ffe5ed77f9b8468ae82b5ce22`
is grandfathered. Newly compiled version-2 plans include a clean repository
SHA in the plan identity and every job. Job success markers also bind that SHA,
so outputs cannot be silently reused under different experiment code.
