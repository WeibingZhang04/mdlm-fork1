# Staged dependence debugging

The purpose is to establish what the joint layer can learn before scaling its
language experiments. No projected results are included.

## Completed mechanism checks

- `tiny/results.json`: 81 exact-distribution runs on AB/BA, AA/BB, and
  independent two-token targets. The original shared scorer cannot fit AB/BA
  with uniform unaries; the restriction is exact, not a failed seed.
- `tiny/rank2_results.json`: 18 runs of the separate-endpoint scorer at half
  rank. With the same 274 active parameters, initialization scale 0.25, and
  100 static warmup updates, it learns both dependent targets in all three
  seeds. No-warmup failures remain in the file.
- `tiny/figure_matched.pdf`: seed 1 fixed before inspecting examples. The
  rightmost panel samples independently from the exact same node marginals.

The proof assumes two tokens, uniform unaries, full vocabulary, and one edge.
It does not say that every shared-factor model has positive association in
every language context.

## Real-backbone checks

The committed launch sequence is `deploy/gcloud/run_staged_debug_v1.sh`.
It first audits serial versus batched backbone outputs, then fits three heads
to 32 fixed corruptions, and evaluates saved factorial checkpoints separately.
The overfit run uses the exact same cached serial hidden states, logits,
candidates, masks, and minibatch order in all arms. The released backbone
computes internally in BF16; storing the outputs in FP32 does not change that.

Shared rank 16 and separate-endpoint rank 8 each train 840,640 parameters.
The candidate-only unary adapter trains 827,312: comparable, not identical.
All preserve the backbone's residual conditional distribution. Therefore,
zero training loss is generally impossible when the target is outside top-K.
`scripts/report_staged_support_floor.py` measures the corresponding lower
bound. A loss plateau above zero is not by itself an optimizer failure.

The data preparation script selects 96 distinct source documents without
model scores: 32 debug-fit, 32 dev, and 32 test. Their source document IDs are
recorded for exclusion from future benchmark claims. The debug-fit subset is
deliberately trained on. This is a memorization diagnostic, not a benchmark.

## Completed real-text diagnostics

`real32/` contains the complete curves, protocol, support floors, and rendered
figure. The run finished all three arms at commit `bdb505b`, on one L4 with a
new snapshot-cloned disk. Final training NLL is 2.14565 (shared), 2.02730
(separate), and 1.19724 (unary), against the backbone's 2.92500. The unchanged
tail imposes a 1.11219 lower bound. All three heads worsen development NLL;
the separate head's training advantage is from better marginals, not a larger
dependence contribution. Pair heads use 100 static warmup updates within the
500 total; the unary head trains contextually throughout.

`backbone-batch-audit.json` records non-identical native-BF16 backbone results
across serial and batched evaluation. All fitting arms use one shared serial
cache. `data/manifest.json` records source-document-disjoint debug-fit/dev/test
selection, made without model scores.

`real32/candidate-identity-audit.json` verifies the actual CUDA candidate paths
at code commit `a7d141d`: full pair-head top-K and compact unary chunk top-K
return identical ordered IDs at all 4,096 masked training/development positions.
This includes 2,770 rows with cutoff ties and 15 tied gold targets. Candidate
membership and retained mass agree exactly; no tie policy was changed.

`checkpoints/` contains 256 observations per saved checkpoint: 32 development
and 32 test examples, each at four mask rates and length 128. These are old
seed-1 shared-factor checkpoints, not the new 32-example heads. The retry
script uses `ae731e4` after an initial uncached tokenizer alias failed before
evaluation. The source data and checkpoint commitments are in each summary;
record hashes verify the accompanying JSONL files.

Both checkpoints choose lambda=1 on development data. Test dependence gains
are +0.00607 and +0.00572 nats/token, but singleton gains relative to the
backbone are -0.01437 and -0.01733. Positive dependence does not offset the
weaker individual predictions. This new small, short-context diagnostic must
not replace or be pooled into the existing four-corpus benchmark.

Full diagnostic heads and logs are backed up in the workspace output folder
and retained on the new experiment disk, not added as large Git objects.

Cloud recovery locations: project `interactive-training-2026`, zone
`us-central1-a`, VM `diffusion-staged-debug-l4-20260907`, new disk
`diffusion-staged-debug-data-20260907`, experiment root
`/mnt/contextual-forest/staged-debug-20260907`. The exact original training
checkout remains detached at `bdb505b` in `code/`; later diagnostics use
separate checkouts. The initial failed tokenizer log is retained along with
the successful retry logs. Saved caches and head weights are under
`overfit32-v1/`; they are not benchmark-trained checkpoints.

## Direct dependence controls

`evaluation/dependence_diagnostics.py` separates the joint score into singleton
scores and observed edge-dependence terms. Marginal-preserving shrinkage uses
one development-selected lambda; test observations never choose it.

`evaluation/fixed_group_sampling.py` reveals exactly 1, 2, 4, 8, or 16 tokens
per call using the same precomputed reveal order in all sampling modes. It
records edges whose endpoints are revealed together, preserves context, and
avoids empty calls. This is a distribution/NFE control, not a wall-clock
benchmark: its callback may compute the structured head even in backbone mode.

These tools are separate from the original production decoder. Existing
factorial and generation artifacts remain results for the original scorer.
