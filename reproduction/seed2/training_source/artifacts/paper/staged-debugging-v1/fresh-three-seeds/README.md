# Fresh-corruption study: three completed seeds

All three paired runs finished 1,000 updates. Each compares shared factors,
separate endpoint factors, and a unary adapter under the unchanged training
protocol. They use the same 2,048 training, 64 development, and 128 test source
documents. Test corruptions vary by seed and remain paired across heads.
Replication was chosen after observing the seed-1 test. This is replication
on the same reserved documents, not an independent second test corpus.

## Results

Separate factors gain 0.004018 nats per masked token over their own marginal
product (95% CI [0.002726, 0.005280]), and 0.003175 over shared factors
([0.002275, 0.004115]). Their mean gain over the trained unary adapter is
0.001386, with an interval spanning zero ([-0.000897, 0.003541]).

At 90% masking, the gain over unary is 0.009480 [0.006374, 0.012388]. Unary
wins at 25% and 50%; the 75% comparison is inconclusive. All rates and
contrasts appear in `test/results.json`, without selecting positive conditions.
The intervals use 5,000 crossed seed/source-document bootstrap draws, seed
1701. They are pointwise, not simultaneous intervals. Conditional prediction
and generation quality remain separate experiments.

## Authentication and reproduction

Seed 1's original artifacts remain in `../fresh-pilot`. The original seed-2
and seed-3 records, seals, and compact training reports are in `seed2` and
`seed3`. Both pair heads select update 900 for seed 1; all other selections
are update 1000. Each original seal predates its test scoring.

The combined seal in `selection` recomputes choices from all 25,344 complete
development records. Its test export combines 4,608 original test records:
three seeds, three heads, 128 documents, four mask rates. Each record retains
its score, masks, candidates, document, checkpoint, and explicit original-file
lineage. Only its selection binding changes. The aggregator reproduces each
original shard's statistics exactly before computing the combined intervals;
no backbone call or test rescoring occurs. `test/lineage.json` records this.

An independent implementation of the crossed bootstrap matches all 240
reported estimates and interval endpoints within 7.2e-15. Large checkpoints,
raw source documents, and encoder caches are not Git objects. All 21 files in
each final training ledger, including every checkpoint, were verified locally
against the ledger authenticated by its original selection seal.

From the full local backup root, reproduce with:

```bash
python scripts/reproduce_staged_fresh_three_seeds_v1.py \
  --experiment-root /path/to/staged-debugging-v1 \
  --output-dir /path/to/new-output-directory
```

The recipe pins all three original selection-manifest and test-result file
hashes, verifies the training backups, seals the combined selection, aggregates
the original scores, and plots every development curve and test rate. It
refuses to overwrite an existing output directory. New timestamps and local
paths change manifest file hashes, not scores or the canonical selection.

Independently check the reported arithmetic, using only NumPy and the Python
standard library (no production statistics imports):

```bash
python scripts/audit_staged_fresh_three_seeds_v1.py \
  --experiment-root /path/to/staged-debugging-v1 \
  --combined-results /path/to/new-output-directory/test/results.json
```

This second check pins the original score files and validates all 240 estimates
and interval endpoints. It complements, rather than replaces, the recipe's
checkpoint and development-grid authentication.

Completed aggregate result file SHA256:
`4aefe030903e42746c79ae9baa67e04758d5bb3bb7247bd7fd67fe88e049be7f`.
Canonical combined selection:
`c0f1525b1722c3f02bb0b8be5c4cce744f07c3ed5d0009e6f5ef660db0f02107`.

## Recovery

The training implementation stayed at `a8d202a` throughout. Seed 2 resumed its
saved update-500 optimizer checkpoint on a recovery L4 worker; seed 3 trained
from initialization on the original worker. Both finished their final
development evaluation and reserved-test scoring before the fixed GPU
shutdown deadline of 2026-09-08 04:49:24 UTC. Both GPU workers are stopped.

Large transfers interrupted during shutdown were recovered using read-only
mounts on bounded CPU-only workers. The original experiment disk is preserved.
Seed-3 backup components, together with the immutable prefix through update
700, reconstruct the complete final training ledger:

- prefix through 700: `57a22dd238187e9367eb2d802166fe0a459d2f0a92bbcf649fdf4f8d772b8209`
- selected checkpoint 1000: `fcc6e939e3da873074eb51966e332e24347303b687ba9f4a44c8178987bff27b`
- checkpoints 800/900 and final head exports: `952c65040a9f4db0f88b333421052f5a0eb7ccb8c8386129b4a250d1967d62ba`
- original selection and test: `d09214ffe850d795e4054eab7a02d5aca083c11074e55de929dd13b8c88cca62`

Lightweight final training files were recovered from the interrupted delta
archive and individually verified against the final ledger and its original
selection manifest. The incomplete archive itself is not a valid backup.
Seed-2 full training and test backups have SHA256
`2e84c0d3214c795c1ec1415c8c1dea534df3741f97e1bba2b61bf6608fd8e8da`
and `8a44fb54d724f724ca2559b2526bde433d951a1242f808533c35c81ce3841966`.
