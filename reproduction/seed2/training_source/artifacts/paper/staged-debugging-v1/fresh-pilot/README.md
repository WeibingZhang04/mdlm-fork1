# Fresh-corruption pilot: completed seed 1

The three heads finished 1,000 updates under the protocol in `training/protocol.json`.
The development-only seal checks all 8,448 observations: 11 checkpoints,
three heads, 64 documents and four mask rates. Both pair heads select update
900; the unary adapter selects update 1000. The selected test evaluation
contains 1,536 records on 128 reserved source documents and all four rates.

## Measured result

Separate factors learn useful held-out dependence: joint prediction improves
on their own marginal product by 0.002837 nats per masked token (95% paired
document-bootstrap CI [0.001925, 0.003746]). They also improve on shared
factors by 0.002538 [0.001666, 0.003406]. They do not outperform the trained
unary adapter overall: -0.000129 [-0.002618, 0.002322].

At 90% masking, separate factors improve on the unary adapter by 0.008257
[0.004239, 0.012153] and dependence contributes 0.006333 [0.004464, 0.008126].
The unary adapter wins at 25% and 50%; the 75% interval includes zero.
`test/results.json` reports every contrast and rate. These are conditional
prediction results for one training seed, not generation quality or
uncertainty across training seeds. The follow-up plan is in `replication-plan.md`.

`learning-and-test.pdf` shows the complete development curves and both primary
test contrasts. Its JSON sidecar binds the figure to the source files and
records the plotted numbers. The maximum test decomposition error is
4.846e-12 nats per sequence with FP64 inference.

## Recovery and provenance

Google Cloud preempted the Spot VM at 2026-09-08 02:33:37 UTC during the
update-800 development evaluation. The immutable parent checkpoint at update
800 has SHA256 `5eb5fc41d8b62770b09b8de3fac9fec681f43ef91eaefd603ecc9c357376a4a4`.
The same dedicated experiment disk was mounted after restart. The unchanged
training implementation at `a8d202a` restored model, optimizer, minibatch and
random states into a new run directory. Development evaluation 800 was
recomputed; 801--1000 continued normally. The original interrupted run is
preserved. A CUDA regression verifies exact resumed-versus-uninterrupted
weights, optimizer moments, losses, corruption hashes and random states.

`training/hashes.json` authenticates the completed run. Large head weights,
optimizer checkpoints and encoder caches are not Git objects. Full small-head
checkpoints, the interrupted parent and raw split JSONL files are backed up
locally and remain on the dedicated experiment disk. Backup archive SHA256:

- full training: `821f1d603b659cd92937c26989b977e66b5843ae9dfc3d4e3b4332a745c8c9f5`
- completed test: `26ae4ba9234a56b7cfbff6ed0971e94326f2879c07b302ea5606939091a2447a`

The original test selection is immutable. Its file SHA256 is
`f05a08df0de9bd81e7b782eae44bd2672f8290288137fe6de77b9fe6f6f68ad0`;
its internal canonical selection SHA256 is
`c0ffd54ac2360e8f2787d8b5cf03e605b63c9ccbdd70637c5f72e11a49b95662`.
`test/commitment.json` was written before reading reserved test tokens and
binds the selected checkpoints, source data, runtime and inference precision.
