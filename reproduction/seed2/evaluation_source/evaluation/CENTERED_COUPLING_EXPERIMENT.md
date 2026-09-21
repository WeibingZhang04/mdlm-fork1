# Learn dependence without moving the unary probabilities

Status: standalone head, generation bridge, correctness tests, and three
old-32-example fitting runs are complete. Fresh-corruption training and
real-text generation comparisons have not yet been run for this head.

The completed three-seed fresh-training study finds positive dependence gain, but the
separate-factor head also worsens singleton scores relative to the trained
unary adapter. This experiment isolates those two effects by construction.

## Established foundations

This is a bounded Sarmanov-type edge model, not a new probability family.
[Sarmanov (1966), section 3, equation (16)](https://www.mathnet.ru/eng/dan32257)
constructs fixed-marginal distributions from bounded, centered endpoint
functions; section 4 permits mixtures with the same marginals. Interpreting
our feature average as an equal-weight mixture is a direct algebraic
identification. The tree extension follows the factorization in
[Chow and Liu (1968), section II](https://cs.nyu.edu/home/people/in_memoriam/roweis/csc2515-2006/readings/chowliu.pdf).
Our leaf-summing argument is its rewrite in terms of joint-to-independent
edge ratios. Any research claim must concern the conditional language-model
implementation and measured comparisons, not these established principles.

## Model

`FrozenUnaryCenteredForestHead` deep-copies and freezes a candidate-only unary
adapter. Let its normalized token probabilities be `b_i`. For each edge of
the natural-order chain of masked positions, learn separate endpoint features
`f_i` and `g_j` with zero mean under `b_i` and `b_j`. Residual-state features
are zero. The edge ratio is

    phi_ij(a,b) = 1 + eta * mean_r f_i(a,r) g_j(b,r).

Features are bounded by one and `0 < eta < 1`, so all ratios are positive.
The joint distribution is `q(x) = product_i b_i(x_i) product_ij phi_ij(x_i,x_j)`.
Each edge ratio integrates to one at either endpoint. Removing leaves proves
that the forest is normalized and has exactly the prescribed marginals.
Observed log scores need only the unary scores and observed edge ratios;
training does not need a partition-function computation.

The implementation exposes positive factors of rank `2*d+1` to the existing
inference and sampling backends. It preserves the selected-test evaluator's
FP32 compact unary lattice followed by FP64 normalization, and normalizes the
within-residual backbone conditional separately in FP64. Recomputing compact
residual mass would change the frozen reference and is not allowed.

At the default `d=8`, `eta=.95`, and GPT-2-sized vocabulary, the new coupling
has 827,312 trainable parameters, in addition to the frozen unary adapter's
827,312 parameters. The positive log-ratio per edge is bounded by
`log(1+eta)`; this family cannot represent arbitrary dependence.

## Gates before a language-model claim

1. **Exactness.** Enumeration, both forest inference backends, direct scores,
   gradients, empty tails, inactive nodes, initialization, and three-seed
   AB/BA learning are covered by the unit tests. Initial scores must equal
   those of the copied unary adapter, including tail targets.
2. **Small training fit.** On a fixed training-only subset, freeze an
   authenticated unary checkpoint and fit coupling parameters. Log the
   joint-minus-unary gain and maximum marginal error at every evaluation.
   Do not inspect reserved-test documents to debug this stage.
3. **Fresh corruptions.** Fit on training documents with fresh masks and choose
   checkpoints using development joint NLL. Include the independence
   checkpoint. Keep the current pilot's test files immutable; any later use
   of these same documents is a follow-up comparison after observing the
   pilot, not new independent confirmation.
4. **Budget-matched control.** Compare against spending the same additional
   trainable parameters and updates on a second unary correction. The frozen
   base plus coupling costs twice the original unary adapter's parameter
   count. A win over the frozen base alone does not establish a better use
   of that budget.
5. **Simultaneous decisions.** The backbone bridge and specialized FP64
   residual decoder now have generation unit tests. With identical contexts,
   reveal order, and RNG, tested group-size-one trajectories agree exactly
   with the selected-test-precision frozen unary reference, including after
   nonzero couplings. Existing ordinary-output sampler paths are unchanged.
   This does not claim bitwise equivalence between FP32 and FP64 residual
   normalization. Add an authenticated real-text launch, then test
   group sizes 2/4/8/16, logging committed edges, quality, and wall time.
   Use a genuinely untouched corpus for the final confirmation.

No production head, training runner, checkpoint, or paper result is replaced
by this prototype. Relevant tests are `test_centered_coupling.py` and
`test_centered_forest.py`.

## Completed small-fit gate, 2026-09-08

`scripts/run_staged_centered_overfit.py` freezes the authenticated seed-1
unary checkpoint at update 1000 and uses the old, fixed 32-example train/dev
encoder caches. It reads no reserved-test examples and makes no backbone
calls. Three coupling initialization/minibatch seeds (1, 2, 3) each run
300 AdamW updates, batch four, learning rate 0.003, feature dimension eight,
and eta 0.95. They share one frozen unary model, not three independently
trained unary bases.

The frozen unary training NLL is 2.909069 nats per masked token. Final joint
NLL is 2.799315, 2.798191, and 2.799838: all three learn dependence, gaining
0.109753, 0.110878, and 0.109230. Maximum marginal error against the separately
computed frozen unary lattice is 6.67e-16; direct scores and forest inference
agree within 5.69e-14 nats. The frozen unary tensors and input caches are
unchanged in every run.

The old development NLL worsens in all three fits, by 0.001163, 0.003250, and
0.001726 nats. This is a successful training-fit and invariance check, not
evidence of generalization. The next gate remains fresh training with
development selection, including update-zero independence. Compact outputs
and the verified backup manifest are in
`artifacts/paper/staged-debugging-v1/centered-old32`.
