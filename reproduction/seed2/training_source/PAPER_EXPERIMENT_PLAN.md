# Paper and Experiment Strategy

## Current direction: 2026-09-07

The dated execution snapshot below is historical, not the current launch
queue. Start with `artifacts/paper/staged-debugging-v1/README.md` for the
completed staged diagnosis and its machine-readable records.

The shared scorer fails balanced AB/BA with uniform unaries for a provable
capacity reason. Separate endpoints repair that restriction. In the completed
32-example text fit, however, the separate head's lower training loss comes
from better token marginals; the learned dependence does not transfer to
development documents. A unary adapter fits much closer to the support bound.

Next: inspect the still-decreasing pair training curves before diagnosing
convergence, then train on fresh corruptions with a strong unary control and
separate singleton/dependence reporting. Only a setting with useful held-out
dependence advances to fixed-group joint-versus-marginal generation at group
sizes 1/2/4/8/16. Replicate a surviving effect before broadening the benchmark.
Do not launch the historical grid or treat old pending-job notes as live jobs.

## Historical implementation snapshot (2026-08-31)

Status: implementation and verified-evidence snapshot, 2026-08-31. This
document is the contract between the paper claims and the experiments. It must
be revised whenever evidence contradicts the current story.

## 0. Current execution status

Completed and usable as evidence:

- exact dense/enumerated versus low-rank forest normalization, marginals,
  gradients, joint-sampling, hard-constraint, and CUDA-repeatability tests;
- a protocol frozen after development on seeds 1--8 and evaluated once on
  held-out seeds 9--13: 900 steps, 300 shared-factor warmup steps, factor
  initialization standard deviation 0.25, and factor initialization seed 1729;
- the held-out contextual-matching result at commit `c8d4b70`: contextual TV
  0.05660, invalid mass 0.00509, edge F1 1.0, and an all-positive paired TV
  improvement whose 95% bootstrap interval is [0.62089, 0.62821];
- the optimized inference profile at commit `9a7129f`, using one L4,
  PyTorch 2.5.1+cu121, three warmups and ten measured calls bracketed by CUDA
  synchronization per backend: exact agreement to machine precision and
  2.67x--10.07x dense over low-rank speed ratios across the reported shapes;
- an opt-in real-data DIT bridge with frozen-backbone loading, streaming
  OpenWebText training, finite validation, conditional-denoising-NLL metrics,
  topology-teacher coverage, and factorized/marginal/joint sampling modes;
- two-step text8 train/resume and structured-sampling plumbing checks; and
- a strict full-length structured pass using the pinned released raw MDLM-OWT
  export (no EMA), with finite logits/marginals and normalized nodes.
- a paired frozen-backbone K=64 pilot on held-out OpenWebText, arXiv, PubMed,
  and WikiText-103, with a pooled conditional-NLL improvement of 0.00692
  (95% document-hierarchical paired-bootstrap CI [0.00624, 0.00760]); all
  eight mask-0.75/0.90 dataset intervals favor the contextual adapter, while
  mask-0.25 effects are mixed;
- a separately prespecified K=128 candidate pilot on arXiv and WikiText-103,
  with a pooled improvement of 0.00998 (95% CI [0.00934, 0.01060]) and all
  four dataset--mask intervals favoring the contextual adapter; and
- frozen routing decisions that promote both broader replicated K=128
  confirmation and a K=256 support pilot. K=128 candidate recall and retained
  mass remain limited (0.6531 and 0.6555 overall), so this is not evidence that
  candidate support is sufficient.

Not yet evidence and therefore not a current paper claim:

- generated-text quality or matched-quality end-to-end latency (a frozen,
  paired WikiText-103 generation run is active but remains non-evidence until
  its complete 32-shard union passes fail-closed aggregation);
- a replicated adapter-training-seed effect, broader K=128 confirmation, or a
  K=256 result;
- the planned real-text 2x2 topology/factor decomposition and K sweep (the
  earlier 20-job, two-arm cloud dry-run plan is superseded and non-executable;
  the four-arm smoke will contain 40 jobs once compiled: 4 train, 4 export,
  and 32 evaluation jobs);
- CoDD, scheduler, and modern 7--8B comparisons; and
- any diffusion ELBO, data likelihood, MAUVE, reference-model perplexity, or
  state-of-the-art result for the contextual forest.

## 1. Decision: pivot from a fixed chain CRF to contextual coupling forests

The old idea correctly identifies a real problem: a masked diffusion language
model (MDLM) predicts clean tokens with a product of marginals, even when it
commits several mutually dependent tokens in one denoising step. However,
"replace factorized predictions with a tractable joint model" is no longer a
novel claim. Discrete Copula Diffusion, EDLM, ADJUST, CoDD, FeF-DLLM, and
CoCommit all address the same marginal--joint gap. CoDD is especially close: it
multiplies the backbone's contextual unaries by a tractable, static structural
prior and samples jointly.

A fixed context-dependent chain is also too incremental. Fast Structured
Decoding (Sun et al., 2019) already combines globally normalized chain CRFs,
top-K candidate truncation, low-rank/context-dependent transitions, and exact
dynamic programming; Cascaded Text Generation with Markov Transformers
(Deng and Rush, 2020) is another close precursor.

The defensible paper is therefore about **adapting the dependency topology to
the corrupted context and diffusion time**:

> A masked-diffusion backbone predicts a sparse forest over currently unknown
> positions and context-conditioned token-pair potentials on its edges. Within
> a top-K candidate lattice, the resulting denoising distribution is globally
> normalized and supports exact sum-product marginals and exact joint sampling.

The linear chain remains an important ablation and implementation milestone,
not the headline contribution.

Current title:

> **Contextual Coupling Forests for Joint Denoising in Diffusion Language
> Models**

## 2. What the paper may and may not claim

### Safe target claims

1. Context- and time-dependent graphical output structure improves recovery of
   dependencies that a factorized denoiser misses.
2. Forest structure permits exact normalization, marginals, and joint sampling
   inside the candidate lattice.
3. The largest gains should occur at high mask ratios and aggressive parallel
   commitment, where conditional total correlation is largest.
4. Modeling and scheduling are complementary: a structured sampler can model
   selected dependencies, while DEMASK/DAPD/PUNT-style scheduling can avoid
   dependencies that remain outside the forest.
5. A contextual forest should improve the quality--latency frontier relative
   to a matched factorized MDLM and a static structured prior.

### Claims to avoid

- first structured or joint diffusion denoiser;
- first identification of the factorization barrier;
- dense full-vocabulary pairwise modeling (the residual state preserves token
  support, but only retained candidates receive explicit pair couplings);
- exact data likelihood unless the induced reverse process and variational
  bound are derived and tested;
- universal long-range dependency modeling;
- state of the art before matched experiments exist.

## 3. Method the experiments must instantiate

Let `M_t` be the masked positions in `x_t`. A backbone produces hidden states
`h_i`, unary token logits `u_i`, and pairwise dependency scores. Candidate sets
`C_i` retain the top K unary tokens. A residual state aggregates all remaining
unary mass and decodes from the normalized tail distribution. Pair factors are
one whenever either endpoint is residual. This retains full token support and
keeps training and inference candidate sets target-independent; recall and
retained mass are still reported as approximation diagnostics.

From a sparse candidate edge graph (local-window edges plus a few
attention/dependency proposals), the model selects a bounded-component maximum
spanning forest `F_theta(x_t,t)`. The conditional energy is

    E(z; x_t,t) = sum_i u_i(z_i)
                  + sum_(i,j in F) psi_ij(z_i,z_j; x_t,t).

The pair potential is a low-rank mixture whose gates depend on `h_i`, `h_j`,
and a time embedding. The forest is globally normalized. Tree sum-product gives
the log partition and node/edge marginals; ancestral tree sampling gives one
joint clean assignment. The absorbing reverse posterior then decides which
tokens are committed. Observed tokens are evidence, never latent variables.

Complexity should be reported as `O(|M_t| K + |E_t| K^2)` for dense potentials
or `O(|M_t| K + |E_t| K R)` when the *positive factor itself* has nonnegative
rank R. Low-rank log potentials do not obtain the latter bound. Forest
construction and every runtime component are measured. Components are capped
so the structured layer can be batched and parallelized.

Training objective:

- structured denoising negative log likelihood on forward-corrupted examples;
- unary auxiliary loss to keep candidate recall high;
- a dependency/topology supervision or distillation loss based on conditional
  influence estimates;
- optional end-to-end fine-tuning after a frozen-backbone adapter phase.

The draft must distinguish this target method from the current recovery code,
which trains a locally normalized directed Markov model and samples independent
marginals.

## 4. Paper organization

1. **Introduction.** The quality--parallelism tradeoff; why selection-only and
   static-coupling methods leave a gap; contributions and scope.
2. **Background and closest work.** MDLM clean-data prediction, conditional
   total correlation, CoDD/DCD/EDLM/ADJUST, dependency-aware schedulers, and
   earlier structured non-autoregressive decoding.
3. **Contextual coupling forests.** Candidate lattice, dynamic topology,
   pairwise potentials, global normalization, exact inference, joint reverse
   sampling, and complexity.
4. **Why topology must adapt.** A controlled construction plus the distinction
   between modeling and avoiding dependencies.
5. **Experiments.** Controlled recovery, MDLM-scale language modeling, a modern
   7--8B adapter, efficiency, and ablations.
6. **Analysis and limitations.** Candidate truncation, tree bias, topology
   discreteness, compute, and claims that remain conditional on results.
7. **Conclusion.** Conservative summary tied to measured evidence.

The main mechanism figure should contrast factorized mixing, a static prior,
and a context-dependent forest on two contexts with different dependency
graphs. The main empirical figure should be quality versus measured wall-clock
latency.

## 5. Backward-designed experiment program

### Stage A: mathematical and implementation correctness (complete)

Run exhaustive enumeration for tiny vocabularies and lengths:

- log partition, node/edge marginals, and gradients;
- empirical frequencies from joint tree sampling;
- reverse-mixture probabilities for absorbing diffusion;
- clamping and top-K behavior.

Every quantity agrees with enumeration to numerical tolerance. The implemented
architecture-count/no-edge control retains inactive pair parameters, so it is
not described as an active-capacity-matched unary adapter.

### Stage B: controlled mechanism experiments (partially complete)

Use three seeds on:

- ambiguous pairs with identical unigram marginals but incompatible joint
  modes;
- first-order Markov languages with adjustable coupling strength;
- a nonlocal copy/agreement task whose relevant edge changes with context.
- a cyclic/XOR counterexample that exposes the forest family's limitation.

Compare factorized, locally normalized chain, global natural-order chain,
static forest, and contextual forest. Report exact KL/TV where available,
conditional NLL, edge recovery, mutual-information recovery, invalid-pair rate,
and speed. Full vocabulary removes top-K as a confound.

Decision gate: do not scale unless joint sampling beats independent sampling
from the same checkpoint and the contextual forest beats both the factorized
and static-structure models on the nonlocal task.

The frozen-protocol context-switching-matching experiment passes this gate on
held-out seeds 9--13. The broader ambiguous-pair, adjustable Markov, and XOR
learned-task matrix remains pending; oracle/table-fit checks are mechanism
sanity evidence and must not be conflated with learned-adapter results.

### Stage C: MDLM-scale controlled language modeling (quality pilots complete;
confirmation and generation active)

Use the released `kuleshov-group/mdlm-owt` checkpoint in the
`yuntian-group/mdlm` codebase, pinned to revision
`d0958fa851335ece6c15260ce0025f030673c0fb` and verified against the released
`model.safetensors` SHA256. The public artifact contains raw `backbone.*`
weights and no EMA. Freeze the backbone first and train output adapters on
streaming OpenWebText. This makes the closest comparisons cheap and fair.
Retain OpenWebText because MDLM, SEDD, and AR checkpoints are public; evaluate
paired fixed examples/corruptions on WikiText-103. Add text8 with its full
35-character vocabulary as an exact-lattice sanity benchmark.

Minimum matched baselines:

- original factorized MDLM, with both its original and confidence reveal rules;
- architecture-count/no-edge control (legacy experiment identifier:
  `parameter_matched_independent`; pair parameters inactive);
- locally normalized chain;
- globally normalized natural-order chain;
- static forest/transition prior;
- contextual forest with independent-marginal versus joint sampling;
- CoDD and a context-modulated CoDD variant following its appendix;
- one dependency-aware scheduler (DEMASK, DAPD, or PUNT), alone and composed
  with the contextual forest.

Evaluate 1, 4, 8, 16, 32, 64, and 128 denoiser calls; K in 8, 16, 32, 64, and 128;
lengths 256, 512, and 1024; batch sizes 1 and 8.

Metrics are separated into:

- **fit:** valid diffusion bound/estimated NLL, structured conditional NLL by
  mask-ratio bin, calibration, candidate recall;
- **generated distribution:** MAUVE, two reference-AR perplexities (labeled as
  proxies), distinct-n, Self-BLEU, repetition-2/4, n-gram JS divergence, and
  adjacent/nonlocal agreement;
- **efficiency:** latency, throughput, denoiser calls, finalized tokens per
  call, peak memory, and structured-layer share of latency.

Decision gates:

- recall@64 at least 99% on masked targets;
- confidence-gated factorized MDLM does not explain the gain;
- parameter matching does not erase the gain;
- contextual structure beats a static prior, especially at high mask ratios;
- per-step cost remains below 4x and yields a better matched-quality latency.

The first recall target above was an aspirational scale target, not the frozen
pilot routing policy. The measured K=64 support was far below it, so the
prespecified support route promoted K=128 rather than terminating the study.
K=128 improved the paired conditional-NLL result but still did not establish
support sufficiency; the next matched support point is therefore K=256. The
quality and support routes remain separate throughout analysis.

### Stage D: modern transfer experiment (pending)

Primary target: Dream-7B; secondary target: LLaDA-8B. Freeze the backbone and
train only the structured adapter first. Use the public competitor settings so
that CoDD, ADJUST, DEMASK/DAPD, Fast-dLLM/APD, and the serial reference are
meaningful comparisons.

Tasks:

- ParallelBench PB80/PB90 and complete accuracy-versus-tokens-per-step curves;
- IFEval;
- GSM8K and MATH-500;
- HumanEval and MBPP;
- WikiText-103 continuation;
- random-span and multi-span infilling.

GPQA is optional because it is costly and weakly diagnostic of the mechanism.

### Stage E: essential ablations (pending)

- no edges / natural chain / static forest / contextual forest;
- fixed topology with dynamic factors / dynamic topology with fixed factors /
  dynamic topology with dynamic factors;
- position-only / context-conditioned / context-and-time-conditioned edges;
- context-shuffled and timestep-shuffled topology controls;
- local candidate graph / attention proposals / learned proposals;
- local versus global normalization;
- independent marginals versus exact joint sampling;
- unary-only versus exact globally temperature-scaled sampling;
- K, pair-potential rank, component-size cap, and mask-ratio bins;
- frozen versus jointly tuned backbone;
- scheduler alone, forest alone, and their composition.

## 6. Measured evidence and unresolved hypotheses

The current learned-quality evidence comprises the frozen-protocol synthetic
held-out experiment and the conditional-denoising K=64/K=128 real-text pilots
described in Section 0. The real-text intervals are conditional on one trained
adapter per arm and therefore do not quantify training-run variability. The
optimized forest profile is a kernel-only measurement: it excludes
candidate/factor and forest construction, so it cannot be presented as
end-to-end decoding latency. The released-backbone and text8 runs are plumbing
checks and cannot be presented as language-quality evidence. Until the active
generation union is complete and verified, none of these results supports a
generated-text-quality claim.

The earlier numerical targets for MAUVE, reference-model perplexity,
dependency/repetition error, end-to-end overhead, and denoiser-call regimes
have been removed. Those cells remain unclaimed until frozen-protocol paired
experiments produce artifacts. Future hypotheses remain qualitative:

- benefits should concentrate at high mask ratios and aggressive commitment;
- contextual topology should matter most when the relevant dependency changes
  with corrupted context;
- a scheduler and a structured output distribution may be complementary; and
- candidate truncation and forest bias may limit gains on long-range cycles.

No predicted numeric result may be inserted into a submission table or prose.

## 7. Code ownership in `yuntian-group/mdlm`

- `models/structured_decoder.py`: new unary, dependency, topology, and
  low-rank pair-potential heads; keep `models/crf_decoder.py` as the legacy
  locally normalized ablation.
- `structured_utils.py`: forest construction, tree partition/marginals,
  ancestral joint sampling, clamping, and reverse-mixture tests.
- `diffusion.py`: structured loss, joint reverse sampler, matched reveal rules,
  and metric hooks.
- `models/dit.py`: hidden-state/unary interface and safe checkpoint loading.
- `dataloader.py`: synthetic languages, text8, deterministic corruptions, and
  fixed evaluation splits.
- `configs/model/contextual-forest-small.yaml`: method configuration; do not
  silently redefine the existing CRF config.
- `configs/experiment/`: named pilot, ablation, and scale configurations.
- `evaluation/`: quality, dependency, infilling, memorization, and latency
  evaluation.
- `tests/test_structured_forest.py`: exhaustive checks.
- `scripts/`: resumable pilot/full launchers, immutable release preparation,
  profile provenance, and paired artifact aggregation; large data and
  checkpoints belong on a dedicated persistent experiment volume.

The branch now contains paper-usable exact-inference, held-out synthetic,
conditional-denoising K=64/K=128, and kernel-profile evidence. The original
32-example recovery smoke remains only a historical mechanics check. Generation,
replicated-training, causal-decomposition, and K=256 claims remain pending.

## 8. Submission-critical execution order (frozen 2026-08-31)

The next experiments are ordered by the paper claim they unlock, not by code
convenience. A failed gate changes the paper claim before it triggers more
compute.

1. **Paired end-to-end WikiText-103 infilling generation.** Evaluate 197
   revision-pinned prompts with four paired draws per prompt (788 draws), a
   document-local contiguous 32-token span inside a length-256 sequence, and
   NFE budgets 8/16/32/64. For the dynamic adapter compare the factorized
   backbone, independent samples from exact structured marginals, and exact
   joint forest samples. Compare the latter with exact joint samples from the
   paired static/static adapter. The primary causal contrast is joint versus
   independent sampling from the same checkpoint and exact node marginals.
   Report reference-span accuracy/exact match, GPT-2-large reference NLL, and
   repetition. Use paired-draw and prompt-cluster bootstrap intervals. The
   current run uses 16 atomic shards per adapter arm and two concurrent L4
   workers, so its timing is descriptive and cannot support a quality--latency
   claim.
2. **Real-text causal decomposition.** From a clean committed revision,
   compile and run the replacement 40-job smoke plan before scaling: 4 train,
   4 export, and 32 evaluation jobs. It contains the four topology/factor arms, an
   algebraically no-edge control without extra active pair capacity, a
   degree/component-preserving permuted-topology diagnostic, and a one-pass
   K=32/64/128/256 support sweep. The old 20-job cloud dry-run is superseded
   and cannot satisfy the manifest or promotion-policy commitments. Only the
   frozen technical-validity route may promote the multi-seed primary suite;
   no NLL sign, effect magnitude, p-value, or confidence-interval position is
   a promotion criterion.
3. **Replication before breadth.** Add independent adapter-training seeds to
   the K=128 confirmation so uncertainty covers training-run variability.
   Then run the promoted K=256 support pilot. Do not pool K=64, K=128, and
   K=256 estimates as a causal K curve unless their strata, corruption draws,
   training protocol, and evaluation depth are matched.
4. **Cross-domain generation.** If the WikiText generation contrast is
   informative, repeat the frozen paired protocol on revision-pinned arXiv and
   PubMed prompts already declared in the manifest. This is the highest-value
   dataset expansion because it tests whether joint sampling helps outside the
   adapter's general-web validation domain without changing the metric stack.
5. **Uncontended end-to-end timing.** After the quality matrix finishes, replay
   a small fixed set of identical batches one process at a time, with explicit
   warmups and repeated measurements. Only this replay may support an
   end-to-end quality--latency claim; the concurrent paper-scale collection may
   not.
6. **Modern transfer/baselines.** Run a batch-1 L4 feasibility gate for the
   pinned Dream-7B checkpoint and official CoDD, DAPD, and ParallelBench code
   before promising a 7B-scale result. If memory, tokenizer, or output-interface
   incompatibility blocks a controlled comparison, report that boundary and
   prioritize the replicated MDLM-scale evidence instead of an incomparable
   headline number.

Remote runs use an experiment-dedicated persistent volume, explicit run labels,
resumable checkpoints, immutable source revisions, and a hard stop. Public
artifacts record hardware/software provenance without private account,
funding, project, instance, or resource identifiers.

### Fail-closed causal smoke-to-primary route

The source plan ID necessarily commits the repository revision that contains
the frozen template, so promotion uses two pre-result stages. First commit the
manifest, template, compiler, evaluator, and tests. Then compile the smoke plan
and immediately finalize the template against that exact, still-unstarted
plan. The finalizer refuses any non-empty source job directory.

```bash
python scripts/compile_experiment_matrix.py \
  --manifest configs/experiment/contextual-forest-causal-evidence-v1.yaml \
  --suite causal_smoke \
  --artifact-root /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1 \
  --output-dir /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke

python scripts/finalize_causal_smoke_policy.py \
  --plan-dir /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke \
  --output /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke/promotion-policy.json
```

After all 40 jobs finish successfully, the causal aggregate binds the union of
all success markers and output hashes and is deterministically replayable from
the raw schema-v2 records. The evaluator checks the full four-arm factorial,
pairing digests and identical masked-token counts for every arm/seed-paired
window, finite and ordered statistics, the algebraic no-edge identity within
its frozen absolute tolerance, the complete support grid with monotonicity
checked at its frozen tolerance, and non-empty forests. Every raw schema-v2
record carries independently recomputed original/permuted degree sequences and
component-size multisets; both must match per record and per condition before
the predeclared topology-change thresholds are consulted. It emits compiler
evidence only if all of those technical checks pass.

```bash
python scripts/aggregate_causal_denoising_eval.py \
  --plan-dir /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke \
  --manifest configs/experiment/contextual-forest-causal-evidence-v1.yaml \
  --suite causal_smoke \
  --output /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke/analysis.json

python scripts/evaluate_causal_promotion.py \
  --policy /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke/promotion-policy.json \
  --analysis /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke/analysis.json \
  --source-plan /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke/compiled-plan.json \
  --output /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke/routing-decision.json \
  --compiler-evidence-dir /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke

python scripts/compile_experiment_matrix.py \
  --manifest configs/experiment/contextual-forest-causal-evidence-v1.yaml \
  --suite causal_primary \
  --promotion-evidence causal_primary /mnt/contextual-forest/experiments/contextual-forest-causal-evidence-v1/plans/causal-smoke/causal_primary-promotion.json
```

The primary compiler selects its causal verifier from a repository-trusted
template registry, replays the source analysis from marker-bound records, and
requires the current checkout to be the exact clean source-plan HEAD while it
checks the policy, plan, repository, job-spec, output, analysis, decision, and
evidence hashes. It never trusts a boolean supplied by the evidence file.
