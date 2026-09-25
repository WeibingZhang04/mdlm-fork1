# Chain CRFs for masked diffusion generation

This harness adds pair potentials to a frozen, released MDLM-OWT model. It
provides count-based bigrams, learned global transitions, contextual transitions,
and a parameter-comparable independent adapter. The main trainer freezes the
backbone; a separate matched-continuation trainer can update it jointly with
the CRF. Neither requires the repository's older structured-decoder pipeline.

## Model

Given a partially masked sequence, the candidate-chain distribution is

```text
q(states | context) ∝ exp(sum_i unary_i(state_i)
                         + sum_i pair_i(state_i, state_{i+1})).
```

Edges connect adjacent positions in the original sentence. Observed tokens are
clamped. Each masked position has its top K backbone tokens plus one residual
state containing all other tokens. The residual has neutral pair scores; its
token is sampled from the backbone distribution restricted to that tail. This
defines a full-support candidate model, not exact inference over an unrestricted
full-vocabulary transition table. Gold tokens are never inserted into candidates.

The pair models are:

- **Counts:** a smoothed empirical bigram joint distribution, scored as PMI or
  log conditional probability. Counts never connect different input rows.
- **Global:** separate left/right token embeddings, with score `L(a)^T R(b)`.
- **Contextual:** `L(a)^T diag(1 + g_i) R(b)`, where a small MLP predicts `g_i`
  from neighboring backbone hidden states and diffusion time.
- **Independent control:** a low-rank contextual unary correction, with no pair
  potentials. Rank 64 approximately matches rank-32 pair-head parameter counts.

Training minimizes conditional joint negative log likelihood, including the
probability of gold tokens inside the residual state. DP uses at least FP32;
categorical sampling uses FP64 probabilities. Pair scores are unrestricted
signed log potentials. Their low-rank parameterization does not make the
exponentiated transition kernel low rank: inference costs `O(L K^2)`.

Generation uses a fixed random reveal order, shared across methods by sample
index. Each step draws a whole completion with forward filtering/backward
sampling, commits the scheduled subset, and discards the remaining draws.
`--sampling marginal` instead draws independently from the same CRF's exact
one-position marginals. This is a dependence ablation, not joint CRF sampling.

## Install and test

Use Python 3.10. Install PyTorch for the intended device, then:

```bash
python -m pip install -r requirements-chain-crf.txt
python -m pytest -q tests/test_chain_*.py
```

The offline tests cover enumerated partitions, likelihoods, marginals and
gradients; joint sampling; clamping and residual states; learning agreement,
disagreement and context-dependent joints; document separation; generation
schedules; training; exact optimizer/RNG resume; segmented inference; and
baseline/evaluation protocols. The synthetic backbone is
only a test fixture, not a reported language-model result.

FlashAttention is optional. `models/dit.py` uses PyTorch SDPA when FlashAttention
is unavailable. Its `encode`/`decode` interface exposes the unchanged native
backbone's hidden states. Runtime comparisons must use the same attention
implementation, device, batch size and precision.

## Data and released weights

`chain_crf/backbone.py` pins the model and tokenizer revisions and verifies the
released safetensors checksum. No remote model Python is executed. A checkpoint
path may point directly to the verified safetensors file, or to a wrapper made
with `scripts/prepare_released_mdlm_owt.py`.

```bash
python scripts/prepare_released_mdlm_owt.py --output checkpoints/mdlm-owt.pt
python scripts/prepare_chain_data.py --output data/chain-owt --length 256 --train-examples 40000 --dev-examples 128 --test-examples 256
```

Data preparation reads the pinned public OWT training split by default. It
assigns original documents to adapter train/dev/test splits by a salted hash,
retains document identities, and writes manifests and file checksums. These are
adapter-data splits; they do not establish exclusion from backbone pretraining.
Provide `--exclude path/to/earlier-evaluation-manifest.json` to exclude documents
used in an earlier study.

For an existing source, use `--source path/to/source.jsonl` with rows containing
either `text`, or `input_ids` and an original `document_id`. A document-preserving
Hugging Face disk cache is also accepted when its pinned provenance sidecar
passes validation. Rows and document boundaries are never joined. Incomplete
chunks are dropped. Keep the generated data and model files out of Git.

## Train one run per configuration

```bash
python scripts/build_chain_counts.py --data data/chain-owt/train.pt --output checkpoints/owt-counts.pt --max-tokens 10000000
python scripts/train_chain_crf.py --data data/chain-owt --checkpoint checkpoints/mdlm-owt.pt --output runs/global --mode global --rank 32 --length 256 --k 64 --steps 10000
python scripts/train_chain_crf.py --data data/chain-owt --checkpoint checkpoints/mdlm-owt.pt --output runs/contextual --mode contextual --rank 32 --length 256 --k 64 --steps 10000
python scripts/train_chain_crf.py --data data/chain-owt --checkpoint checkpoints/mdlm-owt.pt --output runs/independent --mode independent --rank 64 --length 256 --k 64 --steps 10000
```

The default training batch has four sequences: 10,000 updates at length 256
process 10.24M tokens. Corruptions are resampled on each update. The fixed seed
is for reproducibility; these commands do not run seed sweeps or estimate
confidence intervals. Select settings with development data before final tests.

Training saves `last.pt`, `best.pt`, configuration, data/model/source identities,
optimizer and scheduler state, and random-generator/data-cursor state. To resume,
repeat the same training command with `--resume runs/global/last.pt`; `--steps`
is the total desired update count. Identity mismatches fail rather than silently
continuing a different experiment. `--max-seconds` enables a checkpointed time cap.

`--init-global runs/global/best.pt` can warm-start a contextual head. Record that
extra training history when comparing data budgets; the commands above train
each head independently. Adding `--continue-init-stream` continues the initial
run's exact next data batch and corruption draw, with a fresh contextual-head
optimizer. It requires the same data, seed, batch size and sequence length.

## Matched backbone continuation

The separate joint trainer updates the released MDLM backbone. Its two arms
are ordinary independent MDLM continuation and backbone-plus-contextual-CRF
training. Both use the same effective batch size, data stream, corruption
stream, dropout initialization, backbone optimizer, and time weighting.
The contextual head starts with exactly zero pair scores. A small nonconstant
gate initialization lets context-dependent pair scores begin learning.

```bash
python scripts/train_chain_joint.py --arm independent --data data/chain-owt --release-checkpoint checkpoints/mdlm-owt.pt --output runs/continued-mdlm --steps 1000
python scripts/train_chain_joint.py --arm contextual --data data/chain-owt --release-checkpoint checkpoints/mdlm-owt.pt --output runs/joint-contextual --steps 1000
python scripts/evaluate_chain_joint.py --checkpoint runs/joint-contextual/best.pt --output runs/joint-generation --length 1024 --steps 16 --samples 256 --sample-offset 10000 --score-gpt2
```

The contextual objective is a time-weighted joint denoising loss, not a claimed
diffusion ELBO. All-visible corruptions contribute zero without resampling.
Top-K identities are discrete, but unary values and the complete residual
mass remain differentiable. Gradient accumulation divides by the effective
batch token count, not the microbatch size.

Each checkpoint contains the tuned backbone, head, optimizer, scheduler, data
stream and RNG state. Use `scripts/evaluate_chain_joint.py` for these coupled
checkpoints; do not combine their heads with the unchanged released backbone.
`--control own-marginal` keeps the CRF but samples its node marginals;
`--control pair-disabled` removes its edges while keeping the same tuned
backbone. Resume requires matching data, code and training configuration;
`--steps` is the total update target, including updates already completed.

`scripts/prepare_chain_native_parents.py` recovers intact 1,024-token rows
from an authenticated source cache when all four of their 256-token chunks
belong to an existing training or development split. It verifies the previous
data and exclusions, preserves document roles, and excludes incomplete parents;
it never constructs a longer sequence by joining chunks. Supply the audited
expected row counts with `--expected-train` and `--expected-dev`. Native-length
continuation then uses `--length 1024` with those newly prepared data files.

## Generate, score and diagnose

```bash
python scripts/evaluate_chain_crf.py --backbone-checkpoint checkpoints/mdlm-owt.pt --output runs/eval-base --mode backbone --length 256 --steps 16 --samples 256 --score-gpt2
python scripts/evaluate_chain_crf.py --backbone-checkpoint checkpoints/mdlm-owt.pt --output runs/eval-count --mode count --counts checkpoints/owt-counts.pt --count-mode pmi --strength 0.1 --length 256 --steps 16 --samples 256 --score-gpt2
python scripts/evaluate_chain_crf.py --backbone-checkpoint checkpoints/mdlm-owt.pt --output runs/eval-contextual --mode contextual --head runs/contextual/best.pt --length 256 --k 64 --steps 16 --samples 256 --score-gpt2
```

The count strength above is an example setting, not a claimed optimum. Compare
zero strength, counts, learned heads and the independent control on the same
reveal schedule. Sweep step counts explicitly, including a longer-step baseline
that can use the time spent on pair scoring and DP. Prefix continuation is
available through `--prefix`. Use a separate output directory per configuration;
`--resume` verifies its complete manifest.

`--inference segments` eliminates clamped positions and samples independent
contiguous masked runs, absorbing observed boundary factors into their endpoint
unaries. It preserves the dense model's distribution, not its seed-by-seed
samples. Dense inference remains the default. Padded run batching can use more
memory for mixtures of long and short runs; benchmark the intended batch and
mask pattern rather than assuming it is always faster.

Use `--sample-offset 10000` for a final draw set disjoint from an earlier screen
at offset zero. Outputs distinguish the local `sample_id` from its RNG/reveal
`draw_id`. Keep batch size fixed: categorical RNG consumption is batch-based.
Warmup draws lie outside the requested evaluation interval. Old manifests
without draw identities require their original evaluator, not a new-code resume.
An interrupted partial batch is regenerated at its original RNG offset and
shape. Its saved token prefix must match before missing rows are appended.
Recorded generation time covers retained samples; it is not an accounting of
all compute spent on interrupted and replayed jobs.

Outputs include sampled token IDs and text, source/model identities, backbone
calls, end-to-end timing, sampling overhead, entropy, distinct n-grams and
within-sample repetition. External GPT-2-large scoring excludes prefix targets
and reports token-weighted perplexity. It supports samples within the scorer's
context window. Lower evaluator perplexity alone is not evidence of better
generation; examine diversity and the quality–time tradeoff together.

For held-out denoising diagnostics:

```bash
python scripts/evaluate_chain_crf.py --backbone-checkpoint checkpoints/mdlm-owt.pt --output runs/dev-contextual --mode contextual --head runs/contextual/best.pt --dev-data data/chain-owt/dev.pt --length 256 --k 64 --denoise-only
```

This reports joint conditional NLL, own-marginal NLL, backbone NLL, candidate
coverage and retained mass at several mask rates. These conditional diagnostics
are not an exact likelihood of the final multi-step generation distribution.

## Exact-prefix WikiText transfer

Prepare the pinned, original WikiText-103 raw validation and test splits:

```bash
python scripts/prepare_chain_transfer.py --output data/chain-wikitext --lengths 256 1024 --prefix-length 64
python scripts/evaluate_chain_crf.py --backbone-checkpoint checkpoints/mdlm-owt.pt --mode contextual --head runs/contextual/best.pt --continuation-file data/chain-wikitext/validation/length-256/continuation.jsonl --one-per-document --samples 60 --steps 16 --batch-size 4 --sample-offset 30000 --inference segments --score-gpt2 --output runs/wikitext-contextual-validation
```

Preparation checks the original parquet hashes, reconstructs article boundaries
without changing source text, and creates nonoverlapping within-article chunks.
Two equation lines in the pinned test file resemble titles; explicit source-row
assertions keep them inside their article. Validation and test remain separate.
The bundle records source rows, article and chunk IDs, exact GPT-2 token IDs,
tokenizer revision, and dropped tails. It does not add BOS/EOS tokens.

The continuation evaluator clamps each example's exact prefix IDs and generates
only the suffix. Reference token values are never model inputs; their lengths
specify the generation lengths. GPT-2 scoring conditions on the complete prefix
but scores only suffix targets. Equal-shape examples share a batch without
padding or reordering. `--one-per-document` selects the first complete chunk of
each article; request the actual article count for the chosen length. Articles
shorter than that length have no complete chunk.

`--continuation-offset` selects a source window independently of the random-draw
offset. Resume verifies the input file hash, source identities, shapes and
prefix tokens, then replays an interrupted batch at its original shape and draw
offset. Use validation for configuration selection and keep test untouched
until the final settings are fixed. WikiText transfer does not by itself prove
that these documents were absent from the backbone's pretraining corpus.

## Full-vocabulary count baseline

`scripts/evaluate_chain_sparse_count.py` provides a separate exact
full-vocabulary count model. This is important for conditional-bigram scores:
penalizing only explicit candidate pairs leaves the neutral residual state
unpenalized. Even a uniform conditional bigram model then changes the candidate
distribution, although a constant full-vocabulary edge score should not.

For the smoothed count model and any nonnegative strength, the exponentiated
conditional or PMI potential is a rank-one backoff plus nonnegative sparse
corrections on observed bigrams. The implementation caches both sparse CSR
orientations, uses scaled FP64 forward/backward messages, and samples the
joint chain over the whole vocabulary. It preserves visible-token clamps and
original-position adjacencies without a top-K or neutral-tail approximation.
Its work is `O(L*(E+V))`, where E is the number of observed training bigrams.

```bash
python scripts/evaluate_chain_sparse_count.py --counts checkpoints/owt-counts.pt --backbone-checkpoint checkpoints/mdlm-owt.pt --output runs/full-count-conditional --mode conditional --strength 0.1 --length 256 --steps 16 --samples 256 --batch-size 1 --score-gpt2
```

Static potential construction and loading are recorded separately from full
generation time. `--sampling marginal` uses the same model's exact node
marginals; `--sample-offset`, strict manifests, fixed-batch replay and writer
locking support interrupted runs. The example strength is not a claimed
optimum. The tests include dense enumeration, fractional powers, empty counts,
constant-shift invariance, sampling, clamping and optional CUDA parity. Device
checks skip when CUDA is absent; run them on the target GPU before benchmarking.

`--backend gpu` selects a separate FP64 implementation with device-side column
gathers and deferred message validation. Invalid messages still raise before
any result is returned. The default `reference` implementation is unchanged;
the selected backend and its source checksum are included in the run identity.
Tests compare both backends with dense enumeration and verify matching draws,
visible-token clamps, marginal sampling, and interrupted-batch replay. Measure
both single-sample latency and batched throughput; changing batch size changes
the random stream as well as the runtime.

`--backend segments --max-chunk-tokens 512` eliminates visible nodes and
absorbs their edge factors into neighboring masked spans. Equal-length spans
are batched without padding, using the same full-vocabulary FP64 potential.
The joint distribution is unchanged, but bucketed sampling consumes randomness
in a different order. Backend, chunk budget and source hashes are therefore
fixed in the resume identity. The 512-position default bounds each bucket
chunk's real positions, not peak memory; an individual longer span remains
intact. Boundary gathers and full-vocabulary marginals require additional
memory. CPU enumeration and replay tests do not establish production GPU speed
or memory use: run the CUDA checks and profile the intended workload first.

The full-vocabulary evaluator also accepts the exact-prefix continuation files
described above, with the same source selection and suffix-only scorer:

```bash
python scripts/evaluate_chain_sparse_count.py --counts checkpoints/owt-counts.pt --backbone-checkpoint checkpoints/mdlm-owt.pt --backend gpu --mode pmi --strength 0.25 --continuation-file data/chain-wikitext/validation/length-1024/continuation.jsonl --one-per-document --samples 59 --steps 16 --batch-size 4 --sample-offset 30000 --score-gpt2 --output runs/wikitext-count-validation
```

It clamps each row's exact prefix, preserves adjacent equal-shape batches, and
uses reference suffix lengths but never reference token values in generation.
Use `--sampling marginal` for the corresponding own-marginal control. Input
selection, partial-batch replay, and immutable source checks match the learned
head evaluator. Keep an active run's original code snapshot: adding these
options changes its source identity and does not permit resuming old manifests
under the new evaluator.

## Inspect actual generation histories

```bash
python scripts/trace_chain_generation.py --mode contextual --backbone-checkpoint checkpoints/mdlm-owt.pt --head runs/contextual/best.pt --length 1024 --steps 16 --samples 4 --batch-size 1 --sample-offset 40000 --output runs/contextual-traces
```

The recorder executes the production sampler and saves every actual input
state, commitment, and final output without consuming additional random draws.
Structured traces also contain the full sampled proposals, including discarded
tokens. The factorized baseline samples only scheduled positions, so unproposed
entries are `null`, not invented token predictions. Use `--sampling marginal`
for the same-CRF marginal control and `--mode backbone` for the original model.
Keep all requested draw IDs and state any display crop. Trace capture adds CPU
copies and synchronization, so its timings are explicitly not benchmarks.

## Official Tensor-Train baseline adapter

`scripts/evaluate_chain_tensor_train.py` loads the released rank-4 OWT head from
[the authors' repository](https://github.com/ssamt/tensor-train), requiring clean
source at commit `9d0087afd3771ac3e94898ed842858fcc81fb3b0`. It verifies the head
and released MDLM backbone checksums. The head checkpoint does not contain new
backbone weights: both systems use the same frozen MDLM release. Head capacity,
training budget, attention implementation and runtime stack are not matched by
that fact.

```bash
python scripts/evaluate_chain_tensor_train.py --source-root path/to/tensor-train --checkpoint path/to/ttd_4_marg.pt --cache-root path/to/hf-home --output runs/tt-native --length 256 --steps 16 --samples 512 --score-gpt2
```

Use a separate environment for the official implementation. The audited stack
is Python 3.12, PyTorch 2.3.1+cu121, Transformers 4.46.2,
FlashAttention 2.7.4.post1 and Triton 2.3.1. Defaults invoke the unmodified native
sampler, including its FP32 random draws. `--sampling-precision float64` and
`--schedule matched` are explicitly recorded interventions. The common quality
scorer uses original GPT-2 token IDs without EOS truncation or retokenization;
it is not the original paper's scorer protocol. Generation timing excludes
quality scoring and model loading. Use lengths 256 or 1024 and step counts
dividing the length.
The adapter also accepts `--sample-offset` and verifies draw identities on
resume. Its warmup draws are disjoint from the requested interval, and partial
batches use the same replay-and-verify rule as the chain evaluator.

## Plot measured quality and time

`scripts/plot_chain_generation.py` reads a JSON list of series, each with a
`label` and a `runs` list of completed evaluator directories. It checks equal
sample counts, lengths, batch sizes and quality-scoring protocols, then writes
PDF/PNG plots of generative perplexity against denoising steps and elapsed time.
Its numeric sidecar records source-file hashes without local machine paths.
It neither fits curves nor aggregates training seeds. Kernel, precision and
sampler differences still need to be stated in the figure caption; the shared
scorer does not make different implementations a controlled runtime comparison.
Plotting additionally requires Matplotlib; it is not needed for training or
generation.

```bash
python scripts/plot_chain_generation.py --specification runs/plot-series.json --output runs/quality-time-plot --title "Measured generation quality and time"
```

## Distributional quality with MAUVE

`scripts/evaluate_chain_mauve.py` separates reference selection, GPU feature
extraction and CPU clustering. The implementation has offline protocol tests;
no GPU MAUVE result is claimed by this release. It imports the official
[`mauve-text` package](https://github.com/krishnap25/mauve), rather than providing
a replacement implementation. Install `mauve-text==0.4.0` plus compatible
`faiss-cpu`, `scikit-learn`, `scipy`, `joblib` and `threadpoolctl` in the evaluator
environment without replacing the pinned PyTorch/Transformers stack.

The reference source is the pinned raw OWT cache's final 100,000 documents:
the original MDLM **validation** partition, not a newly claimed test set. The
script checks its metadata against the document-preserving processed cache and
an independent document hash/index. It selects the first requested number of
distinct eligible documents after exclusions, taking each document's first L
raw GPT-2 tokens. Thus the reference population is documents of at least L
tokens. It never joins documents, inserts BOS/EOS, stops at EOS, or retokenizes
decoded strings. Pass all adapter train/dev/test JSONLs as exclusions.

```bash
python scripts/evaluate_chain_mauve.py reference --raw-cache path/to/raw-owt-arrow-cache --heldout-cache path/to/pinned-heldout-cache --exclude-documents data/chain-owt/train.jsonl data/chain-owt/dev.jsonl data/chain-owt/test.jsonl --length 256 --samples 5000 --output data/mauve-reference-256
python scripts/evaluate_chain_mauve.py features --input data/mauve-reference-256/references.jsonl --role reference --length 256 --samples 5000 --output runs/mauve-reference-features
python scripts/evaluate_chain_mauve.py features --input runs/final-method/samples.jsonl --role generation --length 256 --samples 5000 --output runs/mauve-method-features
python scripts/evaluate_chain_mauve.py compare --reference runs/mauve-reference-features --generation runs/mauve-method-features --output runs/mauve-method.json
```

For native-length MDLM evaluation, account for the model's documented
BOS/payload/EOS training convention. A terminal-state feature can distinguish
an unwrapped reference from a generated sequence ending in EOS for that reason
alone. `wrap-reference` creates a separate boundary-matched reference from the
same selected documents, without changing their order or exclusion rules:

```bash
python scripts/evaluate_chain_mauve.py wrap-reference --input data/mauve-reference-1024/references.jsonl --output data/mauve-reference-1024-wrapped
python scripts/evaluate_chain_mauve.py features --input data/mauve-reference-1024-wrapped/references.jsonl --role reference --length 1024 --samples 5000 --output runs/mauve-reference-features-wrapped
python scripts/evaluate_chain_mauve.py compare --reference runs/mauve-reference-features-wrapped --generation runs/mauve-method-features --output runs/mauve-method-wrapped.json
```

Each transformed reference is `[50256] + raw_ids[:1022] + [50256]`. This matches
the boundary convention, not the full packed training population. Original
references and scores remain unchanged; new manifests record the transformation
and parent hashes. Use this same reference for every method. Generation IDs
and their existing features are unchanged. The comparator verifies compatibility
with the original identity feature extractor; it never rewrites its manifests.

Feature extraction is offline-cache-only, using the terminal last-layer hidden
state of GPT-2-large revision `32b71b12589c2f8d625668d2335a01cac3249519`, in FP32.
Reference features can be reused across methods. Compare equal sample counts
and identical lengths/settings. The authors recommend several thousand samples
per distribution and use 5000; this harness labels smaller comparisons
exploratory. It fixes the clustering RNG, uses the standard five optimization
restarts, and does not run model-seed sweeps or confidence intervals. Report
MAUVE alongside evaluator perplexity, repetition, diversity and decoding time.

The implementation contains no claimed benchmark improvements or fabricated
results. Use separately recorded runs for any performance claims.
