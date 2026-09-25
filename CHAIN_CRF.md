# Chain CRFs for masked diffusion generation

This harness adds pair potentials to a frozen, released MDLM-OWT model. It
provides count-based bigrams, learned global transitions, contextual transitions,
and a parameter-comparable independent adapter. It does not modify the backbone
weights or require the repository's older structured-decoder pipeline.

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
python -m pytest -q tests/test_chain_crf_core.py tests/test_chain_crf_heads.py tests/test_chain_generation.py tests/test_chain_training.py
```

The 42 offline tests cover enumerated partitions, likelihoods, marginals and
gradients; joint sampling; clamping and residual states; learning agreement,
disagreement and context-dependent joints; document separation; generation
schedules; training; and exact optimizer/RNG resume. The synthetic backbone is
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
each head independently.

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

The implementation contains no claimed benchmark improvements or fabricated
results. Use separately recorded runs for any performance claims.
