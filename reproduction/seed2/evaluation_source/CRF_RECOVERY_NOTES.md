# CRF-MDLM recovery notes

## Scope and formula alignment

The recovery patch keeps the model defined in `diffusion_lm/main.tex`:

- Training still maximises the teacher-forced, locally normalised first-order
  transition likelihood `log P_theta(x_0 | x_t)`.
- Inference still computes the exact forward-backward per-position marginals
  over the pruned candidate lattice and uses those marginals in the reverse
  update.
- The auxiliary unigram objective does not replace the CRF objective. It only
  trains the encoder output head used to choose the top-K lattice candidates.
- The gated reveal changes which still-masked positions are materialised, not
  the marginal probabilities or the diffusion schedule's reveal count.
- Observed tokens now hard-constrain candidate states and transition edges
  *before* forward-backward, as required by the absorbing-process posterior.
- Transition log probabilities are gathered in query chunks. Every chunk still
  uses the exact full-vocabulary log-sum-exp, so this changes peak memory but
  not the locally normalised CRF distribution.

The collaborator-reported `0 -> 0.1` schedule is represented explicitly as a
linear optimizer-step warmup. The endpoint and 10,000-step duration are
configurable; 10,000 is a recovery default, not a value established by the
surviving notes.

## Attention runtime fallback

`models/dit.py` now treats FlashAttention as optional. A compatible installed
FlashAttention package retains the original CUDA varlen-QKV path. When its
import is unavailable (or execution is on CPU), DiT uses PyTorch scaled-dot
product attention and an equivalent non-interleaved rotary implementation.
The recovery CRF always calls the encoder with `seqlens=None`, which is the
supported fallback case; variable-length packed sequences still require
FlashAttention. The package initializer also lazy-loads the autoregressive and
Mamba backbones, so their optional kernels no longer block a CRF-only import.

On the RTX Pro 6000 Blackwell smoke VM, omit/uninstall the incompatible
`flash-attn==2.5.6` package so the fallback is selected. Merely leaving an
importable but kernel-incompatible wheel installed is not sufficient.

## Reproducible modes

- `sampling.crf_reveal=independent`: original random per-position DDPM reveal.
- `sampling.crf_reveal=sequential`: reveal exactly one most-confident site per
  unfinished example per step and take its marginal argmax. Use at least one
  sampling step per initially masked token for a genuinely sequential decode.
- `sampling.crf_reveal=gated`: reveal the most-confident masked positions and
  take each selected marginal argmax (recovery default).
- `sampling.crf_reveal=gated_stochastic`: use the same confidence gate and
  reveal count, but sample selected token identities from their marginals.
- `model.crf.unigram_aux.enabled=false`: reproduce training without the
  pruning-head objective.
- `model.crf.inference_query_chunk_size=64`: bound decoder transition logits
  to 64 queries at once; set to `0` for the original dense allocation. At
  `N=1024`, `K=64`, and `V=50k`, this reduces that allocation from roughly
  6.5 GB (bf16) / 13 GB (fp32) to roughly 6.4 MB / 12.8 MB at batch 1.

## Joint-chain sampling (recovery Method 2) analysis

Independently sampling position marginals discards first-order correlations.
A faithful joint-chain ablation can use the already-computed pruned lattice:

1. Retain every forward log-message from forward-backward.
2. Sample the final candidate from its normalised forward message.
3. Traverse right-to-left and sample candidate `z_i` from
   `softmax(alpha_i[z_i] + transition_i[z_i, z_{i+1}])`.
4. Map candidate indices back to vocabulary IDs, then reveal only the sites
   selected by the same confidence gate.

This should remain an ablation rather than the default. The formula document's
reverse update is written in terms of per-position expectations/marginals, and
a joint proposal changes that sampling approximation. The recovery patch now
hard-constrains observed positions inside the lattice; the joint sampler itself
still needs an exhaustive tiny-chain test against enumerated sequence
probabilities before it should be used for a long run.

## Deliberately excluded from this patch

- The discussed imperfect-generation second training pass. It is less certain and
  changes the training distribution; test probabilities `0, 0.1, 0.25, 0.5`
  only after the two recovery fixes pass a short overfit/sampling gate.
- Full/global CRF normalisation, scratch-vs-checkpoint initialisation, and
  changes to top-K. These are separate scientific ablations, not defect fixes.

## Minimal experiment matrix

Run the same seed/data slice/checkpoint budget for:

1. Original control: auxiliary off, `independent` reveal.
2. Head-only fix: auxiliary `0 -> 0.1`, `independent` reveal.
3. Recovery-parity candidate: auxiliary `0 -> 0.1`, `sequential` reveal, with
   `sampling.steps >= model.length`.
4. Parallel candidate: auxiliary `0 -> 0.1`, `gated` reveal.

Run `gated_stochastic` as the next token-sampling control only if run 4 is
viable; this keeps the first recovery screen to four jobs.

Gate each run first on pruning-head gradient norm, CRF NLL, unigram auxiliary
NLL, top-K target recall, mask count by sampling step, and a small decoded
sample. Only promote viable settings to full OpenWebText evaluation and
generative perplexity. If run 3 is viable, add joint-chain sampling as the next
single-factor ablation; then test imperfect-generation probabilities.

## Recovery validation (2026-08-29)

The bounded recovery gate ran on one NVIDIA RTX PRO 6000 Blackwell GPU. The
32-example, 300-step synthetic overfit reduced the
fixed-mask CRF NLL from `2.83757` to `0.15165`, raised top-K target recall from
`0.05725` to `0.55344`, and produced nonzero final gradients in both the
pruning head (`0.25745`) and transition decoder (`0.66706`).

The checkpoint inference gate then verified:

- marginal probability-sum error at most `1.1921e-7`;
- exactly zero observed-token probability error;
- one revealed site per row for `sequential` and 16 at the half-schedule gate;
- zero unresolved mask tokens after complete sequential and gated sampling.

The generated samples were still 19 token substitutions from their nearest
training sequence and collapsed to the same argmax output in this small run.
The gate therefore validates the recovered training/sampling mechanics, not
the full quality claim; the four-way experiment matrix above remains required
before a full OpenWebText run.

Artifacts are stored in `../results/` as the training JSON, inference JSON,
and 3.1 MB checkpoint. The compute worker was stopped after the gate; its
dedicated persistent experiment volume was retained for reproducibility.
