# K=256 candidate-support pilot

This directory contains the complete plan-bound aggregate for the promoted
K=256 pilot. The frozen matrix has one adapter-training seed per arm, two
corruption seeds, WikiText-103 and Scientific Papers arXiv, mask rates 0.50
and 0.90, and 20 jobs (2 train, 2 export, 16 evaluation).

`analysis.json` is the schema-v2 hierarchical conditional-denoising analysis.
It validates every job marker and scientific-output hash before aggregating
corruptions within source document and bootstrapping source documents. Its
SHA256 is
`bc1c1eb4a8a92cf7778eb912a91a0204b9e68affc988a9d642640b96d84aa6c7`.

`compiled-plan.json` is the clean-revision plan at repository revision
`7f161d3da3f059c5b310d165fa7fa5668d217616`; its SHA256 is
`2677b4afd648d8860cf0415d6f82d17b3fc77d9d3aee3e072d000c9b444919c5`.

The equal-weight four-stratum improvement (static conditional NLL minus
contextual conditional NLL per masked token) is 0.0107072 with a 95% interval
of [0.0100941, 0.0113128]. The four condition means are 0.0092582 (arXiv,
0.50), 0.0119968 (arXiv, 0.90), 0.0110045 (WikiText-103, 0.50), and 0.0105692
(WikiText-103, 0.90); all four intervals exclude zero.

This is conditional-denoising evidence from one training seed, not a diffusion
ELBO, likelihood, perplexity, generation-quality result, or a controlled
causal estimate of changing K.
