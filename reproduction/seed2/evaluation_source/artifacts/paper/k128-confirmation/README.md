# Replicated K=128 confirmation

This directory records the complete promoted `candidate_k_128_confirmation`
grid for the predeclared `contextual_vs_static` contrast. The grid contains
three independent adapter-training seeds per arm, four pinned corpora, mask
rates 0.50/0.75/0.90, five paired corruption seeds, and 500 validation batches
per evaluation job (360 evaluation jobs plus 12 training/export prerequisites).

The fail-closed hierarchical analysis averages corruption replications within
source document, resamples training seeds and documents, equally weights the
12 corpus-by-mask-rate strata, and uses 20,000 PCG64 bootstrap resamples. The
pooled static/static-minus-dynamic/dynamic conditional-NLL improvement is
0.0129935 with 95% CI [0.0128493, 0.0131283]; all 12 stratum intervals are
positive.

The exact clean source revision is
`7f161d3da3f059c5b310d165fa7fa5668d217616`. The analysis SHA-256 is
`2db4c1d85186c279a7e39908cf0f50cb7f8287e37fb2e86b5d74f17b3713e484`;
the compiled-plan SHA-256 is
`4a54b5422dab0b720fc095095961d917d31ed86f0accdb7eb1044e190b60f908`.
