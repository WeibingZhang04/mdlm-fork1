# Released Tensor-Train feasibility matrix

This directory contains the complete fail-closed replay of the six-cell
OpenWebText feasibility matrix comparing the unmodified released marginal and
rank-4 Tensor-Train checkpoints at 8, 16, and 32 NFEs. Each cell contains 256
length-1024 unconditional samples generated with paired seeds and recorded
position schedules, then scored by the same immutable GPT-2-large evaluator.

`analysis.json` validates all six success markers, manifests, scientific-output
hashes, checkpoint and runtime identities, and paired schedule hashes. Its
SHA256 is
`8708305b7c2abfe98fd3c8f9f62878ce19d4831586c192f1a85ad7392b63410c`.
The compiled plan SHA256 is
`4a1efa8240df5b71ca875ada888bf5cba0259d8dca02ac49771b79ef23ea7c02`.
`preflight.json` is the successful exact-stack one-sample compatibility check;
its SHA256 is
`bb3967d36f27289b5a844135339dfbaa720fc9e0f02fb644fa647ca6005064e5`.

Relative to the released marginal checkpoint, the released rank-4
Tensor-Train checkpoint lowers mean GPT-2-large evaluator NLL by 0.26944,
0.14160, and 0.05115 nats at 8, 16, and 32 NFEs. The marginal checkpoint is
2.02x, 1.70x, and 1.39x faster in samples per second at those budgets. These
are descriptive comparisons from one paired generation seed and one recorded
exclusive NVIDIA L4 host.

This matrix compares two released external baselines. It is not a matched
comparison to Contextual Coupling Forests, a likelihood or perplexity result
for the diffusion models, a human evaluation, or a replicated-seed claim.
