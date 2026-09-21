# Why dynamic topology is weaker: current diagnosis

**The evidence points to graph placement and sparse-mask connectivity, with weak anchor supervision. A single proven cause of the generation-PPL gap has not yet been established.**

The completed probe holds each existing 6k checkpoint fixed and changes only its graph. It covers Basic, separate R8 and separate R16 FD/DD; 16 validation chunks; lengths 128, 512 and 1,024; and masking rates 25%, 50%, 75% and 90%. These are conditional NLL measurements, not generated-text PPL.

## 1. DD loses connections when masking becomes sparse

At length 1,024:

| Model | Isolated masked tokens at 25% masking | At 90% masking | NLL change when DD uses a fixed graph* |
|---|---:|---:|---:|
| Basic DD | 20.4% | 0.25% | -0.0034 |
| Separate R8 DD | 23.1% | 0.24% | -0.0068 |
| Separate R16 DD | 21.6% | 0.22% | -0.0065 |
| Fixed graph | about 0.02% | 0% | — |

*Mean change across all four mask rates, nats per masked token; negative is better. The DD checkpoint weights are unchanged. Text-level paired bootstrap intervals exclude zero for these three interventions, but the panel is small and exploratory.

The local-neighbor definitions differ. Dynamic topology proposes local edges only between absolute positions one or two places apart. Fixed topology links consecutive remaining masked positions, even if visible tokens separate them. Thus dynamic topology depends more heavily on learned long-range anchors when masks are sparse. DD also develops higher-degree, longer-range structures at low masking rates. Component-size limits can then reject additional connections. These observations suggest a mechanism; actual generation traces are being collected to check it under the model's own generated contexts.

## 2. Simply increasing the edge count is not the solution

Increasing DD's maximum component size from 32 to 128 increases connectivity but **worsens** conditional NLL by about 0.0073–0.0080 nats/token at length 1,024. Replacing DD's learned graph with the fixed chain improves NLL in all three families. Halving edges helps Basic but not consistently the separate-factor models. Which tokens are connected, and the shape of the components, matter more than raw edge count alone.

## 3. The topology objective is indirect, and anchors look weak

Hard anchor selection and Kruskal forest selection are detached from differentiation. The graph-specific parameters do not receive a direct gradient from the joint token NLL. They learn through the auxiliary gold-reveal influence teacher. Shared context features can receive gradients from both objectives; that does not make hard graph selection differentiable.

On the length-1,024 probe, edge-ranking cross-entropy beats a uniform predictor by 0.37–0.44 nats. Global anchor selection is approximately uniform; slot routing is slightly worse than uniform. This is evidence of a weak part of the learned topology, not evidence that all topology learning failed.

Two design issues need testing: the anchor target varies with the randomly selected revealed source, while anchor selection does not receive that source identity; and pooling anchor logits across slots does not directly require distinct anchors. Raising the topology weight may help optimization, but cannot by itself guarantee a better target or more diverse anchors. Generation traces now record unique anchor locations, duplicate proposals, active-chain coverage, component geometry and isolated nodes.

## Controlled follow-ups already submitted

- **Topology weight:** 0.03 / 0.10 / 0.30, identical Basic DD 6k starting checkpoint, 2,000 additional updates. Compare fixed-mask teacher KL, gradient norms, conditional likelihood, and 100-sample generation PPL. Job array 1556977; pending at the latest check.
- **Generation graph intervention:** Basic DD and separate R16 DD at lengths 128 and 1,024; 8 and 32 denoising steps; 100 samples per cell. Compare the native graph, a fixed graph, and dynamic selection with consecutive masked-token edges added to the proposal set. Checkpoint weights, factor computation, sampler and scorer remain the same. Each case first verifies that tracing alone preserves sampled tokens and NFE. Job array 1556994; pending at the latest check.
- **Longer training:** the eight primary 15k runs and their generation curves continue separately. The first two arms have passed 1k and exported portable checkpoints. These are not replaced by the diagnostic runs.

The strongest candidate improvement is to retain reliable consecutive-masked-token proposals while learning additional connections. It remains an experiment, not an established fix. Fixed-graph inference with DD factors is another low-cost candidate suggested by the conditional tests. Whether either improves generation PPL is still pending.

No other person's jobs or files were modified. All new code is in this campaign's helper directory; original model source and prior experiments remain unchanged.
