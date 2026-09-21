# Local-only CCF speed investigation v3 — September 16, 2026

## Scope and current bottleneck evidence

No production module changed, no new Slurm jobs submitted, no running snapshot
edited. Experiments live in `scripts/audit_ccf_sampling_v3.py` and scoped
patches affect only that diagnostic process. The user permits examining skipped
validation, but model logic, sampling budget, precision, and RNG behavior remain
the comparison target. Production remains GPU-verified v2 (4fc49fb).

The 15-second MDLM number is our server measurement, not a paper claim:

- Job1541925, watgpu308, same factorized generation harness, 10 samples,
  batch1, length1024, 1000 uncached DDPM transitions plus cleanup, 1001 measured
  backbone calls for each sample. First17.989s, later14.456–14.635s/sample.
- Slurm wall time was2m57s total including initialization and GPT-2 scoring.
- The harness timer synchronizes CUDA before and after sampling; excludes
  model construction/loading, text decoding, file output and reference scoring.
  The CCF per-sample timer uses the same boundary, so comparing it against
  MDLM's per-sample timer does not mix loading time with generation time.
- First completed v2 rank8 follow-up samples: FD1500=345.297s on watgpu1208,
  FD2500=312.761s on watgpu508. Total job times6m09s and5m42s respectively.
  These are about5.2–5.8min of generation, still far from MDLM speed.
- Different hardware/checkpoints prevent claiming a controlled exact ratio or
  attributing all time exclusively to one routine. Full-model same-GPU paired
  timing remains needed. Both code paths call the same backbone encode/decode;
  both use its bfloat16 autocast internally. No second backbone call was found
  in each CCF update. MDLM's control does not use ddpm_cache.

H200 question: Slurm identifies watgpu508 as H200. All four LR-decay evaluation
tasks1543754_0–3 completed there with mdlm/PyTorch2.2.2+cu121 (CUDA12.1), including
generation and GPT-2 scoring. Thus mdlm-pro is not required just to run this
workload on H200. This is compatibility evidence, not proof that its older
environment is fastest. NVIDIA lists H200 as compute capability9.0:
https://developer.nvidia.com/cuda/gpus
Hopper compatibility guide:
https://docs.nvidia.com/cuda/archive/12.1.1/hopper-compatibility-guide/index.html

## Candidates tested locally

1. **Unchecked categorical draw.** PyTorch2.2.2's single-sample multinomial
   checks weights with two host scalar reads, even after v2 removed our extra
   checks. Prototype performs the SAME softmax, same-shaped exponential draw,
   in-place probability/noise division, and argmax, omitting native validation.
   CPU profiling: two scalar reads become zero. Unlike replacing this with
   MDLM's uniform-noise implementation, it preserves the tested RNG stream.
   Invalid inputs are no longer rejected here; use only with known-valid rows.
   This is a private local experiment, not a new default or generic public API.
2. **Batched upward messages.** Group independent messages by depth and child
   count; keep per-parent child-addition order identical. Reuse the repo's
   `_batched_low_rank_message`. Sampling itself still uses original serial
   traversal/draw order. Full marginal inference remains unchanged.
3. **Validation ablation.** In the diagnostic process only, remove `_require`
   calls from low-rank input validation, constraint preparation and topology
   building, plus the finite/positive factor scans used only by those checks.
   This measures their cost, not an endorsement of removing all of them from
   every caller. In particular topology building still performs required
   traversal; forest constraints themselves are NOT removed.
4. **Fixed topology unused proposals.** The head still runs proposal generation
   and scoring that fixed edge selection does not consume. Diagnostic bypass
   preserves all tensors consumed by sampling. Proposal diagnostics change, so
   such a path must be generation-only, never silently used in training.
5. **Dynamic Kruskal Python overhead.** Tensor indexing and converting individual
   CPU scalars per edge can be replaced by bulk list conversion while retaining
   the existing stable tensor sort and identical union-find logic/tie breaks.

## Local measurements, not CUDA/full-generation claims

CPU PyTorch2.14.0, one thread, synthetic B2/L1024/K128/R16/V256 inputs,
five measured repetitions after warmup. Exploratory timings, sequential variant
order, not a final controlled GPU benchmark.

| Variant | All-active seconds | Mixed seconds | Speedup vs v2 all / mixed |
| --- | ---: | ---: | ---: |
| Current v2 | .160130 | .126118 | 1 / 1 |
| Unchecked draw | .146574 | .113220 | 1.09 / 1.11 |
| Unchecked + validation ablation | .136290 | .102197 | 1.17 / 1.23 |
| Unchecked + batched upward | .115878 | .100767 | 1.38 / 1.25 |
| All three | .106420 | .089971 | 1.50 / 1.40 |

All four variants passed54 existing exact token/RNG cases each. Additional
L1024 comparisons used two seeds for all/mixed/no-active fixtures and matched.
Unchecked categorical:96 shape/dtype/seed tests matched native tokens/RNG.
Batched upward/root tensors:256/256 bitwise equal in each of all/mixed/none
L128 fixtures. Dense cases in the54 do not exercise low-rank message changes.
No CUDA or full real-model trajectory tests for these candidates yet.

Head-only profiling used synthetic full-size tensors: B1/L1024/V50258/H768,
separate rank16, actual topology widths, active counts1024/512/102. The lattice
(top-K plus vocabulary reductions) cost .124–.180s on CPU; factor tables roughly
.019–.049s. Fixed unused proposal bypass changed head times .183→.163,
.169→.174, .174→.169s (small/noisy, not uniformly beneficial). Dynamic list
Kruskal changed .176→.168, .176→.167, .166→.168s. All consumed tensors matched;
the loop optimization also passes tie/mask/cap/threshold tests. These small
head changes alone cannot explain away the remaining minutes of GPU latency.

## Reaching MDLM-like latency

Removing the remaining host synchronization and batching messages are the next
conservative targets. Even these prototypes retain about1.024 million tiny
categorical invocations per1024-token/1000-update sample, plus serial conditional
pair-row operations. MDLM generates categorical draws across positions in large
tensors instead. Scalar validation removal alone is not a credible guarantee
of20x more speedup. A level-parallel ancestral sampler is the larger target:
parallelize independent components/nodes without dropping correlations. Simply
batching multinomial calls changes RNG consumption; exact seeded parity needs
separate work (e.g. preserving per-node noise generation/order before batching
deterministic conditionals), or explicit permission to change seeded outputs.

Other unused work: factors and logs are computed for padded edges and resolved
positions; residual draws cover every position when any residual is selected;
the backbone output is cloned to mask one vocabulary column. Pruning these must
preserve RNG consumption and not remove conditioning on resolved tokens. No
backbone caching across changed inputs/times, lower precision, fewer timesteps,
changed K/rank, or altered reveal schedule was tested or applied.

We have NOT reached MDLM speed and cannot guarantee parity without a same-GPU
full-generation measurement. The next server test should compare v2 and these
candidates on the same GPU/checkpoint and include a factorized MDLM control.
It must use a new snapshot; current arrays1544304/1544305 are untouched.

## Source / attribution (new external implementation consulted)

`unchecked_rows` adapts PyTorch's multinomial_out single-sample fast path:
https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/native/Distributions.cpp#L599-L642
Retrieved the pinned v2.2.2 source directly and inspected the validation,
exponential draw and argmax sequence. PyTorch copyright/license is retained at
`docs/third-party/pytorch-v2.2.2-LICENSE.txt`, from
https://github.com/pytorch/pytorch/blob/v2.2.2/LICENSE
The remaining prototypes derive from existing repository code. This attribution
applies to v3 only; previous v1/v2 provenance records remain unchanged.
