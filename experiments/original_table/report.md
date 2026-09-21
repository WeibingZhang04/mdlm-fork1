# CCF experiments: shareable TL;DR

We tested whether a **CCF structured output head** improves few-step generation from the released MDLM backbone. The clearest result is that CCF substantially lowers GPT-2-large generative perplexity at **4, 8, and 16 reverse steps**. At 32 steps the best CCF result is also lower, but the gap is smaller. The current main sweep contains **11,020 freshly generated and scored samples**.

**Comparability qualification:** reverse-step schedules match, but CCF and MDLM differ in probability-calculation precision and final cleanup. These are observed pipeline gains; a matched control is still needed to isolate the gain from the CCF head. See the [step and setup audit](README.md#comparability).

## How to read the labels

The two letters describe **graph / pair scores**, in that order:

| Label | Meaning |
|---|---|
| **FF** | **Fixed graph + fixed (static) pair scores.** Internally this was previously called `SS` / `static_static`; **FF is the clearer human-facing label for the same model**, not a new run. |
| **FD** | **Fixed graph + dynamic/contextual pair scores.** |
| **DF** | **Dynamic/contextual graph + fixed pair scores.** |
| **DD** | **Dynamic/contextual graph + dynamic/contextual pair scores.** |

Model-family prefixes:

- **Basic:** one shared rank-16 token-factor table.
- **Separate R8:** separate left/right endpoint tables, rank 8. It is approximately parameter-matched to Basic R16.
- **Separate R16:** separate left/right endpoint tables, rank 16; it has more parameters and is not parameter-matched to the other two.
- **MDLM:** released factorized baseline without the CCF joint head.

## Best results from the 100-sample experiments

Each entry is **PPL @ training checkpoint**. Lower PPL is better. Each model/step entry uses **100 generated samples**; the best checkpoint is selected from the three checkpoints carried into that 100-sample experiment.

| Model | 4 steps | 8 steps | 16 steps | 32 steps |
|---|---:|---:|---:|---:|
| **MDLM baseline** | 1946.88 @ released | 815.06 @ released | 313.78 @ released | 162.56 @ released |
| **Basic FD** | 1618.64 @ 4k | 606.69 @ 6k | **244.05 @ 6k** | **133.42 @ 6k** |
| **Basic DD** | 1667.87 @ 3k | 619.00 @ 3k | 264.12 @ 3k | 145.68 @ 3k |
| **Separate R8 FD** | 1632.05 @ 6k | 625.03 @ 6k | 248.82 @ 5k | 135.74 @ 6k |
| **Separate R8 DD** | 1680.80 @ 2k | 677.00 @ 5k | 255.04 @ 5k | 153.37 @ 2k |
| **Separate R16 FD** | **1582.27 @ 6k** | **605.85 @ 6k** | 248.80 @ 5k | 133.43 @ 5k |
| **Separate R16 DD** | 1733.90 @ 2k | 664.95 @ 6k | 265.51 @ 6k | 140.84 @ 2k |

**Headline:** FD is the consistently strongest design. Separate R16 FD is best at 4 and 8 steps; Basic FD is best at 16 steps and is effectively tied with Separate R16 FD at 32 steps (133.42 vs. 133.43). The extra capacity of Separate R16 does not produce a consistent advantage.

These are **best observed** checkpoints among the three evaluated with the same 100 samples, so this checkpoint-minimum table is exploratory. The cleaner preselected comparisons keep the checkpoint chosen by the earlier 20-sample pilot fixed; those also confirm clear gains for FD at 8 and 16 steps, and for Separate R8 FD at 32 steps.

FF and DF were included in the initial 20-sample sweep but omitted from the 100-sample confirmation because they were generally weaker than FD/DD.

## What was run

| Experiment | Scope | Samples |
|---|---|---:|
| **Initial checkpoint sweep** | MDLM plus 8 CCF variants; seven checkpoints; 8/16/32 reverse steps | **171 configurations × 20 = 3,420** |
| **Independent 100-sample confirmation** | MDLM plus the 6 stronger FD/DD variants; three selected checkpoints at 8/16/32 steps; fresh seeds | **57 configurations × 100 = 5,700** |
| **Four-step follow-up** | MDLM plus the same 6 FD/DD variants and three checkpoints; fresh 4-step generations | **19 configurations × 100 = 1,900** |
| **Quality diagnostics** | Repetition-1/2/4, distinct-2/4, output length, and EOS position on the confirmation outputs | **Reused the same 5,700 samples; no new generation** |

**Main sweep total: 11,020 generated/scored samples.** The 100-sample confirmation uses new generation seeds; it tests sampling stability for fixed trained checkpoints, not stability across independent retraining runs.

Earlier exploratory screens used different, much smaller protocols and should be kept separate:

- **1,000-step screen:** 28 configurations × 5 samples = **140 samples**. Best observed CCF PPL was 73.13; MDLM was 32.78.
- **500-step screen:** 37 configurations × 2 samples = **74 samples**. Best observed CCF PPL was 54.07; MDLM was 45.25.

Those early 2–5 sample estimates were useful for debugging and checkpoint screening, but they are too small for headline comparisons.

## Short glossary

- **PPL:** GPT-2-large generative perplexity on generated text, scored through the first non-leading EOS. **Lower is better.** It is an external fluency score, not training loss or held-out MDLM likelihood.
- **Steps:** reverse diffusion transitions used to generate a sequence. Final cleanup is enabled, so the requested maximum call budget is `steps + 1`. MDLM normally executes the cleanup call; CCF can skip it when no masks remain.
- **4k / 6k checkpoint:** the CCF head after 4,000 / 6,000 training updates. This is unrelated to generation-step count.
- **n = 100:** 100 generated sequences in that table cell. These are generation seeds from one trained checkpoint, not 100 independently trained models.
- **K = 128:** token candidate-set size used throughout these main experiments. This is unrelated to the 4/8/16/32 reverse-step sweep.
- **Rep-n / Distinct-n:** repetition and lexical-diversity diagnostics. Lower repetition and higher distinct-n are usually preferable, but neither alone proves better text.

One important quality caveat: lower PPL did not always mean less repetition. For example, at 16 steps Basic FD 6k improves PPL from **313.78 to 244.05**, while EOS-aligned repeated 4-grams rise from **0.278% to 0.776%** and distinct bigrams fall from **76.45% to 71.97%**. PPL, repetition, diversity, and human inspection should therefore be considered together.
