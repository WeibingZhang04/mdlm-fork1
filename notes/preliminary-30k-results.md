# Preliminary 30k comparison results

All four configurations completed 200 unconditional 1024-token samples with 16 matched reveal steps. Denoising uses the same 32 development rows and fixed masks. These are exploratory measurements from one draw set, not final test results.

Run: `/u401/n23zhang/rework-data/runs/draft-full-scale-training/preliminary_30k_2026_10_01_14_34_01_UTC/`.

## Generation and diversity

| Metric | Baseline | Counts | Global 30k | Contextual 30k |
|---|---:|---:|---:|---:|
| GPT-2-large NLL, nats/token | 5.8557 | 5.6544 | 5.5107 | 5.4782 |
| GPT-2-large perplexity | 349.24 | 285.55 | 247.32 | 239.42 |
| Token entropy, nats | 7.5700 | 7.5934 | 7.5114 | 7.4122 |
| Unique token IDs | 25170 | 25020 | 24537 | 24422 |
| Distinct 2-grams, percent | 68.32 | 67.20 | 64.04 | 63.94 |
| Distinct 3-grams, percent | 92.74 | 92.25 | 89.51 | 88.53 |
| Distinct 4-grams, percent | 98.22 | 98.04 | 96.75 | 96.08 |
| Within-sample repeat 2-grams, percent | 7.02 | 7.10 | 9.42 | 10.61 |
| Within-sample repeat 3-grams, percent | 1.61 | 1.70 | 2.63 | 3.22 |
| Within-sample repeat 4-grams, percent | 0.24 | 0.27 | 0.47 | 0.70 |
| Generation seconds/sample | 0.188 | 5.103 | 5.220 | 5.135 |
| Generated tokens/second | 5438.31 | 200.67 | 196.16 | 199.41 |
| Generation seconds, all samples | 37.66 | 1020.58 | 1044.06 | 1027.04 |
| Backbone seconds, all samples | 34.69 | 35.80 | 36.26 | 36.56 |
| Sampling seconds, all samples | 2.84 | 984.63 | 1007.64 | 990.32 |
| Backbone calls/sample | 16 | 16 | 16 | 16 |

Distinct-n is unique n-grams divided by all n-grams across the generated collection. Repeat-n is the mean, across samples, of one minus the within-sample unique-n-gram fraction. Higher entropy/distinctness indicates diversity, not necessarily better text. Each method generated 204800 tokens; GPT-2-large scored 204600 next-token targets, excluding the first token of each sample consistently. Timing excludes model loading and external scoring.

## Joint denoising NLL

Negative log probability assigned to the complete correct masked-token assignment, divided by the number of masked tokens. Lower is better; units are nats per masked token.

| Mask rate | Baseline | Counts | Global 30k | Contextual 30k |
|---|---:|---:|---:|---:|
| 25% | 1.5707 | 1.5578 | 1.5723 | 1.5371 |
| 50% | 2.6268 | 2.5902 | 2.5766 | 2.5134 |
| 75% | 4.4084 | 4.3415 | 4.2403 | 4.1443 |
| 90% | 5.8366 | 5.7638 | 5.5722 | 5.4735 |

## Own marginal denoising NLL

Negative log probability of each correct masked token under the model's own marginal distribution, averaged across masked tokens. Lower is better; units are nats per masked token.

| Mask rate | Baseline | Counts | Global 30k | Contextual 30k |
|---|---:|---:|---:|---:|
| 25% | 1.5707 | 1.5735 | 1.6032 | 1.5808 |
| 50% | 2.6268 | 2.6298 | 2.6927 | 2.6671 |
| 75% | 4.4084 | 4.4129 | 4.4966 | 4.4690 |
| 90% | 5.8366 | 5.8392 | 5.9022 | 5.8781 |

## Shared candidate support and masking checks

All four completed denoising outputs have identical data checksums, mask rates, example counts, masked-token counts, underlying backbone NLL, candidate coverage, and retained mass.

| Mask rate | Masked tokens | Gold token in top 64 | Backbone mass in top 64 |
|---|---:|---:|---:|
| 25% | 8237 | 94.67% | 94.79% |
| 50% | 16367 | 87.79% | 88.40% |
| 75% | 24541 | 72.94% | 73.52% |
| 90% | 29511 | 58.51% | 58.58% |

Tokens outside the explicit top 64 remain representable through the residual category; coverage is not the fraction of targets included in the NLL.

## Interpretation and limits

Contextual has the lowest external GPT-2-large NLL and the lowest joint denoising NLL at all four tested mask rates. Global improves joint NLL at 50%, 75%, and 90% masking, but slightly worsens it at 25%. All three pair heads have worse own-marginal NLL than the backbone at every tested mask rate; this distinction matters because the training objective is joint NLL. Improved joint probabilities do not require improved individual-token marginals.

Both learned heads generate more repeated n-grams and lower-diversity outputs than the baseline on this draw set. Lower external perplexity therefore does not establish a general improvement in text quality. No causal attribution of the perplexity change to repetition has been established.

The comparison matches reveal steps, not runtime. The baseline uses the branch's matched random reveal schedule, not native ddpm_cache. Counts use all 220106 training rows, while each 30k learned head has processed 120000 sequence exposures. MAUVE, a native-sampler baseline, multiple seeds, and uncertainty intervals were not computed in this preliminary script.

Raw metric values are retained in `preliminary-30k-results.json` alongside this report. Original per-sample token records and scorer outputs remain in the remote run folder.
