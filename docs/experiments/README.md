# CCF generation experiments

Compact registry for the scripts and saved results in this PR. Snapshot: **2026-09-17 20:51:28 EDT**.

## Labels and protocol

The first arm letter is topology and the second is factor scoring: **SS** = static/static, **SD** = static/dynamic (`fixed_dynamic` in code), **DS** = dynamic/static (`dynamic_fixed`), and **DD** = dynamic/dynamic. Shared/separate is the endpoint embedding layout; R8/R16 is interaction rank.

Recent evaluations use length 1,024, batch 1, CCF K=128, paired generation seeds, and GPT-2-large FP32 scoring through the first nonleading EOS. PPL is `exp(total NLL / scored tokens)`. CCF uses the verified level-batched implementation; MDLM uses the factorized harness. Reverse transitions reserve one cleanup call, so requested NFE is steps + 1.

## Primary 100-sample results

Each checkpoint below was fixed by the earlier 20-sample pilot before the 100 fresh confirmation samples were scored. Checkpoint is in parentheses.

| Model | 4 steps | 8 steps | 16 steps | 32 steps |
|---|---:|---:|---:|---:|
| MDLM | 1946.88 | 815.06 | 313.78 | 162.56 |
| Shared R16 SD | 1623.36 (6k) | 606.69 (6k) | 244.05 (6k) | 146.32 (2k) |
| Shared R16 DD | 1667.87 (3k) | 619.00 (3k) | 298.96 (7k) | 152.04 (5k) |
| Separate R8 SD | 1632.05 (6k) | 625.03 (6k) | 248.82 (5k) | 135.74 (6k) |
| Separate R8 DD | 1680.80 (2k) | 687.45 (2k) | 276.08 (6k) | 153.37 (2k) |
| Separate R16 SD | 1582.27 (6k) | 605.85 (6k) | 248.80 (5k) | 150.77 (3k) |
| Separate R16 DD | 1749.49 (6k) | 664.95 (6k) | 282.95 (2k) | 159.24 (3k) |

All **76/76 configurations** completed: 19 at four steps and 57 across 8/16/32 steps, for **7,600 fresh samples**. All CCF gates passed; no unresolved masks remain. The exact three-checkpoint tables are in `results.json`.

SS and DS were not promoted to 100-sample confirmation. Their latest checkpoint minima use 20 pilot samples and remain exploratory:

| Model | 8 steps | 16 steps | 32 steps |
|---|---:|---:|---:|
| Shared R16 SS | 892.74 (1k) | 320.86 (6k) | 178.90 (3k) |
| Shared R16 DS | 835.09 (5k) | 351.65 (5k) | 197.69 (7k) |

At 16 steps, Shared R16 SD 6k improves PPL from **313.78 to 244.05**, while EOS-aligned repeated 4-grams increase from **0.278% to 0.776%** and distinct bigrams fall from **76.45% to 71.97%**. Lower evaluator PPL therefore does not establish better overall text quality.

At 500 steps with two samples per cell, MDLM scores **45.25** and the best observed CCF scores **54.07**. At 1,000 steps with five samples per cell, the corresponding values are **32.78** and **73.13**. These small-sample minima are descriptive.

## Short experiment registry

| Stage | Purpose | Main scripts | Saved scope |
|---|---|---|---|
| Training | Basic four arms through 10k | `train_four_ccf_matched_1k.sh`, `continue_four_ccf_matched_*` | Shared R16 SS/SD/DS/DD |
| Training | Separate endpoint ranks | `train_ccf_separate_7k.sh`, `train_ccf_separate_rank16_7k.sh` | Separate R8/R16 SD/DD through 7k |
| Pilot | Checkpoints at 8/16/32 steps | `evaluate_ccf_selected_five.py`, `evaluate_ccf_*steps*.sh` | 171 cells, 3,420 samples |
| Confirmation | Fresh 100-sample evaluation | `evaluate_ccf_confirmation.py`, `evaluate_ccf_confirmation.sh` | 76 cells, 7,600 samples |
| Longer screens | 500 and 1,000 steps | `evaluate_ccf_500_steps.py`, `evaluate_ccf_selected_five.py` | 33×2 and 28×5 samples |
| Quality | Repetition, diversity, EOS and length | `analyze_ccf_quality_metrics.py` | Reuses the 5,700 8/16/32 outputs |

## Minimal setup

Activate the repository environment, then provide only local roots:

```bash
export CCF_CODE_ROOT=/path/to/checkout
export CCF_CACHE_ROOT=/path/to/checkpoints-cache
```

Run `evaluate_ccf_confirmation.py --inventory` with a selection file and its SHA-256 before any GPU submission. The frozen selections are `experiments/confirmation-4.json` and `experiments/confirmation-8-16-32.json`. Each CCF cell performs a same-runtime first-sample token/NFE/RNG gate. Check for existing completed work before launching anything.

For the next agent: keep pilot seeds 91001–91020 separate from confirmation seeds 100001–100100, do not count the gate duplicate as another scored sample, and do not interpret generation seeds as independent training runs. Record checkpoint/source hashes and actual NFE for every new result.

## Attribution

The backbone and factorized baseline derive from Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (NeurIPS 2024). Tree inference uses standard sum-product; forest construction uses a constrained greedy Kruskal variant. The categorical helper adapts PyTorch 2.2.2's single-sample multinomial implementation; its license is retained under `docs/third-party/`. Distinct-n follows Li et al. (NAACL 2016). Level batching is an implementation optimization and is not presented as a new sampling algorithm.

Model weights, raw text, machine paths, hostnames, accounts, scheduler identifiers, notification addresses, and credentials are intentionally excluded.
