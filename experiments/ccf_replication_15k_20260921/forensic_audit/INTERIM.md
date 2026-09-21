# CCF regression investigation - 21 September 2026

**The PPL reversal is not explained yet.** A startup-monitoring bug changed training cache precision; matched causal training and generation controls are running. This is an interim diagnostic report, not a new efficacy claim.

## New completed check: conditional prediction

Same 16 WikiText validation examples, length1024, four fixed masks per example (.25/.5/.75/.95), all checkpoints evaluated under identical FP32 positional-cache initialization. These are validation diagnostics; not an untouched benchmark. Values are token-weighted NLL in nats per masked token, lower is better. They are **not GPT2 generation PPL**.

| Checkpoint | Joint NLL | Gain over backbone | Token entropy |
|---|---:|---:|---:|
| Frozen MDLM backbone | 4.66108 | - | 4.61885 |
| Old FD, 6k | 4.66614 | -0.00507 | 4.48862 |
| Fresh FD, 6k | 4.66109 | -0.00002 | 4.59257 |
| Fresh FD, 15k | 4.64957 | +0.01150 | 4.62488 |
| Old DD, 6k | 4.66487 | -0.00379 | 4.53427 |
| Fresh DD, 6k | 4.66140 | -0.00033 | 4.58609 |
| Fresh DD, 15k | 4.64980 | +0.01128 | 4.63820 |

On this panel, fresh15k heads improve conditional joint likelihood slightly, while old6k heads worsen it slightly. The old heads lower token entropy more strongly, producing more concentrated probabilities. Fresh15k FD joint-vs-own-marginal NLL gain is0.01755nats/token, versus0.00525 for oldFD. This supports investigating whether old generation-PPL gains came partly from sharpening, rather than stronger joint modeling. It does not establish the causal explanation of the PPL gap or overall text quality. Document-level paired bootstrap intervals are saved in forensic-audit/analysis.json and are descriptive.

Real-GPU controls passed for all six checkpoints and four masks: neutral factors recover backbone marginals (max absolute error5.4e-7) and per-token joint NLL (max error1.54e-5). Native MDLM with FP32 normalization matches structured-backbone probabilities exactly when identical normalization arithmetic is used. The first check attempt compared a fused log-softmax against subtract-logsumexp and failed at3.16e-6; original failure retained, corrected arithmetic passed without changing tolerances.

## Tests running

- **Cache cause, job1557320:** identical oldDD6k weights/optimizer/data, BF16 versus FP32 cache,2k further updates each; both evaluated on the same GPU with100 samples per cell at8/32 denoising steps. BF16 training finished; FP32 training running at the latest check. Paired startup tests passed.
- **Checkpoint and sampler audit, job1557350:** old6k, fresh6k and fresh15k FD/DD; native MDLM; MDLM with FP32 normalization; neutral-factor structured sampling.100 samples per cell, freshseed250001,8/32steps, all cases in one allocation. The conditional gate1557354 passed; generation is running.
- **Scoring audit, same job1557350:** recompute original GPT2-large scores independently from every sample and require agreement. Report fixed-prefix256 and full1024 scores that ignore EOS as secondary sensitivity checks, alongside repetition1/2/4,distinct2/4 and EOS/length metrics. Original scores remain primary and unchanged.

No training architecture or historical model files were changed. Neutral factors and FP32-normalized MDLM are disclosed inference controls. All outcomes will be reported; no checkpoint/sample filtering. The existing follow-up now explicitly tracks both regression studies before final reporting.
