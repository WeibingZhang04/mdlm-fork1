# CCF regression: checkpoint, sampler and scoring audit

This is a predeclared diagnostic study. It preserves all existing positive and negative results. No training architecture or stored checkpoint is changed. It supplements the already-running cache-precision continuation study, job 1557320.

## Questions and controls

1. Does the old checkpoint still beat MDLM with a fresh seed block and the identical GPU/scorer used for fresh checkpoints? Evaluate Basic FD and DD at old 6k, fresh 6k and fresh 15k. Comparing 6k with 6k separates training duration from the changed training process. Report every case, not a best-checkpoint selection.
2. Does a learned head help beyond its sampler and numerical precision? Include native MDLM, native MDLM with FP32 logit normalization (an explicitly labeled numerical control), and the existing neutral-factor ablation with both joint and independent-marginal sampling. The neutral ablation sets pair potentials to one while retaining the candidates, graph and structured sampling implementation. It is not a newly trained model. Also compare old/fresh6k FD joint versus marginal sampling.
3. Did scoring or EOS behavior create the apparent advantage? Independently recompute the original GPT2-large token losses for every generated sample and require agreement with the saved primary score. Add fixed-prefix256 and full1024 scoring that ignore EOS termination, explicitly secondary sensitivity analyses; never replace the original score. Retain all samples, repetition1/2/4, distinct2/4, actual NFE and EOS/length statistics.
4. What does the head learn? On the same first16 WikiText validation examples and four fixed masks (.25/.5/.75/.95), measure joint/marginal/backbone NLL, marginal KL from the backbone, entropy, explicit-candidate mass and graph statistics for all six checkpoints. These are validation diagnostics, not an untouched benchmark. Clean tokens are used only in scoring, never fed to the head at masked positions.

## Fixed generation protocol

100 samples per cell, fresh seed250001, length1024, denoising steps8/32, same candidateK128 and original NFE accounting. All generation cases run serially on one generic GPU allocation; no GPU-model or H200 pin. Positional cache is FP32 in every generation case, asserted at runtime. Original fixed GPT2-large revision, FP32 scoring and EOS policy remain the primary metric. The neutral-factor and FP32-normalization controls are separately labeled in saved artifacts. No ground-truth text is supplied during unconditional generation.

Conditional checks gate generation: on real GPU outputs, neutral factors must recover backbone probabilities and per-token joint NLL within1e-4, and FP32-normalized native MDLM probabilities must match the structured backbone unaries within1e-6. Paired checkpoint comparisons require matching input hashes/masks and generation seed IDs. Report token-weighted PPL and paired descriptive confidence intervals; no outcome is dropped. Multiple controls and a single training seed limit claims of novelty or general efficacy.

## Execution

Run `deploy.py` once to create a unique directory under this campaign's own cache and submit the conditional/identity gate, then generation plus rescoring dependent on that gate. Existing runs, active helpers and source snapshots remain unchanged. Keep failed attempts and create fresh paths for repairs. Finish this audit and the cache-precision causal study before interpreting the PPL reversal as a property of the architecture.
