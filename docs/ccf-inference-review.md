# Four-arm inference: restore now, review later

Requested 2026-09-21: restore inference, test it, and commit this increment.
Manual architecture review remains pending; a passing smoke test is not an
experiment-quality or dataset-integrity certification.

## Run a full training checkpoint

Use the same arm and architectural overrides that produced the checkpoint.
From the repository, in the GPU environment used for training:

```bash
python main.py +experiment=ccf/static_static mode=sample_eval \
  eval.checkpoint_path=/absolute/path/to/full-training.ckpt \
  sampling.steps=1000 sampling.num_sample_batches=1 \
  loader.eval_batch_size=1 eval.compute_generative_perplexity=false
```

Replace `static_static` with `fixed_dynamic`, `dynamic_fixed`, or
`dynamic_dynamic` as appropriate. This command samples; it does not train or
load a dataset. Supply `data.tokenizer_name_or_path` if using a local GPT-2
tokenizer cache. All four presets explicitly select `structured_joint` and
`sampling.predictor=ddpm`. The ordinary frozen-backbone control is selected
with `model.structured_decoder.sampling.mode=factorized`; it bypasses CCF.

A full Lightning checkpoint supplies both the backbone and trained CCF head.
The loader skips loading the original separate backbone initialization file
and strictly restores the full state. Missing head weights and a topology or
factor arm mismatch are errors. The model constructor still requires a
pretrained backbone for new training unless an explicit smoke-test override
is used. This increment does **not** restore historical adapter-only
`.safetensors` loading, CRF generation, marginal ablations, or CCF semi-AR.

Generative perplexity is optional; enable
`eval.compute_generative_perplexity=true` to use the configured external
evaluator. With it disabled, that evaluator/tokenizer is not downloaded and
no perplexity is reported. Generated samples and metrics are separate from
the conditional denoising training loss.

## Review in this order

- [ ] `diffusion.py::_structured_clean_sample`: the CCF head sees frozen
  features, unary logits, the current noise level, and the mask of currently
  hidden positions. It receives no clean targets or training-only teacher
  information. `sample_structured_tokens` uses the retained joint forest
  sampler and residual-state expansion. Already revealed tokens stay fixed.
- [ ] `_structured_ddpm_update`: draw joint clean identities first, then
  draw the original independent Bernoulli reveal mask from the absorbing
  schedule. This order matters for seeded reproducibility. There is no
  confidence-ranked reveal, rejection sampling, or sample reranking.
- [ ] `_sample`: dispatch to that update and use another joint draw to
  remove remaining masks. Do not substitute ordinary MDLM argmax at the end.
- [ ] `restore_model_and_sample`: put the head and backbone in evaluation
  mode, then restore the caller's modes, including on failure. The frozen
  backbone stays frozen/eval. This corrects the historical unconditional
  return to training mode; sampling arithmetic is unchanged.
- [ ] `main.py::_load_from_checkpoint`:
  strict full-state load, matching arm, no dependency on the old external
  initialization file. Review architecture overrides before real experiments.
  The arm comparison reads the saved configuration before Lightning applies
  runtime overrides; this adds a checkpoint read at startup, not per step.
- [ ] Configuration/evaluation: record checkpoint, arm, seed, token length,
  precision, sampling steps, evaluator, and sample count for every real run.
  The example emits one batch to avoid the legacy CLI's last-batch-only
  text return. A complete benchmark/export workflow remains separate.

## Provenance and unchanged computations

The joint draw/reveal/final-denoising path is restored from
[g-experiments at 2051502](https://github.com/WeibingZhang04/mdlm-fork1/blob/2051502329429a252d3b806e0ed195ff379c42b6/diffusion.py),
particularly `_structured_clean_sample`, `_structured_ddpm_update`, and
`_sample`. Unused marginal/CRF/confidence-gating branches were omitted.
The head, forest inference, residual expansion, training loss, corruption,
teacher, optimizer, and dataloader are unchanged by this increment.

The inherited MDLM backbone/schedule attribution remains
[Sahoo et al., Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524).
Restoring these functions is implementation reuse, not a new sampling
algorithm. Existing licensing/attribution review items remain open.

## Still to verify or decide

- [ ] Review this increment manually after returning to architecture work.
- [ ] Run full-size, sufficiently sampled evaluation of actual trained
  experiment checkpoints. Tiny or short smoke runs give no quality claim.
- [ ] Resolve/document the previously observed outer BF16 training versus
  sampling precision difference. This increment preserves those contexts;
  it does not silently fix or waive the known cross-precision failure.
- [ ] Separately establish dataset splits/provenance and exact training
  resume/private-RNG semantics; inference smoke tests cannot establish them.
- [ ] If old adapter-only artifacts must be evaluated, restore and test
  their loader as a separate increment before reporting comparisons.

## Verification recorded 2026-09-21

- **32 historical parity fixtures passed:** actual historical/restored
  generation methods, each branch's actual head/objective/sampler, synthetic
  frozen backbone features on CPU. All intermediate tokens, final samples,
  and RNG states matched exactly (four arms, K=17/128, two seeds, final mask
  removal enabled/disabled). The reveal-update function also matches the
  historical function's syntax tree exactly.
- **109 GPU tests passed**, one unchanged distributed-metric test deselected,
  Slurm **1556874**, RTX 6000 Ada, torch 2.2.2+cu121. Includes all-arm real DiT
  sampling, exact seeded reference trajectories, training-only input boundary,
  frozen-backbone/lifecycle checks, strict full checkpoint restoration,
  rejection of both wrong arm modes and missing head weights, optional PPL
  control, and existing training/optimizer/teacher/diagnostic regressions.
  The previously recorded cross-precision failure was outside this selected
  suite and remains unresolved.
- **Four-arm training/checkpoint/CLI smoke passed**, Slurm **1556870**:
  released OWT backbone, synthetic text fixtures, two optimizer updates per
  arm, then real `main.py` generation (B=1, L=32, four sampling steps).
  All backbone state hashes stayed equal to the released backbone; every
  head updated and strictly reloaded. No generative PPL was requested.
- Failed attempts are retained: **1556868** ran zero tests because a test
  filename was absent on the server. **1556873** passed 104 tests but exposed
  the original arm-check placement error and one test-environment error.
  The arm check now runs before Lightning replaces saved hyperparameters;
  deterministic CUDA tests run with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  No tolerance was relaxed or failing assertion removed.

The new standalone checks can be rerun with:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest -q tests/test_ccf_generation.py
```

Use the CUDA/FlashAttention environment required by this repository. Without
CUDA, GPU-specific cases skip; that is not equivalent to the GPU verification.

Complete evidence, final CLI rerun results, and the commit identifier are in
the shared `problems_revisit/2026-09-21-inference-restoration-01a0b6bf.md` note.
