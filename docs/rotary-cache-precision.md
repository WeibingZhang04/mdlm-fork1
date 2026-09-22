# Rotary-cache precision switch

On `original_table_base_crf-recovery`, `model.rotary_cache_precision` selects
`bf16` (default) or `fp32`. This controls rotary phases and cosine/sine caches,
not all model arithmetic. Keep the usual BF16 mixed trainer setting.

## Existing FD/DD 6k launcher

```bash
# DO NOT touch other people's files, jobs, or processes.
cd /u401/n23zhang/clean_tree_mdlm/mdlm-fork1
unset CCF_STUDY  # prepare a fresh output folder
CCF_ROTARY_CACHE_PRECISION=bf16 bash scripts/launch_original_table_fd_dd_6k.sh
# Or use fp32 instead of bf16 for a separate study.
```

Omitting the variable selects BF16. Preparation validates the value and saves it
in `study.json`. Each training phase gets that recorded Hydra override, so later
shell environment changes cannot alter queued jobs. Invalid values fail before
submission. This does not change the optimizer, loss, data, topology, step
counts, phase resumes, or sample counts. For this launcher, evaluation is
explicitly FP32 rotary for BOTH training choices, preserving the historical
inference policy and isolating training precision. The saved study also records
`eval_rotary_cache_precision: fp32`.

## Direct training and generation

Append `model.rotary_cache_precision=bf16` or
`model.rotary_cache_precision=fp32` to your existing `python main.py ...` command,
retaining its data/checkpoint/other overrides. The contextual-forest-small config
declares this field. Other DiT configs use the same BF16 fallback; use
`++model.rotary_cache_precision=fp32` to add/override the field with Hydra.
Direct generation calls also honor this setting: explicitly hold generation
precision fixed across models when comparing training precision.

BF16 explicitly enters BF16 autocast for the frequency einsum and rounds phases
BEFORE cosine/sine. FP32 explicitly disables autocast with FP32 inputs. A startup
probe outside training autocast therefore cannot silently choose the cache.
Cache reuse checks length, dtype and device. The `inv_freq` state-dict key is
unchanged. The switch is a construction/config choice, not a schedule for
changing precision midway through training.

Old configs/checkpoints without the field now fall back to BF16. Explicitly
choose FP32 to reproduce the previous FP32 ablation or historical FP32 generation.
Do not interpret this switch as converting the whole model to FP32. Avoid
editing a checkout while queued/running jobs depend on it. The running 15k-to-30k
jobs use `mdlm-g-experiments-30k`, a separate worktree, and are unaffected.

## Validation and provenance

`python -m unittest -v test_rotary_cache_precision test_dit_sdpa_fallback`
checks config/default/invalid choices, probe-first versus train-first equality,
cache rebuilding, unchanged state-dict keys, attention gradients, and exact
CUDA agreement with historical BF16 and FP32 computations. Set
`ROTARY_REQUIRE_CUDA=1` to require GPU checks rather than skip them.

This extends the repository's existing MDLM `models/dit.py` implementation;
see `CITATION.cff` for upstream attribution. No external code or dependencies
were introduced. The historical FP32 ablation record remains in
`experiments/original_table/provenance.json`, alongside the new switch record.

Validation on 2026-09-22: GPU job `1560550` passed all eight tests (including
the CUDA reference, no skips). Source-only preparation separately verified
omitted/BF16/FP32 choices, rejected FP16, and checked every phase override
after changing the environment. Logs and metadata evidence are in
`/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/rotary_switch_tests_20260922T190805Z`.
Only this regression test was submitted; no new training campaign was launched.
