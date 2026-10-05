# Preliminary 30k sampling comparison

The user approved comparing frozen released MDLM, the completed count model,
global at 30,000 updates, and contextual at 30,000 updates. This is a small
development experiment using the metrics from `scripts/sample_diagose_all.sh`.
The new entry point is `scripts/sample_diagose_preliminary_30k.sh`; the original
sampling script and active training jobs are unchanged.

## Checkpoint selection and preservation

Training writes `last.pt` every 500 updates and replaces `best.pt` when the
development loss improves. It does not retain numbered historical checkpoints.
Consequently, the requested 10k and 20k weights were not available, although
their development measurements remain in the logs. The user chose the retained
30k checkpoints for this preliminary comparison.

Before another training save could replace them, both actual 30k checkpoint
payloads were copied and their stored steps verified on CPU. These stable files
are in
`/u401/n23zhang/rework-data/runs/draft-full-scale-training/preliminary-checkpoints_2026_10_01_14_29_45_UTC/`:

- `global-step-30000.pt`, SHA256
  `f321ca07cb674fcf3d686a376e5ac46752fc8a91aaf3c1c42e738fa68c6ad518`.
- `contextual-step-30000.pt`, SHA256
  `e53c78a85aaeeeea39a77dd5673c4bd31dc93e83f9798a8d8f18a84cc4a08749`.

The same snapshot directory also preserves the available 33,500-update latest
checkpoints, but those are excluded from this comparison. Its `manifest.json`
records original paths, stored steps, checksums, and training identities.

Counts come from
`/u401/n23zhang/rework-data/runs/draft-full-scale-training/train_2026_10_01_05_46_13_UTC/counts/owt-counts.pt`.
The backbone is `/u401/n23zhang/rework-data/checkpoints/mdlm-owt.pt` and remains
frozen for every configuration.

## Sampling and measurement settings

- Four configurations, 200 unconditional samples each, length 1024, batch one.
- Preserve the old script's 16 reveal steps, temperature 1, joint sampling,
  dense inference, K=64 plus residual state, and one discarded warmup sample.
- Baseline uses the repository's matched random reveal schedule. It is not the
  released model's native `ddpm_cache` sampler; final sampler selection remains
  deferred. All configurations share preliminary draw IDs 1,000,000–1,000,199,
  separate from the earlier pipeline-validation draws.
- Counts use PMI with strength 0.1, as in the original sampling script.
- GPT-2-large scores saved raw token IDs using the existing pinned scorer and
  reports external generation NLL and perplexity. Generated text, exact IDs,
  entropy, distinct/repeated n-grams, and generation timing are also retained.
- Denoising uses the same first 32 rows of the new 1024-token development file,
  with the existing fixed mask draws at rates 0.25, 0.5, 0.75, and 0.9. It reports
  base, joint, and own-marginal NLL, candidate coverage, and retained mass.
- The reserved final-evaluation population is not used. These 200-sample
  measurements are exploratory and do not establish a final ranking.

## Execution and checks

The copied shell launcher submits a four-task Slurm array, at most two tasks
concurrently, with one RTX 6000 Ada GPU, two CPUs, 20 GiB host RAM, and a five-hour
limit per task on watgpu108. Slurm assigns available resources; no existing
jobs are modified or cancelled. The launcher checks input ownership and the
30k checkpoint checksums, creates a new UTC-dated output folder, preserves the
submitted script and snapshot manifest, and records input checksums and job ID.
Each evaluator additionally records its source, backbone, and head identities.

Shell syntax and mocked execution checks passed for all four configurations,
including generation/scoring flags, 32-row diagnostics, checkpoint paths, draw
IDs, and refusal to reuse an existing run folder.

Submitted as Slurm array `1580128` on 2026-10-01 at 14:34:05 UTC, with
baseline task 0, count task 1, global-30k task 2, and contextual-30k task 3.
The verified output directory is
`/u401/n23zhang/rework-data/runs/draft-full-scale-training/preliminary_30k_2026_10_01_14_34_01_UTC/`.
Baseline and count tasks started successfully; learned-head tasks initially
waited for the two-task array concurrency limit. Existing training continued.
