# Original FD same-seed training and generation check

This is a new verification study, separate from the preserved original seed-1 experiment and the completed seed-2 rerun.

Prepare and submit one FD arm:

```bash
python reproduction/same_seed_check/prepare_check.py /u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/CHOOSE_A_NEW_CHECK_NAME --submit
```

It reconstructs and hashes the complete `12a4579` source tree from the preserved99891f9 files plus the17 original additions. It runs the actual original0→1k→3k→6k shell launchers with training seed1. Only working/output/resume directory lines change, and all replacements are recorded. No new startup forward, callback, precision flag, architecture, loss weight, optimizer setting or data order is introduced. The generic GPU allocation ignores the historical launcher's `#SBATCH` resource comments because the scripts execute inside the already obtained allocation.

After each phase, a separate process compares model tensors with the corresponding original checkpoint and extracts existing CSV logs. No training process is instrumented. Historical restart/reseeding behavior is retained.

Predeclared generation comparisons, each at8/16/32 denoising steps and length1024:

| Checkpoint | Samples per cell | Generation seed block | Purpose |
|---|---:|---:|---|
| 2k | 20 | 91001 | Check the original FD32-step pilot result near126.83 |
| 6k | 20 | 91001 | Check the original stronger8/16-step pilot checkpoint |
| 6k | 100 | 100001 | Check the larger confirmation sample |

For each row, regenerate both the archived original adapter and the newly trained adapter on the same allocation, with a matched native MDLM baseline. Keep original GPT2-large scoring/EOS policy, actual NFE, raw tokens, repetition/diversity/length metrics and every result. A separate one-sample evaluator smoke precedes training; it runs in a different process and cannot initialize the training cache.

`weights-*.json` reports exact tensor identity and difference magnitude. `generation-comparisons.json` reports PPL and exact sequence matches. A mismatch is evidence to investigate, not a result to discard. The original historical PPL may itself vary when regenerated on a different GPU; the paired controls help separate this from retraining differences. The same-seed check does not establish general superiority or eliminate checkpoint-selection uncertainty.
