# Exact historical BF16 rerun: training seed 2

This package preserves the executed study `legacy_bf16_seed2/study_qv4ipesi`, which produced FD5k PPL736.76 /278.98 /173.95 at8/16/32 denoising steps,100 samples each. Its matched MDLM values are830.35 /315.86 /161.64. This is the September21 rerun of the earlier setup, not the original September17 seed-1 table.

## What is exact

- `training_source/`: every source/artifact file from the actually executed isolated `99891f9441d9aef7082a088963ae2970bc992563` archive.
- `evaluation_source/`: every source/artifact file from the executed `ccf_confirmation_100.daiam2ji` snapshot, including its sampler optimizations and scorer.
- `submitted/`: the actual Slurm scripts, production Python entry points, saved phase configs and study/deployment records, unchanged.
- `helpers/campaign_common.py`: the actual argument-construction dependency, unchanged.
- `fd5k/`: the exact exported FD5k adapter, its manifest, generation arguments, original samples, scores and runtime manifest.
- `results/`: every completed5k/6k/7k evaluation's metadata and summary for all four arms. `provenance/` retains the gate history and DF preemption/retry information.

Each copied file has its original path and SHA256 in `manifest.json`. Git metadata, Python bytecode and pytest caches are not runtime source and are omitted. The original full checkpoints and pinned backbone/corpus stay in the user's cache. No failed attempt, score or sample is overwritten. The active jobs and old snapshots are untouched.

## Run it again

The original `submitted/train.sh` and `eval.sh` point to the old study and are evidence, not fresh-run commands. From this branch, prepare a **new, nonexistent** output directory:

```bash
python reproduction/seed2/prepare.py /u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/CHOOSE_A_NEW_NAME
```

This verifies every copied file and makes path-only study relocations. The production Python files and scientific settings remain byte-identical. Add `--submit` to submit a fresh GPU gate, all four seed-2 arms through8k, and100-sample evaluations at5k/6k/7k. Nothing is submitted without that flag. It uses generic one-GPU allocations and never modifies existing jobs.

To regenerate the quoted FD5k samples from the preserved adapter, inside a compatible Slurm GPU allocation with the original `mdlm` environment:

```bash
python reproduction/seed2/replay_fd5k.py /u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ANOTHER_NEW_NAME
```

The new gate checks relocated entry points. Its marker is separate from the original gate evidence. The rejected first observer test remains recorded; no added observer or FP32 startup probe is used in production. Original process restarts at1k/3k/6k, seed2, BF16 training,100001 generation seeds and original GPT2-large/EOS scoring are retained. The original `gate.py` is preserved but is not used as the portable gate because it reads archived failed-attempt checkpoints.

## Separate seed-1 reproducibility check

See `../same_seed_check/README.md`. That check answers whether the preserved original setup reproduces its old weights and PPL. It is distinct from the already completed seed-2 study. No architecture or loss changes were made for either package; only fresh output paths and orchestration are new.

Identical source/config/seed does not guarantee identical outputs across GPU types or nondeterministic kernels. Report actual differences; do not describe a short smoke as a full reproduction.
