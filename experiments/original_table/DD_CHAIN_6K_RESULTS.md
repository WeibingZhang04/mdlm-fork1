# Continuous Basic DD 6k: results and code provenance

Audited on 2026-09-24 from completed study artifacts. PPL is GPT-2-large generated-text perplexity (lower is better), with 100 scored length-1024 sequences per cell, seed block 100001, and scorer revision `32b71b12589c2f8d625668d2335a01cac3249519`. The scoring policy retokenizes decoded text through the first nonleading EOS. These are evaluations of one training seed, not independent training replications.

## Source version to associate with these results

All eight study directories named below record the same source base commit, `e9898bafcd979e0f3db5cd6248555ef03bd01255`, plus an uncommitted source snapshot. Their `study.json` files contain identical `source_identities` maps (528 files), identical protocol SHA-256 `227574ce24f8eba2b5f435c3072c317d2c30d35797bdb904e116df6c4935143c`, and source diff SHA-256 `5de66b25bbb900f76c3b47d120343c6f4e82dfc867431ac2aa0dadfb6fdbcf02`. The canonical JSON SHA-256 of each `source_identities` map is `afad7a7b4cbb928bb964ec5fc23a59cf82b0c2e1e2a6b125a5c2cb51f5a1749f` (sorted keys, compact separators).

**The nine code files staged alongside this report match their recorded study SHA-256 identities byte for byte.** The other 519 recorded source files also matched the server checkout at audit time. Thus the planned commit will preserve the base code content used for these runs, assuming those staged files are committed unchanged. The jobs ran from the *uncommitted* snapshot above; they did not run from the future commit hash. This report and the four archived control scripts below were not part of that 528-file base snapshot.

The staged base code files are `evaluation/fixed_group_sampling.py`, `evaluation/generation_harness.py`, `experiments/original_table/run.py`, `experiments/original_table/run_dd_chain_6k.py`, `experiments/original_table/train_dd_chain_6k.sbatch`, `models/structured_decoder.py`, `scripts/launch_dd_chain_6k.sh`, `scripts/run_generation_pilot.py`, and `tests/test_generation_harness.py`. The active-chain proposal algorithm was already in parent commit `3828fd9` and the BF16/FP32 rotary switch in `2417ea4`; the staged files add the continuous runner and edge-source diagnostics around them.

Study paths below are relative to `$CCF_CACHE_ROOT/runs/`. Each `study/evaluation/confirmation/<cell>/generation/summary.json` contains the full-precision score, and the neighboring `adapter.manifest.json` identifies the source checkpoint. Checkpoints, adapters, generated samples, logs, and the GPT-2-large weights remain outside Git.

## Continuous 0→6k run: same 6k checkpoint, different rotary generation paths

The new Basic DD model trained continuously from step 0 to 6,000 with seed 1, `trainer.precision=bf16`, and `model.rotary_cache_precision=bf16` (normal BF16 rotary path). The 6k checkpoint SHA-256 is `0334e0ecaf04c70783182e75f1f2e78f7e825054b58ea4ced50ec8b33cba6bed`; its extracted adapter SHA-256 is `07500393761d6d5c721f56c45d2e8e9efe961fd70b8e7c35fc20bfe06fb68f94`. Both rows below evaluate that same checkpoint and adapter with chain proposals enabled.

| Training rotary path | Generation rotary path | 8 steps | 16 steps | 32 steps | Study |
|---|---|---:|---:|---:|---|
| BF16 | BF16 | 2236.82 | 1533.39 | 1097.22 | `dd_chain_seed1_6k.PJQmhb/study`, cells 000–002 |
| BF16 | FP32 | 692.19 | 271.24 | 155.87 | `dd_chain_ckpt_fp32_eval.zSUokc/study`, cell 000; `dd_chain_ckpt_fp32_sweep.VCTb2Y/study`, cells 009–010 |

The FP32 row is an evaluation-time rotary override of the BF16-trained checkpoint, not a separately trained model. The BF16 8-step cell ran on an RTX 6000 Ada on `watgpu108`, while the FP32 8-step cell ran on an L40S on `watgpu308`. That direct comparison changes GPU model as well as the rotary path; it cannot by itself isolate a precision effect. “BF16 rotary” also means the current `models/dit.py` rounds rotary phases before cosine/sine, not merely that it stores an otherwise FP32-computed cache in BF16.

## FP32 checkpoint sweep of the continuous run

These are checkpoints from the *same* training trajectory, evaluated with FP32 rotary generation. The 6k 8-step result comes from `dd_chain_ckpt_fp32_eval.zSUokc/study`; the other cells come from `dd_chain_ckpt_fp32_sweep.VCTb2Y/study`.

| Training step | Checkpoint SHA-256 prefix | 8 steps | 16 steps | 32 steps |
|---:|---|---:|---:|---:|
| 1k | `07158652` | 703.32 | 311.05 | 164.33 |
| 3k | `7004e713` | 687.37 | 263.94 | 143.91 |
| 5k | `3bdf57f1` | **669.35** | **237.05** | **140.18** |
| 6k | `0334e0ec` | 692.19 | 271.24 | 155.87 |

The lowest PPL among these observed checkpoints is at 5k for every generation budget. The per-cell adapter manifests retain the full checkpoint hashes.

## 8-step control: when rotary values are rounded

This control uses the **historical** Basic DD 6k checkpoint, SHA-256 `dce92574b4a685a62b26a5c4fb6556882e2650b01256b76eea7c5289b6826a11`, with chain proposals on. All three evaluations ran on RTX 6000 Ada GPUs on `watgpu108` with the same checkpoint, paired seeds, and scorer. The FP32 run had a different GPU allocation from the two BF16 runs.

| Rotary calculation | PPL at 8 steps | Study |
|---|---:|---|
| Normal BF16: round phases before trig | 2048.02 | `dd_old_ckpt_new_proposals.sFSVjG/study` |
| FP32 phases and trig | 634.75 | `dd_old_ckpt_fp32_chain_on.drEhrn/study` |
| FP32 phases and trig, then cast cosine/sine to BF16 | 634.75 | `dd_old_ckpt_posttrig_bf16.fV8A9e/study` |

The last two generated token arrays and per-sample reference scores matched exactly for all 100 samples; neither matched the normal BF16 token array. This points to phase rounding *before* trig as the likely source of the large 8-step difference on the historical checkpoint. It does not determine the exact contribution for the new checkpoint across different GPU models, nor at 16/32 steps. The post-trig run used the study-local program archived byte for byte as `dd_chain_6k_controls/evaluate_posttrig.py` with its batch script.

## 8-step chain-proposal inference ablation

These normal-BF16 runs used RTX 6000 Ada GPUs on `watgpu108`, paired seeds, and the same checkpoint within each on/off pair. The chain-off runs masked chain proposals only during inference using the study-local `ablate_chain.py`, archived byte for byte with its batch script under `dd_chain_6k_controls/`. The new checkpoint on/off jobs had different GPU allocations on that node.

| Checkpoint | Chain off PPL | Chain on PPL | Change | Studies (off; on) |
|---|---:|---:|---:|---|
| Historical 6k (`dce92574`) | 2134.94 | 2048.02 | −86.93 (−4.1%) | `dd_old_ckpt_chain_off.9eDwAl/study`; `dd_old_ckpt_new_proposals.sFSVjG/study` |
| New continuous 6k (`0334e0ec`) | 2301.96 | 2236.82 | −65.14 (−2.8%) | `dd_chain_inference_ablation.4vwUP7/study`; `dd_chain_seed1_6k.PJQmhb/study` |

The chain-on runs selected chain edges for 11.6% (historical) and 11.1% (new) of recorded edge events. These percentages describe selected edges, not causal contribution to PPL. Earlier FP32 on/off numbers (historical 665.49→634.75; new 700.88→692.19) changed GPU model within each pair and should not be read as isolated chain effects.

## Archived control scripts and their original identities

The four files under `dd_chain_6k_controls/` are byte-for-byte copies of the programs and Slurm scripts that ran from their study directories; archiving them does **not** retroactively make their study runs originate from a Git commit. Their SHA-256 values let a later reader verify the archive against the original studies:

| Archived file | SHA-256 | Original study |
|---|---|---|
| `ablate_chain.py` | `597b3184accd3b831c9bfbcc6bae4e4ecdd241b71425649899a45a39b23b0e0a` | Both `dd_old_ckpt_chain_off.9eDwAl` and `dd_chain_inference_ablation.4vwUP7` |
| `evaluate_ablation.sbatch` | `4a2d07659b9857e30346fb6d90999de77f4474c931f9a11c224127457f4389a1` | Both chain-off studies |
| `evaluate_posttrig.py` | `3ff5090b3c5e65995d16cc1ceb60f5e568ea2c0fe239d5c2a95202a4d76c6836` | `dd_old_ckpt_posttrig_bf16.fV8A9e` |
| `evaluate_posttrig.sbatch` | `b2e973aebd249be394c2fd5a27ac0b10188cf2754c770da968dee0eb533c3b5d` | `dd_old_ckpt_posttrig_bf16.fV8A9e` |

The archived scripts retain their original server paths and were not rewritten as portable launchers. Study metadata, Slurm allocation records, and result files remain in the named study directories. Before citing a later Git commit as this run's source version, verify that it contains these staged byte identities; do not replace the recorded run-time `source_commit` with that later commit hash.
