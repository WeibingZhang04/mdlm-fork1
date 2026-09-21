# CCF 15,000-step replication and topology diagnosis

This folder versions the files used by the campaign launched on **21 September 2026**. It was added to `g-experiments` after launch so the experiment settings and diagnostics can be reviewed in the branch.

**Main training: eight arms, each with `trainer.max_steps=15000`.** The running jobs use model code pinned to commit **`2051502329429a252d3b806e0ed195ff379c42b6`**, plus the exact helper scripts copied here. Adding this folder does not change an active job or its settings.

## Start here

- [protocol.yaml](protocol.yaml): readable training, generation, GPU and diagnostic settings.
- [configs/resolved/](configs/resolved/): full resolved configurations for all eight primary arms. Search for `max_steps: 15000`.
- [configs/overrides/](configs/overrides/): exact Hydra argument lists passed to `main.py`.
- [configs/observed/](configs/observed/): configuration files copied from jobs that had started when this archive was made. Their resolved values match the reconstructed configs exactly.
- [helpers/train_arm.py](helpers/train_arm.py) and [helpers/campaign_common.py](helpers/campaign_common.py): launch code that applies the overrides.
- [campaign.json](campaign.json): source commit, job IDs, artifact locations and helper hashes.

The repository's generic `configs/config.yaml` keeps its defaults. This campaign sets its own values through Hydra overrides. The resolved files above make those values explicit without changing defaults for unrelated experiments.

## Primary experiments

| Array index | Arm | Embeddings | Rank | Training updates |
|---:|---|---|---:|---:|
| 0 | Basic FF (`static_static`) | shared | 16 | 15,000 |
| 1 | Basic FD | shared | 16 | 15,000 |
| 2 | Basic DF | shared | 16 | 15,000 |
| 3 | Basic DD | shared | 16 | 15,000 |
| 4 | Separate R8 FD | separate | 8 | 15,000 |
| 5 | Separate R8 DD | separate | 8 | 15,000 |
| 6 | Separate R16 FD | separate | 16 | 15,000 |
| 7 | Separate R16 DD | separate | 16 | 15,000 |

The first letter describes topology; the second describes factors. F means fixed form and D means context-dependent. FF was previously called SS. Fixed factors still have learned parameters.

All primary arms start fresh, use seed 1, train on pinned OpenWebText, retain the released frozen MDLM backbone, and use 1,024-token sequences with batch size 4 and top-K 128. Head learning rate is 3e-4. The joint-NLL coefficient is 1; the factorized auxiliary coefficient is 0; the topology coefficient is 0.1 for DF/DD and 0 for FF/FD. The attached paper's topology teacher uses clean tokens only to form training targets, never as generation inputs.

Validation runs every 500 updates. Full resume checkpoints are retained at 500-update intervals; portable head checkpoints are exported every 1,000 updates. The observer records loss components every 50 updates, gradient norms every 100, and a fixed four-example mask panel every 1,000. That small panel is a diagnostic, not an independent test set.

## Generation and scoring

Each checkpoint is evaluated at **4 / 8 / 16 / 32 denoising steps**. These differ from training updates. The requested NFE budgets are **5 / 9 / 17 / 33**; actual model calls are recorded separately.

- Curve: **20 samples per cell** at 1k, 2k, 3k, 4k, 5k, 6k, 7k, 10k, 12k and 15k; seed base 91001.
- Larger checks: **100 samples per cell** at preselected 7k, 10k and 15k; seed base 100001.
- Each arm's evaluation job includes a **same-GPU MDLM baseline** using the 100-sample seeds.
- GPT-2 large scoring uses pinned revision `32b71b12589c2f8d625668d2335a01cac3249519`, float32, and the same first-nonleading-EOS policy as the earlier experiments.
- Reports include repetition-1/2/4, distinct-2/4, output length and EOS positions alongside token-weighted PPL.

The source replay found matching old/new tokens and scores on the same GPU. Historical H200 results differ from the old-code replay on RTX 6000 Ada, so comparisons across hardware should not be attributed to code optimization alone. Fresh continuous training also differs from the earlier checkpoint-restart history. One training seed limits conclusions about training stability.

## Additional diagnoses

| Diagnosis | Design | Helper |
|---|---|---|
| Old/new correctness | 61 tests, real-data losses/gradients, generated tokens/NFE, larger scored replay | `gate.py`, `runtime_probe.py`, `legacy_replay.py` |
| Graph geometry | Six existing FD/DD 6k checkpoints; 16 validation chunks; three lengths; four mask rates; six graph interventions | `topology_diagnosis.py` |
| Topology weight | Basic DD, same 6k starting checkpoint; weights 0.03 / 0.10 / 0.30; 2,000 additional updates to 8k; joint/marginal generation controls | `weight_trial.py`, `weight_observer.py`, `teacher_probe.py`, `weight_eval.py` |
| Graph effects on generation | Basic DD and separate R16 DD; lengths 128/1,024; 8/32 denoising steps; 100 samples; native/fixed/active-chain-proposal graphs | `graph_generation.py`, `run_graph_intervention.py` |

The **3-step runs are smoke tests**, and **8k is the endpoint of the separate weight continuation**. Neither changes the primary arms' 15k target. Graph generation interventions retain checkpoint weights and record the override in an `intervention.json` sidecar. Graph tracing adds overhead; its timings are not performance benchmarks.

## Where the actual run lives

```text
/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/ccf_replication15k_20260921.wmjngfco/
  code/          # pinned model-code snapshot
  helpers/       # exact runtime helpers versioned here
  training/      # full checkpoints, loss logs, saved configs
  exports/       # portable adapter checkpoints and manifests
  evaluation/    # primary generation outputs and PPL
  diagnostics/   # conditional and generation graph diagnoses
  weight*/       # separate topology-weight training runs
  logs/          # Slurm stdout/stderr
```

Large model checkpoints, generated sample files and datasets remain there. `campaign.json`, saved requests and the audit records locate and identify the inputs; they are not silently omitted replacements or different checkpoints. The submitted GPU request is **generic `gpu:1`**, without an H200 constraint. Known unhealthy nodes are excluded. The primary arrays allow two concurrent tasks; diagnostic arrays allow one.

## Reproduce safely

The new convenience launcher creates a **fresh directory** and pins the original model commit. It requires the external backbone, old-code snapshot and old checkpoint paths listed in `campaign.json` to be available. It never reuses or overwrites the original campaign directory.

From the repository root, prepare files only:

```bash
python experiments/ccf_replication_15k_20260921/prepare_campaign.py
```

To prepare another fresh campaign **and explicitly submit its jobs**:

```bash
python experiments/ccf_replication_15k_20260921/prepare_campaign.py --submit
```

Preparation needs standard Python and Git. GPU jobs use the existing `mdlm` Conda environment, activated by the wrappers, with PyTorch 2.2.2/CUDA 12.1, Hydra 1.3.2 and OmegaConf 2.3.0. The backbone requires a compatible BF16/FlashAttention GPU. Do not force Hugging Face offline mode: the pinned streaming loader still needs a metadata lookup.

`prepare_campaign.py` was added for reproducibility and was **not** the launcher used for the historical job IDs. The actual wrappers are preserved under [slurm/](slurm/); exact original `sbatch` commands are in [provenance/slurm-accounting.psv](provenance/slurm-accounting.psv). Do not blindly resubmit the archived wrappers: they point at the original campaign. The historical submission command for the training array shows the first smoke job; its dependency was subsequently corrected to successful smoke job 1556958, without canceling other jobs.

## Analysis, evidence and reports

- [evidence/](evidence/): completed audit, smoke and diagnosis markers, plus numerical comparison evidence.
- [analysis/](analysis/): the original collection, analysis, plotting and report scripts, with a compact result snapshot taken at the time recorded in `provenance/captured_utc.txt`. `analysis/deployment.json` points to the live campaign.
- [outputs/](outputs/): the interim PDF, current topology explanation and figures. **They are not final 15k results.**
- [archive/orchestration/](archive/orchestration/): original submission and retry scripts, retained for provenance. They contain original workspace paths and are not the recommended rerun entry point. The failed first audit launcher used offline mode; the live wrapper and reproduction launcher use the corrected online behavior.

To rerun the saved-snapshot analyses from this folder, with their Python dependencies installed:

```bash
python analysis/analyze-topology.py
python analysis/plot-topology-mechanism.py
python analysis/make-report.py
```

To refresh the live snapshot from the original local environment with the configured SSH alias `n23zhangWatGPU`, run `python analysis/collect.py`. It is read-only. Reports use matplotlib and reportlab; inspect rendered PDF pages before sharing updates.

The paper reviewed was the user-supplied *Learning Token Dependencies in Masked Diffusion Language Models* (`diffusion_LM (9).pdf`, anonymous ICLR 2027 submission). Its source PDF is external; its relevant recommendations and the current evidence are summarized in the reports.

Adding these files changes neither another person's jobs/files nor the model code of the active campaign. Future changes to this branch also do not automatically enter already-running jobs.
