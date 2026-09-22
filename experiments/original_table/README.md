# Original CCF table: compact reproduction

The historical R8, R16 and evaluation changes are applied directly to the repository source on top of `8e1772f4687a78a48b6a330f9f095e9880540de2`. Slurm jobs run this checkout directly; no patch application or source reconstruction happens at runtime. It adds no complete source snapshots, datasets, checkpoints, generated samples, training logs, usernames or personal storage paths to Git. The base commit already contains the existing project; this package does not rewrite that pre-existing history.

## What is preserved

- Basic FF/FD/DF/DD: original BF16 training, seed 1, original full-checkpoint restarts at 1k, 3k and 6k, finishing at 10k.
- Separate R8/R16 FD/DD: original separate-endpoint implementations, fresh seed-1 BF16 training to 7k.
- The released, frozen MDLM backbone and pinned training-data configuration from the original Git tree; the backbone file hash is checked before preparation.
- Original per-phase training override order and values. No added startup forward, numerical probe or training callback. Each training phase runs in a fresh Python process.
- The historical evaluation implementation, including the level-draw sampler and first-sample token/NFE/RNG equivalence gate. Numerical sampler functions are extracted unchanged from the original audit modules; unrelated benchmark code is omitted.
- Pilot: seven checkpoints, all eight arms and MDLM, 8/16/32 denoising steps, 20 samples, seed block 91001. Total: 171 cells / 3,420 samples.
- Confirmation: the original three checkpoint selections for each FD/DD model/budget, MDLM, 4/8/16/32 steps, 100 samples, seed block 100001. Total: 76 cells / 7,600 samples. FF/DF were not in the historical 100-sample table.
- GPT-2-large revision, FP32 scoring, length 1024 and first-non-leading-EOS scoring policy. Budget passed to the sampler is S+1; actual NFE is retained in results.

`protocol.json` holds the shared overrides, phase differences and frozen checkpoint selections. `provenance.json` records historical content hashes; preparation checks that the combined source matches the applied revisions. Shared mode preserves Basic initialization; separate mode enables the historical R8/R16 heads. The optional conditioner remains disabled. `historical_reference.py` retains only the three earlier sampling functions required by the R8-DD first-sample gate. Large immutable historical runs remain outside Git. The old exploratory 500/1000-step generation screens are outside this compact table reproduction.

`report.md` preserves the historical report wording and table; only its audit link points to this README. `expected-table.csv` is the 28-cell historical reference, not a replacement for freshly scored output. The report is historical and does not incorporate subsequent regression findings.

## Prepare

Run from the repository root, in the original compatible `mdlm` Conda environment (historically Python 3.9, PyTorch 2.2.2+cu121, Transformers 4.38.2). This package does not silently install or upgrade dependencies.

Set local paths yourself; do not commit a filled-in environment file:

```bash
conda activate mdlm
export CCF_CACHE_ROOT="/absolute/path/to/your/existing/tree_mdlm_cache"
export CCF_STUDY="$CCF_CACHE_ROOT/runs/original_table_seed1_$(date -u +%Y%m%dT%H%M%SZ)"
export CCF_MAIL_USER="your-notification-email"
python experiments/original_table/run.py prepare --cache "$CCF_CACHE_ROOT" --study "$CCF_STUDY"
```

Preparation creates only study metadata, evaluation-cell lists and a logs directory outside the repository. Checkpoints, resolved configurations, generated samples and logs are written there by jobs. Source code runs directly from this checkout and is hash-checked before each job; do not edit it while the study is running. Preparation checks source and backbone identity without importing a model. `--source-only` validates source without requiring the backbone and deliberately prevents training/generation from that validation directory.

`run.py` is a command helper, not a scheduler: it expands the eight-arm protocol, preserves phase resumes, invokes training/evaluation, and collects results. The Slurm files schedule those commands. Keeping this logic in Python avoids duplicating settings in shell scripts. It no longer archives Git trees, applies patches, copies source, or creates runtime Git repositories. Submit from the repository root so `SLURM_SUBMIT_DIR` points to this checkout.

## Submit yourself

These are the submission commands; no job is submitted by `prepare` or by this package's Python code. They request any compatible single GPU, without a GPU-model or node restriction. Array concurrency is capped, and no job cancellation command is used.

```bash
MAIL_TYPES=ALL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50,ARRAY_TASKS
TRAIN_JOB=$(sbatch --parsable --mail-user="$CCF_MAIL_USER" --mail-type="$MAIL_TYPES" \
  --output="$CCF_STUDY/logs/train-%A_%a.out" --error="$CCF_STUDY/logs/train-%A_%a.err" \
  experiments/original_table/train.sbatch "$CCF_STUDY")
TRAIN_JOB=${TRAIN_JOB%%;*}
sbatch --array=0-170%4 --dependency="afterok:$TRAIN_JOB" \
  --mail-user="$CCF_MAIL_USER" --mail-type="$MAIL_TYPES" \
  --output="$CCF_STUDY/logs/pilot-%A_%a.out" --error="$CCF_STUDY/logs/pilot-%A_%a.err" \
  experiments/original_table/evaluate.sbatch "$CCF_STUDY" pilot
sbatch --array=0-75%4 --dependency="afterok:$TRAIN_JOB" \
  --mail-user="$CCF_MAIL_USER" --mail-type="$MAIL_TYPES" \
  --output="$CCF_STUDY/logs/confirmation-%A_%a.out" --error="$CCF_STUDY/logs/confirmation-%A_%a.err" \
  experiments/original_table/evaluate.sbatch "$CCF_STUDY" confirmation
```

The fixed historical confirmation selection allows both evaluation arrays to depend on training directly. They do not reselect checkpoints after inspecting new confirmation results. This distinguishes a replication from a new best-checkpoint search. Resource/time limits may be adjusted without changing the scientific configuration. Interrupted phases are preserved and fail visibly; automatic extra restarts are disabled because they can change the training trajectory.

After the arrays finish:

```bash
python experiments/original_table/run.py collect --study "$CCF_STUDY"
```

The collector writes the complete cell results and a readable best-of-selected-checkpoints table outside Git. Missing cells remain marked pending; they are not silently dropped. Keep all scored checkpoints, including negative results.

## Comparability

Matching source and arguments does not guarantee identical PPL across GPU models, software environments or nondeterministic kernels. A separate existing seed-1 verification matched the original FD 1k model tensors exactly; that is not proof of the final table. The initial validation did not submit a GPU job; the completed FD/DD retry is recorded below. CPU synthetic comparisons of the combined source are documented in `validation.json`; they do not establish full training or PPL reproduction. Review `validation.json` for the checks actually performed.

Historical CCF and MDLM reverse-step schedules match, but probability-calculation precision and final cleanup differ. CCF often executes S calls and MDLM S+1. The old table measures pipeline performance, not an isolated causal gain from the head. Historical training builds a BF16 rotary cache; evaluation builds an FP32 cache. Preserve this asymmetry when replicating; do not introduce an FP32 startup training probe. A scientifically matched alternative should be reported separately.

The table reports minima over three selected checkpoints and is exploratory. Repetition and diversity can worsen even when PPL improves. Original generation summaries retain repetition/distinct-n; raw outputs remain available outside Git for further length/EOS analysis. The main table uses 100 generated samples per cell from one training seed, not 100 independent training runs.

## Attribution

The applied changes come from the project's preserved historical R8, R16 and confirmation runtime versions, identified by hashes in `provenance.json`. Historical sampler functions originate in `audit_ccf_sampling_v3.py` and `audit_ccf_sampling_v4.py`; their comments retain the PyTorch exponential-race attribution. The numerical architecture, objective and scoring routines were not newly designed for this package. New code handles reconstruction, paths, scheduling arguments and collection.

## FD/DD 6k retry after the September 22 launch failure

Jobs 1558233_0 and 1558233_1 both exited with 105 after 12 seconds on
watgpu608. Their stderr reports `srun: ... Communication connection failure`
and `Application launch failed`; stdout is empty. Training Python did not
start. Evaluation 1558234 therefore remains blocked on its failed afterok
dependency. This identifies a Slurm task-launch communication failure, but
does not identify its underlying network/daemon cause or prove the node is
still faulty. Cluster-side diagnosis requires the administrator's logs.

The FD/DD launcher now keeps its Python entry point and batch script under
`experiments/original_table/` in this checkout. It no longer writes executable
code into a study directory. It preserves the Basic FD/DD seed-1 configuration
and phase resumes at 1k, 3k and 6k, and schedules exactly six confirmation cells:
FD and DD at 8/16/32 steps, 100 samples each, seed block 100001. No pilot is
scheduled for this profile.

Run from the existing branch (the launcher checks it):

```bash
cd /u401/n23zhang/clean_tree_mdlm/mdlm-fork1
git branch --show-current  # must be original_table_base_crf-recovery
unset CCF_STUDY            # choose a fresh study; preserve the failed run
bash scripts/launch_original_table_fd_dd_6k.sh
```

This submits two training tasks and their dependent six-cell evaluation array.
Both arrays exclude watgpu608 by default as a retry mitigation, not a repair of
the cluster. Set `CCF_EXCLUDE_NODES` to a different node list or explicitly to
an empty string to remove that exclusion after the issue is resolved.
Both batch jobs explicitly activate the existing `mdlm` environment and log
their host and interpreter before starting `srun`. No packages are installed
and no jobs are cancelled or automatically retried.

The login node mounts the home directory with `noexec`, so metadata preparation
uses `/usr/bin/python3` (standard library only), rather than the home-installed
Conda Python. Training and evaluation still use `mdlm` inside their allocated
compute jobs. `CCF_PREPARE_PYTHON` can override the metadata interpreter.

For preparation without submission, append `--prepare-only`. This creates a
new study, checks the historical source and backbone hashes, and records the
current checkout's file hashes. Use a fresh study for a later full submission.
Do not resubmit the old generated batch file: use the repository launcher above.
Existing study directories are refused, so old logs and partial checkpoints
cannot be overwritten. Do not edit this checkout between preparation and job
completion; source identity checks reject subsequent changes.

CPU validation (no actual sbatch, models, or GPU work):

```bash
PYTHONDONTWRITEBYTECODE=1 CCF_TEST_CACHE="$CCF_CACHE_ROOT" \
  /usr/bin/python3 tests/test_original_table_fd_dd_launcher.py
```

The tests verify historical training arguments, profile cells, source hashes,
repository script paths, node exclusions and dependency wiring with a mocked
sbatch. Passing these checks does not establish compute-node connectivity or
a successful GPU training run.

## Completed FD/DD 6k run — September 22, 2026 (UTC)

Training array `1558547` and all six evaluation tasks in `1558548` completed
successfully with exit code 0. Both Basic models reached 6,000 training steps.
The last evaluation finished at 2026-09-22 03:55:58 UTC (September 21, 23:55:58 EDT).

GPT-2-large generative perplexity (lower is better), with 100 scored samples
per cell, scored through the first non-leading EOS:

| Model | 8 sampling steps | 16 sampling steps | 32 sampling steps |
|---|---:|---:|---:|
| MDLM baseline (historical, released) | 815.06 | 313.78 | 162.56 |
| Basic FD @ 6k (this run) | 615.85 | 231.58 | 143.99 |
| Basic DD @ 6k (this run) | 665.49 | 248.41 | 145.65 |

The MDLM row comes from `expected-table.csv`; MDLM was not rerun in this
submission. The FD/DD rows use the fixed 6k checkpoint, not a best-of-checkpoints
selection. All six cells have completion markers and zero unresolved mask
tokens. The precision and final-cleanup differences described under
[Comparability](#comparability) still apply; these are pipeline comparisons.

Run provenance:

- Branch: `original_table_base_crf-recovery`.
- Repository: `/u401/n23zhang/clean_tree_mdlm/mdlm-fork1`.
- Study: `/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/fd_dd_exact_seed1_6k.SnCq9S/study`.
- Per-cell results: `evaluation/confirmation/000` through `005`, each containing
  `completed.json` and `generation/summary.json`, relative to the study directory.

Launch command used:

```bash
cd /u401/n23zhang/clean_tree_mdlm/mdlm-fork1
unset CCF_STUDY
bash scripts/launch_original_table_fd_dd_6k.sh
```
