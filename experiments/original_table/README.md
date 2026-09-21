# Original CCF table: compact reproduction

This package reconstructs the original seed-1 experiment from the existing repository commit `8e1772f4687a78a48b6a330f9f095e9880540de2` plus three small patches. It adds no complete source snapshots, datasets, checkpoints, generated samples, training logs, usernames or personal storage paths to Git. The base commit already contains the existing project; this package does not rewrite that pre-existing history.

## What is preserved

- Basic FF/FD/DF/DD: original BF16 training, seed 1, original full-checkpoint restarts at 1k, 3k and 6k, finishing at 10k.
- Separate R8/R16 FD/DD: original separate-endpoint implementations, fresh seed-1 BF16 training to 7k.
- The released, frozen MDLM backbone and pinned training-data configuration from the original Git tree; the backbone file hash is checked before preparation.
- Original per-phase training override order and values. No added startup forward, numerical probe or training callback. Each training phase runs in a fresh Python process.
- The historical evaluation implementation, including the level-draw sampler and first-sample token/NFE/RNG equivalence gate. Numerical sampler functions are extracted unchanged from the original audit modules; unrelated benchmark code is omitted.
- Pilot: seven checkpoints, all eight arms and MDLM, 8/16/32 denoising steps, 20 samples, seed block 91001. Total: 171 cells / 3,420 samples.
- Confirmation: the original three checkpoint selections for each FD/DD model/budget, MDLM, 4/8/16/32 steps, 100 samples, seed block 100001. Total: 76 cells / 7,600 samples. FF/DF were not in the historical 100-sample table.
- GPT-2-large revision, FP32 scoring, length 1024 and first-non-leading-EOS scoring policy. Budget passed to the sampler is S+1; actual NFE is retained in results.

`protocol.json` holds the shared overrides, phase differences and frozen checkpoint selections. The three patches reconstruct R8, R16 and evaluation revisions in sequence. `provenance.json` records content hashes; runtime source reconstruction checks every included file. Large immutable historical runs remain outside Git. The old exploratory 500/1000-step generation screens are outside this compact table reproduction.

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

Preparation creates a new directory outside the repository. Only the required runtime source/configuration files are reconstructed there. Runtime copies, all checkpoints, resolved configurations, local paths and logs stay in that external directory. Preparation checks source and backbone identity without importing a model. `--source-only` is available for CPU source validation and deliberately prevents training/generation from that validation directory.

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
python "$CCF_STUDY/suite/run.py" collect --study "$CCF_STUDY"
```

The collector writes the complete cell results and a readable best-of-selected-checkpoints table outside Git. Missing cells remain marked pending; they are not silently dropped. Keep all scored checkpoints, including negative results.

## Comparability

Matching source and arguments does not guarantee identical PPL across GPU models, software environments or nondeterministic kernels. A separate existing seed-1 verification matched the original FD 1k model tensors exactly; that is not proof of the final table. No new GPU job was submitted to validate this compact wrapper. Review `validation.json` for the checks actually performed.

Historical CCF and MDLM reverse-step schedules match, but probability-calculation precision and final cleanup differ. CCF often executes S calls and MDLM S+1. The old table measures pipeline performance, not an isolated causal gain from the head. Historical training builds a BF16 rotary cache; evaluation builds an FP32 cache. Preserve this asymmetry when replicating; do not introduce an FP32 startup training probe. A scientifically matched alternative should be reported separately.

The table reports minima over three selected checkpoints and is exploratory. Repetition and diversity can worsen even when PPL improves. Original generation summaries retain repetition/distinct-n; raw outputs remain available outside Git for further length/EOS analysis. The main table uses 100 generated samples per cell from one training seed, not 100 independent training runs.

## Attribution

The patches come from the project's preserved historical R8, R16 and confirmation runtime versions, identified by hashes in `provenance.json`. Historical sampler functions originate in `audit_ccf_sampling_v3.py` and `audit_ccf_sampling_v4.py`; their comments retain the PyTorch exponential-race attribution. The numerical architecture, objective and scoring routines were not newly designed for this package. New code handles reconstruction, paths, scheduling arguments and collection.
