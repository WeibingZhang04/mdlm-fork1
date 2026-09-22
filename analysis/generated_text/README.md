# Generated-text evaluation, 500 samples per cell

**DO NOT touch other people's files. DO NOT cancel other people's jobs.**
Only run as `n23zhang`. Inputs and output paths must be owned by that account
and resolve under `/u401/n23zhang`. Existing outputs are refused; no cancellation
commands are used. Do not change another user's files or processes.

This directory adds post-training work only. It does not change existing model,
training, sampling, scoring, or submission code. It is outside the active
`experiments/original_table` directory and the FP32 study's recorded source list.
The scripts run the existing checkout, not a copied runtime. Keep the checkout
unchanged until both the original and the new jobs finish.

## Scope and files

15 cells: Basic FD/DD x BF16/FP32 training-cache precision x 8/16/32 sampling
steps, plus the released factorized MDLM at those three budgets. FD/DD use the
fixed 6k checkpoints. MDLM is the released backbone, not a 6k-trained model.
All generation uses FP32 rotary caches, asserted at runtime. Actual NFE remains
in the generated records; a nominal S-step budget is passed as S+1 exactly as in
the existing evaluator.

- `generate.py`: prepare a manifest, then generate one array cell using the
  existing `run_generation_pilot` and `historical_gate`. It reuses verified
  adapter exports. Factorized MDLM uses the existing baseline path, which
  ignores the loaded structured adapter for sampling.
- `analyze.py`: shared ownership guards, artifact checks, metrics, prefix
  scoring, sample-level plots, and bootstrap summaries.
- `blind_review.py`: 60 randomized blind comparisons and external-rating tally.
- `mauve_eval.py`: held-out reference, cached embeddings, MAUVE, combined table.
- `run.sbatch`: one generation task or the subsequent analysis pipeline.
- `test_analysis.py`: CPU-only boundary, aggregation, blinding, and safety tests.
- `requirements.txt`: pinned additional packages for a separate analysis venv.

We generate 500 **fresh** samples per cell with base seed 300001, shared across
conditions. Existing 100-sample files remain untouched and are not combined into
this dataset. This avoids mixed-run aggregation and uses a fresh evaluation seed
block. Pair keys and seeds are checked, but matching seeds do not guarantee
matching topics or identical stochastic trajectories across different models.
This is not a multi-training-seed experiment. Different GPU allocations are an
additional limitation when interpreting BF16 versus FP32 differences.

## Metric definitions

Primary PPL remains token-weighted GPT-2-large FP32 scoring through the first
non-leading EOS. The scorer revision is
`32b71b12589c2f8d625668d2335a01cac3249519`. Per-sample PPL is used for plots, not
averaged to produce corpus PPL. The original full-sequence metrics are retained
in generation artifacts; new metrics below are separately labeled.

- Rep-4: `1 - unique_4grams / total_4grams` within each sequence, averaged over
  eligible samples. Length <4 is undefined (NA), not a perfect repetition score.
- Distinct-4: unique corpus 4-grams divided by total corpus 4-grams. These two
  metrics use original generated token IDs, omit a leading BOS, and stop before
  the first non-leading EOS. They are length-sensitive; a 256-token prefix
  distinct-4 is also reported with sample coverage.
- Length: stored GPT-2 scored-token count. Early EOS: fraction with fewer than
  128/256 content tokens before a terminal EOS, measured in original generated
  tokens. A separate no-EOS rate distinguishes truncation from stopping.
- Prefix256 PPL: score exactly 256 predicted GPT-2 tokens (257 input tokens),
  strictly before terminal EOS, without changing tokenization by decode/reencode
  cropping. Short samples are ineligible, and eligible counts are reported.
  This is a conditional diagnostic, not a replacement for EOS-based PPL.
- Uncertainty: 1,000 whole-sample bootstrap replicates for corpus PPL and paired
  FP32-minus-BF16 PPL differences. They quantify sampling uncertainty for these
  checkpoints, not model-seed or hardware variability.

MAUVE uses the authors' package, with GPT-2-large last-layer terminal hidden
states, FP32, batch 1, maximum 1024 tokens, automatic bucket count and clustering
seeds 0/1/2. Features are cached once per collection. Generated texts end before
the first non-leading EOS. Empty texts are retained as an EOS-only input.
The fixed reference contains 500 randomly selected nonempty source documents
from OpenWebText rows `[7913769, 8013769)`, revision
`79d93d786212f7344586290adb811d4ae6a1762c`. Those rows are held out from adapter
training; this is not a claim that the released backbone never saw them.
All generated cells share exactly that reference. Reference row IDs, hashes,
package versions and feature policy are saved.

**MAUVE at 500 samples is exploratory.** The authors recommend a few thousand
samples per distribution. Variation across three clustering seeds measures
algorithm sensitivity, not independent-data confidence intervals. Document
length and EOS differences can affect the score; no single metric establishes
overall quality.

## Blind review

Ten randomly selected seed pairs per FD/DD x budget = 60 comparisons. Sides are
balanced 5/5 in each group and shuffled. Full pre-EOS text is shown, without
model labels, PPL, or the key. Text is HTML-escaped and the judging rubric warns
LLMs not to follow instructions inside generated passages. Ratings cover
fluency, coherence, non-repetition and overall A/B/tie/both-poor preference.

Send **only `blind/blind-review.zip`** to fresh LLM conversations. It contains
the passages, HTML reader, rubric and CSV template. `blind/private/answer_key.json`
stays separate. `analysis-results.zip` contains condition labels and must not be
given to blind judges before rating. No LLM API is called by these scripts.
Keep each judge's answers independent and identify the model/version. Agreement
among LLMs is not a substitute for human judgments. Codex knows the experiment
hypothesis, so its initial review is not fully independent.

## Prepare and validate

From the existing checkout, as `n23zhang`:

```bash
cd /u401/n23zhang/clean_tree_mdlm/mdlm-fork1
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 analysis/generated_text/test_analysis.py
export CCF_CACHE_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache
export TEXT_EVAL_ROOT="$CCF_CACHE_ROOT/runs/text_eval_500_$(date -u +%Y%m%dT%H%M%SZ)"
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 analysis/generated_text/generate.py prepare \
  --bf16 "$CCF_CACHE_ROOT/runs/fd_dd_exact_seed1_6k.SnCq9S/study" \
  --fp32 "$CCF_CACHE_ROOT/runs/fd_dd_exact_seed1_6k.PafavQ/study" \
  --out "$TEXT_EVAL_ROOT"
```

Preparation creates metadata and logs only; it does not submit a job. It records
source hashes, refuses unrelated experiment source differences, and permits
only the deliberate rotary/provenance difference plus documentation changes.
Generation verifies those hashes before loading models and waits for each
source cell's completion marker and exported artifact hashes.

The CPU tests do not validate CUDA kernels, downloads, or full-generation PPL.
Run a GPU smoke before relying on an overnight campaign. No job is automatically
retried after failure; preserve its output and use a new output root for a retry.

## Run without MAUVE (current campaign)

The existing `mdlm` environment supports generation, EOS PPL, prefix256 PPL,
repetition/distinct-4, EOS and length diagnostics, sample plots, bootstrap
intervals, and blind-review packaging. No installation or separate environment
is needed for these tasks. `run.sbatch preflight MANIFEST` checks CUDA, the
cached tokenizer and GPT-2-large prefix scorer before generation.

Use `run.sbatch analyze-basic MANIFEST` after all generation tasks succeed.
This runs the diagnostics and creates `analysis-results.zip` plus the separate
`blind/blind-review.zip`. It does not import or run MAUVE or fetch its reference
data. Blind preference still requires a reviewer; packaging alone is not a
completed reading test. MAUVE can be added later using these same saved samples.
All commands retain the account/path safeguards above.

## Separate analysis environment (MAUVE only)

Generation uses the existing `mdlm` environment without package changes.
The optional MAUVE stage additionally needs the pinned packages. **Never install into `mdlm`.**
In your own allocated compute shell (home is noexec on the login node):

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mdlm
export TEXT_ANALYSIS_ENV=/u401/n23zhang/mdlm_data/tree_mdlm_cache/analysis_envs/text_eval_500
test ! -e "$TEXT_ANALYSIS_ENV" || { echo 'Choose a new environment path'; exit 1; }
python -m venv --copies --system-site-packages "$TEXT_ANALYSIS_ENV"
"$TEXT_ANALYSIS_ENV/bin/python" -m pip install --ignore-installed --no-cache-dir \
  -r /u401/n23zhang/clean_tree_mdlm/mdlm-fork1/analysis/generated_text/requirements.txt
export CCF_ANALYSIS_PYTHON="$TEXT_ANALYSIS_ENV/bin/python"
```

`--system-site-packages` reuses the existing Torch/Transformers without upgrading
them; `--ignore-installed` keeps newly installed dependencies inside the new
venv instead of attempting to uninstall shared packages. The expected inherited
versions are Torch 2.2.2+cu121 and Transformers 4.38.2. The venv interpreter must
resolve inside n23zhang's home (hence `--copies`). Do not create this environment
or run pip until authorized. Do not download another revision if pinned data or
models are absent: resolve the cache explicitly first. Production jobs run with
HF offline mode and fail rather than silently substituting data or models.

## Queue after FP32 evaluation

Submit only after environment/cache preflight and approval. `1558911` is the
entire FP32 evaluation array. If it fails, these jobs stay blocked. The next
array contains 15 cells with concurrency capped at four; analysis waits for all
15. On systems that purge completed IDs, verify successful accounting and
completion artifacts before omitting an expired dependency.

```bash
GEN_JOB=$(sbatch --parsable --array=0-14%4 --dependency=afterok:1558911 \
  --exclude=watgpu608 --chdir="$PWD" --export=ALL \
  --output="$TEXT_EVAL_ROOT/logs/generate-%A_%a.out" \
  --error="$TEXT_EVAL_ROOT/logs/generate-%A_%a.err" \
  analysis/generated_text/run.sbatch generate "$TEXT_EVAL_ROOT/manifest.json")
GEN_JOB=${GEN_JOB%%;*}
sbatch --dependency="afterok:$GEN_JOB" --exclude=watgpu608 --chdir="$PWD" --export=ALL \
  --output="$TEXT_EVAL_ROOT/logs/analysis-%j.out" \
  --error="$TEXT_EVAL_ROOT/logs/analysis-%j.err" \
  analysis/generated_text/run.sbatch analyze "$TEXT_EVAL_ROOT/manifest.json"
```

No new generated sample, score, feature, or review is written to the old studies
or repository. The complete report is `comparison.csv`, `analysis/report.md`,
plots under `analysis/`, MAUVE records under `mauve/`, and the blinded review
under `blind/`. Blind preference remains pending until external ratings arrive.

Tally a returned ratings file (new output path per judge):

```bash
/usr/bin/python3 analysis/generated_text/blind_review.py tally \
  --key "$TEXT_EVAL_ROOT/blind/private/answer_key.json" \
  --ratings /u401/n23zhang/path/to/judge-ratings.csv --judge MODEL_VERSION \
  --output "$TEXT_EVAL_ROOT/judge-MODEL_VERSION.csv"
```

## Citations and code attribution

- Pillutla et al. (2021), *MAUVE: Measuring the Gap Between Neural Text and Human
  Text using Divergence Frontiers*, NeurIPS. https://arxiv.org/abs/2102.01454
- Pillutla et al. (2023), *MAUVE Scores for Generative Models: Theory and
  Practice*, JMLR. https://www.jmlr.org/papers/v24/23-0023.html
- MAUVE implementation: https://github.com/krishnap25/mauve (`mauve-text==0.4.0`).
  We call the authors' package, rather than reimplementing its estimator. The
  repository's license applies to that dependency; none of its source is copied.
- Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration*, ICLR.
  https://arxiv.org/abs/1904.09751 — motivation for repetition and quality/diversity
  diagnostics; not a claim that our entire protocol reproduces theirs.
- Li et al. (2016), *A Diversity-Promoting Objective Function for Neural
  Conversation Models*, NAACL. https://aclanthology.org/N16-1014/ — distinct-n;
  our tokenization, order n=4 and aggregation are explicitly specified above.
- Generation and EOS scoring are reused from this repository's
  `scripts/run_generation_pilot.py`, `evaluation/generation_metrics.py` and
  `experiments/original_table/historical_gate.py`, retaining their attribution.
  Prefix loss masking and n-gram definitions follow those existing routines.
  New orchestration, safety checks, plots and review packaging are project glue.
