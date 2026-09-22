# DD topology controls, 500 samples per setting

**DO NOT touch other people's files. DO NOT touch other people's jobs.**
Do not interfere with other people's processes. Run only as n23zhang. All inputs
and outputs must be owned by n23zhang under /u401/n23zhang. Existing outputs are
refused, no cancellation commands are used, and no packages are installed.

Question: does learned edge placement or joint dependence improve generated
text for our existing DD models? These are inference interventions, not new
training. Code stays in original_table_base_crf-recovery in the existing checkout.
Only new files in analysis/topology_controls are added. Existing model, training,
sampling and scoring source files remain unchanged; hooks are process-local.

## Protocol

Both Basic DD 6k checkpoints: BF16-cache training from SnCq9S and FP32-cache
training from PafavQ. Both retain the frozen released backbone, K128, shared rank16
factors, component cap32, sequence length1024 and batch1. Generation always uses
FP32 rotary caches (asserted), with the original internal BF16 autocast unchanged.
The parent manifest is text_eval_500_20260922T062243Z/manifest.json.

For each checkpoint and 8/16/32 reverse steps, run all four variants with 500
fresh samples using base seed400001: 24 cells, 12,000 scored samples. No selection
by results, retraining, checkpoint search, or rejection sampling. All four variants
for one checkpoint/budget run sequentially on the same GPU allocation; six array
tasks use at most four GPUs concurrently. Actual NFE is recorded (budget=S+1).

| Variant | Definition | Interpretation |
|---|---|---|
| native | Existing learned forest, joint sampling | Reference |
| chain | Replace selected edges by the existing cap32 chain over consecutive active masked positions | Changes placement, shape and possibly edge count; not a pure placement control |
| random | Uniformly permute active-token labels of the native selected forest at each call | Exactly preserves edge count, component sizes, degree multiset and isolated count at the current state; changes which tokens have those degrees and edge distances |
| marginal | Existing structured_marginal sampling on the native graph | Independently sample exact node marginals, preserving conditional marginal distributions at a given state, not neutralizing factors |

Dynamic factor network weights are unchanged. For new edges the existing head
gathers factors from their new endpoints. Random relabeling uses a private Python
RNG seeded by SHA256(sample seed, forward index, batch index, graph-v1), so it does
not advance training/sampling RNGs. It is not uniform over all possible forests
and is not distance-matched; one graph randomization per sample is averaged over
500 samples. Diverging generated contexts mean exact graph matching applies to
the native forest at the same current state, not between complete trajectories.
Matching sample seeds does not guarantee matching topics or random draw order.

Primary reported endpoint: paired token-weighted EOS GPT2-large PPL differences
for each intervention versus native, separately by checkpoint and sampling steps.
All 18 contrasts are reported; bootstrap95% intervals are descriptive, not
multiple-comparison-corrected significance claims. These are two checkpoints from
one training seed, not independent training replications. Fixed-chain success would
motivate retraining but does not establish that retrained fixed graphs are optimal.

Secondary diagnostics: repetition4, distinct4, scored length, early EOS<256,
no-EOS rate, prefix256 PPL with eligible counts, prefix distinct4, sample scatter
plots, graph isolation/component/span statistics, actual NFE and GPU identity.
Metric definitions reuse analysis/generated_text/README.md; prefix scores are
conditional diagnostics and do not replace EOS PPL. No MAUVE or automatic LLM judge.

## Files and validation

- interventions.py: graph hooks, validation, tracing and synthetic self-tests.
- run.py: manifest preparation, GPU gate and four-variant group execution.
- report.py: metrics, paired bootstrap, figures and topology-results.zip.
- run.sbatch: existing mdlm environment and Slurm modes; no install step.

The gate checks graph forest/cap/endpoint constraints, preservation of weights,
candidate unaries and global RNG, random graph shape invariants, and chains over
nonconsecutive masked positions. It then generates two samples per variant from
both checkpoints at8 steps, with the production scorer and kernels. Native
instrumented generation must match uninstrumented optimized generation in tokens,
NFE and CPU/CUDA RNG state. Production repeats native parity once per group.
The optimized historical sampler is installed before graph hooks. All variants
are checked at each graph call; random relabeling preserves native shape exactly.

Gate samples (seed490001) are not included in the production500. Source hashes,
checkpoint hashes, full generation arguments, helper hashes via source_sha256,
graph traces, package versions in generation manifests, and Slurm submission
commands in jobs.json make changes reviewable. Source hashes are checked before
each group; do not edit the recorded checkout while jobs are active. New failed
attempts are preserved; any retry must use a fresh output root.

## Reproduce from the existing checkout

```bash
# DO NOT touch other people's files. DO NOT touch other people's jobs.
# Do not interfere with other people's processes.
cd /u401/n23zhang/clean_tree_mdlm/mdlm-fork1
export TOPOLOGY_ROOT=/u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/dd_topology_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 analysis/topology_controls/run.py prepare \
  --parent /u401/n23zhang/mdlm_data/tree_mdlm_cache/runs/text_eval_500_20260922T062243Z/manifest.json \
  --out "$TOPOLOGY_ROOT"
GATE=$(sbatch --parsable --exclude=watgpu608 --time=00:45:00 \
  --output="$TOPOLOGY_ROOT/logs/gate-%j.out" --error="$TOPOLOGY_ROOT/logs/gate-%j.err" \
  analysis/topology_controls/run.sbatch gate "$TOPOLOGY_ROOT/manifest.json")
GATE=${GATE%%;*}
GEN=$(sbatch --parsable --exclude=watgpu608 --array=0-5%4 --dependency="afterok:$GATE" \
  --output="$TOPOLOGY_ROOT/logs/generate-%A_%a.out" --error="$TOPOLOGY_ROOT/logs/generate-%A_%a.err" \
  analysis/topology_controls/run.sbatch generate "$TOPOLOGY_ROOT/manifest.json")
GEN=${GEN%%;*}
sbatch --exclude=watgpu608 --dependency="afterok:$GEN" \
  --output="$TOPOLOGY_ROOT/logs/analysis-%j.out" --error="$TOPOLOGY_ROOT/logs/analysis-%j.err" \
  analysis/topology_controls/run.sbatch analyze "$TOPOLOGY_ROOT/manifest.json"
```

Read analysis/report.md and paired_differences.csv after completion. Raw generated
text remains in cells/*/generation/samples.jsonl; graph-trace.jsonl records every
model graph call. No raw samples are sent to external services.

## Attribution

This code orchestrates existing project functions in models/structured_decoder.py,
diffusion.py, evaluation/generation_harness.py, evaluation/generation_metrics.py,
and experiments/original_table/historical_sampler.py. Their existing attribution
and historical provenance remain applicable. It reuses shared ownership/analysis
functions in analysis/generated_text/analyze.py. New graph relabeling and wrappers
are experimental project code, not a claim of a novel topology-learning method.

- Distinct-n: Li et al., 2016, https://aclanthology.org/N16-1014/ (we use n=4).
- Repetition/quality context: Holtzman et al., 2020, https://arxiv.org/abs/1904.09751.
- Forest selection, inference and sampling are the repository's existing
  implementations; this study does not replace their algorithms.
