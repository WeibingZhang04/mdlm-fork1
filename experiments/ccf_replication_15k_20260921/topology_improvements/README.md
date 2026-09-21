# DD topology improvement experiments

Status: new, predeclared experiments; no improvement has been established yet.

## What changes

All overrides are explicit, reversible, process-local code in `topology_variants.py`.
The main 15k replication and its frozen source are unchanged. Run the variant
launcher to reproduce these experiments; the base adapter manifest alone is not
sufficient. Each export and generation has an additional variant sidecar.

| Label | Change relative to its control |
|---|---|
| native | Original DD; no override |
| active_chain | Add missing edges between consecutive remaining masked positions; learned scorer and cap 32 retained |
| chain_coverage | Active chain plus score-ordered matching, isolated-node attachment, then bounded Kruskal |
| chain_unique_anchors | Active chain plus distinct anchor positions, chosen greedily in fixed slot order; recycle only after exhausting active positions |
| chain_no_absolute_local | Active chain, remove absolute distance-1/2 proposals; keep learned anchor proposals |
| chain_absolute_window4 | Active chain, expand absolute local window from 2 to 4 |
| chain_dense_teacher | Active chain plus training edge-score supervision over every other active position for the sampled source |

The first five interventions change graph construction: these are architecture /
sampling-rule ablations, even though learned parameter counts are unchanged.
Coverage-first selection is **not** a maximum-spanning forest. Forest acyclicity,
active endpoints and the component cap remain enforced. Unique anchor selection
has a fixed slot-order bias and is a hypothesis, not a proven improvement.
The dense teacher changes the training loss, not the inference proposal graph.
It uses the same one reveal / teacher forward as native training; extra edge
scoring costs O(B L D). No new learned parameters, larger component cap, vocabulary
candidate changes, or factor-family changes are introduced. Existing proposal
duplicates remain, but appended chain edges already present are masked out to
avoid an extra duplicate weighting of the training teacher.

## Fixed comparisons

- Inference-only screen: existing **6k** Basic DD and Separate R16 DD, lengths
  **128 and 1024**, **8 and 32 denoising steps**, **20 samples/cell**, seed 210001.
- Fresh-seed confirmation: **all six inference variants**, both model families,
  length 1024, 8/32 steps, **100 samples/cell**, seed 220001. This is not restricted
  to apparent winners. Primary contrast: Basic DD active_chain versus native at
  8 steps. Other contrasts are exploratory and all are reported.
- Training screen: native, active_chain, chain_dense_teacher from the **identical
  old Basic DD 6k checkpoint**, restored optimizer, seed 1, same restarted data
  stream, **2,000 additional updates to 8k**. This is not a fresh 15k run. Every arm
  gets a three-update smoke test first. The existing eight-arm 15k jobs continue.
- Trained confirmation: before6k and all three after8k adapters in one GPU job,
  100 samples/cell, fresh seed 230001, length 1024, 8/32 steps. Report joint and
  each model's own product of marginals, plus a same-job MDLM baseline.
- Same-job native/MDLM controls for each inference matrix; no GPU model/node pin.
  Development and confirmation may run on different GPU types, so compare arms
  *within* each matrix. All learned inference variants within a case share a GPU.
- Fixed backbone/data/scorer hashes and campaign precision/EOS policy. Budget
  S+1 and measured NFE retained; do not call different actual NFE equal compute.

## Research integrity and interpretation

Generation receives corrupted tokens, current mask, time, and learned weights.
It never receives clean targets, teacher outputs or the reference scorer's
feedback. Clean training tokens appear only in the existing supervised likelihood
and reveal-teacher loss. Dense supervision uses features from the original
corrupted input, not the revealed teacher input. Unit tests explicitly perturb
clean labels and check student score/graph invariance.

No sample filtering, rejection sampling, seed search, checkpoint search or EOS
policy changes. Reference GPT2-large scores are never training targets. Preserve
all outputs and failed runs. Report PPL together with repetition-1/2/4, distinct-2/4,
length/EOS and graph statistics. Bootstrap paired sample IDs using token-weighted
NLL before exponentiating; never average per-sample PPL. Treat secondary intervals
as descriptive, not simultaneous significance claims. Lower PPL alone is not
sufficient evidence of better text if repetition or lengths degrade.

These choices were motivated by earlier WikiText diagnostics and generation
results. Thus WikiText remains a development diagnostic, not an untouched test.
New random generation seeds test sampling stability, not independence from the
pretrained data or generalization to an unseen corpus. One training seed and 100
samples limit the strength of paper claims. A winning screened change would need
fresh, matched full training replications before a definitive architecture claim.
Raw total loss across sparse/dense teacher variants is not directly comparable:
the number of teacher alternatives changes. Compare joint NLL, teacher KL/gain
versus uniform, and generation quality separately.

## Repairs to preceding diagnostic jobs

Original jobs 1556977 (weights) and 1556994 (graph generation) failed before usable
results. The weight callback received literal `target`; the tracer used `hidden`
and `unary` instead of the keyword API `hidden_states` and `unary_logits`. Corrected
copies are in `repairs/`; old helpers/results are preserved. Retry jobs and hashes
are in `repair-receipt.json`. These failures must not be interpreted as scientific
negative results. No other jobs were canceled or changed.

## Running

`deploy_improvements.py` records a unique campaign subdirectory, source/module
hashes, fixed protocol and exact sbatch commands. It submits a GPU gate, then
serial bounded arrays with dependencies on the gate. `smoke_gate.py` runs unit
tests, native generation parity and a real 3-update/export smoke for each training
variant. A failed gate prevents downstream jobs from starting. See the deployment
receipt for job IDs. Helpers import the existing audited campaign helpers and
frozen `code`; environment and Slurm scripts are saved in the remote experiment.

## GPU gate retry

Gate 1557038 failed while entering the coverage-first generation variant: the
sampling optimizer inspected the already-replaced Kruskal function. No long
study arm started. The runner now installs the native sampling optimization
first, then the explicit topology override. A regression test checks composition
and restoration for every variant, including that coverage-first remains active.
Ten local tests pass. Replacement gate **1557045** is submitted; the existing dev
and training arrays now depend on it. Failed logs and original helpers remain
under `gate/` and `helpers_before_retry2/`; the retry uses `gate_retry2/`.
