# MDLM CRF 1024 experiment decisions

Updated 2026-10-01. This document records the decisions and reasons for the new
1024-token frozen-backbone experiment. The 256-token runs established a working
pipeline; they do not set the data or training budget for this experiment.
The separate preparation implementation has passed offline checks. Data
preparation is authorized as a draft for full-scale head training. The launch
record below distinguishes a submitted preparation job from a completed dataset.
No head training is authorized by that preparation launch.

## Agreed scope

The released MDLM backbone stays frozen throughout this study. Only the global,
contextual, and independent heads are trained. The count baseline estimates
statistics from training text. Joint backbone training and ordinary MDLM
continuation are out of scope because of the available time.

Retain backbone, count, global, contextual, and independent comparisons. The
choice of generation sampler and sampling-step sweep is deferred until after
training setup. The head trainer learns from randomly corrupted real text; it
does not unroll generation, so choosing a sampling step count does not require
separate head training. The current head evaluator uses the matched random
reveal schedule. A native-style sampler for these heads would require inference
work, not just passing a supported flag.

## Dataset roles

| Role | Population | Use |
| --- | --- | --- |
| train | Selected documents from the pinned source's training interval | Head optimization and count fitting |
| dev | Different documents from that training interval | Checkpoint and training-configuration selection |
| heldout_eval | All of the pinned source's final 100,000 documents, in source order | Final diagnostics after choices are fixed |

Dev and validation usually mean the same role. Evaluation is an activity that
can be performed on either development or final held-out data. Development data
influences model selection even without gradient updates.

The original MDLM paper reserves the final 100,000 OWT documents for validation.
It does not prescribe our additional head-development split within its training
population. That extra split is our design choice so head tuning does not use
the population reserved for final evaluation. We will call the latter the
original MDLM validation population only once its historical source alignment
is established; it is not a newly independent pretraining test set.

## Select documents before packing

The agreed order is: establish source partitions, select documents and their
roles, then tokenize and pack each role independently. Randomly splitting packed
rows could put different parts of one source document in train and development.
Sampling documents also makes the chosen source population explicit, rather
than selecting documents in proportion to the number of full chunks they yield.
During training, longer selected documents still contribute more tokens.

The separate entry point is `scripts/prepare_chain_packed_data.py`. Its concrete
implementation choices are recorded here; these are not claims about MDLM's
original subset selection:

1. Load `Skylion007/openwebtext`, configuration `plain_text`, at revision
   `79d93d786212f7344586290adb811d4ae6a1762c`. Require 8,013,769 text rows.
   Apply the published split rule: training indices `[0, 7913769)` and validation
   indices `[7913769, 8013769)`.
2. Hash the raw UTF-8 text of the entire validation tail. Exclude matching text
   from head training and development, including duplicates at different indices.
3. Generate a deterministic permutation of training indices using a SHA256
   counter and lazy Fisher-Yates sampling, with unbiased bounded draws. This
   avoids dependence on the Python random-number generator's implementation.
4. Take the requested number of eligible development documents first, then the
   requested training documents. Skip blank documents, exact content duplicates,
   and explicit exclusions. Keep selection order as packing order.
5. Preserve the entire validation tail in original source order. Do not sample,
   deduplicate, remove blank documents, or apply prior-experiment exclusions to
   this evaluation population. Record those characteristics and prior overlaps
   in the manifest instead. All matching text remains excluded from head
   training/development. This follows the user's preference for the full
   benchmark population rather than our earlier 10,000-document proposal.

The draft uses dataset seed `mdlm-crf-1024-v1`, 200,000 training documents,
2,000 development documents, and all 100,000 validation documents. These are
mandatory CLI inputs, together with `--evaluation-population full-tail`.
Increasing the train quota with the same seed, dev quota,
source and exclusions extends the same training document list. Changing the dev
quota can change the training subset. Head-training seeds are a separate choice;
all training seeds will use the same fixed prepared dataset.

Pass the previous 256 train/dev/test JSONLs as explicit exclusions for head
training/development. Evaluation preserves the benchmark population, so prior
overlaps there are reported rather than removed. Thus the benchmark is not
claimed to be untouched by every earlier smoke experiment. Exclusions are
content hashes, so duplicate text under another source index is also excluded
from head training/development. We do not claim near-duplicate detection.

The 200,000-document training budget is an initial draft choice, not evidence
that more data cannot help. Report the actual packed token count, tokens seen
by each training run, and development learning curves. The full model still
includes released MDLM; only its small added heads are trained. Every final
method, including the frozen backbone baseline, should use identical evaluation
inputs and scoring. Generative perplexity evaluates generated text and requires
a separately matched generation/scoring protocol.

## Tokenization and packing

Use `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. BOS and EOS both have ID 50256.
Tokenize without automatic special tokens, append EOS after each selected
document, and concatenate within each split. Take consecutive 1,022-token
payloads, then surround each with BOS and EOS to produce 1,024-token rows.

Short documents can share a row; a long document can span rows. Every source
document stays in one role. Document separators remain in the token stream.
Do not add an attention reset at separators: the frozen MDLM convention uses
ordinary attention across each row. Count and learned pair factors include
adjacencies involving EOS, but no factors or counts connect separate rows.

The new script carries tokens across processing batches and drops only the last
incomplete payload of each split. MDLM's original `datasets.map` implementation
can drop remainders at map-batch/worker boundaries. We match its tokenizer and
boundary convention, not its exact historical rows or batch-dependent drops.
The manifest records this deliberate difference and the dropped tail spans.

The old 256 examples are not joined or repurposed. The native-parent recovery
helper is not suitable for this newly selected raw-text subset.

## Reproduction records and checks

The output contains `selected_documents.json`, train/dev/heldout_eval `.pt` and
`.jsonl` files, and `manifest.json`. The manifest is written last and marks a
completed bundle. An interrupted directory is not a valid completed dataset;
the script refuses to overwrite an existing output directory.

Reproduction depends on the source revision/order, selection algorithm and seed,
quotas, exclusions, tokenizer revision, and packing policy. Save all of them,
along with code hashes, environment versions, ordered selected indices/content
hashes, and output checksums. Matching a seed alone is insufficient. The raw
little-endian int64 token hashes allow comparison independent of PyTorch archive
serialization details. The selected index/hash lists are the exact subset record.

Every packed row records all contributing document IDs and token spans into
those documents. `chain_crf/data.py` now accepts these memberships in addition to
legacy single-document rows, and checks every contributing document for overlap.
Do not replace real memberships with a synthetic packed-row ID. Preparation also
rejects identical full token rows across different roles instead of silently
accepting leakage. Count fitting rejects the `heldout_eval` role.

Offline tests cover seeded selection, source-tail duplicates, exclusions,
document separation, exact packing/separators, token-span reconstruction, tail
accounting, changed source text, saved tensor/JSONL agreement, checksums, and
legacy-compatible overlap checks. These are preparation checks, not real-backbone
1024 validation or a convergence result.

On 2026-10-01, the packed-data, training/resume, native-parent, and transfer test
files passed all 35 tests in the WATGPU `mdlm-crf` environment with CUDA disabled.
The new script's `--help` also ran successfully there. Full OWT acquisition and
the real-source preparation path have not been run. Local copies of the files
are retained in `/Users/nina/Research/crf-rework`.

## Source audit and current WATGPU state

The local and remote repositories were observed at upstream commit
`a46d687b3c0a0d0fcec846bdd6d9afb252c82c11` before these changes. Existing staged
remote sampling/training launchers are unrelated and must be preserved.

On 2026-10-01, inspection found the old 256 bundle in
`/u401/n23zhang/rework-data/data/chain-owt`. Neither the configured dataset cache
under `/u401/n23zhang/rework-data/hf-cache/datasets` nor the legacy cache under
`/u401/n23zhang/.cache/huggingface/datasets` contained a full indexed OWT dataset.
The pinned dataset card reports approximately 24.2 GB of compressed downloads
and 39.8 GB of processed dataset bytes. Filesystem inspection reported about
154 GB available; user quota and final artifacts also need to be considered.
No full source download has been initiated.

The script uses the pinned Hugging Face loader and checks row count/schema. This
establishes its requested public source and split rule, but does not independently
authenticate the exact bytes/order used to train the released MDLM checkpoint.
The manifest states this limit. `--source-audit` can attach an evidence JSON file
and its checksum; attaching a file is not itself proof. Historical alignment
remains unresolved and must be described honestly in any paper.

## Parameters still to settle before a real preparation run

- Report the resulting unique payload tokens as well as total row tokens; the
  draft quotas and seed are fixed above, but a formal training budget is not.
- Source acquisition and historical split evidence. Loading the indexed source
  may download the entire corpus even though only a subset is tokenized/trained.
- Head ranks, learning rates, effective batch size, runtime budget, number of
  training seeds, and checkpoint evaluation frequency. The smoke-test budget is
  not a constraint; the trainer's default dev cap of 128 examples also needs an
  explicit decision before launch.
- Final generation sampler, step sweep, scorer/reference boundary policy and
  sample counts. Packed diagnostic rows do not finalize the later MAUVE reference
  population or protocol; retain document IDs so references can be built correctly.

The draft preparation command has this shape. The dated output directory is
created beneath `/u401/n23zhang/rework-data/data/draft-full-scale-training/`.
The shared raw-source cache remains separate and can be reused for a formal run.

```bash
python scripts/prepare_chain_packed_data.py \
  --output "$PACKED_DATA_OUTPUT" \
  --dataset-cache /u401/n23zhang/rework-data/hf-cache/datasets \
  --tokenizer-cache /u401/n23zhang/rework-data/hf-cache/hub \
  --seed mdlm-crf-1024-v1 \
  --train-documents 200000 \
  --dev-documents 2000 \
  --heldout-documents 100000 \
  --evaluation-population full-tail \
  --exclude /u401/n23zhang/rework-data/data/chain-owt/train.jsonl \
  --exclude /u401/n23zhang/rework-data/data/chain-owt/dev.jsonl \
  --exclude /u401/n23zhang/rework-data/data/chain-owt/test.jsonl
```

## Draft preparation launch record

The user authorized preparation on 2026-10-01 and requested a draft/test folder
to distinguish this dataset from a possible future formal full-scale run. All
writes are restricted to their checkout and `/u401/n23zhang/rework-data`.
Only their own allocations may be used; other users' files and jobs must not
be modified, and no other user's job may be cancelled. Submission details and
completion status are recorded below.

- Slurm job `1579295` (`draft-owt-prep`) submitted at 05:04:24 UTC on 2026-10-01.
  CPU-only, four CPUs, 24 GB RAM, eight-hour limit, partition `ALL`, node
  `watgpu108`. Started at 05:04:53 UTC and observed running; completion has not
  yet been verified. The existing interactive job `1579267` was left unchanged.
- Prepared output destination:
  `/u401/n23zhang/rework-data/data/draft-full-scale-training/owt-1024-2026-10-01_050424_UTC`.
- Launch record, source snapshots, batch script and `prepare.log`:
  `/u401/n23zhang/rework-data/runs/draft-full-scale-training/prepare-owt-1024-2026-10-01_050424_UTC/`.
- Raw dataset cache: `/u401/n23zhang/rework-data/hf-cache/datasets`.
- The updated preparation and compatibility checks passed all 37 tests in the
  WATGPU `mdlm-crf` environment before submission. The full-tail test preserves
  source order, duplicate and empty documents, and audited prior overlaps.
- Successful completion requires `manifest.json` in the output directory and
  the final `prepared` event in the log. Training has not been started.

## Step 3 validation

The user authorized step 3 after preparation, followed by preparing the overnight
training launcher. The separate entry point is `scripts/validate_chain_1024.py`.
Job `1579324` (`draft-step3`) was submitted on 2026-10-01 with the Slurm
dependency `afterok:1579295`. It requests one GPU, four CPUs and 24 GB RAM on
`watgpu108`, with a two-hour limit. Its run directory is
`/u401/n23zhang/rework-data/runs/draft-full-scale-training/step3-owt-1024-2026-10-01_051416_UTC/`.
It requires an NVIDIA RTX 6000 Ada Generation GPU and locally cached model files.

The 85 mathematical, head, generation, evaluator-resume, training, packing and
segmented-inference tests passed in the remote environment while download was
in progress. These are not substitutes for the real-data checks below.

The validation runner audits output checksums, all token rows and source
separation, then saves 32 training rows and four development rows as fixed
reference fixtures. It tests exact enumeration and empirical joint sampling on
GPU, neutral factors against real MDLM probabilities at length 1024, and eight
short updates of each learned head with batch size four and K=64. Backbone
parameter hashes and gradient flags must remain unchanged. Optimizer/RNG resume
must recover the next update. Parameter counts, time per update and memory are
recorded to guide step 4; eight updates do not establish convergence.

Global rank 32 and independent rank 64 initialize at identity. Fresh contextual
rank 32 deliberately initializes with small nonzero pair scores, so its identity
control is constructed explicitly by loading a fresh identity global head. The
actual contextual profile still uses the intended fresh contextual initializer.
These ranks are provisional profiling settings, not yet the user's final
training configuration.

The runner also exercises real training CLI resume, generation and partial-batch
resume for backbone/count/global/contextual/independent at length 1024, and
re-scores the same saved sequences twice with the cached GPT-2-large scorer.
Two generated sequences per mode use validation draw IDs 90,000,000 and
90,000,001. Counts from eight diagnostic training rows and the brief trained
heads are labelled validation-only and are not main experiment artifacts. The
current matched sampler and raw-ID scorer are plumbing checks, not a decision
about the final sampling/scoring protocol. Final evaluation text is used only
for integrity/separation auditing at this stage.

`results/validation.json` with `status: passed` is written last on success.
`results/failure.json` records an observed failure. The main training launcher
must require a successful validation record for the exact data manifest before
submitting long training. Historical released-MDLM source alignment remains a
documented limitation rather than a claimed result of these checks.

## Step 4 simple four-command launcher: submitted after confirmation

The user requested the same straightforward style as `sample_diagose_all.sh`,
using the existing `scripts/train_four_models.sh`. Upstream branch
`codex/chain-crf-20260925` was rechecked at commit
`a46d687b3c0a0d0fcec846bdd6d9afb252c82c11`; its fetched `CHAIN_CRF.md` is identical
to the local copy and documents exactly four calls: fit counts, train global,
train contextual, and train independent. Released MDLM itself stays frozen.

The main launcher now contains those four commands directly, with shared flags
and a `case` for array task IDs. It does not call a Python launcher helper.
The earlier helper-based proposal is superseded; its files are unused by this
shell entry point. Other pre-existing launchers were left untouched.

Running `bash scripts/train_four_models.sh` submits one CPU count task (ID 0)
and one GPU array (IDs 1–3, at most two concurrent). Both wait for
`afterok:1579295:1579324`. A small inline check requires step 3 to have passed for
this exact data, backbone and code, then reads the packed counts to calculate
the budget. A failed prerequisite prevents training from starting.

The user confirmed submission. At 2026-10-01 05:46:13 UTC, the launcher submitted
CPU count task `1579395` and GPU head array `1579396` (tasks 1–3, concurrency two),
all owned by `n23zhang`. The verified run directory is
`/u401/n23zhang/rework-data/runs/draft-full-scale-training/train_2026_10_01_05_46_13_UTC/`.
Both were pending on successful step 3 (`1579324`) when checked; the preparation
dependency was already satisfied. No existing jobs were modified or cancelled.
The saved launcher SHA256 is
`cc9f294dea651e43c97f5e94a6f3f0a06dffb3a0a3da5aedd323c3aeeb2050fa`.

The proposed settings remain:

- Fresh global/contextual heads at rank 32 (contextual MLP 128), independent at
  rank 64, seed 1. The rank difference approximately matches trainable capacity.
- Length 1024, K=64 plus residual, batch four, backbone microbatch one.
- All training rows from the 200,000-document subset; one approximate data pass
  per head, with `steps = ceil(train_rows / 4)`. At most three rows are repeated
  to finish the last full batch. This is an initial budget, not convergence.
- AdamW, LR 3e-4, weight decay zero, gradient clipping 1.0, 1,000 warmup updates,
  then constant LR. No warm start from a trained global head.
- All development rows from the 2,000-document subset; evaluate initially,
  every 5,000 updates and at target completion. Keep the best joint denoising
  NLL checkpoint and save resumable state every 500 updates.
- Counts use every prepared training token, smoothing 0.1 and within-row edges.
- GPU tasks use one RTX 6000 Ada, four CPUs and 24 GB host RAM on watgpu108,
  with a 24-hour Slurm limit and checkpointed 23-hour training cap. The separate
  CPU count task gets four CPUs, 24 GB RAM and 12 hours.

All outputs share
`/u401/n23zhang/rework-data/runs/draft-full-scale-training/train_YYYY_MM_DD_HH_MM_SS_UTC/`.
The submitted shell script is copied there, and the existing trainer saves its
code/data/model identities, optimizer/RNG state, `best.pt`, and `last.pt`.
Final held-out evaluation and sampling are deferred. Interrupted or capped runs
must be reported with their actual exposure and resumed before claiming that
all heads completed the common target. Additional seeds and LR searches remain
later development work.

The original truncated `train_four_models.sh` was backed up at
`/u401/n23zhang/rework-data/runs/draft-full-scale-training/launcher-backups/train_four_models-2026_10_01_05_29_39_UTC.sh`.
No existing jobs were changed or cancelled.

## Research sources

- [MDLM Appendix D.4](https://arxiv.org/html/2406.07524v2#A4.SS4): OWT concatenation,
  separators, row boundaries and original validation-tail convention.
- [MDLM original dataloader](https://github.com/kuleshov-group/mdlm/blob/master/dataloader.py):
  operational packing details; the local pinned checkout is the implementation
  reference for this work.
- [BD3-LM Appendix C.1](https://arxiv.org/html/2503.09573v2#A3.SS1): also packs OWT,
  but changes forced row boundaries and retrains its generation baselines. Our
  frozen released MDLM is not that retrained baseline.
- Historical 256 results and the ten-step proposal are in
  `handoffs/mdlm-crf-1024-handoff.md` in the local planning checkout. That
  proposal's numerical budgets remain provisional.
