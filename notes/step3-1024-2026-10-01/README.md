# Step 3: completed 1024-token pipeline validation

**Passed on 2026-10-01 (UTC).** Preparation job `1579295` completed with exit
code 0 in 40m02s. Validation job `1579324` completed with exit code 0 in 3m03s
on an NVIDIA RTX 6000 Ada Generation. This record covers preparation and step 3;
the other chat owns the training launcher and Step 4 decisions.

The machine-readable evidence is [validation.json](validation.json), with
`status: passed`, and [data-manifest.json](data-manifest.json). These are copies
of the completed remote records, not newly generated experimental results.

## Prepared data

Remote dataset directory:
`/u401/n23zhang/rework-data/data/draft-full-scale-training/owt-1024-2026-10-01_050424_UTC/`

| Role | Selected documents | 1024-token rows | Total packed tokens | Retained text tokens |
| --- | ---: | ---: | ---: | ---: |
| Train | 200,000 | 220,106 | 225,388,544 | 224,748,333 |
| Development | 2,000 | 2,154 | 2,205,696 | 2,199,389 |
| Held-out evaluation | 100,000 | 110,520 | 113,172,480 | 112,851,441 |

Total packed tokens include document separators and the two row-boundary tokens.
Retained text tokens exclude these inserted tokens. The final incomplete payload
drops 241 train, 428 development, and 326 evaluation tokens, respectively.

Selection uses the pinned source and tokenizer, seed `mdlm-crf-1024-v1`, and
the document-first split/selection policy recorded in
`../mdlm-crf-1024-decisions.md`. The full evaluation tail stays in source order.
Its audit found zero blank documents, zero exact content duplicates, and zero
matches to the explicitly excluded earlier smoke-test documents. Near duplicates
were not checked. The selection rejected 238 ineligible training-window candidates.

Every output checksum was verified. All 332,780 rows were checked for length,
valid clean token IDs, outer BOS/EOS, token hashes and source memberships.
Document hashes are disjoint across roles, with no identical full token rows
across roles. Held-out text was used only for integrity/separation auditing.

## Checks and results

- All 85 selected mathematical, head, generation, evaluator, training, packing,
  and segmented-inference tests passed in 4.56 seconds inside the validation job.
- All released MDLM parameters remained frozen, had no gradients, and had
  exactly unchanged state hashes after each head's updates. The optimizer
  contained only that head's parameters. All three learned heads changed.
- Identity global/independent heads, an explicitly neutralized contextual head,
  and zero-strength count factors recovered the expected MDLM likelihood:
  observed NLL error per masked token was zero; maximum candidate marginal
  error was `7.092952728271484e-06`. Fresh contextual initialization remains
  nonzero in the actual training profile.
- GPU chain partition functions and marginals matched enumeration of all 27
  states at `rtol=atol=1e-12`. Across 24,000 joint samples, maximum absolute
  frequency error was `0.0040031609157424874` (threshold `0.015`).
- Saved head checkpoints loaded exactly through the real evaluator. Restoring
  the optimizer, scheduler, data stream and masking RNG reproduced the next
  update with maximum parameter error zero for every head.
- The real training CLI completed two updates, then resumed to update three.
- MDLM, count, global, contextual and independent each generated two complete
  1024-token sequences at 16 sampling steps. IDs, lengths and counts checked
  out. A deliberately interrupted partial batch resumed with identical token
  sequences for every method. Reference outputs were preserved before truncating
  the disposable test outputs.
- GPT-2-large scored the same ten saved sequences twice: 10,230 next-token
  targets, identical mean NLL, and matching per-sequence scores within tolerance.
- All recorded source hashes were identical at the start and end of validation.

## Short training profiles

Each head used eight diagnostic updates, batch four, backbone microbatch one,
K=64, AdamW at constant LR `3e-4`, and gradient clipping at 1.0. Medians exclude
the first update. These are feasibility measurements, not convergence results
or estimates of complete training time including validation and checkpoints.

| Head | Rank | Trainable parameters | Median seconds/update | Peak allocated GPU GiB |
| --- | ---: | ---: | ---: | ---: |
| Global | 32 | 3,216,512 | 0.878 | 5.348 |
| Contextual | 32 | 3,419,040 | 0.890 | 5.358 |
| Independent | 64 | 3,267,328 | 0.401 | 5.349 |

Peak reserved GPU memory was approximately 6.15–6.26 GiB. Full precision and
backbone provenance are in `validation.json`; the checkpoint is the raw released
MDLM weights without EMA. No MDLM fine-tuning occurred.

## Evidence locations and identity

Remote validation run:
`/u401/n23zhang/rework-data/runs/draft-full-scale-training/step3-owt-1024-2026-10-01_051416_UTC/`

Within that directory:

- `validation.log` and `launch.json`: execution and submission records.
- `results/validation.json`: authoritative success marker, data/model/code
  identities, complete profiles, generation checks and limitations.
- `results/reference-data/`: 32 train and four development rows with source
  memberships, original file hashes and matching JSONL records.
- `results/neutral-reference.pt`, `*-resume-reference.pt`, and
  `*-validation-head.pt`: fixed inputs and diagnostic checkpoints.
- `results/generation-*/samples.reference.jsonl`: complete reference samples.
- `results/scorer-first.json` and `scorer-repeat.json`: repeated scoring results.
- `results/unit-tests.log` and individual CLI logs: detailed check outputs.

Dataset manifest SHA256:
`b709b877288a22c0ab7073b18a794e59032ebdd01e64ea084b61be4d4c72bfc9`

Validation script SHA256:
`33185aac41e38fa15624a3d4a38cb566d60cd57ec462b5c9f3cddc28b118e16d`

The script is `scripts/validate_chain_1024.py`. Its recorded command is in
`launch.json`; a repeat must use a fresh output directory. A first, early GPU
attempt exposed an unsupported CUDA int64 matrix-vector operation in the
sampling-test histogram helper. It was replaced with equivalent elementwise
integer arithmetic. The failed log and original script snapshot were retained;
the corrected early check and complete validation both passed. Model mathematics
and weights were unchanged by that helper fix.

## Interpretation limits

The diagnostic heads and counts fitted on eight fixture rows are test artifacts;
they are not the main trained models. This validation establishes pipeline
correctness and feasibility for the tested settings, not model quality or
convergence. It exercises the current matched sampler and raw-token-ID scorer;
the final sampling/scoring protocol remains a later decision, and native
`ddpm_cache` was not exercised here. The pinned OWT source follows the published
100,000-document tail rule, but its exact historical alignment with released
MDLM pretraining is not independently authenticated. Packing retains one stream
per role and drops one tail per role, rather than reproducing historical
map-batch remainder drops.
