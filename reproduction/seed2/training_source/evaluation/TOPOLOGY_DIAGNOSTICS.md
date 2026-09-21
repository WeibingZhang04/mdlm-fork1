# Contextual-forest topology diagnostics

This is an eval-only, descriptive protocol for recording the exact selected
forest. It does not estimate likelihood, generation quality, or a causal
quality effect. The frozen protocol is
`configs/evaluation/contextual-forest-topology-diagnostics-v1.json`.

## Frozen design

The source grid contains the first 32 units in pinned document-local
evaluation order for WikiText-103 and arXiv and the `dynamic_dynamic` adapter
at candidate support K=128 from training seeds 1, 2, and 3. Each
dataset-by-training-seed job emits the
exact Cartesian product

```
32 sources x 3 corruption seeds x 5 requested times x 5 interventions
= 2,400 records
```

The six-job union therefore contains exactly 14,400 records. The ordered
source-selection artifact records indices `0..31`; a set with the right size
but the wrong examples or order is invalid.

The time grid is an absorbing-mask probability grid, not the raw scalar fed
to the topology head. The head receives `-log1p(-p)`, matching the log-linear
absorbing diffusion's sigma conditioning. For each source unit and corruption
seed, the evaluator creates one private vector of deterministic SHA-256-derived
53-bit integers. Position `i` is masked exactly when its integer is below
`floor(p * 2^53)`. The same vector is reused over the complete probability grid
and every adapter training seed. The corrupted-token commitment, attention
mask, active mask, and base-noise commitment are recorded separately. Active
sets must therefore be nested as `p` increases. `active_nodes` is exactly the
increasing set of positions where the corrupted token is the mask token and
the attention mask is true.

The backbone hidden states and unary logits are computed once for the
requested corrupted input and requested probability's transformed sigma.
Time interventions reuse those tensors and change only the explicit
probability-derived sigma input to the dynamic topology head. The model is in
deterministic evaluation mode under
`torch.no_grad`, with dropout disabled:

* `learned`: requested topology-head time;
* `matched_permuted`: a frozen bijection maps the learned undirected edges
  within the same active-node set; it does not rerun the model;
* `fixed_time`: topology-head time 0.5;
* `zero_time`: topology-head time 0.0;
* `timestep_shuffled`: a deterministic derangement of the requested time
  grid. The donor record chooses only the effective time; its edge set must
  never be copied into the requested corruption.

When requested time is 0.5, `fixed_time` must reproduce `learned` exactly.
Every edge is copied from `edge_index[edge_mask]`, endpoint-canonicalized and
lexicographically sorted. Forests must be acyclic, use only active endpoints,
and respect the frozen component cap of 32.

## Provenance and two-stage binding

Record bundles contain only commitments knowable before or during model
execution: protocol, compiled plan and job specification, clean repository,
adapter bytes and export manifest, data configuration and runtime provenance,
evaluator source, and ordered source selection. They deliberately do **not**
contain a success-marker hash. A compiled-job success marker is created only
after output hashes are known, so embedding each hash in the other would form
an impossible cycle.

The authoritative aggregator accepts a dedicated compiled topology plan
directory, not an arbitrary list of manifests. The dedicated compiler derives
that plan from the authenticated K=128 confirmation plan and copies only the
three required train/export dependency pairs. Their execution digests are
unchanged, so completed adapters are reused, while six new topology jobs are
added. Runtime validation and aggregation first reconstruct the K=128 parent
in memory from the repository-trusted manifest through its registered
promotion verifier. They then rerun the topology derivation and require
canonical-JSON equality for the complete parent and derived plans and every
job, including argv, dependencies, external inputs, and outputs. Neither
authoritative replay step writes or replaces plan artifacts. The aggregator:

1. validates the current clean checkout, dedicated plan, frozen protocol, and
   rehashed parent K=128 compiled plan;
2. discovers exactly one topology eval job per dataset and training seed;
3. recursively validates adapter-export dependencies, job specifications,
   success markers, and every required output hash;
4. rehashes and validates the adapter plus export manifest;
5. validates data-config, dataset-provenance, and GPU-exclusivity evidence;
6. verifies the ordered source-selection artifact and all 14,400 raw cells;
7. enforces identical corruption inputs across training seeds; and
8. writes a post-run source-integrity commitment containing the real marker
   and output hashes.

Topology eval jobs must name these required outputs exactly:

* `topology_records` -> `topology_records.jsonl`;
* `topology_record_manifest` -> `topology_records.manifest.json`;
* `topology_source_selection` -> the ordered source artifact;
* `dataset_provenance` -> pinned runtime data provenance; and
* `gpu_exclusivity` -> `gpu_exclusivity.json`.

The job must have exactly one matching adapter-export dependency with outputs
named `adapter` and `adapter_manifest`.

Compile the runnable six-job diagnostic grid after the promoted K=128
confirmation plan exists:

```bash
python scripts/compile_topology_diagnostics.py \
  --source-plan-dir /path/to/candidate-k-128-confirmation-plan \
  --output-dir /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/plans/topology-diagnostics
```

Run it through the normal fail-closed launcher (the copied train/export jobs
are skipped only when their authenticated markers and outputs still verify):

```bash
python scripts/run_compiled_job.py \
  --plan-dir /mnt/contextual-forest/experiments/contextual-forest-expansion-v1/plans/topology-diagnostics \
  --all
```

`scripts/run_topology_diagnostics.py` is the compiled evaluator. It loads the
pinned document-local dataset in map order, literally selects rows `0..31`,
loads the released backbone and strict adapter export, and invokes
`emit_topology_records`. Each requested corruption batch executes the backbone
once, then executes the topology head at learned, fixed, zero, and shuffled
time inputs while reusing the exact same hidden states and unary logits. The
matched-permuted control is derived without a model call. Each of the six jobs
writes and self-validates 2,400 records before its success marker can exist.
The complete CUDA phase holds the submission-wide nonblocking `flock` at
`/mnt/contextual-forest/experiments/.submission-gpu.lock` and reuses the
Tensor-Train harness's one-second foreign-compute-PID monitor. A contended
lock, `nvidia-smi` query failure, or any foreign CUDA PID fails the fresh
attempt. The pre/during/post monitor evidence is a required, post-run-hashed
output and must show no foreign process or monitor error.

Aggregate a completed plan with:

```bash
python scripts/aggregate_topology_diagnostics.py \
  --protocol configs/evaluation/contextual-forest-topology-diagnostics-v1.json \
  --plan-dir /path/to/compiled-topology-plan \
  --output /fresh/path/topology-diagnostics.json
```

Replay an existing analysis from the same authenticated plan with
`--verify-analysis` instead of `--output`. Replay compares the parsed,
hash-committed scientific content; it does not claim that arbitrary JSON
whitespace is byte-identical.

## Metric definitions

The natural-order control connects consecutive **active** positions. A
natural-chain edge can still be nonlocal in original token coordinates when
there is a gap between active positions. A nonlocal edge has absolute token
position distance greater than one.

Across-corruption Jaccard compares learned edge sets for the same source,
training seed, and requested time under two independent corruption seeds.
Across-time Jaccard compares the same source, training seed, and base-noise
draw at two requested times. Because active sets can change, both analyses
report all-edge Jaccard, Jaccard after restricting both forests to shared
active nodes, and shared-active-node fraction. Thus across-time Jaccard is a
trajectory statistic; the same-context fixed/zero/shuffled interventions are
the more isolated topology-head time diagnostic. Empty restricted edge unions
are excluded and their reduced counts remain visible.

Because corruptions are required to be bit-identical across trained adapters,
the analysis also reports learned-edge Jaccard across training seeds at the
same source, corruption, and requested time. Pooled learned summaries are
accompanied by dataset-specific summaries to prevent a pooled corpus mixture
from hiding domain differences.

The output also includes edge-distance histograms, active-order chain
precision/recall, nonlocal-edge fraction, component sizes, depth from each
component's minimum-position root, component diameter, and learned-versus-
intervention edge retention. All quantities replay from canonical raw edge
lists on CPU.
