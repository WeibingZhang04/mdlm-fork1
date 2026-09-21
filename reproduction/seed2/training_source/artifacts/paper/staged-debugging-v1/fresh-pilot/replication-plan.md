# Unchanged-seed replication, 2026-09-08

Completed: all three seeds finished training and held-out scoring before the
fixed GPU deadline. The authenticated combined results and full rate profile
are in `../fresh-three-seeds`. The unchanged plan and recovery history below
are retained as provenance, not a pending launch queue.

The completed seed-1 reserved-test comparison shows useful dependence for the
separate-factor head (+0.002837 nats per masked token, paired document 95% CI
[0.001925, 0.003746]), but no overall advantage over the trained unary adapter
(-0.000129, [-0.002618, 0.002322]). At 90% masking it improves on the unary
adapter by 0.008257 [0.004239, 0.012153]; the unary adapter wins at 25% and
50%. These are the pilot results, not outcomes for additional seeds.

Replicate the entire experiment at training seeds 2 and 3. Keep all four mask
rates, all three arms, the 2,048/64/128 source-document split, 1,000 updates,
learning rate 0.0003, and the original development selection rule unchanged.
Do not restrict evaluation to 90% masking or select a checkpoint by dependence
gain. Each seed has fresh training order and masks; masks stay paired across
its arms. Report every seed and mask rate, whether it agrees with seed 1 or not.

This replication was chosen after reading the seed-1 test result. The original
test artifact and its seal remain immutable. New seeds select checkpoints on
development data only. Each completed replica is sealed and scored separately.
A separate aggregate seal binds the union of the three choices; the aggregate
export combines authenticated existing test shards without re-scoring seed 1,
then uses the implemented crossed seed/document bootstrap. This is replication on the same
reserved documents, not a new independent test corpus or a new tuning search.

The existing L4 VM and dedicated experiment disk are reused. The original
absolute shutdown deadline, 2026-09-08 04:49:24 UTC, is unchanged. A bounded
sequential queue runs seeds 2 and 3; an interrupted or failed run is retained
and cannot pass the complete-run selection sealer.

The first queue was preempted at 03:01:44 UTC during seed 2's initial
development evaluation, before any training updates were saved. Its
step-zero checkpoint and partial outputs are retained. The retry uses fresh
`retry1-v1` output directories and the same initialization and settings;
this is a restart from initialization, not an optimizer resume. The original
absolute shutdown deadline is restored after VM restart.

## Checkpoint recovery and parallel completion

The retry was preempted at 03:31:51 UTC after saving update 500. Its checkpoint
SHA256 is `ff5f5675e77dcc41ba53f992c5284242a8f9b713d9955c9c543565f150e071f7`.
The checkpoint includes optimizer, minibatch, corruption, and random states.
No learning setting changes in the recovery.

The stopped VM was copied to machine image
`diffusion-staged-recovery-20260908t0338`. Seed 2 resumed into the new
`fresh-replication-seed2-resume-v1` directory on on-demand L4 worker
`diffusion-staged-debug-l4-c-20260908` in `us-central1-c`. Its copied 300 GB
data disk is `diffusion-staged-debug-l4-c-20260908-1`; its device name and
mount path match the original experiment. The original disk remains intact.
All earlier checkpoints and the interrupted parent are preserved.

After capacity failures for a second copied worker in zones c and b, the
original experiment VM obtained on-demand capacity in `us-central1-a`.
Seed 3 starts from initialization in `fresh-replication-seed3-retry1-v1` on
that VM. Both workers have an absolute guest shutdown timer at
2026-09-08 04:49:24 UTC. The training implementation remains `a8d202a`;
launchers and analysis use separate authenticated checkouts. Individual
replica sealing and test scoring use `406df18`.
