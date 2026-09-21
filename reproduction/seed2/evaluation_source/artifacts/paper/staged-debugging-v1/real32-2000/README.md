# 2,000-update fixed-corruption diagnostic

This extends the original 500-update run, not the fresh-document experiment.
The runner restarts all three heads with the original initialization, optimizer
settings, cached backbone tensors and minibatch RNG sequence. It does not
restore an optimizer checkpoint. All recorded scores at shared checkpoints
0, 100, 200, 300, 400 and 500 reproduce exactly (maximum difference zero;
predeclared tolerance 0.0001 nats per masked token).

The support-bound reports are copied from `../real32`: both source caches have
matching tensor hashes, and the candidate policy is unchanged. The original
500-update artifacts are retained. Full small-head checkpoints are backed up
under the workspace's `output/results/staged-debugging-v1/overfit32-2000-v1`.

Source archive SHA256:
`9b462e6acb98aebc9ec00dbd7c2dc3fcfdc10241755209163a122afbfc709f82`.

One training seed, 32 fixed training corruptions and 32 development documents.
This diagnoses fitting; it does not measure fresh-corruption generalization
or generation quality.
