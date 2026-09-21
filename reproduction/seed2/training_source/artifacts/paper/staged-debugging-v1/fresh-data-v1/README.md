# Fresh-corruption pilot

The fixed-corruption run checks fitting. This next stage checks transfer to new
documents and new masks. The frozen MDLM-OWT encoder feeds three heads on exactly
the same online-corrupted examples: shared factors (rank 16), separate endpoint
factors (rank 8), and a candidate-only unary adapter (rank 16). The pair heads
each train 840,640 parameters; the unary adapter trains 827,312. All use top-64
backbone candidates, the same residual conditionals and a natural-order chain
for the pair models.

The pilot uses one training seed, 2,048 training documents, 64 development
documents and 128 reserved test documents, each represented by its first 128
cached tokens. Selection follows source order among eligible chunk-zero
documents. Train, development, test, and the 96 earlier debugging documents
are disjoint in source identity. Duplicate content and token prefixes are
removed across the new splits. The new held-out source window starts well
after the prefix used by earlier benchmark evaluation (see prefix-audit.json).

Training uses 1,000 AdamW updates, batch size four, learning rate 0.0003,
weight decay zero and gradient norm clipping at one. One of four scheduled
mask rates (0.25, 0.50, 0.75, 0.90) is drawn per update, then fresh uniformly
chosen fixed-count masks are drawn for each example. Rounded active counts
are 32, 64, 96 and 115; sigma uses the scheduled rate. All three heads receive
the same encoded minibatch. The encoder processes one example per call in
both training and evaluation. Pair factors use a 100-update static warmup;
the unary adapter is contextual throughout.

Every 100 updates, including step zero, we evaluate the same 64 development
documents at all four rates. Each head selects the checkpoint with lowest
pooled masked-token NLL; ties choose the earlier checkpoint. The selection is
sealed before any reserved-test scoring. Test masks are generated with seed
training_seed + 1,900,000 + rate_index. Selected head forwards retain FP32;
forest inference and residual normalization use FP64.

The primary held-out contrast is separate factors versus the trained unary
adapter. The dependence check compares the joint model with its own exact
marginal product. We report both, together with the full NLL decomposition and
all four mask rates, using 5,000 paired document-bootstrap draws. All masks
for a document remain in the same resampled cluster. These intervals describe
document variation for one training seed, not variation across training runs.
Neither conditional-likelihood contrast by itself establishes generation
quality. Fixed-group generation and additional seeds follow only if this
stage shows useful held-out dependence.

The dataset manifest pins cache provenance, every selected source ID and token
prefix hash, and all input JSONL byte hashes. Raw training and reserved-test
tokens stay on the dedicated experiment disk; this directory stores provenance.
