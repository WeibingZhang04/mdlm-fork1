# Frozen-backbone real-text confirmation

The structured adapter was trained for 1,000 updates on streaming OpenWebText
with the released raw MDLM-OWT backbone frozen.  The final checkpoint, selected
by step rather than validation performance, was evaluated on the finite
WikiText-103 validation split with sequence length 1024.

`independent-vs-structured.json` and `fixed-vs-structured.json` contain five
paired validation-corruption seeds (201--205), per-seed conditional denoising
NLL, 20,000-resample paired-seed bootstrap intervals, exact SHA-256 pairing
commitments, input-file hashes, and protocol metadata.  Each commitment covers
the ordered clean token IDs, attention mask, sampled times, corrupted token
IDs, and active mask.  Pairing diagnostics match exactly within every pair.

`topology-diagnostic.json` records a separate seed-201 run used only to verify
that all teacher-derived topology targets were defined.  Its NLL is not
compared with the paired arms because enabling the diagnostic changes the
random stream.

`structured-adapter.safetensors` contains the 21 learned structured-head
tensors (984,417 FP32 parameters) with the 131 bitwise-frozen backbone tensors
omitted.  Its manifest pins the source checkpoint and released backbone by
SHA-256, records every tensor name/shape/dtype, and names the strict loader.
`structured-adapter-preflight.json` records a successful strict rehydration and
length-1024 structured forward/inference pass atop that released backbone.
`structured-adapter-replay.json` records that the portable path reproduces the
full-checkpoint seed-201 metrics CSV and pairing digest byte for byte.

Scientific scope: these artifacts establish conditional denoising NLL per
masked token for one training checkpoint and five validation-corruption seeds.
They do not establish a diffusion ELBO, data likelihood, perplexity,
generation quality, or variation across independent training seeds.
