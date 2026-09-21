# Matched length-1024 Tensor-Train comparison

This directory contains the complete fail-closed descriptive comparison between
the promoted seed-1, K=128 dynamic/dynamic Contextual Coupling Forest (CCF)
adapter and the released marginal and rank-4 Tensor-Train OpenWebText systems.
All systems use 256 unconditional length-1024 samples at 8, 16, and 32 NFEs,
generation seed 260703, batch size 1, the same immutable GPT-2-large evaluator,
and exclusive NVIDIA L4 measurements.

`ccf-union.json` verifies all 32 atomic CCF shards, 768 generation records, no
unresolved masks, the clean source revision, the released-backbone bytes, the
structured-adapter bytes and manifest, and machine-readable evidence that the
adapter came from the frozen four-arm factorial plan. Its SHA256 is
`6717d8ec0847ec914debce1f8ed6d112d0908668431a0018bcae50f9f9f5a9d2`.

`comparison.json` revalidates that union against the completed released
Tensor-Train matrix. Its SHA256 is
`95ab62f5d073abb88fb627ce490498e8b68ebc52b17d5c267c7e8cc991958cd4`.
At 8, 16, and 32 NFEs, CCF evaluator NLL is 6.8140, 5.8528, and 5.1251,
respectively. Relative to released rank-4 Tensor-Train, CCF is worse by 0.3467
and 0.1102 nats at 8 and 16 NFEs, then better by 0.0878 nats at 32 NFEs. The
rank-4 system is 6.43x, 11.52x, and 18.30x faster in samples per second.

The comparison aligns nominal NFE, length, sample count, seed, evaluator,
batch sizes, and GPU model. It does not pair model-native reverse schedules or
checkpoint-training histories, so the differences are descriptive system
comparisons rather than causal effects of output parameterization.
