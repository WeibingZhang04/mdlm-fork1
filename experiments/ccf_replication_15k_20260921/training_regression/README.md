# CCF training regression audit - 21 September 2026

**Confirmed: the added monitoring callback changed the numerical training setup.** This was not an equivalent replication of the historical training process, even though the core configurations matched. The speed-optimization changes are not implicated by the controlled replay below. The amount of the generation-PPL reversal attributable to this difference still requires a controlled training rerun.

## What changed

`CampaignObserver.on_train_start` runs a validation probe before the first training batch. That probe runs outside Lightning's BF16 autocast. In `models/dit.py`, `Rotary.forward` computes its position rotations using an autocast-sensitive `einsum` and caches them using only sequence length as its key. The probe populates a **FP32** cache. Subsequent BF16 training reuses it.

The historical training had no startup probe. Its first call occurred inside BF16 training and populated a **BF16** cache. Both old and new source have the same `models/dit.py` bytes. Changing the observer therefore changes backbone predictions without changing any backbone weights. Restoring module training flags and random-number states does not restore this cache.

Generation source inspection shows that both the native MDLM and structured generation paths initialize this cache outside an outer BF16 autocast. Their DIT transformer blocks still use BF16 internally. Historical adapters therefore trained with a BF16 positional cache and generated with a FP32 cache; the fresh monitored runs use FP32 for both. This is a numerical execution difference, not an architecture change or a data-selection change.

## Controlled GPU replay

Job **1557313**, completed successfully on an NVIDIA RTX 6000 Ada Generation, Torch 2.2.2+cu121. Separate R8 FD; identical initial head parameter hashes, ten input-batch hashes, seed, corruption stream and frozen backbone. These are ten forward replays with **no optimizer updates**; they isolate the frozen-backbone discrepancy, not full learned-checkpoint PPL.

| Replay | Cached rotations | Frozen-backbone NLL on batch 10 |
|---|---|---:|
| Old source, no startup probe | BF16 | 5.5327525 |
| g-experiments, no startup probe | BF16 | 5.5327525 |
| g-experiments, existing FP32 probe | FP32 | 3.8848100 |
| g-experiments, probe under BF16 | BF16 | 5.5327525 |

The first and third values exactly match the historical and fresh training logs at update 10. All loss/metric values across all ten batches match exactly among the three BF16-cache cases. The old replay uses the preserved confirmation snapshot; saved original training hashes additionally verify the backbone, diffusion module, decoder and training-loss module. The historical objective/utilities differed from that snapshot in sampling optimizations, documented in `source-evidence.json`.

## What stayed the same; other differences

- Saved train/validation provenance matches byte for byte across all eight arms. Same OpenWebText revision, tokenizer, released frozen backbone, length 1024, batch 4, seed 1, head LR 0.0003, optimizer settings, 50-step warmup and objective weights.
- Historical Separate R8 FD was also a fresh run. Resume history cannot explain its discrepancy. Basic historical arms were resumed in stages, whereas the new arms started fresh; that is an additional difference for Basic.
- New target is 15,000 updates versus 7,000 for historical Separate runs; checkpoint retention and monitoring were added. Compare equal checkpoints before attributing differences to longer training.
- GPU allocations and some evaluation seed blocks differ across historical tables. Prior same-GPU old/new-code replays retain identical generated tokens. This does not validate equivalence of the training process.

## Interpretation and next control

The fresh PPL results remain real measured results, including their losses to MDLM. They must not be presented as a faithful rerun of the old training or as evidence that runtime optimization alone harmed quality. The old improvements are also empirical results, but their training/inference precision mismatch needs disclosure and confirmation under an explicit consistent protocol.

The earlier audit compared one-batch loss/gradients without Lightning's training autocast and replayed old checkpoints at inference; it did not exercise the startup callback plus first mixed-precision batch. It missed this side effect. A smaller conditional loss is not proof of better generation PPL.

For the next controlled experiment, explicitly fix positional-cache precision and record it, test that monitoring on/off cannot change training inputs, logits or gradients, and compare matched training trajectories. Preserve a separately labeled BF16-startup historical reproduction if needed. Do not silently modify active snapshots, overwrite results or change scoring to improve a table. No model or architecture files were changed by this audit. The audit only added a bounded diagnostic job and evidence; no existing jobs were canceled.
