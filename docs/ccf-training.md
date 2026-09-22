# CCF four-arm training

Select one preset with `+experiment=ccf/<arm>`:

| Arm | Topology | Pair factors | Topology teacher weight |
| --- | --- | --- | --- |
| `static_static` | fixed | fixed | 0.0 |
| `fixed_dynamic` | fixed | dynamic | 0.0 |
| `dynamic_fixed` | dynamic | fixed | 0.1 |
| `dynamic_dynamic` | dynamic | dynamic | 0.1 |

The four-arm launcher is `scripts/train_four_ccf_matched_8k.sh`: K=128, rank16,
length1024, frozen raw MDLM backbone, shared endpoint tables, original additive
FiLM, head LR0.0003, no EMA, batch4, 50 warmup steps, and 8000 updates. Dynamic
topology uses gold-reveal supervision. The standalone experiment presets still
default to 1000 updates; the launcher overrides that limit. The normal MDLM
default configuration is unchanged.

The presets deliberately leave `data.train`, `data.valid`, and `data.cache_dir`
mandatory, along with `model.structured_decoder.training.backbone_checkpoint`.
The launcher selects `train_openwebtext_pinned`, which records dataset and
tokenizer revisions but does not establish MDLM's unrecorded historical
snapshots. It saves full Lightning checkpoints; adapter-only export is not part
of this launcher.

A training command has this shape (replace each placeholder deliberately):

```bash
python main.py +experiment=ccf/static_static \
  data.train=TRAIN_SOURCE data.valid=VALID_SOURCE data.cache_dir=/absolute/cache \
  model.structured_decoder.training.backbone_checkpoint=/absolute/backbone.pt \
  checkpointing.save_dir=/absolute/new-run hydra.run.dir=/absolute/new-run/hydra
```

No four-arm experiment is launched by adding these configs. Use a fresh run directory
per arm/seed. Resume is disabled by default because paired private-RNG/data resumption
is not yet verified. A one-device strategy is selected, with unused parameters allowed
for the arm-specific inactive parts. Multi-GPU training still needs separate verification.

## Released backbone

`scripts/prepare_released_mdlm_owt.py` is restored verbatim from2051502. It verifies the
pinned safetensors bytes and tensor schema, then packages the raw `backbone.*` weights
for the existing strict loader. It does not run Hugging Face remote model code or add EMA.
Use an already cached source to avoid downloads:

```bash
python scripts/prepare_released_mdlm_owt.py \
  --source /absolute/model.safetensors --output /absolute/new-backbone.pt
```

## Short full-training smoke

From the repo root on one allocated CUDA GPU, with the `mdlm` environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python scripts/smoke_ccf_training.py \
  --source /absolute/cached/released/model.safetensors \
  --tokenizer /absolute/cached/gpt2/snapshot \
  --output /absolute/new-smoke-directory
```

This runs the real `main._train`, cached-data loader, frozen released backbone,
CCF objective, automatic Lightning backward/AdamW, BF16 mixed precision, validation,
and checkpoint callbacks for all four arms. It uses four synthetic training sentences
and two distinct synthetic validation sentences: a plumbing fixture, not an accuracy
benchmark. Smoke overrides are length32, batch2, two optimizer updates, no LR warmup,
one validation batch. K=128/rank16 and the released backbone dimensions stay unchanged.
No training/loss/backbone method is mocked. An observer callback checks finite gradients,
head updates, exact frozen-backbone state, teacher gating, matched masks/times/data across
arms, and strict checkpoint reload. This does not test continued training from a checkpoint
or exact mid-epoch RNG/data resume.

Outputs include resolved configs, per-arm audit JSON and Lightning logs, and `summary.json`.
The output directory must be new. By default, the newly created large checkpoints and
backbone wrapper are deleted **after** successful verification; use `--keep-checkpoints`
to retain them. Source caches are read only. Interrupted/failed runs may retain their
own partial artifacts for diagnosis. No existing run directory is overwritten.

## Metrics and checkpoints

Checkpoint selection monitors `val/conditional_nll_per_masked_token`, not diffusion
ELBO/perplexity. Additional metrics live under `train/structured/` and `val/structured/`.
Generation evaluation stays disabled until CCF joint inference is connected. In particular,
a successful training smoke does not validate the existing ordinary-MDLM generation route
for CCF checkpoints.
