# MDLM contextual coupling forest: visualization branch

This branch keeps the current OpenWebText contextual coupling forest (CCF)
training and inference path and removes historical experiment runners,
alternative backbones, reports, and generated artifacts. It is intended to
make the active architecture easy to inspect while remaining runnable.

## Architecture map

```text
main.py
  -> dataloader.py                         pinned OpenWebText batches
  -> diffusion.py                         noising, loss, reverse sampling
       -> models/dit.py                    frozen MDLM denoiser
       -> models/structured_decoder.py     contextual coupling forest head
       -> structured_training.py           structured training loss
       -> structured_objective.py          exact forest likelihood/sampling
       -> structured_utils.py              tree inference primitives
```

The current experiment freezes the released MDLM-OWT DiT backbone and trains
only the structured decoder. Array task `0` uses fixed topology and task `1`
uses dynamic topology. Both use dynamic pair factors. Rank 8 is the default;
rank 16 is selected with `CCF_FACTOR_RANK=16`.

## Five experiment scripts

1. `scripts/prepare_released_mdlm_owt.py` downloads, verifies, and wraps the
   pinned released backbone.
2. `scripts/run_ccf_separate_7k.sh` submits or runs the two matched 7,000-step
   training arms, then exports and samples from each completed model.
3. `scripts/ccf_separate_resume.py` selects and validates the newest compatible
   checkpoint when a training job resumes.
4. `scripts/export_structured_adapter.py` exports the learned head to
   safetensors with a provenance manifest.
5. `scripts/run_generation_pilot.py` runs paired factorized and structured
   generation/infilling and writes samples, metrics, config, and provenance.

## Run

Create the environment with `requirements.yaml`, activate it, and set the
shared cache/run root. Prepare the released backbone once:

```bash
python scripts/prepare_released_mdlm_owt.py \
  --output "$CCF_CACHE_ROOT/checkpoints/mdlm-owt-backbone.pt"
```

Submit the rank-8 screen:

```bash
CCF_CACHE_ROOT=/path/to/cache sbatch scripts/run_ccf_separate_7k.sh
```

Submit the rank-16 screen:

```bash
CCF_CACHE_ROOT=/path/to/cache CCF_FACTOR_RANK=16 \
  sbatch --export=ALL scripts/run_ccf_separate_7k.sh
```

The runner records the commit, working-tree status, source hashes, backbone
hash, checkpoint hash, adapter hash, manifest hash, resolved configuration,
and generated samples. It refuses incompatible resume checkpoints and does not
overwrite completed results.

Set `MDLM_DEBUG_VALIDATION=1` to enable expensive whole-tensor diagnostic
checks. Shape, dtype, device, topology, checkpoint, and final finite-result
checks remain active in normal training and inference.
