#!/usr/bin/env python3
"""Evaluate pinned official Tensor-Train weights with the chain study's metrics.

The upstream source and model are unchanged. Default generation is its native
random-order batch_sample. Optional matched reveal sets and FP64 Gumbel draws
are explicitly labeled interventions, not silently substituted reproduction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chain_crf.backbone import file_sha256
from chain_crf.data import atomic_json, canonical_hash
from chain_crf.generation import synchronize, token_statistics

SOURCE_REVISION = "9d0087afd3771ac3e94898ed842858fcc81fb3b0"
MODEL_REVISION = "d0958fa851335ece6c15260ce0025f030673c0fb"
MODEL_SHA256 = "47149e73f7552f39ea9776dbe74d925d25237bcf2ed2e2ec03cdff9d51c82aa4"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
CHECKPOINTS = {
    "rank4": ("8ad8d956af127795686489e9f3496e7a634da18ecf79464df1319668a2a3a7a2", 149999),
    "marginal": ("84fc03cacd818df293602987d4367b8ead7c96539b9b536b748bc86a6cd7079c", 599999),
}


def validate_dimensions(length, steps, samples, batch_size):
    if min(length, steps, samples, batch_size) < 1 or length % steps:
        raise ValueError("Positive dimensions and length divisible by steps required")
    return length // steps


def validate_sample_records(records, *, sample_offset, samples):
    """Resume only the same contiguous, explicitly identified draw range."""
    if sample_offset < 0:
        raise ValueError("Sample offset must be nonnegative")
    if [row.get("sample_id") for row in records] != list(range(len(records))) or len(records) > samples:
        raise ValueError("Invalid existing sample IDs")
    expected = list(range(sample_offset, sample_offset + len(records)))
    if [row.get("draw_id") for row in records] != expected:
        raise ValueError("Stored draw IDs do not match the requested sample offset")


class MatchedReveal:
    """The chain harness's sample-ID reveal sets, sorted for upstream TT cores."""

    def __init__(self, length, batch_size, offset, tokens_per_step):
        self.order = torch.stack([
            torch.randperm(length, generator=torch.Generator().manual_seed(1729 + i))
            for i in range(offset, offset + batch_size)])
        self.cursor, self.k = 0, tokens_per_step
        self.chunks = []

    def __call__(self, ordering, x, K, mask_id, logprobs=None):
        if ordering != "random" or K != self.k or logprobs is not None:
            raise ValueError("Matched reveal only supports the audited random top-k path")
        selected = self.order[:, self.cursor:self.cursor + K].sort(-1).values.to(x.device)
        if selected.shape != (len(x), K) or not x.gather(1, selected).eq(mask_id).all():
            raise ValueError("Reveal schedule exhausted or selected a committed token")
        self.cursor += K
        self.chunks.append(selected.cpu().tolist())
        return selected


@contextmanager
def sampling_precision(modules, precision):
    """Change only Gumbel draw precision, retaining the official sampler code."""
    if precision not in ("native", "float64"):
        raise ValueError("Unknown sampling precision")
    originals = [(module, module.sample) for module in modules]
    if precision == "float64":
        for module, original in originals:
            def precise(logits, temperature=1., _original=original):
                return _original(logits.double(), temperature=temperature)
            module.sample = precise
    try:
        yield
    finally:
        for module, original in originals:
            module.sample = original


def load_official(args):
    source = args.source_root.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True)
    if revision != SOURCE_REVISION or dirty:
        raise ValueError("Official source must be clean at the pinned revision")
    expected_hash, expected_step = CHECKPOINTS[args.arm]
    if file_sha256(args.checkpoint) != expected_hash:
        raise ValueError("Official checkpoint checksum mismatch")
    cache = args.cache_root.resolve()
    backbone_weights = (cache / "hub" / "models--kuleshov-group--mdlm-owt" /
                        "snapshots" / MODEL_REVISION / "model.safetensors")
    if file_sha256(backbone_weights) != MODEL_SHA256:
        raise ValueError("Pinned frozen MDLM backbone checksum mismatch")
    for key, value in {"HF_HOME":str(cache), "HF_HUB_CACHE":str(cache / "hub"),
                       "HF_MODULES_CACHE":str(cache / "modules"), "HF_HUB_OFFLINE":"1",
                       "TRANSFORMERS_OFFLINE":"1", "HF_DATASETS_OFFLINE":"1"}.items():
        os.environ[key] = value
    sys.path.insert(0, str(source))
    upstream = importlib.import_module("generate")
    mdlm = importlib.import_module("mdlm")
    for module in (upstream, mdlm):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError("Official source import collision")
    from omegaconf import OmegaConf
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    class PinnedTokenizer:
        @staticmethod
        def from_pretrained(identifier, **kwargs):
            if identifier != "gpt2":
                raise ValueError("Unexpected upstream tokenizer")
            return AutoTokenizer.from_pretrained("openai-community/gpt2", revision=TOKENIZER_REVISION,
                                                 local_files_only=True, **kwargs)

    class PinnedModel:
        @staticmethod
        def from_pretrained(identifier, **kwargs):
            if identifier != "kuleshov-group/mdlm-owt":
                raise ValueError("Unexpected upstream backbone")
            return AutoModelForMaskedLM.from_pretrained(identifier, revision=MODEL_REVISION,
                                                        local_files_only=True, **kwargs)

    mdlm.AutoTokenizer, mdlm.AutoModelForMaskedLM = PinnedTokenizer, PinnedModel
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["step"] != expected_step:
        raise ValueError("Official checkpoint training step mismatch")
    config = OmegaConf.create(payload["config"])
    if (config.algo.decomp != ("tt" if args.arm == "rank4" else "cp")
            or (args.arm == "rank4" and (config.algo.tt.rank != 4 or not config.algo.tt.marginal_head))):
        raise ValueError("Checkpoint architecture differs from selected arm")
    config.generation.length = args.length
    config.generation.k = args.length // args.steps
    config.generation.batch_size = args.batch_size
    config.generation.total_samples = args.samples
    config.generation.temperature = 1.
    config.generation.ordering = "random"
    config.generation.sampling = "top-k"
    model = mdlm.MDLM(config).to(args.device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise ValueError("Released learned-head state was not loaded exactly")
    model.eval()
    identity = {"source_revision":revision, "checkpoint_sha256":expected_hash,
                "checkpoint_step":expected_step, "backbone_revision":MODEL_REVISION,
                "backbone_sha256":MODEL_SHA256,
                "checkpoint_contains_backbone":any(key.startswith("backbone.")
                                                    for key in payload["model"]),
                "tokenizer_revision":TOKENIZER_REVISION,
                "checkpoint_config_sha256":canonical_hash(payload["config"]),
                "output_dtype":config.algo.output_dtype,
                "runtime":{name:importlib.metadata.version(name)
                           for name in ("torch", "transformers", "flash-attn", "triton", "omegaconf")}}
    return model, config, upstream, identity


@torch.no_grad()
def generate_batch(model, config, upstream, *, sample_offset, sampling, schedule):
    device = next(model.parameters()).device
    torch.manual_seed(2718 + sample_offset)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2718 + sample_offset)
    random.seed(2718 + sample_offset)
    original_pick = upstream.pick_tokens_to_unmask
    recorder = MatchedReveal(config.generation.length, config.generation.batch_size,
                             sample_offset, config.generation.k) if schedule == "matched" else None
    if recorder is not None:
        upstream.pick_tokens_to_unmask = recorder
    calls = []
    hook = model.register_forward_hook(lambda *_: calls.append(1))
    modules = [importlib.import_module("tensor.ttd"), importlib.import_module("tensor.cpd")]
    try:
        synchronize(device)
        start = time.perf_counter()
        with sampling_precision(modules, sampling):
            tokens, average_steps = upstream.batch_sample(model, config, model.mask_id)
        synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        hook.remove()
        upstream.pick_tokens_to_unmask = original_pick
    if tokens.eq(model.mask_id).any() or len(calls) != config.generation.length // config.generation.k:
        raise RuntimeError("Official sampler did not finish at the requested step budget")
    return tokens, {"elapsed_seconds":elapsed, "backbone_calls":len(calls),
                    "reported_steps":float(average_steps),
                    "schedule_sha256":canonical_hash(recorder.chunks) if recorder else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=CHECKPOINTS, default="rank4")
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--sample-offset", type=int, default=0,
                   help="First draw ID; use a separate range for final evaluation after pilot selection")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--schedule", choices=["native", "matched"], default="native")
    p.add_argument("--sampling-precision", choices=["native", "float64"], default="native")
    p.add_argument("--score-gpt2", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    validate_dimensions(args.length, args.steps, args.samples, args.batch_size)
    validate_sample_records([], sample_offset=args.sample_offset, samples=args.samples)
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or explicit --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    model, config, upstream, identity = load_official(args)
    manifest = {"arguments":{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
                             if k not in ("resume", "score_gpt2")},
                "identity":identity, "wrapper_sha256":file_sha256(Path(__file__)),
                "quality_scoring":"raw_GPT2_token_ids_shared_chain_evaluator_not_native_retokenization"}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume identity mismatch")
    else:
        atomic_json(manifest, manifest_path)
    records_path = args.output / "samples.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line] if records_path.exists() else []
    validate_sample_records(records, sample_offset=args.sample_offset, samples=args.samples)
    # Warmup draws are outside the requested range and excluded from metrics.
    generate_batch(model, config, upstream, sample_offset=args.sample_offset + args.samples,
                   sampling=args.sampling_precision, schedule=args.schedule)
    # A native batch uses one seeded RNG stream. If an append was interrupted,
    # replay that original batch and verify its saved prefix before any writes.
    first_batch = (len(records) if len(records) == args.samples else
                   len(records) // args.batch_size * args.batch_size)
    for offset in range(first_batch, args.samples, args.batch_size):
        config.generation.batch_size = min(args.batch_size, args.samples - offset)
        tokens, timing = generate_batch(model, config, upstream, sample_offset=args.sample_offset + offset,
                                       sampling=args.sampling_precision, schedule=args.schedule)
        batch = [{"sample_id":offset+i, "draw_id":args.sample_offset+offset+i,
                  "token_ids":ids, "prefix_length":0,
                  "text":model.tokenizer.decode(ids), "batch_size":len(tokens), **timing,
                  "elapsed_seconds":timing["elapsed_seconds"] / len(tokens)}
                 for i,ids in enumerate(tokens.cpu().tolist())]
        saved_prefix = max(0, len(records) - offset)
        for i in range(saved_prefix):
            if records[offset+i]["token_ids"] != batch[i]["token_ids"]:
                raise ValueError("Replayed partial-batch token IDs differ from the saved prefix")
        batch = batch[saved_prefix:]
        with records_path.open("a") as f:
            for row in batch:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
        records.extend(batch)
        print(json.dumps({"completed":len(records), "target":args.samples, "batch":timing}), flush=True)
    elapsed = sum(row["elapsed_seconds"] for row in records)
    metrics = {"samples":len(records), "elapsed_seconds":elapsed,
               "seconds_per_sample":elapsed / len(records),
               "backbone_calls_per_sample":sum(row["backbone_calls"] for row in records) / len(records),
               **token_statistics([row["token_ids"] for row in records])}
    atomic_json(metrics, args.output / "metrics.json")
    print(json.dumps(metrics), flush=True)
    if args.score_gpt2:
        del model
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        from scripts.evaluate_chain_crf import score_gpt2
        atomic_json(score_gpt2(records, args.device), args.output / "gpt2-large.json")


if __name__ == "__main__":
    main()
