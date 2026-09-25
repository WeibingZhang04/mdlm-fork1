#!/usr/bin/env python3
"""Evaluate one coupled tuned-backbone/CRF checkpoint without mixing weights.

--control joint uses joint CRF draws; own-marginal samples its nodes separately;
pair-disabled removes pairs but preserves the same tuned backbone. Independent
continued-MDLM checkpoints use the backbone sampler directly. Timing excludes
loading, discarded warmups and the external quality scorer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from chain_crf.backbone import file_sha256
from chain_crf.data import atomic_json, load_token_data
from chain_crf.generation import denoising, generate, token_statistics
from chain_crf.joint import evaluation_components, models_from_checkpoint
from scripts.evaluate_chain_crf import score_gpt2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--control", choices=("joint", "own-marginal", "pair-disabled"), default="joint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--k", type=int, help="Default: checkpoint training K")
    p.add_argument("--inference", choices=("dense", "segments"), default="segments")
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--prefix", default="")
    p.add_argument("--dev-data", type=Path)
    p.add_argument("--dev-examples", type=int, default=128)
    p.add_argument("--denoise-only", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--score-gpt2", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    if min(args.length, args.steps, args.samples, args.batch_size, args.dev_examples, args.threads) < 1:
        raise ValueError("Dimensions, sample counts and threads must be positive")
    if args.sample_offset < 0 or args.warmup < 0 or args.temperature <= 0 or args.k is not None and args.k < 0:
        raise ValueError("Invalid generation configuration")
    if args.denoise_only and args.dev_data is None:
        raise ValueError("--denoise-only requires --dev-data")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists: choose a new directory or explicit --resume")
    torch.set_num_threads(args.threads)
    if torch.device(args.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    backbone, coupled_head = models_from_checkpoint(payload, device=args.device, cache_dir=args.cache_dir)
    head, mode, sampling = evaluation_components(backbone, coupled_head, args.control)
    synthetic = payload["config"]["synthetic_only"]
    if synthetic and (args.prefix or args.score_gpt2):
        raise ValueError("Synthetic fixtures cannot tokenize text or report GPT-2 quality")
    k = payload["config"]["k"] if args.k is None else args.k
    prefix = backbone.tokenizer.encode(args.prefix, add_special_tokens=False) if args.prefix else []
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                     if key not in ("resume", "warmup", "score_gpt2")}
    configuration["effective_k"] = k
    sources = ("scripts/evaluate_chain_joint.py", "scripts/evaluate_chain_crf.py", "chain_crf/joint.py",
               "chain_crf/generation.py", "chain_crf/core.py", "chain_crf/heads.py", "chain_crf/backbone.py",
               "chain_crf/data.py", "models/dit.py", "chain_crf/segments.py")
    manifest = {"config": configuration, "checkpoint_sha256": file_sha256(args.checkpoint),
                "training_identity": payload["identity"], "training_step": payload["step"],
                "trained_tokens": payload["trained_tokens"], "synthetic_only": synthetic,
                "backbone": backbone.provenance, "effective_mode": mode, "effective_sampling": sampling,
                "dev_data_sha256": file_sha256(args.dev_data) if args.dev_data else None,
                "runtime": {"torch_version": str(torch.__version__), "device_type": torch.device(args.device).type,
                            "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None},
                "source_sha256": {name: file_sha256(ROOT/name) for name in sources}}
    del payload
    records_path, manifest_path = args.output/"samples.jsonl", args.output/"manifest.json"
    if records_path.exists() and not manifest_path.exists():
        raise ValueError("Existing samples lack a verified coupled-checkpoint manifest")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume manifest differs from code, checkpoint or configuration")
    args.output.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        atomic_json(manifest, manifest_path)
    if args.dev_data:
        tokens, _, _ = load_token_data(args.dev_data, length=args.length, vocab_size=backbone.vocab_size,
                                       mask_id=backbone.mask_id, max_examples=args.dev_examples)
        result = denoising(backbone, tokens, head, mode, k=k, device=args.device, batch_size=args.batch_size)
        atomic_json({"rows": result, "data_sha256": file_sha256(args.dev_data), "control": args.control},
                    args.output/"denoising.json")
    if args.denoise_only:
        return
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()] if records_path.exists() else []
    if [row["sample_id"] for row in records] != list(range(len(records))):
        raise ValueError("Incomplete or duplicated sample IDs")
    if [row["draw_id"] for row in records] != list(range(args.sample_offset, args.sample_offset+len(records))):
        raise ValueError("Stored draw IDs do not match requested sample offset")
    if len(records) > args.samples:
        raise ValueError("Stored samples exceed requested target")
    kwargs = dict(length=args.length, steps=args.steps, k=k, sampling=sampling, device=args.device,
                  temperature=args.temperature, prefix=prefix, inference=args.inference)
    for index in range(args.warmup):
        generate(backbone, head, mode, batch_size=args.batch_size,
                 sample_offset=args.sample_offset+args.samples+index*args.batch_size, **kwargs)
    # The categorical RNG belongs to an original batch. Replay a partial batch
    # from its original offset, verify saved rows, and only append missing rows.
    start = len(records)//args.batch_size*args.batch_size if len(records) < args.samples else args.samples
    for offset in range(start, args.samples, args.batch_size):
        size = min(args.batch_size, args.samples-offset)
        tokens, timing = generate(backbone, head, mode, batch_size=size,
                                   sample_offset=args.sample_offset+offset, **kwargs)
        batch = []
        for index, ids in enumerate(tokens.cpu().tolist()):
            row = {"sample_id": offset+index, "draw_id": args.sample_offset+offset+index,
                   "token_ids": ids, "prefix_length": len(prefix), "batch_id": offset, "batch_size": size,
                   "text": backbone.tokenizer.decode(ids) if backbone.tokenizer else " ".join(map(str, ids)), **timing}
            for key in ("elapsed_seconds", "backbone_seconds", "sampling_seconds"):
                row[key] /= size
            batch.append(row)
        overlap = min(len(records)-offset, size)
        for old, new in zip(records[offset:offset+overlap], batch[:overlap]):
            keys = ("sample_id", "draw_id", "token_ids", "prefix_length", "batch_id", "batch_size")
            if any(old[key] != new[key] for key in keys):
                raise ValueError("Replayed partial batch differs from saved prefix")
        with records_path.open("a") as f:
            for row in batch[overlap:]:
                f.write(json.dumps(row, allow_nan=False)+"\n")
            f.flush()
        records.extend(batch[overlap:])
        print(json.dumps({"completed": len(records), "target": args.samples,
                          "replayed_rows": overlap, "batch_timing": timing}), flush=True)
    elapsed = sum(row["elapsed_seconds"] for row in records)
    result = {"samples": len(records), "elapsed_seconds": elapsed, "seconds_per_sample": elapsed/len(records),
              "backbone_seconds": sum(row["backbone_seconds"] for row in records),
              "sampling_seconds": sum(row["sampling_seconds"] for row in records),
              "backbone_calls_per_sample": sum(row["backbone_calls"] for row in records)/len(records),
              "generated_tokens_per_second": len(records)*args.length/elapsed, "synthetic_only": synthetic,
              "timing_scope": "completed samples; excludes loading, warmups, scorer and repeated interruption work",
              **token_statistics([row["token_ids"][row["prefix_length"]:] for row in records])}
    atomic_json(result, args.output/"metrics.json")
    if args.score_gpt2:
        del head, coupled_head, backbone
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        atomic_json(score_gpt2(records, args.device), args.output/"gpt2-large.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
