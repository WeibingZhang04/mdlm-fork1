#!/usr/bin/env python3
"""Continue MDLM alone or jointly with a fresh zero-pair contextual chain.

Both arms use the same data/corruption streams, ordinary dropout, 1/t weights,
effective batch size and token budget. This is not the frozen-head trainer.
Example: --arm contextual --data DATA --output NEW_RUN --steps 10000.
Run the independent control separately with the same settings and --arm
independent. --steps is a total optimizer-step target, including on resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from chain_crf.backbone import file_sha256
from chain_crf.data import BatchStream, assert_disjoint, atomic_json, atomic_torch_save, canonical_hash, load_token_data
from chain_crf.generation import synchronize
from chain_crf.joint import (TrainableMDLM, SyntheticTrainableMDLM, checkpoint_payload,
                             evaluate_joint, make_joint_head, make_optimizer,
                             restore_training, train_update)

SOURCE_FILES = ("chain_crf/joint.py", "chain_crf/backbone.py", "chain_crf/core.py",
                "chain_crf/heads.py", "chain_crf/data.py", "chain_crf/generation.py",
                "scripts/train_chain_joint.py", "scripts/prepare_released_mdlm_owt.py",
                "models/dit.py", "configs/model/small.yaml")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("independent", "contextual"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--dev-data", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release-checkpoint", type=Path, help="Verified initial release, never a tuned checkpoint")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--resume", type=Path, help="Coupled joint checkpoint with training state")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4, help="Effective batch per optimizer update")
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--mlp-size", type=int, default=128)
    p.add_argument("--gate-init-std", type=float, default=.01)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--gradient-clip", type=float, default=1.)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--max-train-examples", type=int)
    p.add_argument("--max-dev-examples", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--synthetic-backbone", action="store_true", help="Offline plumbing tests only")
    p.add_argument("--synthetic-vocab-size", type=int, default=17)
    args = p.parse_args(argv)
    if min(args.steps, args.batch_size, args.microbatch_size, args.length, args.rank,
           args.mlp_size, args.eval_every, args.save_every, args.threads) < 1 or args.k < 0:
        raise ValueError("Positive dimensions/intervals and nonnegative K required")
    if (min(args.backbone_lr, args.head_lr, args.gate_init_std, args.gradient_clip) <= 0
            or args.warmup_steps < 0 or args.max_seconds < 0):
        raise ValueError("Invalid learning configuration")
    if args.max_train_examples is not None and args.max_train_examples < 1 or args.max_dev_examples < 1:
        raise ValueError("Positive data selection sizes required")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Output is nonempty: choose a new run or explicit --resume")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    backbone = (SyntheticTrainableMDLM(vocab_size=args.synthetic_vocab_size, device=args.device)
                if args.synthetic_backbone else TrainableMDLM.from_release(
                    args.release_checkpoint, device=args.device, cache_dir=args.cache_dir))
    train_path = args.data / "train.pt" if args.data.is_dir() else args.data
    dev_path = args.dev_data or (args.data / "dev.pt" if args.data.is_dir() else None)
    if dev_path is None:
        raise ValueError("--dev-data required when --data is a file")
    train, train_docs, train_source = load_token_data(train_path, length=args.length,
        vocab_size=backbone.vocab_size, mask_id=backbone.mask_id, max_examples=args.max_train_examples)
    dev, dev_docs, dev_source = load_token_data(dev_path, length=args.length,
        vocab_size=backbone.vocab_size, mask_id=backbone.mask_id, max_examples=args.max_dev_examples)
    assert_disjoint(train, train_docs, dev, dev_docs)
    torch.manual_seed(args.seed)
    head = make_joint_head(args.arm, backbone, args.rank, args.mlp_size, args.gate_init_std)
    # Head construction must not advance the backbone's dropout RNG in one
    # arm only. Corruptions use their own CPU generator in both arms.
    torch.manual_seed(args.seed + 3000)
    config = {key: getattr(args, key) for key in (
        "arm", "batch_size", "microbatch_size", "length", "k", "rank", "mlp_size",
        "gate_init_std", "backbone_lr", "head_lr", "warmup_steps", "gradient_clip",
        "seed", "eval_every", "save_every", "threads")}
    config.update(loss="sum_sequence_nll_over_t / effective_batch_tokens",
                  corruption="antithetic_uniform_t_[.001,1); Bernoulli(.999*t); allow_zero_masks",
                  clipping="separate_parameter_group_norms", weight_decay=0., ema=False,
                  synthetic_only=args.synthetic_backbone)
    identity = {"config": config, "model_spec": backbone.model_spec,
                "initialization_backbone": backbone.initialization,
                "train_sha256": file_sha256(train_path), "dev_sha256": file_sha256(dev_path),
                "train_examples": len(train), "dev_examples": len(dev),
                "source_sha256": {name: file_sha256(ROOT/name) for name in SOURCE_FILES},
                "torch_version": str(torch.__version__), "device_type": torch.device(args.device).type,
                "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None}
    optimizer, scheduler = make_optimizer(backbone, head, backbone_lr=args.backbone_lr,
                                          head_lr=args.head_lr, warmup_steps=args.warmup_steps)
    stream = BatchStream(train, args.batch_size, args.seed+1000)
    mask_rng = torch.Generator().manual_seed(args.seed+2000)
    step, best = 0, None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        step, best = restore_training(payload, backbone, head, optimizer, scheduler, stream, mask_rng, identity)
        del payload
        if step > args.steps:
            raise ValueError("Total target steps precede resumed step")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"identity": identity, "identity_sha256": canonical_hash(identity),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "train_source": train_source, "dev_source": dev_source,
                "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
                "head_parameters": sum(p.numel() for p in head.parameters()) if head is not None else 0,
                "backbone_frozen": False, "test_data_access": False,
                "objective_claim": "ordinary MDLM for independent; time-weighted joint denoising for contextual, not an ELBO",
                "candidate_policy": "no gold insertion; differentiable full residual mass; discrete top-K identities",
                "resume_sha256": file_sha256(args.resume) if args.resume else None}
    atomic_json(protocol, args.output/"protocol.json")
    stopped = []
    previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in previous_handlers:
        signal.signal(s, lambda signum, _: stopped.append(signal.Signals(signum).name))
    started = time.perf_counter()

    def emit(event, **values):
        record = {"event": event, "step": step, **values}
        with (args.output/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False)+"\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def save(name="last.pt"):
        atomic_torch_save(checkpoint_payload(backbone, head, optimizer, scheduler, stream, mask_rng,
                                            step=step, identity=identity, best_dev=best), args.output/name)

    def development():
        nonlocal best
        result = evaluate_joint(backbone, head, dev, k=args.k, batch_size=args.batch_size,
                                microbatch_size=args.microbatch_size, seed=args.seed+100002)
        emit("dev", **result)
        if best is None or result["weighted_loss"] < best["weighted_loss"]:
            best = {"step": step, "weighted_loss": result["weighted_loss"],
                    "nll_per_masked_token": result["nll_per_masked_token"]}
            save("best.pt")
        save()

    try:
        development()
        while step < args.steps and not stopped:
            if args.max_seconds and time.perf_counter()-started >= args.max_seconds:
                stopped.append("wall_time_limit")
                break
            synchronize(args.device)
            tick = time.perf_counter()
            metrics = train_update(backbone, head, optimizer, scheduler, stream.next(), mask_rng,
                k=args.k, microbatch_size=args.microbatch_size, gradient_clip=args.gradient_clip)
            synchronize(args.device)
            step += 1
            emit("train", **metrics, step_seconds=time.perf_counter()-tick, epoch=stream.epoch,
                 learning_rates={g["name"]: g["lr"] for g in optimizer.param_groups})
            if step % args.eval_every == 0 or step == args.steps:
                development()
            elif step % args.save_every == 0:
                save()
        save()
        summary = {"complete": step == args.steps, "step": step, "target_steps": args.steps,
                   "trained_tokens": step*args.batch_size*args.length, "best_dev": best,
                   "stop_reason": stopped or ["finished"], "elapsed_seconds": time.perf_counter()-started,
                   "synthetic_only": args.synthetic_backbone, "checkpoint_sha256": file_sha256(args.output/"last.pt")}
        atomic_json(summary, args.output/"results.json")
        emit("finished", **summary)
    finally:
        for s, handler in previous_handlers.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    main()
