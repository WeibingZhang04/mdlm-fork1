#!/usr/bin/env python3
"""Train one frozen-MDLM chain head with fresh corruptions and exact joint NLL.

Checkpoints include optimizer/scheduler, RNG, data cursor and exact provenance.
--steps is a total target on resume. Seed controls reproducibility only; this
script does not run seed sweeps or compute confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from chain_crf.backbone import FrozenMDLM, SyntheticBackbone, file_sha256
from chain_crf.core import build_candidates, gold_log_prob
from chain_crf.data import BatchStream, assert_disjoint, atomic_json, atomic_torch_save, canonical_hash, load_token_data
from chain_crf.heads import GlobalPairHead, ContextualPairHead, IndependentHead

SOURCE_FILES = ("chain_crf/backbone.py", "chain_crf/core.py", "chain_crf/heads.py",
                "chain_crf/data.py", "scripts/train_chain_crf.py", "models/dit.py")


def make_head(mode, vocab_size, hidden_size, rank, mlp_size=128):
    if mode == "global":
        return GlobalPairHead(vocab_size, rank=rank)
    if mode == "contextual":
        return ContextualPairHead(vocab_size, hidden_size, rank=rank, mlp_size=mlp_size)
    if mode == "independent":
        return IndependentHead(vocab_size, hidden_size, rank=rank)
    raise ValueError(f"Unknown mode {mode}")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def corrupt(clean, mask_id, generator, noise_eps=1e-3):
    """Uniform normalized t and Bernoulli forward corruption, on CPU RNG."""
    times = .001 + .999 * torch.rand(len(clean), generator=generator)
    active = torch.rand(clean.shape, generator=generator) < ((1-noise_eps) * times[:, None])
    # An all-visible batch has zero training weight. Redraw that rare batch,
    # rather than inserting a position selected from the clean target.
    while not bool(active.any()):
        times = .001 + .999 * torch.rand(len(clean), generator=generator)
        active = torch.rand(clean.shape, generator=generator) < ((1-noise_eps) * times[:, None])
    return torch.where(active, mask_id, clean), active, times


def score_batch(backbone, head, mode, clean, *, k, device, generator, backbone_batch_size=1):
    corrupted, active, times = corrupt(clean.cpu(), backbone.mask_id, generator, backbone.noise_eps)
    clean, corrupted, active, times = [x.to(device) for x in (clean, corrupted, active, times)]
    synchronize(device)
    started = time.perf_counter()
    outputs = []
    for offset in range(0, len(clean), backbone_batch_size):
        with torch.no_grad():
            outputs.append(backbone(corrupted[offset:offset+backbone_batch_size],
                                    times[offset:offset+backbone_batch_size]))
    log_probs = torch.cat([o["log_probs"] for o in outputs])
    hidden = torch.cat([o["hidden"] for o in outputs])
    synchronize(device)
    backbone_seconds = time.perf_counter() - started
    started = time.perf_counter()
    packet = build_candidates(log_probs, corrupted, backbone.mask_id, k, gold=clean)
    scores = head(packet.candidate_ids, hidden=hidden, time=times)
    if mode == "independent":
        scores = scores * active.unsqueeze(-1)
        edges = torch.zeros((len(clean), clean.shape[1]-1, scores.shape[-1], scores.shape[-1]),
                            device=device, dtype=scores.dtype)
        joint = gold_log_prob(packet, edges, unary_delta=scores)
    else:
        # Known-known factors are constants; removing them avoids subtracting
        # large irrelevant values in gold-score minus log-partition.
        active_edges = active[:, :-1] | active[:, 1:]
        scores = torch.where(active_edges[..., None, None], scores, torch.zeros_like(scores))
        joint = gold_log_prob(packet, scores)
    n_masked = active.sum()
    loss = -joint.sum() / n_masked
    baseline_sum = -(log_probs.gather(-1, clean[..., None]).squeeze(-1) * active).sum()
    covered = (packet.candidate_ids[..., :-1] == clean[..., None]).any(-1)
    adjacent = active[:, :-1] & active[:, 1:]
    pair_covered = covered[:, :-1] & covered[:, 1:]
    retained_mass = packet.unary[..., :-1].exp().sum(-1)
    synchronize(device)
    metrics = {
        "masked_tokens": int(n_masked), "tokens": clean.numel(),
        "nll_sum": float((-joint.sum()).detach()), "baseline_nll_sum": float(baseline_sum),
        "nll_per_masked_token": float(loss.detach()),
        "baseline_nll_per_masked_token": float(baseline_sum/n_masked),
        "gold_candidate_covered": int((covered & active).sum()),
        "gold_candidate_coverage": float((covered & active).sum()/n_masked),
        "retained_mass_sum": float((retained_mass * active).sum()),
        "retained_mass_mean": float((retained_mass * active).sum()/n_masked),
        "masked_adjacent_pairs": int(adjacent.sum()),
        "gold_adjacent_pair_covered": int((pair_covered & adjacent).sum()),
        "mean_time": float(times.mean()), "backbone_seconds": backbone_seconds,
        "head_forward_dp_seconds": time.perf_counter()-started,
        "backbone_calls": len(outputs),
    }
    if not bool(torch.isfinite(loss)) or not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite training/evaluation loss or metrics")
    return loss, metrics


@torch.no_grad()
def evaluate(backbone, head, mode, tokens, *, k, device, batch_size, backbone_batch_size, seed):
    head.eval()
    generator = torch.Generator().manual_seed(seed + 100002)
    sums = {}
    for offset in range(0, len(tokens), batch_size):
        _, metrics = score_batch(backbone, head, mode, tokens[offset:offset+batch_size],
                                 k=k, device=device, generator=generator,
                                 backbone_batch_size=backbone_batch_size)
        for key, value in metrics.items():
            if key.endswith("_sum") or key in {"masked_tokens", "tokens", "gold_candidate_covered",
                    "masked_adjacent_pairs", "gold_adjacent_pair_covered", "backbone_seconds",
                    "head_forward_dp_seconds", "backbone_calls"}:
                sums[key] = sums.get(key, 0) + value
    count = sums["masked_tokens"]
    sums.update(nll_per_masked_token=sums["nll_sum"]/count,
                baseline_nll_per_masked_token=sums["baseline_nll_sum"]/count,
                gold_candidate_coverage=sums["gold_candidate_covered"]/count,
                retained_mass_mean=sums["retained_mass_sum"]/count)
    head.train()
    return sums


def checkpoint_payload(head, optimizer, scheduler, stream, mask_rng, step, identity, best, config):
    return {"schema_version": 1, "head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "stream": stream.state_dict(),
            "mask_rng": mask_rng.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(), "step": step, "identity": identity,
            "identity_sha256": canonical_hash(identity), "best_dev": best, "config": config}


def restore(payload, head, optimizer, scheduler, stream, mask_rng, identity):
    if payload.get("identity_sha256") != canonical_hash(identity) or payload.get("identity") != identity:
        raise ValueError("Resume identity differs: data, source, model, or training configuration changed")
    head.load_state_dict(payload["head"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    stream.load_state_dict(payload["stream"])
    mask_rng.set_state(payload["mask_rng"].cpu())
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in payload["cuda_rng"]])
    random.setstate(payload["python_rng"])
    return int(payload["step"]), payload["best_dev"]


def continue_initialization_stream(initial, stream, mask_rng, *, train_sha256, config):
    """Continue the warm-start run's next data/mask draw, not its optimizer.

    The contextual head adds parameters, so it intentionally gets a fresh
    optimizer. Its examples and corruption stream instead continue exactly
    after the global initialization's final update.
    """
    if initial["identity"]["train_sha256"] != train_sha256:
        raise ValueError("Warm-start stream uses a different training data file")
    for key in ("seed", "batch_size", "length"):
        if initial["config"][key] != config[key]:
            raise ValueError(f"Warm-start stream requires matching {key}")
    if initial["identity"]["train_examples"] != len(stream.tokens):
        raise ValueError("Warm-start stream has a different training selection")
    stream.load_state_dict(initial["stream"])
    mask_rng.set_state(initial["mask_rng"].cpu())
    return int(initial["step"])


def initialization_token_exposures(initial):
    """Count the initial global run using its own validated training dimensions."""
    identity, config = initial["identity"], initial["config"]
    if initial.get("identity_sha256") != canonical_hash(identity) or identity.get("config") != config:
        raise ValueError("Global initialization training identity checksum/configuration mismatch")
    values = (initial["step"], config["batch_size"], config["length"])
    if (any(type(value) is not int for value in values)
            or values[0] < 0 or min(values[1:]) < 1):
        raise ValueError("Invalid global initialization training dimensions")
    expected = values[0] * values[1] * values[2]
    if "trained_tokens" in initial and initial["trained_tokens"] != expected:
        raise ValueError("Global initialization token count differs from its training configuration")
    return expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Prepared directory or training JSONL/PT")
    p.add_argument("--dev-data", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["global", "contextual", "independent"], required=True)
    p.add_argument("--checkpoint", type=Path, help="Pinned raw MDLM backbone; otherwise download release")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--init-global", type=Path, help="Initialize contextual head from a global-head checkpoint")
    p.add_argument("--continue-init-stream", action="store_true",
                   help="Continue global initialization's exact next data/corruption draw; contextual optimizer stays fresh")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--backbone-batch-size", type=int, default=1)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--mlp-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--gradient-clip", type=float, default=1.)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--max-train-examples", type=int)
    p.add_argument("--max-dev-examples", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--synthetic-backbone", action="store_true", help="Offline plumbing tests ONLY")
    p.add_argument("--synthetic-vocab-size", type=int, default=17)
    args = p.parse_args(argv)
    if min(args.steps, args.batch_size, args.backbone_batch_size, args.length, args.k,
           args.rank, args.eval_every, args.save_every, args.threads) < 1:
        raise ValueError("Training dimensions and intervals must be positive")
    if args.length < 2 or args.learning_rate <= 0 or args.warmup_steps < 0 or args.gradient_clip <= 0:
        raise ValueError("Invalid learning configuration")
    if args.init_global and (args.mode != "contextual" or args.resume):
        raise ValueError("--init-global is only for a fresh contextual-head run")
    if args.continue_init_stream and (not args.init_global or args.resume):
        raise ValueError("--continue-init-stream requires a fresh --init-global run")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Output is nonempty: use a new directory or explicit --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    backbone = (SyntheticBackbone(vocab_size=args.synthetic_vocab_size, device=args.device)
                if args.synthetic_backbone else FrozenMDLM(args.checkpoint, device=args.device, cache_dir=args.cache_dir))
    train_path = args.data / "train.pt" if args.data.is_dir() else args.data
    dev_path = args.dev_data or (args.data / "dev.pt" if args.data.is_dir() else None)
    if dev_path is None:
        raise ValueError("--dev-data is required when --data is a file")
    train, train_docs, train_source = load_token_data(train_path, length=args.length,
        vocab_size=backbone.vocab_size, mask_id=backbone.mask_id, max_examples=args.max_train_examples)
    dev, dev_docs, dev_source = load_token_data(dev_path, length=args.length,
        vocab_size=backbone.vocab_size, mask_id=backbone.mask_id, max_examples=args.max_dev_examples)
    assert_disjoint(train, train_docs, dev, dev_docs)
    torch.manual_seed(args.seed)
    head = make_head(args.mode, backbone.vocab_size, backbone.hidden_size, args.rank, args.mlp_size).to(args.device)
    if args.init_global:
        initial = torch.load(args.init_global, map_location="cpu", weights_only=True)
        initialization_tokens = initialization_token_exposures(initial)
        if initial["config"]["mode"] != "global" or initial["config"]["rank"] != args.rank:
            raise ValueError("Global initialization must have matching rank")
        if initial["identity"]["backbone"] != backbone.provenance:
            raise ValueError("Global initialization belongs to a different backbone")
        global_head = make_head("global", backbone.vocab_size, backbone.hidden_size, args.rank)
        global_head.load_state_dict(initial["head"], strict=True)
        head.load_global(global_head)
    resume_payload = (torch.load(args.resume, map_location=args.device, weights_only=True)
                      if args.resume else None)
    init_global_sha256 = (file_sha256(args.init_global) if args.init_global else
                         resume_payload["identity"].get("init_global_sha256") if resume_payload else None)
    stream_continuation = (bool(resume_payload["identity"].get("continue_init_stream", False))
                           if resume_payload else args.continue_init_stream)
    initialization_steps = (int(initial["step"]) if args.init_global else
                            int(resume_payload["identity"].get("initialization_steps", 0)) if resume_payload else 0)
    if not args.init_global:
        initialization_tokens = (int(resume_payload["identity"].get("initialization_token_exposures", 0))
                                 if resume_payload else 0)
    config = {"mode": args.mode, "vocab_size": backbone.vocab_size, "hidden_size": backbone.hidden_size,
              "rank": args.rank, "mlp_size": args.mlp_size, "k": args.k, "length": args.length,
              "seed": args.seed, "batch_size": args.batch_size, "backbone_batch_size": args.backbone_batch_size,
              "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
              "warmup_steps": args.warmup_steps, "gradient_clip": args.gradient_clip,
              "eval_every": args.eval_every, "save_every": args.save_every,
              "synthetic_only": args.synthetic_backbone}
    identity = {"config": config, "backbone": backbone.provenance,
                "init_global_sha256": init_global_sha256,
                "continue_init_stream": stream_continuation,
                "initialization_steps": initialization_steps,
                "initialization_token_exposures": initialization_tokens,
                "train_sha256": file_sha256(train_path), "dev_sha256": file_sha256(dev_path),
                "train_examples": len(train), "dev_examples": len(dev),
                "source_sha256": {name: file_sha256(ROOT/name) for name in SOURCE_FILES},
                "torch_version": str(torch.__version__), "device_type": torch.device(args.device).type,
                "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None}
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1., (step+1)/max(1,args.warmup_steps)))
    stream = BatchStream(train, args.batch_size, args.seed+1000)
    mask_rng = torch.Generator().manual_seed(args.seed+2000)
    if args.continue_init_stream:
        continue_initialization_stream(initial, stream, mask_rng,
            train_sha256=identity["train_sha256"], config=config)
    step, best = 0, None
    if args.resume:
        step, best = restore(resume_payload, head, optimizer, scheduler, stream, mask_rng, identity)
        if step > args.steps:
            raise ValueError("--steps total target precedes resume checkpoint")
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unknown"
    atomic_json({"identity": identity, "identity_sha256": canonical_hash(identity), "git_head": git_head,
                 "arguments": {key: str(v) if isinstance(v, Path) else v for key,v in vars(args).items()},
                 "train_source": train_source, "dev_source": dev_source,
                 "parameter_count": sum(p.numel() for p in head.parameters()),
                 "backbone_frozen": True, "test_data_access": False,
                 "candidate_policy": "base top-K plus neutral residual; never gold insertion",
                 "loss": "joint clean-token NLL / masked-token count, including residual conditional likelihood",
                 "corruption": "uniform t in [.001,1]; independent Bernoulli mask probability .999*t",
                 "resume_sha256": file_sha256(args.resume) if args.resume else None,
                 "init_global_sha256": init_global_sha256,
                 "initialization_steps": initialization_steps,
                 "initialization_token_exposures": initialization_tokens,
                 "continue_init_stream": stream_continuation,
                 "contextual_optimizer": "fresh; data and corruption RNG continued" if stream_continuation else None},
                args.output/"protocol.json")
    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append("SIGTERM"))
    signal.signal(signal.SIGINT, lambda *_: stopped.append("SIGINT"))
    start = time.perf_counter()

    def emit(event, **values):
        record = {"event": event, "step": step, **values}
        with (args.output/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False)+"\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def save(name="last.pt"):
        atomic_torch_save(checkpoint_payload(head, optimizer, scheduler, stream, mask_rng,
                                            step, identity, best, config), args.output/name)

    def development():
        nonlocal best
        result = evaluate(backbone, head, args.mode, dev, k=args.k, device=args.device,
                          batch_size=args.batch_size, backbone_batch_size=args.backbone_batch_size, seed=args.seed)
        emit("dev", **result)
        if best is None or result["nll_per_masked_token"] < best["nll_per_masked_token"]:
            best = {"step": step, "nll_per_masked_token": result["nll_per_masked_token"]}
            save("best.pt")
        save()

    development()
    while step < args.steps and not stopped:
        if args.max_seconds and time.perf_counter()-start >= args.max_seconds:
            stopped.append("wall_time_limit")
            break
        head.train()
        optimizer.zero_grad(set_to_none=True)
        synchronize(args.device)
        tick = time.perf_counter()
        loss, metrics = score_batch(backbone, head, args.mode, stream.next(), k=args.k,
                                    device=args.device, generator=mask_rng,
                                    backbone_batch_size=args.backbone_batch_size)
        backward = time.perf_counter()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.gradient_clip, error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        synchronize(args.device)
        step += 1
        emit("train", **metrics, grad_norm=float(norm), learning_rate=scheduler.get_last_lr()[0],
             backward_optimizer_seconds=time.perf_counter()-backward,
             total_step_seconds=time.perf_counter()-tick, epoch=stream.epoch)
        if step % args.eval_every == 0 or step == args.steps:
            development()
        elif step % args.save_every == 0:
            save()
    save()
    summary = {"complete": step == args.steps, "step": step, "target_steps": args.steps,
               "trained_tokens": step*args.batch_size*args.length, "best_dev": best,
               "initialization_steps": initialization_steps,
               "initialization_token_exposures": initialization_tokens,
               "total_training_token_exposures": step*args.batch_size*args.length+initialization_tokens,
               "stop_reason": stopped or ["finished"], "elapsed_seconds": time.perf_counter()-start,
               "synthetic_only": args.synthetic_backbone, "last_sha256": file_sha256(args.output/"last.pt")}
    atomic_json(summary, args.output/"results.json")
    emit("finished", **summary)


if __name__ == "__main__":
    main()
