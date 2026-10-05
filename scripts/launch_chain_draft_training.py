#!/usr/bin/env python3
"""Prepare or submit frozen-MDLM draft training after successful step-3 checks.

Required budget arguments prevent silently recycling the old smoke-run setup.
By default this writes a reviewable plan and batch scripts; --submit launches
one CPU count-fitting job and a learned-head GPU array. Jobs run a saved source
snapshot, so later checkout edits cannot change an in-flight experiment.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/u401/n23zhang/rework-data")
MODES = ("global", "contextual", "independent")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def check_validation(data, validation):
    report = json.loads((validation / "validation.json").read_text())
    if report.get("status") != "passed" or report["data_manifest_sha256"] != sha(data / "manifest.json"):
        raise ValueError("Successful step-3 validation for this exact dataset is required")
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["length"] != 1024 or manifest["selection_report"]["evaluation_population"] != "full-tail":
        raise ValueError("Expected the audited full-tail 1024 dataset")
    sources = json.loads((validation / "cli-train/protocol.json").read_text())["identity"]["source_sha256"]
    sources.update(json.loads((validation / "generation-backbone/manifest.json").read_text())["source_sha256"])
    for name, expected in sources.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Code changed since step-3 validation: {name}")
    for role in ("train", "dev"):
        if sha(data / f"{role}.pt") != manifest["files"][f"{role}.pt"]:
            raise ValueError(f"Data changed since validation: {role}")
    return report, manifest


def worker(run):
    """Execute a saved array task; resume only that task's own checkpoint."""
    import torch
    plan = json.loads((run / "plan.json").read_text())
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    task = plan["tasks"][index]
    if torch.cuda.get_device_name() != plan["gpu"]:
        raise ValueError("Training GPU differs from the validated GPU model")
    snapshot = run / "source"
    for name, expected in plan["source_sha256"].items():
        if sha(snapshot / name) != expected:
            raise ValueError(f"Saved source changed: {name}")
    for role in ("train", "dev"):
        if sha(Path(plan["data"]) / f"{role}.pt") != plan["data_sha256"][role]:
            raise ValueError(f"Prepared data changed: {role}")
    if sha(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise ValueError("Released backbone file changed")
    command = task["command"]
    last = Path(task["output"]) / "last.pt"
    if last.exists():
        command = [*command, "--resume", str(last)]
    os.chdir(snapshot)
    os.execv(sys.executable, [sys.executable, *command])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True, help="Step-3 results directory")
    p.add_argument("--steps", type=int, required=True, help="Common total update target for every learned head")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--parallel-heads", type=int, required=True)
    p.add_argument("--wall-hours", type=int, required=True, help="Slurm limit per head")
    p.add_argument("--eval-every", type=int, required=True)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--pair-rank", type=int, default=32)
    p.add_argument("--independent-rank", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.parallel_heads, args.wall_hours, args.eval_every, args.save_every,
           args.warmup_steps, args.pair_rank, args.independent_rank, args.batch_size) < 1:
        raise ValueError("Budgets, dimensions and intervals must be positive")
    if args.learning_rate <= 0 or len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        raise ValueError("Require positive learning rate and distinct nonnegative seeds")
    if BASE.stat().st_uid != os.getuid() or not args.data.resolve().is_relative_to(BASE.resolve()):
        raise ValueError("This draft launcher writes only within the owner's rework-data directory")
    report, manifest = check_validation(args.data, args.validation)
    # Do not pretend that unprofiled memory/configuration has been validated.
    for mode in MODES:
        expected_rank = args.independent_rank if mode == "independent" else args.pair_rank
        tested = report["training_profiles"][mode]["config"]
        if (tested["batch_size"], tested["rank"], tested["k"]) != (args.batch_size, expected_rank, 64):
            raise ValueError(f"Re-profile the requested {mode} rank/batch size before launch")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    run = BASE / "runs/draft-full-scale-training" / ("training-" + stamp)
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "source"
    files = [*ROOT.glob("chain_crf/*.py"), ROOT / "models/__init__.py", ROOT / "models/dit.py",
        ROOT / "configs/model/small.yaml", ROOT / "scripts/train_chain_crf.py",
        ROOT / "scripts/prepare_released_mdlm_owt.py", ROOT / "scripts/build_chain_counts.py",
        Path(__file__).resolve(), ROOT / "notes/mdlm-crf-1024-decisions.md"]
    hashes = {}
    for original in files:
        relative = original.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        hashes[str(relative)] = sha(destination)
    shutil.copy2(args.data / "manifest.json", run / "data-manifest.json")
    shutil.copy2(args.validation / "validation.json", run / "validation.json")
    checkpoint = BASE / "checkpoints/mdlm-owt.pt"
    if sha(checkpoint) != report["backbone"]["loaded_file_sha256"]:
        raise ValueError("Released backbone differs from step-3 validation")
    plan = {"kind": "draft_frozen_mdlm_heads", "data": str(args.data.resolve()),
        "validation": str(args.validation.resolve()), "gpu": report["gpu"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "data_sha256": {role: manifest["files"][f"{role}.pt"] for role in ("train", "dev")},
        "source_sha256": hashes, "settings": vars(args).copy(),
        "unique_train_rows": manifest["splits"]["train"]["rows"],
        "train_payload_tokens": manifest["splits"]["train"]["packed_text_tokens"],
        "tokens_seen_per_head": args.steps * args.batch_size * 1024,
        "all_dev_rows": manifest["splits"]["dev"]["rows"],
        "checkpoint_selection": "minimum joint denoising NLL on all prepared development rows",
        "backbone_frozen": True, "sampling_launched": False, "tasks": [], "submissions": []}
    plan["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in plan["settings"].items()}
    for seed in args.seeds:
        for mode in MODES:
            target = run / f"{mode}-seed-{seed}"
            rank = args.independent_rank if mode == "independent" else args.pair_rank
            command = [str(snapshot / "scripts/train_chain_crf.py"), "--data", str(args.data.resolve()),
                "--output", str(target), "--mode", mode, "--checkpoint", str(checkpoint),
                "--cache-dir", str(BASE / "hf-cache/hub"), "--length", "1024", "--k", "64",
                "--rank", str(rank), "--mlp-size", "128", "--steps", str(args.steps),
                "--seed", str(seed), "--batch-size", str(args.batch_size), "--backbone-batch-size", "1",
                "--learning-rate", str(args.learning_rate), "--weight-decay", "0", "--gradient-clip", "1",
                "--warmup-steps", str(args.warmup_steps), "--eval-every", str(args.eval_every),
                "--save-every", str(args.save_every), "--max-dev-examples", str(plan["all_dev_rows"]),
                "--max-seconds", str(args.wall_hours * 3600 - 600), "--threads", "4"]
            plan["tasks"].append({"mode": mode, "seed": seed, "output": str(target), "command": command})
    preamble = ["#!/bin/bash", "set -eo pipefail", "source /opt/anaconda3/etc/profile.d/conda.sh",
        "conda activate mdlm-crf", "export PYTHONNOUSERSITE=1", "export TOKENIZERS_PARALLELISM=false",
        "export OMP_NUM_THREADS=4", "export HF_HUB_OFFLINE=1", "export TRANSFORMERS_OFFLINE=1",
        "export HF_HOME=" + shlex.quote(str(BASE / "hf-cache")), "cd " + shlex.quote(str(snapshot))]
    head_script = run / "train-heads.sbatch"
    head_script.write_text("\n".join([*preamble, "exec python " + shlex.join([
        str(snapshot / "scripts/launch_chain_draft_training.py"), "--worker", str(run)]), ""]))
    count_script = run / "fit-counts.sbatch"
    count_command = ["python", str(snapshot / "scripts/build_chain_counts.py"), "--data", str(args.data.resolve() / "train.pt"),
        "--output", str(run / "counts/owt-counts.pt"), "--max-tokens", str(manifest["splits"]["train"]["packed_tokens"])]
    count_script.write_text("\n".join([*preamble, "export CUDA_VISIBLE_DEVICES=", "exec " + shlex.join(count_command), ""]))
    common = ["sbatch", "--parsable", "--partition=ALL", "--nodelist=watgpu108", "--cpus-per-task=4", "--mem=24G"]
    submissions = [common + ["--job-name=draft-counts", "--time=08:00:00",
        "--output=" + str(run / "counts-%j.log"), str(count_script)],
        common + ["--job-name=draft-heads", "--gres=gpu:1", f"--time={args.wall_hours}:00:00",
        f"--array=0-{len(plan['tasks']) - 1}%{args.parallel_heads}", "--signal=TERM@300",
        "--output=" + str(run / "head-%A_%a.log"), str(head_script)]]
    plan["submission_commands"] = submissions
    write_json(run / "plan.json", plan)
    print(json.dumps({"run_directory": str(run), "plan": str(run / "plan.json"),
        "steps": args.steps, "tokens_seen_per_head": plan["tokens_seen_per_head"],
        "dev_rows_per_check": plan["all_dev_rows"], "submitted": False}, indent=2), flush=True)
    if args.submit:
        for command in submissions:
            result = subprocess.run(command, capture_output=True, text=True)
            plan["submissions"].append({"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()})
            write_json(run / "plan.json", plan)
            result.check_returncode()
            print(json.dumps({"submitted_job": result.stdout.strip()}), flush=True)
    else:
        for command in submissions:
            print(shlex.join(command))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
    else:
        main()
